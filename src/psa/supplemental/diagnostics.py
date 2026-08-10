from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.supplemental.formal_run import (
    EXPECTED_GROUP_COUNT,
    EXPECTED_RAW_RECORD_COUNT,
    SUPPLEMENTAL_SET_DIGEST,
)
from psa.supplemental.set_generation import verify_exp001b_supplemental_set_package


EXPECTED_RAW_PAYLOAD_DIGEST = (
    "6926a932220f34b37c6b4e86fa65edc230e414726bd2d4d308bf471d1af290f6"
)
EXPECTED_TOKENIZER_SHA256 = (
    "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _quantile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def prefix_failure_flags(output: Mapping[str, Any]) -> dict[str, bool]:
    metadata = output.get("metadata")
    prefix = metadata.get("forced_prefix") if isinstance(metadata, Mapping) else None
    present = isinstance(prefix, Mapping)
    text_exact = bool(present and prefix.get("text") == ">\n")
    greedy_exact = bool(present and prefix.get("greedy_exact") is True)
    roundtrip_exact = bool(present and prefix.get("roundtrip_exact") is True)
    return {
        "missing": not present,
        "text_mismatch": not text_exact,
        "greedy_mismatch": not greedy_exact,
        "roundtrip_mismatch": not roundtrip_exact,
        "valid": present and text_exact and greedy_exact and roundtrip_exact,
    }


def prefix_token_divergence(output: Mapping[str, Any]) -> dict[str, Any]:
    metadata = output.get("metadata")
    prefix = metadata.get("forced_prefix") if isinstance(metadata, Mapping) else None
    if not isinstance(prefix, Mapping):
        raise ValueError("forced-prefix metadata is missing")
    expected = prefix.get("token_ids")
    greedy = prefix.get("greedy_token_ids")
    if (
        not isinstance(expected, list)
        or not isinstance(greedy, list)
        or any(not isinstance(item, int) for item in expected + greedy)
    ):
        raise ValueError("forced-prefix token IDs are invalid")
    divergence_index = None
    limit = max(len(expected), len(greedy))
    for index in range(limit):
        expected_id = expected[index] if index < len(expected) else None
        greedy_id = greedy[index] if index < len(greedy) else None
        if expected_id != greedy_id:
            divergence_index = index
            break
    return {
        "expected_token_ids": list(expected),
        "greedy_token_ids": list(greedy),
        "divergence_index": divergence_index,
        "expected_divergent_token_id": (
            expected[divergence_index]
            if divergence_index is not None and divergence_index < len(expected)
            else None
        ),
        "greedy_divergent_token_id": (
            greedy[divergence_index]
            if divergence_index is not None and divergence_index < len(greedy)
            else None
        ),
    }


def summarize_prefix_cells(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[dict[str, bool]]] = defaultdict(list)
    for frozen, output in rows:
        cells[(str(frozen["condition"]), str(frozen["task_type"]))].append(
            prefix_failure_flags(output)
        )
    reports = []
    for (condition, task_type), flags in sorted(cells.items()):
        if len(flags) != 32:
            raise ValueError("control diagnostic cell must contain 32 records")
        failure_counts = {
            key: sum(int(item[key]) for item in flags)
            for key in (
                "missing",
                "text_mismatch",
                "greedy_mismatch",
                "roundtrip_mismatch",
            )
        }
        invalid_count = sum(int(not item["valid"]) for item in flags)
        reports.append(
            {
                "condition": condition,
                "task_type": task_type,
                "record_count": len(flags),
                "invalid_prefix_count": invalid_count,
                "valid_prefix_rate": 1.0 - invalid_count / len(flags),
                "failure_counts": failure_counts,
            }
        )
    return {
        "cell_count": len(reports),
        "cells_with_failures": sum(item["invalid_prefix_count"] > 0 for item in reports),
        "invalid_record_count": sum(item["invalid_prefix_count"] for item in reports),
        "cells": reports,
    }


def summarize_control_concordance(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {}
    for frozen, output in rows:
        sample_id = str(frozen["source_control_sample_id"])
        task_type = str(frozen["task_type"])
        condition = str(frozen["condition"])
        bucket = samples.setdefault(
            sample_id,
            {"task_type": task_type, "conditions": {}},
        )
        if bucket["task_type"] != task_type or condition in bucket["conditions"]:
            raise ValueError("control source sample is inconsistent")
        bucket["conditions"][condition] = prefix_failure_flags(output)[
            "greedy_mismatch"
        ]
    condition_failure_sets: dict[str, set[str]] = defaultdict(set)
    pattern_counts: Counter[tuple[str, ...]] = Counter()
    condition_count_distribution: Counter[int] = Counter()
    task_any_failure_counts: Counter[str] = Counter()
    for sample_id, sample in samples.items():
        if len(sample["conditions"]) != 8:
            raise ValueError("each control source sample must cover eight conditions")
        failed = tuple(
            sorted(
                condition
                for condition, failure in sample["conditions"].items()
                if failure
            )
        )
        pattern_counts[failed] += 1
        condition_count_distribution[len(failed)] += 1
        task_any_failure_counts[sample["task_type"]] += int(bool(failed))
        for condition in failed:
            condition_failure_sets[condition].add(sample_id)
    conditions = sorted(
        {condition for sample in samples.values() for condition in sample["conditions"]}
    )
    pairwise = []
    for index, left in enumerate(conditions):
        for right in conditions[index + 1 :]:
            left_set = condition_failure_sets[left]
            right_set = condition_failure_sets[right]
            overlap = len(left_set & right_set)
            union = len(left_set | right_set)
            if overlap:
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "overlap_count": overlap,
                        "jaccard": overlap / union,
                    }
                )
    return {
        "source_sample_count": len(samples),
        "samples_with_any_greedy_mismatch": sum(
            count for failures, count in condition_count_distribution.items() if failures
        ),
        "failed_condition_count_distribution": {
            str(key): condition_count_distribution[key]
            for key in sorted(condition_count_distribution)
        },
        "samples_with_any_failure_by_task": dict(sorted(task_any_failure_counts.items())),
        "condition_failure_sample_counts": {
            condition: len(condition_failure_sets[condition]) for condition in conditions
        },
        "failure_condition_patterns": [
            {"conditions": list(pattern), "sample_count": count}
            for pattern, count in sorted(
                pattern_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "pairwise_failure_overlaps": sorted(
            pairwise,
            key=lambda item: (-item["overlap_count"], item["left"], item["right"]),
        ),
    }


def summarize_control_greedy_tokens(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    pattern_counts: Counter[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = Counter()
    divergence_counts: Counter[tuple[int | None, int | None, int | None]] = Counter()
    failure_count = 0
    for frozen, output in rows:
        flags = prefix_failure_flags(output)
        if not flags["greedy_mismatch"]:
            continue
        failure_count += 1
        divergence = prefix_token_divergence(output)
        expected = tuple(divergence["expected_token_ids"])
        greedy = tuple(divergence["greedy_token_ids"])
        pattern_counts[
            (str(frozen["condition"]), str(frozen["task_type"]), expected, greedy)
        ] += 1
        divergence_counts[
            (
                divergence["divergence_index"],
                divergence["expected_divergent_token_id"],
                divergence["greedy_divergent_token_id"],
            )
        ] += 1
    return {
        "greedy_mismatch_record_count": failure_count,
        "token_patterns": [
            {
                "condition": condition,
                "task_type": task_type,
                "expected_token_ids": list(expected),
                "greedy_token_ids": list(greedy),
                "record_count": count,
            }
            for (condition, task_type, expected, greedy), count in sorted(
                pattern_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "first_divergence_patterns": [
            {
                "divergence_index": index,
                "expected_token_id": expected,
                "greedy_token_id": greedy,
                "record_count": count,
            }
            for (index, expected, greedy), count in sorted(
                divergence_counts.items(),
                key=lambda item: (-item[1], repr(item[0])),
            )
        ],
    }


def decode_control_greedy_tokens(
    report: Mapping[str, Any],
    decoder: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    result = {
        "greedy_mismatch_record_count": int(report["greedy_mismatch_record_count"]),
        "token_patterns": [],
        "first_divergence_patterns": [],
    }
    for item in report["token_patterns"]:
        decorated = dict(item)
        decorated["expected_text"] = decoder(item["expected_token_ids"])
        decorated["greedy_text"] = decoder(item["greedy_token_ids"])
        result["token_patterns"].append(decorated)
    for item in report["first_divergence_patterns"]:
        decorated = dict(item)
        expected = item["expected_token_id"]
        greedy = item["greedy_token_id"]
        decorated["expected_token_text"] = (
            decoder([expected]) if expected is not None else None
        )
        decorated["greedy_token_text"] = decoder([greedy]) if greedy is not None else None
        result["first_divergence_patterns"].append(decorated)
    return result


def classify_control_prefix_failures(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    decoder: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    condition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cell_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    source_combo_counts: dict[str, Counter[str]] = defaultdict(Counter)
    factorial_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    records = []
    answer_codes = {"A", "B", "C", "D"}

    for frozen, output in rows:
        if not prefix_failure_flags(output)["greedy_mismatch"]:
            continue
        divergence = prefix_token_divergence(output)
        expected_text = decoder(divergence["expected_token_ids"])
        greedy_text = decoder(divergence["greedy_token_ids"])
        target_code = str(frozen["target_code"])
        if divergence["divergence_index"] == 0:
            category = "first_token_corruption"
        elif greedy_text == f">{target_code}":
            category = "correct_answer_emitted_immediately"
        elif (
            len(greedy_text) == 2
            and greedy_text.startswith(">")
            and greedy_text[1] in answer_codes
        ):
            category = "wrong_answer_emitted_immediately"
        else:
            category = "other"

        condition = str(frozen["condition"])
        task_type = str(frozen["task_type"])
        source_combo = frozen.get("assigned_source_combo")
        if isinstance(source_combo, list):
            source_combo_label = json.dumps(source_combo, separators=(",", ":"))
        else:
            source_combo_label = "null"
        factorial_group = str(frozen.get("assigned_factorial_group_id"))
        category_counts[category] += 1
        condition_counts[condition][category] += 1
        task_counts[task_type][category] += 1
        cell_counts[(condition, task_type)][category] += 1
        source_combo_counts[source_combo_label][category] += 1
        factorial_group_counts[factorial_group][category] += 1
        records.append(
            {
                "record_id": str(frozen["record_id"]),
                "source_control_sample_id": str(frozen["source_control_sample_id"]),
                "condition": condition,
                "task_type": task_type,
                "target_code": target_code,
                "assigned_source_combo": source_combo,
                "assigned_factorial_group_id": factorial_group,
                "category": category,
                "expected_token_ids": divergence["expected_token_ids"],
                "greedy_token_ids": divergence["greedy_token_ids"],
                "divergence_index": divergence["divergence_index"],
                "expected_text": expected_text,
                "greedy_text": greedy_text,
            }
        )

    def decorated_counts(counts: Mapping[str, Counter[str]]) -> list[dict[str, Any]]:
        return [
            {
                "label": label,
                "failure_count": sum(counter.values()),
                "category_counts": dict(sorted(counter.items())),
            }
            for label, counter in sorted(counts.items())
        ]

    failure_count = len(records)
    format_only = category_counts["correct_answer_emitted_immediately"]
    return {
        "greedy_mismatch_record_count": failure_count,
        "classification_complete": sum(category_counts.values()) == failure_count,
        "category_counts": dict(sorted(category_counts.items())),
        "semantic_preserving_format_only_count": format_only,
        "semantic_preserving_format_only_rate": (
            format_only / failure_count if failure_count else None
        ),
        "by_condition": decorated_counts(condition_counts),
        "by_task_type": decorated_counts(task_counts),
        "by_condition_and_task_type": [
            {
                "condition": condition,
                "task_type": task_type,
                "failure_count": sum(counter.values()),
                "category_counts": dict(sorted(counter.items())),
            }
            for (condition, task_type), counter in sorted(cell_counts.items())
        ],
        "by_assigned_source_combo": decorated_counts(source_combo_counts),
        "by_assigned_factorial_group": decorated_counts(factorial_group_counts),
        "records": sorted(records, key=lambda item: item["record_id"]),
    }


def summarize_nonrandom_failure_samples(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    token_counter: Callable[[str], int],
    decoder: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    samples: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for frozen, output in rows:
        samples[str(frozen["source_control_sample_id"])].append((frozen, output))
    reports = []
    task_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    token_counts = []
    for sample_id, sample_rows in sorted(samples.items()):
        if len(sample_rows) != 8:
            raise ValueError("each diagnostic source sample must contain eight conditions")
        prompts = {str(frozen["prompt"]) for frozen, _ in sample_rows}
        tasks = {str(frozen["task_type"]) for frozen, _ in sample_rows}
        targets = {str(frozen["target_code"]) for frozen, _ in sample_rows}
        digests = {str(frozen["prompt_digest_sha256"]) for frozen, _ in sample_rows}
        if len(prompts) != 1 or len(tasks) != 1 or len(targets) != 1 or len(digests) != 1:
            raise ValueError("control source sample fields differ across conditions")
        failures = []
        for frozen, output in sample_rows:
            condition = str(frozen["condition"])
            if condition == "random_matched" or not prefix_failure_flags(output)[
                "greedy_mismatch"
            ]:
                continue
            divergence = prefix_token_divergence(output)
            divergence["expected_text"] = decoder(divergence["expected_token_ids"])
            divergence["greedy_text"] = decoder(divergence["greedy_token_ids"])
            failures.append({"condition": condition, **divergence})
        if not failures:
            continue
        prompt = next(iter(prompts))
        task = next(iter(tasks))
        target = next(iter(targets))
        prompt_tokens = int(token_counter(prompt))
        task_counts[task] += 1
        target_counts[target] += 1
        token_counts.append(prompt_tokens)
        reports.append(
            {
                "source_control_sample_id": sample_id,
                "task_type": task,
                "target_code": target,
                "prompt_digest_sha256": next(iter(digests)),
                "prompt_character_count": len(prompt),
                "prompt_line_count": prompt.count("\n") + 1,
                "prompt_token_count": prompt_tokens,
                "failed_nonrandom_conditions": sorted(
                    item["condition"] for item in failures
                ),
                "greedy_divergences": sorted(
                    failures, key=lambda item: item["condition"]
                ),
                "prompt": prompt,
            }
        )
    return {
        "sample_count": len(reports),
        "by_task_type": dict(sorted(task_counts.items())),
        "by_target_code": dict(sorted(target_counts.items())),
        "prompt_token_count_distribution": {
            "minimum": min(token_counts) if token_counts else None,
            "median": _quantile(token_counts, 0.5),
            "maximum": max(token_counts) if token_counts else None,
        },
        "samples": reports,
    }


def summarize_matched_norms(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path_counts: Counter[str] = Counter()
    max_ratios = []
    records_with_alerts = 0
    total_alerts = 0
    prefix_failures = Counter()
    for output in outputs:
        metadata = output.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("matched diagnostic metadata is missing")
        alert_count = int(metadata.get("state_norm_alert_count", -1))
        paths = metadata.get("state_norm_alert_paths")
        max_ratio = float(metadata.get("state_norm_max_alert_ratio", float("nan")))
        if (
            alert_count < 0
            or not isinstance(paths, list)
            or any(not isinstance(path, str) for path in paths)
            or len(paths) != alert_count
            or not math.isfinite(max_ratio)
        ):
            raise ValueError("matched state-norm diagnostic fields are invalid")
        total_alerts += alert_count
        records_with_alerts += int(alert_count > 0)
        path_counts.update(paths)
        max_ratios.append(max_ratio)
        flags = prefix_failure_flags(output)
        for key, value in flags.items():
            if key != "valid" and value:
                prefix_failures[key] += 1
    count = len(outputs)
    if count == 0:
        raise ValueError("matched diagnostics require records")
    return {
        "record_count": count,
        "records_with_alerts": records_with_alerts,
        "record_alert_rate": records_with_alerts / count,
        "total_component_alerts": total_alerts,
        "alerts_per_record_mean": total_alerts / count,
        "unique_alert_path_count": len(path_counts),
        "top_alert_paths": [
            {"path": path, "count": path_counts[path]}
            for path in sorted(path_counts, key=lambda item: (-path_counts[item], item))[:25]
        ],
        "max_alert_ratio_distribution": {
            "median": _quantile(max_ratios, 0.5),
            "q95": _quantile(max_ratios, 0.95),
            "q99": _quantile(max_ratios, 0.99),
            "maximum": max(max_ratios),
        },
        "prefix_failure_counts": dict(sorted(prefix_failures.items())),
    }


def run_exp001b_posthoc_diagnostics(
    *,
    supplemental_raw_output_dir: str | Path,
    supplemental_raw_verification_path: str | Path,
    supplemental_set_package_dir: str | Path,
    analysis_output_dir: str | Path,
    diagnostic_output_dir: str | Path,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    raw_root = Path(supplemental_raw_output_dir).resolve()
    set_root = Path(supplemental_set_package_dir).resolve()
    analysis_root = Path(analysis_output_dir).resolve()
    destination = Path(diagnostic_output_dir).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("diagnostic output directory must be empty")
    verification = _load_object(
        supplemental_raw_verification_path, "supplemental raw verification"
    )
    raw_manifest = _load_object(raw_root / "manifest.json", "supplemental raw manifest")
    supplemental_set = _load_object(set_root / "supplemental_set.json", "supplemental set")
    set_report = verify_exp001b_supplemental_set_package(set_root)
    analysis_summary = _load_object(analysis_root / "summary.json", "analysis summary")
    analysis_report_path = analysis_root / "supplemental_report.json"
    prerequisites = {
        "raw_verification_valid": (
            verification.get("valid") is True
            and verification.get("status") == "raw_package_verified_unanalyzed"
            and verification.get("verified_record_count") == EXPECTED_RAW_RECORD_COUNT
            and verification.get("group_payload_digest_sha256")
            == EXPECTED_RAW_PAYLOAD_DIGEST
        ),
        "supplemental_set_valid": (
            set_report.get("valid") is True
            and supplemental_set.get("supplemental_set_digest_sha256")
            == SUPPLEMENTAL_SET_DIGEST
        ),
        "analysis_package_valid": (
            analysis_summary.get("valid") is True
            and analysis_summary.get("supplemental_raw_payload_digest_sha256")
            == EXPECTED_RAW_PAYLOAD_DIGEST
            and sha256_file(analysis_report_path)
            == analysis_summary.get("supplemental_report_sha256")
        ),
        "group_count_valid": (
            len(raw_manifest.get("expected_group_ids", [])) == EXPECTED_GROUP_COUNT
        ),
        "tokenizer_valid": (
            tokenizer_file.is_file()
            and sha256_file(tokenizer_file) == EXPECTED_TOKENIZER_SHA256
        ),
    }
    if not all(prerequisites.values()):
        failed = [key for key, value in prerequisites.items() if not value]
        raise ValueError(f"diagnostic prerequisites failed: {failed}")
    tokenizer_module = import_module("rwkv.rwkv_tokenizer")
    tokenizer = tokenizer_module.TRIE_TOKENIZER(str(tokenizer_file))

    def decode(token_ids: Sequence[int]) -> str:
        return str(tokenizer.decode(list(token_ids)))

    def token_count(text: str) -> int:
        return len(list(tokenizer.encode(text)))

    controls_by_id = {
        str(record["record_id"]): record for record in supplemental_set["records"]["controls"]
    }
    control_rows = []
    matched_outputs = []
    generation_prefix_failures = Counter()
    total_records = 0
    for group_id in raw_manifest["expected_group_ids"]:
        payload = _load_object(raw_root / "groups" / f"{group_id}.json", str(group_id))
        for output in payload.get("records", []):
            if not isinstance(output, Mapping):
                raise ValueError("raw diagnostic record must be an object")
            total_records += 1
            kind = output.get("record_kind")
            if kind == "matched_context":
                matched_outputs.append(output)
            elif kind == "general_capability_control_condition":
                frozen = controls_by_id.get(str(output.get("record_id")))
                if frozen is None:
                    raise ValueError("control output is not bound to the frozen set")
                control_rows.append((frozen, output))
            elif kind == "formal_generation_readout":
                flags = prefix_failure_flags(output)
                for key, value in flags.items():
                    if key != "valid" and value:
                        generation_prefix_failures[key] += 1
            else:
                raise ValueError("unknown supplemental record kind")
    if (
        total_records != EXPECTED_RAW_RECORD_COUNT
        or len(control_rows) != 768
        or len(matched_outputs) != 5_120
    ):
        raise ValueError("diagnostic did not consume the complete supplemental package")

    token_diagnostics = decode_control_greedy_tokens(
        summarize_control_greedy_tokens(control_rows), decode
    )
    report = {
        "report_version": "1.3-posthoc",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "EXP-001B",
        "exploratory_posthoc": True,
        "confirmatory_decision_changed": False,
        "automatic_rerun_authorized": False,
        "supplemental_raw_payload_digest_sha256": EXPECTED_RAW_PAYLOAD_DIGEST,
        "analysis_package_digest_sha256": analysis_summary[
            "analysis_package_digest_sha256"
        ],
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "prerequisite_checks": prerequisites,
        "control_prefix_diagnostics": summarize_prefix_cells(control_rows),
        "control_failure_concordance": summarize_control_concordance(control_rows),
        "control_greedy_token_diagnostics": token_diagnostics,
        "control_prefix_failure_classification": classify_control_prefix_failures(
            control_rows,
            decode,
        ),
        "nonrandom_failure_sample_diagnostics": summarize_nonrandom_failure_samples(
            control_rows,
            token_count,
            decode,
        ),
        "matched_state_norm_diagnostics": summarize_matched_norms(matched_outputs),
        "generation_prefix_failure_counts": dict(
            sorted(generation_prefix_failures.items())
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "diagnostic_report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    summary = {
        "summary_version": "1.3-posthoc",
        "experiment_id": "EXP-001B",
        "status": "posthoc_failure_diagnostics_complete",
        "valid": True,
        "exploratory_posthoc": True,
        "confirmatory_decision_changed": False,
        "automatic_rerun_authorized": False,
        "diagnostic_report_sha256": sha256_file(report_path),
        "diagnostic_package_digest_sha256": sha256_json(
            {"diagnostic_report.json": sha256_file(report_path)}
        ),
        "control_cells_with_prefix_failures": report[
            "control_prefix_diagnostics"
        ]["cells_with_failures"],
        "control_source_samples_with_any_greedy_mismatch": report[
            "control_failure_concordance"
        ]["samples_with_any_greedy_mismatch"],
        "control_source_samples_with_nonrandom_greedy_mismatch": report[
            "nonrandom_failure_sample_diagnostics"
        ]["sample_count"],
        "control_prefix_failure_category_counts": report[
            "control_prefix_failure_classification"
        ]["category_counts"],
        "control_semantic_preserving_format_only_count": report[
            "control_prefix_failure_classification"
        ]["semantic_preserving_format_only_count"],
        "matched_records_with_state_norm_alerts": report[
            "matched_state_norm_diagnostics"
        ]["records_with_alerts"],
        "matched_total_component_alerts": report[
            "matched_state_norm_diagnostics"
        ]["total_component_alerts"],
        "reports": ["diagnostic_report.json"],
    }
    (destination / "summary.json").write_bytes(canonical_json_bytes(summary))
    return summary
