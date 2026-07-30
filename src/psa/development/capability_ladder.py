from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.development.impl3 import (
    greedy_format_probe,
    inspect_answer_codes,
    normalized_probabilities,
    score_continuations,
    write_jsonl,
)
from psa.evaluation import bca_mean_interval
from psa.model import RWKV7Adapter, load_model_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _load_object(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _require_codes(value: Any) -> tuple[str, str, str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(code, str) and code for code in value)
        or len(set(value)) != 4
    ):
        raise ValueError("answer_codes must contain four distinct strings")
    return value[0], value[1], value[2], value[3]


def _require_thresholds(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("thresholds must be an object")
    fields = (
        "accuracy_lower_bound",
        "format_valid_rate",
        "max_answer_position_accuracy_gap",
        "max_infrastructure_failure_rate",
    )
    result = {}
    for field in fields:
        value = payload.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"thresholds.{field} must be in [0, 1]")
        result[field] = float(value)
    return result


def _copy_prompt(code: str) -> str:
    return (
        "COPY TASK.\n"
        f"TARGET CODE: {code}\n"
        "Reply with the target code only.\n"
        "ANSWER:"
    )


def _single_field_prompt(
    target_symbol: str,
    options: Sequence[tuple[str, str]],
) -> str:
    rendered_options = "\n".join(
        f"{code}. SYMBOL: {symbol}" for code, symbol in options
    )
    return (
        "TASK: exact one-field match.\n"
        "Choose the option whose SYMBOL equals CURRENT SYMBOL.\n"
        f"CURRENT SYMBOL: {target_symbol}\n"
        "OPTIONS:\n"
        f"{rendered_options}\n"
        "Reply with the matching option code only.\n"
        "ANSWER:"
    )


def generate_capability_manifest(
    *,
    answer_codes: Sequence[str],
    symbols: Sequence[str],
    repetitions: int,
    base_seed: int,
) -> dict[str, Any]:
    if len(answer_codes) != 4 or len(set(answer_codes)) != 4:
        raise ValueError("answer_codes must contain four distinct values")
    if len(symbols) != 4 or len(set(symbols)) != 4:
        raise ValueError("symbols must contain four distinct values")
    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")

    rng = random.Random(base_seed)
    trials = []
    for repetition in range(repetitions):
        block_id = f"block-{repetition:03d}"
        for code in answer_codes:
            prompt = _copy_prompt(code)
            trials.append(
                {
                    "sample_id": "diag-" + sha256_json(
                        {
                            "level": "copy_code",
                            "block": block_id,
                            "target_code": code,
                            "seed": base_seed,
                        }
                    )[:24],
                    "task_level": "copy_code",
                    "block_id": block_id,
                    "prompt": prompt,
                    "prompt_digest_sha256": sha256_json(prompt),
                    "target_code": code,
                    "target_symbol": None,
                    "option_mapping": None,
                }
            )

        shuffled_symbols = list(symbols)
        rng.shuffle(shuffled_symbols)
        options = list(zip(answer_codes, shuffled_symbols, strict=True))
        code_by_symbol = {symbol: code for code, symbol in options}
        for symbol in symbols:
            prompt = _single_field_prompt(symbol, options)
            trials.append(
                {
                    "sample_id": "diag-" + sha256_json(
                        {
                            "level": "single_field",
                            "block": block_id,
                            "target_symbol": symbol,
                            "mapping": options,
                            "seed": base_seed,
                        }
                    )[:24],
                    "task_level": "single_field",
                    "block_id": block_id,
                    "prompt": prompt,
                    "prompt_digest_sha256": sha256_json(prompt),
                    "target_code": code_by_symbol[symbol],
                    "target_symbol": symbol,
                    "option_mapping": [
                        {"code": code, "symbol": option_symbol}
                        for code, option_symbol in options
                    ],
                }
            )
    return {
        "manifest_version": "0.1",
        "development_only": True,
        "base_seed": base_seed,
        "repetitions": repetitions,
        "answer_codes": list(answer_codes),
        "symbols": list(symbols),
        "trial_count": len(trials),
        "trials": trials,
        "manifest_digest_sha256": sha256_json(trials),
    }


def evaluate_capability_level(
    *,
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    task_level: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    targets = {
        trial["sample_id"]: trial
        for trial in manifest["trials"]
        if trial["task_level"] == task_level
    }
    selected_records = [
        record for record in records if record["task_level"] == task_level
    ]
    outcomes = []
    failures = 0
    for record in selected_records:
        target = targets[record["sample_id"]]
        if record["status"] != "success":
            failures += 1
            continue
        outcomes.append(
            {
                "sample_id": record["sample_id"],
                "block_id": target["block_id"],
                "target_code": target["target_code"],
                "predicted_code": record["argmax_choice"],
                "correct": record["argmax_choice"] == target["target_code"],
                "format_valid": bool(record["format_valid"]),
            }
        )

    group_metrics = []
    for repetition in range(manifest["repetitions"]):
        block_id = f"block-{repetition:03d}"
        block = [item for item in outcomes if item["block_id"] == block_id]
        if len(block) != 4:
            continue
        group_metrics.append(
            {
                "block_id": block_id,
                "accuracy": sum(item["correct"] for item in block) / 4.0,
            }
        )
    accuracies = [item["accuracy"] for item in group_metrics]
    interval: tuple[float | None, float | None]
    if len(accuracies) >= 2:
        interval = bca_mean_interval(
            accuracies,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    else:
        interval = None, None

    denominator = len(selected_records)
    accuracy = (
        sum(item["correct"] for item in outcomes) / denominator
        if denominator
        else 0.0
    )
    format_valid_rate = (
        sum(item["format_valid"] for item in outcomes) / denominator
        if denominator
        else 0.0
    )
    infrastructure_failure_rate = (
        failures / denominator if denominator else 1.0
    )
    per_code = {}
    for code in manifest["answer_codes"]:
        items = [item for item in outcomes if item["target_code"] == code]
        per_code[code] = (
            sum(item["correct"] for item in items) / len(items)
            if items
            else 0.0
        )
    position_gap = max(per_code.values()) - min(per_code.values())
    checks = {
        "accuracy_lower_bound": bool(
            interval[0] is not None
            and interval[0] >= thresholds["accuracy_lower_bound"]
        ),
        "format_valid_rate": (
            format_valid_rate >= thresholds["format_valid_rate"]
        ),
        "answer_position_balance": (
            position_gap <= thresholds["max_answer_position_accuracy_gap"]
        ),
        "infrastructure_failure_rate": (
            infrastructure_failure_rate
            <= thresholds["max_infrastructure_failure_rate"]
        ),
    }
    return {
        "task_level": task_level,
        "planned_trial_count": len(targets),
        "successful_trial_count": len(outcomes),
        "failed_trial_count": failures,
        "complete_block_count": len(group_metrics),
        "accuracy": accuracy,
        "accuracy_interval": list(interval),
        "format_valid_rate": format_valid_rate,
        "infrastructure_failure_rate": infrastructure_failure_rate,
        "answer_position_accuracy": per_code,
        "max_answer_position_accuracy_gap": position_gap,
        "thresholds": thresholds,
        "checks": checks,
        "valid": all(checks.values()),
    }


def classify_capability_route(
    *,
    copy_valid: bool,
    single_field_valid: bool,
    two_field_valid: bool,
) -> str:
    if not copy_valid:
        return "revise_checkpoint_or_answer_interface"
    if not single_field_valid:
        return "revise_single_field_matching"
    if not two_field_valid:
        return "revise_compositional_matching"
    return "go_batch2"


def _verify_v02_evidence(root: Path, relative_path: str) -> dict[str, Any]:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("v0.2 evidence path must stay inside project root")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Impl-3 v0.2 evidence is missing: {path}")
    payload = _load_object(path, "Impl-3 v0.2 evidence")
    valid = bool(
        payload.get("gate") == "impl3_development"
        and payload.get("revision_id") == "prompt-template-v0.2"
        and payload.get("prompt_template_version") == "explicit-match-v0.2"
        and payload.get("batch0_valid") is True
        and payload.get("label_pool_valid") is True
        and payload.get("task_leakage_valid") is True
        and payload.get("task_tokenization_valid") is True
        and payload.get("resource_estimate_valid") is True
        and payload.get("decision") == "revise"
        and payload.get("prompt_visible_valid") is False
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "summary": payload,
        "valid": valid,
    }


def run_capability_ladder_gate(
    *,
    config_path: str | Path,
    gate_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    gate_path = Path(gate_config_path).resolve()
    gate_config = _load_object(gate_path, "capability ladder config")
    if gate_config.get("gate") != "impl3b_capability_ladder":
        raise ValueError("gate config is not for impl3b_capability_ladder")
    started_at = _utc_now()

    evidence = _verify_v02_evidence(
        root,
        str(gate_config.get("impl3_v02_summary")),
    )
    _write_json(destination / "impl3_v02_evidence.json", evidence)
    if not evidence["valid"]:
        raise RuntimeError("Impl-3 v0.2 evidence is incomplete or incompatible")

    selected_pairs = evidence["summary"].get("selected_identity_pairs")
    if (
        not isinstance(selected_pairs, list)
        or len(selected_pairs) != 2
        or not all(isinstance(pair, list) and len(pair) == 2 for pair in selected_pairs)
    ):
        raise ValueError("v0.2 summary does not contain two identity label pairs")
    symbols = tuple(str(label) for pair in selected_pairs for label in pair)
    if len(set(symbols)) != 4:
        raise ValueError("diagnostic symbols must be distinct")

    answer_codes = _require_codes(gate_config.get("answer_codes"))
    continuation_prefix = str(gate_config.get("answer_continuation_prefix", ""))
    repetitions = gate_config.get("repetitions")
    base_seed = gate_config.get("base_seed")
    bootstrap_replicates = gate_config.get("bootstrap_replicates")
    bootstrap_seed = gate_config.get("bootstrap_seed")
    max_generation_tokens = gate_config.get("max_generation_tokens")
    for field, value, minimum in (
        ("repetitions", repetitions, 2),
        ("base_seed", base_seed, 0),
        ("bootstrap_replicates", bootstrap_replicates, 100),
        ("bootstrap_seed", bootstrap_seed, 0),
        ("max_generation_tokens", max_generation_tokens, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    thresholds = _require_thresholds(gate_config.get("thresholds"))

    manifest = generate_capability_manifest(
        answer_codes=answer_codes,
        symbols=symbols,
        repetitions=repetitions,
        base_seed=base_seed,
    )
    _write_json(destination / "capability_manifest.json", manifest)

    model_config = load_model_config(config_path, root, verify_files=True)
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(model_config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    answer_report = inspect_answer_codes(
        adapter,
        answer_codes,
        continuation_prefix=continuation_prefix,
        require_equal_token_count=True,
    )
    _write_json(destination / "answer_interface_report.json", answer_report)
    if not answer_report["valid"]:
        raise RuntimeError("answer interface tokenization is invalid")

    rendered_answers = {
        code: f"{continuation_prefix}{code}" for code in answer_codes
    }
    records = []
    run_started = time.perf_counter()
    for trial in manifest["trials"]:
        trial_started = time.perf_counter()
        base_record = {
            "record_version": "0.1",
            "experiment_id": "EXP-001",
            "batch_id": "impl3b_capability_ladder",
            "run_id": "run-" + trial["sample_id"].removeprefix("diag-"),
            "sample_id": trial["sample_id"],
            "task_level": trial["task_level"],
            "block_id": trial["block_id"],
            "prompt_digest_sha256": trial["prompt_digest_sha256"],
        }
        try:
            scores, logits, state, prompt_token_count = score_continuations(
                adapter,
                trial["prompt"],
                rendered_answers,
            )
            format_probe = greedy_format_probe(
                adapter,
                logits,
                state,
                answer_codes=answer_codes,
                max_tokens=max_generation_tokens,
            )
            records.append(
                {
                    **base_record,
                    "prompt_token_count": prompt_token_count,
                    "option_scores": scores,
                    "option_probabilities": normalized_probabilities(scores),
                    "argmax_choice": max(scores, key=scores.__getitem__),
                    **format_probe,
                    "timing_seconds": time.perf_counter() - trial_started,
                    "status": "success",
                    "error": None,
                }
            )
        except Exception as exc:
            records.append(
                {
                    **base_record,
                    "prompt_token_count": None,
                    "option_scores": {},
                    "option_probabilities": {},
                    "argmax_choice": None,
                    "generated_token_ids": [],
                    "generated_text": "",
                    "generated_choice": None,
                    "format_valid": False,
                    "timing_seconds": time.perf_counter() - trial_started,
                    "status": "failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started
    write_jsonl(destination / "raw_capability_ladder.jsonl", records)

    copy_report = evaluate_capability_level(
        manifest=manifest,
        records=records,
        task_level="copy_code",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        thresholds=thresholds,
    )
    single_report = evaluate_capability_level(
        manifest=manifest,
        records=records,
        task_level="single_field",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        thresholds=thresholds,
    )
    ladder_report = {
        "report_version": "0.1",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "copy_code": copy_report,
        "single_field": single_report,
        "two_field": {
            "source": str(evidence["path"]),
            "prompt_template_version": evidence["summary"][
                "prompt_template_version"
            ],
            "metrics": evidence["summary"]["prompt_visible_metrics"],
            "valid": evidence["summary"]["prompt_visible_valid"],
        },
    }
    route = classify_capability_route(
        copy_valid=copy_report["valid"],
        single_field_valid=single_report["valid"],
        two_field_valid=bool(
            evidence["summary"]["prompt_visible_valid"]
        ),
    )
    ladder_report["route_decision"] = route
    ladder_report["capability_gate_passed"] = route == "go_batch2"
    _write_json(destination / "capability_ladder_report.json", ladder_report)

    diagnostic_valid = bool(
        evidence["valid"]
        and answer_report["valid"]
        and copy_report["failed_trial_count"] == 0
        and single_report["failed_trial_count"] == 0
    )
    summary = {
        "gate": "impl3b_capability_ladder",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "development_only": True,
        "model_id": model_config.model_id,
        "gate_config_sha256": sha256_file(gate_path),
        "impl3_v02_evidence_valid": evidence["valid"],
        "answer_interface_valid": answer_report["valid"],
        "copy_code_valid": copy_report["valid"],
        "single_field_valid": single_report["valid"],
        "two_field_valid": bool(
            evidence["summary"]["prompt_visible_valid"]
        ),
        "capability_gate_passed": route == "go_batch2",
        "route_decision": route,
        "copy_code_metrics": {
            "accuracy": copy_report["accuracy"],
            "accuracy_interval": copy_report["accuracy_interval"],
            "format_valid_rate": copy_report["format_valid_rate"],
            "answer_position_accuracy": copy_report[
                "answer_position_accuracy"
            ],
        },
        "single_field_metrics": {
            "accuracy": single_report["accuracy"],
            "accuracy_interval": single_report["accuracy_interval"],
            "format_valid_rate": single_report["format_valid_rate"],
            "answer_position_accuracy": single_report[
                "answer_position_accuracy"
            ],
        },
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "reports": [
            "impl3_v02_evidence.json",
            "answer_interface_report.json",
            "capability_manifest.json",
            "raw_capability_ladder.jsonl",
            "capability_ladder_report.json",
        ],
        "valid": diagnostic_valid,
    }
    _write_json(destination / "summary.json", summary)
    return summary
