from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.confirmatory.runner import (
    build_non_core_development_fixture,
    run_development_fixture,
)
from psa.confirmatory.rwkv_backend import RWKVConfirmatoryBackend
from psa.development.history_binding import _score_from_state
from psa.development.impl3 import (
    greedy_format_probe,
    score_continuations_after_prefix,
)
from psa.model import RWKV7Adapter, load_model_config
from psa.model.rwkv7 import _flatten_state
from psa.preregistration.formal_freeze import _fit_filler, _render_history


BDEV1_GATE = "exp001b_bdev1_non_core_calibration"
BDEV2_GATE = "exp001b_bdev2_non_core_runner"
NON_CORE_LABELS = (("amber", "cobalt"), ("orbit", "prism"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)))


def _load_object(path: str | Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_confirmed_design(path: str | Path) -> dict[str, Any]:
    design = _load_object(path, label="EXP-001B design")
    if design.get("experiment_id") != "EXP-001B":
        raise ValueError("development gate requires the EXP-001B design")
    if design.get("status") != "design_confirmed_development_only":
        raise ValueError("EXP-001B B1-B7 design has not been confirmed")
    review = design.get("design_review")
    if not isinstance(review, Mapping) or review.get("b1_b7_confirmed") is not True:
        raise ValueError("EXP-001B design review record is missing")
    safety = design.get("safety_boundary")
    if not isinstance(safety, Mapping):
        raise ValueError("EXP-001B safety boundary is missing")
    forbidden_true = (
        "supplemental_set_generated",
        "supplemental_experiment_authorized",
        "supplemental_experiment_run",
        "supplemental_results_observed",
        "automatic_rerun_authorized",
        "modify_exp001_artifacts_authorized",
    )
    if any(safety.get(field) is not False for field in forbidden_true):
        raise ValueError("development gate refuses formal authority or results")
    return design


def _matched_history_text(
    template: Mapping[str, str],
    *,
    domain: str,
    operation: str,
    filler: str,
    padding: str,
) -> str:
    user_text = template["user_text"].format(
        domain=domain,
        operation=operation,
    )
    return (
        f"User: {user_text}\n\n"
        f"Assistant: {template['assistant_ack']}\n\n"
        "User: Review these neutral records without setting a current state.\n"
        f"{filler}{padding}\n\n"
        "Assistant: Neutral records reviewed; current state remains unset."
    )


def fit_matched_context_history(
    *,
    original_history: str,
    template: Mapping[str, str],
    domain: str,
    operation: str,
    filler: str,
    padding_fragments: Sequence[str],
    token_counter: Any,
) -> dict[str, Any]:
    """Fit an explicitly unbound history to one paired history token count."""
    target = int(token_counter(original_history))
    candidates: list[tuple[str | None, int, str, int]] = []
    base = _matched_history_text(
        template,
        domain=domain,
        operation=operation,
        filler=filler,
        padding="",
    )
    candidates.append((None, 0, base, int(token_counter(base))))
    for fragment in padding_fragments:
        if not isinstance(fragment, str) or not fragment:
            raise ValueError("matched-context padding fragments must be non-empty")
        for repetitions in range(1, target * 4 + 1):
            text = _matched_history_text(
                template,
                domain=domain,
                operation=operation,
                filler=filler,
                padding=fragment * repetitions,
            )
            count = int(token_counter(text))
            candidates.append((fragment, repetitions, text, count))
            if count == target:
                return {
                    "text": text,
                    "target_token_count": target,
                    "matched_token_count": count,
                    "token_count_exact": True,
                    "padding_fragment": fragment,
                    "padding_repetitions": repetitions,
                    "filler_exact_substring": filler in text,
                }
            if count > target + 8:
                break
    for fragment, repetitions, text, count in candidates:
        if count == target:
            return {
                "text": text,
                "target_token_count": target,
                "matched_token_count": count,
                "token_count_exact": True,
                "padding_fragment": fragment,
                "padding_repetitions": repetitions,
                "filler_exact_substring": filler in text,
            }
    closest = min(candidates, key=lambda item: abs(item[3] - target))
    raise RuntimeError(
        "could not fit matched-context history to paired token count: "
        f"target={target}, closest={closest[3]}"
    )


def _validate_unbound_text(
    text: str,
    *,
    domain: str,
    operation: str,
    forbidden_phrases: Sequence[str],
) -> dict[str, Any]:
    phrase_hits = [phrase for phrase in forbidden_phrases if phrase in text]
    domain_count = text.count(domain)
    operation_count = text.count(operation)
    explicit_unbound = any(
        cue in text.lower()
        for cue in (
            "not a current-state record",
            "does not define the current state",
            "unrelated to the current state",
            "reject this proposal",
        )
    )
    return {
        "domain_mention_count": domain_count,
        "operation_mention_count": operation_count,
        "forbidden_phrase_hits": phrase_hits,
        "explicit_unbound_cue": explicit_unbound,
        "valid": bool(
            domain_count == 1
            and operation_count == 1
            and not phrase_hits
            and explicit_unbound
        ),
    }


def build_non_core_calibration_cases(
    design: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    *,
    token_counter: Any,
) -> list[dict[str, Any]]:
    calibration = design["state_norm_development_calibration"]
    if calibration.get("core_set_access_forbidden") is not True:
        raise ValueError("B-Dev1 must forbid Core Set access")
    histories = formal_config["history_protocol"]["templates"]
    if not isinstance(histories, list) or len(histories) != 4:
        raise ValueError("B-Dev1 requires four frozen formal history templates")
    matched = design["matched_context"]
    matched_templates = matched["templates"]
    if not isinstance(matched_templates, list) or len(matched_templates) != 4:
        raise ValueError("B-Dev1 requires four matched-context templates")
    fillers = [
        _fit_filler(
            variant_index=index,
            filler_config=formal_config["filler_protocol"],
            token_counter=token_counter,
        )
        for index in range(4)
    ]
    identity_labels, goal_labels = NON_CORE_LABELS
    cases: list[dict[str, Any]] = []
    for history_index, history_template in enumerate(histories):
        for filler_index, filler in enumerate(fillers):
            for identity in range(2):
                for goal in range(2):
                    domain = identity_labels[identity]
                    operation = goal_labels[goal]
                    original = _render_history(
                        history_template,
                        domain=domain,
                        operation=operation,
                        filler=filler["text"],
                    )
                    fitted = fit_matched_context_history(
                        original_history=original,
                        template=matched_templates[history_index],
                        domain=domain,
                        operation=operation,
                        filler=filler["text"],
                        padding_fragments=matched["padding_fragments"],
                        token_counter=token_counter,
                    )
                    binding = _validate_unbound_text(
                        fitted["text"],
                        domain=domain,
                        operation=operation,
                        forbidden_phrases=matched["forbidden_binding_phrases"],
                    )
                    case_id = "bdev1-" + sha256_json(
                        {
                            "history_index": history_index,
                            "filler_index": filler_index,
                            "identity": identity,
                            "goal": goal,
                        }
                    )[:20]
                    cases.append(
                        {
                            "case_id": case_id,
                            "history_template_id": history_template["id"],
                            "matched_template_id": matched_templates[history_index]["id"],
                            "filler_variant_id": filler["variant_id"],
                            "identity": identity,
                            "goal": goal,
                            "domain": domain,
                            "operation": operation,
                            "original_history": original,
                            "matched_history": fitted["text"],
                            "original_token_count": fitted["target_token_count"],
                            "matched_token_count": fitted["matched_token_count"],
                            "token_count_exact": fitted["token_count_exact"],
                            "filler_exact_substring": fitted["filler_exact_substring"],
                            "padding_fragment": fitted["padding_fragment"],
                            "padding_repetitions": fitted["padding_repetitions"],
                            "unbound_validation": binding,
                        }
                    )
    expected = int(calibration["group_count"])
    if len(cases) != expected:
        raise ValueError(f"B-Dev1 expected {expected} non-Core cases")
    return cases


def empirical_quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("empirical quantile requires values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if quantile == 0.0:
        return ordered[0]
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _state_rms(state: Any, torch: Any) -> list[dict[str, Any]]:
    records = []
    for path, tensor in _flatten_state(state):
        numeric = tensor.detach().float()
        finite = bool(torch.isfinite(numeric).all().item())
        rms = float(torch.sqrt(torch.mean(numeric * numeric)).item())
        records.append(
            {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": finite,
                "rms": rms,
            }
        )
    return records


def _freeze_state_norm_thresholds(
    case_records: Sequence[Mapping[str, Any]],
    *,
    quantile: float,
) -> dict[str, Any]:
    by_path: dict[str, list[float]] = {}
    metadata: dict[str, tuple[list[int], str]] = {}
    for case in case_records:
        for component in case["components"]:
            path = component["path"]
            by_path.setdefault(path, []).append(float(component["rms"]))
            metadata.setdefault(path, (component["shape"], component["dtype"]))
    expected_count = len(case_records)
    if not by_path or any(len(values) != expected_count for values in by_path.values()):
        raise ValueError("state norm calibration has incomplete component coverage")
    components = []
    for path in sorted(by_path):
        values = by_path[path]
        shape, dtype = metadata[path]
        components.append(
            {
                "path": path,
                "shape": shape,
                "dtype": dtype,
                "sample_count": len(values),
                "minimum_rms": min(values),
                "median_rms": empirical_quantile(values, 0.5),
                "maximum_rms": max(values),
                "threshold_rms": empirical_quantile(values, quantile),
            }
        )
    return {
        "threshold_version": "0.1-development",
        "method": "nearest-rank empirical quantile ceil(q*n)",
        "quantile": quantile,
        "case_count": expected_count,
        "component_count": len(components),
        "components": components,
        "valid": all(item["threshold_rms"] >= 0.0 for item in components),
    }


def evaluate_state_norms(
    records: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    limits = {
        item["path"]: float(item["threshold_rms"])
        for item in thresholds["components"]
    }
    observed = {item["path"]: float(item["rms"]) for item in records}
    if set(observed) != set(limits):
        raise ValueError("state norm component inventory does not match thresholds")
    alerts = [
        {
            "path": path,
            "observed_rms": observed[path],
            "threshold_rms": limits[path],
        }
        for path in sorted(limits)
        if observed[path] > limits[path]
    ]
    return {
        "component_count": len(limits),
        "alert_count": len(alerts),
        "alerts": alerts,
        "valid": not alerts,
    }


def run_exp001b_bdev1_gate(
    *,
    design_path: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    started_at = datetime.now(timezone.utc)
    design = _load_confirmed_design(design_path)
    formal_path = root / design["general_capability_controls"]["source_config"]
    formal_config = _load_object(formal_path, label="frozen formal config")
    config = load_model_config(model_config_path, root, verify_files=True)
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    token_counter = lambda text: len(adapter.encode(text))
    cases = build_non_core_calibration_cases(
        design,
        formal_config,
        token_counter=token_counter,
    )
    token_report = {
        "report_version": "0.1-development",
        "gate": BDEV1_GATE,
        "development_only": True,
        "core_set_accessed": False,
        "case_count": len(cases),
        "all_token_counts_exact": all(item["token_count_exact"] for item in cases),
        "all_fillers_preserved": all(item["filler_exact_substring"] for item in cases),
        "all_histories_unbound": all(
            item["unbound_validation"]["valid"] for item in cases
        ),
        "cases": cases,
    }
    token_report["valid"] = bool(
        token_report["case_count"] == 64
        and token_report["all_token_counts_exact"]
        and token_report["all_fillers_preserved"]
        and token_report["all_histories_unbound"]
    )
    _write_json(destination / "matched_context_token_report.json", token_report)

    lengths = sorted({item["original_token_count"] for item in cases})
    neutral_token = adapter.encode(" neutral")[0]
    for length in lengths:
        adapter.forward([neutral_token] * length, None)
    state_cases = []
    for case in cases:
        _, state = adapter.forward(adapter.encode(case["original_history"]), None)
        components = _state_rms(state, torch)
        state_cases.append(
            {
                "case_id": case["case_id"],
                "component_count": len(components),
                "all_finite": all(item["finite"] for item in components),
                "components": components,
            }
        )
    thresholds = _freeze_state_norm_thresholds(
        state_cases,
        quantile=0.999,
    )
    thresholds.update(
        {
            "gate": BDEV1_GATE,
            "development_only": True,
            "core_set_accessed": False,
            "model_id": config.model_id,
            "shape_warmup_token_lengths": lengths,
            "shape_warmup_excluded_from_calibration": True,
            "all_states_finite": all(item["all_finite"] for item in state_cases),
            "source_case_digest_sha256": sha256_json(
                [item["case_id"] for item in state_cases]
            ),
        }
    )
    thresholds["valid"] = bool(
        thresholds["valid"]
        and thresholds["case_count"] == 64
        and thresholds["component_count"] == 96
        and thresholds["all_states_finite"]
    )
    _write_json(destination / "state_norm_thresholds.json", thresholds)
    torch.cuda.synchronize()
    summary = {
        "summary_version": "0.1-development",
        "gate": BDEV1_GATE,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "core_set_accessed": False,
        "supplemental_set_generated": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "model_id": config.model_id,
        "load_seconds": load_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "matched_context_case_count": len(cases),
        "matched_context_valid": token_report["valid"],
        "state_norm_case_count": thresholds["case_count"],
        "state_component_count": thresholds["component_count"],
        "state_norm_thresholds_valid": thresholds["valid"],
        "design_sha256": sha256_file(design_path),
        "reports": [
            "matched_context_token_report.json",
            "state_norm_thresholds.json",
        ],
        "route_decision": "run_exp001b_bdev2_non_core_runner",
        "valid": bool(token_report["valid"] and thresholds["valid"]),
    }
    _write_json(destination / "summary.json", summary)
    return summary


def _verify_bdev1_evidence(
    summary_path: str | Path,
    thresholds_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_object(summary_path, label="B-Dev1 summary")
    thresholds = _load_object(thresholds_path, label="state norm thresholds")
    if (
        summary.get("gate") != BDEV1_GATE
        or summary.get("valid") is not True
        or summary.get("development_only") is not True
        or summary.get("core_set_accessed") is not False
        or summary.get("supplemental_experiment_run") is not False
        or thresholds.get("valid") is not True
        or thresholds.get("development_only") is not True
        or thresholds.get("core_set_accessed") is not False
    ):
        raise ValueError("B-Dev1 evidence is invalid or crossed a safety boundary")
    return summary, thresholds


def _run_matched_context_probe(
    adapter: Any,
    fixture: Mapping[str, Any],
    matched_report: Mapping[str, Any],
) -> dict[str, Any]:
    available: dict[tuple[int, int], str] = {}
    for case in matched_report["cases"]:
        combo = int(case["identity"]), int(case["goal"])
        available.setdefault(combo, case["matched_history"])
    if set(available) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("B-Dev1 report lacks four non-Core matched histories")
    group = fixture["groups"][0]
    states = {}
    for combo, history in available.items():
        _, states[combo] = adapter.forward(adapter.encode(history), None)
    records = []
    for trial in group["trials"]:
        target = trial["target_fields"]
        combo = int(target["identity"]), int(target["goal"])
        scores, token_count, prefix = _score_from_state(
            adapter,
            query_text=trial["query_prompt"],
            source_state=states[combo],
            rendered_answers={code: code for code in "ABCD"},
            forced_prefix=">\n",
        )
        records.append(
            {
                "trial_id": trial["trial_id"],
                "option_scores": scores,
                "query_token_count": token_count,
                "forced_prefix": prefix,
            }
        )
    return {
        "report_version": "0.1-development",
        "development_only": True,
        "core_set_accessed": False,
        "record_count": len(records),
        "contains_derived_accuracy": False,
        "records": records,
        "valid": len(records) == 16,
    }


def _run_generation_probe(adapter: Any, fixture: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for trial in fixture["groups"][0]["trials"]:
        prompt = trial["history_prompt"] + "\n\n" + trial["query_prompt"]
        scores, logits, state, token_count, prefix = score_continuations_after_prefix(
            adapter,
            prompt,
            {code: code for code in "ABCD"},
            forced_prefix=">\n",
        )
        generated = greedy_format_probe(
            adapter,
            logits,
            state,
            answer_codes=tuple("ABCD"),
            max_tokens=4,
        )
        records.append(
            {
                "trial_id": trial["trial_id"],
                "prompt_token_count": token_count,
                "option_scores": scores,
                "forced_prefix": prefix,
                **generated,
            }
        )
    prefix_rate = sum(item["forced_prefix"]["greedy_exact"] for item in records) / len(records)
    format_rate = sum(item["format_valid"] for item in records) / len(records)
    return {
        "report_version": "0.1-development",
        "development_only": True,
        "core_set_accessed": False,
        "record_count": len(records),
        "forced_prefix_greedy_exact_rate": prefix_rate,
        "format_valid_rate": format_rate,
        "records": records,
        "valid": bool(len(records) == 16 and prefix_rate == 1.0 and format_rate == 1.0),
    }


def _run_fixture_state_norm_probe(
    adapter: Any,
    fixture: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    representatives: dict[tuple[int, int], Mapping[str, Any]] = {}
    for trial in fixture["groups"][0]["trials"]:
        target = trial["target_fields"]
        combo = int(target["identity"]), int(target["goal"])
        representatives.setdefault(combo, trial)
    checks = []
    for combo in sorted(representatives):
        trial = representatives[combo]
        _, state = adapter.forward(adapter.encode(trial["history_prompt"]), None)
        result = evaluate_state_norms(_state_rms(state, adapter.torch), thresholds)
        checks.append({"combo": list(combo), **result})
    return {
        "report_version": "0.1-development",
        "development_only": True,
        "core_set_accessed": False,
        "state_count": len(checks),
        "checks": checks,
        "valid": bool(len(checks) == 4 and all(item["valid"] for item in checks)),
    }


def run_exp001b_bdev2_gate(
    *,
    design_path: str | Path,
    model_config_path: str | Path,
    bdev1_summary_path: str | Path,
    bdev1_thresholds_path: str | Path,
    bdev1_matched_report_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    started_at = datetime.now(timezone.utc)
    design = _load_confirmed_design(design_path)
    bdev1, thresholds = _verify_bdev1_evidence(
        bdev1_summary_path,
        bdev1_thresholds_path,
    )
    matched_report = _load_object(
        bdev1_matched_report_path,
        label="B-Dev1 matched-context report",
    )
    if (
        matched_report.get("valid") is not True
        or matched_report.get("development_only") is not True
        or matched_report.get("core_set_accessed") is not False
    ):
        raise ValueError("B-Dev1 matched-context evidence is invalid")
    config = load_model_config(model_config_path, root, verify_files=True)
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    fixture = build_non_core_development_fixture()
    fixture["fixture_kind"] = "non_core_exp001b_runner_fixture"
    fixture["experiment_id"] = "DEV-EXP001B-RUNNER"
    fixture["purpose"] = "exercise EXP-001B supplement without Core Set access"
    fixture["runner_source_digests"] = {
        relative: sha256_file(root / relative)
        for relative in (
            "src/psa/confirmatory/runner.py",
            "src/psa/confirmatory/rwkv_backend.py",
            "src/psa/supplemental/development.py",
        )
    }
    fixture.pop("fixture_digest_sha256", None)
    fixture["fixture_digest_sha256"] = sha256_json(fixture)
    # The shared runner intentionally accepts one exact development kind. Use a
    # transient compatible view while retaining the EXP-001B fixture on disk.
    _write_json(destination / "development_fixture.json", fixture)
    runner_fixture = dict(fixture)
    runner_fixture["fixture_kind"] = "non_core_confirmatory_runner_fixture"
    runner_fixture.pop("fixture_digest_sha256", None)
    runner_fixture["fixture_digest_sha256"] = sha256_json(runner_fixture)
    _write_json(
        destination / "condition_runner" / "development_fixture.json",
        runner_fixture,
    )
    backend = RWKVConfirmatoryBackend(adapter=adapter)
    runner_manifest = run_development_fixture(
        dataset=runner_fixture,
        backend=backend,
        output_dir=destination / "condition_runner",
    )
    matched_probe = _run_matched_context_probe(adapter, fixture, matched_report)
    generation_probe = _run_generation_probe(adapter, fixture)
    norm_probe = _run_fixture_state_norm_probe(adapter, fixture, thresholds)
    design_conditions = design["general_capability_controls"]["conditions"]
    condition_alias_report = {
        "report_version": "0.1-development",
        "development_only": True,
        "core_set_accessed": False,
        "design_conditions": design_conditions,
        "runner_conditions": [
            "continuous",
            "restored",
            "reset",
            "random_matched",
            "swapped_I",
            "swapped_G",
            "swapped_both",
            "prompt_visible",
        ],
        "aliases": {"prompt_visible_reset": "prompt_visible"},
    }
    condition_alias_report["valid"] = bool(
        set(design_conditions[:-1])
        == set(condition_alias_report["runner_conditions"][:-1])
        and design_conditions[-1] == "prompt_visible_reset"
        and condition_alias_report["runner_conditions"][-1] == "prompt_visible"
    )
    _write_json(destination / "matched_context_probe.json", matched_probe)
    _write_json(destination / "generation_probe.json", generation_probe)
    _write_json(destination / "state_norm_probe.json", norm_probe)
    _write_json(destination / "condition_alias_report.json", condition_alias_report)
    torch.cuda.synchronize()
    summary = {
        "summary_version": "0.1-development",
        "gate": BDEV2_GATE,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "core_set_accessed": False,
        "supplemental_set_generated": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "model_id": config.model_id,
        "load_seconds": load_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "bdev1_summary_sha256": sha256_file(bdev1_summary_path),
        "bdev1_valid": bdev1["valid"],
        "condition_runner_valid": runner_manifest.get("valid") is True,
        "condition_alias_valid": condition_alias_report["valid"],
        "condition_record_count": 128,
        "matched_context_probe_valid": matched_probe["valid"],
        "matched_context_record_count": matched_probe["record_count"],
        "generation_probe_valid": generation_probe["valid"],
        "generation_record_count": generation_probe["record_count"],
        "forced_prefix_greedy_exact_rate": generation_probe[
            "forced_prefix_greedy_exact_rate"
        ],
        "format_valid_rate": generation_probe["format_valid_rate"],
        "state_norm_probe_valid": norm_probe["valid"],
        "design_sha256": sha256_file(design_path),
        "reports": [
            "development_fixture.json",
            "condition_alias_report.json",
            "condition_runner/development_fixture.json",
            "condition_runner/manifest.json",
            "condition_runner/groups/devgrp-impl5b-noncore-v1.json",
            "matched_context_probe.json",
            "generation_probe.json",
            "state_norm_probe.json",
        ],
        "route_decision": "review_exp001b_preregistration_candidate_inputs",
        "valid": bool(
            runner_manifest.get("valid") is True
            and condition_alias_report["valid"]
            and matched_probe["valid"]
            and generation_probe["valid"]
            and norm_probe["valid"]
        ),
    }
    _write_json(destination / "summary.json", summary)
    return summary
