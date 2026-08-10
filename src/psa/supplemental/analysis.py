from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.confirmatory.analysis import _describe, _semantic_condition_scores
from psa.evaluation.contrasts import joint_margin
from psa.preregistration import verify_core_set_package
from psa.supplemental.formal_run import (
    SUPPLEMENTAL_SET_DIGEST,
    SUPPLEMENTAL_SET_PACKAGE_DIGEST,
)
from psa.supplemental.set_generation import (
    PARENT_CORE_SET_DIGEST,
    PARENT_CORE_SET_PACKAGE_DIGEST,
    verify_exp001b_supplemental_set_package,
)


EXPECTED_GROUP_COUNT = 320
EXPECTED_SUPPLEMENTAL_RECORD_COUNT = 11_008
EXPECTED_PARENT_RAW_RECORD_COUNT = 40_960
EXPECTED_ANALYSIS_CONFIG_SHA256 = "2c49170f01e79a721fe93bb49856187c4b397ff51dc80bf0631eb88dd283d26a"
COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))
CONTROL_CONDITIONS = (
    "continuous",
    "restored",
    "reset",
    "random_matched",
    "swapped_I",
    "swapped_G",
    "swapped_both",
    "prompt_visible_reset",
)
ANALYSIS_SOURCE_FILES = (
    "configs/analysis/exp001b_supplemental_v1.json",
    "docs/exp001b_supplemental_analysis_plan.md",
    "src/psa/supplemental/analysis.py",
    "src/psa/supplemental/__init__.py",
    "src/psa/cli.py",
    "src/psa/confirmatory/analysis.py",
    "src/psa/evaluation/contrasts.py",
    "src/psa/evaluation/resampling.py",
    "scripts/analyze_exp001b_supplemental.sh",
    "tests/test_exp001b_analysis.py",
)


Combo = tuple[int, int]
Scores = dict[Combo, float]


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _combo(value: Any, label: str) -> Combo:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two binary values")
    result = int(value[0]), int(value[1])
    if result not in COMBOS:
        raise ValueError(f"{label} is not an I x G combination")
    return result


def _finite_option_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set("ABCD"):
        raise ValueError("option scores must contain exactly A-D")
    result = {code: float(value[code]) for code in "ABCD"}
    if not all(math.isfinite(score) for score in result.values()):
        raise ValueError("option scores must be finite")
    return result


def _semantic_rotated_scores(
    frozen_records: Sequence[Mapping[str, Any]],
    outputs_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[Combo, Scores]:
    buckets: dict[str, dict[str, Any]] = {}
    for frozen in frozen_records:
        record_id = str(frozen.get("record_id"))
        output = outputs_by_id.get(record_id)
        if output is None:
            raise ValueError("supplemental output is missing a frozen record")
        target_fields = frozen.get("target_fields")
        option_mapping = frozen.get("option_mapping")
        if not isinstance(target_fields, Mapping) or not isinstance(option_mapping, list):
            raise ValueError("frozen semantic record is incomplete")
        target = _combo(
            (target_fields.get("identity"), target_fields.get("goal")),
            "semantic target",
        )
        case_id = str(frozen.get("source_semantic_case_id"))
        bucket = buckets.setdefault(
            case_id,
            {"target": target, "rotations": set(), "scores": defaultdict(list)},
        )
        if bucket["target"] != target:
            raise ValueError("semantic case has inconsistent targets")
        bucket["rotations"].add(int(frozen.get("rotation_index")))
        option_scores = _finite_option_scores(output.get("option_scores"))
        for option in option_mapping:
            if not isinstance(option, Mapping):
                raise ValueError("option mapping entry must be an object")
            semantic = _combo(
                (option.get("identity"), option.get("goal")),
                "semantic option",
            )
            bucket["scores"][semantic].append(option_scores[str(option["code"])])
    result: dict[Combo, Scores] = {}
    for bucket in buckets.values():
        if bucket["rotations"] != {0, 1, 2, 3}:
            raise ValueError("semantic record is missing a code rotation")
        if set(bucket["scores"]) != set(COMBOS) or any(
            len(values) != 4 for values in bucket["scores"].values()
        ):
            raise ValueError("semantic rotation is incomplete")
        target = bucket["target"]
        if target in result:
            raise ValueError("duplicate semantic target in group")
        result[target] = {
            combo: mean(values) for combo, values in bucket["scores"].items()
        }
    if set(result) != set(COMBOS):
        raise ValueError("group does not cover all four semantic targets")
    return result


def _prefix_valid(output: Mapping[str, Any]) -> bool:
    metadata = output.get("metadata")
    prefix = metadata.get("forced_prefix") if isinstance(metadata, Mapping) else None
    return bool(
        isinstance(prefix, Mapping)
        and prefix.get("text") == ">\n"
        and prefix.get("greedy_exact") is True
        and prefix.get("roundtrip_exact") is True
    )


def analyze_supplemental_group(
    *,
    core_group: Mapping[str, Any],
    parent_payload: Mapping[str, Any],
    supplemental_payload: Mapping[str, Any],
    frozen_matched: Sequence[Mapping[str, Any]],
    frozen_generation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = supplemental_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("supplemental group records are missing")
    outputs_by_id = {
        str(item["record_id"]): item
        for item in records
        if isinstance(item, Mapping) and isinstance(item.get("record_id"), str)
    }
    if len(outputs_by_id) != len(records):
        raise ValueError("supplemental group contains duplicate record IDs")
    continuous = _semantic_condition_scores(core_group, parent_payload)["continuous"]
    matched = _semantic_rotated_scores(frozen_matched, outputs_by_id)
    matched_delta = mean(
        joint_margin(continuous[combo], combo) - joint_margin(matched[combo], combo)
        for combo in COMBOS
    )
    matched_outputs = [outputs_by_id[str(item["record_id"])] for item in frozen_matched]
    norm_alert_count = sum(
        int(item.get("metadata", {}).get("state_norm_alert_count", 0))
        for item in matched_outputs
    )
    max_alert_ratio = max(
        (
            float(item.get("metadata", {}).get("state_norm_max_alert_ratio", 0.0))
            for item in matched_outputs
        ),
        default=0.0,
    )

    generation_rows = []
    for frozen in frozen_generation:
        output = outputs_by_id[str(frozen["record_id"])]
        choice = output.get("generated_choice")
        target_code = str(frozen["target_code"])
        chosen = next(
            (
                item
                for item in frozen["option_mapping"]
                if isinstance(item, Mapping) and item.get("code") == choice
            ),
            None,
        )
        target = frozen["target_fields"]
        generation_rows.append(
            {
                "target_code": target_code,
                "format_valid": float(output.get("format_valid") is True),
                "prefix_valid": float(_prefix_valid(output)),
                "joint_correct": float(choice == target_code),
                "identity_correct": float(
                    isinstance(chosen, Mapping)
                    and chosen.get("identity") == target.get("identity")
                ),
                "goal_correct": float(
                    isinstance(chosen, Mapping)
                    and chosen.get("goal") == target.get("goal")
                ),
            }
        )
    if len(generation_rows) != 16:
        raise ValueError("each group requires 16 formal generation records")
    return {
        "factorial_group_id": core_group["factorial_group_id"],
        "matched_context": {
            "continuous_minus_matched_joint_margin": matched_delta,
            "state_norm_alert_count": norm_alert_count,
            "state_norm_max_alert_ratio": max_alert_ratio,
        },
        "generation": {
            key: mean(float(row[key]) for row in generation_rows)
            for key in (
                "format_valid",
                "prefix_valid",
                "joint_correct",
                "identity_correct",
                "goal_correct",
            )
        },
        "generation_position": {
            code: {
                "count": sum(row["target_code"] == code for row in generation_rows),
                "accuracy": mean(
                    row["joint_correct"]
                    for row in generation_rows
                    if row["target_code"] == code
                ),
            }
            for code in "ABCD"
        },
    }


def analyze_control_records(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"accuracy": [], "prefix_format": [], "target_log_probability": []}
    )
    for frozen, output in rows:
        condition = str(frozen.get("condition"))
        task_type = str(frozen.get("task_type"))
        if condition not in CONTROL_CONDITIONS:
            raise ValueError("control record has an unknown condition")
        scores = _finite_option_scores(output.get("option_scores"))
        target_code = str(frozen.get("target_code"))
        cell = cells[(condition, task_type)]
        cell["accuracy"].append(float(max(scores, key=scores.__getitem__) == target_code))
        cell["prefix_format"].append(float(_prefix_valid(output)))
        cell["target_log_probability"].append(scores[target_code])
    task_types = sorted({task for _, task in cells})
    baseline_condition = str(config["baseline_condition"])
    reports = []
    for condition in CONTROL_CONDITIONS:
        for task_type in task_types:
            values = cells.get((condition, task_type))
            baseline = cells.get((baseline_condition, task_type))
            if values is None or baseline is None or any(len(items) != 32 for items in values.values()):
                raise ValueError("control condition x task cell is incomplete")
            metrics = {key: mean(items) for key, items in values.items()}
            baseline_metrics = {key: mean(items) for key, items in baseline.items()}
            drops = {
                "accuracy": baseline_metrics["accuracy"] - metrics["accuracy"],
                "prefix_format": baseline_metrics["prefix_format"] - metrics["prefix_format"],
                "target_log_probability": (
                    baseline_metrics["target_log_probability"]
                    - metrics["target_log_probability"]
                ),
            }
            passes = {
                "accuracy": drops["accuracy"] <= float(config["maximum_accuracy_drop"]),
                "prefix_format": drops["prefix_format"]
                <= float(config["maximum_prefix_format_drop"]),
                "target_log_probability": drops["target_log_probability"]
                <= float(config["maximum_target_log_probability_drop"]),
            }
            reports.append(
                {
                    "condition": condition,
                    "task_type": task_type,
                    "sample_count": len(values["accuracy"]),
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                    "drops": drops,
                    "passes": passes,
                    "measured_valid": all(passes.values()),
                }
            )
    return {
        "cell_count": len(reports),
        "cells": reports,
        "measured_alerts_pass": all(item["measured_valid"] for item in reports),
        "state_norm_diagnostic": "not_recorded_in_frozen_control_outputs",
        "source_invariance_diagnostic": "not_recorded_in_frozen_control_outputs",
        "required_diagnostics_complete": False,
        "full_control_gate": False,
    }


def summarize_supplemental_analysis(
    groups: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = config["bootstrap"]
    matched_values = [
        float(group["matched_context"]["continuous_minus_matched_joint_margin"])
        for group in groups
    ]
    matched = _describe(
        matched_values,
        bootstrap=bootstrap,
        permutation=config["permutation"],
    )
    matched_alerts = sum(
        int(group["matched_context"]["state_norm_alert_count"]) for group in groups
    )
    matched_cfg = config["matched_context"]
    matched_go = bool(
        matched["mean"] >= float(matched_cfg["mean_joint_margin_advantage_sesoi"])
        and matched["confidence_interval"][0]
        > float(matched_cfg["confidence_interval_lower_must_exceed"])
        and matched["raw_p_value"] < float(matched_cfg["one_sided_p_below"])
        and (
            not matched_cfg.get("require_zero_state_norm_alerts", False)
            or matched_alerts == 0
        )
    )
    matched["state_norm_alert_count"] = matched_alerts
    matched["go"] = matched_go

    generation_reports = {
        metric: _describe(
            [float(group["generation"][metric]) for group in groups],
            bootstrap=bootstrap,
        )
        for metric in (
            "format_valid",
            "prefix_valid",
            "joint_correct",
            "identity_correct",
            "goal_correct",
        )
    }
    position_reports = {}
    for code in "ABCD":
        correct = sum(
            float(group["generation_position"][code]["accuracy"])
            * int(group["generation_position"][code]["count"])
            for group in groups
        )
        count = sum(int(group["generation_position"][code]["count"]) for group in groups)
        position_reports[code] = {"count": count, "accuracy": correct / count}
    position_gap = max(item["accuracy"] for item in position_reports.values()) - min(
        item["accuracy"] for item in position_reports.values()
    )
    generation_cfg = config["generation"]
    generation_go = bool(
        generation_reports["format_valid"]["mean"]
        >= float(generation_cfg["minimum_format_valid_rate"])
        and generation_reports["joint_correct"]["confidence_interval"][0]
        >= float(generation_cfg["minimum_joint_accuracy_ci_lower"])
        and generation_reports["identity_correct"]["confidence_interval"][0]
        >= float(generation_cfg["minimum_identity_accuracy_ci_lower"])
        and generation_reports["goal_correct"]["confidence_interval"][0]
        >= float(generation_cfg["minimum_goal_accuracy_ci_lower"])
        and position_gap
        <= float(generation_cfg["maximum_answer_position_accuracy_gap"])
        and generation_reports["prefix_valid"]["mean"]
        == float(generation_cfg["require_forced_prefix_greedy_exact_rate"])
    )
    generation = {
        "metrics": generation_reports,
        "answer_positions": position_reports,
        "maximum_answer_position_accuracy_gap": position_gap,
        "go": generation_go,
    }
    measured_controls_go = controls.get("measured_alerts_pass") is True
    diagnostics_complete = controls.get("required_diagnostics_complete") is True
    measured_package_go = matched_go and generation_go and measured_controls_go
    if not measured_package_go:
        gate_status = "revise_or_stop_measured_supplemental_control_failure"
        route = "review_frozen_failures_without_rerun"
        allowed = "supplemental_measured_controls_do_not_close_phase_2"
    elif not diagnostics_complete:
        gate_status = "not_assessable_no_full_go"
        route = "hold_phase_2_missing_frozen_control_diagnostics"
        allowed = "measured_supplemental_controls_pass_but_full_gate_remains_unassessable"
    else:
        gate_status = "go"
        route = "begin_phase_3_explicit_self_model"
        allowed = "exp001_and_exp001b_close_phase_2_behavioral_control_package"
    return {
        "matched_context": matched,
        "controls": dict(controls),
        "generation": generation,
        "measured_supplemental_package_go": measured_package_go,
        "required_diagnostics_complete": diagnostics_complete,
        "gate_2_single_variable_causal_transfer": {"status": gate_status},
        "gate_4_native_state_carrier_qualification": {"status": gate_status},
        "allowed_conclusion": allowed,
        "route_decision": route,
    }


def run_exp001b_supplemental_analysis(
    *,
    parent_raw_output_dir: str | Path,
    parent_raw_verification_path: str | Path,
    supplemental_raw_output_dir: str | Path,
    supplemental_raw_verification_path: str | Path,
    core_set_package_dir: str | Path,
    supplemental_set_package_dir: str | Path,
    analysis_config_path: str | Path,
    analysis_output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    parent_raw = Path(parent_raw_output_dir).resolve()
    supplemental_raw = Path(supplemental_raw_output_dir).resolve()
    core_root = Path(core_set_package_dir).resolve()
    set_root = Path(supplemental_set_package_dir).resolve()
    config_path = Path(analysis_config_path).resolve()
    destination = Path(analysis_output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("analysis output directory must be empty; results are immutable")
    if sha256_file(config_path) != EXPECTED_ANALYSIS_CONFIG_SHA256:
        raise ValueError("analysis config differs from the pinned pre-observation plan")
    config = _load_object(config_path, "EXP-001B analysis config")
    parent_verification = _load_object(parent_raw_verification_path, "parent raw verification")
    supplemental_verification = _load_object(
        supplemental_raw_verification_path, "supplemental raw verification"
    )
    parent_completion = _load_object(parent_raw / "completion.json", "parent completion")
    supplemental_completion = _load_object(
        supplemental_raw / "completion.json", "supplemental completion"
    )
    parent_manifest = _load_object(parent_raw / "manifest.json", "parent manifest")
    supplemental_manifest = _load_object(
        supplemental_raw / "manifest.json", "supplemental manifest"
    )
    core_set = _load_object(core_root / "core_set.json", "Core Set")
    core_manifest = _load_object(core_root / "manifest.json", "Core Set manifest")
    supplemental_set = _load_object(set_root / "supplemental_set.json", "supplemental set")
    supplemental_set_manifest = _load_object(
        set_root / "manifest.json", "supplemental set manifest"
    )
    core_report = verify_core_set_package(core_root)
    set_report = verify_exp001b_supplemental_set_package(set_root)
    expected_parent_digest = config["expected_parent_raw_payload_digest_sha256"]
    expected_supplemental_digest = config[
        "expected_supplemental_raw_payload_digest_sha256"
    ]
    prerequisites = {
        "parent_raw_verification_valid": (
            parent_verification.get("valid") is True
            and parent_verification.get("failed_checks") == []
            and parent_verification.get("failed_group_count") == 0
            and parent_verification.get("verified_record_count")
            == EXPECTED_PARENT_RAW_RECORD_COUNT
        ),
        "supplemental_raw_verification_valid": (
            supplemental_verification.get("valid") is True
            and supplemental_verification.get("status")
            == "raw_package_verified_unanalyzed"
            and supplemental_verification.get("failed_checks") == []
            and supplemental_verification.get("failed_group_count") == 0
            and supplemental_verification.get("verified_record_count")
            == EXPECTED_SUPPLEMENTAL_RECORD_COUNT
        ),
        "parent_raw_payload_bound": (
            parent_verification.get("group_payload_digest_sha256")
            == parent_completion.get("group_payload_digest_sha256")
            == parent_manifest.get("group_payload_digest_sha256")
            == expected_parent_digest
        ),
        "supplemental_raw_payload_bound": (
            supplemental_verification.get("group_payload_digest_sha256")
            == supplemental_completion.get("group_payload_digest_sha256")
            == supplemental_manifest.get("group_payload_digest_sha256")
            == expected_supplemental_digest
        ),
        "supplemental_results_unobserved": (
            supplemental_completion.get("supplemental_results_observed") is False
            and supplemental_verification.get("supplemental_results_observed") is False
        ),
        "core_package_valid": core_report.get("valid") is True,
        "supplemental_set_package_valid": set_report.get("valid") is True,
        "frozen_package_identity_valid": (
            core_set.get("core_set_digest_sha256") == PARENT_CORE_SET_DIGEST
            and core_manifest.get("core_set_package_digest_sha256")
            == PARENT_CORE_SET_PACKAGE_DIGEST
            and supplemental_set.get("supplemental_set_digest_sha256")
            == SUPPLEMENTAL_SET_DIGEST
            and supplemental_set_manifest.get("supplemental_set_package_digest_sha256")
            == SUPPLEMENTAL_SET_PACKAGE_DIGEST
        ),
        "design_counts_valid": (
            len(core_set.get("groups", [])) == EXPECTED_GROUP_COUNT
            and supplemental_completion.get("completed_group_count")
            == EXPECTED_GROUP_COUNT
            and supplemental_completion.get("raw_record_count")
            == EXPECTED_SUPPLEMENTAL_RECORD_COUNT
        ),
    }
    if not all(prerequisites.values()):
        failed = [key for key, value in prerequisites.items() if not value]
        raise ValueError(f"analysis prerequisites failed: {failed}")

    groups_by_id = {group["factorial_group_id"]: group for group in core_set["groups"]}
    frozen_records = supplemental_set["records"]
    matched_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    generation_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    frozen_by_id = {}
    for kind, destination_map in (
        ("matched_context", matched_by_group),
        ("formal_generation", generation_by_group),
    ):
        for record in frozen_records[kind]:
            destination_map[str(record["source_factorial_group_id"])].append(record)
            frozen_by_id[str(record["record_id"])] = record
    for record in frozen_records["controls"]:
        frozen_by_id[str(record["record_id"])] = record

    group_reports = []
    control_rows = []
    for group_id in supplemental_manifest["expected_group_ids"]:
        parent_payload = _load_object(parent_raw / "groups" / f"{group_id}.json", group_id)
        supplemental_payload = _load_object(
            supplemental_raw / "groups" / f"{group_id}.json", group_id
        )
        group_reports.append(
            analyze_supplemental_group(
                core_group=groups_by_id[group_id],
                parent_payload=parent_payload,
                supplemental_payload=supplemental_payload,
                frozen_matched=matched_by_group[group_id],
                frozen_generation=generation_by_group[group_id],
            )
        )
        for output in supplemental_payload["records"]:
            if output.get("record_kind") == "general_capability_control_condition":
                control_rows.append((frozen_by_id[str(output["record_id"])], output))
    if len(group_reports) != EXPECTED_GROUP_COUNT or len(control_rows) != 768:
        raise ValueError("analysis did not consume the complete frozen record set")
    controls = analyze_control_records(control_rows, config["controls"])
    aggregate = summarize_supplemental_analysis(group_reports, controls, config)
    source_digests = {path: sha256_file(root / path) for path in ANALYSIS_SOURCE_FILES}

    destination.mkdir(parents=True, exist_ok=True)
    groups_payload = {
        "group_report_version": "1.0",
        "experiment_id": "EXP-001B",
        "factorial_group_count": len(group_reports),
        "supplemental_results_observed": True,
        "groups": group_reports,
    }
    report_payload = {
        "report_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "EXP-001B",
        "analysis_id": config["analysis_id"],
        "analysis_config_sha256": EXPECTED_ANALYSIS_CONFIG_SHA256,
        "parent_raw_payload_digest_sha256": expected_parent_digest,
        "supplemental_raw_payload_digest_sha256": expected_supplemental_digest,
        "supplemental_results_observed": True,
        "analysis_read_only": True,
        "prerequisite_checks": prerequisites,
        **aggregate,
    }
    groups_path = destination / "group_level_metrics.json"
    report_path = destination / "supplemental_report.json"
    groups_path.write_bytes(canonical_json_bytes(groups_payload))
    report_path.write_bytes(canonical_json_bytes(report_payload))
    summary = {
        "summary_version": "1.0",
        "experiment_id": "EXP-001B",
        "status": "supplemental_analysis_complete",
        "valid": True,
        "supplemental_results_observed": True,
        "analysis_read_only": True,
        "analysis_config_sha256": EXPECTED_ANALYSIS_CONFIG_SHA256,
        "parent_raw_payload_digest_sha256": expected_parent_digest,
        "supplemental_raw_payload_digest_sha256": expected_supplemental_digest,
        "group_level_metrics_sha256": sha256_file(groups_path),
        "supplemental_report_sha256": sha256_file(report_path),
        "analysis_source_digests": source_digests,
        "gate_2_status": aggregate["gate_2_single_variable_causal_transfer"]["status"],
        "gate_4_status": aggregate["gate_4_native_state_carrier_qualification"]["status"],
        "allowed_conclusion": aggregate["allowed_conclusion"],
        "route_decision": aggregate["route_decision"],
        "reports": ["group_level_metrics.json", "supplemental_report.json"],
    }
    summary["analysis_package_digest_sha256"] = sha256_json(
        {
            "group_level_metrics.json": summary["group_level_metrics_sha256"],
            "supplemental_report.json": summary["supplemental_report_sha256"],
        }
    )
    (destination / "summary.json").write_bytes(canonical_json_bytes(summary))
    return summary
