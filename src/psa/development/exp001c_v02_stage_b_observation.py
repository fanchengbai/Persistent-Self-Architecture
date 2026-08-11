from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)
from psa.development.exp001c_v02_stage_b_run import (
    verify_exp001c_v02_stage_b_result,
)


OBSERVATION_VERSION = "0.1-exploratory"
SEMANTIC_CONDITIONS = (
    "continuous",
    "restored",
    "swapped_I",
    "swapped_G",
    "swapped_both",
)
DIAGNOSTIC_CONDITIONS = ("reset", "random_matched")
OBSERVATION_SOURCE_FILES = (
    "configs/analysis/exp001c_v02_stage_b_observation_v01.json",
    "schemas/exp001c_v02_stage_b_observation.schema.json",
    "scripts/analyze_exp001c_v02_stage_b.py",
    "src/psa/development/exp001c_v02_stage_b_observation.py",
    "tests/test_exp001c_v02_stage_b_observation.py",
)


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("analysis_version") != OBSERVATION_VERSION
        or config.get("experiment_id") != "EXP-001C"
        or config.get("scope") != "v02_stage_b_noncore_observation"
        or config.get("expected_record_count") != 224
        or config.get("expected_condition_count") != 7
        or config.get("rotation_count_per_semantic_case") != 4
        or config.get("semantic_conditions") != list(SEMANTIC_CONDITIONS)
        or config.get("diagnostic_conditions") != list(DIAGNOSTIC_CONDITIONS)
        or config.get("readout")
        != "mean_log_score_by_semantic_option_across_four_code_rotations"
        or config.get("result_observation_authorized") is not True
        or config.get("confirmatory_decision_authorized") is not False
        or config.get("formal_test_set_access_authorized") is not False
        or config.get("formal_run_authorized") is not False
        or config.get("automatic_rerun_authorized") is not False
        or config.get("thresholds_or_go_no_go_decisions") is not False
        or not isinstance(config.get("expected_stage_b_result_sha256"), str)
        or len(str(config.get("expected_stage_b_result_sha256"))) != 64
    ):
        raise ValueError("Stage B observation config violates the frozen boundary")


def _fields(value: Mapping[str, Any]) -> tuple[str, str]:
    return str(value["domain"]), str(value["operation"])


def _field_object(value: tuple[str, str]) -> dict[str, str]:
    return {"domain": value[0], "operation": value[1]}


def _condition_report(
    *,
    condition: str,
    records: list[Mapping[str, Any]],
    trials_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["semantic_case_id"])].append(record)
    if len(groups) != 8:
        raise ValueError(f"{condition} must contain eight semantic cases")

    case_reports = []
    raw_correct = 0
    predicted_codes: Counter[str] = Counter()
    for record in records:
        predicted_codes[str(record["predicted_code"])] += 1
        expected_code = record.get("expected_state_semantic_target_code")
        if expected_code is not None and record.get("predicted_code") == expected_code:
            raw_correct += 1

    for case_id, case_records in sorted(groups.items()):
        if (
            len(case_records) != 4
            or {int(item["rotation_index"]) for item in case_records}
            != {0, 1, 2, 3}
        ):
            raise ValueError(f"{condition}/{case_id} lacks four rotations")
        semantic_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        reference_fields = set()
        target_fields = set()
        for record in case_records:
            trial = trials_by_id[str(record["query_sample_id"])]
            reference_fields.add(_fields(trial["target_fields"]))
            source_fields = record.get("state_source_fields")
            if isinstance(source_fields, Mapping):
                target_fields.add(_fields(source_fields))
            scores = record["option_log_probabilities"]
            for option in trial["option_mapping"]:
                semantic = str(option["domain"]), str(option["operation"])
                score = float(scores[str(option["code"])])
                if not math.isfinite(score):
                    raise ValueError("Stage B observation received a non-finite score")
                semantic_scores[semantic].append(score)
        if len(reference_fields) != 1 or any(len(values) != 4 for values in semantic_scores.values()):
            raise ValueError(f"{condition}/{case_id} semantic mapping is incomplete")
        if condition in SEMANTIC_CONDITIONS and len(target_fields) != 1:
            raise ValueError(f"{condition}/{case_id} target fields drifted")
        means = {semantic: mean(values) for semantic, values in semantic_scores.items()}
        prediction = max(sorted(means), key=lambda key: means[key])
        reference = next(iter(reference_fields))
        target = next(iter(target_fields)) if target_fields else None
        scored_target = target if target is not None else reference
        other_scores = [value for key, value in means.items() if key != scored_target]
        case_reports.append(
            {
                "semantic_case_id": case_id,
                "prediction": _field_object(prediction),
                "target_fields": _field_object(target) if target is not None else None,
                "reference_stage_a_fields": _field_object(reference),
                "joint_match": prediction == target if target is not None else None,
                "domain_match": prediction[0] == target[0] if target is not None else None,
                "operation_match": prediction[1] == target[1] if target is not None else None,
                "diagnostic_reference_match": prediction == reference,
                "semantic_target_or_reference_margin": (
                    means[scored_target] - max(other_scores)
                ),
                "semantic_scores": [
                    {
                        **_field_object(key),
                        "mean_log_score": means[key],
                        "rotation_count": 4,
                    }
                    for key in sorted(means)
                ],
            }
        )

    predictions = Counter(
        (item["prediction"]["domain"], item["prediction"]["operation"])
        for item in case_reports
    )
    semantic = condition in SEMANTIC_CONDITIONS
    return {
        "condition": condition,
        "role": "state_semantic_primary" if semantic else "diagnostic_control",
        "record_count": len(records),
        "semantic_case_count": len(case_reports),
        "raw_code_accuracy": raw_correct / len(records) if semantic else None,
        "label_marginalized_joint_accuracy": (
            sum(item["joint_match"] is True for item in case_reports) / len(case_reports)
            if semantic
            else None
        ),
        "label_marginalized_domain_accuracy": (
            sum(item["domain_match"] is True for item in case_reports) / len(case_reports)
            if semantic
            else None
        ),
        "label_marginalized_operation_accuracy": (
            sum(item["operation_match"] is True for item in case_reports) / len(case_reports)
            if semantic
            else None
        ),
        "diagnostic_reference_match_rate": (
            sum(item["diagnostic_reference_match"] for item in case_reports)
            / len(case_reports)
            if not semantic
            else None
        ),
        "mean_semantic_target_or_reference_margin": mean(
            item["semantic_target_or_reference_margin"] for item in case_reports
        ),
        "predicted_code_counts": dict(sorted(predicted_codes.items())),
        "predicted_semantic_counts": {
            f"{key[0]}|{key[1]}": value for key, value in sorted(predictions.items())
        },
        "cases": case_reports,
    }


def analyze_exp001c_v02_stage_b(
    *,
    analysis_config_path: str | Path,
    design_manifest_path: str | Path,
    stage_b_result_path: str | Path,
    stage_b_summary_path: str | Path,
    protocol_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = _resolve(analysis_config_path, root)
    design_path = _resolve(design_manifest_path, root)
    result_path = _resolve(stage_b_result_path, root)
    summary_path = _resolve(stage_b_summary_path, root)
    config = _object(config_path, "Stage B observation config")
    _validate_config(config)
    result_sha = sha256_file(result_path)
    if result_sha != config["expected_stage_b_result_sha256"]:
        raise ValueError("Stage B result digest does not match the frozen observation plan")
    verification = verify_exp001c_v02_stage_b_result(
        result_path=result_path,
        design_manifest_path=design_path,
        project_root=root,
    )
    if verification.get("valid") is not True:
        raise ValueError("Stage B raw result verification failed")
    summary = _object(summary_path, "Stage B summary")
    if not (
        summary.get("valid") is True
        and summary.get("status") == "stage_b_raw_result_complete_verified_unobserved"
        and summary.get("stage_b_result_observation_authorized") is True
        and summary.get("stage_b_result_sha256") == result_sha
        and summary.get("record_count") == 224
        and summary.get("stage_a_rerun") is False
        and summary.get("formal_test_set_accessed") is False
        and summary.get("formal_run") is False
        and summary.get("automatic_rerun_authorized") is False
    ):
        raise PermissionError("Stage B result observation authority is invalid")
    protocol = build_exp001c_protocol_v02_manifest(
        config_path=_resolve(protocol_config_path, root),
        project_root=root,
    )
    result = _object(result_path, "Stage B raw result")
    design = _object(design_path, "Stage B design manifest")
    if protocol.get("manifest_digest_sha256") != design.get("protocol_manifest_digest_sha256"):
        raise ValueError("Stage B protocol digest drifted")
    trials_by_id = {str(item["sample_id"]): item for item in protocol["trials"]}
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in result["records"]:
        by_condition[str(record["condition"])].append(record)
    if set(by_condition) != set(SEMANTIC_CONDITIONS + DIAGNOSTIC_CONDITIONS):
        raise ValueError("Stage B condition inventory is invalid")
    condition_reports = {
        condition: _condition_report(
            condition=condition,
            records=by_condition[condition],
            trials_by_id=trials_by_id,
        )
        for condition in SEMANTIC_CONDITIONS + DIAGNOSTIC_CONDITIONS
    }
    predictions = {
        condition: {
            item["semantic_case_id"]: item["prediction"]
            for item in condition_reports[condition]["cases"]
        }
        for condition in condition_reports
    }
    continuous = predictions["continuous"]
    restored = predictions["restored"]
    source_digests = {
        relative: sha256_file(root / relative) for relative in OBSERVATION_SOURCE_FILES
    }
    plan_payload = {
        "analysis_config_sha256": sha256_file(config_path),
        "stage_b_result_sha256": result_sha,
        "design_manifest_digest_sha256": design["design_manifest_digest_sha256"],
        "protocol_manifest_digest_sha256": protocol["manifest_digest_sha256"],
        "locked_source_digests": dict(sorted(source_digests.items())),
    }
    return {
        "observation_version": OBSERVATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "EXP-001C",
        "status": "stage_b_exploratory_observation_complete_no_confirmatory_decision",
        "valid": True,
        "development_only": True,
        "non_core": True,
        "result_observed": True,
        "model_executed_by_analysis": False,
        "stage_b_result_sha256": result_sha,
        "analysis_plan_digest_sha256": sha256_json(plan_payload),
        "analysis_plan": plan_payload,
        "condition_reports": condition_reports,
        "descriptive_contrasts": {
            "continuous_restored_prediction_agreement": (
                sum(continuous[key] == restored[key] for key in continuous)
                / len(continuous)
            ),
            "swap_joint_tracking_accuracy": {
                condition: condition_reports[condition][
                    "label_marginalized_joint_accuracy"
                ]
                for condition in ("swapped_I", "swapped_G", "swapped_both")
            },
            "diagnostic_reference_match_rate": {
                condition: condition_reports[condition][
                    "diagnostic_reference_match_rate"
                ]
                for condition in DIAGNOSTIC_CONDITIONS
            },
        },
        "contains_confirmatory_decision": False,
        "thresholds_or_go_no_go_decisions": False,
        "stage_a_rerun": False,
        "formal_test_set_accessed": False,
        "formal_run": False,
        "automatic_rerun_authorized": False,
    }
