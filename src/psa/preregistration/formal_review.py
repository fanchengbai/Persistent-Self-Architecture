from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"{path.name}:{line_number} must contain an object"
            )
        records.append(value)
    return records


def _metric(rows: Sequence[bool]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "correct_count": sum(rows),
        "accuracy": sum(rows) / len(rows) if rows else 0.0,
    }


def _group_metrics(
    cases: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values: dict[str, list[bool]] = {}
    for case in cases:
        values.setdefault(str(case[field]), []).append(
            bool(case["correct"])
        )
    return {
        key: _metric(outcomes) for key, outcomes in sorted(values.items())
    }


def review_template_interactions(
    *,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    trial_by_case: dict[str, dict[str, Any]] = {}
    for trial in manifest["trials"]:
        trial_by_case.setdefault(trial["semantic_case_id"], trial)
    rotation = report["rotation_report"]
    cases = []
    errors = []
    for case in rotation["case_reports"]:
        trial = trial_by_case[case["semantic_case_id"]]
        target = case["target_fields"]
        prediction = case["label_marginalized_prediction"]
        filler_id = trial["filler_variant_id"]
        identity_labels = sorted(
            {item["domain"] for item in trial["option_mapping"]}
        )
        goal_labels = sorted(
            {item["operation"] for item in trial["option_mapping"]}
        )
        target_score = next(
            (
                float(item["mean_log_score"])
                for item in case["label_marginalized_scores"]
                if (
                    item["domain"] == target["domain"]
                    and item["operation"] == target["operation"]
                )
            ),
            None,
        )
        predicted_score = next(
            (
                float(item["mean_log_score"])
                for item in case["label_marginalized_scores"]
                if (
                    prediction is not None
                    and item["domain"] == prediction["domain"]
                    and item["operation"] == prediction["operation"]
                )
            ),
            None,
        )
        row = {
            "semantic_case_id": case["semantic_case_id"],
            "history_template_id": trial["history_template_id"],
            "query_template_id": trial["query_template_id"],
            "history_query_pair": (
                f"{trial['history_template_id']}|"
                f"{trial['query_template_id']}"
            ),
            "filler_variant_id": filler_id,
            "identity_label_pair": "|".join(identity_labels),
            "goal_label_pair": "|".join(goal_labels),
            "target_identity": int(target["identity"]),
            "target_goal": int(target["goal"]),
            "target_combo": (
                f"{int(target['identity'])}{int(target['goal'])}"
            ),
            "correct": bool(case["label_marginalized_correct"]),
        }
        cases.append(row)
        if not row["correct"]:
            errors.append(
                {
                    **row,
                    "target_fields": target,
                    "predicted_fields": prediction,
                    "target_mean_log_score": target_score,
                    "predicted_mean_log_score": predicted_score,
                    "target_minus_predicted": (
                        target_score - predicted_score
                        if (
                            target_score is not None
                            and predicted_score is not None
                        )
                        else None
                    ),
                    "error_target_codes": case["error_target_codes"],
                }
            )
    return {
        "semantic_case_count": len(cases),
        "label_marginalized_accuracy": _metric(
            [bool(case["correct"]) for case in cases]
        )["accuracy"],
        "history_template_metrics": _group_metrics(
            cases,
            "history_template_id",
        ),
        "query_template_metrics": _group_metrics(
            cases,
            "query_template_id",
        ),
        "history_query_pair_metrics": _group_metrics(
            cases,
            "history_query_pair",
        ),
        "filler_variant_metrics": _group_metrics(
            cases,
            "filler_variant_id",
        ),
        "identity_label_pair_metrics": _group_metrics(
            cases,
            "identity_label_pair",
        ),
        "goal_label_pair_metrics": _group_metrics(
            cases,
            "goal_label_pair",
        ),
        "target_combo_metrics": _group_metrics(cases, "target_combo"),
        "error_count": len(errors),
        "errors": errors,
        "valid": len(cases) == manifest["semantic_case_count"],
    }


def _semantic_key(
    task_type: str,
    option: Mapping[str, Any],
) -> tuple[str, ...] | None:
    if task_type == "single_field_lexical_match":
        return (str(option["symbol"]),)
    if task_type == "unrelated_two_field_symbol_match":
        return (str(option["marker"]), str(option["pattern"]))
    return None


def _target_key(
    task_type: str,
    target: Mapping[str, Any],
) -> tuple[str, ...] | None:
    if task_type == "single_field_lexical_match":
        return (str(target["symbol"]),)
    if task_type == "unrelated_two_field_symbol_match":
        return (str(target["marker"]), str(target["pattern"]))
    return None


def review_control_rotation(
    *,
    manifest: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    minimum_accuracy: float,
) -> dict[str, Any]:
    record_by_id = {record["sample_id"]: record for record in records}
    task_reports = {}
    for task_type in manifest["task_types"]:
        trials = [
            trial
            for trial in manifest["trials"]
            if trial["task_type"] == task_type
        ]
        code_outcomes = []
        per_target_code: dict[str, list[bool]] = {}
        confusion: dict[str, dict[str, int]] = {}
        cases: dict[str, list[dict[str, Any]]] = {}
        for trial in trials:
            record = record_by_id.get(trial["sample_id"])
            success = bool(record and record.get("status") == "success")
            predicted = record.get("argmax_choice") if record else None
            correct = bool(success and predicted == trial["target_code"])
            code_outcomes.append(correct)
            per_target_code.setdefault(
                trial["target_code"], []
            ).append(correct)
            confusion.setdefault(trial["target_code"], {})
            predicted_key = str(predicted)
            confusion[trial["target_code"]][predicted_key] = (
                confusion[trial["target_code"]].get(predicted_key, 0) + 1
            )
            cases.setdefault(trial["semantic_case_id"], []).append(
                {
                    "trial": trial,
                    "record": record,
                }
            )

        marginalized_cases = []
        for case_id, rotations in sorted(cases.items()):
            target = _target_key(
                task_type,
                rotations[0]["trial"]["target_fields"],
            )
            if target is None:
                continue
            semantic_scores: dict[tuple[str, ...], list[float]] = {}
            complete = len(rotations) == 4
            for item in rotations:
                trial = item["trial"]
                record = item["record"]
                if not record or record.get("status") != "success":
                    complete = False
                    continue
                raw_scores = record.get("option_scores")
                if not isinstance(raw_scores, dict):
                    complete = False
                    continue
                for option in trial["option_mapping"]:
                    semantic = _semantic_key(task_type, option)
                    score = raw_scores.get(option["code"])
                    if (
                        semantic is not None
                        and isinstance(score, (int, float))
                    ):
                        semantic_scores.setdefault(semantic, []).append(
                            float(score)
                        )
            mean_scores = {
                key: sum(values) / len(values)
                for key, values in semantic_scores.items()
                if values
            }
            prediction = (
                max(mean_scores, key=mean_scores.__getitem__)
                if mean_scores
                else None
            )
            marginalized_cases.append(
                {
                    "semantic_case_id": case_id,
                    "target": list(target),
                    "prediction": (
                        list(prediction)
                        if prediction is not None
                        else None
                    ),
                    "correct": prediction == target,
                    "rotation_count": len(rotations),
                    "complete": complete,
                    "target_mean_log_score": mean_scores.get(target),
                    "predicted_mean_log_score": mean_scores.get(prediction),
                }
            )
        marginalized_accuracy = (
            sum(case["correct"] for case in marginalized_cases)
            / len(marginalized_cases)
            if marginalized_cases
            else None
        )
        task_reports[task_type] = {
            "trial_count": len(trials),
            "code_level_accuracy": (
                sum(code_outcomes) / len(code_outcomes)
                if code_outcomes
                else 0.0
            ),
            "per_target_code": {
                code: _metric(outcomes)
                for code, outcomes in sorted(per_target_code.items())
            },
            "confusion_matrix": confusion,
            "semantic_case_count": len(marginalized_cases),
            "label_marginalized_accuracy": marginalized_accuracy,
            "label_marginalized_pass_threshold": (
                marginalized_accuracy >= minimum_accuracy
                if marginalized_accuracy is not None
                else None
            ),
            "label_marginalized_errors": [
                case
                for case in marginalized_cases
                if not case["correct"]
            ],
            "diagnostic_complete": bool(
                len(trials) == 32
                and len(
                    [
                        trial
                        for trial in trials
                        if trial["sample_id"] in record_by_id
                    ]
                )
                == len(trials)
                and all(
                    case["complete"] for case in marginalized_cases
                )
            ),
        }
    two_field = task_reports["unrelated_two_field_symbol_match"]
    if two_field["label_marginalized_pass_threshold"] is True:
        route = "control_code_bias_controlled_by_rotation"
    else:
        route = "control_two_field_semantic_failure_after_rotation"
    return {
        "report_version": "1.0",
        "development_only": True,
        "confirmatory_results_observed": False,
        "minimum_accuracy": minimum_accuracy,
        "task_reports": task_reports,
        "route_decision": route,
        "valid": all(
            report["diagnostic_complete"]
            for report in task_reports.values()
        ),
    }


def run_formal_freeze_review(
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    template_manifest_path = (
        destination / "template_qualification_manifest.json"
    )
    template_report_path = (
        destination / "template_qualification_report.json"
    )
    control_manifest_path = destination / "control_manifest.json"
    control_report_path = destination / "control_baseline_report.json"
    control_raw_path = destination / "raw_control_baseline.jsonl"
    source_summary_path = destination / "summary.json"
    template_manifest = _load_object(
        template_manifest_path,
        "template qualification manifest",
    )
    template_report = _load_object(
        template_report_path,
        "template qualification report",
    )
    control_manifest = _load_object(
        control_manifest_path,
        "control manifest",
    )
    control_report = _load_object(
        control_report_path,
        "control baseline report",
    )
    source_summary = _load_object(source_summary_path, "Impl-3q summary")
    control_records = _load_jsonl(control_raw_path)
    template_review = review_template_interactions(
        manifest=template_manifest,
        report=template_report,
    )
    control_review = review_control_rotation(
        manifest=control_manifest,
        records=control_records,
        minimum_accuracy=float(
            control_report["minimum_accuracy_per_task"]
        ),
    )
    if control_review["route_decision"] == (
        "control_code_bias_controlled_by_rotation"
    ):
        route = (
            "revise_formal_template_family_and_use_"
            "rotation_marginalized_control_readout"
        )
    else:
        route = "revise_formal_and_control_two_field_prompt_families"
    review = {
        "review_version": "1.0",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "confirmatory_results_observed": False,
        "core_set_generated": False,
        "source_gate": source_summary["gate"],
        "source_candidate_digest_sha256": source_summary[
            "candidate_digest_sha256"
        ],
        "source_files": {
            path.name: sha256_file(path)
            for path in (
                template_manifest_path,
                template_report_path,
                control_manifest_path,
                control_report_path,
                control_raw_path,
                source_summary_path,
            )
        },
        "template_review": template_review,
        "control_review": control_review,
        "route_decision": route,
        "valid": bool(
            source_summary["valid"]
            and not source_summary["freeze_candidate_ready"]
            and template_review["valid"]
            and control_review["valid"]
        ),
    }
    output_path = destination / "formal_freeze_review.json"
    output_path.write_bytes(canonical_json_bytes(review))
    return review
