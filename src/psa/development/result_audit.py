from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from psa.artifacts import canonical_json_bytes


_TASK_LEVELS = ("copy_code", "single_field", "two_field")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _output_variant(record: dict[str, Any]) -> tuple[Any, ...]:
    token_ids = record.get("generated_token_ids", [])
    if not isinstance(token_ids, list):
        token_ids = []
    return (
        str(record.get("generated_text", "")),
        tuple(token_ids),
        record.get("generated_choice"),
        bool(record.get("format_valid")),
    )


def audit_g1_capability_records(
    *,
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_trials = manifest.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("capability manifest trials must be a list")
    trials = [
        trial for trial in raw_trials if isinstance(trial, dict)
    ]
    if len(trials) != len(raw_trials):
        raise ValueError("capability manifest contains a non-object trial")

    trials_by_id: dict[str, dict[str, Any]] = {}
    for trial in trials:
        sample_id = trial.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("capability manifest trial is missing sample_id")
        if sample_id in trials_by_id:
            raise ValueError(f"duplicate manifest sample_id: {sample_id}")
        trials_by_id[sample_id] = trial

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("capability record is missing sample_id")
        if sample_id in records_by_id:
            raise ValueError(f"duplicate record sample_id: {sample_id}")
        records_by_id[sample_id] = record

    missing_record_ids = sorted(set(trials_by_id) - set(records_by_id))
    unexpected_record_ids = sorted(set(records_by_id) - set(trials_by_id))
    level_reports: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    total_format_invalid = 0
    total_failed = 0
    total_error_and_format_invalid = 0

    for task_level in _TASK_LEVELS:
        level_trials = [
            trial
            for trial in trials
            if trial.get("task_level") == task_level
        ]
        confusion: dict[str, Counter[str]] = {}
        variants: Counter[tuple[Any, ...]] = Counter()
        errors: list[dict[str, Any]] = []
        format_invalid_count = 0
        error_and_format_invalid_count = 0
        failed_count = 0
        correct_count = 0

        for trial in level_trials:
            sample_id = str(trial["sample_id"])
            record = records_by_id.get(sample_id)
            if record is None:
                continue
            target = str(trial.get("target_code"))
            predicted_raw = record.get("argmax_choice")
            predicted = (
                str(predicted_raw) if predicted_raw is not None else "<none>"
            )
            confusion.setdefault(target, Counter())[predicted] += 1
            variants[_output_variant(record)] += 1

            if record.get("status") != "success":
                failed_count += 1
            if not bool(record.get("format_valid")):
                format_invalid_count += 1
            if predicted == target:
                correct_count += 1
                continue

            scores = record.get("option_scores")
            if not isinstance(scores, dict):
                scores = {}
            target_score = scores.get(target)
            predicted_score = scores.get(predicted)
            error = {
                "sample_id": sample_id,
                "task_level": task_level,
                "target": target,
                "predicted": predicted,
                "target_fields": trial.get("target_fields"),
                "option_mapping": trial.get("option_mapping"),
                "target_score": target_score,
                "predicted_score": predicted_score,
                "target_minus_predicted": (
                    float(target_score) - float(predicted_score)
                    if isinstance(target_score, (int, float))
                    and isinstance(predicted_score, (int, float))
                    else None
                ),
                "generated_text": record.get("generated_text", ""),
                "generated_token_ids": record.get(
                    "generated_token_ids",
                    [],
                ),
                "generated_choice": record.get("generated_choice"),
                "format_valid": bool(record.get("format_valid")),
            }
            errors.append(error)
            all_errors.append(error)
            if not bool(record.get("format_valid")):
                error_and_format_invalid_count += 1

        variant_rows = [
            {
                "count": count,
                "generated_text": variant[0],
                "generated_token_ids": list(variant[1]),
                "generated_choice": variant[2],
                "format_valid": variant[3],
            }
            for variant, count in sorted(
                variants.items(),
                key=lambda item: (-item[1], repr(item[0])),
            )
        ]
        level_reports[task_level] = {
            "trial_count": len(level_trials),
            "record_count": sum(
                trial["sample_id"] in records_by_id for trial in level_trials
            ),
            "failed_trial_count": failed_count,
            "scoring_correct_count": correct_count,
            "scoring_error_count": len(errors),
            "format_invalid_count": format_invalid_count,
            "scoring_error_and_format_invalid_count": (
                error_and_format_invalid_count
            ),
            "confusion_matrix": {
                target: dict(sorted(predictions.items()))
                for target, predictions in sorted(confusion.items())
            },
            "generated_output_variants": variant_rows,
            "scoring_errors": errors,
        }
        total_format_invalid += format_invalid_count
        total_failed += failed_count
        total_error_and_format_invalid += error_and_format_invalid_count

    return {
        "audit_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "manifest_digest_sha256": manifest.get(
            "manifest_digest_sha256"
        ),
        "trial_count": len(trials),
        "record_count": len(records),
        "missing_record_ids": missing_record_ids,
        "unexpected_record_ids": unexpected_record_ids,
        "failed_trial_count": total_failed,
        "scoring_error_count": len(all_errors),
        "format_invalid_count": total_format_invalid,
        "scoring_error_and_format_invalid_count": (
            total_error_and_format_invalid
        ),
        "levels": level_reports,
        "scoring_errors": all_errors,
        "valid": bool(
            not missing_record_ids
            and not unexpected_record_ids
            and total_failed == 0
            and len(records) == len(trials)
        ),
    }


def run_g1_capability_audit(
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    manifest = _load_object(
        destination / "capability_manifest.json",
        "capability manifest",
    )
    records = []
    for line in (
        destination / "raw_capability_ladder.jsonl"
    ).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("capability JSONL record must be an object")
        records.append(record)

    report = audit_g1_capability_records(
        manifest=manifest,
        records=records,
    )
    (destination / "audit_report.json").write_bytes(
        canonical_json_bytes(report)
    )
    return report
