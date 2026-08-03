from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import payload_digest, sha256_file, sha256_json
from psa.confirmatory.preflight import (
    EXPECTED_CORE_PACKAGE_DIGEST,
    EXPECTED_CORE_SET_DIGEST,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_FINAL_DIGEST,
    EXPECTED_MODEL_ID,
    verify_confirmatory_run_authorization,
)
from psa.confirmatory.runner import CONDITIONS
from psa.preregistration import verify_core_set_package


EXPECTED_GROUP_COUNT = 320
EXPECTED_TRIAL_COUNT = 5120
EXPECTED_RAW_RECORD_COUNT = EXPECTED_TRIAL_COUNT * len(CONDITIONS)
PLAN_RECORD_KEYS = (
    "trial_id",
    "condition",
    "query_target_combo",
    "state_source_combo",
    "evaluation_combo",
    "evaluation_option_code",
    "semantic_rule",
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _verify_group_payload(
    payload: Mapping[str, Any],
    *,
    expected_group: Mapping[str, Any],
) -> dict[str, bool]:
    records = payload.get("records")
    expected_trials = expected_group.get("trials")
    if not isinstance(records, list) or not isinstance(expected_trials, list):
        return {"structure_valid": False}
    trial_ids = {
        trial.get("trial_id")
        for trial in expected_trials
        if isinstance(trial, Mapping)
    }
    observed_pairs = Counter(
        (record.get("trial_id"), record.get("condition"))
        for record in records
        if isinstance(record, Mapping)
    )
    expected_pairs = {
        (trial_id, condition)
        for trial_id in trial_ids
        for condition in CONDITIONS
    }
    scores_valid = True
    for record in records:
        if not isinstance(record, Mapping):
            scores_valid = False
            break
        scores = record.get("option_scores")
        if (
            not isinstance(scores, Mapping)
            or set(scores) != set("ABCD")
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in scores.values()
            )
        ):
            scores_valid = False
            break
    planned_records = [
        {key: record.get(key) for key in PLAN_RECORD_KEYS}
        for record in records
        if isinstance(record, Mapping)
    ]
    return {
        "group_identity_valid": payload.get("factorial_group_id")
        == expected_group.get("factorial_group_id"),
        "record_count_valid": payload.get("record_count") == 128
        and len(records) == 128,
        "trial_condition_coverage_valid": (
            set(observed_pairs) == expected_pairs
            and all(count == 1 for count in observed_pairs.values())
        ),
        "option_scores_finite": scores_valid,
        "plan_digest_valid": payload.get("plan_digest_sha256")
        == sha256_json(planned_records),
        "record_digest_valid": payload.get("group_result_digest_sha256")
        == sha256_json(records),
        "no_derived_accuracy": payload.get("contains_derived_accuracy")
        is False,
        "no_interim_decision": payload.get("contains_interim_decision")
        is False,
    }


def verify_exp001_confirmatory_raw_package(
    *,
    output_dir: str | Path,
    core_set_package_dir: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Verify the complete raw package without deriving research metrics."""
    destination = Path(output_dir).resolve()
    core_root = Path(core_set_package_dir).resolve()
    manifest = _load_object(destination / "manifest.json", "run manifest")
    completion = _load_object(
        destination / "completion.json",
        "completion report",
    )
    preflight = _load_object(preflight_path, "preflight")
    authorization = _load_object(authorization_path, "authorization")
    core_set = _load_object(core_root / "core_set.json", "Core Set")
    core_report = verify_core_set_package(core_root)
    authorization_report = verify_confirmatory_run_authorization(
        authorization,
        preflight=preflight,
    )

    groups = core_set.get("groups")
    groups_by_id = {
        group.get("factorial_group_id"): group
        for group in groups
        if isinstance(group, Mapping)
        and isinstance(group.get("factorial_group_id"), str)
    } if isinstance(groups, list) else {}
    expected_group_ids = list(groups_by_id)
    expected_files = {f"{group_id}.json" for group_id in expected_group_ids}
    group_dir = destination / "groups"
    actual_files = {
        path.name for path in group_dir.glob("*.json") if path.is_file()
    } if group_dir.is_dir() else set()
    completed = manifest.get("completed_group_files")
    completed = completed if isinstance(completed, dict) else {}

    failed_group_ids: list[str] = []
    group_file_digests: dict[str, str] = {}
    verified_record_count = 0
    for group_id in expected_group_ids:
        path = group_dir / f"{group_id}.json"
        if not path.is_file():
            failed_group_ids.append(group_id)
            continue
        actual_digest = sha256_file(path)
        group_file_digests[group_id] = actual_digest
        if completed.get(group_id) != actual_digest:
            failed_group_ids.append(group_id)
            continue
        group_payload = _load_object(path, f"group {group_id}")
        checks = _verify_group_payload(
            group_payload,
            expected_group=groups_by_id[group_id],
        )
        if not checks or not all(checks.values()):
            failed_group_ids.append(group_id)
            continue
        verified_record_count += int(group_payload["record_count"])

    authorization_sha256 = sha256_file(authorization_path)
    computed_payload_digest = payload_digest(group_file_digests)
    checks = {
        "core_package_valid": core_report.get("valid") is True,
        "authorization_valid": authorization_report.get("valid") is True,
        "experiment_identity_valid": (
            manifest.get("experiment_id")
            == completion.get("experiment_id")
            == core_set.get("experiment_id")
            == EXPECTED_EXPERIMENT_ID
        ),
        "model_identity_valid": manifest.get("model_id")
        == EXPECTED_MODEL_ID,
        "final_preregistration_digest_valid": core_set.get(
            "final_preregistration_digest_sha256"
        )
        == EXPECTED_FINAL_DIGEST,
        "core_set_digest_valid": (
            manifest.get("core_set_digest_sha256")
            == completion.get("core_set_digest_sha256")
            == core_set.get("core_set_digest_sha256")
            == EXPECTED_CORE_SET_DIGEST
        ),
        "core_package_digest_valid": manifest.get(
            "core_set_package_digest_sha256"
        )
        == EXPECTED_CORE_PACKAGE_DIGEST,
        "preflight_digest_valid": (
            manifest.get("preflight_digest_sha256")
            == completion.get("preflight_digest_sha256")
            == preflight.get("preflight_digest_sha256")
        ),
        "authorization_file_digest_valid": manifest.get(
            "authorization_file_sha256"
        )
        == authorization_sha256,
        "run_status_complete": (
            manifest.get("status") == "confirmatory_raw_complete"
            and completion.get("status") == "confirmatory_raw_complete"
            and manifest.get("valid") is True
            and completion.get("valid") is True
        ),
        "design_counts_valid": (
            core_set.get("factorial_group_count") == EXPECTED_GROUP_COUNT
            and core_set.get("trial_count") == EXPECTED_TRIAL_COUNT
            and len(expected_group_ids) == EXPECTED_GROUP_COUNT
            and manifest.get("expected_group_count") == EXPECTED_GROUP_COUNT
            and manifest.get("completed_group_count") == EXPECTED_GROUP_COUNT
            and completion.get("completed_group_count")
            == EXPECTED_GROUP_COUNT
        ),
        "group_file_set_exact": actual_files == expected_files,
        "completed_group_ledger_exact": set(completed)
        == set(expected_group_ids),
        "all_group_files_valid": not failed_group_ids
        and len(group_file_digests) == EXPECTED_GROUP_COUNT,
        "raw_record_count_valid": (
            verified_record_count == EXPECTED_RAW_RECORD_COUNT
            and manifest.get("raw_record_count") == EXPECTED_RAW_RECORD_COUNT
            and completion.get("raw_record_count")
            == EXPECTED_RAW_RECORD_COUNT
        ),
        "group_payload_digest_valid": (
            computed_payload_digest
            == manifest.get("group_payload_digest_sha256")
            == completion.get("group_payload_digest_sha256")
        ),
        "no_derived_accuracy": (
            manifest.get("contains_derived_accuracy") is False
            and completion.get("contains_derived_accuracy") is False
        ),
        "no_interim_decision": (
            manifest.get("contains_interim_decision") is False
            and completion.get("contains_interim_decision") is False
        ),
        "results_still_unobserved": (
            manifest.get("confirmatory_results_observed") is False
            and completion.get("confirmatory_results_observed") is False
        ),
    }
    valid = all(checks.values())
    return {
        "verification_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": (
            "raw_package_verified_unanalyzed"
            if valid
            else "raw_package_verification_failed"
        ),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "failed_group_count": len(failed_group_ids),
        "failed_group_ids": failed_group_ids,
        "completed_group_count": manifest.get("completed_group_count"),
        "verified_record_count": verified_record_count,
        "group_payload_digest_sha256": computed_payload_digest,
        "contains_derived_accuracy": False,
        "confirmatory_experiment_run": True,
        "confirmatory_results_observed": False,
        "route_decision": (
            "begin_frozen_read_only_analysis"
            if valid
            else "hold_without_analysis_and_repair_integrity"
        ),
        "valid": valid,
    }
