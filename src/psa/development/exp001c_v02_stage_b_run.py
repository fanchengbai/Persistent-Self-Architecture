from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from psa.artifacts import canonical_json_bytes, sha256_file
from psa.development.exp001c_v02_stage_b_design import (
    EXPERIMENT_ID,
    STAGE_B_CONDITIONS,
    STATE_SEMANTIC_CONDITIONS,
    verify_exp001c_v02_stage_b_design_manifest,
)
from psa.development.prefix_instrumentation import answer_boundary_evidence


STAGE_B_EXECUTION_LOCK = "AUTHORIZED_EXP001C_V02_STAGE_B_NONCORE_ONCE"
STAGE_B_RUN_VERSION = "0.1-development"


class Exp001CV02StageBBackend(Protocol):
    def run_stage_b(
        self,
        design_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _load_object(
    path: str | Path,
    label: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    candidate = Path(path)
    if root is not None and not candidate.is_absolute():
        candidate = root / candidate
    value = json.loads(candidate.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_claim(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("Stage B single-use execution claim already exists") from exc


def validate_exp001c_v02_stage_b_authority(
    *,
    design_manifest_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    from psa.development.exp001c_v02_stage_b_preflight import (
        validate_exp001c_v02_stage_b_machine_authority,
    )

    return validate_exp001c_v02_stage_b_machine_authority(
        design_manifest_path=design_manifest_path,
        preflight_path=preflight_path,
        authorization_path=authorization_path,
        model_config_path=model_config_path,
        project_root=project_root,
    )


def _authority_is_valid(
    authority: Mapping[str, Any],
    design: Mapping[str, Any],
) -> bool:
    return bool(
        authority.get("valid") is True
        and authority.get("experiment_id") == EXPERIMENT_ID
        and authority.get("scope")
        == "v02_stage_b_recurrent_state_noncore_pilot_once"
        and authority.get("design_manifest_digest_sha256")
        == design.get("design_manifest_digest_sha256")
        and isinstance(authority.get("preflight_digest_sha256"), str)
        and len(str(authority.get("preflight_digest_sha256"))) == 64
        and isinstance(authority.get("stage_a_result_sha256"), str)
        and len(str(authority.get("stage_a_result_sha256"))) == 64
        and authority.get("model_execution_authorized") is True
        and isinstance(
            authority.get("stage_b_result_observation_authorized"),
            bool,
        )
        and authority.get("stage_a_rerun_authorized") is False
        and authority.get("formal_test_set_access_authorized") is False
        and authority.get("formal_run_authorized") is False
        and authority.get("confirmatory_decision_authorized") is False
        and authority.get("automatic_rerun_authorized") is False
    )


def _scores(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or set(value) != set("ABCD"):
        return None
    scores = {code: float(value[code]) for code in "ABCD"}
    return scores if all(math.isfinite(score) for score in scores.values()) else None


def _prefix_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("instrumentation_version") == "0.1-development"
        and value.get("development_only") is True
        and value.get("text") == ">\n"
        and isinstance(value.get("token_ids"), list)
        and len(value["token_ids"]) == 2
        and isinstance(value.get("greedy_token_ids"), list)
        and len(value["greedy_token_ids"]) == 2
        and isinstance(value.get("greedy_exact"), bool)
        and isinstance(value.get("roundtrip_exact"), bool)
        and value.get("top_k") == 10
        and isinstance(value.get("positions"), list)
        and len(value["positions"]) == 2
    )


def _validate_result_payload(
    result: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    result_records = result.get("records")
    design_records = design.get("records")
    top_level_valid = bool(
        result.get("result_version") == "0.2-stage-b-development"
        and result.get("experiment_id") == EXPERIMENT_ID
        and result.get("status") == "v02_stage_b_recurrent_state_complete"
        and result.get("development_only") is True
        and result.get("non_core") is True
        and result.get("model_executed") is True
        and result.get("recurrent_state_accessed") is True
        and result.get("source_states_cloned_per_route") is True
        and result.get("stage_a_rerun") is False
        and result.get("formal_test_set_accessed") is False
        and result.get("formal_run") is False
        and result.get("contains_confirmatory_decision") is False
        and result.get("automatic_rerun_authorized") is False
        and result.get("design_manifest_digest_sha256")
        == design.get("design_manifest_digest_sha256")
        and result.get("protocol_manifest_digest_sha256")
        == design.get("protocol_manifest_digest_sha256")
        and result.get("condition_count") == 7
        and result.get("record_count") == 224
        and isinstance(result.get("warmup_token_lengths"), list)
        and bool(result["warmup_token_lengths"])
        and isinstance(result.get("snapshot_roundtrip_reports"), Mapping)
        and set(result["snapshot_roundtrip_reports"])
        == {"block-000", "block-001"}
        and isinstance(result_records, list)
        and len(result_records) == 224
        and isinstance(design_records, list)
        and len(design_records) == 224
    )
    failed_record_ids = []
    condition_counts: Counter[str] = Counter()
    if top_level_valid:
        route_fields = (
            "record_id",
            "condition",
            "condition_role",
            "query_sample_id",
            "semantic_case_id",
            "block_id",
            "rotation_index",
            "query_history_key",
            "state_source_sample_id",
            "state_source_history_key",
            "state_source_fields",
            "reference_stage_a_target_code",
            "expected_state_semantic_target_code",
            "semantic_endpoint_role",
        )
        for route, record in zip(design_records, result_records):
            valid = isinstance(route, Mapping) and isinstance(record, Mapping)
            if valid:
                valid = all(record.get(field) == route.get(field) for field in route_fields)
            scores = _scores(record.get("option_log_probabilities")) if valid else None
            if valid and scores is not None:
                predicted = max("ABCD", key=lambda code: scores[code])
                valid = record.get("predicted_code") == predicted
            else:
                valid = False
            if valid:
                valid = (
                    isinstance(record.get("query_token_count"), int)
                    and record["query_token_count"] > 0
                    and _prefix_valid(record.get("prefix_evidence"))
                )
            target = record.get("expected_state_semantic_target_code") if valid else None
            condition = str(record.get("condition", "")) if valid else ""
            if valid and condition in STATE_SEMANTIC_CONDITIONS:
                expected_boundary = answer_boundary_evidence(
                    scores,
                    target_code=str(target),
                )
                valid = record.get("answer_boundary_evidence") == expected_boundary
            elif valid and condition in {"reset", "random_matched"}:
                valid = (
                    target is None
                    and record.get("answer_boundary_evidence") is None
                )
            else:
                valid = False
            if valid:
                condition_counts[condition] += 1
            else:
                failed_record_ids.append(str(record.get("record_id", "<missing>")))
    record_inventory_valid = bool(
        top_level_valid
        and not failed_record_ids
        and condition_counts
        == Counter({condition: 32 for condition in STAGE_B_CONDITIONS})
        and len({record["record_id"] for record in result_records}) == 224
    )
    return {
        "valid": bool(top_level_valid and record_inventory_valid),
        "top_level_valid": top_level_valid,
        "record_inventory_valid": record_inventory_valid,
        "failed_record_ids": failed_record_ids,
        "condition_counts": dict(condition_counts),
        "contains_derived_accuracy": False,
        "contains_research_decision": False,
    }


def verify_exp001c_v02_stage_b_result(
    *,
    result_path: str | Path,
    design_manifest_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    design_verification = verify_exp001c_v02_stage_b_design_manifest(
        design_manifest_path,
        project_root=root,
    )
    if design_verification.get("valid") is not True:
        raise ValueError("Stage B design manifest verification failed")
    design = _load_object(
        design_manifest_path,
        "EXP-001C v02 Stage B design manifest",
        root=root,
    )
    result_file = Path(result_path).resolve()
    result = _load_object(result_file, "EXP-001C v02 Stage B result")
    payload_verification = _validate_result_payload(result, design)
    return {
        "verification_version": STAGE_B_RUN_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "stage_b_raw_result_verified_unobserved"
            if payload_verification["valid"]
            else "invalid"
        ),
        **payload_verification,
        "design_manifest_digest_sha256": design[
            "design_manifest_digest_sha256"
        ],
        "stage_b_result_sha256": sha256_file(result_file),
        "model_executed_by_verifier": False,
        "formal_test_set_accessed": False,
    }


def run_exp001c_v02_stage_b(
    *,
    design_manifest_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    backend_factory: Callable[[bool], Exp001CV02StageBBackend],
    execution_lock: str,
    project_root: str | Path,
) -> dict[str, Any]:
    if execution_lock != STAGE_B_EXECUTION_LOCK:
        raise PermissionError("Stage B single-use execution lock is absent")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Stage B output directory must be empty")
    root = Path(project_root).resolve()
    design_verification = verify_exp001c_v02_stage_b_design_manifest(
        design_manifest_path,
        project_root=root,
    )
    if design_verification.get("valid") is not True:
        raise ValueError("Stage B design manifest verification failed")
    design = _load_object(
        design_manifest_path,
        "EXP-001C v02 Stage B design manifest",
        root=root,
    )
    authority = validate_exp001c_v02_stage_b_authority(
        design_manifest_path=design_manifest_path,
        preflight_path=preflight_path,
        authorization_path=authorization_path,
        model_config_path=model_config_path,
        project_root=root,
    )
    if not isinstance(authority, Mapping) or not _authority_is_valid(authority, design):
        raise PermissionError("Stage B machine authorization is invalid")

    claim = {
        "claim_version": STAGE_B_RUN_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "stage_b_single_use_execution_claim_consumed",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_use": True,
        "design_manifest_digest_sha256": design[
            "design_manifest_digest_sha256"
        ],
        "preflight_digest_sha256": authority["preflight_digest_sha256"],
        "stage_a_result_sha256": authority["stage_a_result_sha256"],
        "model_execution_authorized": True,
        "stage_b_result_observation_authorized": authority[
            "stage_b_result_observation_authorized"
        ],
        "stage_a_rerun_authorized": False,
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
    }
    claim_path = destination / "execution_claim.json"
    _exclusive_claim(claim_path, claim)

    backend = backend_factory(True)
    result = backend.run_stage_b(design)
    if not isinstance(result, Mapping):
        raise ValueError("Stage B backend result must be an object")
    payload_verification = _validate_result_payload(result, design)
    if payload_verification.get("valid") is not True:
        raise ValueError("Stage B backend result violates the locked contract")

    result_path = destination / "stage_b_result.json"
    _atomic_write(result_path, result)
    verification = verify_exp001c_v02_stage_b_result(
        result_path=result_path,
        design_manifest_path=design_manifest_path,
        project_root=root,
    )
    if verification.get("valid") is not True:
        raise ValueError("Stage B persisted result verification failed")
    _atomic_write(destination / "result_verification.json", verification)
    summary = {
        "summary_version": STAGE_B_RUN_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "stage_b_raw_result_complete_verified_unobserved",
        "valid": True,
        "development_only": True,
        "non_core": True,
        "model_executed": True,
        "recurrent_state_accessed": True,
        "single_use_execution_claim_consumed": True,
        "stage_b_result_observation_authorized": authority[
            "stage_b_result_observation_authorized"
        ],
        "design_manifest_digest_sha256": design[
            "design_manifest_digest_sha256"
        ],
        "preflight_digest_sha256": authority["preflight_digest_sha256"],
        "stage_a_result_sha256": authority["stage_a_result_sha256"],
        "stage_b_result_sha256": verification["stage_b_result_sha256"],
        "record_count": 224,
        "condition_count": 7,
        "contains_derived_accuracy": False,
        "contains_research_decision": False,
        "stage_a_rerun": False,
        "formal_test_set_accessed": False,
        "formal_run": False,
        "automatic_rerun_authorized": False,
    }
    _atomic_write(destination / "summary.json", summary)
    return summary
