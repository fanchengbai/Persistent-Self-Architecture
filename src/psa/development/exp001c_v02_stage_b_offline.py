from __future__ import annotations

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
    verify_exp001c_v02_stage_b_design_manifest,
)


OFFLINE_RESULT_VERSION = "0.1-offline-contract"
OFFLINE_TEST_LOCK = "EXP001C_V02_STAGE_B_OFFLINE_FAKE_ADAPTER_ONLY"


class Exp001CV02StageBOfflineAdapter(Protocol):
    offline_fake_adapter: bool
    model_loaded: bool

    def score_route(self, record: Mapping[str, Any]) -> Mapping[str, float]: ...


class Exp001CV02StageBOfflineBackend(Protocol):
    offline_contract_backend: bool

    def run_offline_contract(
        self,
        manifest: Mapping[str, Any],
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


def _validated_scores(value: Mapping[str, Any]) -> dict[str, float]:
    if set(value) != set("ABCD"):
        raise ValueError("offline Stage B scores must contain exactly A-D")
    scores = {code: float(value[code]) for code in "ABCD"}
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("offline Stage B scores must be finite")
    return scores


def _answer_boundary(
    scores: Mapping[str, float],
    target_code: str | None,
) -> dict[str, Any] | None:
    if target_code is None:
        return None
    incorrect = [code for code in "ABCD" if code != target_code]
    best_incorrect = max(incorrect, key=lambda code: scores[code])
    return {
        "target_code": target_code,
        "target_answer_log_probability": float(scores[target_code]),
        "best_incorrect_code": best_incorrect,
        "best_incorrect_answer_log_probability": float(
            scores[best_incorrect]
        ),
        "target_margin_over_best_incorrect": float(
            scores[target_code] - scores[best_incorrect]
        ),
    }


class OfflineFakeStageBContractBackend:
    """Exercise Stage B routing with a fake adapter and no model capability."""

    offline_contract_backend = True

    def __init__(self, *, adapter: Exp001CV02StageBOfflineAdapter) -> None:
        if (
            getattr(adapter, "offline_fake_adapter", None) is not True
            or getattr(adapter, "model_loaded", None) is not False
        ):
            raise PermissionError(
                "Stage B offline backend accepts only an unloaded fake adapter"
            )
        self.adapter = adapter

    def run_offline_contract(
        self,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        records = manifest.get("records")
        if (
            manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("status")
            != "offline_stage_b_design_verified_execution_unapproved"
            or manifest.get("model_executed") is not False
            or manifest.get("execution_authorized") is not False
            or manifest.get("formal_test_set_accessed") is not False
            or manifest.get("stage_a_rerun_included") is not False
            or manifest.get("conditions") != list(STAGE_B_CONDITIONS)
            or not isinstance(records, list)
            or len(records) != 224
        ):
            raise ValueError("Stage B offline backend received an invalid design")

        output_records = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("Stage B offline route must be an object")
            scores = _validated_scores(self.adapter.score_route(record))
            target_code = record.get("expected_state_semantic_target_code")
            if target_code is not None and target_code not in set("ABCD"):
                raise ValueError("Stage B offline target code is invalid")
            output_records.append(
                {
                    "record_id": record["record_id"],
                    "condition": record["condition"],
                    "condition_role": record["condition_role"],
                    "query_sample_id": record["query_sample_id"],
                    "semantic_case_id": record["semantic_case_id"],
                    "rotation_index": record["rotation_index"],
                    "state_source_sample_id": record[
                        "state_source_sample_id"
                    ],
                    "state_source_fields": record["state_source_fields"],
                    "reference_stage_a_target_code": record[
                        "reference_stage_a_target_code"
                    ],
                    "expected_state_semantic_target_code": target_code,
                    "semantic_endpoint_role": record[
                        "semantic_endpoint_role"
                    ],
                    "option_log_probabilities": scores,
                    "predicted_code": max(scores, key=scores.__getitem__),
                    "answer_boundary_evidence": _answer_boundary(
                        scores,
                        target_code,
                    ),
                    "synthetic_output": True,
                }
            )
        return {
            "result_version": OFFLINE_RESULT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "stage_b_offline_fake_contract_complete",
            "development_only": True,
            "non_core": True,
            "offline_fake_adapter_only": True,
            "synthetic_output_not_research_evidence": True,
            "model_loaded": False,
            "model_executed": False,
            "stage_a_rerun": False,
            "formal_test_set_accessed": False,
            "formal_run": False,
            "contains_confirmatory_decision": False,
            "automatic_rerun_authorized": False,
            "design_manifest_digest_sha256": manifest.get(
                "design_manifest_digest_sha256"
            ),
            "condition_count": 7,
            "record_count": len(output_records),
            "records": output_records,
        }


def _validate_offline_result(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    records = result.get("records")
    expected_records = manifest.get("records")
    expected_ids = (
        [record.get("record_id") for record in expected_records]
        if isinstance(expected_records, list)
        and all(isinstance(record, Mapping) for record in expected_records)
        else []
    )
    actual_ids = (
        [record.get("record_id") for record in records]
        if isinstance(records, list)
        and all(isinstance(record, Mapping) for record in records)
        else []
    )
    if (
        result.get("result_version") != OFFLINE_RESULT_VERSION
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("status") != "stage_b_offline_fake_contract_complete"
        or result.get("development_only") is not True
        or result.get("non_core") is not True
        or result.get("offline_fake_adapter_only") is not True
        or result.get("synthetic_output_not_research_evidence") is not True
        or result.get("model_loaded") is not False
        or result.get("model_executed") is not False
        or result.get("stage_a_rerun") is not False
        or result.get("formal_test_set_accessed") is not False
        or result.get("formal_run") is not False
        or result.get("contains_confirmatory_decision") is not False
        or result.get("automatic_rerun_authorized") is not False
        or result.get("design_manifest_digest_sha256")
        != manifest.get("design_manifest_digest_sha256")
        or result.get("condition_count") != 7
        or result.get("record_count") != 224
        or actual_ids != expected_ids
        or len(set(actual_ids)) != 224
    ):
        raise ValueError("Stage B offline result violates the safety contract")


def run_exp001c_v02_stage_b_offline_contract(
    *,
    design_manifest_path: str | Path,
    output_dir: str | Path,
    backend_factory: Callable[[], Exp001CV02StageBOfflineBackend],
    offline_test_lock: str,
    project_root: str | Path,
) -> dict[str, Any]:
    if offline_test_lock != OFFLINE_TEST_LOCK:
        raise PermissionError("Stage B offline fake-adapter lock is absent")
    root = Path(project_root).resolve()
    verification = verify_exp001c_v02_stage_b_design_manifest(
        design_manifest_path,
        project_root=root,
    )
    if verification.get("valid") is not True:
        raise ValueError("Stage B design manifest verification failed")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Stage B offline output directory must be empty")
    manifest = _load_object(
        design_manifest_path,
        "EXP-001C v02 Stage B design manifest",
        root=root,
    )
    backend = backend_factory()
    if getattr(backend, "offline_contract_backend", None) is not True:
        raise PermissionError("Stage B backend is not an offline contract backend")
    result = backend.run_offline_contract(manifest)
    if not isinstance(result, Mapping):
        raise ValueError("Stage B offline backend result must be an object")
    _validate_offline_result(result, manifest)

    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "stage_b_offline_contract_result.json"
    _atomic_write(result_path, result)
    summary = {
        "summary_version": OFFLINE_RESULT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "stage_b_offline_fake_contract_complete",
        "valid": True,
        "development_only": True,
        "offline_fake_adapter_only": True,
        "synthetic_output_not_research_evidence": True,
        "model_loaded": False,
        "model_executed": False,
        "stage_a_rerun": False,
        "formal_test_set_accessed": False,
        "formal_run": False,
        "contains_confirmatory_decision": False,
        "automatic_rerun_authorized": False,
        "design_manifest_digest_sha256": manifest[
            "design_manifest_digest_sha256"
        ],
        "offline_result_sha256": sha256_file(result_path),
        "condition_count": 7,
        "record_count": 224,
    }
    _atomic_write(destination / "summary.json", summary)
    return summary


def run_exp001c_v02_stage_b_model(*args: Any, **kwargs: Any) -> None:
    """Fail closed until a live preflight and exact owner authority exist."""
    del args, kwargs
    raise PermissionError(
        "Stage B model execution is unavailable until live preflight and "
        "separate owner authorization are implemented"
    )
