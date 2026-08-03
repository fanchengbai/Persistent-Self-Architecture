from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)
from psa.confirmatory.preflight import (
    EXPECTED_CORE_PACKAGE_DIGEST,
    EXPECTED_CORE_SET_DIGEST,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_FINAL_DIGEST,
    EXPECTED_MODEL_ID,
    build_confirmatory_preflight,
    verify_confirmatory_run_authorization,
)
from psa.confirmatory.runner import CONDITIONS, execute_group
from psa.preregistration import verify_core_set_package


EXPECTED_FACTORIAL_GROUP_COUNT = 320
EXPECTED_ROTATION_TRIAL_COUNT = 5120
EXPECTED_RAW_RECORD_COUNT = EXPECTED_ROTATION_TRIAL_COUNT * len(CONDITIONS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        data = canonical_json_bytes(dict(payload))
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_core_dataset(dataset: Mapping[str, Any]) -> None:
    groups = dataset.get("groups")
    if (
        dataset.get("experiment_id") != EXPECTED_EXPERIMENT_ID
        or dataset.get("status") != "core_set_frozen_unrun"
        or dataset.get("core_set_digest_sha256") != EXPECTED_CORE_SET_DIGEST
        or dataset.get("final_preregistration_digest_sha256")
        != EXPECTED_FINAL_DIGEST
        or dataset.get("factorial_group_count")
        != EXPECTED_FACTORIAL_GROUP_COUNT
        or dataset.get("trial_count") != EXPECTED_ROTATION_TRIAL_COUNT
        or dataset.get("confirmatory_experiment_run") is not False
        or dataset.get("confirmatory_results_observed") is not False
        or tuple(dataset.get("conditions", ())) != CONDITIONS
        or not isinstance(groups, list)
        or len(groups) != EXPECTED_FACTORIAL_GROUP_COUNT
    ):
        raise ValueError("frozen EXP-001 Core Set identity or design is invalid")
    group_ids = [
        group.get("factorial_group_id")
        if isinstance(group, Mapping)
        else None
        for group in groups
    ]
    if any(
        not isinstance(group_id, str) or not group_id.startswith("coregrp-")
        for group_id in group_ids
    ):
        raise ValueError("formal groups must use frozen coregrp identifiers")
    if len(set(group_ids)) != EXPECTED_FACTORIAL_GROUP_COUNT:
        raise ValueError("formal factorial_group_id values must be unique")
    if any(
        not isinstance(group, Mapping)
        or group.get("trial_count") != 16
        or not isinstance(group.get("trials"), list)
        or len(group["trials"]) != 16
        for group in groups
    ):
        raise ValueError("every formal group must contain 16 frozen trials")


def prepare_exp001_confirmatory_launch(
    *,
    project_root: str | Path,
    final_package_dir: str | Path,
    core_set_package_dir: str | Path,
    model_config_path: str | Path,
    asset_manifest_path: str | Path,
    asset_root: str | Path,
    runner_evidence_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Rebuild every non-inference lock before any model can be loaded."""
    root = Path(project_root).resolve()
    persisted_preflight = _load_object(preflight_path, "persisted preflight")
    authorization = _load_object(authorization_path, "run authorization")
    live_preflight = build_confirmatory_preflight(
        project_root=root,
        final_package_dir=final_package_dir,
        core_set_package_dir=core_set_package_dir,
        model_config_path=model_config_path,
        asset_manifest_path=asset_manifest_path,
        asset_root=asset_root,
        runner_evidence_path=runner_evidence_path,
    )
    preflight_matches = bool(
        persisted_preflight.get("preflight_digest_sha256")
        == live_preflight.get("preflight_digest_sha256")
        and persisted_preflight.get("run_plan_candidate")
        == live_preflight.get("run_plan_candidate")
        and persisted_preflight.get("status")
        == "preflight_valid_authorization_still_required"
    )
    if not preflight_matches:
        raise ValueError("persisted preflight does not match the live host and sources")
    authorization_report = verify_confirmatory_run_authorization(
        authorization,
        preflight=live_preflight,
    )
    if not authorization_report["valid"]:
        failed = [
            key
            for key, value in authorization_report["checks"].items()
            if not value
        ]
        raise ValueError(
            "confirmatory authorization is invalid: " + ", ".join(failed)
        )
    core_report = verify_core_set_package(core_set_package_dir)
    if not core_report["valid"]:
        raise ValueError("frozen Core Set package is invalid")
    dataset = _load_object(
        Path(core_set_package_dir) / "core_set.json",
        "Core Set",
    )
    _validate_core_dataset(dataset)
    return {
        "launch_lock_version": "0.1",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_id": EXPECTED_MODEL_ID,
        "preflight_digest_sha256": live_preflight[
            "preflight_digest_sha256"
        ],
        "authorization_file_sha256": sha256_file(authorization_path),
        "authorization_scope_sha256": sha256_json(
            authorization["authorization"]
        ),
        "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
        "core_set_package_digest_sha256": EXPECTED_CORE_PACKAGE_DIGEST,
        "expected_group_ids": [
            group["factorial_group_id"] for group in dataset["groups"]
        ],
        "expected_group_count": EXPECTED_FACTORIAL_GROUP_COUNT,
        "expected_rotation_trial_count": EXPECTED_ROTATION_TRIAL_COUNT,
        "expected_condition_count": len(CONDITIONS),
        "expected_raw_record_count": EXPECTED_RAW_RECORD_COUNT,
        "conditions": list(CONDITIONS),
        "safety_rules": {
            "partial_core_runs_forbidden": True,
            "intermediate_accuracy_reporting_forbidden": True,
            "manual_resume_required_after_interruption": True,
            "rerun_after_completion_forbidden": True,
            "derived_metrics_emitted_by_runner": False,
        },
        "valid": True,
    }


def _new_manifest(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_manifest_version": "0.1",
        "run_kind": "exp001_confirmatory",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_id": EXPECTED_MODEL_ID,
        "preflight_digest_sha256": lock["preflight_digest_sha256"],
        "authorization_file_sha256": lock["authorization_file_sha256"],
        "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
        "core_set_package_digest_sha256": EXPECTED_CORE_PACKAGE_DIGEST,
        "expected_group_ids": list(lock["expected_group_ids"]),
        "expected_group_count": lock["expected_group_count"],
        "expected_records_per_group": 16 * len(CONDITIONS),
        "expected_raw_record_count": lock["expected_raw_record_count"],
        "status": "initialized",
        "completed_group_files": {},
        "completed_group_count": 0,
        "contains_derived_accuracy": False,
        "contains_interim_decision": False,
        "confirmatory_experiment_authorized": True,
        "confirmatory_experiment_run": True,
        "confirmatory_results_observed": False,
    }


def _validate_existing_manifest(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    expected = _new_manifest(lock)
    immutable_keys = (
        "run_kind",
        "experiment_id",
        "model_id",
        "preflight_digest_sha256",
        "authorization_file_sha256",
        "core_set_digest_sha256",
        "core_set_package_digest_sha256",
        "expected_group_ids",
        "expected_group_count",
        "expected_records_per_group",
        "expected_raw_record_count",
        "contains_derived_accuracy",
        "contains_interim_decision",
        "confirmatory_experiment_authorized",
        "confirmatory_results_observed",
    )
    if any(manifest.get(key) != expected.get(key) for key in immutable_keys):
        raise ValueError("existing formal manifest does not match launch lock")
    if manifest.get("confirmatory_experiment_run") is not True:
        raise ValueError("existing formal manifest has an invalid run marker")


def _validate_completed_files(
    *,
    destination: Path,
    completed: Mapping[str, Any],
    groups_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    for group_id, expected_digest in completed.items():
        group_path = destination / "groups" / f"{group_id}.json"
        if (
            group_id not in groups_by_id
            or not isinstance(expected_digest, str)
            or not group_path.is_file()
            or sha256_file(group_path) != expected_digest
        ):
            raise ValueError(f"completed formal group integrity failed: {group_id}")


def _check_start_boundary(
    *,
    destination: Path,
    launch_lock: Mapping[str, Any],
    resume: bool,
) -> None:
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = _load_object(manifest_path, "formal run manifest")
        _validate_existing_manifest(manifest, launch_lock)
        if manifest.get("status") == "confirmatory_raw_complete":
            raise ValueError("completed confirmatory run cannot be rerun")
        if not resume:
            raise ValueError("interrupted formal run requires explicit resume")
    else:
        if resume:
            raise ValueError("cannot resume before a formal manifest exists")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("new formal output directory must be empty")


def run_locked_confirmatory_groups(
    *,
    dataset: Mapping[str, Any],
    backend: Any,
    output_dir: str | Path,
    launch_lock: Mapping[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Run all locked groups, emitting no accuracy or interim decision."""
    _validate_core_dataset(dataset)
    if launch_lock.get("valid") is not True:
        raise ValueError("a valid launch lock is required")
    destination = Path(output_dir).resolve()
    manifest_path = destination / "manifest.json"
    _check_start_boundary(
        destination=destination,
        launch_lock=launch_lock,
        resume=resume,
    )
    if manifest_path.exists():
        manifest = _load_object(manifest_path, "formal run manifest")
    else:
        manifest = _new_manifest(launch_lock)
        _atomic_write(manifest_path, manifest)

    completed = manifest.get("completed_group_files")
    if not isinstance(completed, dict):
        raise ValueError("completed_group_files must be an object")
    groups_by_id = {
        group["factorial_group_id"]: group for group in dataset["groups"]
    }
    _validate_completed_files(
        destination=destination,
        completed=completed,
        groups_by_id=groups_by_id,
    )
    manifest["status"] = "running"
    manifest["confirmatory_experiment_run"] = True
    manifest.setdefault("started_at_utc", _utc_now())
    _atomic_write(manifest_path, manifest)
    try:
        for group_id in launch_lock["expected_group_ids"]:
            if group_id in completed:
                continue
            result = execute_group(groups_by_id[group_id], backend)
            group_path = destination / "groups" / f"{group_id}.json"
            _atomic_write(group_path, result)
            completed[group_id] = sha256_file(group_path)
            manifest["completed_group_files"] = dict(completed)
            manifest["completed_group_count"] = len(completed)
            _atomic_write(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "failed_at_utc": _utc_now(),
        }
        _atomic_write(manifest_path, manifest)
        raise

    if len(completed) != launch_lock["expected_group_count"]:
        raise RuntimeError("formal run ended without every locked group")
    manifest.pop("failure", None)
    manifest["status"] = "confirmatory_raw_complete"
    manifest["completed_at_utc"] = _utc_now()
    manifest["raw_record_count"] = sum(
        int(groups_by_id[group_id]["trial_count"]) * len(CONDITIONS)
        for group_id in completed
    )
    manifest["group_payload_digest_sha256"] = payload_digest(completed)
    manifest["valid"] = (
        manifest["raw_record_count"] == EXPECTED_RAW_RECORD_COUNT
    )
    completion = {
        "completion_version": "0.1",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": manifest["status"],
        "preflight_digest_sha256": manifest["preflight_digest_sha256"],
        "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
        "completed_group_count": manifest["completed_group_count"],
        "raw_record_count": manifest["raw_record_count"],
        "group_payload_digest_sha256": manifest[
            "group_payload_digest_sha256"
        ],
        "contains_derived_accuracy": False,
        "contains_interim_decision": False,
        "confirmatory_experiment_run": True,
        "confirmatory_results_observed": False,
        "route_decision": "verify_complete_raw_package_before_analysis",
        "valid": manifest["valid"],
    }
    _atomic_write(destination / "completion.json", completion)
    _atomic_write(manifest_path, manifest)
    return completion


def run_exp001_confirmatory(
    *,
    project_root: str | Path,
    final_package_dir: str | Path,
    core_set_package_dir: str | Path,
    model_config_path: str | Path,
    asset_manifest_path: str | Path,
    asset_root: str | Path,
    runner_evidence_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Authorized full EXP-001 execution; never called by development gates."""
    launch_lock = prepare_exp001_confirmatory_launch(
        project_root=project_root,
        final_package_dir=final_package_dir,
        core_set_package_dir=core_set_package_dir,
        model_config_path=model_config_path,
        asset_manifest_path=asset_manifest_path,
        asset_root=asset_root,
        runner_evidence_path=runner_evidence_path,
        preflight_path=preflight_path,
        authorization_path=authorization_path,
    )
    dataset = _load_object(
        Path(core_set_package_dir) / "core_set.json",
        "Core Set",
    )
    _check_start_boundary(
        destination=Path(output_dir).resolve(),
        launch_lock=launch_lock,
        resume=resume,
    )

    # Keep model imports and allocation strictly after every authorization lock.
    from psa.confirmatory.rwkv_backend import RWKVConfirmatoryBackend
    from psa.model import RWKV7Adapter, load_model_config

    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    config = load_model_config(model_config_path, project_root, verify_files=True)
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    backend = RWKVConfirmatoryBackend(adapter=adapter)
    completion = run_locked_confirmatory_groups(
        dataset=dataset,
        backend=backend,
        output_dir=output_dir,
        launch_lock=launch_lock,
        resume=resume,
    )
    completion["model_load_seconds"] = load_seconds
    completion["cuda_peak_memory_bytes"] = int(
        torch.cuda.max_memory_allocated()
    )
    _atomic_write(Path(output_dir).resolve() / "completion.json", completion)
    return completion
