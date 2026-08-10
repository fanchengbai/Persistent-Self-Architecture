from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json


EXPERIMENT_ID = "EXP-001C"
PROBE_MANIFEST_VERSION = "0.1-development"
PROBE_EXECUTION_ENV = "PSA_EXP001C_NONCORE_PILOT"
PROBE_EXECUTION_LOCK = "AUTHORIZED_EXP001C_NONCORE_DEVELOPMENT_PILOT"
PROBE_SOURCE_FILES = (
    "configs/development/exp001c_noncore_formal_shape_fixture.v0.1.json",
    "configs/preregistration/exp001c_prefix_semantics.draft.json",
    "docs/exp001c_prospective_design.md",
    "schemas/exp001c_noncore_pilot_authorization.schema.json",
    "schemas/exp001c_noncore_probe_result.schema.json",
    "schemas/exp001c_prefix_evidence.schema.json",
    "src/psa/cli.py",
    "src/psa/development/__init__.py",
    "src/psa/development/exp001c_probe.py",
    "src/psa/development/exp001c_rwkv_backend.py",
    "src/psa/development/prefix_instrumentation.py",
    "tests/fixtures/exp001c_prefix_logits_fixture.json",
    "tests/test_exp001c_design.py",
    "tests/test_exp001c_prefix_instrumentation.py",
    "tests/test_exp001c_probe.py",
    "tests/test_exp001c_rwkv_backend.py",
)


class Exp001CProbeBackend(Protocol):
    def run_probe(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


def _safe_locked_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        normalized = _relative(candidate, root)
    except ValueError:
        return None
    return candidate if normalized == relative.replace("\\", "/") else None


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "manifest_digest_sha256"
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        data = canonical_json_bytes(dict(value))
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_exp001c_probe_manifest(
    *,
    design_config_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    design_path = Path(design_config_path).resolve()
    model_path = Path(model_config_path).resolve()
    design = _load_object(design_path, "EXP-001C design config")
    _load_object(model_path, "model config")
    authority = design.get("authority")
    if (
        design.get("experiment_id") != EXPERIMENT_ID
        or not isinstance(authority, Mapping)
        or authority.get("development_implementation_authorized") is not True
    ):
        raise ValueError("EXP-001C offline development is not authorized")
    source_digests = {}
    for relative in PROBE_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"probe source file is missing: {relative}")
        source_digests[relative] = sha256_file(path)
    design_relative = _relative(design_path, root)
    model_relative = _relative(model_path, root)
    if source_digests.get(design_relative) != sha256_file(design_path):
        raise ValueError("design config must be present in the locked source inventory")
    manifest = {
        "manifest_version": PROBE_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "status": "development_probe_manifest_unrun",
        "development_only": True,
        "formal_test_set_accessed": False,
        "model_executed": False,
        "design_config": {
            "path": design_relative,
            "sha256": sha256_file(design_path),
            "noncore_pilot_authorized_at_build": authority.get(
                "pilot_run_authorized"
            )
            is True,
        },
        "model_config": {
            "path": model_relative,
            "sha256": sha256_file(model_path),
        },
        "execution_lock": {
            "environment_variable": PROBE_EXECUTION_ENV,
            "required_value": PROBE_EXECUTION_LOCK,
            "satisfied_at_build": False,
        },
        "locked_source_digests": dict(sorted(source_digests.items())),
        "pilot_authorization_file_present": False,
        "formal_run_authorized": False,
        "test_set_generation_authorized": False,
        "result_observation_authorized": False,
    }
    manifest["manifest_digest_sha256"] = sha256_json(_manifest_payload(manifest))
    return manifest


def verify_exp001c_probe_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = _load_object(manifest_path, "EXP-001C probe manifest")
    stored_sources = manifest.get("locked_source_digests")
    inventory_complete = bool(
        isinstance(stored_sources, Mapping)
        and set(str(key) for key in stored_sources) == set(PROBE_SOURCE_FILES)
    )
    source_checks = {}
    for relative in PROBE_SOURCE_FILES:
        path = _safe_locked_path(root, relative)
        expected = (
            stored_sources.get(relative)
            if isinstance(stored_sources, Mapping)
            else None
        )
        source_checks[relative] = bool(
            path is not None
            and path.is_file()
            and isinstance(expected, str)
            and sha256_file(path) == expected
        )
    design = manifest.get("design_config")
    model = manifest.get("model_config")
    digest_valid = manifest.get("manifest_digest_sha256") == sha256_json(
        _manifest_payload(manifest)
    )
    execution_lock = manifest.get("execution_lock")
    boundary_valid = bool(
        manifest.get("manifest_version") == PROBE_MANIFEST_VERSION
        and manifest.get("experiment_id") == EXPERIMENT_ID
        and manifest.get("status") == "development_probe_manifest_unrun"
        and manifest.get("development_only") is True
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("model_executed") is False
        and manifest.get("formal_run_authorized") is False
        and manifest.get("test_set_generation_authorized") is False
        and manifest.get("result_observation_authorized") is False
        and manifest.get("pilot_authorization_file_present") is False
        and isinstance(execution_lock, Mapping)
        and execution_lock.get("environment_variable") == PROBE_EXECUTION_ENV
        and execution_lock.get("required_value") == PROBE_EXECUTION_LOCK
        and execution_lock.get("satisfied_at_build") is False
    )
    design_path = (
        _safe_locked_path(root, design.get("path"))
        if isinstance(design, Mapping)
        else None
    )
    design_valid = bool(
        isinstance(design, Mapping)
        and design.get("path")
        == "configs/preregistration/exp001c_prefix_semantics.draft.json"
        and design_path is not None
        and design_path.is_file()
        and sha256_file(design_path) == design.get("sha256")
    )
    model_path = (
        _safe_locked_path(root, model.get("path"))
        if isinstance(model, Mapping)
        else None
    )
    model_valid = bool(
        isinstance(model, Mapping)
        and model_path is not None
        and model_path.is_file()
        and sha256_file(model_path) == model.get("sha256")
    )
    valid = bool(
        digest_valid
        and boundary_valid
        and design_valid
        and model_valid
        and inventory_complete
        and all(source_checks.values())
    )
    return {
        "verification_version": PROBE_MANIFEST_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "development_probe_manifest_verified" if valid else "invalid",
        "valid": valid,
        "manifest_digest_valid": digest_valid,
        "safety_boundary_valid": boundary_valid,
        "design_config_valid": design_valid,
        "model_config_valid": model_valid,
        "source_inventory_complete": inventory_complete,
        "source_checks": source_checks,
        "model_executed": False,
        "formal_test_set_accessed": False,
    }


def validate_exp001c_probe_execution_authority(
    *,
    manifest_path: str | Path,
    authorization_path: str | Path,
    execution_lock: str,
    project_root: str | Path,
) -> dict[str, Any]:
    if execution_lock != PROBE_EXECUTION_LOCK:
        raise PermissionError("EXP-001C non-Core pilot execution lock is absent")
    verification = verify_exp001c_probe_manifest(
        manifest_path,
        project_root=project_root,
    )
    if verification.get("valid") is not True:
        raise ValueError("EXP-001C probe manifest verification failed")
    root = Path(project_root).resolve()
    manifest = _load_object(manifest_path, "EXP-001C probe manifest")
    design_entry = manifest["design_config"]
    if design_entry.get("noncore_pilot_authorized_at_build") is not True:
        raise PermissionError(
            "EXP-001C probe manifest was built without pilot authority"
        )
    design = _load_object(
        root / str(design_entry["path"]),
        "EXP-001C design config",
    )
    authority = design.get("authority")
    development_authorization = design.get("development_authorization")
    if (
        not isinstance(authority, Mapping)
        or authority.get("pilot_run_authorized") is not True
        or not isinstance(development_authorization, Mapping)
        or development_authorization.get("noncore_pilot_authorized") is not True
        or development_authorization.get("model_execution_authorized") is not True
        or authority.get("test_set_generation_authorized") is not False
        or authority.get("formal_run_authorized") is not False
        or authority.get("result_observation_authorized") is not False
        or authority.get("automatic_rerun_authorized") is not False
        or development_authorization.get("formal_data_generation_authorized")
        is not False
    ):
        raise PermissionError("EXP-001C non-Core pilot is not authorized by design")
    authorization = _load_object(
        authorization_path,
        "EXP-001C non-Core pilot authorization",
    )
    authorization_valid = bool(
        authorization.get("authorization_version") == "0.1"
        and authorization.get("experiment_id") == EXPERIMENT_ID
        and authorization.get("scope") == "noncore_development_pilot_only"
        and authorization.get("authorized") is True
        and authorization.get("probe_manifest_digest_sha256")
        == manifest.get("manifest_digest_sha256")
        and authorization.get("formal_test_set_access_authorized") is False
        and authorization.get("formal_run_authorized") is False
        and authorization.get("result_observation_authorized") is False
    )
    if not authorization_valid:
        raise PermissionError("EXP-001C non-Core pilot authorization is invalid")
    return {
        "valid": True,
        "experiment_id": EXPERIMENT_ID,
        "scope": "noncore_development_pilot_only",
        "manifest_digest_sha256": manifest["manifest_digest_sha256"],
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "result_observation_authorized": False,
    }


def run_exp001c_development_probe(
    *,
    manifest_path: str | Path,
    authorization_path: str | Path,
    output_dir: str | Path,
    backend_factory: Callable[[], Exp001CProbeBackend],
    execution_lock: str,
    project_root: str | Path,
) -> dict[str, Any]:
    authority = validate_exp001c_probe_execution_authority(
        manifest_path=manifest_path,
        authorization_path=authorization_path,
        execution_lock=execution_lock,
        project_root=project_root,
    )
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("EXP-001C probe output directory must be empty")
    manifest = _load_object(manifest_path, "EXP-001C probe manifest")
    backend = backend_factory()
    result = backend.run_probe(manifest)
    if (
        not isinstance(result, Mapping)
        or result.get("development_only") is not True
        or result.get("model_executed") is not True
        or result.get("formal_test_set_accessed") is not False
        or result.get("contains_confirmatory_decision") is not False
    ):
        raise ValueError("EXP-001C development probe result violates safety boundary")
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "probe_result.json"
    _atomic_write(result_path, result)
    summary = {
        "summary_version": PROBE_MANIFEST_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "noncore_development_probe_complete",
        "valid": True,
        "development_only": True,
        "model_executed": True,
        "manifest_digest_sha256": authority["manifest_digest_sha256"],
        "probe_result_sha256": sha256_file(result_path),
        "formal_test_set_accessed": False,
        "contains_confirmatory_decision": False,
        "formal_run_authorized": False,
        "result_observation_authorized": False,
    }
    _atomic_write(destination / "summary.json", summary)
    return summary
