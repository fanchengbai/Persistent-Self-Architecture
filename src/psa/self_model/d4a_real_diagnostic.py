from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.model.rwkv7 import RWKV7Adapter, load_model_config
from psa.self_model.d4a_failure_diagnostic_runtime import (
    D4A_RECORDED_ROUNDS,
    D4A_TOKEN_IDS,
    RWKV7RecompiledUnmodifiedRuntime,
    execute_d4a_fake_or_authorized_diagnostic,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    RWKV7InstrumentedOffRuntime,
)


REPORT_VERSION = "0.1-d4a-real-diagnostic"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d4a_real_diagnostic.json"
)
D4A_EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D4A_REAL_DIAGNOSTIC"
D4A_EXECUTION_LOCK_VALUE = "AUTHORIZED_D4A_REAL_2_9B_DIAGNOSTIC_ONCE"
D4A_OWNER_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 D4A 真实2.9B最小诊断一次，并授权观察本次诊断结果；"
    "不授权重跑D4、自动重跑、D5、active injection或Self效果实验。"
)
D4A_CLOUD_STATIC_REPORT_DIGEST = (
    "f8e74653ecb170c5a5bcd870b1c8efc90cbd311417f43adf3ed61f3658386e57"
)
ENTRY_SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d4a_real_diagnostic_entry.md",
    "scripts/run_self_model_v0_1_d4a_real_diagnostic.py",
    "scripts/verify_self_model_v0_1_d4a_real_diagnostic_entry.py",
    "src/psa/self_model/d4a_real_diagnostic.py",
    "src/psa/self_model/d4a_failure_diagnostic_runtime.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d4a_real_diagnostic.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise PermissionError(f"{field} must equal the frozen D4A value")


def _object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _read_spec(path: str | Path) -> dict[str, Any]:
    payload = _object(path, "D4A real diagnostic config")
    exact = {
        "protocol_version": REPORT_VERSION,
        "stage": "D4A_real_2_9b_minimal_diagnostic_entry",
        "status": "entry_implemented_execution_not_authorized",
        "development_only": True,
        "diagnostic_only": True,
        "model_config_path": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_sha256": (
            "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
        ),
        "model_id": "rwkv7-g1h-2.9b-20260710",
        "routes": [
            "original_baseline",
            "g0_recompiled_unmodified",
            "off_g2_instrumented",
        ],
        "recorded_rounds": D4A_RECORDED_ROUNDS,
        "model_forward_call_count": 9,
        "discarded_warmup_call_count": 0,
        "within_route_comparison_count": 9,
        "cross_route_comparison_count": 27,
        "execution_lock_env": D4A_EXECUTION_LOCK_ENV,
        "execution_lock_value": D4A_EXECUTION_LOCK_VALUE,
        "required_owner_authorization_text": D4A_OWNER_AUTHORIZATION_TEXT,
        "entry_implementation_authorized": True,
        "no_model_verification_authorized": True,
        "execution_authorized_at_implementation": False,
        "future_exact_owner_authorization_required": True,
        "future_machine_authorization_required": True,
        "single_use_claim_required": True,
        "claim_consumed_before_weights_or_model": True,
        "result_observation_requires_same_authorization": True,
        "d4_status_can_change": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "failure_action": "persist_failure_and_stop_claim_consumed",
    }
    for field, expected in exact.items():
        _require_exact(payload.get(field), expected, field)
    prerequisite = payload.get("prerequisite")
    _require_exact(
        prerequisite,
        {
            "d4_status": "failed_preserved",
            "d4a_cloud_static_status": "d4a_cloud_static_verified",
            "d4a_cloud_static_report_digest_sha256": D4A_CLOUD_STATIC_REPORT_DIGEST,
        },
        "prerequisite",
    )
    _require_exact(
        payload.get("fixture"),
        {"token_ids": D4A_TOKEN_IDS, "state_input": "none", "full_output": False},
        "fixture",
    )
    return payload


def _git_metadata(project_root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _require_clean_main(git: Mapping[str, str]) -> None:
    if git.get("branch") != "main" or git.get("status_porcelain"):
        raise RuntimeError("D4A requires a clean main worktree")
    commit = git.get("commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("D4A requires a full lowercase Git commit")


def _resolve_inside(root: Path, value: str | Path, parent: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    allowed = (root / parent).resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise PermissionError(f"D4A path must remain inside {parent}")
    return resolved


def build_d4a_real_authorization(
    *,
    config_path: str | Path,
    project_root: str | Path,
    authorization_text: str,
    git_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if authorization_text != D4A_OWNER_AUTHORIZATION_TEXT:
        raise PermissionError("D4A owner authorization text is not exact")
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    expected_config = (root / CONFIG_RELATIVE_PATH).resolve()
    if config_file != expected_config:
        raise PermissionError("D4A authorization config path is not frozen")
    spec = _read_spec(config_file)
    git = dict(git_metadata) if git_metadata is not None else _git_metadata(root)
    _require_clean_main(git)
    authorization = {
        "authorization_version": REPORT_VERSION,
        "stage": spec["stage"],
        "scope": "real_2_9b_d4a_minimal_diagnostic_once",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "model_execution_authorized": True,
        "diagnostic_result_observation_authorized": True,
        "d4_status_change_authorized": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_d4a_real_authorization(
    *,
    authorization_path: str | Path,
    config_path: str | Path,
    git: Mapping[str, str],
) -> dict[str, Any]:
    authorization = _object(authorization_path, "D4A machine authorization")
    stored_digest = authorization.get("authorization_digest_sha256")
    digest_payload = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_digest_sha256"
    }
    expected = {
        "authorization_version": REPORT_VERSION,
        "scope": "real_2_9b_d4a_minimal_diagnostic_once",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": D4A_OWNER_AUTHORIZATION_TEXT,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "model_execution_authorized": True,
        "diagnostic_result_observation_authorized": True,
        "d4_status_change_authorized": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    for field, value in expected.items():
        _require_exact(authorization.get(field), value, f"authorization.{field}")
    if not isinstance(authorization.get("authorized_at_utc"), str):
        raise PermissionError("authorization timestamp is missing")
    if not isinstance(stored_digest, str) or sha256_json(digest_payload) != stored_digest:
        raise PermissionError("D4A machine authorization digest is invalid")
    return authorization


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def _create_claim(
    *,
    output_dir: Path,
    config_path: Path,
    authorization_path: Path,
    git: Mapping[str, str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D4A output directory is not empty; rerun refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d4a_single_use_execution_claim_consumed",
        "created_at_utc": _utc_now(),
        "single_use": True,
        "automatic_rerun_authorized": False,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
    }
    return write_json_exclusive(output_dir / "execution_claim.json", claim)


def _installed_source() -> tuple[str, Path, bytes, str]:
    installed_version = version("rwkv")
    source_path = Path(distribution("rwkv").locate_file("rwkv/model.py")).resolve()
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    _require_exact(installed_version, EXPECTED_RWKV_PACKAGE_VERSION, "rwkv version")
    _require_exact(source_digest, EXPECTED_RWKV_MODEL_SOURCE_SHA256, "model.py digest")
    return installed_version, source_path, source_bytes, source_digest


def run_d4a_real_diagnostic(
    *,
    config_path: str | Path,
    authorization_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D4A execution config path is not frozen")
    spec = _read_spec(config_file)
    if os.environ.get(D4A_EXECUTION_LOCK_ENV) != D4A_EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D4A execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D4A")
    authorization_file = _resolve_inside(root, authorization_path, "results/authorizations")
    destination = _resolve_inside(root, output_dir, "results/development")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d4a_real_authorization(
        authorization_path=authorization_file,
        config_path=config_file,
        git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _installed_source()

    claim_path = _create_claim(
        output_dir=destination,
        config_path=config_file,
        authorization_path=authorization_file,
        git=git,
    )
    started = time.perf_counter()
    try:
        model_config_path = (root / spec["model_config_path"]).resolve()
        _require_exact(
            sha256_file(model_config_path), spec["model_config_sha256"], "model config digest"
        )
        model_config = load_model_config(model_config_path, root, verify_files=True)
        _require_exact(model_config.model_id, spec["model_id"], "loaded model id")
        for key, value in model_config.environment.items():
            os.environ[key] = value
        adapter = RWKV7Adapter.load(model_config)
        torch = adapter.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        rwkv_module = sys.modules["rwkv.model"]
        g0 = RWKV7RecompiledUnmodifiedRuntime(
            base_model=adapter.model,
            upstream_source_bytes=source_bytes,
            upstream_globals=vars(rwkv_module),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
        )
        off_g2 = RWKV7InstrumentedOffRuntime(
            base_model=adapter.model,
            upstream_source_bytes=source_bytes,
            upstream_globals=vars(rwkv_module),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
        )
        diagnostic = execute_d4a_fake_or_authorized_diagnostic(
            base_model=adapter.model,
            g0=g0,
            off_g2=off_g2,
            torch=torch,
        )
        torch.cuda.synchronize()
        report = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": (
                "d4a_real_diagnostic_complete"
                if diagnostic["valid"]
                else "d4a_real_diagnostic_incomplete"
            ),
            "valid": diagnostic["valid"],
            "development_only": True,
            "diagnostic_only": True,
            "git": git,
            "config": {"path": str(config_file), "sha256": sha256_file(config_file)},
            "authorization_digest_sha256": authorization[
                "authorization_digest_sha256"
            ],
            "execution_claim_sha256": sha256_file(claim_path),
            "model": adapter.model_metadata(),
            "upstream": {
                "package": "rwkv",
                "version": installed_version,
                "model_source_path": str(source_path),
                "model_source_sha256": source_digest,
            },
            "diagnostic": diagnostic,
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "safety": {
                "d4_status_changed": False,
                "d4_rerun": False,
                "active_injection_implemented": False,
                "active_injection_executed": False,
                "self_projection_constructed": False,
                "self_effect_experiment_run": False,
                "d5_authorized": False,
                "confirmatory_decision_made": False,
                "automatic_rerun_authorized": False,
                "diagnostic_result_observation_authorized": True,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        (destination / "report.json").write_bytes(canonical_json_bytes(report))
        return report
    except BaseException as error:
        failure = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d4a_execution_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "d4_status_changed": False,
            "d5_authorized": False,
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        (destination / "failure.json").write_bytes(canonical_json_bytes(failure))
        raise


def _call_lines(source: str, function_name: str) -> dict[str, int]:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    lines: dict[str, int] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        lines.setdefault(name, node.lineno)
    return lines


def build_d4a_real_entry_static_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D4A static entry config path is not frozen")
    spec = _read_spec(config_file)
    module_path = root / "src/psa/self_model/d4a_real_diagnostic.py"
    module_source = module_path.read_text(encoding="utf-8")
    calls = _call_lines(module_source, "run_d4a_real_diagnostic")
    required_calls = (
        "validate_d4a_real_authorization",
        "_installed_source",
        "_create_claim",
        "load_model_config",
        "load",
        "execute_d4a_fake_or_authorized_diagnostic",
    )
    source_digests = {path: sha256_file(root / path) for path in ENTRY_SOURCE_PATHS}
    checks = {
        "config_path_frozen": True,
        "config_valid": bool(spec),
        "d4_failure_prerequisite_preserved": spec["prerequisite"]["d4_status"]
        == "failed_preserved",
        "cloud_static_digest_frozen": spec["prerequisite"][
            "d4a_cloud_static_report_digest_sha256"
        ]
        == D4A_CLOUD_STATIC_REPORT_DIGEST,
        "fixture_is_only_failed_cell": spec["fixture"]
        == {"token_ids": [2764], "state_input": "none", "full_output": False},
        "latin_schedule_frozen": spec["recorded_rounds"] == D4A_RECORDED_ROUNDS,
        "nine_calls_no_discarded_warmup": spec["model_forward_call_count"] == 9
        and spec["discarded_warmup_call_count"] == 0,
        "comparison_counts_frozen": spec["within_route_comparison_count"] == 9
        and spec["cross_route_comparison_count"] == 27,
        "implementation_authorized": spec["entry_implementation_authorized"] is True,
        "execution_not_authorized_at_implementation": spec[
            "execution_authorized_at_implementation"
        ]
        is False,
        "future_exact_owner_authorization_required": spec[
            "future_exact_owner_authorization_required"
        ]
        is True,
        "single_use_claim_required": spec["single_use_claim_required"] is True,
        "all_required_entry_calls_present": all(name in calls for name in required_calls),
        "claim_precedes_weight_verification_and_model_load": calls["_create_claim"]
        < calls["load_model_config"]
        < calls["load"],
        "diagnostic_core_runs_after_claim": calls["_create_claim"]
        < calls["execute_d4a_fake_or_authorized_diagnostic"],
        "d4_cannot_change": spec["d4_status_can_change"] is False,
        "d4_rerun_not_authorized": spec["d4_rerun_authorized"] is False,
        "active_not_authorized": spec["active_injection_authorized"] is False,
        "self_effect_not_authorized": spec["self_effect_experiment_authorized"]
        is False,
        "d5_not_authorized": spec["d5_authorized"] is False,
        "automatic_rerun_not_authorized": spec["automatic_rerun_authorized"]
        is False,
        "source_inventory_complete": len(source_digests) == len(ENTRY_SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D4A real entry static verification failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": "d4a_real_diagnostic_entry_static_verified",
        "valid": True,
        "checks": checks,
        "source_digests": source_digests,
        "entry_call_lines": {name: calls[name] for name in required_calls},
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "execution_claim_created": False,
            "machine_authorization_created": False,
            "active_injection_implemented": False,
            "self_effect_experiment_run": False,
            "d5_authorized": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
