from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.model.rwkv7 import RWKV7Adapter, load_model_config
from psa.self_model.d4a_real_diagnostic import _installed_source
from psa.self_model.d4b_real_off_equivalence import (
    _git_metadata,
    _require_clean_main,
    write_json_exclusive,
)
from psa.self_model.d5c_mechanism_runtime import (
    D5CSyntheticProbe,
    FIXTURES,
    RWKV7D5CActiveRuntime,
)
from psa.self_model.d5c_p1_engineering_validation import (
    ACTIVE_COMPARISONS,
    CONTROL_COMPARISONS,
    ROUTE_ORDER,
    execute_d5c_p1_engineering_core,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)


REPORT_VERSION = "0.1-d5c-p1-real-engineering-entry"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_p1_real_engineering_validation.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d5c_p1_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "results/authorizations/self_model_v0_1_d5c_p1_real_engineering_v01.json"
)
OUTPUT_RELATIVE_DIR = (
    "results/development/self_model_v0_1_d5c_p1_real_engineering_v01"
)
EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D5C_P1_REAL_ENGINEERING"
EXECUTION_LOCK_VALUE = "AUTHORIZED_D5C_P1_PATCHED_REAL_2_9B_ENGINEERING_ONCE"
IMPLEMENTATION_CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 D5C-P1 补丁后真实2.9B非Core工程验证设计与无模型安全入口实现；"
    "不授权模型加载或执行、不重跑原D5C、不改变历史失败结论，也不授权D5D/D5E、"
    "正式测试集、Self效果结论、真实Self projection、Self Updater或自动重跑。"
)
FUTURE_EXECUTION_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 D5C-P1 补丁后真实2.9B非Core工程验证一次"
    "（固定两个夹具、每夹具6次、共12次调用），并授权观察本次工程结果；"
    "不授权重跑原D5C、不改变历史失败结论，也不授权D5D/D5E、正式测试集、"
    "Self效果结论、真实Self projection、Self Updater或自动重跑。"
)
PATCH_REPORT_DIGEST = "49f7444c6de98f7f751f15242ad43183ef76ea98fd1fd5455a8d521c3e6ac731"
PATCH_RUNTIME_DIGEST = "e4ae5c5bee74a85a4dea8a9b8eb16e3b6e19ef6b375020ffc849a09cbd7bbc32"
INSTRUMENTER_DIGEST = "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"
HISTORICAL_D5C_REPORT_DIGEST = "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21"
HISTORICAL_D5C_CLAIM_DIGEST = "75d69ae3ad4550361cc53d03ae5d89fd636f045d31a6cd62974c4dc15496f12f"
MODEL_CONFIG_DIGEST = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
AUTHORIZATION_FIELDS = {
    "authorization_version", "stage", "scope", "authorized",
    "authorization_basis", "authorization_text", "authorized_at_utc",
    "git_commit", "config_sha256", "entry_static_report_sha256",
    "model_execution_authorized", "result_observation_authorized",
    "engineering_validation_only", "historical_d5c_rerun_authorized",
    "historical_d5c_conclusion_change_authorized", "d5d_authorized",
    "d5e_authorized", "formal_test_set_authorized",
    "self_effect_conclusion_authorized", "real_self_projection_authorized",
    "self_updater_authorized", "automatic_rerun_authorized", "single_use",
    "authorization_digest_sha256",
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_p1_implementation_authorization.md",
    "docs/self_model_v0_1_d5c_p1_real_engineering_validation.md",
    "scripts/run_self_model_v0_1_d5c_p1_real_engineering_validation.py",
    "scripts/verify_self_model_v0_1_d5c_p1_real_entry.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/d5c_p1_engineering_validation.py",
    "src/psa/self_model/d5c_p1_real_entry.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d5c_p1_real_entry.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise PermissionError(f"D5C-P1 {field} differs from the frozen value")


def _require_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved != (root / relative).resolve():
        raise PermissionError(f"D5C-P1 {label} path is not frozen")
    return resolved


def read_spec(path: str | Path) -> dict[str, Any]:
    spec = _object(path, "D5C-P1 config")
    exact = {
        "protocol_version": REPORT_VERSION,
        "stage": "D5C-P1_patched_real_2_9b_noncore_engineering_entry",
        "status": "design_and_no_model_entry_implemented_execution_not_authorized",
        "development_only": True,
        "implementation_confirmation_text": IMPLEMENTATION_CONFIRMATION_TEXT,
        "frozen_prerequisites": {
            "runtime_patch_report_sha256": PATCH_REPORT_DIGEST,
            "patched_runtime_source_sha256": PATCH_RUNTIME_DIGEST,
            "instrumenter_source_sha256": INSTRUMENTER_DIGEST,
            "historical_d5c_report_sha256": HISTORICAL_D5C_REPORT_DIGEST,
            "historical_d5c_claim_sha256": HISTORICAL_D5C_CLAIM_DIGEST,
            "historical_d5c_status": "d5c_mechanism_smoke_failed",
            "historical_d5c_decision": "stop_without_rerun",
        },
        "model_config_path": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_sha256": MODEL_CONFIG_DIGEST,
        "model_id": "rwkv7-g1h-2.9b-20260710",
        "upstream": {
            "package": "rwkv",
            "version": EXPECTED_RWKV_PACKAGE_VERSION,
            "model_source_sha256": EXPECTED_RWKV_MODEL_SOURCE_SHA256,
            "rwkv_de_version": "unset",
        },
        "fixtures": [dict(value) for value in FIXTURES],
        "route_order": list(ROUTE_ORDER),
        "control_comparisons": [list(value) for value in CONTROL_COMPARISONS],
        "active_comparisons": [list(value) for value in ACTIVE_COMPARISONS],
        "counts": {
            "fixture_count": 2,
            "calls_per_fixture": 6,
            "model_forward_calls_total": 12,
            "wrapped_forward_calls_total": 8,
            "active_forward_calls_total": 2,
            "control_comparisons_total": 10,
            "active_comparisons_total": 4,
            "cleanup_evidence_records_total": 8,
            "active_callback_calls_total": 64,
            "active_probe_applications_total": 2,
        },
        "acceptance": {
            "all_outputs_finite": True,
            "original_before_after_active_exact": True,
            "all_control_comparisons_exact": True,
            "active_differs_from_before_and_after": True,
            "temporary_bindings_absent_after_every_wrapped_call": True,
            "callback_counts_exact": True,
            "engineering_validation_only": True,
        },
        "execution_lock_env": EXECUTION_LOCK_ENV,
        "execution_lock_value": EXECUTION_LOCK_VALUE,
        "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
        "authorization_path": AUTHORIZATION_RELATIVE_PATH,
        "output_dir": OUTPUT_RELATIVE_DIR,
        "future_execution_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "design_and_entry_implementation_authorized": True,
        "no_model_verification_authorized": True,
        "execution_authorized_at_implementation": False,
        "future_exact_owner_authorization_required": True,
        "future_machine_authorization_required": True,
        "single_use_claim_required": True,
        "claim_consumed_before_model_config_weights_or_load": True,
        "historical_d5c_authorization_reusable": False,
        "historical_d5c_claim_reusable": False,
        "historical_d5c_rerun_authorized": False,
        "historical_d5c_conclusion_change_authorized": False,
        "d5d_authorized": False,
        "d5e_authorized": False,
        "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "real_self_projection_authorized": False,
        "self_updater_authorized": False,
        "automatic_rerun_authorized": False,
        "failure_action": "persist_failure_and_stop_p1_claim_consumed",
    }
    for field, expected in exact.items():
        _require_exact(spec.get(field), expected, field)
    return spec


def _authorization_schema(root: Path) -> dict[str, Any]:
    schema = _object(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH, "D5C-P1 authorization schema")
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        not isinstance(properties, dict)
        or set(properties) != AUTHORIZATION_FIELDS
        or not isinstance(required, list)
        or set(required) != AUTHORIZATION_FIELDS
        or schema.get("additionalProperties") is not False
    ):
        raise RuntimeError("D5C-P1 authorization schema changed")
    return schema


def _require_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D5C-P1 authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D5C-P1 authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D5C-P1 authorization timestamp must be UTC")


def build_p1_authorization(
    *,
    config_path: str | Path,
    project_root: str | Path,
    authorization_text: str,
    git_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    read_spec(config_file)
    _authorization_schema(root)
    _require_exact(authorization_text, FUTURE_EXECUTION_AUTHORIZATION_TEXT, "authorization text")
    git = dict(git_metadata or _git_metadata(root))
    _require_clean_main(git)
    authorization = {
        "authorization_version": REPORT_VERSION,
        "stage": "D5C-P1_patched_real_2_9b_noncore_engineering_execution",
        "scope": "one_fixed_12_call_engineering_validation_and_observation",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_future_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "entry_static_report_sha256": build_p1_entry_static_report(
            config_path=config_file, project_root=root
        )["report_digest_sha256"],
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "engineering_validation_only": True,
        "historical_d5c_rerun_authorized": False,
        "historical_d5c_conclusion_change_authorized": False,
        "d5d_authorized": False,
        "d5e_authorized": False,
        "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "real_self_projection_authorized": False,
        "self_updater_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_p1_authorization(
    *, authorization_path: str | Path, config_path: str | Path,
    project_root: str | Path, git: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    authorization = _object(authorization_path, "D5C-P1 machine authorization")
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D5C-P1 machine authorization fields changed")
    expected = build_p1_authorization(
        config_path=config_path, project_root=root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        git_metadata=git,
    )
    for field, value in expected.items():
        if field not in {"authorized_at_utc", "authorization_digest_sha256"}:
            _require_exact(authorization.get(field), value, f"authorization.{field}")
    _require_utc_timestamp(authorization.get("authorized_at_utc"))
    stored_digest = authorization.get("authorization_digest_sha256")
    payload = {key: value for key, value in authorization.items() if key != "authorization_digest_sha256"}
    if not isinstance(stored_digest, str) or sha256_json(payload) != stored_digest:
        raise PermissionError("D5C-P1 machine authorization digest is invalid")
    return authorization


def _create_claim(
    *, output_dir: Path, config_path: Path, authorization_path: Path,
    git: Mapping[str, str], entry_static_report_sha256: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D5C-P1 output directory is not empty; reuse refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d5c_p1_single_use_engineering_claim_consumed",
        "created_at_utc": _utc_now(),
        "single_use": True,
        "automatic_rerun_authorized": False,
        "historical_d5c_rerun_authorized": False,
        "historical_d5c_claim_reused": False,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "entry_static_report_sha256": entry_static_report_sha256,
        "model_forward_call_count": 12,
        "engineering_validation_only": True,
    }
    return write_json_exclusive(output_dir / "execution_claim.json", claim)


def run_p1_real_engineering_validation(
    *, config_path: str | Path, authorization_path: str | Path,
    project_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D5C-P1 execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D5C-P1")
    authorization_file = _require_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_p1_authorization(
        authorization_path=authorization_file, config_path=config_file,
        project_root=root, git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _installed_source()
    _require_exact(installed_version, EXPECTED_RWKV_PACKAGE_VERSION, "installed package version")
    _require_exact(source_digest, EXPECTED_RWKV_MODEL_SOURCE_SHA256, "installed source digest")
    claim_path = _create_claim(
        output_dir=destination, config_path=config_file,
        authorization_path=authorization_file, git=git,
        entry_static_report_sha256=authorization["entry_static_report_sha256"],
    )
    started = time.perf_counter()
    try:
        model_config_path = (root / spec["model_config_path"]).resolve()
        _require_exact(sha256_file(model_config_path), spec["model_config_sha256"], "model config")
        model_config = load_model_config(model_config_path, root, verify_files=True)
        _require_exact(model_config.model_id, spec["model_id"], "model id")
        for key, value in model_config.environment.items():
            os.environ[key] = value
        adapter = RWKV7Adapter.load(model_config)
        torch = adapter.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        claim_digest = sha256_file(claim_path)
        authorization_digest = sha256_file(authorization_file)
        probe = D5CSyntheticProbe(
            torch=torch, execution_claim_sha256=claim_digest,
            machine_authorization_sha256=authorization_digest,
        )
        runtime = RWKV7D5CActiveRuntime(
            base_model=adapter.model, upstream_source_bytes=source_bytes,
            upstream_globals=vars(sys.modules["rwkv.model"]),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
            execution_claim_sha256=claim_digest,
            machine_authorization_sha256=authorization_digest,
        )
        core = execute_d5c_p1_engineering_core(
            base_model=adapter.model, active_runtime=runtime, probe=probe, torch=torch
        )
        torch.cuda.synchronize()
        report = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": core["status"],
            "valid": core["valid"],
            "development_only": True,
            "git": git,
            "config_sha256": sha256_file(config_file),
            "authorization_digest_sha256": authorization["authorization_digest_sha256"],
            "execution_claim_sha256": claim_digest,
            "model": adapter.model_metadata(),
            "upstream": {
                "package": "rwkv", "version": installed_version,
                "model_source_path": str(source_path),
                "model_source_sha256": source_digest,
            },
            "runtime_core": core,
            "interpretation": "post_patch_engineering_validation_only_no_self_effect_conclusion",
            "historical_d5c": {
                "report_sha256": HISTORICAL_D5C_REPORT_DIGEST,
                "claim_sha256": HISTORICAL_D5C_CLAIM_DIGEST,
                "result_changed": False,
                "rerun": False,
            },
            "decision_effect": "engineering_gate_observation_only",
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "safety": {
                "real_2_9b_model_loaded": True,
                "real_2_9b_model_executed": True,
                "historical_d5c_rerun": False,
                "historical_d5c_conclusion_changed": False,
                "d5d_authorized": False, "d5e_authorized": False,
                "formal_test_set_used": False,
                "self_effect_conclusion_made": False,
                "real_self_projection_constructed": False,
                "self_updater_used": False,
                "automatic_rerun_authorized": False,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        write_json_exclusive(destination / "report.json", report)
        return report
    except BaseException as error:
        failure = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d5c_p1_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "historical_d5c_rerun_authorized": False,
            "historical_d5c_conclusion_change_authorized": False,
            "d5d_authorized": False, "d5e_authorized": False,
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        write_json_exclusive(destination / "failure.json", failure)
        raise


def _call_lines(source: str, function_name: str) -> dict[str, int]:
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    lines: dict[str, int] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            lines[name] = min(lines.get(name, node.lineno), node.lineno)
    return lines


def build_p1_entry_static_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    schema = _authorization_schema(root)
    module_source = (root / "src/psa/self_model/d5c_p1_real_entry.py").read_text(encoding="utf-8")
    core_source = (root / "src/psa/self_model/d5c_p1_engineering_validation.py").read_text(encoding="utf-8")
    runtime_path = root / "src/psa/self_model/d5c_mechanism_runtime.py"
    instrumenter_path = root / "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
    calls = _call_lines(module_source, "run_p1_real_engineering_validation")
    required_calls = (
        "validate_p1_authorization", "_installed_source", "_create_claim",
        "sha256_file", "load_model_config", "load", "D5CSyntheticProbe",
        "RWKV7D5CActiveRuntime", "execute_d5c_p1_engineering_core",
    )
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    properties = schema["properties"]
    checks = {
        "config_valid": bool(spec),
        "implementation_confirmation_recorded": IMPLEMENTATION_CONFIRMATION_TEXT in (
            root / "docs/self_model_v0_1_d5c_p1_implementation_authorization.md"
        ).read_text(encoding="utf-8"),
        "historical_failure_and_claim_frozen": spec["frozen_prerequisites"]
        ["historical_d5c_report_sha256"] == HISTORICAL_D5C_REPORT_DIGEST
        and spec["frozen_prerequisites"]["historical_d5c_claim_sha256"]
        == HISTORICAL_D5C_CLAIM_DIGEST,
        "patched_runtime_digest_frozen": sha256_file(runtime_path) == PATCH_RUNTIME_DIGEST,
        "instrumenter_digest_unchanged": sha256_file(instrumenter_path) == INSTRUMENTER_DIGEST,
        "two_noncore_fixtures_frozen": spec["fixtures"] == [dict(value) for value in FIXTURES],
        "six_call_route_order_frozen": spec["route_order"] == list(ROUTE_ORDER),
        "twelve_total_calls_frozen": spec["counts"]["model_forward_calls_total"] == 12,
        "comparison_counts_frozen": spec["counts"]["control_comparisons_total"] == 10
        and spec["counts"]["active_comparisons_total"] == 4,
        "cleanup_evidence_count_frozen": spec["counts"]["cleanup_evidence_records_total"] == 8,
        "core_checks_managed_bindings_after_wrapped_calls": (
            "temporary_bindings_absent_after_every_wrapped_call" in core_source
            and "base_model.__dict__" in core_source
        ),
        "authorization_schema_exact": set(properties) == AUTHORIZATION_FIELDS
        and set(schema["required"]) == AUTHORIZATION_FIELDS
        and properties["authorization_text"]["const"]
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "execution_not_authorized_at_implementation": spec[
            "execution_authorized_at_implementation"
        ] is False,
        "future_exact_owner_authorization_required": spec[
            "future_exact_owner_authorization_required"
        ] is True,
        "historical_authorization_and_claim_not_reusable": not spec[
            "historical_d5c_authorization_reusable"
        ] and not spec["historical_d5c_claim_reusable"],
        "single_use_claim_required": spec["single_use_claim_required"] is True,
        "all_required_entry_calls_present": all(name in calls for name in required_calls),
        "claim_precedes_model_config_weights_and_load": calls["_create_claim"]
        < calls["sha256_file"] < calls["load_model_config"] < calls["load"],
        "core_runs_after_claim": calls["_create_claim"]
        < calls["execute_d5c_p1_engineering_core"],
        "historical_d5c_rerun_and_conclusion_change_closed": not spec[
            "historical_d5c_rerun_authorized"
        ] and not spec["historical_d5c_conclusion_change_authorized"],
        "d5d_d5e_closed": not spec["d5d_authorized"] and not spec["d5e_authorized"],
        "formal_self_and_automatic_gates_closed": not spec["formal_test_set_authorized"]
        and not spec["self_effect_conclusion_authorized"]
        and not spec["real_self_projection_authorized"]
        and not spec["self_updater_authorized"]
        and not spec["automatic_rerun_authorized"],
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C-P1 entry static verification failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": "d5c_p1_real_engineering_entry_static_verified",
        "valid": True,
        "checks": checks,
        "source_digests": source_digests,
        "entry_call_lines": {name: calls[name] for name in required_calls},
        "future_exact_owner_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "next_gate": "remote_no_model_static_verification_then_separate_execution_authorization",
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "execution_claim_created": False,
            "machine_authorization_created": False,
            "historical_d5c_rerun": False,
            "historical_d5c_conclusion_changed": False,
            "d5d_authorized": False, "d5e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "real_self_projection_constructed": False,
            "self_updater_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
