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
    CONTROL_ROUTES,
    FIXTURES,
    HIDDEN_DIMENSION,
    N_LAYER,
    PRECONDITION_ORDER,
    ROUTES,
    SCORED_ROUNDS,
    TARGET_LAYER_INDEX,
    TARGET_RESIDUAL_RMS_RATIO,
    D5CSyntheticProbe,
    RWKV7D5CActiveRuntime,
    execute_d5c_mechanism_core,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)


REPORT_VERSION = "0.1-coupling-d5c-real-mechanism"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d5c_real_mechanism.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d5c_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "results/authorizations/self_model_v0_1_d5c_real_mechanism_v01.json"
)
OUTPUT_RELATIVE_DIR = (
    "results/development/self_model_v0_1_d5c_real_mechanism_v01"
)
D5C_EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D5C_REAL_MECHANISM_SMOKE"
D5C_EXECUTION_LOCK_VALUE = "AUTHORIZED_D5C_REAL_2_9B_NONCORE_MECHANISM_ONCE"
D5C_OWNER_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 Coupling-D5C 真实2.9B非Core机制冒烟一次"
    "（固定synthetic probe、0-based第15层、42次调用），并授权观察本次机制结果；"
    "不授权重跑D4/D4B/D5C、自动重跑、D5D/D5E、正式测试集、Self效果结论、"
    "真实Self projection或Self Updater。"
)
IMPLEMENTATION_CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 Coupling-D5C 真实2.9B非Core机制冒烟设计与无模型安全入口实现；"
    "不授权模型加载或执行、D5D/D5E、正式测试集、Self效果结论、Self Updater或自动重跑。"
)
D5A_REPORT_DIGEST = "48c6f609f53a2d5223366abcc9b8b3af3936ede282fcd880113edb4f1cff89d3"
D5B_REPORT_DIGEST = "b6ca7f56caabff0c64d85bda6735d6dd185d41fdc832cbc8d4f30f19a1bc8d16"
D4B_REAL_REPORT_DIGEST = "8befb5f4b2ce90241b66aff1f43bce59645d367c14f6594169e9c454fcf36a20"
MODEL_CONFIG_SHA256 = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
AUTHORIZATION_FIELDS = {
    "authorization_version", "stage", "scope", "authorized",
    "authorization_basis", "authorization_text", "authorized_at_utc",
    "git_commit", "config_sha256", "entry_static_report_sha256",
    "model_execution_authorized", "result_observation_authorized",
    "mechanism_only", "d4_rerun_authorized", "d4b_rerun_authorized",
    "d5c_rerun_authorized", "d5d_authorized", "d5e_authorized",
    "formal_test_set_authorized", "self_effect_conclusion_authorized",
    "real_self_projection_authorized", "self_updater_authorized",
    "automatic_rerun_authorized", "single_use", "authorization_digest_sha256",
}
ENTRY_SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d5c_implementation_authorization.md",
    "docs/self_model_v0_1_coupling_d5c_real_mechanism.md",
    "scripts/run_self_model_v0_1_coupling_d5c_real_mechanism.py",
    "scripts/verify_self_model_v0_1_coupling_d5c_entry.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/d5c_real_mechanism.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d5c_real_mechanism.py",
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
        raise PermissionError(f"{field} must equal the frozen D5C value")


def _require_exact_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if resolved != (root / relative).resolve():
        raise PermissionError(f"D5C {label} path is not frozen")
    return resolved


def _read_spec(path: str | Path) -> dict[str, Any]:
    spec = _object(path, "D5C real mechanism config")
    exact = {
        "protocol_version": REPORT_VERSION,
        "stage": "Coupling-D5C_real_2_9b_noncore_mechanism_smoke_entry",
        "status": "design_and_no_model_entry_implemented_execution_not_authorized",
        "development_only": True,
        "prerequisite": {
            "d5a_report_digest_sha256": D5A_REPORT_DIGEST,
            "d5b_report_digest_sha256": D5B_REPORT_DIGEST,
            "d4b_real_report_digest_sha256": D4B_REAL_REPORT_DIGEST,
        },
        "model_config_path": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_id": "rwkv7-g1h-2.9b-20260710",
        "upstream": {
            "package": "rwkv", "version": EXPECTED_RWKV_PACKAGE_VERSION,
            "model_source_sha256": EXPECTED_RWKV_MODEL_SOURCE_SHA256,
            "rwkv_de_version": "unset",
        },
        "fixtures": [dict(value) for value in FIXTURES],
        "routes": list(ROUTES),
        "fixed_preconditioning_order": list(PRECONDITION_ORDER),
        "scored_rounds": [list(value) for value in SCORED_ROUNDS],
        "probe": {
            "kind": "deterministic_synthetic_unit_rms_vector_not_self_representation",
            "hidden_dimension": HIDDEN_DIMENSION,
            "n_layer": N_LAYER,
            "target_layer_index_zero_based": TARGET_LAYER_INDEX,
            "target_layer_rule": "floor((n_layer-1)/2)",
            "phase": "post_ffn_residual",
            "target_residual_rms_ratio": TARGET_RESIDUAL_RMS_RATIO,
            "gate": 1.0,
            "sequence_policy": "broadcast_same_delta_to_each_position",
            "projection_trained": False,
            "real_self_projection": False,
            "effect_layer_selected": False,
        },
        "counts": {
            "fixture_count": 2, "prefix_calls_per_fixture": 1,
            "preconditioning_calls_per_fixture": 4, "scored_calls_per_fixture": 16,
            "model_forward_calls_per_fixture": 21, "model_forward_calls_total": 42,
            "within_route_comparisons_total": 24,
            "control_cross_route_comparisons_total": 24,
            "active_control_comparisons_total": 24,
            "active_callback_calls_total": 320,
            "active_probe_applications_total": 10,
        },
        "acceptance": {
            "within_route_all_exact": True, "all_control_pairs_exact": True,
            "active_differs_from_each_control_in_logits_or_state": True,
            "all_outputs_finite": True,
            "all_shapes_dtypes_devices_compatible": True,
            "callback_count_exact": True, "claim_is_mechanism_only": True,
        },
        "execution_lock_env": D5C_EXECUTION_LOCK_ENV,
        "execution_lock_value": D5C_EXECUTION_LOCK_VALUE,
        "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
        "authorization_path": AUTHORIZATION_RELATIVE_PATH,
        "output_dir": OUTPUT_RELATIVE_DIR,
        "required_owner_authorization_text": D5C_OWNER_AUTHORIZATION_TEXT,
        "design_and_entry_implementation_authorized": True,
        "no_model_verification_authorized": True,
        "execution_authorized_at_implementation": False,
        "future_exact_owner_authorization_required": True,
        "future_machine_authorization_required": True,
        "single_use_claim_required": True,
        "claim_consumed_before_model_config_or_weights_or_load": True,
        "result_observation_requires_same_authorization": True,
        "d4_rerun_authorized": False, "d4b_rerun_authorized": False,
        "d5c_rerun_authorized": False, "d5d_authorized": False,
        "d5e_authorized": False, "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "real_self_projection_authorized": False,
        "self_updater_authorized": False, "automatic_rerun_authorized": False,
        "failure_action": "persist_failure_and_stop_claim_consumed",
    }
    for field, expected in exact.items():
        _require_exact(spec.get(field), expected, field)
    return spec


def _authorization_schema(root: Path) -> dict[str, Any]:
    schema = _object(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH, "D5C authorization schema")
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        not isinstance(properties, dict) or set(properties) != AUTHORIZATION_FIELDS
        or not isinstance(required, list) or set(required) != AUTHORIZATION_FIELDS
        or schema.get("additionalProperties") is not False
    ):
        raise RuntimeError("D5C authorization schema changed")
    return schema


def _require_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D5C authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D5C authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D5C authorization timestamp must be UTC")


def _entry_static_digest(*, config_path: Path, project_root: Path) -> str:
    return build_d5c_entry_static_report(
        config_path=config_path, project_root=project_root
    )["report_digest_sha256"]


def build_d5c_real_authorization(
    *,
    config_path: str | Path,
    project_root: str | Path,
    authorization_text: str,
    git_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_exact_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    _read_spec(config_file)
    _authorization_schema(root)
    _require_exact(authorization_text, D5C_OWNER_AUTHORIZATION_TEXT, "owner authorization text")
    git = dict(git_metadata or _git_metadata(root))
    _require_clean_main(git)
    authorization = {
        "authorization_version": REPORT_VERSION,
        "stage": "Coupling-D5C_real_2_9b_noncore_mechanism_smoke",
        "scope": "one_fixed_42_call_mechanism_smoke_and_observation",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "entry_static_report_sha256": _entry_static_digest(
            config_path=config_file, project_root=root
        ),
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "mechanism_only": True,
        "d4_rerun_authorized": False, "d4b_rerun_authorized": False,
        "d5c_rerun_authorized": False, "d5d_authorized": False,
        "d5e_authorized": False, "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "real_self_projection_authorized": False,
        "self_updater_authorized": False, "automatic_rerun_authorized": False,
        "single_use": True,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_d5c_real_authorization(
    *,
    authorization_path: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    git: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    authorization = _object(authorization_path, "D5C machine authorization")
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D5C machine authorization fields changed")
    stored_digest = authorization.get("authorization_digest_sha256")
    digest_payload = {
        key: value for key, value in authorization.items()
        if key != "authorization_digest_sha256"
    }
    expected = build_d5c_real_authorization(
        config_path=config_file, project_root=root,
        authorization_text=D5C_OWNER_AUTHORIZATION_TEXT,
        git_metadata=git,
    )
    for field, value in expected.items():
        if field not in {"authorized_at_utc", "authorization_digest_sha256"}:
            _require_exact(authorization.get(field), value, f"authorization.{field}")
    _require_utc_timestamp(authorization.get("authorized_at_utc"))
    if not isinstance(stored_digest, str) or sha256_json(digest_payload) != stored_digest:
        raise PermissionError("D5C machine authorization digest is invalid")
    return authorization


def _create_claim(
    *, output_dir: Path, config_path: Path, authorization_path: Path,
    git: Mapping[str, str], entry_static_report_sha256: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D5C output directory is not empty; rerun refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d5c_single_use_execution_claim_consumed",
        "created_at_utc": _utc_now(), "single_use": True,
        "automatic_rerun_authorized": False,
        "git_commit": git["commit"], "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "entry_static_report_sha256": entry_static_report_sha256,
        "model_forward_call_count": 42, "mechanism_only": True,
    }
    return write_json_exclusive(output_dir / "execution_claim.json", claim)


def run_d5c_real_mechanism(
    *, config_path: str | Path, authorization_path: str | Path,
    project_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_exact_path(root, config_path, CONFIG_RELATIVE_PATH, "execution config")
    spec = _read_spec(config_file)
    if os.environ.get(D5C_EXECUTION_LOCK_ENV) != D5C_EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D5C execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D5C")
    authorization_file = _require_exact_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_exact_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d5c_real_authorization(
        authorization_path=authorization_file, config_path=config_file,
        project_root=root, git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _installed_source()
    _require_exact(installed_version, EXPECTED_RWKV_PACKAGE_VERSION, "installed RWKV version")
    _require_exact(source_digest, EXPECTED_RWKV_MODEL_SOURCE_SHA256, "installed RWKV source")
    claim_path = _create_claim(
        output_dir=destination, config_path=config_file,
        authorization_path=authorization_file, git=git,
        entry_static_report_sha256=authorization["entry_static_report_sha256"],
    )
    started = time.perf_counter()
    try:
        model_config_path = (root / spec["model_config_path"]).resolve()
        model_config_digest = sha256_file(model_config_path)
        _require_exact(model_config_digest, spec["model_config_sha256"], "model config digest")
        model_config = load_model_config(model_config_path, root, verify_files=True)
        _require_exact(model_config.model_id, spec["model_id"], "loaded model id")
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
        core = execute_d5c_mechanism_core(
            base_model=adapter.model, active_runtime=runtime, probe=probe, torch=torch
        )
        torch.cuda.synchronize()
        report = {
            "report_version": REPORT_VERSION, "created_at_utc": _utc_now(),
            "status": core["status"], "valid": core["valid"],
            "development_only": True, "git": git,
            "config": {"path": str(config_file), "sha256": sha256_file(config_file)},
            "authorization_digest_sha256": authorization["authorization_digest_sha256"],
            "execution_claim_sha256": claim_digest,
            "model": adapter.model_metadata(),
            "upstream": {"package": "rwkv", "version": installed_version,
                         "model_source_path": str(source_path),
                         "model_source_sha256": source_digest},
            "runtime_core": core,
            "interpretation": "mechanism_connectivity_only_no_self_effect_conclusion",
            "decision_effect": "d5d_design_review_candidate_only" if core["valid"] else "stop_without_rerun",
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "safety": {
                "real_2_9b_model_loaded": True, "real_2_9b_model_executed": True,
                "mechanism_only": True, "d4_rerun": False, "d4b_rerun": False,
                "d5c_rerun": False, "d5d_authorized": False, "d5e_authorized": False,
                "formal_test_set_used": False, "self_effect_conclusion_made": False,
                "real_self_projection_constructed": False, "self_updater_used": False,
                "automatic_rerun_authorized": False,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        write_json_exclusive(destination / "report.json", report)
        return report
    except BaseException as error:
        failure = {
            "report_version": REPORT_VERSION, "created_at_utc": _utc_now(),
            "status": "d5c_execution_attempt_failed_claim_consumed", "valid": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "d5c_rerun_authorized": False, "d5d_authorized": False,
            "d5e_authorized": False, "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        write_json_exclusive(destination / "failure.json", failure)
        raise


def _call_lines(source: str, function_name: str) -> dict[str, int]:
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
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
            lines.setdefault(name, node.lineno)
    return lines


def build_d5c_entry_static_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_exact_path(root, config_path, CONFIG_RELATIVE_PATH, "static config")
    spec = _read_spec(config_file)
    schema = _authorization_schema(root)
    module_source = (root / "src/psa/self_model/d5c_real_mechanism.py").read_text(encoding="utf-8")
    runtime_source = (root / "src/psa/self_model/d5c_mechanism_runtime.py").read_text(encoding="utf-8")
    calls = _call_lines(module_source, "run_d5c_real_mechanism")
    required_calls = (
        "validate_d5c_real_authorization", "_installed_source", "_create_claim",
        "sha256_file", "load_model_config", "load", "D5CSyntheticProbe",
        "RWKV7D5CActiveRuntime", "execute_d5c_mechanism_core",
    )
    source_digests = {path: sha256_file(root / path) for path in ENTRY_SOURCE_PATHS}
    properties = schema["properties"]
    checks = {
        "config_valid": bool(spec),
        "implementation_confirmation_recorded": IMPLEMENTATION_CONFIRMATION_TEXT in (
            root / "docs/self_model_v0_1_coupling_d5c_implementation_authorization.md"
        ).read_text(encoding="utf-8"),
        "two_noncore_fixtures_frozen": spec["fixtures"] == [dict(value) for value in FIXTURES],
        "latin_schedule_frozen": spec["routes"] == list(ROUTES)
        and spec["scored_rounds"] == [list(value) for value in SCORED_ROUNDS],
        "forty_two_calls_frozen": spec["counts"]["model_forward_calls_total"] == 42,
        "comparison_counts_frozen": all(
            spec["counts"][name] == 24 for name in (
                "within_route_comparisons_total", "control_cross_route_comparisons_total",
                "active_control_comparisons_total",
            )
        ),
        "probe_is_synthetic_not_self": spec["probe"]["real_self_projection"] is False
        and spec["probe"]["effect_layer_selected"] is False,
        "layer_rule_frozen": TARGET_LAYER_INDEX == 15
        and spec["probe"]["target_layer_rule"] == "floor((n_layer-1)/2)",
        "authorization_schema_exact": set(properties) == AUTHORIZATION_FIELDS
        and set(schema["required"]) == AUTHORIZATION_FIELDS
        and properties["authorization_text"]["const"] == D5C_OWNER_AUTHORIZATION_TEXT,
        "execution_not_authorized_at_implementation": spec["execution_authorized_at_implementation"] is False,
        "future_exact_owner_authorization_required": spec["future_exact_owner_authorization_required"] is True,
        "single_use_claim_required": spec["single_use_claim_required"] is True,
        "all_required_entry_calls_present": all(name in calls for name in required_calls),
        "claim_precedes_model_config_weight_verification_and_load": calls["_create_claim"]
        < calls["sha256_file"] < calls["load_model_config"] < calls["load"],
        "mechanism_core_runs_after_claim": calls["_create_claim"] < calls["execute_d5c_mechanism_core"],
        "temporary_binding_has_finally_cleanup": "finally:" in runtime_source
        and "_restore_bindings" in runtime_source
        and "_verify_restored_bindings" in runtime_source
        and "D5CCleanupTransactionError" in runtime_source
        and "instance_dict.pop(name, None)" not in runtime_source,
        "d4_d4b_d5c_reruns_closed": not spec["d4_rerun_authorized"]
        and not spec["d4b_rerun_authorized"] and not spec["d5c_rerun_authorized"],
        "d5d_d5e_closed": not spec["d5d_authorized"] and not spec["d5e_authorized"],
        "formal_and_self_claims_closed": not spec["formal_test_set_authorized"]
        and not spec["self_effect_conclusion_authorized"]
        and not spec["real_self_projection_authorized"]
        and not spec["self_updater_authorized"],
        "automatic_rerun_closed": not spec["automatic_rerun_authorized"],
        "source_inventory_complete": len(source_digests) == len(ENTRY_SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C entry static verification failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": "d5c_real_mechanism_entry_static_verified", "valid": True,
        "checks": checks, "source_digests": source_digests,
        "entry_call_lines": {name: calls[name] for name in required_calls},
        "future_exact_owner_authorization_text": D5C_OWNER_AUTHORIZATION_TEXT,
        "next_gate": "wait_for_exact_single_use_real_2_9b_d5c_authorization",
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False, "model_loaded": False, "model_executed": False,
            "execution_claim_created": False, "machine_authorization_created": False,
            "d5d_authorized": False, "d5e_authorized": False,
            "formal_test_set_used": False, "self_effect_conclusion_made": False,
            "real_self_projection_constructed": False, "self_updater_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
