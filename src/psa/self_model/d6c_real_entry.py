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
from psa.self_model.d5c_mechanism_runtime import D5CSyntheticProbe, FIXTURES
from psa.self_model.d6c_persistent_mechanism import (
    ACTIVE_CALLBACK_CALLS_TOTAL,
    ACTIVE_FORWARD_CALLS_TOTAL,
    ACTIVE_PROBE_APPLICATIONS_TOTAL,
    HIDDEN_DIMENSION,
    LATIN_ROUNDS,
    MODEL_FORWARD_CALLS_PER_FIXTURE,
    MODEL_FORWARD_CALLS_TOTAL,
    N_LAYER,
    ROUTES,
    STATE_COMPONENTS,
    TARGET_LAYER_INDEX,
    RWKV7D6CPersistentRuntime,
    execute_d6c_mechanism_core,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)


REPORT_VERSION = "0.1-coupling-d6c-real-persistent-mechanism"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6c_real_persistent_mechanism.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d6c_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "results/authorizations/self_model_v0_1_d6c_real_persistent_mechanism_v01.json"
)
OUTPUT_RELATIVE_DIR = (
    "results/development/self_model_v0_1_d6c_real_persistent_mechanism_v01"
)
EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D6C_REAL_PERSISTENT_MECHANISM"
EXECUTION_LOCK_VALUE = "AUTHORIZED_D6C_REAL_2_9B_PERSISTENT_MECHANISM_ONCE"
IMPLEMENTATION_CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 Coupling-D6C 真实2.9B persistent-instrumented非Core机制验证"
    "设计与无模型安全入口实现；冻结两个非Core形状、每形状1次OFF预条件和4轮OFF/zero/active"
    "拉丁调度（共26次调用）、固定synthetic probe与层访问计数，并实现新授权Schema、唯一输出目录、"
    "single-use claim及未来逐字执行授权门；本轮不探测installed source、不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型，也不授权D6C真实执行、D6D/D6E、D5C/P1/P2重跑、"
    "raw-original路线、真实层选择、真实Self projection、Self效果实验、Self Updater或自动重跑。"
)
FUTURE_EXECUTION_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 Coupling-D6C 真实2.9B persistent-instrumented非Core机制验证一次"
    "（冻结两个非Core形状、每形状1次OFF预条件和4轮OFF/zero/active拉丁调度，共26次调用；"
    "固定synthetic probe与层访问计数），并授权观察本次机制结果；不授权重跑D5C/P1/P2或D6C、"
    "自动重跑、D6D/D6E、raw-original路线、正式测试集、真实层选择、真实Self projection、"
    "Self效果结论或Self Updater。"
)
D6B_REMOTE_REPORT_DIGEST = "e51c1248d3f51f81273d8d3e088418f68a3cb7b5198abafe266a4d499a7d9007"
D6B_SOURCE_DIGEST = "4e051ead7d04767b2614266be98562fe826638b13bf5f363a42844a7d2d42b91"
INSTRUMENTER_DIGEST = "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"
D5_RUNTIME_DIGEST = "e4ae5c5bee74a85a4dea8a9b8eb16e3b6e19ef6b375020ffc849a09cbd7bbc32"
MODEL_CONFIG_DIGEST = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
CLASSIFICATION = (
    "d6c_real_persistent_mechanism_design_and_single_use_entry_static_verified_"
    "execution_not_authorized"
)
AUTHORIZATION_FIELDS = {
    "authorization_version", "stage", "scope", "authorized",
    "authorization_basis", "authorization_text", "authorized_at_utc",
    "git_commit", "config_sha256", "entry_static_report_sha256",
    "installed_source_probe_authorized", "model_execution_authorized",
    "result_observation_authorized", "mechanism_only",
    "raw_original_route_authorized", "d5c_rerun_authorized",
    "p1_rerun_authorized", "p2_authorized", "d6c_rerun_authorized",
    "d6d_authorized", "d6e_authorized", "formal_test_set_authorized",
    "real_layer_selection_authorized", "real_self_projection_authorized",
    "self_effect_conclusion_authorized", "self_updater_authorized",
    "automatic_rerun_authorized", "single_use", "authorization_digest_sha256",
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d6c_real_persistent_mechanism.md",
    "scripts/run_self_model_v0_1_coupling_d6c_real_persistent_mechanism.py",
    "scripts/verify_self_model_v0_1_coupling_d6c_real_entry.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/d6b_persistent_ast.py",
    "src/psa/self_model/d6c_persistent_mechanism.py",
    "src/psa/self_model/d6c_real_entry.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d6c_real_entry.py",
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
        raise PermissionError(f"D6C {field} differs from the frozen value")


def _require_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if resolved != (root / relative).resolve():
        raise PermissionError(f"D6C {label} path is not frozen")
    return resolved


def read_spec(path: str | Path) -> dict[str, Any]:
    spec = _object(path, "D6C config")
    exact = {
        "protocol_version": REPORT_VERSION,
        "stage": "Coupling-D6C_real_2_9b_persistent_instrumented_noncore_mechanism_entry",
        "status": "design_and_no_model_entry_implemented_execution_not_authorized",
        "development_only": True,
        "implementation_confirmation_text": IMPLEMENTATION_CONFIRMATION_TEXT,
        "frozen_prerequisites": {
            "d6b_remote_report_sha256": D6B_REMOTE_REPORT_DIGEST,
            "d6b_source_sha256": D6B_SOURCE_DIGEST,
            "instrumenter_source_sha256": INSTRUMENTER_DIGEST,
            "historical_d5_runtime_sha256": D5_RUNTIME_DIGEST,
            "d5_line_status": "stopped_no_rerun",
            "p2_allowed": False,
        },
        "model_config_path": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_sha256": MODEL_CONFIG_DIGEST,
        "model_id": "rwkv7-g1h-2.9b-20260710",
        "upstream": {
            "package": "rwkv", "version": EXPECTED_RWKV_PACKAGE_VERSION,
            "model_source_sha256": EXPECTED_RWKV_MODEL_SOURCE_SHA256,
            "rwkv_de_version": "unset",
        },
        "fixtures": [dict(value) for value in FIXTURES],
        "routes": list(ROUTES),
        "off_precondition_calls_per_fixture": 1,
        "latin_rounds": [list(value) for value in LATIN_ROUNDS],
        "counts": {
            "fixture_count": 2,
            "calls_per_fixture": MODEL_FORWARD_CALLS_PER_FIXTURE,
            "model_forward_calls_total": MODEL_FORWARD_CALLS_TOTAL,
            "off_precondition_calls_total": 2,
            "scored_rounds_per_fixture": 4,
            "scored_calls_per_fixture": 12,
            "active_forward_calls_total": ACTIVE_FORWARD_CALLS_TOTAL,
            "active_callback_calls_total": ACTIVE_CALLBACK_CALLS_TOTAL,
            "active_probe_applications_total": ACTIVE_PROBE_APPLICATIONS_TOTAL,
            "control_comparisons_total": 8,
            "active_comparisons_total": 16,
            "within_route_comparisons_total": 18,
        },
        "probe": {
            "kind": "deterministic_synthetic_unit_rms_vector_not_self_representation",
            "hidden_dimension": HIDDEN_DIMENSION,
            "n_layer": N_LAYER,
            "target_layer_index_zero_based": TARGET_LAYER_INDEX,
            "target_layer_rule": "frozen_midpoint_mechanism_probe_not_effect_selected",
            "phase": "post_ffn_residual", "target_residual_rms_ratio": 0.01,
            "gate": 1.0, "projection_trained": False,
            "real_layer_selection_performed": False, "real_self_projection": False,
        },
        "execution_lock_env": EXECUTION_LOCK_ENV,
        "execution_lock_value": EXECUTION_LOCK_VALUE,
        "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
        "authorization_path": AUTHORIZATION_RELATIVE_PATH,
        "output_dir": OUTPUT_RELATIVE_DIR,
        "future_execution_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "design_and_entry_implementation_authorized": True,
        "no_model_verification_authorized": True,
        "installed_source_probe_authorized_at_implementation": False,
        "execution_authorized_at_implementation": False,
        "future_exact_owner_authorization_required": True,
        "future_machine_authorization_required": True,
        "single_use_claim_required": True,
        "claim_consumed_before_model_config_weights_or_load": True,
        "unique_output_required": True,
        "raw_original_route_authorized": False,
        "d5c_rerun_authorized": False, "p1_rerun_authorized": False,
        "p2_authorized": False, "d6c_rerun_authorized": False,
        "d6d_authorized": False, "d6e_authorized": False,
        "formal_test_set_authorized": False,
        "real_layer_selection_authorized": False,
        "real_self_projection_authorized": False,
        "self_effect_conclusion_authorized": False,
        "self_updater_authorized": False,
        "automatic_rerun_authorized": False,
        "failure_action": "persist_failure_and_stop_d6c_claim_consumed",
    }
    for field, expected in exact.items():
        _require_exact(spec.get(field), expected, field)
    acceptance = spec.get("acceptance")
    expected_acceptance = {
        "all_outputs_finite": True, "within_route_all_exact": True,
        "off_zero_all_exact": True, "active_differs_from_off_and_zero": True,
        "persistent_binding_identity_stable": True,
        "runtime_model_attribute_mutation_allowed": False,
        "callback_counts_exact": True, "mechanism_only": True,
    }
    _require_exact(acceptance, expected_acceptance, "acceptance")
    return spec


def _authorization_schema(root: Path) -> dict[str, Any]:
    schema = _object(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH, "D6C authorization schema")
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        not isinstance(properties, dict) or set(properties) != AUTHORIZATION_FIELDS
        or not isinstance(required, list) or set(required) != AUTHORIZATION_FIELDS
        or schema.get("additionalProperties") is not False
    ):
        raise RuntimeError("D6C authorization schema changed")
    return schema


def _require_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D6C authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D6C authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D6C authorization timestamp must be UTC")


def build_d6c_authorization(
    *, config_path: str | Path, project_root: str | Path,
    authorization_text: str, git_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    read_spec(config_file)
    _authorization_schema(root)
    _require_exact(
        authorization_text, FUTURE_EXECUTION_AUTHORIZATION_TEXT, "authorization text"
    )
    git = dict(git_metadata or _git_metadata(root))
    _require_clean_main(git)
    authorization = {
        "authorization_version": REPORT_VERSION,
        "stage": "Coupling-D6C_real_2_9b_persistent_instrumented_noncore_mechanism",
        "scope": "one_fixed_26_call_persistent_mechanism_validation_and_observation",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_future_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "entry_static_report_sha256": build_d6c_entry_static_report(
            config_path=config_file, project_root=root
        )["report_digest_sha256"],
        "installed_source_probe_authorized": True,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "mechanism_only": True,
        "raw_original_route_authorized": False,
        "d5c_rerun_authorized": False, "p1_rerun_authorized": False,
        "p2_authorized": False, "d6c_rerun_authorized": False,
        "d6d_authorized": False, "d6e_authorized": False,
        "formal_test_set_authorized": False,
        "real_layer_selection_authorized": False,
        "real_self_projection_authorized": False,
        "self_effect_conclusion_authorized": False,
        "self_updater_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_d6c_authorization(
    *, authorization_path: str | Path, config_path: str | Path,
    project_root: str | Path, git: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    authorization = _object(authorization_path, "D6C machine authorization")
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D6C machine authorization fields changed")
    expected = build_d6c_authorization(
        config_path=config_path, project_root=root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT, git_metadata=git,
    )
    for field, value in expected.items():
        if field not in {"authorized_at_utc", "authorization_digest_sha256"}:
            _require_exact(authorization.get(field), value, f"authorization.{field}")
    _require_utc_timestamp(authorization.get("authorized_at_utc"))
    stored = authorization.get("authorization_digest_sha256")
    payload = {
        key: value for key, value in authorization.items()
        if key != "authorization_digest_sha256"
    }
    if not isinstance(stored, str) or sha256_json(payload) != stored:
        raise PermissionError("D6C machine authorization digest is invalid")
    return authorization


def _create_claim(
    *, output_dir: Path, config_path: Path, authorization_path: Path,
    git: Mapping[str, str], entry_static_report_sha256: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D6C output directory is not empty; reuse refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d6c_single_use_execution_claim_consumed",
        "created_at_utc": _utc_now(), "single_use": True,
        "automatic_rerun_authorized": False,
        "d5c_p1_p2_rerun_authorized": False,
        "d6c_rerun_authorized": False,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "entry_static_report_sha256": entry_static_report_sha256,
        "model_forward_call_count": MODEL_FORWARD_CALLS_TOTAL,
        "mechanism_only": True,
    }
    return write_json_exclusive(output_dir / "execution_claim.json", claim)


def run_d6c_real_persistent_mechanism(
    *, config_path: str | Path, authorization_path: str | Path,
    project_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D6C execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D6C")
    authorization_file = _require_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d6c_authorization(
        authorization_path=authorization_file, config_path=config_file,
        project_root=root, git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _installed_source()
    _require_exact(installed_version, EXPECTED_RWKV_PACKAGE_VERSION, "installed version")
    _require_exact(source_digest, EXPECTED_RWKV_MODEL_SOURCE_SHA256, "installed source")
    claim_path = _create_claim(
        output_dir=destination, config_path=config_file,
        authorization_path=authorization_file, git=git,
        entry_static_report_sha256=authorization["entry_static_report_sha256"],
    )
    started = time.perf_counter()
    try:
        model_config_path = (root / spec["model_config_path"]).resolve()
        _require_exact(sha256_file(model_config_path), MODEL_CONFIG_DIGEST, "model config")
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
        runtime = RWKV7D6CPersistentRuntime(
            base_model=adapter.model, upstream_source_bytes=source_bytes,
            upstream_globals=vars(sys.modules["rwkv.model"]),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
            execution_claim_sha256=claim_digest,
            machine_authorization_sha256=authorization_digest,
        )
        core = execute_d6c_mechanism_core(runtime=runtime, probe=probe, torch=torch)
        torch.cuda.synchronize()
        report = {
            "report_version": REPORT_VERSION, "created_at_utc": _utc_now(),
            "status": core["status"], "valid": core["valid"],
            "development_only": True, "git": git,
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
            "interpretation": "persistent_mechanism_connectivity_only_no_self_effect_conclusion",
            "decision_effect": "d6d_design_review_candidate_only" if core["valid"] else "stop_without_rerun",
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "safety": {
                "mechanism_only": True, "raw_original_route_used": False,
                "d5c_p1_p2_rerun": False, "d6c_rerun": False,
                "d6d_authorized": False, "d6e_authorized": False,
                "formal_test_set_used": False, "real_layer_selected": False,
                "real_self_projection_constructed": False,
                "self_effect_conclusion_made": False,
                "self_updater_used": False, "automatic_rerun_authorized": False,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        write_json_exclusive(destination / "report.json", report)
        return report
    except BaseException as error:
        failure = {
            "report_version": REPORT_VERSION, "created_at_utc": _utc_now(),
            "status": "d6c_execution_attempt_failed_claim_consumed", "valid": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "d6c_rerun_authorized": False, "d6d_authorized": False,
            "d6e_authorized": False, "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        write_json_exclusive(destination / "failure.json", failure)
        raise


def _entry_ast_audit() -> dict[str, int]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_d6c_real_persistent_mechanism"
    )
    wanted = {
        "_git_metadata", "validate_d6c_authorization", "_installed_source",
        "_create_claim", "load_model_config", "load",
        "execute_d6c_mechanism_core",
    }
    lines: dict[str, int] = {}
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        name = None
        if isinstance(call.func, ast.Name):
            name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            name = call.func.attr
        if name in wanted:
            lines.setdefault(name, call.lineno)
    if set(lines) != wanted:
        missing = sorted(wanted - set(lines))
        raise RuntimeError("D6C entry call inventory changed: " + ", ".join(missing))
    return lines


def _runtime_forward_mutation_audit(root: Path) -> dict[str, Any]:
    path = root / "src/psa/self_model/d6c_persistent_mechanism.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RWKV7D6CPersistentRuntime"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "forward"
    )
    mutations = [
        call.func.id
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"setattr", "delattr"}
    ]
    return {
        "forward_line": function.lineno,
        "model_attribute_mutation_calls": mutations,
        "model_attribute_mutation_call_count": len(mutations),
    }


def build_d6c_entry_static_report(
    *, config_path: str | Path, project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    _authorization_schema(root)
    lines = _entry_ast_audit()
    runtime_audit = _runtime_forward_mutation_audit(root)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    authorization_path = root / AUTHORIZATION_RELATIVE_PATH
    claim_path = root / OUTPUT_RELATIVE_DIR / "execution_claim.json"
    checks = {
        "config_valid": True,
        "implementation_confirmation_recorded": spec["implementation_confirmation_text"]
        == IMPLEMENTATION_CONFIRMATION_TEXT,
        "two_fixtures_and_twenty_six_calls_frozen": len(spec["fixtures"]) == 2
        and spec["counts"]["model_forward_calls_total"] == 26,
        "off_precondition_and_four_rounds_frozen": spec["off_precondition_calls_per_fixture"] == 1
        and len(spec["latin_rounds"]) == 4,
        "only_three_persistent_routes": spec["routes"] == list(ROUTES)
        and all("original" not in route for route in spec["routes"]),
        "probe_counts_frozen": spec["counts"]["active_callback_calls_total"] == 256
        and spec["counts"]["active_probe_applications_total"] == 8,
        "persistent_shape_frozen": spec["probe"]["n_layer"] == 32
        and spec["probe"]["hidden_dimension"] == 2560
        and STATE_COMPONENTS == 96,
        "authorization_schema_exact": set(_authorization_schema(root)["properties"])
        == AUTHORIZATION_FIELDS,
        "unique_paths_frozen": spec["authorization_path"] == AUTHORIZATION_RELATIVE_PATH
        and spec["output_dir"] == OUTPUT_RELATIVE_DIR,
        "future_exact_authorization_required": spec["future_execution_authorization_text"]
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "execution_not_authorized_at_implementation": spec["execution_authorized_at_implementation"]
        is False,
        "installed_source_probe_not_authorized_at_implementation": spec[
            "installed_source_probe_authorized_at_implementation"
        ] is False,
        "authorization_precedes_installed_source": lines["validate_d6c_authorization"]
        < lines["_installed_source"],
        "claim_precedes_model_config_and_load": lines["_create_claim"]
        < lines["load_model_config"] < lines["load"],
        "core_runs_after_model_load": lines["load"]
        < lines["execute_d6c_mechanism_core"],
        "runtime_forward_has_no_model_attribute_mutation": runtime_audit[
            "model_attribute_mutation_call_count"
        ] == 0,
        "d6b_source_frozen": source_digests[
            "src/psa/self_model/d6b_persistent_ast.py"
        ] == D6B_SOURCE_DIGEST,
        "instrumenter_frozen": source_digests[
            "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
        ] == INSTRUMENTER_DIGEST,
        "historical_d5_runtime_frozen": source_digests[
            "src/psa/self_model/d5c_mechanism_runtime.py"
        ] == D5_RUNTIME_DIGEST,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
        "machine_authorization_not_created": not authorization_path.exists(),
        "execution_claim_not_created": not claim_path.exists(),
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6C static entry verification failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": "d6c_real_persistent_mechanism_entry_static_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "entry_call_lines": lines,
        "runtime_forward_ast_audit": runtime_audit,
        "future_exact_owner_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "source_digests": source_digests,
        "next_gate": "remote_no_model_static_verification_then_separate_d6c_execution_authorization",
        "safety": {
            "installed_source_probed": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False, "model_loaded": False,
            "model_executed": False, "machine_authorization_created": False,
            "execution_claim_created": False, "raw_original_route_used": False,
            "d5c_p1_p2_rerun": False, "d6c_rerun": False,
            "d6d_authorized": False, "d6e_authorized": False,
            "formal_test_set_used": False, "real_layer_selected": False,
            "real_self_projection_constructed": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False, "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
