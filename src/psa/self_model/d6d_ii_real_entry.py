from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.model.rwkv7 import RWKV7Adapter, load_model_config
from psa.self_model.d4b_real_off_equivalence import (
    _git_metadata,
    _require_clean_main,
    write_json_exclusive,
)
from psa.self_model.d6d_ii_joint_runtime import (
    D6DIIWrapperOwnedRuntime,
    execute_joint_pilot,
    execute_projection_training,
)
from psa.self_model.d6d_ii_manifests import (
    PILOT_FORWARD_CALLS,
    PILOT_MANIFEST_RELATIVE_PATH,
    TOTAL_FORWARD_CALLS,
    TRAINING_FORWARD_CALLS,
    TRAINING_MANIFEST_RELATIVE_PATH,
    build_manifest_report,
    load_pilot_manifest,
    load_training_manifest,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    TARGET_METHODS,
    build_instrumented_method_asts,
    compile_instrumented_methods,
    inspect_instrumented_source,
)


REPORT_VERSION = "0.1-coupling-d6d-ii-real-entry"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6d_ii_real_entry.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d6d_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "results/authorizations/self_model_v0_1_d6d_joint_v01.json"
)
OUTPUT_RELATIVE_DIR = "results/development/self_model_v0_1_d6d_joint_v01"
EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D6D_REAL_JOINT"
EXECUTION_LOCK_VALUE = "AUTHORIZED_D6D_REAL_2_9B_JOINT_TRAINING_PILOT_ONCE"
MODEL_CONFIG_DIGEST = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
IMPLEMENTATION_CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 Coupling-D6D-II installed source静态兼容、联合训练/试验"
    "manifest与单次真实入口的无模型实现；只允许探测并静态编译锁定installed source、冻结"
    "projection训练与pilot清单、实现新Schema/唯一目录/single-use claim入口，不访问权重、"
    "不加载或执行模型，不构造真实projection，也不授权D6D真实执行、D6E、正式测试集、"
    "Self效果结论、Self Updater、任何历史重跑或自动重跑。"
)
FUTURE_EXECUTION_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 Coupling-D6D 真实2.9B单一联合projection训练与非Core pilot一次"
    "（同一进程、同一wrapper、16次只读训练capture后冻结真实projection，再按12个fixture"
    "各1次OFF预条件和11条件调度执行144次pilot，共160次forward），并授权观察本次工程结果；"
    "不授权重跑D5C/P1/P2/D6C或D6D、自动重跑、D6E、正式测试集、Self效果结论、"
    "Self Updater、raw-original路线或任何拆分机制运行。"
)
D6D_DESIGN_REPORT_DIGEST = "3862a681d3658b645f141eb543b1076aa008a3be0ed805a31e8d3d022b081f75"
D6D_I_REPORT_DIGEST = "59f93fc9881578b1b1eed2aefaf41edcaf033a30937039e6b8ef1cd26e7e4625"
D6D_I_WRAPPER_DIGEST = "a7905cd798640ea62b10d1d5c7ac264d3129f3c06d172b95bc102c8024b4ae81"
D6D_I_PROJECTION_DIGEST = "64846ccc1ea54bca8a110b3b93f79ada4736c937c159077604596e2366124343"
INSTRUMENTER_DIGEST = "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"
CLASSIFICATION = (
    "d6d_ii_manifests_and_single_use_real_entry_no_model_verified_"
    "installed_source_static_probe_optional_until_remote_execution_not_authorized"
)
AUTHORIZATION_FIELDS = {
    "authorization_version", "stage", "scope", "authorized",
    "authorization_basis", "authorization_text", "authorized_at_utc",
    "git_commit", "config_sha256", "training_manifest_sha256",
    "pilot_manifest_sha256", "installed_source_static_report_sha256",
    "installed_source_sha256", "projection_training_authorized",
    "real_projection_construction_authorized", "weights_access_authorized",
    "model_load_authorized", "model_execution_authorized",
    "result_observation_authorized", "single_joint_experiment", "single_use",
    "raw_original_route_authorized", "historical_rerun_authorized",
    "d6d_rerun_authorized", "d6e_authorized", "formal_test_set_authorized",
    "self_effect_conclusion_authorized", "self_updater_authorized",
    "automatic_rerun_authorized", "authorization_digest_sha256",
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    TRAINING_MANIFEST_RELATIVE_PATH,
    PILOT_MANIFEST_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d6d_ii_real_entry.md",
    "scripts/run_self_model_v0_1_coupling_d6d_joint.py",
    "scripts/verify_self_model_v0_1_coupling_d6d_ii_real_entry.py",
    "src/psa/self_model/d6d_core_approach_design.py",
    "src/psa/self_model/d6d_wrapper_runtime.py",
    "src/psa/self_model/d6d_projection_artifact.py",
    "src/psa/self_model/d6d_ii_manifests.py",
    "src/psa/self_model/d6d_ii_joint_runtime.py",
    "src/psa/self_model/d6d_ii_real_entry.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d6d_ii_real_entry.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D6D-II {label} must be an object")
    return value


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise PermissionError(f"D6D-II {field} differs from the frozen value")


def _require_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if resolved != (root / relative).resolve():
        raise PermissionError(f"D6D-II {label} path is not frozen")
    return resolved


def read_spec(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    spec = _object(path, "config")
    exact = {
        "protocol_version": REPORT_VERSION,
        "stage": "Coupling-D6D-II_installed_source_static_compatibility_manifests_and_single_use_real_entry",
        "status": "no_model_entry_implementation_execution_not_authorized",
        "development_only": True,
        "implementation_confirmation_text": IMPLEMENTATION_CONFIRMATION_TEXT,
        "training_manifest_path": TRAINING_MANIFEST_RELATIVE_PATH,
        "pilot_manifest_path": PILOT_MANIFEST_RELATIVE_PATH,
        "model_config_path": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_sha256": MODEL_CONFIG_DIGEST,
        "model_id": "rwkv7-g1h-2.9b-20260710",
        "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
        "authorization_path": AUTHORIZATION_RELATIVE_PATH,
        "output_dir": OUTPUT_RELATIVE_DIR,
        "execution_lock_env": EXECUTION_LOCK_ENV,
        "execution_lock_value": EXECUTION_LOCK_VALUE,
        "future_execution_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "failure_action": "persist_failure_consume_claim_stop_without_rerun_or_route_splitting",
        "next_gate": "remote_installed_source_no_model_verification_then_separate_exact_d6d_execution_authorization",
    }
    for field, expected in exact.items():
        _require_exact(spec.get(field), expected, field)
    prerequisites = spec.get("frozen_prerequisites", {})
    _require_exact(
        prerequisites,
        {
            "d6d_design_report_sha256": D6D_DESIGN_REPORT_DIGEST,
            "d6d_i_remote_report_sha256": D6D_I_REPORT_DIGEST,
            "d6d_i_wrapper_source_sha256": D6D_I_WRAPPER_DIGEST,
            "d6d_i_projection_source_sha256": D6D_I_PROJECTION_DIGEST,
            "locked_instrumenter_sha256": INSTRUMENTER_DIGEST,
            "d6c_status": "failed_claim_consumed_no_rerun",
            "d6c_execution_claim_sha256": "82b94c33513da0137127ce44a85513c48c381d92b87e3a7c27916931821fe6a3",
        },
        "frozen prerequisites",
    )
    manifest_report = build_manifest_report(
        training_path=root / TRAINING_MANIFEST_RELATIVE_PATH,
        pilot_path=root / PILOT_MANIFEST_RELATIVE_PATH,
    )
    _require_exact(
        spec.get("training_manifest_sha256"),
        manifest_report["training_manifest_sha256"],
        "training manifest digest",
    )
    _require_exact(
        spec.get("pilot_manifest_sha256"),
        manifest_report["pilot_manifest_sha256"],
        "pilot manifest digest",
    )
    _require_exact(
        spec.get("manifest_load_order"),
        [
            "training_manifest_before_model_load",
            "projection_parameter_and_artifact_digests_persisted",
            "pilot_manifest_payload_loaded_after_artifact_freeze",
        ],
        "manifest load order",
    )
    static = spec.get("installed_source_static_contract", {})
    _require_exact(
        static,
        {
            "package": "rwkv",
            "version": EXPECTED_RWKV_PACKAGE_VERSION,
            "model_source_sha256": EXPECTED_RWKV_MODEL_SOURCE_SHA256,
            "target_class": "RWKV_x070",
            "target_methods": list(TARGET_METHODS),
            "rwkv_de_version": "unset",
            "compile_mode": "ast_transform_then_builtin_compile_without_exec",
            "rwkv_module_import_allowed": False,
            "torch_import_allowed": False,
            "installed_source_probe_authorized_this_round": True,
            "static_compile_authorized_this_round": True,
        },
        "installed source static contract",
    )
    _require_exact(
        spec.get("joint_run_counts"),
        {
            "projection_training_capture_calls": TRAINING_FORWARD_CALLS,
            "pilot_forward_calls": PILOT_FORWARD_CALLS,
            "total_model_forward_calls": TOTAL_FORWARD_CALLS,
            "pilot_fixture_count": 12,
            "pilot_conditions_per_fixture": 11,
            "pilot_off_precondition_per_fixture": 1,
        },
        "joint run counts",
    )
    authority = spec.get("authority", {})
    allowed = {
        "d6d_ii_implementation_authorized",
        "manifest_freeze_authorized",
        "installed_source_probe_authorized",
        "installed_source_static_compile_authorized",
        "real_entry_implementation_authorized",
        "authorization_schema_implementation_authorized",
        "single_use_claim_implementation_authorized",
    }
    _require_exact(set(authority), allowed | {
        "rwkv_import_authorized", "torch_import_authorized",
        "weights_access_authorized", "model_load_authorized",
        "model_execution_authorized", "real_projection_training_authorized",
        "real_projection_construction_authorized", "d6d_real_execution_authorized",
        "d6e_authorized", "formal_test_set_authorized",
        "self_effect_conclusion_authorized", "self_updater_authorized",
        "raw_original_route_authorized", "historical_rerun_authorized",
        "automatic_rerun_authorized",
    }, "authority fields")
    if not all(authority.get(name) is True for name in allowed):
        raise PermissionError("D6D-II implementation authority is incomplete")
    if not all(authority.get(name) is False for name in set(authority) - allowed):
        raise PermissionError("D6D-II model or later authority opened early")
    return spec


def _authorization_schema(root: Path) -> dict[str, Any]:
    schema = _object(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH, "authorization schema")
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        not isinstance(properties, dict)
        or set(properties) != AUTHORIZATION_FIELDS
        or not isinstance(required, list)
        or set(required) != AUTHORIZATION_FIELDS
        or schema.get("additionalProperties") is not False
    ):
        raise RuntimeError("D6D-II authorization schema changed")
    return schema


def _default_installed_source() -> tuple[str, Path, bytes, str]:
    installed_version = version("rwkv")
    source_path = Path(distribution("rwkv").locate_file("rwkv/model.py")).resolve()
    source_bytes = source_path.read_bytes()
    return (
        installed_version,
        source_path,
        source_bytes,
        hashlib.sha256(source_bytes).hexdigest(),
    )


def probe_installed_source_compatibility(
    provider: Callable[[], tuple[str, Path, bytes, str]] | None = None,
) -> dict[str, Any]:
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D6D-II")
    before = set(sys.modules)
    installed_version, source_path, source_bytes, source_digest = (
        provider or _default_installed_source
    )()
    calculated_source_digest = hashlib.sha256(source_bytes).hexdigest()
    _require_exact(source_digest, calculated_source_digest, "provider source digest")
    _require_exact(installed_version, EXPECTED_RWKV_PACKAGE_VERSION, "installed version")
    _require_exact(source_digest, EXPECTED_RWKV_MODEL_SOURCE_SHA256, "installed source digest")
    source = source_bytes.decode("utf-8")
    inspection = inspect_instrumented_source(source)
    methods, counts = build_instrumented_method_asts(source, rwkv_de_version=None)
    module = ast.Module(body=list(methods.values()), type_ignores=[])
    ast.fix_missing_locations(module)
    compiled = compile(module, "<psa-d6d-ii-static-installed-source>", "exec")
    imported = set(sys.modules) - before
    checks = {
        "package_version_exact": installed_version == EXPECTED_RWKV_PACKAGE_VERSION,
        "source_digest_exact": source_digest == EXPECTED_RWKV_MODEL_SOURCE_SHA256,
        "instrumentation_inspection_valid": inspection["valid"],
        "both_methods_transformed": set(methods) == set(TARGET_METHODS),
        "one_injection_per_method": counts == {"forward_one": 1, "forward_seq": 1},
        "builtin_compile_without_exec_succeeded": isinstance(compiled, type(compile("", "", "exec"))),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
        "probe_added_no_rwkv_or_torch_module": not any(
            name == "torch" or name.startswith("torch.")
            or name == "rwkv.model" or name.startswith("rwkv.model.")
            for name in imported
        ),
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D-II installed source static probe failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": "d6d_ii_installed_source_static_compatible_no_model",
        "valid": True,
        "checks": checks,
        "installed_package_version": installed_version,
        "installed_source_path": str(source_path),
        "installed_source_sha256": source_digest,
        "inspection": inspection,
        "compile_only": True,
        "exec_called": False,
        "model_loaded": False,
        "model_executed": False,
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report


def _entry_ast_audit() -> dict[str, int]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_d6d_real_joint"
    )
    wanted = {
        "_git_metadata", "validate_d6d_authorization",
        "probe_installed_source_compatibility", "_create_claim",
        "load_training_manifest", "load_model_config", "load",
        "compile_instrumented_methods", "D6DIIWrapperOwnedRuntime",
        "execute_projection_training", "write_json_exclusive", "load_pilot_manifest",
        "execute_joint_pilot",
    }
    lines = {}
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        name = call.func.id if isinstance(call.func, ast.Name) else (
            call.func.attr if isinstance(call.func, ast.Attribute) else None
        )
        if name in wanted:
            lines.setdefault(name, call.lineno)
    if set(lines) != wanted:
        missing = sorted(wanted - set(lines))
        raise RuntimeError("D6D-II entry call inventory changed: " + ", ".join(missing))
    return lines


def _execution_artifacts_absent(root: Path) -> dict[str, bool]:
    return {
        "machine_authorization_absent": not (root / AUTHORIZATION_RELATIVE_PATH).exists(),
        "execution_claim_absent": not (
            root / OUTPUT_RELATIVE_DIR / "execution_claim.json"
        ).exists(),
        "real_projection_artifact_absent": not (
            root / OUTPUT_RELATIVE_DIR / "projection_artifact.json"
        ).exists(),
    }


def build_d6d_ii_static_report(
    *, config_path: str | Path, project_root: str | Path,
    probe_installed_source: bool = False,
    installed_source_provider: Callable[[], tuple[str, Path, bytes, str]] | None = None,
    verify_execution_artifacts_absent: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file, root)
    _authorization_schema(root)
    manifest_report = build_manifest_report(
        training_path=root / TRAINING_MANIFEST_RELATIVE_PATH,
        pilot_path=root / PILOT_MANIFEST_RELATIVE_PATH,
    )
    lines = _entry_ast_audit()
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    artifacts = _execution_artifacts_absent(root)
    installed_report = (
        probe_installed_source_compatibility(installed_source_provider)
        if probe_installed_source else {
            "status": "remote_installed_source_probe_pending",
            "valid": None,
            "installed_source_sha256": None,
            "model_loaded": False,
            "model_executed": False,
        }
    )
    checks = {
        "config_valid": True,
        "implementation_confirmation_recorded": spec["implementation_confirmation_text"]
        == IMPLEMENTATION_CONFIRMATION_TEXT,
        "d6d_design_and_d6d_i_reports_frozen": spec["frozen_prerequisites"]
        ["d6d_design_report_sha256"] == D6D_DESIGN_REPORT_DIGEST
        and spec["frozen_prerequisites"]["d6d_i_remote_report_sha256"]
        == D6D_I_REPORT_DIGEST,
        "d6d_i_wrapper_frozen": source_digests[
            "src/psa/self_model/d6d_wrapper_runtime.py"
        ] == D6D_I_WRAPPER_DIGEST,
        "d6d_i_projection_frozen": source_digests[
            "src/psa/self_model/d6d_projection_artifact.py"
        ] == D6D_I_PROJECTION_DIGEST,
        "instrumenter_frozen": source_digests[
            "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
        ] == INSTRUMENTER_DIGEST,
        "training_and_pilot_manifests_valid": manifest_report["valid"],
        "manifest_digests_bound_in_config": spec["training_manifest_sha256"]
        == manifest_report["training_manifest_sha256"]
        and spec["pilot_manifest_sha256"] == manifest_report["pilot_manifest_sha256"],
        "sixteen_plus_one_hundred_forty_four_calls_frozen": all(
            manifest_report["counts"][name] == expected
            for name, expected in {
                "training_forward_calls": 16,
                "pilot_forward_calls": 144,
                "total_forward_calls": 160,
            }.items()
        ),
        "authorization_schema_exact": set(_authorization_schema(root)["properties"])
        == AUTHORIZATION_FIELDS,
        "unique_paths_frozen": spec["authorization_path"] == AUTHORIZATION_RELATIVE_PATH
        and spec["output_dir"] == OUTPUT_RELATIVE_DIR,
        "future_exact_authorization_frozen": spec["future_execution_authorization_text"]
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "authorization_precedes_installed_probe": lines["validate_d6d_authorization"]
        < lines["probe_installed_source_compatibility"],
        "claim_precedes_weight_verification_and_load": lines["_create_claim"]
        < lines["load_model_config"] < lines["load"],
        "one_wrapper_after_model_load": lines["load"]
        < lines["compile_instrumented_methods"] < lines["D6DIIWrapperOwnedRuntime"],
        "training_precedes_artifact_persist": lines["execute_projection_training"]
        < lines["write_json_exclusive"],
        "pilot_payload_loads_after_artifact_persist": lines["write_json_exclusive"]
        < lines["load_pilot_manifest"] < lines["execute_joint_pilot"],
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
        "machine_authorization_not_created": artifacts["machine_authorization_absent"]
        if verify_execution_artifacts_absent else True,
        "execution_claim_not_created": artifacts["execution_claim_absent"]
        if verify_execution_artifacts_absent else True,
        "real_projection_not_constructed": artifacts["real_projection_artifact_absent"]
        if verify_execution_artifacts_absent else True,
        "installed_source_probe_matches_requested_mode": (
            installed_report["valid"] is True if probe_installed_source
            else installed_report["valid"] is None
        ),
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D-II static entry verification failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": (
            "d6d_ii_installed_source_and_real_entry_no_model_verified"
            if probe_installed_source
            else "d6d_ii_real_entry_static_verified_installed_source_probe_pending"
        ),
        "valid": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "entry_call_lines": lines,
        "manifest_report": manifest_report,
        "installed_source_report": installed_report,
        "source_digests": source_digests,
        "future_exact_owner_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "next_gate": (
            "separate_exact_d6d_real_execution_authorization"
            if probe_installed_source
            else "remote_installed_source_no_model_verification"
        ),
        "safety": {
            "installed_source_probed": bool(probe_installed_source),
            "installed_source_static_compiled_without_exec": bool(probe_installed_source),
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "machine_authorization_created": False,
            "execution_claim_created": False,
            "real_projection_trained": False,
            "real_projection_constructed": False,
            "pilot_run": False,
            "historical_rerun": False,
            "d6d_execution_authorized": False,
            "d6e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report


def _require_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D6D authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D6D authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D6D authorization timestamp must be UTC")


def build_d6d_authorization(
    *, config_path: str | Path, project_root: str | Path,
    authorization_text: str, git_metadata: Mapping[str, str] | None = None,
    installed_source_provider: Callable[[], tuple[str, Path, bytes, str]] | None = None,
    verify_execution_artifacts_absent: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file, root)
    _authorization_schema(root)
    _require_exact(authorization_text, FUTURE_EXECUTION_AUTHORIZATION_TEXT, "authorization text")
    git = dict(git_metadata or _git_metadata(root))
    _require_clean_main(git)
    static = build_d6d_ii_static_report(
        config_path=config_file,
        project_root=root,
        probe_installed_source=True,
        installed_source_provider=installed_source_provider,
        verify_execution_artifacts_absent=verify_execution_artifacts_absent,
    )
    authorization = {
        "authorization_version": REPORT_VERSION,
        "stage": "Coupling-D6D_real_2_9b_single_joint_projection_training_and_noncore_pilot",
        "scope": "one_process_one_wrapper_sixteen_training_plus_one_hundred_forty_four_pilot_calls",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_future_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "training_manifest_sha256": spec["training_manifest_sha256"],
        "pilot_manifest_sha256": spec["pilot_manifest_sha256"],
        "installed_source_static_report_sha256": static["report_digest_sha256"],
        "installed_source_sha256": static["installed_source_report"]["installed_source_sha256"],
        "projection_training_authorized": True,
        "real_projection_construction_authorized": True,
        "weights_access_authorized": True,
        "model_load_authorized": True,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "single_joint_experiment": True,
        "single_use": True,
        "raw_original_route_authorized": False,
        "historical_rerun_authorized": False,
        "d6d_rerun_authorized": False,
        "d6e_authorized": False,
        "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "self_updater_authorized": False,
        "automatic_rerun_authorized": False,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_d6d_authorization(
    *, authorization_path: str | Path, config_path: str | Path,
    project_root: str | Path, git: Mapping[str, str],
    installed_source_provider: Callable[[], tuple[str, Path, bytes, str]] | None = None,
) -> dict[str, Any]:
    authorization = _object(authorization_path, "machine authorization")
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D6D machine authorization fields changed")
    expected = build_d6d_authorization(
        config_path=config_path,
        project_root=project_root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        git_metadata=git,
        installed_source_provider=installed_source_provider,
        verify_execution_artifacts_absent=False,
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
        raise PermissionError("D6D machine authorization digest is invalid")
    return authorization


def _create_claim(
    *, output_dir: Path, config_path: Path, authorization_path: Path,
    git: Mapping[str, str], installed_source_static_report_sha256: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D6D output directory is not empty; reuse refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d6d_single_use_joint_execution_claim_consumed",
        "created_at_utc": _utc_now(),
        "single_use": True,
        "automatic_rerun_authorized": False,
        "historical_rerun_authorized": False,
        "d6d_rerun_authorized": False,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "installed_source_static_report_sha256": installed_source_static_report_sha256,
        "training_forward_calls": TRAINING_FORWARD_CALLS,
        "pilot_forward_calls": PILOT_FORWARD_CALLS,
        "total_forward_calls": TOTAL_FORWARD_CALLS,
        "single_joint_experiment": True,
    }
    return write_json_exclusive(output_dir / "execution_claim.json", claim)


def run_d6d_real_joint(
    *, config_path: str | Path, authorization_path: str | Path,
    project_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file, root)
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D6D execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D6D")
    authorization_file = _require_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d6d_authorization(
        authorization_path=authorization_file,
        config_path=config_file,
        project_root=root,
        git=git,
    )
    installed = probe_installed_source_compatibility()
    claim_path = _create_claim(
        output_dir=destination,
        config_path=config_file,
        authorization_path=authorization_file,
        git=git,
        installed_source_static_report_sha256=installed["report_digest_sha256"],
    )
    started = time.perf_counter()
    try:
        training_manifest = load_training_manifest(root / TRAINING_MANIFEST_RELATIVE_PATH)
        model_config_path = root / spec["model_config_path"]
        _require_exact(sha256_file(model_config_path), MODEL_CONFIG_DIGEST, "model config digest")
        model_config = load_model_config(model_config_path, root, verify_files=True)
        _require_exact(model_config.model_id, spec["model_id"], "model id")
        adapter = RWKV7Adapter.load(model_config)
        torch = adapter.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        source_bytes = Path(installed["installed_source_path"]).read_bytes()
        methods, counts = compile_instrumented_methods(
            upstream_source=source_bytes.decode("utf-8"),
            upstream_globals=vars(sys.modules["rwkv.model"]),
            rwkv_de_version=os.environ.get("RWKV_DE_VERSION"),
        )
        runtime = D6DIIWrapperOwnedRuntime(
            base_model=adapter.model,
            compiled_methods=methods,
            injection_counts=counts,
        )
        claim_digest = sha256_file(claim_path)
        authorization_digest = sha256_file(authorization_file)
        training = execute_projection_training(
            adapter=adapter,
            runtime=runtime,
            torch=torch,
            training_manifest=training_manifest,
            training_manifest_sha256=spec["training_manifest_sha256"],
            pilot_manifest_sha256=spec["pilot_manifest_sha256"],
        )
        artifact_path = destination / "projection_artifact.json"
        write_json_exclusive(artifact_path, training["artifact"])
        pilot_manifest = load_pilot_manifest(root / PILOT_MANIFEST_RELATIVE_PATH)
        pilot = execute_joint_pilot(
            adapter=adapter,
            runtime=runtime,
            torch=torch,
            artifact=training["artifact"],
            pilot_manifest=pilot_manifest,
            seed_material=f"{claim_digest}|{authorization_digest}",
        )
        torch.cuda.synchronize()
        report = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d6d_real_joint_training_and_noncore_pilot_complete",
            "valid": training["valid"] and pilot["valid"],
            "development_only": True,
            "git": git,
            "config_sha256": sha256_file(config_file),
            "authorization_digest_sha256": authorization["authorization_digest_sha256"],
            "execution_claim_sha256": claim_digest,
            "installed_source_static_report_sha256": installed["report_digest_sha256"],
            "model": adapter.model_metadata(),
            "training": {key: value for key, value in training.items() if key != "artifact"},
            "projection_artifact_sha256": sha256_file(artifact_path),
            "pilot": pilot,
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "classification": "noncore_engineering_pilot_not_self_effect_conclusion",
            "self_effect_conclusion": False,
            "safety": {
                "single_joint_experiment": True,
                "raw_original_route_used": False,
                "historical_rerun": False,
                "d6d_rerun": False,
                "d6e_authorized": False,
                "formal_test_set_used": False,
                "self_effect_conclusion_made": False,
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
            "status": "d6d_real_joint_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "d6d_rerun_authorized": False,
            "d6e_authorized": False,
            "self_effect_conclusion": False,
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        write_json_exclusive(destination / "failure.json", failure)
        raise
