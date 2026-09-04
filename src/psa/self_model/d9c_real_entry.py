"""D9-C single-use real entry with a no-model static verification surface."""

from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence, TextIO

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.self_model.d9a_within_wrapper_causal_isolation import evaluate_candidate
from psa.self_model.d9c_projection_contract import (
    ARTIFACT_VERSION,
    CALIBRATION_COMMITMENT,
    CALIBRATION_RELATIVE_PATH,
    CALIBRATION_SHA256,
    CalibrationCapture,
    CONTRACT_RELATIVE_PATH as PROJECTION_CONTRACT_RELATIVE_PATH,
    HELDOUT_COMMITMENT,
    HELDOUT_RELATIVE_PATH,
    HELDOUT_SHA256,
    SCHEDULE_COMMITMENT,
    SCHEDULE_RELATIVE_PATH,
    SCHEDULE_SHA256,
    audit_frozen_projection_artifact,
    build_frozen_projection_artifact,
    project_condition,
    verify_projection_contract_files,
)


ENTRY_VERSION = "0.1-self-model-d9c-projection-entry"
CONFIG_RELATIVE_PATH = "configs/development/self_model_v0_1_d9c_projection_entry.json"
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d9_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = "results/authorizations/self_model_v0_1_d9_real_v01.json"
OUTPUT_RELATIVE_DIR = "results/development/self_model_v0_1_d9_real_v01"
DESIGN_RELATIVE_PATH = (
    "configs/preregistration/"
    "self_model_v0_1_d9a_within_wrapper_causal_isolation.draft.json"
)
D9B_CONTRACT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9b_manifest_endpoint_contract.json"
)
ENDPOINT_RELATIVE_PATH = "configs/development/self_model_v0_1_d9_causal_endpoint.json"
MODEL_CONFIG_RELATIVE_PATH = "configs/models/rwkv7_g1h_2.9b.candidate.json"
CONFIG_SHA256 = "f73881c3aedbb52057057d8984764d891577033612b6aafe2fdaf28401d77288"
AUTHORIZATION_SCHEMA_SHA256 = (
    "a8f02222351e445489a76d32ad2039c0ebd4c40a0a4669f95df6909a2a1d27fc"
)
DESIGN_SHA256 = "430926dbf6eafad2246e25e695ca9783ea73df2f91b9c427f9105de44e9858a8"
D9B_CONTRACT_SHA256 = (
    "8b2b2926fdda172b136b1a5223e7e61fc3ea091c553dfb01b4394cf910316eda"
)
D9B_REMOTE_REPORT_SHA256 = (
    "6fa53a0ae84db81bcb1ec2294876bfdd15d0a9f3717dbcbae621c6110565ac91"
)
ENDPOINT_SHA256 = (
    "1323b15f269f2bd4123992e213ec4d6dd1262376ad5c7b9563279f6b2543f562"
)
PROJECTION_CONTRACT_SHA256 = (
    "6e3a5b6b9ce690ab81e659d05d3c86d429371ecbd7f020b8445f4467ac07a01d"
)
CALL_IDS_SHA256 = "1cf8da2f335921588ee0dd4b2f034346f10821aa99f8a69296183d3ff90866a4"
MODEL_CONFIG_SHA256 = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
EXPECTED_PACKAGE_VERSION = "0.8.32"
EXPECTED_MODEL_SOURCE_SHA256 = (
    "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
)
D6D_RUNTIME_SHA256 = (
    "883fbbf314ad1cd93bd2547cfe1672c3d27b84419a0ee30881058e44905c609e"
)
INSTRUMENTER_SHA256 = (
    "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"
)
EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D9_REAL_CAUSAL_ISOLATION"
EXECUTION_LOCK_VALUE = "AUTHORIZED_D9D_REAL_2_9B_WITHIN_WRAPPER_CAUSAL_ISOLATION_ONCE"
IMPLEMENTATION_CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 D9-C calibration-only冻结projection contract与真实2.9B "
    "single-use安全入口的无模型实现；只允许绑定D9-A/D9-B冻结的32条calibration、64条"
    "held-out、七类同wrapper对照、448个pair/928次forward及全部阈值，定义仅使用calibration "
    "capture拟合、在访问held-out前冻结的字段分离projection artifact/schema，并实现全新"
    "authorization、single-use claim、唯一output、完整有序ledger和失败即停止入口及纯Python/"
    "AST验收；所有计分路线必须保持persistent wrapper，不得引入public计分路线或复用D8-C数据。"
    "本轮不探测installed source、不创建真实projection或机器authorization/claim/output、不导入"
    "RWKV/Torch、不访问权重、不加载或执行模型；不授权D9-D真实执行、D8-C或历史重跑、"
    "D7-D/D7-E、正式测试集、Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
FUTURE_EXECUTION_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 D9-D 真实2.9B within-wrapper causal isolation联合验证一次"
    "（同一进程、同一persistent wrapper，固定32次calibration capture后冻结真实projection，"
    "再执行64个held-out fixture的448个pair/896次forward，共928次forward），并授权观察本次"
    "工程结果；不授权重跑D9-D、D8-C或任何历史实验、自动重跑、D7-D/D7-E、正式测试集、"
    "Self效果结论、Self Updater或raw-original路线。"
)
CLASSIFICATION = (
    "d9c_calibration_only_projection_contract_and_single_use_entry_static_verified_"
    "execution_not_authorized"
)
NEXT_GATE = (
    "remote_no_model_d9c_static_verification_then_separate_exact_d9d_execution_"
    "authorization"
)
CHOICE_TOKEN_IDS = {"A": 66, "B": 67, "C": 68, "D": 69}
CODE_LABELS = ("A", "B", "C", "D")
CONDITION_MAP = {
    "wrapper_zero": "wrapper_zero",
    "active_true": "self_matched",
    "mask_identity": "self_identity_mask",
    "mask_goal": "self_goal_mask",
    "swap_identity": "self_identity_swap",
    "swap_goal": "self_goal_swap",
    "matched_random": "self_identity_goal_norm_matched_random",
    "synthetic_active": "synthetic_positive",
}
AUTHORIZATION_FIELDS = {
    "authorization_version", "stage", "scope", "authorized",
    "authorization_basis", "authorization_text", "authorized_at_utc",
    "git_commit", "entry_config_sha256", "entry_static_report_sha256",
    "d9a_design_sha256", "d9b_contract_sha256", "d9b_remote_report_sha256",
    "calibration_manifest_sha256", "heldout_manifest_sha256",
    "schedule_manifest_sha256", "endpoint_manifest_sha256",
    "projection_contract_sha256", "expanded_call_ids_sha256",
    "installed_source_probe_authorized", "projection_training_authorized",
    "real_projection_construction_authorized", "weights_access_authorized",
    "model_load_authorized", "model_execution_authorized",
    "result_observation_authorized", "model_forward_calls",
    "d9d_rerun_authorized", "d8c_rerun_authorized",
    "historical_rerun_authorized", "d7d_authorized", "d7e_authorized",
    "formal_test_set_authorized", "self_effect_conclusion_authorized",
    "self_updater_authorized", "raw_original_route_authorized",
    "automatic_rerun_authorized", "single_use", "authorization_digest_sha256",
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    D9B_CONTRACT_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    HELDOUT_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    PROJECTION_CONTRACT_RELATIVE_PATH,
    MODEL_CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d9c_projection_entry.md",
    "scripts/run_self_model_v0_1_d9d_real_causal_isolation.py",
    "scripts/verify_self_model_v0_1_d9c_projection_entry.py",
    "src/psa/self_model/d6d_ii_joint_runtime.py",
    "src/psa/self_model/d9a_within_wrapper_causal_isolation.py",
    "src/psa/self_model/d9b_manifest_endpoint_contract.py",
    "src/psa/self_model/d9c_projection_contract.py",
    "src/psa/self_model/d9c_real_entry.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d9c_projection_entry.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D9-C {label} must be an object")
    return value


def _require_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    expected = (root / relative).resolve()
    supplied = Path(value)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected:
        raise PermissionError(f"D9-C {label} path is not frozen")
    return supplied


def _git_metadata(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--porcelain"),
    }


def _require_clean_main(git: Mapping[str, str]) -> None:
    if git.get("branch") != "main" or git.get("status") != "":
        raise PermissionError("D9-C future execution requires a clean main checkout")
    commit = git.get("commit", "")
    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
        raise PermissionError("D9-C git commit is invalid")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
    return path


def _write_jsonl_record(handle: TextIO, payload: Mapping[str, Any]) -> None:
    handle.write(canonical_json_bytes(payload).decode("utf-8"))
    handle.write("\n")
    handle.flush()


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config.get("frozen_prerequisites", {})
    model = config.get("model", {})
    plan = config.get("execution_plan", {})
    artifacts = config.get("artifact_contract", {})
    ledger = config.get("ledger_contract", {})
    authority = config.get("implementation_authority", {})
    checks = {
        "identity_exact": config.get("entry_version") == ENTRY_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D9-C_calibration_only_projection_contract_and_single_use_entry"
        and config.get("status")
        == "entry_implemented_static_only_execution_not_authorized",
        "confirmation_exact": config.get("implementation_confirmation_text")
        == IMPLEMENTATION_CONFIRMATION_TEXT,
        "frozen_prerequisites_exact": frozen
        == {
            "d9a_design_path": DESIGN_RELATIVE_PATH,
            "d9a_design_sha256": DESIGN_SHA256,
            "d9b_contract_path": D9B_CONTRACT_RELATIVE_PATH,
            "d9b_contract_sha256": D9B_CONTRACT_SHA256,
            "d9b_remote_report_sha256": D9B_REMOTE_REPORT_SHA256,
            "calibration_manifest_sha256": CALIBRATION_SHA256,
            "heldout_manifest_sha256": HELDOUT_SHA256,
            "schedule_manifest_sha256": SCHEDULE_SHA256,
            "endpoint_manifest_sha256": ENDPOINT_SHA256,
            "projection_contract_sha256": PROJECTION_CONTRACT_SHA256,
            "expanded_call_ids_sha256": CALL_IDS_SHA256,
        },
        "model_shape_and_source_exact": model.get("config_path")
        == MODEL_CONFIG_RELATIVE_PATH
        and model.get("config_sha256") == MODEL_CONFIG_SHA256
        and model.get("expected_package_version") == EXPECTED_PACKAGE_VERSION
        and model.get("expected_model_source_sha256")
        == EXPECTED_MODEL_SOURCE_SHA256
        and model.get("n_layer") == 32
        and model.get("hidden_dimension") == 2560
        and model.get("state_component_count") == 96
        and model.get("target_layer_index_zero_based") == 15,
        "single_wrapper_928_plan_exact": plan
        == {
            "process_count": 1,
            "wrapper_count": 1,
            "calibration_capture_calls": 32,
            "heldout_fixtures": 64,
            "heldout_pair_blocks": 448,
            "heldout_forward_calls": 896,
            "total_forward_calls": 928,
            "scored_route": "persistent_wrapper",
            "public_scoring_route_allowed": False,
            "heldout_access_before_projection_freeze": False,
            "adaptive_retry_allowed": False,
            "discarded_scored_output_allowed": False,
        },
        "unique_single_use_paths_exact": artifacts.get("authorization_path")
        == AUTHORIZATION_RELATIVE_PATH
        and artifacts.get("output_dir") == OUTPUT_RELATIVE_DIR
        and artifacts.get("claim_filename") == "execution_claim.json"
        and artifacts.get("raw_ledger_filename") == "raw_ledger.jsonl"
        and artifacts.get("projection_artifact_filename") == "projection.json"
        and artifacts.get("report_filename") == "report.json"
        and artifacts.get("failure_filename") == "failure.json"
        and artifacts.get("integrity_filename") == "integrity.json"
        and artifacts.get("single_use") is True
        and artifacts.get("exclusive_create_only") is True,
        "ledger_exact": ledger.get("calibration_records") == 32
        and ledger.get("heldout_pair_records") == 448
        and ledger.get("total_records") == 480
        and ledger.get("represented_forward_calls") == 928
        and ledger.get("finite_values_required") is True
        and ledger.get("exact_frozen_order_required") is True
        and ledger.get("projection_digest_required_on_every_heldout_pair") is True
        and ledger.get("endpoint_after_complete_ledger_only") is True,
        "future_authorization_exact": config.get("execution_lock_env")
        == EXECUTION_LOCK_ENV
        and config.get("execution_lock_value") == EXECUTION_LOCK_VALUE
        and config.get("future_execution_authorization_text")
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "implementation_authority_exact": all(
            authority.get(name) is True
            for name in (
                "projection_contract_implementation_authorized",
                "authorization_schema_implementation_authorized",
                "single_use_entry_implementation_authorized",
                "claim_ledger_report_lifecycle_implementation_authorized",
                "pure_python_ast_verification_authorized",
            )
        ),
        "execution_and_later_authority_closed": all(
            authority.get(name) is False
            for name in (
                "installed_source_probe_authorized_at_implementation",
                "real_projection_construction_authorized_at_implementation",
                "machine_authorization_created_at_implementation",
                "execution_claim_created_at_implementation",
                "output_created_at_implementation",
                "rwkv_import_authorized_at_implementation",
                "torch_import_authorized_at_implementation",
                "weights_access_authorized_at_implementation",
                "model_load_authorized_at_implementation",
                "model_execution_authorized_at_implementation",
                "d9d_real_execution_authorized",
                "d8c_rerun_authorized",
                "historical_rerun_authorized",
                "d7d_authorized",
                "d7e_authorized",
                "formal_test_set_authorized",
                "self_effect_conclusion_authorized",
                "self_updater_authorized",
                "raw_original_route_authorized",
                "automatic_rerun_authorized",
            )
        ),
        "next_gate_exact": config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        raise PermissionError(
            "D9-C entry config failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def read_spec(path: str | Path) -> dict[str, Any]:
    spec = _object(path, "entry config")
    validate_config(spec)
    return spec


def build_call_plan(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = [
        {
            "call_id": item["call_id"],
            "phase": "calibration",
            "fixture_id": item["fixture_id"],
            "condition": "calibration_capture",
            "route": "persistent_wrapper_capture",
        }
        for item in schedule["calibration_calls"]
    ]
    for block in schedule["heldout_pair_blocks"]:
        for position, condition in enumerate(block["condition_order"], start=1):
            plan.append(
                {
                    "call_id": f"{block['pair_block_id']}-{position:02d}-{condition}",
                    "phase": "heldout",
                    "fixture_id": block["fixture_id"],
                    "pair_block_id": block["pair_block_id"],
                    "condition": condition,
                    "route": "persistent_wrapper",
                }
            )
    return plan


def validate_call_plan(plan: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    ids = [item.get("call_id") for item in plan]
    checks = {
        "call_count_exact": len(plan) == 928,
        "ids_unique": len(ids) == len(set(ids)),
        "call_id_digest_exact": sha256_json(ids) == CALL_IDS_SHA256,
        "calibration_first_exact": all(
            item.get("phase") == "calibration"
            and item.get("route") == "persistent_wrapper_capture"
            for item in plan[:32]
        ),
        "heldout_same_wrapper_only": all(
            item.get("phase") == "heldout"
            and item.get("route") == "persistent_wrapper"
            and item.get("condition") in CONDITION_MAP
            for item in plan[32:]
        ),
        "no_public_route": all("public" not in str(item.get("route")) for item in plan),
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-C call plan failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def _authorization_schema(root: Path) -> dict[str, Any]:
    schema = _object(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH, "authorization schema")
    if sha256_file(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH) != AUTHORIZATION_SCHEMA_SHA256:
        raise RuntimeError("D9-C authorization schema digest changed")
    if schema.get("additionalProperties") is not False:
        raise RuntimeError("D9-C authorization schema must fail closed")
    if set(schema.get("required", [])) != AUTHORIZATION_FIELDS:
        raise RuntimeError("D9-C authorization schema fields changed")
    return schema


def _authorization_payload(
    *, git_commit: str, entry_static_report_sha256: str, authorized_at_utc: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "authorization_version": ENTRY_VERSION,
        "stage": "Self-Model-v0.1-D9-D_real_within_wrapper_causal_isolation",
        "scope": "one_process_one_persistent_wrapper_calibration_freeze_then_heldout_928_forward",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_future_chat_authorization",
        "authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "authorized_at_utc": authorized_at_utc,
        "git_commit": git_commit,
        "entry_config_sha256": CONFIG_SHA256,
        "entry_static_report_sha256": entry_static_report_sha256,
        "d9a_design_sha256": DESIGN_SHA256,
        "d9b_contract_sha256": D9B_CONTRACT_SHA256,
        "d9b_remote_report_sha256": D9B_REMOTE_REPORT_SHA256,
        "calibration_manifest_sha256": CALIBRATION_SHA256,
        "heldout_manifest_sha256": HELDOUT_SHA256,
        "schedule_manifest_sha256": SCHEDULE_SHA256,
        "endpoint_manifest_sha256": ENDPOINT_SHA256,
        "projection_contract_sha256": PROJECTION_CONTRACT_SHA256,
        "expanded_call_ids_sha256": CALL_IDS_SHA256,
        "installed_source_probe_authorized": True,
        "projection_training_authorized": True,
        "real_projection_construction_authorized": True,
        "weights_access_authorized": True,
        "model_load_authorized": True,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "model_forward_calls": 928,
        "d9d_rerun_authorized": False,
        "d8c_rerun_authorized": False,
        "historical_rerun_authorized": False,
        "d7d_authorized": False,
        "d7e_authorized": False,
        "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "self_updater_authorized": False,
        "raw_original_route_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    payload["authorization_digest_sha256"] = sha256_json(payload)
    return payload


def build_d9_authorization(
    *, project_root: str | Path, authorization_text: str,
    entry_static_report_sha256: str, git: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    _authorization_schema(root)
    if authorization_text != FUTURE_EXECUTION_AUTHORIZATION_TEXT:
        raise PermissionError("D9-D future authorization text is not exact")
    metadata = dict(git) if git is not None else _git_metadata(root)
    _require_clean_main(metadata)
    if len(entry_static_report_sha256) != 64:
        raise PermissionError("D9-C static report digest is invalid")
    return _authorization_payload(
        git_commit=metadata["commit"],
        entry_static_report_sha256=entry_static_report_sha256,
        authorized_at_utc=_utc_now(),
    )


def _require_utc(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D9-D authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D9-D authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D9-D authorization timestamp must be UTC")


def validate_d9_authorization(
    *, authorization_path: str | Path, project_root: str | Path,
    git: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    _authorization_schema(root)
    value = _object(authorization_path, "machine authorization")
    if set(value) != AUTHORIZATION_FIELDS:
        raise PermissionError("D9-D machine authorization fields changed")
    _require_utc(value.get("authorized_at_utc"))
    expected = _authorization_payload(
        git_commit=git["commit"],
        entry_static_report_sha256=str(value.get("entry_static_report_sha256")),
        authorized_at_utc=str(value.get("authorized_at_utc")),
    )
    if value != expected:
        raise PermissionError("D9-D machine authorization payload differs")
    return value


def _create_claim(
    *, output_dir: Path, config_path: Path, authorization_path: Path,
    git: Mapping[str, str], entry_static_report_sha256: str,
    installed_source_sha256: str,
) -> Path:
    claim = {
        "claim_version": ENTRY_VERSION,
        "status": "d9d_single_use_joint_execution_claim_consumed",
        "single_use": True,
        "created_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "entry_static_report_sha256": entry_static_report_sha256,
        "installed_source_sha256": installed_source_sha256,
        "calibration_forward_calls": 32,
        "heldout_forward_calls": 896,
        "total_forward_calls": 928,
        "d9d_rerun_authorized": False,
        "automatic_rerun_authorized": False,
    }
    return _write_json_exclusive(output_dir / "execution_claim.json", claim)


def _validate_launcher_environment() -> dict[str, bool]:
    checks = {
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "python_hash_seed_exact": os.environ.get("PYTHONHASHSEED") == "29083101",
        "rwkv_de_unset": os.environ.get("RWKV_DE_VERSION") is None,
    }
    if not all(checks.values()):
        raise PermissionError("D9-D launcher deterministic environment is incomplete")
    return checks


def _probe_installed_source() -> tuple[str, Path, bytes, str]:
    module = import_module("psa.self_model.d4a_real_diagnostic")
    return module._installed_source()


def _runtime_dependencies() -> dict[str, Any]:
    torch = import_module("torch")
    model_module = import_module("psa.model.rwkv7")
    instrumenter = import_module("psa.self_model.rwkv7_instrumented_off_runtime")
    runtime = import_module("psa.self_model.d6d_ii_joint_runtime")
    return {
        "torch": torch,
        "RWKV7Adapter": model_module.RWKV7Adapter,
        "load_model_config": model_module.load_model_config,
        "clone_state": model_module.clone_state,
        "compile_instrumented_methods": instrumenter.compile_instrumented_methods,
        "Wrapper": runtime.D6DIIWrapperOwnedRuntime,
        "TrainingCaptureCallback": runtime.TrainingCaptureCallback,
        "SyntheticPositiveCallback": runtime.SyntheticPositiveCallback,
        "FrozenProjectionCallback": runtime.FrozenProjectionCallback,
        "request_for_training_capture": runtime.request_for_training_capture,
        "request_for_pilot_condition": runtime.request_for_pilot_condition,
    }


def _apply_runtime_determinism(torch: Any) -> dict[str, Any]:
    random.seed(29083101)
    torch.manual_seed(29083101)
    torch.cuda.manual_seed_all(29083101)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "runtime_seed": 29083101,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _tensor_digest(tensor: Any, torch: Any) -> str:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("D9-D output tensor is nonfinite")
    byte_view = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _scores(output: tuple[Any, Sequence[Any]], torch: Any) -> dict[str, Any]:
    logits, state = output
    shape = tuple(logits.shape)
    selected = logits[-1] if len(shape) == 2 else logits
    if tuple(selected.shape) != (65536,) or len(state) != 96:
        raise ValueError("D9-D logits or state shape changed")
    if not all(bool(torch.isfinite(component).all().item()) for component in state):
        raise ValueError("D9-D state is nonfinite")
    values = {
        code: float(selected[token_id].detach().float().item())
        for code, token_id in CHOICE_TOKEN_IDS.items()
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("D9-D choice score is nonfinite")
    return {
        "choice_scores": values,
        "logits_sha256": _tensor_digest(selected, torch),
        "state_component_count": len(state),
    }


def _target_codes(identity: int, goal: int, rotation: int) -> dict[str, str]:
    def code(i: int, g: int) -> str:
        return CODE_LABELS[((i * 3 + g) % 4 + rotation) % 4]
    return {
        "true": code(identity, goal),
        "identity_swap": code((identity + 1) % 4, goal),
        "goal_swap": code(identity, (goal + 1) % 4),
    }


def _margins(scores: Mapping[str, float], codes: Mapping[str, str]) -> dict[str, float]:
    true = codes["true"]
    return {
        "target_alignment_margin": scores[true]
        - max(value for code, value in scores.items() if code != true),
        "identity_margin": scores[true] - scores[codes["identity_swap"]],
        "goal_margin": scores[true] - scores[codes["goal_swap"]],
        "identity_swap_advantage": scores[codes["identity_swap"]] - scores[true],
        "goal_swap_advantage": scores[codes["goal_swap"]] - scores[true],
    }


def _bootstrap_lower_bound(values: Sequence[float], seed: int) -> float:
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError("D9-D bootstrap requires sixteen finite base-case values")
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(16)] for _ in range(16)) / 16
        for _ in range(100000)
    ]
    means.sort()
    return means[999]


def _real_endpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 448:
        raise ValueError("D9-D endpoint requires exactly 448 pair records")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(str(record["base_case_id"]), {}).setdefault(
            str(record["contrast"]), []
        ).append(record)
    if len(grouped) != 16 or any(
        len(values) != 4
        for contrasts in grouped.values()
        for values in contrasts.values()
    ):
        raise ValueError("D9-D endpoint rotation groups are incomplete")
    base_ids = sorted(grouped)
    active: list[float] = []
    true_random: list[float] = []
    mask_identity_count = 0
    mask_goal_count = 0
    swap_identity_count = 0
    swap_goal_count = 0
    synthetic_count = 0
    identity_levels: dict[int, list[float]] = {index: [] for index in range(4)}
    goal_levels: dict[int, list[float]] = {index: [] for index in range(4)}
    for base_id in base_ids:
        contrasts = grouped[base_id]
        deltas = {
            name: statistics.fmean(
                item["condition_margins"]["target_alignment_margin"]
                - item["zero_margins"]["target_alignment_margin"]
                for item in items
            )
            for name, items in contrasts.items()
        }
        active.append(deltas["active_true"])
        true_random.append(deltas["active_true"] - deltas["matched_random"])
        identity = int(contrasts["active_true"][0]["identity_index"])
        goal = int(contrasts["active_true"][0]["goal_index"])
        identity_levels[identity].append(deltas["active_true"])
        goal_levels[goal].append(deltas["active_true"])
        mask_i = contrasts["mask_identity"]
        if statistics.fmean(
            item["condition_margins"]["identity_margin"]
            - item["zero_margins"]["identity_margin"] for item in mask_i
        ) < 0.0 and statistics.fmean(
            item["condition_margins"]["goal_margin"]
            - item["zero_margins"]["goal_margin"] for item in mask_i
        ) > 0.0:
            mask_identity_count += 1
        mask_g = contrasts["mask_goal"]
        if statistics.fmean(
            item["condition_margins"]["goal_margin"]
            - item["zero_margins"]["goal_margin"] for item in mask_g
        ) < 0.0 and statistics.fmean(
            item["condition_margins"]["identity_margin"]
            - item["zero_margins"]["identity_margin"] for item in mask_g
        ) > 0.0:
            mask_goal_count += 1
        if statistics.fmean(
            item["condition_margins"]["identity_swap_advantage"]
            for item in contrasts["swap_identity"]
        ) > 0.0:
            swap_identity_count += 1
        if statistics.fmean(
            item["condition_margins"]["goal_swap_advantage"]
            for item in contrasts["swap_goal"]
        ) > 0.0:
            swap_goal_count += 1
        synthetic_count += sum(
            bool(item["synthetic_output_changed"])
            for item in contrasts["synthetic_active"]
        )
    metrics = {
        "active_minus_zero_mean": statistics.fmean(active),
        "active_minus_zero_lb99": _bootstrap_lower_bound(active, 29083102),
        "positive_base_cases": sum(value > 0.0 for value in active),
        "identity_level_min_positive": min(
            sum(value > 0.0 for value in values) for values in identity_levels.values()
        ),
        "goal_level_min_positive": min(
            sum(value > 0.0 for value in values) for values in goal_levels.values()
        ),
        "true_minus_random_lb99": _bootstrap_lower_bound(true_random, 29083103),
        "mask_identity_specific_count": mask_identity_count,
        "mask_goal_specific_count": mask_goal_count,
        "swap_identity_follow_count": swap_identity_count,
        "swap_goal_follow_count": swap_goal_count,
        "synthetic_active_changed_fixture_count": synthetic_count,
    }
    decision = evaluate_candidate(metrics)
    return {
        "metrics": metrics,
        "checks": decision["checks"],
        "all_gates_pass": decision["all_gates_pass"],
        "decision": decision["decision"],
        "self_effect_conclusion": False,
    }


def _persist_integrity(
    *, output_dir: Path, claim_path: Path, raw_path: Path,
    projection_path: Path, report_path: Path,
) -> Path:
    integrity = {
        "integrity_version": ENTRY_VERSION,
        "status": "d9d_real_artifact_integrity_complete",
        "execution_claim_sha256": sha256_file(claim_path),
        "raw_ledger_sha256": sha256_file(raw_path),
        "projection_artifact_sha256": sha256_file(projection_path),
        "report_sha256": sha256_file(report_path),
        "d9d_rerun_authorized": False,
        "automatic_rerun_authorized": False,
    }
    integrity["integrity_digest_sha256"] = sha256_json(integrity)
    return _write_json_exclusive(output_dir / "integrity.json", integrity)


def _load_heldout_after_freeze(
    *, root: Path, projection_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not projection_path.is_file():
        raise RuntimeError("D9-D projection must be frozen before heldout access")
    return (
        _object(root / HELDOUT_RELATIVE_PATH, "heldout manifest"),
        _object(root / SCHEDULE_RELATIVE_PATH, "heldout schedule"),
    )


def run_d9d_real_causal_isolation(
    *, config_path: str | Path, authorization_path: str | Path,
    project_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    read_spec(config_file)
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D9-D execution lock is absent")
    launcher_checks = _validate_launcher_environment()
    authorization_file = _require_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d9_authorization(
        authorization_path=authorization_file, project_root=root, git=git
    )
    installed_version, source_path, source_bytes, source_digest = _probe_installed_source()
    if installed_version != EXPECTED_PACKAGE_VERSION or source_digest != EXPECTED_MODEL_SOURCE_SHA256:
        raise RuntimeError("D9-D installed source lock differs")
    claim_path = _create_claim(
        output_dir=destination,
        config_path=config_file,
        authorization_path=authorization_file,
        git=git,
        entry_static_report_sha256=authorization["entry_static_report_sha256"],
        installed_source_sha256=source_digest,
    )
    started = time.perf_counter()
    raw_path = destination / "raw_ledger.jsonl"
    projection_path = destination / "projection.json"
    calibration_count = 0
    heldout_forward_count = 0
    pair_records: list[dict[str, Any]] = []
    try:
        dependencies = _runtime_dependencies()
        model_config_path = root / MODEL_CONFIG_RELATIVE_PATH
        if sha256_file(model_config_path) != MODEL_CONFIG_SHA256:
            raise RuntimeError("D9-D model config digest changed")
        model_config = dependencies["load_model_config"](
            model_config_path, root, verify_files=True
        )
        for key, value in model_config.environment.items():
            os.environ[key] = value
        torch = dependencies["torch"]
        determinism = _apply_runtime_determinism(torch)
        adapter = dependencies["RWKV7Adapter"].load(model_config)
        for code, token_id in CHOICE_TOKEN_IDS.items():
            encoded = adapter.encode(code)
            if encoded != [token_id] or adapter.decode(encoded) != code:
                raise RuntimeError("D9-D frozen choice-token roundtrip changed")
        methods, injection_counts = dependencies["compile_instrumented_methods"](
            upstream_source=source_bytes.decode("utf-8"),
            upstream_globals=vars(sys.modules["rwkv.model"]),
            rwkv_de_version=None,
        )
        wrapper = dependencies["Wrapper"](
            base_model=adapter.model,
            compiled_methods=methods,
            injection_counts=injection_counts,
        )
        calibration_manifest = _object(
            root / CALIBRATION_RELATIVE_PATH, "calibration manifest"
        )
        captures: list[CalibrationCapture] = []
        with raw_path.open("x", encoding="utf-8", newline="\n") as raw_handle:
            for fixture in calibration_manifest["fixtures"]:
                callback = dependencies["TrainingCaptureCallback"](torch)
                source_state = adapter.model.generate_zero_state()
                with torch.inference_mode():
                    output = wrapper.forward(
                        list(fixture["token_ids"]),
                        dependencies["clone_state"](source_state),
                        full_output=bool(fixture["full_output"]),
                        coupling=dependencies["request_for_training_capture"](callback),
                    )
                if output is None or callback.vector is None:
                    raise RuntimeError("D9-D calibration capture produced no vector")
                if callback.invocation_count != 32 or callback.application_count != 1:
                    raise RuntimeError("D9-D calibration callback count changed")
                capture = CalibrationCapture(
                    fixture_id=fixture["fixture_id"],
                    identity_index=int(fixture["identity_index"]),
                    goal_index=int(fixture["goal_index"]),
                    replicate=int(fixture["replicate"]),
                    vector=tuple(callback.vector),
                )
                captures.append(capture)
                calibration_count += 1
                _write_jsonl_record(
                    raw_handle,
                    {
                        "record_type": "calibration_capture",
                        "call_id": f"{fixture['fixture_id']}-capture",
                        "fixture_id": fixture["fixture_id"],
                        "phase": "calibration",
                        "route": "persistent_wrapper_capture",
                        "capture_sha256": sha256_json(
                            [format(value, ".12e") for value in callback.vector]
                        ),
                        "callback_invocations": callback.invocation_count,
                        "target_layer_applications": callback.application_count,
                        "heldout_scored": False,
                    },
                )
            artifact = build_frozen_projection_artifact(
                captures=captures,
                calibration_manifest_sha256=CALIBRATION_SHA256,
                calibration_commitment_sha256=CALIBRATION_COMMITMENT,
                heldout_manifest_sha256=HELDOUT_SHA256,
                heldout_commitment_sha256=HELDOUT_COMMITMENT,
                schedule_commitment_sha256=SCHEDULE_COMMITMENT,
                output_dimension=2560,
                fixture_only=False,
            )
            audit_frozen_projection_artifact(
                artifact, expected_dimension=2560, fixture_only=False
            )
            _write_json_exclusive(projection_path, artifact)

            # The semantic held-out payload is intentionally first parsed only
            # after the projection artifact has been audited and frozen.
            heldout_manifest, schedule = _load_heldout_after_freeze(
                root=root, projection_path=projection_path
            )
            call_plan = build_call_plan(schedule)
            validate_call_plan(call_plan)
            fixtures = {
                item["fixture_id"]: item for item in heldout_manifest["fixtures"]
            }
            source_states = {
                fixture_id: adapter.model.generate_zero_state() for fixture_id in fixtures
            }
            for block in schedule["heldout_pair_blocks"]:
                fixture = fixtures[block["fixture_id"]]
                codes = _target_codes(
                    int(fixture["identity_index"]),
                    int(fixture["goal_index"]),
                    int(fixture["code_rotation"]),
                )
                if codes["true"] != fixture["target_code"]:
                    raise RuntimeError("D9-D heldout target-code derivation changed")
                observations: list[dict[str, Any]] = []
                callbacks: list[Any] = []
                for condition in block["condition_order"]:
                    callback = None
                    mapped = CONDITION_MAP[condition]
                    projection_digest = artifact["artifact_digest_sha256"]
                    if condition == "synthetic_active":
                        callback = dependencies["SyntheticPositiveCallback"](
                            torch, f"{sha256_file(claim_path)}|{fixture['fixture_id']}"
                        )
                    elif condition != "wrapper_zero":
                        vector = project_condition(
                            artifact,
                            condition=condition,
                            identity_index=int(fixture["identity_index"]),
                            goal_index=int(fixture["goal_index"]),
                            fixture_id=fixture["fixture_id"],
                        )
                        callback = dependencies["FrozenProjectionCallback"](torch, vector)
                    request = dependencies["request_for_pilot_condition"](mapped, callback)
                    with torch.inference_mode():
                        output = wrapper.forward(
                            list(fixture["token_ids"]),
                            dependencies["clone_state"](
                                source_states[fixture["fixture_id"]]
                            ),
                            full_output=bool(fixture["full_output"]),
                            coupling=request,
                        )
                    scored = _scores(output, torch)
                    margins = _margins(scored["choice_scores"], codes)
                    if callback is not None:
                        callbacks.append(callback)
                        if callback.invocation_count != 32 or callback.application_count != 1:
                            raise RuntimeError("D9-D heldout callback count changed")
                    observations.append(
                        {
                            "condition": condition,
                            "choice_scores": scored["choice_scores"],
                            "margins": margins,
                            "logits_sha256": scored["logits_sha256"],
                            "state_component_count": scored["state_component_count"],
                            "projection_artifact_sha256": projection_digest,
                        }
                    )
                    heldout_forward_count += 1
                by_condition = {item["condition"]: item for item in observations}
                zero = by_condition["wrapper_zero"]
                condition = by_condition[block["contrast"]]
                record = {
                    "record_type": "heldout_pair",
                    "pair_block_id": block["pair_block_id"],
                    "fixture_id": block["fixture_id"],
                    "base_case_id": block["base_case_id"],
                    "identity_index": block["identity_index"],
                    "goal_index": block["goal_index"],
                    "code_rotation": block["code_rotation"],
                    "contrast": block["contrast"],
                    "latin_position": block["latin_position"],
                    "pair_order": block["pair_order"],
                    "condition_order": block["condition_order"],
                    "route": "persistent_wrapper",
                    "source_state_contract": block["source_state_contract"],
                    "zero_margins": zero["margins"],
                    "condition_margins": condition["margins"],
                    "observations": observations,
                    "synthetic_output_changed": (
                        condition["logits_sha256"] != zero["logits_sha256"]
                        if block["contrast"] == "synthetic_active" else False
                    ),
                    "projection_artifact_sha256": artifact[
                        "artifact_digest_sha256"
                    ],
                }
                pair_records.append(record)
                _write_jsonl_record(raw_handle, record)
        if calibration_count != 32 or heldout_forward_count != 896 or len(pair_records) != 448:
            raise RuntimeError("D9-D completed call or ledger counts changed")
        endpoint = _real_endpoint(pair_records)
        report: dict[str, Any] = {
            "report_version": ENTRY_VERSION,
            "status": "d9d_real_within_wrapper_causal_isolation_completed_claim_consumed",
            "valid": True,
            "classification": endpoint["decision"],
            "self_effect_conclusion": False,
            "git_commit": git["commit"],
            "authorization_digest_sha256": authorization[
                "authorization_digest_sha256"
            ],
            "execution_claim_sha256": sha256_file(claim_path),
            "installed_source": {
                "version": installed_version,
                "path": str(source_path),
                "sha256": source_digest,
            },
            "launcher_checks": launcher_checks,
            "runtime_determinism": determinism,
            "projection": {
                "artifact_version": ARTIFACT_VERSION,
                "artifact_digest_sha256": artifact["artifact_digest_sha256"],
                "parameter_digest_sha256": artifact["parameter_digest_sha256"],
                "calibration_only": True,
                "frozen_before_heldout_access": True,
            },
            "counts": {
                "calibration_forward_calls": calibration_count,
                "heldout_pair_records": len(pair_records),
                "heldout_forward_calls": heldout_forward_count,
                "total_forward_calls": calibration_count + heldout_forward_count,
                "ledger_records": calibration_count + len(pair_records),
            },
            "endpoint": endpoint,
            "elapsed_seconds": time.perf_counter() - started,
            "d9d_rerun_authorized": False,
            "automatic_rerun_authorized": False,
        }
        report["report_digest_sha256"] = sha256_json(report)
        report_path = _write_json_exclusive(destination / "report.json", report)
        _persist_integrity(
            output_dir=destination,
            claim_path=claim_path,
            raw_path=raw_path,
            projection_path=projection_path,
            report_path=report_path,
        )
        return report
    except Exception as error:
        failure: dict[str, Any] = {
            "report_version": ENTRY_VERSION,
            "status": "d9d_real_joint_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "calibration_forward_calls": calibration_count,
            "heldout_forward_calls": heldout_forward_count,
            "heldout_pair_records": len(pair_records),
            "real_projection_created": projection_path.exists(),
            "self_effect_conclusion": False,
            "d9d_rerun_authorized": False,
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        _write_json_exclusive(destination / "failure.json", failure)
        return failure


def _entry_ast_audit() -> dict[str, int]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    target = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_d9d_real_causal_isolation"
    )
    names = {
        "validate_d9_authorization",
        "_probe_installed_source",
        "_create_claim",
        "_runtime_dependencies",
        "_apply_runtime_determinism",
        "load",
        "build_frozen_projection_artifact",
        "_load_heldout_after_freeze",
        "_real_endpoint",
    }
    found: dict[str, int] = {}
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in names:
            found.setdefault(name, node.lineno)
    if set(found) != names:
        raise RuntimeError("D9-C entry AST call inventory changed")
    return found


def _execution_artifacts_absent(root: Path) -> dict[str, bool]:
    output = root / OUTPUT_RELATIVE_DIR
    return {
        "machine_authorization_absent": not (root / AUTHORIZATION_RELATIVE_PATH).exists(),
        "execution_claim_absent": not (output / "execution_claim.json").exists(),
        "projection_artifact_absent": not (output / "projection.json").exists(),
        "raw_ledger_absent": not (output / "raw_ledger.jsonl").exists(),
        "report_absent": not (output / "report.json").exists(),
        "failure_absent": not (output / "failure.json").exists(),
        "integrity_absent": not (output / "integrity.json").exists(),
    }


def run_pure_python_acceptance(root: Path) -> dict[str, Any]:
    schedule = _object(root / SCHEDULE_RELATIVE_PATH, "schedule")
    plan = build_call_plan(schedule)
    plan_checks = validate_call_plan(plan)
    before = copy.deepcopy(plan)
    missing = plan[:-1]
    duplicate = copy.deepcopy(plan)
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    reordered = copy.deepcopy(plan)
    reordered[32], reordered[33] = reordered[33], reordered[32]
    public = copy.deepcopy(plan)
    public[32]["route"] = "public"

    def fails(value: Sequence[Mapping[str, Any]]) -> bool:
        try:
            validate_call_plan(value)
        except ValueError:
            return True
        return False

    fake_git = {"commit": "a" * 40, "branch": "main", "status": ""}
    payload = _authorization_payload(
        git_commit=fake_git["commit"],
        entry_static_report_sha256="b" * 64,
        authorized_at_utc="2026-09-04T00:00:00+00:00",
    )
    tampered = copy.deepcopy(payload)
    tampered["model_forward_calls"] = 927
    checks = {
        "plan_valid": all(plan_checks.values()),
        "missing_call_rejected": fails(missing),
        "duplicate_call_rejected": fails(duplicate),
        "reordered_call_rejected": fails(reordered),
        "public_route_rejected": fails(public),
        "plan_input_unchanged": plan == before,
        "authorization_fields_exact": set(payload) == AUTHORIZATION_FIELDS,
        "authorization_digest_valid": payload["authorization_digest_sha256"]
        == sha256_json(
            {key: value for key, value in payload.items()
             if key != "authorization_digest_sha256"}
        ),
        "authorization_tamper_changes_digest": tampered["authorization_digest_sha256"]
        != sha256_json(
            {key: value for key, value in tampered.items()
             if key != "authorization_digest_sha256"}
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "call_plan_checks": plan_checks,
        "counts": {
            "calibration_calls": 32,
            "heldout_calls": 896,
            "heldout_pairs": 448,
            "total_calls": len(plan),
        },
    }


def build_static_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    schema = _authorization_schema(root)
    projection = verify_projection_contract_files(root)
    pure = run_pure_python_acceptance(root)
    ast_lines = _entry_ast_audit()
    artifacts = _execution_artifacts_absent(root)
    locked_hashes = {
        CONFIG_RELATIVE_PATH: CONFIG_SHA256,
        AUTHORIZATION_SCHEMA_RELATIVE_PATH: AUTHORIZATION_SCHEMA_SHA256,
        DESIGN_RELATIVE_PATH: DESIGN_SHA256,
        D9B_CONTRACT_RELATIVE_PATH: D9B_CONTRACT_SHA256,
        CALIBRATION_RELATIVE_PATH: CALIBRATION_SHA256,
        HELDOUT_RELATIVE_PATH: HELDOUT_SHA256,
        SCHEDULE_RELATIVE_PATH: SCHEDULE_SHA256,
        ENDPOINT_RELATIVE_PATH: ENDPOINT_SHA256,
        PROJECTION_CONTRACT_RELATIVE_PATH: PROJECTION_CONTRACT_SHA256,
        MODEL_CONFIG_RELATIVE_PATH: MODEL_CONFIG_SHA256,
        "src/psa/self_model/d6d_ii_joint_runtime.py": D6D_RUNTIME_SHA256,
        "src/psa/self_model/rwkv7_instrumented_off_runtime.py": INSTRUMENTER_SHA256,
    }
    lock_checks = {
        path: sha256_file(root / path) == digest for path, digest in locked_hashes.items()
    }
    checks = {
        "config_valid": all(validate_config(spec).values()),
        "projection_contract_and_fake_acceptance_valid": projection["valid"],
        "call_plan_and_authorization_fake_valid": pure["valid"],
        "authorization_schema_exact": set(schema["required"]) == AUTHORIZATION_FIELDS,
        "all_prerequisite_hashes_frozen": all(lock_checks.values()),
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "execution_artifacts_absent": all(artifacts.values()),
        "authorization_precedes_installed_source_probe": ast_lines[
            "validate_d9_authorization"
        ] < ast_lines["_probe_installed_source"],
        "claim_precedes_model_stack_and_load": ast_lines["_create_claim"]
        < ast_lines["_runtime_dependencies"] < ast_lines["load"],
        "determinism_precedes_model_load": ast_lines["_apply_runtime_determinism"]
        < ast_lines["load"],
        "projection_frozen_before_heldout_manifest_load": ast_lines[
            "build_frozen_projection_artifact"
        ] < ast_lines["_load_heldout_after_freeze"],
        "endpoint_runs_after_projection_and_heldout": ast_lines[
            "build_frozen_projection_artifact"
        ] < ast_lines["_real_endpoint"],
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "D9-C static verification failed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    report: dict[str, Any] = {
        "report_version": ENTRY_VERSION,
        "status": "d9c_projection_contract_and_single_use_entry_static_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": validate_config(spec),
        "projection_contract": projection,
        "pure_python_acceptance": pure,
        "entry_call_lines": ast_lines,
        "locked_source_checks": lock_checks,
        "execution_artifacts": artifacts,
        "future_exact_owner_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "next_gate": NEXT_GATE,
        "safety": {
            "installed_source_probed": False,
            "real_projection_constructed": False,
            "machine_authorization_created": False,
            "execution_claim_created": False,
            "output_created": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d9d_real_execution_authorized": False,
            "d8c_or_historical_rerun": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
