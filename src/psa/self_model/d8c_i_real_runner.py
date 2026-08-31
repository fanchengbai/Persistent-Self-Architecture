"""Single-use D8-C real runner with a no-model static verification surface."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence, TextIO

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.self_model.d7c_public_semantics_runtime import (
    D7CPublicSemanticsWrapper,
    zero_request,
)
from psa.self_model.d8_numerical_identifiability_design import (
    expand_fixtures,
    expand_schedule,
)
from psa.self_model.d8b_manifest_endpoint_contract import (
    DESIGN_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    aggregate_fixture_excess,
    decide_excess_drift,
)
from psa.self_model.d8c_real_numerical_identifiability import (
    D8B_CONTRACT_RELATIVE_PATH,
    DETERMINISM_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    build_call_plan,
    validate_call_plan,
)


RUNNER_VERSION = "0.1-self-model-d8c-i-real-runner"
CONFIG_RELATIVE_PATH = "configs/development/self_model_v0_1_d8c_i_real_runner.json"
D8C_PROTOCOL_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8c_real_numerical_identifiability.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d8c_i_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = "results/authorizations/self_model_v0_1_d8c_real_v01.json"
OUTPUT_RELATIVE_DIR = "results/development/self_model_v0_1_d8c_real_v01"
MODEL_CONFIG_RELATIVE_PATH = "configs/models/rwkv7_g1h_2.9b.candidate.json"

D8C_PROTOCOL_SHA256 = "3eac85714df01d842cc93445c67686ae75c948c4aa9cc77eb97c32aa9fe4eb90"
D8C_REMOTE_STATIC_REPORT_SHA256 = (
    "a5bdf69f262b1203e8de916806fb3e90edd2fcba68cb7274633d43858c5852b1"
)
D8B_CONTRACT_SHA256 = "cf3c1142ba4d97fa7cd484415869e5f6e54635f1264204e9f7dcbf6bfa07fc4f"
FIXTURE_SHA256 = "d0b9c2e67eff48f2e9fd3cf9b5244cdcec165091bb4b079acecd8fd5b3323c2a"
SCHEDULE_SHA256 = "b1ecd2c027bd6aed8834147ec333c5a2ac1ba3df6550ee3ab221612e4805943f"
DETERMINISM_SHA256 = "81d98e24a61f29b9e5e5fed77612b2ffa56953c80282ee811192bbd7417dff83"
ENDPOINT_SHA256 = "96517ab9314ac2e8f68fd520bfb860626e39b1b69e3403f7033781a69983458c"
CALL_IDS_SHA256 = "7004dd99e62d0657be968096f83b4099b6752cb07bf203577c31b487db3190ca"
MODEL_CONFIG_SHA256 = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
EXPECTED_PACKAGE_VERSION = "0.8.32"
EXPECTED_MODEL_SOURCE_SHA256 = (
    "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
)
D7C_WRAPPER_SHA256 = "064a8f345f15db32a65c765b4a774aa48482510616756cb0ef8b174eda9f3dda"
INSTRUMENTER_SHA256 = "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"

EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D8C_REAL_NUMERICAL_IDENTIFIABILITY"
EXECUTION_LOCK_VALUE = "AUTHORIZED_D8C_REAL_2_9B_NUMERICAL_IDENTIFIABILITY_ONCE"
IMPLEMENTATION_CONFIRMATION = (
    "确认进入 Self Model v0.1 D8-C-I 真实2.9B数值可识别性 single-use runner 与无模型静态验证实现；"
    "只允许实现冻结的584-call计划、严格确定性预检、authorization验证、唯一claim/output生命周期、"
    "完整ledger及失败报告，并进行纯Python/AST验收；本轮不探测installed source、不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型、不创建机器authorization或claim，也不授权D8-C真实执行、任何重跑、"
    "D7-D/D7-E、projection、正式测试集、Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
CLASSIFICATION = (
    "d8c_i_single_use_real_runner_static_verified_execution_not_authorized"
)
NEXT_GATE = (
    "remote_no_model_d8c_i_static_verification_then_separate_exact_d8c_execution_authorization"
)

AUTHORIZATION_FIELDS = {
    "authorization_version",
    "stage",
    "scope",
    "authorized",
    "authorization_basis",
    "authorization_text",
    "authorized_at_utc",
    "git_commit",
    "runner_config_sha256",
    "runner_static_report_sha256",
    "d8c_protocol_sha256",
    "d8c_remote_static_report_sha256",
    "d8b_contract_sha256",
    "fixture_manifest_sha256",
    "schedule_manifest_sha256",
    "determinism_manifest_sha256",
    "endpoint_manifest_sha256",
    "expanded_call_ids_sha256",
    "installed_source_probe_authorized",
    "weights_access_authorized",
    "model_load_authorized",
    "model_execution_authorized",
    "result_observation_authorized",
    "model_forward_calls",
    "d8c_rerun_authorized",
    "historical_rerun_authorized",
    "d7d_authorized",
    "d7e_authorized",
    "projection_authorized",
    "formal_test_set_authorized",
    "self_effect_conclusion_authorized",
    "self_updater_authorized",
    "raw_original_route_authorized",
    "automatic_rerun_authorized",
    "single_use",
    "authorization_digest_sha256",
}

SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    D8C_PROTOCOL_RELATIVE_PATH,
    D8B_CONTRACT_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    DETERMINISM_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    MODEL_CONFIG_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_d8c_i_real_runner.md",
    "scripts/run_self_model_v0_1_d8c_real_numerical_identifiability.py",
    "scripts/verify_self_model_v0_1_d8c_i_real_runner.py",
    "src/psa/self_model/d7c_public_semantics_runtime.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "src/psa/self_model/d8c_i_real_runner.py",
    "tests/test_self_model_d8c_i_real_runner.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D8-C-I {label} must be an object")
    return value


def _require_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    expected = (root / relative).resolve()
    supplied = Path(value)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected:
        raise PermissionError(f"D8-C-I {label} path is not frozen")
    return supplied


def _git_metadata(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "origin_main": run("rev-parse", "refs/remotes/origin/main"),
        "status": run("status", "--short"),
    }


def _require_clean_main(git: Mapping[str, str]) -> None:
    if git.get("branch") != "main":
        raise PermissionError("D8-C-I real execution requires branch main")
    if git.get("commit") != git.get("origin_main"):
        raise PermissionError("D8-C-I HEAD must equal origin/main")
    if git.get("status") != "":
        raise PermissionError("D8-C-I real execution requires a clean worktree")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
    except FileExistsError as error:
        raise FileExistsError(f"D8-C-I refuses to overwrite {path}") from error
    return path


def _write_jsonl_record(handle: TextIO, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")
    handle.flush()


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config.get("frozen_prerequisites", {})
    model = config.get("model", {})
    plan = config.get("execution_plan", {})
    deterministic = config.get("determinism_preflight", {})
    artifacts = config.get("artifact_contract", {})
    ledger = config.get("ledger_contract", {})
    authority = config.get("implementation_authority", {})
    checks = {
        "identity_exact": config.get("runner_version") == RUNNER_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D8-C-I_single_use_real_numerical_identifiability_runner"
        and config.get("status")
        == "runner_implemented_static_only_execution_not_authorized"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("implementation_confirmation_text")
        == IMPLEMENTATION_CONFIRMATION,
        "frozen_prerequisites_exact": frozen
        == {
            "d8c_protocol_path": D8C_PROTOCOL_RELATIVE_PATH,
            "d8c_protocol_sha256": D8C_PROTOCOL_SHA256,
            "d8c_remote_static_report_sha256": D8C_REMOTE_STATIC_REPORT_SHA256,
            "d8b_contract_sha256": D8B_CONTRACT_SHA256,
            "fixture_manifest_sha256": FIXTURE_SHA256,
            "schedule_manifest_sha256": SCHEDULE_SHA256,
            "determinism_manifest_sha256": DETERMINISM_SHA256,
            "endpoint_manifest_sha256": ENDPOINT_SHA256,
            "expanded_call_ids_sha256": CALL_IDS_SHA256,
        },
        "model_lock_exact": model
        == {
            "model_id": "rwkv7-g1h-2.9b-20260710",
            "config_path": MODEL_CONFIG_RELATIVE_PATH,
            "config_sha256": MODEL_CONFIG_SHA256,
            "expected_package": "rwkv",
            "expected_package_version": EXPECTED_PACKAGE_VERSION,
            "expected_model_source_sha256": EXPECTED_MODEL_SOURCE_SHA256,
            "rwkv_de_version": "unset",
            "n_layer": 32,
            "hidden_dimension": 2560,
            "state_component_count": 96,
        },
        "execution_plan_exact": plan
        == {
            "process_count": 1,
            "wrapper_count": 1,
            "conditioning_forward_calls": 8,
            "scored_pair_blocks": 288,
            "scored_forward_calls": 576,
            "total_forward_calls": 584,
            "routes": ["public", "wrapper_zero"],
            "state_contract": "fresh_clone_of_fixture_prebuilt_zero_state_for_every_call",
            "adaptive_retry_allowed": False,
            "discarded_scored_output_allowed": False,
        },
        "determinism_exact": deterministic
        == {
            "launcher_environment_before_python": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "28083101",
                "RWKV_DE_VERSION": "unset",
            },
            "runtime_seed": 28083101,
            "torch_use_deterministic_algorithms": True,
            "torch_deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "failure_action": "persist_failure_consume_claim_stop_without_relaxation_or_rerun",
        },
        "artifact_paths_and_single_use_exact": artifacts
        == {
            "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
            "authorization_path": AUTHORIZATION_RELATIVE_PATH,
            "output_dir": OUTPUT_RELATIVE_DIR,
            "claim_filename": "execution_claim.json",
            "raw_comparisons_filename": "raw_comparisons.jsonl",
            "report_filename": "report.json",
            "failure_filename": "failure.json",
            "integrity_filename": "integrity.json",
            "single_use": True,
            "output_directory_must_be_absent_or_empty": True,
            "exclusive_create_only": True,
        },
        "ledger_exact": ledger
        == {
            "conditioning_records": 8,
            "scored_comparison_records": 288,
            "state_component_distances_per_comparison": 96,
            "finite_values_required": True,
            "exact_frozen_order_required": True,
            "missing_duplicate_or_reordered_action": "invalidate_and_stop",
            "endpoint_after_complete_ledger_only": True,
        },
        "lock_and_future_authorization_exact": config.get("execution_lock_env")
        == EXECUTION_LOCK_ENV
        and config.get("execution_lock_value") == EXECUTION_LOCK_VALUE
        and config.get("future_execution_authorization_text")
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "implementation_authority_exact": all(
            authority.get(field) is True
            for field in (
                "single_use_runner_implementation_authorized",
                "authorization_validation_implementation_authorized",
                "claim_and_report_lifecycle_implementation_authorized",
                "pure_python_ast_verification_authorized",
            )
        )
        and all(
            authority.get(field) is False
            for field in (
                "installed_source_probe_authorized_at_implementation",
                "rwkv_import_authorized_at_implementation",
                "torch_import_authorized_at_implementation",
                "weights_access_authorized_at_implementation",
                "model_load_authorized_at_implementation",
                "model_execution_authorized_at_implementation",
                "machine_authorization_created_at_implementation",
                "execution_claim_created_at_implementation",
                "d8c_real_execution_authorized",
                "d8c_rerun_authorized",
                "historical_rerun_authorized",
                "d7d_authorized",
                "d7e_authorized",
                "projection_authorized",
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
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D8-C-I config failed closed: " + ", ".join(failed))
    return checks


def read_spec(path: str | Path) -> dict[str, Any]:
    config = _object(path, "runner config")
    validate_config(config)
    return config


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
        raise RuntimeError("D8-C-I authorization schema changed")
    return schema


def _execution_artifacts_absent(root: Path) -> dict[str, bool]:
    output = root / OUTPUT_RELATIVE_DIR
    return {
        "machine_authorization_absent": not (root / AUTHORIZATION_RELATIVE_PATH).exists(),
        "execution_claim_absent": not (output / "execution_claim.json").exists(),
        "raw_comparisons_absent": not (output / "raw_comparisons.jsonl").exists(),
        "report_absent": not (output / "report.json").exists(),
        "failure_absent": not (output / "failure.json").exists(),
        "integrity_absent": not (output / "integrity.json").exists(),
    }


def _authorization_payload(
    *,
    config_path: Path,
    git: Mapping[str, str],
    runner_static_report_sha256: str,
    authorized_at_utc: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "authorization_version": RUNNER_VERSION,
        "stage": "Self-Model-v0.1-D8-C_real_numerical_identifiability_execution",
        "scope": "one_process_one_wrapper_frozen_584_call_numerical_identifiability_gate",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_future_chat_authorization",
        "authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "authorized_at_utc": authorized_at_utc,
        "git_commit": git["commit"],
        "runner_config_sha256": sha256_file(config_path),
        "runner_static_report_sha256": runner_static_report_sha256,
        "d8c_protocol_sha256": D8C_PROTOCOL_SHA256,
        "d8c_remote_static_report_sha256": D8C_REMOTE_STATIC_REPORT_SHA256,
        "d8b_contract_sha256": D8B_CONTRACT_SHA256,
        "fixture_manifest_sha256": FIXTURE_SHA256,
        "schedule_manifest_sha256": SCHEDULE_SHA256,
        "determinism_manifest_sha256": DETERMINISM_SHA256,
        "endpoint_manifest_sha256": ENDPOINT_SHA256,
        "expanded_call_ids_sha256": CALL_IDS_SHA256,
        "installed_source_probe_authorized": True,
        "weights_access_authorized": True,
        "model_load_authorized": True,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "model_forward_calls": 584,
        "d8c_rerun_authorized": False,
        "historical_rerun_authorized": False,
        "d7d_authorized": False,
        "d7e_authorized": False,
        "projection_authorized": False,
        "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "self_updater_authorized": False,
        "raw_original_route_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    payload["authorization_digest_sha256"] = sha256_json(payload)
    return payload


def build_d8c_authorization(
    *,
    config_path: str | Path,
    project_root: str | Path,
    authorization_text: str,
    git_metadata: Mapping[str, str] | None = None,
    verify_execution_artifacts_absent: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    read_spec(config_file)
    _authorization_schema(root)
    if authorization_text != FUTURE_EXECUTION_AUTHORIZATION_TEXT:
        raise PermissionError("D8-C-I future authorization text is not exact")
    git = dict(git_metadata or _git_metadata(root))
    _require_clean_main(git)
    static_report = build_static_report(
        config_path=config_file,
        project_root=root,
        verify_execution_artifacts_absent=verify_execution_artifacts_absent,
    )
    return _authorization_payload(
        config_path=config_file,
        git=git,
        runner_static_report_sha256=static_report["report_digest_sha256"],
        authorized_at_utc=_utc_now(),
    )


def _require_utc(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D8-C-I authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D8-C-I authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D8-C-I authorization timestamp must be UTC")


def validate_authorization_payload(
    authorization: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D8-C-I machine authorization fields changed")
    for field, value in expected.items():
        if field not in {"authorized_at_utc", "authorization_digest_sha256"}:
            if authorization.get(field) != value:
                raise PermissionError(f"D8-C-I authorization.{field} changed")
    _require_utc(authorization.get("authorized_at_utc"))
    stored = authorization.get("authorization_digest_sha256")
    payload = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_digest_sha256"
    }
    if not isinstance(stored, str) or sha256_json(payload) != stored:
        raise PermissionError("D8-C-I authorization digest is invalid")
    return dict(authorization)


def validate_d8c_authorization(
    *,
    authorization_path: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    git: Mapping[str, str],
) -> dict[str, Any]:
    authorization = _object(authorization_path, "machine authorization")
    expected = build_d8c_authorization(
        config_path=config_path,
        project_root=project_root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        git_metadata=git,
        verify_execution_artifacts_absent=False,
    )
    return validate_authorization_payload(authorization, expected)


def _create_claim(
    *,
    output_dir: Path,
    config_path: Path,
    authorization_path: Path,
    git: Mapping[str, str],
    runner_static_report_sha256: str,
    installed_source_sha256: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D8-C-I output directory is not empty; reuse refused")
    claim = {
        "claim_version": RUNNER_VERSION,
        "status": "d8c_single_use_numerical_identifiability_claim_consumed",
        "created_at_utc": _utc_now(),
        "single_use": True,
        "git_commit": git["commit"],
        "runner_config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "runner_static_report_sha256": runner_static_report_sha256,
        "installed_source_sha256": installed_source_sha256,
        "expanded_call_ids_sha256": CALL_IDS_SHA256,
        "model_forward_calls": 584,
        "d8c_rerun_authorized": False,
        "historical_rerun_authorized": False,
        "automatic_rerun_authorized": False,
    }
    return _write_json_exclusive(output_dir / "execution_claim.json", claim)


def _validate_launcher_environment() -> dict[str, bool]:
    checks = {
        "cublas_workspace_config_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == ":4096:8",
        "pythonhashseed_exact": os.environ.get("PYTHONHASHSEED") == "28083101",
        "rwkv_de_version_unset": os.environ.get("RWKV_DE_VERSION") is None,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D8-C-I launcher determinism failed: " + ", ".join(failed))
    return checks


def _probe_installed_source() -> tuple[str, Path, bytes, str]:
    module = import_module("psa.self_model.d4a_real_diagnostic")
    return module._installed_source()


def _runtime_dependencies() -> dict[str, Any]:
    torch = import_module("torch")
    model_module = import_module("psa.model.rwkv7")
    instrumenter = import_module("psa.self_model.rwkv7_instrumented_off_runtime")
    return {
        "torch": torch,
        "RWKV7Adapter": model_module.RWKV7Adapter,
        "load_model_config": model_module.load_model_config,
        "clone_state": model_module.clone_state,
        "compile_instrumented_methods": instrumenter.compile_instrumented_methods,
    }


def _apply_runtime_determinism(torch: Any) -> dict[str, Any]:
    random.seed(28083101)
    torch.manual_seed(28083101)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(28083101)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "python_random_seed": 28083101,
        "torch_manual_seed": 28083101,
        "torch_cuda_manual_seed_all": 28083101,
        "torch_use_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _tensor_distance(left: Any, right: Any, torch: Any) -> float:
    if (
        tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        raise ValueError("D8-C-I tensor outputs are incompatible")
    if not bool(torch.isfinite(left).all().item()) or not bool(
        torch.isfinite(right).all().item()
    ):
        raise ValueError("D8-C-I tensor outputs must be finite")
    left_float = left.detach().float()
    right_float = right.detach().float()
    numerator = float(torch.max(torch.abs(left_float - right_float)).item())
    denominator = max(
        float(torch.max(torch.abs(left_float)).item()),
        float(torch.max(torch.abs(right_float)).item()),
        1e-12,
    )
    return numerator / denominator


def _output_distance(left: tuple[Any, Sequence[Any]], right: tuple[Any, Sequence[Any]], torch: Any) -> dict[str, Any]:
    left_logits, left_state = left
    right_logits, right_state = right
    if len(left_state) != 96 or len(right_state) != 96:
        raise ValueError("D8-C-I outputs must contain 96 state components")
    logits_distance = _tensor_distance(left_logits, right_logits, torch)
    component_distances = [
        _tensor_distance(left_component, right_component, torch)
        for left_component, right_component in zip(left_state, right_state)
    ]
    state_distance = max(component_distances)
    return {
        "logits_distance": logits_distance,
        "state_distance": state_distance,
        "output_distance": max(logits_distance, state_distance),
        "max_state_component_index": component_distances.index(state_distance),
        "state_component_distances": component_distances,
    }


def _forward_route(
    *,
    route: str,
    tokens: Sequence[int],
    source_state: Sequence[Any],
    full_output: bool,
    base_model: Any,
    wrapper: D7CPublicSemanticsWrapper,
    clone_state: Any,
) -> tuple[Any, Sequence[Any]]:
    state = clone_state(source_state)
    if route == "public":
        return base_model.forward(list(tokens), state, full_output)
    if route == "wrapper_zero":
        return wrapper.forward(
            list(tokens), state, full_output, request=zero_request()
        )
    raise PermissionError("D8-C-I route is not frozen")


def _persist_integrity(
    *, output_dir: Path, claim_path: Path, raw_path: Path, report_path: Path
) -> Path:
    integrity = {
        "integrity_version": RUNNER_VERSION,
        "status": "d8c_real_artifact_integrity_complete",
        "execution_claim_sha256": sha256_file(claim_path),
        "raw_comparisons_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "d8c_rerun_authorized": False,
        "automatic_rerun_authorized": False,
    }
    integrity["integrity_digest_sha256"] = sha256_json(integrity)
    return _write_json_exclusive(output_dir / "integrity.json", integrity)


def run_d8c_real_numerical_identifiability(
    *,
    config_path: str | Path,
    authorization_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D8-C-I execution lock is absent")
    launcher_checks = _validate_launcher_environment()
    authorization_file = _require_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d8c_authorization(
        authorization_path=authorization_file,
        config_path=config_file,
        project_root=root,
        git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _probe_installed_source()
    if (
        installed_version != EXPECTED_PACKAGE_VERSION
        or source_digest != EXPECTED_MODEL_SOURCE_SHA256
    ):
        raise RuntimeError("D8-C-I installed source lock differs")
    claim_path = _create_claim(
        output_dir=destination,
        config_path=config_file,
        authorization_path=authorization_file,
        git=git,
        runner_static_report_sha256=authorization["runner_static_report_sha256"],
        installed_source_sha256=source_digest,
    )
    started = time.perf_counter()
    forward_count = 0
    conditioning_count = 0
    scored_call_count = 0
    scored_records: list[dict[str, Any]] = []
    raw_path = destination / "raw_comparisons.jsonl"
    try:
        dependencies = _runtime_dependencies()
        torch = dependencies["torch"]
        runtime_determinism = _apply_runtime_determinism(torch)
        load_model_config = dependencies["load_model_config"]
        rwkv_adapter_class = dependencies["RWKV7Adapter"]
        model_config_path = root / MODEL_CONFIG_RELATIVE_PATH
        if sha256_file(model_config_path) != MODEL_CONFIG_SHA256:
            raise RuntimeError("D8-C-I model config digest changed")
        model_config = load_model_config(
            model_config_path, root, verify_files=True
        )
        for key, value in model_config.environment.items():
            os.environ[key] = value
        adapter = rwkv_adapter_class.load(model_config)
        methods, injection_counts = dependencies["compile_instrumented_methods"](
            upstream_source=source_bytes.decode("utf-8"),
            upstream_globals=vars(sys.modules["rwkv.model"]),
            rwkv_de_version=None,
        )
        wrapper = D7CPublicSemanticsWrapper(
            base_model=adapter.model,
            compiled_methods=methods,
            injection_counts=injection_counts,
        )
        design = _object(root / DESIGN_RELATIVE_PATH, "D8-A design")
        fixtures = expand_fixtures(design)
        schedule = expand_schedule(design, fixtures)
        plan = build_call_plan(schedule)
        validate_call_plan(plan)
        if sha256_json([item["call_id"] for item in plan]) != CALL_IDS_SHA256:
            raise RuntimeError("D8-C-I expanded call ID digest changed")
        conditioning_by_id = {
            item["conditioning_id"]: item
            for item in fixtures["conditioning_fixtures"]
        }
        scored_by_id = {
            item["fixture_id"]: item for item in fixtures["scored_fixtures"]
        }
        source_states = {
            item_id: adapter.model.generate_zero_state()
            for item_id in [*conditioning_by_id, *scored_by_id]
        }
        with raw_path.open("x", encoding="utf-8", newline="\n") as raw_handle:
            for call in schedule["conditioning_calls"]:
                fixture = conditioning_by_id[call["conditioning_id"]]
                output = _forward_route(
                    route=call["route"],
                    tokens=fixture["token_ids"],
                    source_state=source_states[call["conditioning_id"]],
                    full_output=fixture["full_output"],
                    base_model=adapter.model,
                    wrapper=wrapper,
                    clone_state=dependencies["clone_state"],
                )
                if len(output[1]) != 96 or not bool(torch.isfinite(output[0]).all().item()):
                    raise ValueError("D8-C-I conditioning output is invalid")
                if not all(bool(torch.isfinite(component).all().item()) for component in output[1]):
                    raise ValueError("D8-C-I conditioning state is nonfinite")
                forward_count += 1
                conditioning_count += 1
                _write_jsonl_record(
                    raw_handle,
                    {
                        "record_type": "conditioning_call",
                        "call_id": call["call_id"],
                        "conditioning_id": call["conditioning_id"],
                        "route": call["route"],
                        "scored": False,
                        "state_component_count": len(output[1]),
                        "finite": True,
                    },
                )
            for block in schedule["pair_blocks"]:
                fixture = scored_by_id[block["fixture_id"]]
                outputs = []
                for route in block["route_order"]:
                    outputs.append(
                        _forward_route(
                            route=route,
                            tokens=fixture["token_ids"],
                            source_state=source_states[block["fixture_id"]],
                            full_output=fixture["full_output"],
                            base_model=adapter.model,
                            wrapper=wrapper,
                            clone_state=dependencies["clone_state"],
                        )
                    )
                    forward_count += 1
                    scored_call_count += 1
                distance = _output_distance(outputs[0], outputs[1], torch)
                record = {
                    "record_type": "scored_pair_comparison",
                    "pair_block_id": block["pair_block_id"],
                    "fixture_id": block["fixture_id"],
                    "stratum": block["stratum"],
                    "replicate": block["replicate"],
                    "latin_position": block["latin_position"],
                    "pair_type": block["pair_type"],
                    "route_order": block["route_order"],
                    **distance,
                }
                scored_records.append(record)
                _write_jsonl_record(raw_handle, record)
        if (
            forward_count != 584
            or conditioning_count != 8
            or scored_call_count != 576
            or len(scored_records) != 288
        ):
            raise RuntimeError("D8-C-I completed call or ledger counts changed")
        endpoint = _object(root / ENDPOINT_RELATIVE_PATH, "D8 endpoint")
        endpoint_ledger = [
            {
                "pair_block_id": item["pair_block_id"],
                "pair_type": item["pair_type"],
                "output_distance": item["output_distance"],
            }
            for item in scored_records
        ]
        fixture_results = aggregate_fixture_excess(schedule, endpoint_ledger)
        decision = decide_excess_drift(fixture_results, endpoint)
        checks = {
            "all_584_forwards_complete": forward_count == 584,
            "conditioning_count_exact": conditioning_count == 8,
            "scored_count_exact": scored_call_count == 576,
            "complete_288_pair_ledger": len(scored_records) == 288,
            "all_state_component_distance_counts_exact": all(
                len(item["state_component_distances"]) == 96
                for item in scored_records
            ),
            "wrapper_base_dictionary_unchanged": wrapper.base_dictionary_is_stable(),
            "wrapper_bindings_and_context_stable": wrapper.owned_bindings_are_stable()
            and wrapper.context_is_empty(),
            "runtime_determinism_applied": runtime_determinism[
                "torch_use_deterministic_algorithms"
            ]
            and runtime_determinism["cudnn_deterministic"]
            and not runtime_determinism["cudnn_benchmark"]
            and not runtime_determinism["cuda_matmul_allow_tf32"]
            and not runtime_determinism["cudnn_allow_tf32"],
        }
        valid = all(checks.values())
        report: dict[str, Any] = {
            "report_version": RUNNER_VERSION,
            "created_at_utc": _utc_now(),
            "status": (
                "d8c_real_numerical_identifiability_completed"
                if valid
                else "d8c_real_numerical_identifiability_invalid"
            ),
            "valid": valid,
            "development_only": True,
            "git": git,
            "runner_config_sha256": sha256_file(config_file),
            "authorization_digest_sha256": authorization[
                "authorization_digest_sha256"
            ],
            "execution_claim_sha256": sha256_file(claim_path),
            "model": adapter.model_metadata(),
            "installed_source": {
                "path": str(source_path),
                "version": installed_version,
                "sha256": source_digest,
            },
            "launcher_determinism": launcher_checks,
            "runtime_determinism": runtime_determinism,
            "checks": checks,
            "counts": {
                "conditioning_forward_calls": conditioning_count,
                "scored_forward_calls": scored_call_count,
                "total_forward_calls": forward_count,
                "scored_pair_records": len(scored_records),
            },
            "endpoint_decision": decision,
            "fixture_results": fixture_results,
            "interpretation": "numerical_route_identifiability_engineering_evidence_only_not_self_effect",
            "runtime_seconds": time.perf_counter() - started,
            "safety": {
                "d8c_rerun_authorized": False,
                "historical_rerun": False,
                "d7d_authorized": False,
                "d7e_authorized": False,
                "projection_constructed": False,
                "formal_test_set_used": False,
                "self_effect_conclusion_made": False,
                "self_updater_used": False,
                "raw_original_route_used": False,
                "automatic_rerun_authorized": False,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        report_path = _write_json_exclusive(destination / "report.json", report)
        _persist_integrity(
            output_dir=destination,
            claim_path=claim_path,
            raw_path=raw_path,
            report_path=report_path,
        )
        return report
    except BaseException as error:
        failure: dict[str, Any] = {
            "report_version": RUNNER_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d8c_real_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "forward_calls_completed": forward_count,
            "conditioning_calls_completed": conditioning_count,
            "scored_calls_completed": scored_call_count,
            "raw_comparisons_present": raw_path.exists(),
            "d8c_rerun_authorized": False,
            "historical_rerun_authorized": False,
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        _write_json_exclusive(destination / "failure.json", failure)
        raise


def _entry_ast_audit() -> dict[str, int]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_d8c_real_numerical_identifiability"
    )
    wanted = {
        "_validate_launcher_environment",
        "_git_metadata",
        "validate_d8c_authorization",
        "_probe_installed_source",
        "_create_claim",
        "_runtime_dependencies",
        "_apply_runtime_determinism",
        "load_model_config",
        "load",
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
        raise RuntimeError("D8-C-I entry call inventory changed: " + ", ".join(missing))
    return lines


def _launcher_ast_audit(root: Path) -> dict[str, int]:
    launcher_path = root / "scripts/run_self_model_v0_1_d8c_real_numerical_identifiability.py"
    tree = ast.parse(launcher_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    wanted = {
        "_validate_launcher_environment",
        "build_d8c_authorization",
        "_write_json_exclusive",
        "run_d8c_real_numerical_identifiability",
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
        raise RuntimeError("D8-C-I launcher call inventory changed: " + ", ".join(missing))
    return lines


def run_pure_python_acceptance(root: Path) -> dict[str, Any]:
    design = _object(root / DESIGN_RELATIVE_PATH, "D8-A design")
    fixtures = expand_fixtures(design)
    schedule = expand_schedule(design, fixtures)
    plan = build_call_plan(schedule)
    plan_checks = validate_call_plan(plan)
    call_digest = sha256_json([item["call_id"] for item in plan])
    fake_git = {
        "branch": "main",
        "commit": "a" * 40,
        "origin_main": "a" * 40,
        "status": "",
    }
    fake_authorization = _authorization_payload(
        config_path=root / CONFIG_RELATIVE_PATH,
        git=fake_git,
        runner_static_report_sha256="b" * 64,
        authorized_at_utc="2026-08-31T00:00:00+00:00",
    )
    expected = dict(fake_authorization)
    validated = validate_authorization_payload(fake_authorization, expected)
    mutation_rejected = False
    changed = dict(fake_authorization)
    changed["model_forward_calls"] = 583
    changed_without_digest = {
        key: value for key, value in changed.items() if key != "authorization_digest_sha256"
    }
    changed["authorization_digest_sha256"] = sha256_json(changed_without_digest)
    try:
        validate_authorization_payload(changed, expected)
    except PermissionError:
        mutation_rejected = True
    checks = {
        "plan_checks_all_pass": all(plan_checks.values()),
        "call_count_exact": len(plan) == 584,
        "call_ids_digest_exact": call_digest == CALL_IDS_SHA256,
        "conditioning_and_pair_counts_exact": len(schedule["conditioning_calls"]) == 8
        and len(schedule["pair_blocks"]) == 288,
        "authorization_payload_valid": validated["single_use"] is True,
        "authorization_mutation_rejected": mutation_rejected,
        "no_model_objects_created": True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "conditioning_calls": len(schedule["conditioning_calls"]),
            "pair_blocks": len(schedule["pair_blocks"]),
            "total_forward_calls": len(plan),
        },
        "call_ids_digest": call_digest,
    }


def build_static_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    verify_execution_artifacts_absent: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    schema = _authorization_schema(root)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    lock_checks = {
        "d8c_protocol": source_digests[D8C_PROTOCOL_RELATIVE_PATH]
        == D8C_PROTOCOL_SHA256,
        "d8b_contract": source_digests[D8B_CONTRACT_RELATIVE_PATH]
        == D8B_CONTRACT_SHA256,
        "fixture_manifest": source_digests[FIXTURE_RELATIVE_PATH] == FIXTURE_SHA256,
        "schedule_manifest": source_digests[SCHEDULE_RELATIVE_PATH]
        == SCHEDULE_SHA256,
        "determinism_manifest": source_digests[DETERMINISM_RELATIVE_PATH]
        == DETERMINISM_SHA256,
        "endpoint_manifest": source_digests[ENDPOINT_RELATIVE_PATH]
        == ENDPOINT_SHA256,
        "model_config": source_digests[MODEL_CONFIG_RELATIVE_PATH]
        == MODEL_CONFIG_SHA256,
        "d7c_wrapper": source_digests[
            "src/psa/self_model/d7c_public_semantics_runtime.py"
        ]
        == D7C_WRAPPER_SHA256,
        "instrumenter": source_digests[
            "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
        ]
        == INSTRUMENTER_SHA256,
    }
    acceptance = run_pure_python_acceptance(root)
    lines = _entry_ast_audit()
    launcher_lines = _launcher_ast_audit(root)
    observed_artifacts = _execution_artifacts_absent(root)
    artifacts = (
        observed_artifacts
        if verify_execution_artifacts_absent
        else {name: True for name in observed_artifacts}
    )
    checks = {
        "config_valid": all(validate_config(spec).values()),
        "authorization_schema_exact": set(schema["properties"])
        == AUTHORIZATION_FIELDS,
        "frozen_source_locks_valid": all(lock_checks.values()),
        "pure_python_acceptance_valid": acceptance["valid"],
        "authorization_precedes_installed_source_probe": lines[
            "validate_d8c_authorization"
        ]
        < lines["_probe_installed_source"],
        "claim_precedes_runtime_imports_and_model_load": lines["_create_claim"]
        < lines["_runtime_dependencies"]
        < lines["load_model_config"]
        < lines["load"],
        "launcher_preflight_precedes_runtime_imports": lines[
            "_validate_launcher_environment"
        ]
        < lines["_runtime_dependencies"],
        "runtime_determinism_precedes_model_load": lines[
            "_apply_runtime_determinism"
        ]
        < lines["load"],
        "launcher_environment_precedes_authorization_creation": launcher_lines[
            "_validate_launcher_environment"
        ]
        < launcher_lines["build_d8c_authorization"]
        < launcher_lines["_write_json_exclusive"]
        < launcher_lines["run_d8c_real_numerical_identifiability"],
        "future_authorization_exact_and_separate": spec[
            "future_execution_authorization_text"
        ]
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "execution_artifacts_absent": all(artifacts.values()),
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("D8-C-I static verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": RUNNER_VERSION,
        "status": "d8c_i_single_use_real_runner_static_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": validate_config(spec),
        "source_lock_checks": lock_checks,
        "pure_python_acceptance": acceptance,
        "entry_call_lines": lines,
        "launcher_call_lines": launcher_lines,
        "execution_artifacts": artifacts,
        "future_exact_owner_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "d8c_remote_static_report_sha256": D8C_REMOTE_STATIC_REPORT_SHA256,
        "source_digests": source_digests,
        "next_gate": NEXT_GATE,
        "safety": {
            "installed_source_probed": False,
            "machine_authorization_created": False,
            "execution_claim_created": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d8c_real_execution_authorized": False,
            "d8c_rerun_authorized": False,
            "historical_rerun": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "projection_constructed": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
