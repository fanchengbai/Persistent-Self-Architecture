from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
from importlib import import_module
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.self_model.d7c_public_semantics_runtime import (
    D7CPublicSemanticsWrapper,
    FULL_OUTPUT_VALUES,
    N_LAYER,
    SEQUENCE_TOKEN_IDS,
    SINGLE_TOKEN_IDS,
    STATE_COMPONENTS,
    STATE_INPUTS,
    TARGET_LAYER_INDEX,
    TARGET_LAYER_RULE_ID,
    active_request,
    compatibility_cells,
    run_synthetic_compatibility_acceptance,
    zero_request,
)


REPORT_VERSION = "0.1-self-model-d7c-real-public-semantics-compatibility"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d7c_real_public_semantics_compatibility.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d7c_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "results/authorizations/self_model_v0_1_d7_compatibility_v01.json"
)
OUTPUT_RELATIVE_DIR = "results/development/self_model_v0_1_d7_compatibility_v01"
EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D7C_REAL_COMPATIBILITY"
EXECUTION_LOCK_VALUE = "AUTHORIZED_D7C_REAL_2_9B_PUBLIC_SEMANTICS_COMPATIBILITY_ONCE"
MODEL_CONFIG_RELATIVE_PATH = "configs/models/rwkv7_g1h_2.9b.candidate.json"
MODEL_CONFIG_SHA256 = "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
EXPECTED_PACKAGE_VERSION = "0.8.32"
EXPECTED_MODEL_SOURCE_SHA256 = (
    "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
)
D7_DESIGN_SHA256 = "94687cc07f06a72e784e21338b554cb1b57fadeb35f7a052eb02bb1b580bb647"
D7B_CONTRACT_SHA256 = "2bac0cce49673c1252478bd1a832d7a5b991e02903a7f9a77d0037a5cc1c66d3"
D7B_REMOTE_REPORT_SHA256 = (
    "9800d221f13b3351ffb175815fa02e9c0659c3c6f6471ca8b76424a3a59ff683"
)
CALIBRATION_MANIFEST_SHA256 = (
    "f655629c736bb18b7a4de5f0493975b0c2e8f48e63217d7fe471493853353e24"
)
HELDOUT_MANIFEST_SHA256 = (
    "8af9c69c96330e067fb6502bd798160585842e79a8cd9a52c1d101a4a13e46ed"
)
IMPLEMENTATION_CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 D7-C 真实public语义兼容门设计与无模型安全入口实现；只允许冻结"
    "与held-out payload无关的固定synthetic token fixture、8个兼容cell（forward_one/"
    "forward_seq × state=None/prebuilt × full_output=false/true）、每cell各1次public OFF与"
    "wrapper zero共16次调用，以及2次synthetic active调用的18-call计划；允许定义独立的目标层"
    "选择规则、public zero-state初始化、logits/state逐项torch.equal、32层callback计数、目标层"
    "单次应用、基础实例字典不变性和失败即停止标准，并实现新的授权Schema、唯一authorization/"
    "claim/output命名空间及无模型静态入口验证；本轮不探测installed source、不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型、不创建机器授权或claim、不访问calibration/held-out payload，"
    "也不授权D7-C真实执行、D7-D/D7-E、projection实现或构造、D6D重跑、正式测试集、Self效果"
    "结论、Self Updater、raw-original路线或自动重跑。"
)
FUTURE_EXECUTION_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 D7-C 真实2.9B public语义兼容门一次（固定8个public OFF/"
    "wrapper zero等价cell共16次调用，加2次synthetic active，共18次forward；不访问"
    "calibration/held-out payload），并授权观察本次兼容结果；不授权D7-C重跑、自动重跑、"
    "D7-D/D7-E、projection实现或构造、D6D重跑、正式测试集、Self效果结论、Self Updater或"
    "raw-original路线。"
)
CLASSIFICATION = (
    "d7c_real_public_semantics_compatibility_design_and_single_use_entry_"
    "static_verified_execution_not_authorized"
)
NEXT_GATE = (
    "remote_no_model_d7c_static_verification_then_separate_exact_execution_authorization"
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
    "config_sha256",
    "entry_static_report_sha256",
    "d7b_remote_report_sha256",
    "installed_source_probe_authorized",
    "weights_access_authorized",
    "model_load_authorized",
    "model_execution_authorized",
    "result_observation_authorized",
    "compatibility_only",
    "model_forward_calls",
    "heldout_payload_accessed",
    "target_layer_rule_id",
    "d7c_rerun_authorized",
    "d7d_authorized",
    "d7e_authorized",
    "projection_implementation_authorized",
    "projection_construction_authorized",
    "d6d_rerun_authorized",
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
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "configs/development/self_model_v0_1_d7b_manifest_runtime_contract.json",
    "configs/development/self_model_v0_1_d7_projection_training_manifest.json",
    "configs/development/self_model_v0_1_d7_heldout_transfer_manifest.json",
    "configs/preregistration/self_model_v0_1_d7_heldout_causal_transfer.draft.json",
    "docs/self_model_v0_1_d7b_remote_observation.md",
    "docs/self_model_v0_1_d7c_real_public_semantics_compatibility.md",
    "scripts/run_self_model_v0_1_d7c_real_public_semantics_compatibility.py",
    "scripts/verify_self_model_v0_1_d7c_real_public_semantics_entry.py",
    "src/psa/self_model/d7c_public_semantics_runtime.py",
    "src/psa/self_model/d7c_real_compatibility_entry.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d7c_public_semantics_runtime.py",
    "tests/test_self_model_d7c_real_compatibility_entry.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D7-C {label} must be an object")
    return value


def _require_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if resolved != (root / relative).resolve():
        raise PermissionError(f"D7-C {label} path is not frozen")
    return resolved


def _git_metadata(root: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--porcelain"),
    }


def _require_clean_main(git: Mapping[str, str]) -> None:
    if git.get("branch") != "main" or git.get("status_porcelain") != "":
        raise PermissionError("D7-C execution requires a clean main worktree")
    commit = git.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise PermissionError("D7-C git commit is invalid")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("frozen_prerequisites", {})
    model = config.get("model", {})
    separation = config.get("payload_separation", {})
    cells = config.get("compatibility_cells")
    active = config.get("active_calls")
    counts = config.get("counts", {})
    public = config.get("public_semantics_contract", {})
    layer = config.get("target_layer_rule", {})
    acceptance = config.get("acceptance", {})
    checks = {
        "identity_exact": config.get("protocol_version") == REPORT_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D7-C_real_public_semantics_compatibility_entry"
        and config.get("status")
        == "design_and_no_model_entry_implemented_execution_not_authorized"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("implementation_confirmation_text")
        == IMPLEMENTATION_CONFIRMATION_TEXT,
        "d7_and_d7b_prerequisites_frozen": prerequisites.get("d7_design_config_sha256")
        == D7_DESIGN_SHA256
        and prerequisites.get("d7b_contract_sha256") == D7B_CONTRACT_SHA256
        and prerequisites.get("d7b_remote_report_sha256") == D7B_REMOTE_REPORT_SHA256
        and prerequisites.get("calibration_manifest_sha256")
        == CALIBRATION_MANIFEST_SHA256
        and prerequisites.get("heldout_manifest_sha256") == HELDOUT_MANIFEST_SHA256
        and prerequisites.get("d6d_fixture_seed_claim_authorization_or_result_reused")
        is False,
        "model_lock_exact": model.get("model_id") == "rwkv7-g1h-2.9b-20260710"
        and model.get("config_path") == MODEL_CONFIG_RELATIVE_PATH
        and model.get("config_sha256") == MODEL_CONFIG_SHA256
        and model.get("expected_package_version") == EXPECTED_PACKAGE_VERSION
        and model.get("expected_model_source_sha256")
        == EXPECTED_MODEL_SOURCE_SHA256
        and model.get("rwkv_de_version") == "unset"
        and model.get("n_layer") == N_LAYER
        and model.get("hidden_dimension") == 2560,
        "payload_is_synthetic_and_separate": separation.get("fixture_kind")
        == "fixed_synthetic_token_protocol_only"
        and separation.get("single_token_ids") == list(SINGLE_TOKEN_IDS)
        and separation.get("sequence_token_ids") == list(SEQUENCE_TOKEN_IDS)
        and all(
            separation.get(field) is False
            for field in (
                "calibration_payload_accessed",
                "heldout_payload_accessed",
                "capability_payload_accessed",
                "formal_test_set_accessed",
            )
        ),
        "eight_cells_exact": isinstance(cells, list)
        and cells == [dict(cell) for cell in compatibility_cells()],
        "two_active_calls_exact": active
        == [
            {
                "call_id": "d7c-active-one",
                "execution_path": "forward_one",
                "state_input": "none",
                "full_output": False,
            },
            {
                "call_id": "d7c-active-seq",
                "execution_path": "forward_seq",
                "state_input": "none",
                "full_output": True,
            },
        ],
        "counts_exact": counts
        == {
            "equivalence_cells": 8,
            "public_off_calls": 8,
            "wrapper_zero_calls": 8,
            "equivalence_forward_calls": 16,
            "synthetic_active_forward_calls": 2,
            "model_forward_calls_total": 18,
            "active_callback_invocations_total": 64,
            "active_target_layer_applications_total": 2,
        },
        "public_semantics_exact": public
        == {
            "wrapper_owned_external_to_base_instance": True,
            "state_none_calls_generate_zero_state_before_child_dispatch": True,
            "prebuilt_state_skips_zero_state_generation": True,
            "single_token_dispatch": "forward_one",
            "sequence_dispatch": "forward_seq",
            "full_output_forwarded_to_sequence": True,
            "base_instance_dictionary_mutation_allowed": False,
            "runtime_model_attribute_switching_allowed": False,
        },
        "independent_target_layer_rule_exact": layer
        == {
            "rule_id": TARGET_LAYER_RULE_ID,
            "derivation": "n_layer_div_2_minus_1",
            "n_layer": N_LAYER,
            "target_layer_index_zero_based": TARGET_LAYER_INDEX,
            "derived_independently_from_d6d_results": True,
            "effect_selected": False,
            "projection_selected": False,
        },
        "acceptance_and_failure_stop_exact": acceptance.get(
            "each_cell_public_off_vs_wrapper_zero_logits_torch_equal"
        )
        is True
        and acceptance.get("each_cell_public_off_vs_wrapper_zero_state_torch_equal")
        is True
        and acceptance.get("each_cell_state_component_inventory_equal") is True
        and acceptance.get("wrapper_none_initializes_once_before_child_dispatch") is True
        and acceptance.get("wrapper_prebuilt_initializes_zero_times") is True
        and acceptance.get("active_callback_invocations_exact") is True
        and acceptance.get("active_target_layer_applications_exact") is True
        and acceptance.get("active_output_differs_from_zero") is True
        and acceptance.get("base_instance_dictionary_unchanged") is True
        and acceptance.get("all_outputs_finite") is True
        and acceptance.get("failure_action")
        == "persist_failure_consume_claim_stop_without_d7d_d7e_or_rerun",
        "paths_and_future_authorization_frozen": config.get("execution_lock_env")
        == EXECUTION_LOCK_ENV
        and config.get("execution_lock_value") == EXECUTION_LOCK_VALUE
        and config.get("authorization_schema_path")
        == AUTHORIZATION_SCHEMA_RELATIVE_PATH
        and config.get("authorization_path") == AUTHORIZATION_RELATIVE_PATH
        and config.get("output_dir") == OUTPUT_RELATIVE_DIR
        and config.get("future_execution_authorization_text")
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "implementation_only_authority_exact": config.get(
            "design_and_entry_implementation_authorized"
        )
        is True
        and config.get("no_model_static_verification_authorized") is True
        and config.get("future_exact_owner_authorization_required") is True
        and config.get("future_machine_authorization_required") is True
        and config.get("single_use_claim_required") is True
        and config.get("claim_precedes_model_config_weights_and_load") is True
        and config.get("unique_output_required") is True,
        "execution_projection_and_later_authority_closed": all(
            config.get(field) is False
            for field in (
                "installed_source_probe_authorized_at_implementation",
                "machine_authorization_created_at_implementation",
                "execution_claim_created_at_implementation",
                "execution_authorized_at_implementation",
                "d7c_rerun_authorized",
                "d7d_authorized",
                "d7e_authorized",
                "projection_implementation_authorized",
                "projection_construction_authorized",
                "d6d_rerun_authorized",
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
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D7-C config failed closed: " + ", ".join(failed))
    return checks


def read_spec(path: str | Path) -> dict[str, Any]:
    spec = _object(path, "config")
    validate_config(spec)
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
        raise RuntimeError("D7-C authorization schema changed")
    return schema


def _execution_artifacts_absent(root: Path) -> dict[str, bool]:
    return {
        "machine_authorization_absent": not (root / AUTHORIZATION_RELATIVE_PATH).exists(),
        "execution_claim_absent": not (
            root / OUTPUT_RELATIVE_DIR / "execution_claim.json"
        ).exists(),
        "output_report_absent": not (root / OUTPUT_RELATIVE_DIR / "report.json").exists(),
        "failure_report_absent": not (root / OUTPUT_RELATIVE_DIR / "failure.json").exists(),
    }


def build_d7c_authorization(
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
        raise PermissionError("D7-C future authorization text is not exact")
    git = dict(git_metadata or _git_metadata(root))
    _require_clean_main(git)
    report = build_d7c_entry_static_report(
        config_path=config_file,
        project_root=root,
        verify_execution_artifacts_absent=verify_execution_artifacts_absent,
    )
    authorization: dict[str, Any] = {
        "authorization_version": REPORT_VERSION,
        "stage": "Self-Model-v0.1-D7-C_real_public_semantics_compatibility",
        "scope": "one_fixed_18_call_public_off_wrapper_zero_and_synthetic_active_compatibility_gate",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_future_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "entry_static_report_sha256": report["report_digest_sha256"],
        "d7b_remote_report_sha256": D7B_REMOTE_REPORT_SHA256,
        "installed_source_probe_authorized": True,
        "weights_access_authorized": True,
        "model_load_authorized": True,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "compatibility_only": True,
        "model_forward_calls": 18,
        "heldout_payload_accessed": False,
        "target_layer_rule_id": TARGET_LAYER_RULE_ID,
        "d7c_rerun_authorized": False,
        "d7d_authorized": False,
        "d7e_authorized": False,
        "projection_implementation_authorized": False,
        "projection_construction_authorized": False,
        "d6d_rerun_authorized": False,
        "formal_test_set_authorized": False,
        "self_effect_conclusion_authorized": False,
        "self_updater_authorized": False,
        "raw_original_route_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def _require_utc(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D7-C authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D7-C authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D7-C authorization timestamp must be UTC")


def validate_d7c_authorization(
    *,
    authorization_path: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    git: Mapping[str, str],
) -> dict[str, Any]:
    authorization = _object(authorization_path, "machine authorization")
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D7-C machine authorization fields changed")
    expected = build_d7c_authorization(
        config_path=config_path,
        project_root=project_root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        git_metadata=git,
        verify_execution_artifacts_absent=False,
    )
    for field, value in expected.items():
        if field not in {"authorized_at_utc", "authorization_digest_sha256"}:
            if authorization.get(field) != value:
                raise PermissionError(f"D7-C authorization.{field} changed")
    _require_utc(authorization.get("authorized_at_utc"))
    stored = authorization.get("authorization_digest_sha256")
    payload = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_digest_sha256"
    }
    if not isinstance(stored, str) or sha256_json(payload) != stored:
        raise PermissionError("D7-C authorization digest is invalid")
    return authorization


def _create_claim(
    *,
    output_dir: Path,
    config_path: Path,
    authorization_path: Path,
    git: Mapping[str, str],
    entry_static_report_sha256: str,
    installed_source_sha256: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D7-C output directory is not empty; reuse refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d7c_single_use_compatibility_execution_claim_consumed",
        "created_at_utc": _utc_now(),
        "single_use": True,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "entry_static_report_sha256": entry_static_report_sha256,
        "installed_source_sha256": installed_source_sha256,
        "model_forward_calls": 18,
        "heldout_payload_accessed": False,
        "d7c_rerun_authorized": False,
        "d7d_authorized": False,
        "d7e_authorized": False,
        "automatic_rerun_authorized": False,
    }
    return _write_json_exclusive(output_dir / "execution_claim.json", claim)


def _probe_installed_source() -> tuple[str, Path, bytes, str]:
    module = import_module("psa.self_model.d4a_real_diagnostic")
    return module._installed_source()


def _runtime_dependencies() -> dict[str, Any]:
    model_module = import_module("psa.model.rwkv7")
    instrumenter = import_module("psa.self_model.rwkv7_instrumented_off_runtime")
    return {
        "RWKV7Adapter": model_module.RWKV7Adapter,
        "load_model_config": model_module.load_model_config,
        "clone_state": model_module.clone_state,
        "compare_tensors": model_module.compare_tensors,
        "compare_states": model_module.compare_states,
        "inventory_state": model_module.inventory_state,
        "compile_instrumented_methods": instrumenter.compile_instrumented_methods,
    }


class D7CTorchSyntheticProbe:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.invocation_count = 0
        self.application_count = 0
        self.layer_counts = {layer: 0 for layer in range(N_LAYER)}

    def __call__(self, **payload: Any) -> Any:
        residual = payload["residual_x"]
        layer_index = int(payload["layer_index"])
        self.invocation_count += 1
        self.layer_counts[layer_index] += 1
        if layer_index != TARGET_LAYER_INDEX:
            return residual
        self.application_count += 1
        delta = self.torch.zeros_like(residual)
        flattened = delta.reshape(-1)
        rms = self.torch.sqrt(self.torch.mean(residual.detach().float().square()))
        magnitude = rms * 0.01
        if not bool(self.torch.isfinite(magnitude).item()) or float(magnitude.item()) == 0.0:
            magnitude = self.torch.tensor(0.01, device=residual.device)
        flattened[0] = magnitude.to(dtype=residual.dtype)
        return residual + delta


def run_d7c_real_compatibility(
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
        raise PermissionError("the exact single-use D7-C execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D7-C")
    authorization_file = _require_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_path(root, output_dir, OUTPUT_RELATIVE_DIR, "output")
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d7c_authorization(
        authorization_path=authorization_file,
        config_path=config_file,
        project_root=root,
        git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _probe_installed_source()
    if installed_version != EXPECTED_PACKAGE_VERSION or source_digest != EXPECTED_MODEL_SOURCE_SHA256:
        raise RuntimeError("D7-C installed source lock differs")
    claim_path = _create_claim(
        output_dir=destination,
        config_path=config_file,
        authorization_path=authorization_file,
        git=git,
        entry_static_report_sha256=authorization["entry_static_report_sha256"],
        installed_source_sha256=source_digest,
    )
    started = time.perf_counter()
    try:
        dependencies = _runtime_dependencies()
        model_config_path = root / MODEL_CONFIG_RELATIVE_PATH
        if sha256_file(model_config_path) != MODEL_CONFIG_SHA256:
            raise RuntimeError("D7-C model config digest changed")
        model_config = dependencies["load_model_config"](
            model_config_path, root, verify_files=True
        )
        for key, value in model_config.environment.items():
            os.environ[key] = value
        adapter = dependencies["RWKV7Adapter"].load(model_config)
        torch = adapter.torch
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
        cells = []
        wrapper_zero_logits: dict[tuple[str, str, bool], Any] = {}
        for cell in compatibility_cells():
            tokens = (
                SINGLE_TOKEN_IDS
                if cell["execution_path"] == "forward_one"
                else SEQUENCE_TOKEN_IDS
            )
            source_state = (
                None
                if cell["state_input"] == "none"
                else adapter.model.generate_zero_state()
            )
            public_logits, public_state = adapter.model.forward(
                list(tokens),
                dependencies["clone_state"](source_state),
                cell["full_output"],
            )
            before_initializations = wrapper.zero_state_initialization_count
            wrapper_logits, wrapper_state = wrapper.forward(
                list(tokens),
                dependencies["clone_state"](source_state),
                cell["full_output"],
                request=zero_request(),
            )
            wrapper_zero_logits[
                (cell["execution_path"], cell["state_input"], cell["full_output"])
            ] = dependencies["clone_state"](wrapper_logits)
            logits_comparison = dependencies["compare_tensors"](
                public_logits, wrapper_logits, torch
            )
            state_comparison = dependencies["compare_states"](
                public_state, wrapper_state, torch
            )
            public_inventory = dependencies["inventory_state"](public_state, torch)
            wrapper_inventory = dependencies["inventory_state"](wrapper_state, torch)
            cells.append(
                {
                    **cell,
                    "logits": logits_comparison,
                    "state": state_comparison,
                    "state_component_inventory_equal": public_inventory["components"]
                    == wrapper_inventory["components"],
                    "wrapper_zero_state_initializations": wrapper.zero_state_initialization_count
                    - before_initializations,
                }
            )
        probe = D7CTorchSyntheticProbe(torch)
        active_reports = []
        for call in spec["active_calls"]:
            tokens = (
                SINGLE_TOKEN_IDS
                if call["execution_path"] == "forward_one"
                else SEQUENCE_TOKEN_IDS
            )
            zero_logits = wrapper_zero_logits[
                (call["execution_path"], "none", call["full_output"])
            ]
            active_logits, active_state = wrapper.forward(
                list(tokens), None, call["full_output"], request=active_request(probe)
            )
            active_reports.append(
                {
                    **call,
                    "active_differs_from_zero": not bool(
                        torch.equal(zero_logits, active_logits)
                    ),
                    "state_inventory": dependencies["inventory_state"](
                        active_state, torch
                    ),
                }
            )
        checks = {
            "eight_cells_complete": len(cells) == 8,
            "all_logits_exact": all(cell["logits"]["exact"] for cell in cells),
            "all_states_exact": all(cell["state"]["exact"] for cell in cells),
            "all_state_inventories_equal": all(
                cell["state_component_inventory_equal"] for cell in cells
            ),
            "none_and_prebuilt_initialization_counts_exact": all(
                cell["wrapper_zero_state_initializations"]
                == (1 if cell["state_input"] == "none" else 0)
                for cell in cells
            ),
            "active_callback_count_exact": probe.invocation_count == 64
            and set(probe.layer_counts.values()) == {2},
            "active_target_application_count_exact": probe.application_count == 2,
            "active_outputs_differ": all(
                item["active_differs_from_zero"] for item in active_reports
            ),
            "base_instance_dictionary_unchanged": wrapper.base_dictionary_is_stable(),
            "wrapper_bindings_and_context_stable": wrapper.owned_bindings_are_stable()
            and wrapper.context_is_empty(),
        }
        valid = all(checks.values())
        report: dict[str, Any] = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": (
                "d7c_real_public_semantics_compatibility_passed"
                if valid
                else "d7c_real_public_semantics_compatibility_failed"
            ),
            "valid": valid,
            "development_only": True,
            "git": git,
            "config_sha256": sha256_file(config_file),
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
            "checks": checks,
            "cells": cells,
            "active_calls": active_reports,
            "counts": {
                "model_forward_calls": 18,
                "public_off_calls": 8,
                "wrapper_zero_calls": 8,
                "synthetic_active_calls": 2,
                "active_callback_invocations": probe.invocation_count,
                "active_target_layer_applications": probe.application_count,
            },
            "interpretation": "public_semantics_compatibility_only_not_projection_or_self_effect",
            "runtime_seconds": time.perf_counter() - started,
            "safety": {
                "heldout_payload_accessed": False,
                "d7c_rerun_authorized": False,
                "d7d_authorized": False,
                "d7e_authorized": False,
                "projection_implemented": False,
                "projection_constructed": False,
                "d6d_rerun": False,
                "formal_test_set_used": False,
                "self_effect_conclusion_made": False,
                "self_updater_used": False,
                "raw_original_route_used": False,
                "automatic_rerun_authorized": False,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        _write_json_exclusive(destination / "report.json", report)
        return report
    except BaseException as error:
        failure: dict[str, Any] = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d7c_real_compatibility_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "d7c_rerun_authorized": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
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
        and node.name == "run_d7c_real_compatibility"
    )
    wanted = {
        "_git_metadata",
        "validate_d7c_authorization",
        "_probe_installed_source",
        "_create_claim",
        "_runtime_dependencies",
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
        raise RuntimeError("D7-C entry call inventory changed")
    return lines


def build_d7c_entry_static_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    verify_execution_artifacts_absent: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_path(root, config_path, CONFIG_RELATIVE_PATH, "config")
    spec = read_spec(config_file)
    schema = _authorization_schema(root)
    acceptance = run_synthetic_compatibility_acceptance()
    lines = _entry_ast_audit()
    observed_artifacts = _execution_artifacts_absent(root)
    artifacts = (
        observed_artifacts
        if verify_execution_artifacts_absent
        else {name: True for name in observed_artifacts}
    )
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    prerequisite_checks = {
        "d7_design_config": source_digests[
            "configs/preregistration/self_model_v0_1_d7_heldout_causal_transfer.draft.json"
        ]
        == D7_DESIGN_SHA256,
        "d7b_contract": source_digests[
            "configs/development/self_model_v0_1_d7b_manifest_runtime_contract.json"
        ]
        == D7B_CONTRACT_SHA256,
        "calibration_manifest": source_digests[
            "configs/development/self_model_v0_1_d7_projection_training_manifest.json"
        ]
        == CALIBRATION_MANIFEST_SHA256,
        "heldout_manifest": source_digests[
            "configs/development/self_model_v0_1_d7_heldout_transfer_manifest.json"
        ]
        == HELDOUT_MANIFEST_SHA256,
    }
    checks = {
        "config_valid": all(validate_config(spec).values()),
        "authorization_schema_exact": set(schema["properties"])
        == AUTHORIZATION_FIELDS,
        "synthetic_public_semantics_acceptance_valid": acceptance["valid"],
        "eight_cells_and_eighteen_calls_frozen": len(spec["compatibility_cells"])
        == 8
        and spec["counts"]["model_forward_calls_total"] == 18,
        "target_layer_rule_is_independent_and_derived": TARGET_LAYER_INDEX == 15
        and spec["target_layer_rule"]["derived_independently_from_d6d_results"]
        is True,
        "payloads_remain_unaccessed": all(
            spec["payload_separation"][field] is False
            for field in (
                "calibration_payload_accessed",
                "heldout_payload_accessed",
                "capability_payload_accessed",
                "formal_test_set_accessed",
            )
        ),
        "authorization_precedes_installed_source_probe": lines[
            "validate_d7c_authorization"
        ]
        < lines["_probe_installed_source"],
        "claim_precedes_runtime_dependencies_and_model_load": lines["_create_claim"]
        < lines["_runtime_dependencies"]
        < lines["load"],
        "future_authorization_exact_and_separate": spec[
            "future_execution_authorization_text"
        ]
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "execution_artifacts_absent": all(artifacts.values()),
        "prerequisite_digests_valid": all(prerequisite_checks.values()),
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D7-C static entry verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "status": "d7c_real_public_semantics_entry_static_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": validate_config(spec),
        "prerequisite_checks": prerequisite_checks,
        "synthetic_acceptance": acceptance,
        "entry_call_lines": lines,
        "execution_artifacts": artifacts,
        "future_exact_owner_authorization_text": FUTURE_EXECUTION_AUTHORIZATION_TEXT,
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
            "calibration_payload_accessed": False,
            "heldout_payload_accessed": False,
            "projection_implemented": False,
            "projection_constructed": False,
            "d7c_executed": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "d6d_rerun": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
