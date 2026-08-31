"""D8-C real numerical-identifiability protocol and no-model safe entry.

This module deliberately stops at protocol validation.  It never imports a model
runtime, opens weights, reads calibration/held-out payloads, or creates an
authorization/claim artifact.  A future real runner must consume this contract
only after a separate owner authorization and single-use claim.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d8b_manifest_endpoint_contract import (
    validate_contract_config,
    validate_determinism_manifest,
    validate_endpoint_manifest,
    validate_fixture_manifest,
    validate_schedule_manifest,
)


CONTRACT_VERSION = "0.1-self-model-d8c-real-numerical-identifiability-entry"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8c_real_numerical_identifiability.json"
)
D8B_CONTRACT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8b_manifest_endpoint_contract.json"
)
FIXTURE_RELATIVE_PATH = "configs/development/self_model_v0_1_d8_fixture_manifest.json"
SCHEDULE_RELATIVE_PATH = "configs/development/self_model_v0_1_d8_counterbalanced_schedule.json"
DETERMINISM_RELATIVE_PATH = "configs/development/self_model_v0_1_d8_determinism_policy.json"
ENDPOINT_RELATIVE_PATH = "configs/development/self_model_v0_1_d8_excess_drift_endpoint.json"
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d8c_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = "results/authorizations/self_model_v0_1_d8c_real_v01.json"
CLAIM_RELATIVE_PATH = "results/development/self_model_v0_1_d8c_real_v01/execution_claim.json"
OUTPUT_RELATIVE_DIR = "results/development/self_model_v0_1_d8c_real_v01"

D8B_CONTRACT_SHA256 = "cf3c1142ba4d97fa7cd484415869e5f6e54635f1264204e9f7dcbf6bfa07fc4f"
FIXTURE_SHA256 = "d0b9c2e67eff48f2e9fd3cf9b5244cdcec165091bb4b079acecd8fd5b3323c2a"
SCHEDULE_SHA256 = "b1ecd2c027bd6aed8834147ec333c5a2ac1ba3df6550ee3ab221612e4805943f"
DETERMINISM_SHA256 = "81d98e24a61f29b9e5e5fed77612b2ffa56953c80282ee811192bbd7417dff83"
ENDPOINT_SHA256 = "96517ab9314ac2e8f68fd520bfb860626e39b1b69e3403f7033781a69983458c"

REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D8-C 真实数值可识别性协议设计与无模型安全入口实现；"
    "仅设计新的真实执行协议、确定性策略、授权/claim/output 命名空间和失败即停止门，"
    "不执行模型、不访问权重、不探测或修改真实 runner，不重跑 D7-C/D6D，也不授权 D8-C"
    "真实执行、正式测试集、Self 效果实验、Self Updater、raw-original 路线或自动重跑。"
)
FUTURE_EXECUTION_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 D8-C 真实2.9B数值可识别性验证一次（固定8次conditioning不计分、"
    "24个fixture/288个pair/584次forward、严格确定性策略与完整有序ledger），并授权观察本次结果；"
    "不授权重跑D8-C或历史实验、自动重跑、D7-C/D6D重跑、D7-D/D7-E、projection、正式测试集、"
    "Self效果结论、Self Updater、raw-original路线。"
)
CLASSIFICATION = (
    "d8c_real_numerical_identifiability_safe_entry_static_verified_"
    "execution_not_authorized"
)
NEXT_GATE = "remote_no_model_d8c_verification_then_separate_d8c_real_execution_authorization"

SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    D8B_CONTRACT_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    DETERMINISM_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_d8c_real_numerical_identifiability.md",
    "scripts/verify_self_model_v0_1_d8c_real_entry.py",
    "src/psa/self_model/d8c_real_numerical_identifiability.py",
    "tests/test_self_model_d8c_real_entry.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D8-C {label} must be an object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError(f"D8-C {label} path is not frozen")
    resolved = (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise PermissionError(f"D8-C {label} escapes project root")
    return resolved


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config.get("frozen_inputs", {})
    protocol = config.get("protocol", {})
    determinism = config.get("determinism", {})
    authority = config.get("authority", {})
    boundary = config.get("historical_boundary", {})
    gates = config.get("gate_sequence", [])
    namespaces = config.get("future_namespaces", {})
    required_false = (
        "installed_source_probe_authorized",
        "real_runner_modification_authorized",
        "rwkv_import_authorized",
        "torch_import_authorized",
        "weights_access_authorized",
        "model_load_authorized",
        "model_execution_authorized",
        "d8c_real_execution_authorized",
        "d8c_rerun_authorized",
        "d7c_fix_authorized",
        "d7c_rerun_authorized",
        "d6d_rerun_authorized",
        "d7d_authorized",
        "d7e_authorized",
        "projection_authorized",
        "formal_test_set_authorized",
        "self_effect_conclusion_authorized",
        "self_updater_authorized",
        "raw_original_route_authorized",
        "automatic_rerun_authorized",
    )
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D8-C_real_numerical_identifiability_protocol_and_safe_entry"
        and config.get("status") == "offline_safe_entry_implemented_unrun_no_model"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "frozen_inputs_exact": frozen
        == {
            "d8b_contract_path": D8B_CONTRACT_RELATIVE_PATH,
            "d8b_contract_sha256": D8B_CONTRACT_SHA256,
            "fixture_manifest_path": FIXTURE_RELATIVE_PATH,
            "fixture_manifest_sha256": FIXTURE_SHA256,
            "schedule_manifest_path": SCHEDULE_RELATIVE_PATH,
            "schedule_manifest_sha256": SCHEDULE_SHA256,
            "determinism_manifest_path": DETERMINISM_RELATIVE_PATH,
            "determinism_manifest_sha256": DETERMINISM_SHA256,
            "endpoint_manifest_path": ENDPOINT_RELATIVE_PATH,
            "endpoint_manifest_sha256": ENDPOINT_SHA256,
        },
        "protocol_exact": protocol.get("future_forward_calls") == 584
        and protocol.get("conditioning_forward_calls") == 8
        and protocol.get("scored_forward_calls") == 576
        and protocol.get("model_process_contract")
        == "one process, one frozen launcher policy, one complete ledger, no adaptive retries"
        and protocol.get("payload_boundary")
        == "no calibration or held-out payload may be opened by the safe entry"
        and protocol.get("failure_action")
        == "invalidate_and_stop_without_rerun_or_relaxation",
        "determinism_exact": determinism
        == {
            "launcher_environment_before_python": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "28083101",
                "RWKV_DE_VERSION": "unset",
            },
            "runtime_flags_before_model_load": {
                "torch_use_deterministic_algorithms": True,
                "torch_deterministic_warn_only": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "policy_seed": 28083101,
            "bootstrap_seed": 28083102,
            "unavailable_policy_action": "stop_without_relaxing_policy_or_scoring_results",
        },
        "stop_gates_frozen": config.get("stop_gates")
        == [
            "authorization_missing_or_digest_mismatch",
            "D8-B_manifest_hash_or_commitment_mismatch",
            "deterministic_policy_unavailable_or_changed",
            "payload_boundary_violation",
            "nonfinite_or_incompatible_output",
            "missing_duplicate_or_reordered_call",
            "forward_exception_or_process_interruption",
            "call_count_not_equal_to_584",
            "any_attempted_rerun_or_adaptive_retry",
        ],
        "namespaces_exact": namespaces
        == {
            "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
            "authorization_path": AUTHORIZATION_RELATIVE_PATH,
            "claim_path": CLAIM_RELATIVE_PATH,
            "output_dir": OUTPUT_RELATIVE_DIR,
            "raw_comparisons_path": OUTPUT_RELATIVE_DIR + "/raw_comparisons.jsonl",
            "report_path": OUTPUT_RELATIVE_DIR + "/report.json",
            "integrity_path": OUTPUT_RELATIVE_DIR + "/integrity.json",
        }
        and len(set(namespaces.values())) == len(namespaces),
        "authority_exact": authority.get("protocol_design_authorized") is True
        and authority.get("offline_safe_entry_implementation_authorized") is True
        and authority.get("offline_fixture_acceptance_authorized") is True
        and all(authority.get(field) is False for field in required_false),
        "historical_boundary_exact": boundary
        == {
            "d8b_data_reused_as_new_data": False,
            "d7c_claim_or_result_reused": False,
            "d6d_claim_or_result_reused": False,
            "historical_rerun": False,
            "thresholds_changed": False,
        },
        "gates_exact": gates
        == [
            {"gate_id": "D8-A", "implemented": True, "model_execution": False},
            {"gate_id": "D8-B", "implemented": True, "model_execution": False},
            {"gate_id": "D8-C-design", "implemented": True, "model_execution": False},
            {
                "gate_id": "D8-C-real",
                "implemented": False,
                "authorized": False,
                "model_execution": True,
                "future_forward_calls": 584,
            },
        ],
        "classification_and_next_gate_exact": config.get("required_classification")
        == CLASSIFICATION
        and config.get("next_gate") == NEXT_GATE,
        "future_authorization_text_frozen": config.get("future_execution_authorization_text")
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D8-C config failed closed: " + ", ".join(failed))
    return checks


def validate_authorization_schema(schema: Mapping[str, Any]) -> dict[str, bool]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    checks = {
        "object_and_no_extras": schema.get("type") == "object"
        and schema.get("additionalProperties") is False,
        "version_const_exact": properties.get("authorization_version", {}).get("const")
        == "0.1-self-model-d8c-real-numerical-identifiability",
        "stage_const_exact": properties.get("stage", {}).get("const")
        == "Self-Model-v0.1-D8-C_real_numerical_identifiability_execution",
        "single_use_and_scope_const": properties.get("single_use", {}).get("const") is True
        and properties.get("scope", {}).get("const")
        == "one_d8c_process_584_forward_calls_no_payload_access",
        "required_fields_complete": set(required) == set(properties)
        and len(required) == 29,
        "execution_flags_true_only_future": all(
            properties.get(field, {}).get("const") is True
            for field in (
                "authorized",
                "model_execution_authorized",
                "model_load_authorized",
                "weights_access_authorized",
                "result_observation_authorized",
            )
        ),
        "future_authorization_text_const": properties.get("authorization_text", {}).get(
            "const"
        )
        == FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        "later_gates_false": all(
            properties.get(field, {}).get("const") is False
            for field in (
                "d8c_rerun_authorized",
                "d7c_rerun_authorized",
                "d6d_rerun_authorized",
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
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D8-C authorization schema changed: " + ", ".join(failed))
    return checks


def build_call_plan(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditioning = schedule.get("conditioning_calls", [])
    blocks = schedule.get("pair_blocks", [])
    plan: list[dict[str, Any]] = []
    for call in conditioning:
        plan.append(
            {
                "call_id": call["call_id"],
                "kind": "conditioning",
                "route": call["route"],
                "scored": False,
            }
        )
    for block in blocks:
        for position, route in enumerate(block["route_order"], start=1):
            plan.append(
                {
                    "call_id": f'{block["pair_block_id"]}-call-{position}',
                    "kind": "scored_pair_call",
                    "pair_block_id": block["pair_block_id"],
                    "pair_type": block["pair_type"],
                    "route": route,
                    "scored": True,
                }
            )
    return plan


def validate_call_plan(plan: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    ids = [record.get("call_id") for record in plan]
    checks = {
        "total_calls_exact": len(plan) == 584,
        "call_ids_unique": len(ids) == len(set(ids)) and all(isinstance(item, str) for item in ids),
        "conditioning_prefix_exact": len(plan) >= 8
        and all(record.get("kind") == "conditioning" for record in plan[:8])
        and all(record.get("scored") is False for record in plan[:8]),
        "scored_suffix_exact": len(plan) == 584
        and all(record.get("kind") == "scored_pair_call" for record in plan[8:])
        and all(record.get("scored") is True for record in plan[8:]),
        "pair_calls_have_two_routes": all(
            record.get("route") in {"public", "wrapper_zero"} for record in plan
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError("D8-C call plan failed closed: " + ", ".join(failed))
    return checks


def validate_ledger_order(
    plan: Sequence[Mapping[str, Any]], ledger: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [record["call_id"] for record in plan]
    actual = [record.get("call_id") for record in ledger]
    if actual != expected:
        raise ValueError("D8-C ledger must contain the complete frozen call order")
    return {"valid": True, "record_count": len(ledger), "call_ids_digest": sha256_json(actual)}


def run_fake_acceptance(schedule: Mapping[str, Any]) -> dict[str, Any]:
    plan = build_call_plan(schedule)
    plan_checks = validate_call_plan(plan)
    ledger = [{"call_id": item["call_id"], "status": "planned"} for item in plan]
    ledger_snapshot = copy.deepcopy(ledger)
    valid_ledger = validate_ledger_order(plan, ledger)
    rejection_checks: dict[str, bool] = {}
    for label, candidate in (
        ("missing_call_rejected", ledger[:-1]),
        ("duplicate_call_rejected", ledger[:-1] + [copy.deepcopy(ledger[0])]),
        ("reordered_call_rejected", ledger[:2][::-1] + ledger[2:]),
    ):
        try:
            validate_ledger_order(plan, candidate)
            rejection_checks[label] = False
        except ValueError:
            rejection_checks[label] = True
    checks = {
        "plan_checks_all_pass": all(plan_checks.values()),
        "ledger_complete_and_ordered": valid_ledger["record_count"] == 584,
        "fake_ledger_input_unchanged": ledger == ledger_snapshot,
        "all_fail_closed_rejections_pass": all(rejection_checks.values()),
        "no_numeric_or_model_objects_created": True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "plan_checks": plan_checks,
        "rejection_checks": rejection_checks,
        "counts": {
            "conditioning_forward_calls": 8,
            "scored_forward_calls": 576,
            "total_future_forward_calls": len(plan),
        },
        "call_ids_digest": valid_ledger["call_ids_digest"],
    }


def build_static_report(*, config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    expected_config = _inside(root, CONFIG_RELATIVE_PATH, "config")
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected_config:
        raise PermissionError("D8-C config path is not frozen")
    config = _object(expected_config, "config")
    config_checks = validate_config(config)
    d8b = _object(_inside(root, D8B_CONTRACT_RELATIVE_PATH, "D8-B contract"), "D8-B contract")
    fixture = _object(_inside(root, FIXTURE_RELATIVE_PATH, "fixture manifest"), "fixture manifest")
    schedule = _object(_inside(root, SCHEDULE_RELATIVE_PATH, "schedule manifest"), "schedule manifest")
    determinism = _object(_inside(root, DETERMINISM_RELATIVE_PATH, "determinism manifest"), "determinism manifest")
    endpoint = _object(_inside(root, ENDPOINT_RELATIVE_PATH, "endpoint manifest"), "endpoint manifest")
    schema = _object(_inside(root, AUTHORIZATION_SCHEMA_RELATIVE_PATH, "authorization schema"), "authorization schema")
    d8b_checks = validate_contract_config(d8b)
    manifest_checks = {
        "fixture_manifest_valid": all(validate_fixture_manifest(fixture).values()),
        "schedule_manifest_valid": all(validate_schedule_manifest(schedule).values()),
        "determinism_manifest_valid": all(validate_determinism_manifest(determinism).values()),
        "endpoint_manifest_valid": all(validate_endpoint_manifest(endpoint).values()),
    }
    hash_checks = {
        D8B_CONTRACT_RELATIVE_PATH: sha256_file(_inside(root, D8B_CONTRACT_RELATIVE_PATH, "D8-B contract")) == D8B_CONTRACT_SHA256,
        FIXTURE_RELATIVE_PATH: sha256_file(_inside(root, FIXTURE_RELATIVE_PATH, "fixture manifest")) == FIXTURE_SHA256,
        SCHEDULE_RELATIVE_PATH: sha256_file(_inside(root, SCHEDULE_RELATIVE_PATH, "schedule manifest")) == SCHEDULE_SHA256,
        DETERMINISM_RELATIVE_PATH: sha256_file(_inside(root, DETERMINISM_RELATIVE_PATH, "determinism manifest")) == DETERMINISM_SHA256,
        ENDPOINT_RELATIVE_PATH: sha256_file(_inside(root, ENDPOINT_RELATIVE_PATH, "endpoint manifest")) == ENDPOINT_SHA256,
    }
    schema_checks = validate_authorization_schema(schema)
    acceptance = run_fake_acceptance(schedule)
    namespace_checks = {
        "authorization_absent": not _inside(root, AUTHORIZATION_RELATIVE_PATH, "authorization").exists(),
        "claim_absent": not _inside(root, CLAIM_RELATIVE_PATH, "claim").exists(),
        "output_namespace_not_materialized": not _inside(root, OUTPUT_RELATIVE_DIR, "output").exists(),
        "no_payload_paths_declared": all(
            token not in json.dumps(config, ensure_ascii=False).lower()
            for token in ("calibration_payload", "held_out_payload", "heldout_payload")
        ),
    }
    checks = {
        "config_valid": all(config_checks.values()),
        "d8b_contract_valid": all(d8b_checks.values()),
        **manifest_checks,
        "frozen_input_hashes_valid": all(hash_checks.values()),
        "authorization_schema_valid": all(schema_checks.values()),
        "fake_acceptance_valid": acceptance["valid"],
        "future_call_count_exact": acceptance["counts"]["total_future_forward_calls"] == 584,
        "d8b_thresholds_unchanged": config["historical_boundary"]["thresholds_changed"] is False,
        "historical_nonreuse_preserved": all(not value for key, value in config["historical_boundary"].items() if key != "thresholds_changed"),
        "future_artifacts_not_created": all(namespace_checks.values()),
        "source_inventory_complete": all((_inside(root, path, "source")).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("D8-C static verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": CONTRACT_VERSION,
        "status": "d8c_real_numerical_identifiability_safe_entry_static_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "manifest_checks": manifest_checks,
        "frozen_input_hash_checks": hash_checks,
        "authorization_schema_checks": schema_checks,
        "namespace_checks": namespace_checks,
        "fake_acceptance": acceptance,
        "next_gate": NEXT_GATE,
        "safety": {
            "installed_source_probed": False,
            "real_runner_modified": False,
            "authorization_created": False,
            "execution_claim_created": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "payload_accessed": False,
            "d8c_real_execution_authorized": False,
            "d8c_rerun_authorized": False,
            "d7c_rerun": False,
            "d6d_rerun": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "projection_constructed": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
        "source_digests": {path: sha256_file(_inside(root, path, "source")) for path in SOURCE_PATHS},
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
