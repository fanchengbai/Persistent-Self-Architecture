from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-d5c-p1-reporter-fix-design"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_p1_reporter_fix_design.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D5C-P1失败纯离线Tensor/fixture类型判别诊断与reporter修复设计；"
    "仅使用现有authorization、claim、failure、冻结源码和合成fixture，不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型、不实现或授权修复后真实重跑，也不改变原D5C/P1结论或开放"
    "D5D/D5E、正式测试集、Self效果、真实Self projection、Self Updater或自动重跑。"
)
P1_CORE_DIGEST = "30f1f91186cf6e45a775cb28732556ff39c1a44f92b7940d21f8682880d67a78"
AUTHORIZATION_INTERNAL_DIGEST = "c7d78281d93e85814a61fb8fdbb71696495b24981f5b2e944f42cf3e30daf37e"
AUTHORIZATION_FILE_DIGEST = "8b0f34cbc0b2b57a5ffffefb801128ff7e3243283ce6ae182c37ee414c3c2f0f"
CLAIM_FILE_DIGEST = "7c49107b33a223be7ac11f3412328abc07e24e5fc6bcf68accfbe34b7ca97628"
FAILURE_REPORT_DIGEST = "930c31ef6f70c431066cda3637c97fcc35344b8caabaeb2f2f7147a0b5d54483"
CLASSIFICATION = (
    "callable_values_name_collision_reproduced_reporter_dispatch_defect_"
    "explicit_injected_offline_adapter_recommended_no_fix_or_rerun"
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_p1_real_engineering_observation.md",
    "docs/self_model_v0_1_d5c_p1_reporter_fix_design.md",
    "scripts/verify_self_model_v0_1_d5c_p1_reporter_fix_design.py",
    "src/psa/self_model/d5c_p1_engineering_validation.py",
    "src/psa/self_model/d5c_p1_reporter_fix_design.py",
    "tests/test_self_model_d5c_p1_reporter_fix_design.py",
)
ACCEPTANCE_CATEGORIES = (
    "callable_values_collision_reproduces_serialization_typeerror",
    "real_like_tensor_with_callable_values_uses_default_real_serializer",
    "exact_offline_fixture_uses_explicit_adapter",
    "unregistered_data_values_object_fails_closed",
    "callability_guard_rejects_method_but_still_misclassifies_data_property",
    "object_marker_can_be_spoofed_and_is_insufficient_alone",
    "adapter_does_not_read_callable_values_member",
    "default_path_does_not_depend_on_values_name",
    "verification_does_not_modify_frozen_reporter",
)


def _object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C-P1 reporter fix design config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    evidence = config.get("frozen_failure_evidence")
    authority = config.get("authority")
    strategies = config.get("strategy_review")
    recommended = config.get("recommended_design")
    if not isinstance(evidence, Mapping) or not isinstance(authority, Mapping):
        raise ValueError("D5C-P1 reporter fix design config is incomplete")
    checks = {
        "identity_exact": config.get("design_version") == DESIGN_VERSION
        and config.get("stage") == "D5C-P1_failure_offline_reporter_fix_design"
        and config.get("status") == "offline_diagnosis_and_design_only_no_fix_no_model"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("owner_confirmation_text") == REQUIRED_CONFIRMATION,
        "failure_evidence_exact": evidence
        == {
            "execution_commit": "1bc58579ddc0ff91a8ad37f83044f2046d2ccc16",
            "authorization_internal_digest_sha256": AUTHORIZATION_INTERNAL_DIGEST,
            "authorization_file_sha256": AUTHORIZATION_FILE_DIGEST,
            "execution_claim_sha256": CLAIM_FILE_DIGEST,
            "failure_report_sha256": FAILURE_REPORT_DIGEST,
            "failure_status": "d5c_p1_attempt_failed_claim_consumed",
            "failure_type": "TypeError",
            "failure_message": "Object of type builtin_function_or_method is not JSON serializable",
            "completed_model_forwards": 1,
            "patched_routes_reached": False,
        },
        "reporter_digest_exact": config.get("frozen_reporter_source_sha256")
        == P1_CORE_DIGEST,
        "strategies_exact": strategies
        == [
            {
                "strategy": "callability_guard_on_values",
                "decision": "insufficient_accidental_data_property_still_misclassified",
            },
            {
                "strategy": "object_owned_offline_marker",
                "decision": "insufficient_spoofable_and_production_object_coupled",
            },
            {
                "strategy": "explicit_injected_offline_adapter",
                "decision": "recommended_for_future_fake_first_implementation",
            },
        ],
        "recommended_design_exact": recommended
        == {
            "production_default": "always_use_real_tensor_serializer_without_values_name_dispatch",
            "offline_test_path": "requires_explicit_adapter_argument_with_exact_fixture_acceptance",
            "unknown_object_policy": "fail_closed_without_guessing_from_attribute_names",
            "adapter_scope": "test_only_not_constructed_by_real_runner",
            "real_runner_adapter_value": None,
            "output_commit_rule": "fingerprint_all_outputs_or_fail_without_reclassifying_tensor_kind",
        },
        "acceptance_exact": config.get("future_fake_acceptance")
        == list(ACCEPTANCE_CATEGORIES),
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "authority_exact": authority
        == {
            "offline_failure_diagnosis_authorized": True,
            "reporter_fix_design_authorized": True,
            "synthetic_fixture_execution_authorized": True,
            "reporter_fix_implementation_authorized": False,
            "rwkv_import_authorized": False,
            "torch_import_authorized": False,
            "weights_access_authorized": False,
            "model_load_authorized": False,
            "model_execution_authorized": False,
            "p1_rerun_authorized": False,
            "historical_d5c_conclusion_change_authorized": False,
            "p1_conclusion_change_authorized": False,
            "d5d_authorized": False,
            "d5e_authorized": False,
            "formal_test_set_authorized": False,
            "self_effect_conclusion_authorized": False,
            "real_self_projection_authorized": False,
            "self_updater_authorized": False,
            "automatic_rerun_authorized": False,
        },
        "next_gate_exact": config.get("next_gate")
        == "fake_first_reporter_dispatch_fix_requires_separate_owner_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C-P1 reporter fix design failed closed: " + ", ".join(failed))
    return checks


def inspect_frozen_reporter(root: Path) -> dict[str, Any]:
    path = root / "src/psa/self_model/d5c_p1_engineering_validation.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tensor_payload"
    )
    hasattrs = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hasattr"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "values"
    ]
    serialized_values = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sha256_json"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "values"
    ]
    return {
        "source_sha256": sha256_file(path),
        "historical_source_sha256": P1_CORE_DIGEST,
        "tensor_payload_line": function.lineno,
        "hasattr_values_call_count": len(hasattrs),
        "hasattr_values_line": hasattrs[0].lineno if len(hasattrs) == 1 else None,
        "sha256_json_values_call_count": len(serialized_values),
        "sha256_json_values_line": serialized_values[0].lineno
        if len(serialized_values) == 1 else None,
        "explicit_adapter_present": "offline_adapter" in source,
        "torch_import_present": any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                any(alias.name == "torch" for alias in node.names)
                if isinstance(node, ast.Import) else node.module == "torch"
            )
            for node in tree.body
        ),
    }


class CallableValuesTensorLike:
    def __init__(self) -> None:
        self.real_serializer_calls = 0

    def values(self) -> tuple[int, ...]:
        return (1, 2, 3)


class OfflineFixtureTensor:
    values = (1.0, 2.0, 3.0)


class AccidentalDataValuesObject:
    values = (9.0, 8.0)


class SpoofedMarkerObject:
    offline_tensor_fixture = True
    values = (7.0,)


class ExactOfflineAdapter:
    def accepts(self, value: Any) -> bool:
        return type(value) is OfflineFixtureTensor

    def payload(self, value: OfflineFixtureTensor) -> dict[str, Any]:
        return {"kind": "offline_fixture", "sha256": sha256_json(value.values)}


def _legacy_dispatch(value: Any) -> dict[str, Any]:
    if hasattr(value, "values"):
        return {"sha256": sha256_json(value.values)}
    raise TypeError("legacy default tensor serializer unavailable in synthetic fixture")


def _callability_guard_dispatch(value: Any) -> str:
    member = getattr(value, "values", None)
    return "offline" if member is not None and not callable(member) else "real"


def _marker_dispatch(value: Any) -> str:
    return "offline" if getattr(value, "offline_tensor_fixture", False) is True else "real"


def _proposed_dispatch(
    value: Any, *, offline_adapter: ExactOfflineAdapter | None = None
) -> dict[str, Any]:
    if offline_adapter is not None and offline_adapter.accepts(value):
        return offline_adapter.payload(value)
    if type(value) is CallableValuesTensorLike:
        value.real_serializer_calls += 1
        return {"kind": "real_tensor", "sha256": "synthetic-real-digest"}
    raise TypeError("unregistered tensor object rejected without attribute-name guessing")


def run_synthetic_dispatch_diagnostic() -> dict[str, Any]:
    real_like = CallableValuesTensorLike()
    try:
        _legacy_dispatch(real_like)
    except TypeError as error:
        reproduced_type = type(error).__name__
        reproduced_message = str(error)
    else:
        reproduced_type = None
        reproduced_message = None

    offline = OfflineFixtureTensor()
    adapter = ExactOfflineAdapter()
    real_result = _proposed_dispatch(real_like, offline_adapter=adapter)
    offline_result = _proposed_dispatch(offline, offline_adapter=adapter)
    try:
        _proposed_dispatch(AccidentalDataValuesObject(), offline_adapter=adapter)
    except TypeError:
        unknown_failed_closed = True
    else:
        unknown_failed_closed = False
    checks = {
        "callable_values_collision_reproduced": reproduced_type == "TypeError"
        and reproduced_message
        == "Object of type method is not JSON serializable",
        "callable_values_member_confirmed": callable(real_like.values),
        "legacy_offline_fixture_still_serializes": _legacy_dispatch(offline)["sha256"]
        == sha256_json(offline.values),
        "callability_guard_avoids_method_collision": _callability_guard_dispatch(real_like)
        == "real",
        "callability_guard_still_misclassifies_accidental_data": _callability_guard_dispatch(
            AccidentalDataValuesObject()
        ) == "offline",
        "marker_strategy_is_spoofable": _marker_dispatch(SpoofedMarkerObject()) == "offline",
        "explicit_adapter_routes_exact_fixture_only": offline_result["kind"]
        == "offline_fixture" and not adapter.accepts(AccidentalDataValuesObject()),
        "real_default_ignores_callable_values_member": real_result["kind"] == "real_tensor"
        and real_like.real_serializer_calls == 1,
        "unregistered_data_values_object_fails_closed": unknown_failed_closed,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "reproduced_error": {
            "type": reproduced_type,
            "message": reproduced_message,
        },
        "recommended_dispatch": {
            "real_runner_offline_adapter": None,
            "offline_tests_require_explicit_adapter": True,
            "attribute_name_inference": False,
        },
    }


def build_reporter_fix_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C-P1 reporter fix design config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    source_audit = inspect_frozen_reporter(root)
    diagnostic = run_synthetic_dispatch_diagnostic()
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "failure_integrity_chain_frozen": config["frozen_failure_evidence"]
        ["failure_report_sha256"] == FAILURE_REPORT_DIGEST,
        "claim_consumed_and_patched_routes_not_reached": config["frozen_failure_evidence"]
        ["completed_model_forwards"] == 1
        and config["frozen_failure_evidence"]["patched_routes_reached"] is False,
        "historical_reporter_digest_frozen_in_config": config[
            "frozen_reporter_source_sha256"
        ] == P1_CORE_DIGEST,
        "current_reporter_differs_from_failure_source": source_audit[
            "source_sha256"
        ] != P1_CORE_DIGEST,
        "historical_values_dispatch_removed": source_audit[
            "hasattr_values_call_count"
        ] == 0 and source_audit["sha256_json_values_call_count"] == 0,
        "real_failure_shape_reproduced_synthetically": diagnostic["valid"],
        "callability_guard_rejected_as_incomplete": config["strategy_review"][0][
            "decision"
        ].startswith("insufficient"),
        "object_marker_rejected_as_incomplete": config["strategy_review"][1][
            "decision"
        ].startswith("insufficient"),
        "explicit_adapter_design_recommended": config["strategy_review"][2]["decision"]
        == "recommended_for_future_fake_first_implementation",
        "explicit_adapter_transition_detected": source_audit[
            "explicit_adapter_present"
        ] is True,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C-P1 reporter fix design verification failed: " + ", ".join(failed))
    report = {
        "design_version": DESIGN_VERSION,
        "status": "d5c_p1_offline_reporter_fix_design_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "source_audit": source_audit,
        "synthetic_diagnostic": diagnostic,
        "strategy_review": config["strategy_review"],
        "recommended_design": config["recommended_design"],
        "future_fake_acceptance": config["future_fake_acceptance"],
        "decision": {
            "root_cause_boundary": "reporter_dispatch_attribute_name_collision",
            "runtime_patch_evaluated_by_p1": False,
            "reporter_fix_implemented": False,
            "current_tree_reporter_fix_detected": True,
            "real_rerun_authorized": False,
            "historical_d5c_or_p1_conclusion_changed": False,
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "reporter_fix_implemented": False,
            "current_tree_reporter_fix_detected": True,
            "p1_rerun": False,
            "historical_d5c_conclusion_changed": False,
            "p1_conclusion_changed": False,
            "d5d_authorized": False,
            "d5e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "real_self_projection_constructed": False,
            "self_updater_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
