from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Any, Callable, Mapping

from psa.artifacts import sha256_file, sha256_json


FIXTURE_VERSION = "0.1-d5c-decorator-object-protocol-offline"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_decorator_object_protocol_fixture.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D5C失败纯离线 decorator/object-protocol 边界 fixture 实现；"
    "只使用合成 Python fixture 检查 decorator、descriptor、`setattr`/`delattr`/直接字典删除及方法解析，"
    "不修改真实 runtime，不导入 RWKV/Torch，不访问权重，不加载或执行模型，不授权修复或重跑 D5C，"
    "也不授权 D5D/D5E、正式测试集、Self 效果、真实 Self projection、Self Updater 或自动重跑。"
)
CLASSIFICATION = (
    "synthetic_side_dispatch_state_is_sufficient_for_direct_pop_contamination_"
    "standard_decorators_restore_real_root_cause_unresolved"
)
CALLBACK_ATTRIBUTE = "_psa_post_ffn_residual_callback"
TARGET_METHODS = ("forward_one", "forward_seq")
MANAGED_NAMES = (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
EXECUTION_PATHS = TARGET_METHODS
DECORATOR_KINDS = ("plain", "identity_decorator", "non_caching_descriptor")
CLEANUP_MODES = ("direct_instance_dict_pop", "delattr")
TRUE_AUTHORITY_FIELDS = {
    "synthetic_fixture_implementation_authorized",
    "existing_source_and_report_observation_authorized",
}
FALSE_AUTHORITY_FIELDS = {
    "real_runtime_modification_authorized",
    "fix_implementation_authorized",
    "rwkv_import_authorized",
    "torch_import_authorized",
    "weights_access_authorized",
    "model_load_authorized",
    "model_execution_authorized",
    "d5c_rerun_authorized",
    "d5d_authorized",
    "d5e_authorized",
    "formal_test_set_authorized",
    "self_effect_conclusion_authorized",
    "real_self_projection_authorized",
    "self_updater_authorized",
    "automatic_rerun_authorized",
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_decorator_object_protocol_fixture.md",
    "scripts/verify_self_model_v0_1_d5c_decorator_object_protocol_fixture.py",
    "src/psa/self_model/d5c_decorator_object_protocol_fixture.py",
    "src/psa/self_model/d5c_dispatch_cache_source_audit.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d5c_decorator_object_protocol_fixture.py",
)


class NonCachingDescriptor:
    """A synthetic non-data descriptor; it deliberately owns no dispatch cache."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        self.access_count = 0

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        self.access_count += 1
        return types.MethodType(self.function, instance)


class CallbackCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, execution_path: str) -> None:
        if execution_path not in EXECUTION_PATHS:
            raise ValueError("synthetic callback received an invalid execution path")
        self.count += 1


def identity_decorator(function: Callable[..., Any]) -> Callable[..., Any]:
    return function


def _original_function(execution_path: str) -> Callable[..., tuple[str, str]]:
    def original(self: Any, payload: Any) -> tuple[str, str]:
        return "original", execution_path

    original.__name__ = execution_path
    original._fixture_origin = "original"  # type: ignore[attr-defined]
    return original


def _active_function(execution_path: str) -> Callable[..., tuple[str, str]]:
    def active(self: Any, payload: Any) -> tuple[str, str]:
        callback = getattr(self, CALLBACK_ATTRIBUTE, None)
        if callback is not None:
            callback(execution_path)
        return "active", execution_path

    active.__name__ = execution_path
    active._fixture_origin = "active"  # type: ignore[attr-defined]
    return active


def _build_standard_class(decorator_kind: str) -> type:
    if decorator_kind not in DECORATOR_KINDS:
        raise ValueError("unsupported synthetic decorator kind")
    namespace: dict[str, Any] = {}
    for execution_path in EXECUTION_PATHS:
        function = _original_function(execution_path)
        if decorator_kind == "identity_decorator":
            function = identity_decorator(function)
        elif decorator_kind == "non_caching_descriptor":
            function = NonCachingDescriptor(function)  # type: ignore[assignment]
        namespace[execution_path] = function

    def forward(self: Any, execution_path: str, payload: Any) -> tuple[str, str]:
        if execution_path not in EXECUTION_PATHS:
            raise ValueError("unsupported synthetic execution path")
        return getattr(self, execution_path)(payload)

    namespace["forward"] = forward
    namespace["fixture_decorator_kind"] = decorator_kind
    return type(f"Synthetic_{decorator_kind}", (), namespace)


def _build_side_cache_class() -> type:
    base = _build_standard_class("non_caching_descriptor")

    def __init__(self: Any) -> None:
        object.__setattr__(self, "_side_dispatch", {})
        object.__setattr__(self, "_protocol_events", [])

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if name in MANAGED_NAMES:
            self._protocol_events.append({"operation": "setattr", "name": name})
            self._side_dispatch[name] = value
        object.__setattr__(self, name, value)

    def __getattribute__(self: Any, name: str) -> Any:
        if name in MANAGED_NAMES:
            side_dispatch = object.__getattribute__(self, "_side_dispatch")
            if name in side_dispatch:
                object.__getattribute__(self, "_protocol_events").append(
                    {"operation": "side_get", "name": name}
                )
                return side_dispatch[name]
        return object.__getattribute__(self, name)

    def __delattr__(self: Any, name: str) -> None:
        if name in MANAGED_NAMES:
            self._protocol_events.append({"operation": "delattr", "name": name})
            self._side_dispatch.pop(name, None)
        object.__delattr__(self, name)

    return type(
        "Synthetic_non_caching_descriptor_side_cache",
        (base,),
        {
            "__init__": __init__,
            "__setattr__": __setattr__,
            "__getattribute__": __getattribute__,
            "__delattr__": __delattr__,
            "fixture_protocol": "explicit_side_dispatch",
        },
    )


def _descriptor_access_count(model: Any, execution_path: str) -> int | None:
    for owner in type(model).__mro__:
        candidate = owner.__dict__.get(execution_path)
        if isinstance(candidate, NonCachingDescriptor):
            return candidate.access_count
    return None


def _resolved_origin(value: Any) -> str | None:
    function = getattr(value, "__func__", value)
    return getattr(function, "_fixture_origin", None)


def run_fixture_case(
    *,
    decorator_kind: str,
    cleanup_mode: str,
    execution_path: str,
    side_dispatch: bool,
) -> dict[str, Any]:
    if decorator_kind not in DECORATOR_KINDS:
        raise ValueError("unsupported synthetic decorator kind")
    if cleanup_mode not in CLEANUP_MODES:
        raise ValueError("unsupported synthetic cleanup mode")
    if execution_path not in EXECUTION_PATHS:
        raise ValueError("unsupported synthetic execution path")
    if side_dispatch and decorator_kind != "non_caching_descriptor":
        raise ValueError("side-dispatch scenario is frozen to the descriptor boundary")

    model_class = (
        _build_side_cache_class() if side_dispatch else _build_standard_class(decorator_kind)
    )
    model = model_class()
    baseline = model.forward(execution_path, "payload")
    descriptor_count_after_baseline = _descriptor_access_count(model, execution_path)
    callback = CallbackCounter()
    instance_dict = model.__dict__
    if any(name in instance_dict for name in MANAGED_NAMES):
        raise RuntimeError("synthetic model starts with conflicting managed names")

    setattr(model, CALLBACK_ATTRIBUTE, callback)
    for name in TARGET_METHODS:
        setattr(model, name, types.MethodType(_active_function(name), model))
    active = model.forward(execution_path, "payload")
    callback_count_after_active = callback.count

    if cleanup_mode == "direct_instance_dict_pop":
        for name in MANAGED_NAMES:
            instance_dict.pop(name, None)
    else:
        for name in MANAGED_NAMES:
            if name in instance_dict:
                delattr(model, name)

    instance_keys_after_cleanup = sorted(
        name for name in MANAGED_NAMES if name in instance_dict
    )
    side_keys_after_cleanup = sorted(
        getattr(model, "_side_dispatch", {}).keys()
        if side_dispatch else []
    )
    resolved = getattr(model, execution_path)
    resolved_origin_after_cleanup = _resolved_origin(resolved)
    post_cleanup = model.forward(execution_path, "payload")
    callback_count_after_post_cleanup = callback.count
    protocol_events = list(getattr(model, "_protocol_events", []))
    descriptor_count_after_post_cleanup = _descriptor_access_count(model, execution_path)
    restored = (
        post_cleanup == baseline
        and resolved_origin_after_cleanup == "original"
        and callback_count_after_post_cleanup == callback_count_after_active
    )
    contaminated = (
        post_cleanup == active
        and resolved_origin_after_cleanup == "active"
        and callback_count_after_post_cleanup == callback_count_after_active + 1
    )
    expected_contamination = side_dispatch and cleanup_mode == "direct_instance_dict_pop"
    checks = {
        "baseline_is_original": baseline == ("original", execution_path),
        "active_is_active": active == ("active", execution_path),
        "active_callback_exact": callback_count_after_active == 1,
        "instance_keys_absent_after_cleanup": instance_keys_after_cleanup == [],
        "outcome_matches_frozen_expectation": contaminated if expected_contamination else restored,
        "side_state_matches_frozen_expectation": (
            set(side_keys_after_cleanup) == set(MANAGED_NAMES)
            if expected_contamination
            else side_keys_after_cleanup == []
        ),
    }
    return {
        "case_id": (
            f"{'side_cache' if side_dispatch else 'standard'}:{decorator_kind}:"
            f"{cleanup_mode}:{execution_path}"
        ),
        "decorator_kind": decorator_kind,
        "cleanup_mode": cleanup_mode,
        "execution_path": execution_path,
        "side_dispatch": side_dispatch,
        "valid": all(checks.values()),
        "checks": checks,
        "observed": {
            "baseline": list(baseline),
            "active": list(active),
            "post_cleanup": list(post_cleanup),
            "restored": restored,
            "contaminated": contaminated,
            "resolved_origin_after_cleanup": resolved_origin_after_cleanup,
            "instance_keys_after_cleanup": instance_keys_after_cleanup,
            "side_keys_after_cleanup": side_keys_after_cleanup,
            "callback_count_after_active": callback_count_after_active,
            "callback_count_after_post_cleanup": callback_count_after_post_cleanup,
            "descriptor_count_after_baseline": descriptor_count_after_baseline,
            "descriptor_count_after_post_cleanup": descriptor_count_after_post_cleanup,
            "protocol_events": protocol_events,
        },
    }


def run_fixture_matrix() -> list[dict[str, Any]]:
    standard = [
        run_fixture_case(
            decorator_kind=decorator_kind,
            cleanup_mode=cleanup_mode,
            execution_path=execution_path,
            side_dispatch=False,
        )
        for decorator_kind in DECORATOR_KINDS
        for cleanup_mode in CLEANUP_MODES
        for execution_path in EXECUTION_PATHS
    ]
    side_cache = [
        run_fixture_case(
            decorator_kind="non_caching_descriptor",
            cleanup_mode=cleanup_mode,
            execution_path=execution_path,
            side_dispatch=True,
        )
        for cleanup_mode in CLEANUP_MODES
        for execution_path in EXECUTION_PATHS
    ]
    return standard + side_cache


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C decorator/object-protocol fixture config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("frozen_prerequisites")
    matrix = config.get("matrix")
    expected = config.get("expected_observations")
    authority = config.get("authority")
    if not all(isinstance(item, Mapping) for item in (prerequisites, matrix, expected, authority)):
        raise ValueError("D5C decorator/object-protocol fixture config is incomplete")
    checks = {
        "identity_exact": config.get("fixture_version") == FIXTURE_VERSION
        and config.get("stage") == "Coupling-D5C_failure_decorator_object_protocol_fixture"
        and config.get("status") == "synthetic_fixture_authorized_no_model_no_fix"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "failed_prerequisites_preserved": prerequisites
        == {
            "d5c_real_report_sha256": "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21",
            "lifecycle_diagnostic_report_sha256": "3dfa640d32bfbea2594e7d58afb6c71552b8fc8619a215d72ae8b23e5e0a4150",
            "source_audit_report_sha256": "652b1a4cc0bcf3f8c5b03f304133cc151f3af07160d8f8ecdddad9afb32d1342",
            "d5c_wrapper_source_sha256": "e1de359da6d2087721dfd433a3e6ad90c6439bb474325a768c2f1d07fb08b5b7",
            "instrumenter_source_sha256": "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5",
            "d5c_status": "d5c_mechanism_smoke_failed",
            "decision_effect": "stop_without_rerun",
        },
        "matrix_exact": matrix
        == {
            "execution_paths": list(EXECUTION_PATHS),
            "standard_decorator_kinds": list(DECORATOR_KINDS),
            "cleanup_modes": list(CLEANUP_MODES),
            "side_cache_decorator_kind": "non_caching_descriptor",
            "standard_case_count": 12,
            "side_cache_case_count": 4,
            "total_case_count": 16,
        },
        "expectations_exact": expected
        == {
            "all_standard_cases_restore_original": True,
            "side_cache_direct_pop_cases_contaminate": True,
            "side_cache_delattr_cases_restore_original": True,
            "direct_pop_can_leave_instance_keys_absent_while_side_cache_remains": True,
            "synthetic_sufficient_mechanism_is_real_root_cause": False,
        },
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "authority_exact": set(authority) == TRUE_AUTHORITY_FIELDS | FALSE_AUTHORITY_FIELDS
        and all(authority.get(name) is True for name in TRUE_AUTHORITY_FIELDS)
        and all(authority.get(name) is False for name in FALSE_AUTHORITY_FIELDS),
        "next_gate_exact": config.get("next_gate")
        == "offline_fix_design_requires_separate_owner_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C decorator fixture config failed closed: " + ", ".join(failed))
    return checks


def build_fixture_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C decorator fixture config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    cases = run_fixture_matrix()
    standard = [case for case in cases if not case["side_dispatch"]]
    side_direct = [
        case for case in cases
        if case["side_dispatch"] and case["cleanup_mode"] == "direct_instance_dict_pop"
    ]
    side_delattr = [
        case for case in cases
        if case["side_dispatch"] and case["cleanup_mode"] == "delattr"
    ]
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "matrix_has_16_cases": len(cases) == 16,
        "all_cases_self_consistent": all(case["valid"] for case in cases),
        "all_12_standard_cases_restore": len(standard) == 12
        and all(case["observed"]["restored"] for case in standard),
        "plain_cases_restore": all(
            case["observed"]["restored"] for case in standard
            if case["decorator_kind"] == "plain"
        ),
        "identity_decorator_cases_restore": all(
            case["observed"]["restored"] for case in standard
            if case["decorator_kind"] == "identity_decorator"
        ),
        "non_caching_descriptor_cases_restore": all(
            case["observed"]["restored"] for case in standard
            if case["decorator_kind"] == "non_caching_descriptor"
        ),
        "side_cache_direct_pop_both_paths_contaminate": len(side_direct) == 2
        and all(case["observed"]["contaminated"] for case in side_direct),
        "side_cache_direct_pop_instance_keys_are_absent": all(
            case["observed"]["instance_keys_after_cleanup"] == [] for case in side_direct
        ),
        "side_cache_direct_pop_side_keys_remain": all(
            set(case["observed"]["side_keys_after_cleanup"]) == set(MANAGED_NAMES)
            for case in side_direct
        ),
        "side_cache_delattr_both_paths_restore": len(side_delattr) == 2
        and all(case["observed"]["restored"] for case in side_delattr),
        "side_cache_delattr_invokes_protocol": all(
            sum(event["operation"] == "delattr" for event in case["observed"]["protocol_events"])
            == 3 for case in side_delattr
        ),
        "synthetic_result_not_promoted_to_real_root_cause": True,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C decorator/object-protocol fixture failed: " + ", ".join(failed))
    report = {
        "fixture_version": FIXTURE_VERSION,
        "status": "d5c_decorator_object_protocol_fixture_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "matrix_summary": {
            "case_count": len(cases),
            "standard_restored": sum(case["observed"]["restored"] for case in standard),
            "side_cache_direct_pop_contaminated": sum(
                case["observed"]["contaminated"] for case in side_direct
            ),
            "side_cache_delattr_restored": sum(
                case["observed"]["restored"] for case in side_delattr
            ),
        },
        "cases": cases,
        "findings": {
            "confirmed_synthetic": [
                "plain_identity_and_non_caching_descriptor_boundaries_restore_under_standard_python",
                "direct_dict_pop_bypasses_custom_delattr_protocol",
                "explicit_side_dispatch_state_can_retain_active_methods_and_callback_after_instance_keys_are_removed",
                "delattr_clears_that_state_in_the_frozen_synthetic_protocol",
            ],
            "sufficient_not_necessary": [
                "side_dispatch_state_plus_direct_dict_pop_can_generate_the_d5c_contamination_shape",
            ],
            "not_supported": [
                "rwkv_or_torch_has_the_synthetic_side_dispatch_protocol",
                "direct_dict_pop_is_the_real_d5c_root_cause",
                "delattr_is_an_authorized_or_verified_real_fix",
                "permission_to_modify_runtime_or_rerun_d5c",
                "d5c_pass_or_self_effect_conclusion",
            ],
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "real_runtime_modified": False,
            "fix_implemented": False,
            "existing_real_report_reexecuted": False,
            "d5c_rerun": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d5c_conclusion_changed": False,
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
