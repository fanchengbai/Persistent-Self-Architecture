from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d6d_ii_joint_runtime import (
    D6DIIWrapperOwnedRuntime,
    request_for_pilot_condition,
)
from psa.self_model.rwkv7_instrumented_off_runtime import compile_instrumented_methods


DIAGNOSTIC_VERSION = "0.1-coupling-d6d-failure-state-initialization-offline"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d6d_failure_state_initialization_diagnostic.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6D失败纯离线state初始化语义诊断与研究路线审查；"
    "仅使用现有authorization、claim、failure、冻结源码和合成Python fixture，验证public forward的"
    "zero-state初始化、wrapper-owned分派边界及测试覆盖缺口，并判断后续是否存在科学上独立且不构成"
    "D6D重跑的新实验路线；不导入RWKV/Torch、不访问权重、不加载或执行模型、不修改真实runner、"
    "不实现或授权D6D修复后重跑，也不授权D6E、正式测试集、Self效果结论、Self Updater、"
    "raw-original路线、拆分机制运行或自动重跑。"
)
CLASSIFICATION = (
    "d6d_failure_root_cause_converged_on_skipped_public_zero_state_initialization_"
    "no_independent_successor_established"
)
NEXT_GATE = "owner_reviews_offline_stop_decision_no_model_or_fix_authorized"
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d6d_failure_state_initialization_diagnostic.md",
    "docs/self_model_v0_1_coupling_d6d_real_joint_observation.md",
    "scripts/verify_self_model_v0_1_d6d_failure_state_initialization_diagnostic.py",
    "src/psa/self_model/d6d_failure_state_initialization_diagnostic.py",
    "src/psa/self_model/d6d_ii_joint_runtime.py",
    "src/psa/self_model/d6d_ii_real_entry.py",
    "src/psa/self_model/rwkv_interface_audit.py",
    "tests/test_self_model_d6d_failure_state_initialization_diagnostic.py",
    "tests/test_self_model_d6d_ii_real_entry.py",
)


SYNTHETIC_UPSTREAM_SOURCE = """
class RWKV_x070:
    def __init__(self):
        self.n_layer = 32
        self.zero_state_calls = 0

    def generate_zero_state(self):
        self.zero_state_calls += 1
        return [0 for _ in range(self.n_layer * 3)]

    def forward(self, idx, state, full_output=False):
        if state == None:
            state = self.generate_zero_state()
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)

    def forward_one(self, idx, state):
        x = idx
        for i in range(self.n_layer):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_one(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state

    def forward_seq(self, idx, state, full_output=False):
        x = sum(idx)
        for i in range(self.n_layer):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state
""".strip()


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D6D failure diagnostic {label} must be an object")
    return value


def _project_path(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise PermissionError(f"D6D failure diagnostic {label} path is not frozen")
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise PermissionError(f"D6D failure diagnostic {label} escapes project root")
    return resolved


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    evidence = config.get("frozen_failure_evidence", {})
    fixture = config.get("synthetic_fixture", {})
    route = config.get("route_review", {})
    authority = config.get("authority", {})
    checks = {
        "identity_exact": config.get("diagnostic_version") == DIAGNOSTIC_VERSION
        and config.get("stage")
        == "Coupling-D6D_failure_state_initialization_offline_diagnostic_and_route_review"
        and config.get("status")
        == "offline_diagnostic_authorized_no_model_no_fix_no_rerun"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "failure_identity_frozen": evidence.get("execution_commit")
        == "563c3144d23f1a10e27c3e4377952f165fd0230f"
        and evidence.get("failure_report_digest_sha256")
        == "5fc3570762c4d64e30de328711d1f5b9596876d87adf30898fe89a1e2a7ba65d"
        and evidence.get("failure_status")
        == "d6d_real_joint_attempt_failed_claim_consumed"
        and evidence.get("failure_error_type") == "TypeError"
        and evidence.get("failure_error") == "'NoneType' object is not subscriptable",
        "claim_consumed_and_rerun_closed": evidence.get("execution_claim_sha256")
        == "421a4c811722e2600abe08bf742b43faa56f2162e9075c1668710187f5fed909"
        and evidence.get("claim_consumed") is True
        and evidence.get("d6d_rerun_authorized") is False
        and evidence.get("automatic_rerun_authorized") is False,
        "no_capture_projection_or_pilot": evidence.get("training_forward_calls_planned") == 16
        and evidence.get("training_captures_completed") == 0
        and evidence.get("projection_artifact_constructed") is False
        and evidence.get("pilot_forward_calls_planned") == 144
        and evidence.get("pilot_forward_calls_completed") == 0,
        "fixture_exact": fixture.get("kind")
        == "pure_python_public_forward_vs_wrapper_owned_dispatch_triangle"
        and fixture.get("n_layer") == 32
        and fixture.get("state_components") == 96
        and fixture.get("tokens") == [3, 5, 8]
        and fixture.get("required_cases")
        == [
            "public_forward_none_initializes_and_succeeds",
            "wrapper_direct_sequence_none_reproduces_type_error_before_dispatch",
            "wrapper_direct_sequence_prebuilt_state_succeeds",
        ]
        and all(
            fixture.get(field) is False
            for field in ("rwkv_imported", "torch_imported", "model_loaded", "model_executed")
        ),
        "route_decision_frozen": route.get(
            "repair_then_repeat_same_model_manifests_and_160_call_plan_is_d6d_rerun"
        )
        is True
        and route.get("d6e_prerequisite_satisfied") is False
        and route.get("independent_new_experiment_route_established") is False
        and route.get("decision")
        == "stop_d6d_no_independent_successor_established_by_this_diagnostic"
        and len(route.get("minimum_future_independence_requirements", [])) == 5,
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "offline_authority_exact": authority.get("offline_failure_diagnostic_authorized")
        is True
        and authority.get("offline_synthetic_fixture_authorized") is True
        and authority.get("research_route_review_authorized") is True,
        "all_model_fix_and_later_authority_closed": all(
            authority.get(field) is False
            for field in (
                "rwkv_import_authorized",
                "torch_import_authorized",
                "weights_access_authorized",
                "model_load_authorized",
                "model_execution_authorized",
                "real_runner_modification_authorized",
                "d6d_fix_implementation_authorized",
                "d6d_rerun_authorized",
                "d6e_authorized",
                "formal_test_set_authorized",
                "self_effect_conclusion_authorized",
                "self_updater_authorized",
                "raw_original_route_authorized",
                "split_mechanism_run_authorized",
                "automatic_rerun_authorized",
            )
        ),
        "next_gate_exact": config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6D failure diagnostic config changed: " + ", ".join(failed))
    return checks


def _function(tree: ast.AST, name: str, class_name: str | None = None) -> ast.FunctionDef:
    scope: ast.AST = tree
    if class_name is not None:
        classes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise RuntimeError(f"D6D diagnostic expected one {class_name}")
        scope = classes[0]
    matches = [
        node for node in ast.walk(scope)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"D6D diagnostic expected one {name}")
    return matches[0]


def _attribute_call(node: ast.Call, owner: str, method: str) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


def audit_frozen_source(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runtime_path = root / "src/psa/self_model/d6d_ii_joint_runtime.py"
    audit_path = root / "src/psa/self_model/rwkv_interface_audit.py"
    test_path = root / "tests/test_self_model_d6d_ii_real_entry.py"
    runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    wrapper_forward = _function(runtime_tree, "forward", "D6DIIWrapperOwnedRuntime")
    training = _function(runtime_tree, "execute_projection_training")
    direct_calls = [
        node for node in ast.walk(wrapper_forward)
        if isinstance(node, ast.Call)
        and (
            _attribute_call(node, "self", "forward_one")
            or _attribute_call(node, "self", "forward_seq")
        )
    ]
    generator_calls = [
        node for node in ast.walk(wrapper_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate_zero_state"
    ]
    training_calls = [
        node for node in ast.walk(training)
        if isinstance(node, ast.Call) and _attribute_call(node, "runtime", "forward")
    ]
    training_none = bool(
        len(training_calls) == 1
        and len(training_calls[0].args) >= 2
        and isinstance(training_calls[0].args[1], ast.Constant)
        and training_calls[0].args[1].value is None
    )
    audit_source = audit_path.read_text(encoding="utf-8")
    upstream_contract = all(
        marker in audit_source
        for marker in (
            '"none_state_is_initialized": "if state == None:"',
            '"sequence_dispatch": "return self.forward_seq(idx, state, full_output)"',
            '"single_token_dispatch": "return self.forward_one(idx[0], state)"',
            '"zero_state_has_three_components_per_layer": (',
            '"state = [None for _ in range(self.args.n_layer * 3)]"',
        )
    )
    test_tree = ast.parse(test_path.read_text(encoding="utf-8"))
    wrapper_test = _function(
        test_tree, "test_new_wrapper_keeps_base_dictionary_unchanged_on_off_forward"
    )
    test_calls = [
        node for node in ast.walk(wrapper_test)
        if isinstance(node, ast.Call) and _attribute_call(node, "runtime", "forward")
    ]
    test_uses_prebuilt_state = bool(
        len(test_calls) == 1
        and len(test_calls[0].args) >= 2
        and isinstance(test_calls[0].args[1], ast.Call)
        and isinstance(test_calls[0].args[1].func, ast.Name)
        and test_calls[0].args[1].func.id == "_state"
    )
    checks = {
        "training_entry_passes_none_state": training_none,
        "wrapper_dispatches_directly_to_two_child_methods": len(direct_calls) == 2
        and {node.func.attr for node in direct_calls} == {"forward_one", "forward_seq"},
        "wrapper_has_no_zero_state_generation": not generator_calls,
        "frozen_upstream_contract_places_none_initialization_in_public_forward": upstream_contract,
        "existing_real_entry_wrapper_test_uses_prebuilt_state": test_uses_prebuilt_state,
        "existing_real_entry_wrapper_test_does_not_cover_none_state": bool(test_calls)
        and not any(
            len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value is None
            for call in test_calls
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "training_runtime_forward_line": training_calls[0].lineno if training_calls else None,
        "wrapper_direct_dispatch_lines": {
            node.func.attr: node.lineno for node in direct_calls
        },
        "wrapper_zero_state_call_count": len(generator_calls),
        "existing_wrapper_test_state_argument": "_state()" if test_uses_prebuilt_state else None,
    }


def _synthetic_namespace() -> tuple[dict[str, Any], type]:
    def cmix(x: int, state_value: int, layer_index: int) -> tuple[int, int]:
        del x, layer_index
        return 1, state_value + 1

    namespace: dict[str, Any] = {
        "RWKV_x070_CMix_one": cmix,
        "RWKV_x070_CMix_seq": cmix,
    }
    exec(SYNTHETIC_UPSTREAM_SOURCE, namespace)
    return namespace, namespace["RWKV_x070"]


def run_synthetic_dispatch_triangle(tokens: Sequence[int] = (3, 5, 8)) -> dict[str, Any]:
    before_rwkv = "rwkv.model" in sys.modules
    before_torch = "torch" in sys.modules
    namespace, fixture_type = _synthetic_namespace()

    public_fixture = fixture_type()
    public_output = public_fixture.forward(list(tokens), None, False)

    none_fixture = fixture_type()
    none_methods, none_counts = compile_instrumented_methods(
        upstream_source=SYNTHETIC_UPSTREAM_SOURCE,
        upstream_globals=namespace,
        rwkv_de_version=None,
    )
    none_runtime = D6DIIWrapperOwnedRuntime(
        base_model=none_fixture,
        compiled_methods=none_methods,
        injection_counts=none_counts,
    )
    none_error_type = None
    none_error = None
    try:
        none_runtime.forward(
            list(tokens), None, full_output=False,
            coupling=request_for_pilot_condition("wrapper_off"),
        )
    except BaseException as error:
        none_error_type = type(error).__name__
        none_error = str(error)

    prebuilt_fixture = fixture_type()
    prebuilt_methods, prebuilt_counts = compile_instrumented_methods(
        upstream_source=SYNTHETIC_UPSTREAM_SOURCE,
        upstream_globals=namespace,
        rwkv_de_version=None,
    )
    prebuilt_runtime = D6DIIWrapperOwnedRuntime(
        base_model=prebuilt_fixture,
        compiled_methods=prebuilt_methods,
        injection_counts=prebuilt_counts,
    )
    source_state = [0 for _ in range(96)]
    source_snapshot = copy.deepcopy(source_state)
    prebuilt_output = prebuilt_runtime.forward(
        list(tokens), source_state, full_output=False,
        coupling=request_for_pilot_condition("wrapper_off"),
    )
    checks = {
        "public_forward_none_initializes_and_succeeds": public_output is not None
        and public_fixture.zero_state_calls == 1,
        "wrapper_direct_sequence_none_reproduces_type_error_before_dispatch": none_error_type
        == "TypeError"
        and none_error == "'NoneType' object is not subscriptable"
        and none_runtime.dispatcher.dispatch_count == 0,
        "wrapper_none_does_not_call_public_initializer": none_fixture.zero_state_calls == 0,
        "wrapper_none_context_restored_and_base_stable": none_runtime.context_is_empty()
        and none_runtime.base_dictionary_is_stable()
        and none_runtime.owned_bindings_are_stable(),
        "wrapper_direct_sequence_prebuilt_state_succeeds": prebuilt_output is not None
        and prebuilt_runtime.dispatcher.dispatch_count == 32,
        "prebuilt_source_state_unchanged": source_state == source_snapshot,
        "prebuilt_wrapper_does_not_call_public_initializer": prebuilt_fixture.zero_state_calls == 0,
        "synthetic_fixture_imports_no_real_model_modules": before_rwkv
        == ("rwkv.model" in sys.modules)
        and before_torch == ("torch" in sys.modules),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "none_case": {
            "error_type": none_error_type,
            "error": none_error,
            "execution_count": none_runtime.execution_count,
            "dispatcher_calls": none_runtime.dispatcher.dispatch_count,
            "zero_state_calls": none_fixture.zero_state_calls,
        },
        "prebuilt_case": {
            "execution_count": prebuilt_runtime.execution_count,
            "dispatcher_calls": prebuilt_runtime.dispatcher.dispatch_count,
            "zero_state_calls": prebuilt_fixture.zero_state_calls,
        },
        "public_case": {"zero_state_calls": public_fixture.zero_state_calls},
    }


def build_diagnostic_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    expected_config = (root / CONFIG_RELATIVE_PATH).resolve()
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected_config:
        raise PermissionError("D6D failure diagnostic config path is not frozen")
    config = _object(supplied, "config")
    config_checks = validate_config(config)
    locks = config["frozen_source_locks"]
    lock_checks = {
        relative: sha256_file(_project_path(root, relative, "source lock")) == digest
        for relative, digest in locks.items()
    }
    if not all(lock_checks.values()):
        failed = [path for path, valid in lock_checks.items() if not valid]
        raise RuntimeError("D6D frozen source lock changed: " + ", ".join(failed))
    source_audit = audit_frozen_source(root)
    fixture = run_synthetic_dispatch_triangle()
    evidence = config["frozen_failure_evidence"]
    route = config["route_review"]
    checks = {
        "config_valid": all(config_checks.values()),
        "frozen_source_locks_valid": all(lock_checks.values()),
        "source_audit_valid": source_audit["valid"],
        "synthetic_dispatch_triangle_valid": fixture["valid"],
        "synthetic_error_matches_real_failure": fixture["none_case"]["error_type"]
        == evidence["failure_error_type"]
        and fixture["none_case"]["error"] == evidence["failure_error"],
        "failure_precedes_first_capture": evidence["training_captures_completed"] == 0
        and evidence["projection_artifact_constructed"] is False,
        "pilot_never_started": evidence["pilot_forward_calls_completed"] == 0,
        "claim_consumed_and_rerun_closed": evidence["claim_consumed"] is True
        and evidence["d6d_rerun_authorized"] is False
        and evidence["automatic_rerun_authorized"] is False,
        "root_cause_converges_on_skipped_public_initialization": source_audit["checks"][
            "training_entry_passes_none_state"
        ]
        and source_audit["checks"]["wrapper_has_no_zero_state_generation"]
        and source_audit["checks"][
            "frozen_upstream_contract_places_none_initialization_in_public_forward"
        ]
        and fixture["checks"]["public_forward_none_initializes_and_succeeds"]
        and fixture["checks"][
            "wrapper_direct_sequence_none_reproduces_type_error_before_dispatch"
        ],
        "existing_test_gap_identified": source_audit["checks"][
            "existing_real_entry_wrapper_test_uses_prebuilt_state"
        ]
        and source_audit["checks"][
            "existing_real_entry_wrapper_test_does_not_cover_none_state"
        ],
        "repair_and_same_plan_is_rerun": route[
            "repair_then_repeat_same_model_manifests_and_160_call_plan_is_d6d_rerun"
        ]
        is True,
        "d6e_remains_blocked": route["d6e_prerequisite_satisfied"] is False,
        "no_independent_successor_established": route[
            "independent_new_experiment_route_established"
        ]
        is False,
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D failure diagnostic failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": DIAGNOSTIC_VERSION,
        "status": "d6d_failure_state_initialization_offline_diagnostic_complete",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "source_lock_checks": lock_checks,
        "source_audit": source_audit,
        "synthetic_fixture": fixture,
        "frozen_failure_evidence": evidence,
        "findings": {
            "root_cause": "wrapper_direct_child_dispatch_skips_public_forward_zero_state_initialization",
            "test_gap": "prebuilt_state_fixture_did_not_cover_real_entry_state_none_semantics",
            "d6d_decision": route["decision"],
            "independent_successor_established": False,
            "minimum_future_independence_requirements": route[
                "minimum_future_independence_requirements"
            ],
        },
        "next_gate": NEXT_GATE,
        "safety": {
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_runner_modified": False,
            "d6d_fix_implemented": False,
            "d6d_rerun": False,
            "d6e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "split_mechanism_run": False,
            "automatic_rerun_authorized": False,
            "historical_failure_conclusion_changed": False,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
