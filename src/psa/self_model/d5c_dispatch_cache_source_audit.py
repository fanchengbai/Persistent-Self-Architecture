from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


AUDIT_VERSION = "0.1-d5c-dispatch-cache-source-offline"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_dispatch_cache_source_audit.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D5C失败纯离线源码级dispatch/cache边界审计；"
    "只审计冻结AST变换、wrapper源码、decorator/方法分派和现有fake，不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型、不实现或授权修复后重跑，也不授权D5D/D5E、"
    "正式测试集、Self效果、真实Self projection、Self Updater或自动重跑。"
)
CLASSIFICATION = (
    "ast_object_mutation_excluded_wrapper_protocol_asymmetry_and_decorator_"
    "coverage_gap_confirmed_cache_root_cause_unresolved"
)
TRUE_AUTHORITY_FIELDS = {
    "source_audit_authorized",
    "existing_fake_audit_authorized",
    "existing_report_observation_authorized",
}
FALSE_AUTHORITY_FIELDS = {
    "fix_implementation_authorized",
    "rwkv_import_authorized",
    "torch_import_authorized",
    "weights_access_authorized",
    "model_load_authorized",
    "model_execution_authorized",
    "d4_rerun_authorized",
    "d4b_rerun_authorized",
    "d5c_rerun_authorized",
    "d5d_authorized",
    "d5e_authorized",
    "formal_test_set_authorized",
    "self_effect_conclusion_authorized",
    "real_self_projection_authorized",
    "self_updater_authorized",
    "automatic_rerun_authorized",
}
REAL_REPORT_DIGEST = "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21"
LIFECYCLE_REPORT_DIGEST = (
    "3dfa640d32bfbea2594e7d58afb6c71552b8fc8619a215d72ae8b23e5e0a4150"
)
UPSTREAM_SOURCE_DIGEST = (
    "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_dispatch_cache_source_audit.md",
    "docs/self_model_v0_1_d4a_cloud_static_observation.md",
    "docs/self_model_v0_1_coupling_d5c_real_mechanism_observation.md",
    "scripts/verify_self_model_v0_1_d5c_dispatch_cache_source_audit.py",
    "src/psa/self_model/d5c_dispatch_cache_source_audit.py",
    "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "src/psa/self_model/rwkv_interface_audit.py",
    "tests/test_self_model_d5c_dispatch_cache_source_audit.py",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C dispatch/cache source audit config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    evidence = config.get("frozen_evidence")
    authority = config.get("authority")
    if not isinstance(evidence, Mapping) or not isinstance(authority, Mapping):
        raise ValueError("D5C dispatch/cache source audit config is incomplete")
    checks = {
        "identity_exact": config.get("audit_version") == AUDIT_VERSION
        and config.get("stage") == "Coupling-D5C_failure_dispatch_cache_source_audit"
        and config.get("status") == "source_audit_authorized_no_model_no_fix"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "upstream_lock_exact": evidence.get("rwkv_package_version") == "0.8.32"
        and evidence.get("rwkv_model_source_sha256") == UPSTREAM_SOURCE_DIGEST
        and evidence.get("rwkv_jit_on") == "0",
        "decorator_boundary_frozen": evidence.get("real_original_method_decorators")
        == {"forward_one": ["MyFunction"], "forward_seq": ["MyFunction"]}
        and evidence.get("recompiled_method_decorators")
        == {"forward_one": [], "forward_seq": []},
        "dispatch_markers_frozen": evidence.get("public_forward_dispatch_markers")
        == [
            "return self.forward_seq(idx, state, full_output)",
            "return self.forward_one(idx[0], state)",
        ],
        "failure_evidence_exact": evidence.get("d5c_real_report_sha256")
        == REAL_REPORT_DIGEST
        and evidence.get("d5c_real_status") == "d5c_mechanism_smoke_failed"
        and evidence.get("lifecycle_diagnostic_report_sha256")
        == LIFECYCLE_REPORT_DIGEST,
        "count_signature_exact": evidence.get("post_active_original_calls") == 8
        and evidence.get("extra_callback_invocations") == 256
        and evidence.get("extra_probe_applications") == 8
        and evidence.get("layers_per_call") == 32,
        "classification_exact": config.get("required_classification")
        == CLASSIFICATION,
        "authority_exact": set(authority)
        == TRUE_AUTHORITY_FIELDS | FALSE_AUTHORITY_FIELDS
        and all(authority.get(name) is True for name in TRUE_AUTHORITY_FIELDS)
        and all(authority.get(name) is False for name in FALSE_AUTHORITY_FIELDS),
        "next_gate_exact": config.get("next_gate")
        == "separate_offline_boundary_fixture_or_fix_design_requires_owner_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C source audit config failed closed: " + ", ".join(failed))
    return checks


def _function(tree: ast.AST, class_name: str | None, function_name: str) -> ast.FunctionDef:
    scope: ast.AST = tree
    if class_name is not None:
        classes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise RuntimeError(f"expected one {class_name} class")
        scope = classes[0]
    matches = [
        node for node in ast.walk(scope)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {function_name} function")
    return matches[0]


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == name)
             or (isinstance(node.func, ast.Attribute) and node.func.attr == name))
    ]


def _source_boundary_analysis(root: Path) -> dict[str, Any]:
    transform_path = root / "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
    wrapper_path = root / "src/psa/self_model/d5c_mechanism_runtime.py"
    fake_path = root / "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py"
    interface_path = root / "src/psa/self_model/rwkv_interface_audit.py"
    transform_source = transform_path.read_text(encoding="utf-8")
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    fake_source = fake_path.read_text(encoding="utf-8")
    interface_source = interface_path.read_text(encoding="utf-8")
    transform_tree = ast.parse(transform_source)
    wrapper_tree = ast.parse(wrapper_source)
    fake_tree = ast.parse(fake_source)

    plan = _function(transform_tree, None, "_build_instrumented_method_plan")
    compile_methods = _function(transform_tree, None, "compile_instrumented_methods")
    wrapper_forward = _function(wrapper_tree, "RWKV7D5CActiveRuntime", "forward")
    restore_bindings = _function(wrapper_tree, None, "_restore_bindings")
    verify_bindings = _function(wrapper_tree, None, "_verify_restored_bindings")

    parse_calls = _calls(plan, "parse")
    exec_calls = _calls(compile_methods, "exec")
    setattr_calls = _calls(wrapper_forward, "setattr")
    pop_calls = _calls(wrapper_forward, "pop")
    delattr_calls = _calls(wrapper_forward, "delattr")
    getattr_calls = _calls(wrapper_forward, "getattr")
    restore_delattr_calls = _calls(restore_bindings, "delattr")
    verify_getattr_calls = _calls(verify_bindings, "getattr")
    restore_helper_calls = _calls(wrapper_forward, "_restore_bindings")
    verify_helper_calls = _calls(wrapper_forward, "_verify_restored_bindings")
    decorator_clears = [
        node for node in ast.walk(plan)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "decorator_list"
                for target in node.targets)
        and isinstance(node.value, ast.List) and not node.value.elts
    ]
    copied_globals = [
        node for node in ast.walk(compile_methods)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    ]
    method_type_calls = _calls(wrapper_forward, "MethodType")

    fake_source_values = [
        node.value.value for node in ast.walk(fake_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DIAGNOSTIC_SOURCE"
                for target in node.targets)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    ]
    if len(fake_source_values) != 1:
        raise RuntimeError("existing lifecycle fake source was not found")
    embedded_fake_tree = ast.parse(fake_source_values[0])
    fake_methods = [
        node for node in ast.walk(embedded_fake_tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"forward_one", "forward_seq"}
    ]
    fake_decorators = {
        node.name: [ast.unparse(item) for item in node.decorator_list]
        for node in fake_methods
    }
    source_inventory = "\n".join(
        (transform_source, wrapper_source, fake_source, interface_source)
    )
    myfunction_definitions = any(
        marker in source_inventory
        for marker in ("def MyFunction", "MyFunction =", "class MyFunction")
    )
    public_dispatch_markers = {
        "sequence": '"sequence_dispatch": "return self.forward_seq(idx, state, full_output)"'
        in interface_source,
        "single": '"single_token_dispatch": "return self.forward_one(idx[0], state)"'
        in interface_source,
    }

    return {
        "fresh_ast_parse_present": len(parse_calls) == 1,
        "decorator_lists_cleared": len(decorator_clears) == 1,
        "copied_globals_namespace_present": bool(copied_globals),
        "independent_exec_compile_present": len(exec_calls) == 1,
        "wrapper_setattr_count": len(setattr_calls),
        "wrapper_methodtype_present": bool(method_type_calls),
        "wrapper_direct_dict_pop_count": len(pop_calls),
        "wrapper_delattr_count": len(delattr_calls),
        "wrapper_getattr_count": len(getattr_calls),
        "restore_delattr_count": len(restore_delattr_calls),
        "verify_getattr_count": len(verify_getattr_calls),
        "restore_helper_called": len(restore_helper_calls) == 1,
        "verify_helper_called": len(verify_helper_calls) == 1,
        "fake_method_decorators": fake_decorators,
        "myfunction_definition_in_audited_sources": myfunction_definitions,
        "public_forward_dynamic_dispatch_markers": public_dispatch_markers,
        "source_level_implications": {
            "ast_transform_can_mutate_loaded_original_method_objects": not (
                len(parse_calls) == 1 and len(exec_calls) == 1
            ),
            "installation_and_cleanup_use_symmetric_object_protocol": bool(
                setattr_calls and restore_delattr_calls and not pop_calls
            ),
            "cleanup_verifies_resolved_method_identity_after_restore": bool(
                verify_getattr_calls and verify_helper_calls
            ),
            "cleanup_synchronizes_unknown_framework_or_decorator_caches": False,
            "existing_fake_covers_real_decorator_boundary": any(
                fake_decorators.values()
            ),
        },
    }


def build_source_audit_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C dispatch/cache source audit config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    analysis = _source_boundary_analysis(root)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    implications = analysis["source_level_implications"]
    checks = {
        "config_valid": all(config_checks.values()),
        "fresh_ast_is_independent_of_loaded_methods": analysis["fresh_ast_parse_present"]
        and analysis["independent_exec_compile_present"],
        "compiled_methods_remove_decorators": analysis["decorator_lists_cleared"],
        "compiled_namespace_copies_upstream_globals": analysis[
            "copied_globals_namespace_present"
        ],
        "wrapper_installs_bound_methods_with_setattr": analysis["wrapper_setattr_count"] == 2
        and analysis["wrapper_methodtype_present"],
        "historical_direct_pop_removed_from_wrapper": analysis[
            "wrapper_direct_dict_pop_count"
        ] == 0,
        "transactional_restore_uses_delattr": analysis["restore_delattr_count"] == 1
        and analysis["restore_helper_called"],
        "transactional_cleanup_verifies_resolution": analysis["verify_getattr_count"] >= 2
        and analysis["verify_helper_called"],
        "public_forward_dynamically_dispatches_both_paths": all(
            analysis["public_forward_dynamic_dispatch_markers"].values()
        ),
        "existing_fake_has_plain_undecorated_methods": analysis["fake_method_decorators"]
        == {"forward_one": [], "forward_seq": []},
        "myfunction_definition_not_in_audited_sources": not analysis[
            "myfunction_definition_in_audited_sources"
        ],
        "ast_loaded_object_mutation_excluded": not implications[
            "ast_transform_can_mutate_loaded_original_method_objects"
        ],
        "historical_object_protocol_asymmetry_now_closed_in_source": implications[
            "installation_and_cleanup_use_symmetric_object_protocol"
        ],
        "decorator_coverage_gap_confirmed": not implications[
            "existing_fake_covers_real_decorator_boundary"
        ],
        "cache_root_cause_not_claimed": not implications[
            "cleanup_synchronizes_unknown_framework_or_decorator_caches"
        ],
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C dispatch/cache source audit failed: " + ", ".join(failed))
    report = {
        "audit_version": AUDIT_VERSION,
        "status": "d5c_dispatch_cache_source_audit_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "boundary_analysis": analysis,
        "findings": {
            "confirmed": [
                "fresh_ast_compilation_cannot_mutate_loaded_original_method_objects",
                "active_compiled_methods_remove_real_myfunction_decorators",
                "historical_wrapper_used_setattr_then_direct_instance_dict_pop",
                "current_wrapper_uses_protocol_restore_and_resolution_verification",
                "existing_plain_python_fake_does_not_cover_real_decorator_descriptor_boundary",
                "public_forward_source_markers_resolve_forward_one_or_forward_seq_dynamically",
            ],
            "risk_not_root_cause": [
                "setattr_and_direct_dict_pop_object_protocol_asymmetry",
                "compiled_undecorated_to_class_decorated_method_boundary",
                "post_active_schedule_cannot_separate_route_from_predecessor_state",
            ],
            "unresolved_due_to_missing_frozen_source": [
                "myfunction_definition_and_runtime_behavior",
                "torch_nn_module_attribute_protocol_internals",
                "any_descriptor_or_compilation_cache_state_after_active",
            ],
            "not_supported": [
                "one_proven_cache_or_dispatch_root_cause",
                "a_fix_design",
                "permission_to_rerun_d5c",
                "d5c_pass_or_self_effect_conclusion",
            ],
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "runtime_modified": False,
            "fix_implemented": False,
            "existing_real_report_reexecuted": False,
            "d4_rerun": False,
            "d4b_rerun": False,
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
