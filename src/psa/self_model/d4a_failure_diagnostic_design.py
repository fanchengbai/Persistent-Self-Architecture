from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-d4a-failure-diagnostic-design"
DESIGN_CONFIG = (
    "configs/development/self_model_v0_1_d4a_failure_diagnostic_design.json"
)
LOCKED_FILES = (
    DESIGN_CONFIG,
    "docs/self_model_v0_1_d4_failure_observation.md",
    "docs/self_model_v0_1_d4a_failure_diagnostic_design.md",
    "scripts/verify_self_model_v0_1_d4a_failure_diagnostic_design.py",
    "src/psa/self_model/d4_real_off_equivalence.py",
    "src/psa/self_model/d4a_failure_diagnostic_design.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d4a_failure_diagnostic_design.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("D4A design must be an object")
    return payload


def _dict_literal_order(source: str, function_name: str, variable_name: str) -> list[str]:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets)
        and isinstance(node.value, ast.Dict)
    )
    return [key.value for key in assignment.value.keys if isinstance(key, ast.Constant)]


def _cell_ids_from_runtime() -> list[str]:
    return [
        "forward_one__none__full_output_false",
        "forward_one__restored__full_output_false",
        "forward_seq__none__full_output_false",
        "forward_seq__none__full_output_true",
        "forward_seq__restored__full_output_false",
        "forward_seq__restored__full_output_true",
    ]


def validate_d4a_design(
    *,
    design: Mapping[str, Any],
    d4_source: str,
    off_g2_source: str,
) -> dict[str, bool]:
    evidence = design.get("d4_failure_evidence")
    findings = design.get("offline_audit_findings")
    diagnostic = design.get("minimal_future_diagnostic")
    stop = design.get("future_stop_rules")
    authority = design.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (evidence, findings, diagnostic, stop, authority)
    ):
        raise ValueError("D4A design structure is incomplete")

    route_order = _dict_literal_order(
        d4_source, "execute_equivalence_matrix", "routes"
    )
    expected_rounds = [
        ["original_baseline", "g0_recompiled_unmodified", "off_g2_instrumented"],
        ["g0_recompiled_unmodified", "off_g2_instrumented", "original_baseline"],
        ["off_g2_instrumented", "original_baseline", "g0_recompiled_unmodified"],
    ]
    positions = {
        route: [round_.index(route) for round_ in expected_rounds]
        for route in expected_rounds[0]
    }
    expected_authority = {
        "offline_failure_audit_authorized": True,
        "diagnostic_design_authorized": True,
        "diagnostic_runtime_implementation_authorized": False,
        "rwkv_model_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "diagnostic_result_observation_authorized": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    checks = {
        "design_is_offline_only": (
            design.get("design_version") == DESIGN_VERSION
            and design.get("stage") == "D4A_post_failure_minimal_diagnostic_design"
            and design.get("status") == "design_only_unimplemented_unexecuted"
            and design.get("development_only") is True
        ),
        "d4_failure_identity_frozen": (
            evidence.get("executed_git_commit")
            == "a4d110c5de5a8e638137c0559a15de8941172eed"
            and evidence.get("report_digest_sha256")
            == "39d4611a6d50791f1677f9eb27e6fb2ea702151a26236fa4094699b821ca721a"
            and evidence.get("execution_claim_sha256")
            == "2900bf111031f878bf18004d3f7123439fdfea62dabf8da0a67eefb54e7479de"
            and evidence.get("valid") is False
            and evidence.get("passing_cell_count") == 5
            and evidence.get("total_cell_count") == 6
            and evidence.get("failed_cell_id")
            == "forward_one__none__full_output_false"
            and evidence.get("failed_state_component_count") == 92
            and evidence.get("automatic_rerun_authorized") is False
        ),
        "d4_cell_order_audited": findings.get("d4_cell_order")
        == _cell_ids_from_runtime(),
        "d4_route_order_audited": (
            route_order
            == ["original_baseline", "off_g1_passthrough", "off_g2_instrumented"]
            and findings.get("d4_route_order") == route_order
        ),
        "d4_warmup_is_unobserved": (
            findings.get("warmup_outputs_recorded") is False
            and findings.get("within_route_repeatability_recorded") is False
            and '_invoke(route, cell["tokens"], warm_state' in d4_source
            and 'outputs[route_name] = _invoke(' in d4_source
        ),
        "d4_report_lacks_numeric_localization": (
            findings.get("numeric_error_magnitude_recorded") is False
            and findings.get("per_tensor_content_digest_recorded") is False
            and "max_abs_error" not in d4_source
            and "unequal_element_count" not in d4_source
        ),
        "state_mode_and_route_age_are_confounded": findings.get(
            "state_mode_and_route_age_separated"
        )
        is False,
        "off_g2_recompile_boundary_audited": (
            findings.get("off_g2_recompiles_selected_method_ast") is True
            and findings.get("off_g2_removes_method_decorators") is True
            and findings.get("off_g2_copies_upstream_globals") is True
            and "method.decorator_list = []" in off_g2_source
            and "namespace = dict(upstream_globals)" in off_g2_source
            and 'compile(module, "<psa-rwkv7-instrumented-off>", "exec")'
            in off_g2_source
        ),
        "off_g2_binding_boundary_audited": (
            findings.get("off_g2_temporarily_binds_both_methods") is True
            and findings.get("off_g2_temporarily_sets_none_callback") is True
            and "managed_names = (*TARGET_METHODS, CALLBACK_ATTRIBUTE)"
            in off_g2_source
            and "setattr(self._base_model, CALLBACK_ATTRIBUTE, None)"
            in off_g2_source
            and "types.MethodType(function, self._base_model)" in off_g2_source
        ),
        "diagnostic_scope_is_failed_cell_only": (
            diagnostic.get("non_core_token_ids") == [2764]
            and diagnostic.get("state_input") == "none"
            and diagnostic.get("full_output") is False
            and diagnostic.get("diagnostic_only_not_d4_retest") is True
        ),
        "g0_is_a_binding_boundary_control": diagnostic.get("g0_boundary")
        == (
            "same_selected_ast_decorator_removal_copied_globals_and_temporary_"
            "binding_as_g2_but_no_callback_attribute_or_injected_branch"
        ),
        "latin_schedule_is_balanced": (
            diagnostic.get("recorded_rounds") == expected_rounds
            and all(sorted(value) == [0, 1, 2] for value in positions.values())
            and diagnostic.get("model_forward_call_count") == 9
            and diagnostic.get("discarded_warmup_call_count") == 0
            and diagnostic.get("each_route_occupies_each_order_position_once") is True
        ),
        "all_outputs_and_differences_will_be_recorded": (
            diagnostic.get("required_per_call_records")
            == [
                "route",
                "round_index",
                "order_position",
                "logits_shape_dtype_device",
                "logits_sha256",
                "state_component_count",
                "state_component_shape_dtype_device",
                "state_component_sha256",
            ]
            and diagnostic.get("required_comparisons")
            == [
                "all_within_route_pairs_torch_equal",
                "all_cross_route_pairs_torch_equal",
                "unequal_element_count",
                "max_abs_error",
                "mean_abs_error",
                "first_mismatch_component",
            ]
        ),
        "diagnostic_cannot_upgrade_d4_or_d5": (
            diagnostic.get("interpretation_only") is True
            and diagnostic.get("cannot_change_d4_status") is True
            and diagnostic.get("cannot_authorize_d5") is True
        ),
        "stop_rules_fail_closed": dict(stop)
        == {
            "always_stop_after_one_complete_or_failed_attempt": True,
            "no_automatic_rerun": True,
            "no_tolerance_as_pass_criterion": True,
            "no_top1_as_pass_criterion": True,
            "no_active_injection": True,
            "no_self_effect_experiment": True,
        },
        "authority_is_design_only": dict(authority) == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D4A design failed closed: " + ", ".join(failed))
    return checks


def build_d4a_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = Path(config_path).resolve()
    if config != (root / DESIGN_CONFIG).resolve():
        raise PermissionError("D4A design config path is not frozen")
    design = _load_object(config)
    d4_source = (root / "src/psa/self_model/d4_real_off_equivalence.py").read_text(
        encoding="utf-8"
    )
    off_g2_source = (
        root / "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
    ).read_text(encoding="utf-8")
    checks = validate_d4a_design(
        design=design, d4_source=d4_source, off_g2_source=off_g2_source
    )
    report = {
        "report_version": DESIGN_VERSION,
        "status": "d4a_failure_diagnostic_design_static_verified",
        "valid": all(checks.values()),
        "development_only": True,
        "checks": checks,
        "source_digests": {
            relative: sha256_file(root / relative) for relative in LOCKED_FILES
        },
        "future_diagnostic": design["minimal_future_diagnostic"],
        "authority": design["authority"],
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "diagnostic_runtime_implemented": False,
            "d4_rerun": False,
            "active_injection_implemented": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
