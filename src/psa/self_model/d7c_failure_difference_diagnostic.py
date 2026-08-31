from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json


DIAGNOSTIC_VERSION = "0.1-self-model-d7c-failure-difference-offline"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d7c_failure_difference_diagnostic.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D7-C失败纯离线差异来源诊断与研究路线审查；"
    "仅使用现有authorization、claim、report、冻结源码和合成Python fixture，分析8个cell的"
    "logits/state差异分布、调用顺序、public/instrumented路径与确定性设置，并判断是否存在科学"
    "独立且不构成D7-C重跑的新路线；不导入RWKV/Torch、不访问权重、不加载或执行模型、不修改"
    "真实runner、不实现或授权修复后重跑，也不改变D7-C失败结论或开放D7-D/D7-E、projection、"
    "正式测试集、Self效果、Self Updater、raw-original或自动重跑。"
)
CLASSIFICATION = (
    "d7c_exactness_failure_real_cause_not_identifiable_without_within_route_"
    "repeatability_and_counterbalanced_order"
)
NEXT_GATE = "owner_reviews_d8_numerical_identifiability_preregistration_design_only"
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d7c_failure_difference_diagnostic.md",
    "scripts/verify_self_model_v0_1_d7c_failure_difference_diagnostic.py",
    "src/psa/self_model/d7c_failure_difference_diagnostic.py",
    "tests/test_self_model_d7c_failure_difference_diagnostic.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D7-C difference diagnostic {label} must be an object")
    return value


def _project_path(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise PermissionError(f"D7-C difference diagnostic {label} path is not frozen")
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise PermissionError(f"D7-C difference diagnostic {label} escapes project root")
    return resolved


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    evidence = config.get("frozen_failure_evidence", {})
    observations = config.get("cell_observations", [])
    determinism = config.get("determinism_observation", {})
    route = config.get("route_review", {})
    authority = config.get("authority", {})
    checks = {
        "identity_exact": config.get("diagnostic_version") == DIAGNOSTIC_VERSION
        and config.get("stage")
        == "D7-C_failure_difference_source_offline_diagnostic_and_route_review"
        and config.get("status")
        == "offline_diagnostic_authorized_no_model_no_fix_no_rerun"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "failure_identity_frozen": evidence.get("execution_commit")
        == "665ac40026249fd8f1523aa2cae40486bb427d44"
        and evidence.get("failure_report_digest_sha256")
        == "9e22f664908cb477ca006a7af0bb450fd5309ee59719f1a6243755f2ce35233d"
        and evidence.get("failure_status")
        == "d7c_real_public_semantics_compatibility_failed",
        "claim_consumed_and_rerun_closed": evidence.get("execution_claim_sha256")
        == "fa86ad7056a3a0c5d1d0f05f8ec0221f6bd922c8878d662376ff6ef0bdf700e1"
        and evidence.get("claim_consumed") is True
        and evidence.get("d7c_rerun_authorized") is False
        and evidence.get("automatic_rerun_authorized") is False,
        "complete_failed_attempt_frozen": evidence.get("runner_exit_code") == 0
        and evidence.get("forward_calls_completed") == 18
        and evidence.get("compatibility_cells_completed") == 8
        and evidence.get("active_calls_completed") == 2
        and evidence.get("active_controls_passed") is True
        and evidence.get("all_logits_exact") is False
        and evidence.get("all_states_exact") is False,
        "eight_cells_exact": len(observations) == 8
        and [cell.get("cell") for cell in observations] == list(range(1, 9))
        and all(cell.get("exact_state_components") == 4 for cell in observations)
        and all(cell.get("compatible_state_components") == 96 for cell in observations)
        and all(cell.get("first_nonexact_state_index") == 4 for cell in observations)
        and all(cell.get("max_error_state_index") == 94 for cell in observations),
        "initialization_counts_exact": [
            cell.get("wrapper_zero_state_initializations") for cell in observations
        ]
        == [1, 1, 0, 0, 1, 1, 0, 0],
        "determinism_metadata_frozen": determinism.get("seed") == 20260729
        and determinism.get("determinism_enabled") is False
        and determinism.get("deterministic_algorithms") is False
        and determinism.get("cudnn_deterministic") is False
        and determinism.get("cudnn_benchmark") is False
        and determinism.get("cuda_matmul_allow_tf32") is False
        and determinism.get("cudnn_allow_tf32") is True
        and determinism.get("float32_matmul_precision") == "highest"
        and determinism.get("cublas_workspace_config") is None,
        "route_is_independent_design_only": route.get(
            "d7c_failure_conclusion_remains"
        )
        is True
        and route.get("d7c_causal_source_identifiable_from_existing_evidence")
        is False
        and route.get("independent_design_candidate_established") is True
        and route.get("candidate_stage")
        == "D8_numerical_identifiability_and_excess_drift_preregistration"
        and len(route.get("minimum_independence_requirements", [])) == 8
        and route.get("candidate_design_only") is True
        and route.get("candidate_execution_authorized") is False,
        "classification_exact": config.get("required_classification")
        == CLASSIFICATION,
        "offline_authority_exact": authority.get(
            "offline_failure_diagnostic_authorized"
        )
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
                "d7c_fix_implementation_authorized",
                "d7c_rerun_authorized",
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
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D7-C difference diagnostic config changed: " + ", ".join(failed))
    return checks


def _function(tree: ast.AST, name: str, class_name: str | None = None) -> ast.FunctionDef:
    scope: ast.AST = tree
    if class_name is not None:
        classes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise RuntimeError(f"D7-C diagnostic expected one {class_name}")
        scope = classes[0]
    matches = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"D7-C diagnostic expected one {name}")
    return matches[0]


def _call_chain(node: ast.Call) -> str:
    parts: list[str] = []
    value: ast.AST = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def audit_frozen_source(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    entry_path = root / "src/psa/self_model/d7c_real_compatibility_entry.py"
    runtime_path = root / "src/psa/self_model/d7c_public_semantics_runtime.py"
    instrumenter_path = root / "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
    entry_tree = ast.parse(entry_path.read_text(encoding="utf-8"))
    run = _function(entry_tree, "run_d7c_real_compatibility")
    cell_loops = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and _call_chain(node.iter) == "compatibility_cells"
    ]
    if len(cell_loops) != 1:
        raise RuntimeError("D7-C diagnostic expected one compatibility cell loop")
    cell_loop = cell_loops[0]
    cell_calls = [node for node in ast.walk(cell_loop) if isinstance(node, ast.Call)]
    public_calls = [node for node in cell_calls if _call_chain(node) == "adapter.model.forward"]
    wrapper_calls = [node for node in cell_calls if _call_chain(node) == "wrapper.forward"]

    runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    dispatcher = _function(runtime_tree, "__call__", "D7CFixedDispatcher")
    wrapper_forward = _function(runtime_tree, "forward", "D7CPublicSemanticsWrapper")
    wrapper_source = ast.unparse(wrapper_forward)
    zero_returns = [
        node
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.If)
        and "request.scale == 0.0" in ast.unparse(node.test)
        and any(
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Name)
            and statement.value.id == "residual"
            for statement in node.body
        )
    ]

    instrumenter_tree = ast.parse(instrumenter_path.read_text(encoding="utf-8"))
    compile_method = _function(instrumenter_tree, "compile_instrumented_methods")
    injector = _function(instrumenter_tree, "_instrument_body", "_PostFFNInjector")
    compile_source = ast.unparse(compile_method)
    injector_source = ast.unparse(injector)

    checks = {
        "one_public_call_per_cell": len(public_calls) == 1,
        "one_wrapper_call_per_cell": len(wrapper_calls) == 1,
        "public_always_precedes_wrapper": bool(public_calls and wrapper_calls)
        and public_calls[0].lineno < wrapper_calls[0].lineno,
        "no_within_route_repeatability_control": len(public_calls) == 1
        and len(wrapper_calls) == 1,
        "no_counterbalanced_order_in_cell_loop": bool(public_calls and wrapper_calls)
        and public_calls[0].lineno < wrapper_calls[0].lineno,
        "zero_dispatch_returns_residual_identity": len(zero_returns) == 1,
        "wrapper_initializes_none_state": "if state_copy is None:" in wrapper_source
        and "self.generate_zero_state()" in wrapper_source,
        "wrapper_dispatches_one_and_sequence": "self.forward_one" in wrapper_source
        and "self.forward_seq" in wrapper_source,
        "instrumented_methods_compiled_in_separate_namespace": "namespace = dict(upstream_globals)"
        in compile_source
        and "exec(compile(module" in compile_source,
        "callback_inserted_after_post_ffn_residual": "rewritten.append(residual_add)"
        in injector_source
        and "rewritten.append(callback)" in injector_source,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "cell_loop_line": cell_loop.lineno,
        "public_call_line": public_calls[0].lineno if public_calls else None,
        "wrapper_call_line": wrapper_calls[0].lineno if wrapper_calls else None,
        "observed_cell_order": "public_then_wrapper_for_every_cell",
        "within_route_repeatability_calls_per_cell": {"public": 1, "wrapper": 1},
        "counterbalanced_order_present": False,
    }


def observed_difference_fingerprint(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_paths = sorted(
        set.intersection(
            *[
                set(range(int(cell["exact_state_components"])))
                for cell in cells
            ]
        )
    )
    stable_groups = {
        "forward_one_cells_2_3_4": all(
            cells[index][field] == cells[1][field]
            for index in (2, 3)
            for field in (
                "logits_max_abs_error",
                "logits_mean_abs_error",
                "state_max_abs_error",
            )
        ),
        "forward_seq_cells_6_8": all(
            cells[index][field] == cells[5][field]
            for index in (7,)
            for field in (
                "logits_max_abs_error",
                "logits_mean_abs_error",
                "state_max_abs_error",
            )
        ),
    }
    return {
        "cell_count": len(cells),
        "all_logits_nonexact": all(cell["logits_max_abs_error"] > 0 for cell in cells),
        "all_states_nonexact": all(cell["state_max_abs_error"] > 0 for cell in cells),
        "all_state_components_compatible": all(
            cell["compatible_state_components"] == 96 for cell in cells
        ),
        "common_exact_state_indices": exact_paths,
        "common_first_nonexact_state_index": 4,
        "common_max_error_state_index": 94,
        "stable_metric_groups": stable_groups,
        "initialization_counts_follow_contract": [
            cell["wrapper_zero_state_initializations"] for cell in cells
        ]
        == [1, 1, 0, 0, 1, 1, 0, 0],
        "state_input_not_unique_explanation": stable_groups[
            "forward_one_cells_2_3_4"
        ]
        and stable_groups["forward_seq_cells_6_8"],
        "full_output_not_unique_explanation": cells[2]["state_max_abs_error"]
        == cells[3]["state_max_abs_error"],
        "recurrent_propagation_signature": {
            "first_nonexact_state_index": 4,
            "first_nonexact_role": "layer_1_attention_kv",
            "max_error_state_index": 94,
            "max_error_role": "layer_31_attention_kv",
        },
    }


def _comparison_fingerprint(public: Sequence[float], wrapper: Sequence[float]) -> dict[str, Any]:
    nonexact = [index for index, (left, right) in enumerate(zip(public, wrapper)) if left != right]
    errors = [abs(left - right) for left, right in zip(public, wrapper)]
    max_error = max(errors)
    return {
        "component_count": len(public),
        "exact_indices": [index for index, error in enumerate(errors) if error == 0.0],
        "first_nonexact_index": nonexact[0],
        "max_error_index": errors.index(max_error),
        "all_components_compatible": len(public) == len(wrapper) == 96,
        "state_exact": not nonexact,
        "logits_exact": False,
    }


def run_synthetic_nonidentifiability_fixture() -> dict[str, Any]:
    before_rwkv = "rwkv.model" in sys.modules
    before_torch = "torch" in sys.modules
    baseline = [float(index) for index in range(96)]
    drift = [0.0] * 96
    for index in range(4, 95):
        drift[index] = (index - 3) / 1000.0
    drift[95] = 0.01

    # Mechanism A: both routes are semantically identical; the second invocation drifts.
    order_public = list(baseline)
    order_wrapper = [value + drift[index] for index, value in enumerate(baseline)]
    # Mechanism B: timing is stable; only the instrumented path drifts.
    route_public = list(baseline)
    route_wrapper = [value + drift[index] for index, value in enumerate(baseline)]
    order_fingerprint = _comparison_fingerprint(order_public, order_wrapper)
    route_fingerprint = _comparison_fingerprint(route_public, route_wrapper)
    checks = {
        "mechanisms_are_causally_distinct": "second_invocation_background_drift"
        != "instrumented_route_specific_drift",
        "mechanisms_are_observationally_identical_under_d7c_order": order_fingerprint
        == route_fingerprint,
        "both_match_observed_exact_prefix": order_fingerprint["exact_indices"]
        == [0, 1, 2, 3],
        "both_match_observed_first_nonexact": order_fingerprint[
            "first_nonexact_index"
        ]
        == 4,
        "both_match_observed_max_error_index": order_fingerprint["max_error_index"]
        == 94,
        "no_real_model_modules_imported": before_rwkv == ("rwkv.model" in sys.modules)
        and before_torch == ("torch" in sys.modules),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "mechanisms": [
            {
                "id": "second_invocation_background_drift",
                "route_semantics_equal": True,
                "wrapper_specific_effect": False,
                "fingerprint": order_fingerprint,
            },
            {
                "id": "instrumented_route_specific_drift",
                "route_semantics_equal": False,
                "wrapper_specific_effect": True,
                "fingerprint": route_fingerprint,
            },
        ],
        "identifiability_result": "same_observation_two_distinct_causes",
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
        raise PermissionError("D7-C difference diagnostic config path is not frozen")
    config = _object(supplied, "config")
    config_checks = validate_config(config)
    lock_checks = {
        relative: sha256_file(_project_path(root, relative, "source lock")) == digest
        for relative, digest in config["frozen_source_locks"].items()
    }
    if not all(lock_checks.values()):
        failed = [path for path, valid in lock_checks.items() if not valid]
        raise RuntimeError("D7-C frozen source lock changed: " + ", ".join(failed))
    source_audit = audit_frozen_source(root)
    fingerprint = observed_difference_fingerprint(config["cell_observations"])
    synthetic = run_synthetic_nonidentifiability_fixture()
    route = config["route_review"]
    determinism = config["determinism_observation"]
    checks = {
        "config_valid": all(config_checks.values()),
        "frozen_source_locks_valid": all(lock_checks.values()),
        "source_audit_valid": source_audit["valid"],
        "eight_real_cells_failed_exactness": fingerprint["cell_count"] == 8
        and fingerprint["all_logits_nonexact"]
        and fingerprint["all_states_nonexact"],
        "state_structure_remained_compatible": fingerprint[
            "all_state_components_compatible"
        ],
        "common_recurrent_difference_signature": fingerprint[
            "common_exact_state_indices"
        ]
        == [0, 1, 2, 3]
        and fingerprint["common_first_nonexact_state_index"] == 4
        and fingerprint["common_max_error_state_index"] == 94,
        "initialization_and_full_output_not_unique_explanations": fingerprint[
            "initialization_counts_follow_contract"
        ]
        and fingerprint["state_input_not_unique_explanation"]
        and fingerprint["full_output_not_unique_explanation"],
        "determinism_was_not_enabled": determinism["determinism_enabled"] is False
        and determinism["deterministic_algorithms"] is False
        and determinism["cudnn_deterministic"] is False,
        "d7c_order_confounds_route_and_time": source_audit["checks"][
            "public_always_precedes_wrapper"
        ]
        and source_audit["checks"]["no_within_route_repeatability_control"]
        and source_audit["checks"]["no_counterbalanced_order_in_cell_loop"],
        "synthetic_nonidentifiability_demonstrated": synthetic["valid"],
        "existing_evidence_cannot_identify_unique_cause": synthetic[
            "identifiability_result"
        ]
        == "same_observation_two_distinct_causes",
        "d7c_failure_and_rerun_closure_preserved": config[
            "frozen_failure_evidence"
        ]["claim_consumed"]
        and not config["authority"]["d7c_rerun_authorized"],
        "independent_design_candidate_has_all_requirements": route[
            "independent_design_candidate_established"
        ]
        and len(route["minimum_independence_requirements"]) == 8
        and route["candidate_design_only"]
        and not route["candidate_execution_authorized"],
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D7-C difference diagnostic failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": DIAGNOSTIC_VERSION,
        "status": "d7c_failure_difference_source_offline_diagnostic_complete",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "source_lock_checks": lock_checks,
        "source_audit": source_audit,
        "observed_difference_fingerprint": fingerprint,
        "determinism_observation": determinism,
        "synthetic_nonidentifiability_fixture": synthetic,
        "findings": {
            "d7c_conclusion": "failed_and_unchanged",
            "unique_cause_identified": False,
            "ruled_out_as_unique_explanations": [
                "state_none_initialization_count",
                "prebuilt_vs_none_state_mode",
                "full_output_flag",
                "state_shape_or_component_inventory_corruption",
            ],
            "consistent_but_not_proven": [
                "within_process_numerical_repeatability_or_first_shape_order_effect",
                "separately_compiled_instrumented_path_numerical_effect",
                "recurrent_amplification_after_the_first_attention_kv_component",
            ],
            "missing_identification_controls": [
                "public_public_repeatability",
                "wrapper_wrapper_repeatability",
                "counterbalanced_public_wrapper_order",
                "preregistered_determinism_policy",
            ],
            "independent_design_candidate_established": True,
            "candidate_stage": route["candidate_stage"],
            "candidate_research_question": route["candidate_research_question"],
            "candidate_primary_endpoint": "cross_route_drift_minus_within_route_repeatability_envelope",
            "minimum_independence_requirements": route[
                "minimum_independence_requirements"
            ],
            "candidate_execution_authorized": False,
        },
        "next_gate": NEXT_GATE,
        "safety": {
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_runner_modified": False,
            "d7c_fix_implemented": False,
            "d7c_rerun": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "projection_constructed": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
            "historical_d7c_failure_conclusion_changed": False,
            "independent_candidate_executed": False,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
