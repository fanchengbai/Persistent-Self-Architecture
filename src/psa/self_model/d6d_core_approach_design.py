from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-coupling-d6d-core-approach"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6d_core_approach_design.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6D 核心趋近非Core实验设计与无模型审查；"
    "同一实验必须包含wrapper-owned persistent synthetic正控制和真实冻结Self projection的"
    "identity/goal swap、mask、random、OFF因果对照，不修改真实RWKV实例字典，不拆分新的"
    "纯机制执行轮；本轮不导入RWKV/Torch、不访问权重、不加载或执行模型、不重跑"
    "D5C/P1/P2/D6C，也不授权D6D真实执行、D6E、正式测试集、Self效果结论、"
    "Self Updater或自动重跑。"
)
NEXT_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6D-I wrapper-owned真实路径与冻结Self projection"
    "构建工具的无模型实现；必须保持D6D单一联合实验、不得修改真实RWKV实例字典，只实现"
    "wrapper、projection训练冻结与artifact审计接口及纯Python验收；不探测installed source、"
    "不导入RWKV/Torch、不访问权重、不加载或执行模型，不授权D6D真实执行、D6E、正式测试集、"
    "Self效果结论、Self Updater、D5C/P1/P2/D6C重跑或自动重跑。"
)
CLASSIFICATION = (
    "d6d_joint_core_approach_design_no_model_verified_real_projection_not_constructed_"
    "execution_not_authorized"
)
CONDITIONS = (
    "wrapper_off",
    "wrapper_zero",
    "synthetic_positive",
    "self_matched",
    "self_identity_swap",
    "self_goal_swap",
    "self_identity_goal_swap",
    "self_identity_mask",
    "self_goal_mask",
    "self_identity_goal_mask",
    "self_identity_goal_norm_matched_random",
)
SELF_CONDITIONS = CONDITIONS[3:]
TASK_FAMILIES = (
    ("identity_bound_choice", "identity_anchors"),
    ("goal_bound_choice", "active_goals"),
    ("identity_goal_conflict", "both"),
)
CONDITION_CONTRACT = {
    "wrapper_off": "disabled_and_no_projection_constructed",
    "wrapper_zero": "enabled_zero_scale_and_no_projection_constructed",
    "synthetic_positive": (
        "wrapper_owned_unit_rms_probe_at_same_frozen_layer_not_self_evidence"
    ),
    "self_matched": "frozen_projection_of_fixture_matched_identity_and_goal",
    "self_identity_swap": "paired_state_identity_only_goal_matched",
    "self_goal_swap": "identity_matched_paired_state_goal_only",
    "self_identity_goal_swap": "paired_state_identity_and_goal",
    "self_identity_mask": "identity_branch_exact_zero_goal_matched",
    "self_goal_mask": "identity_matched_goal_branch_exact_zero",
    "self_identity_goal_mask": "both_branches_exact_zero_must_equal_wrapper_zero",
    "self_identity_goal_norm_matched_random": (
        "both_encoded_branches_seeded_randomized_independently_with_branch_l2_norms_"
        "preserved"
    ),
}
ENGINEERING_CHECKS = (
    "wrapper_never_mutates_real_model_instance_dictionary",
    "off_zero_and_double_mask_are_exact",
    "synthetic_positive_changes_output_at_frozen_layer",
    "all_outputs_finite_and_counts_exact",
    "projection_and_data_digests_match_preflight",
    "matched_identity_margin_exceeds_identity_swap_and_identity_mask_on_identity_family",
    "matched_goal_margin_exceeds_goal_swap_and_goal_mask_on_goal_family",
    "matched_joint_margin_exceeds_both_swap_both_mask_and_norm_random_on_joint_family",
    "field_specific_cross_controls_do_not_reverse_prespecified_specificity",
    "general_capability_sentinels_do_not_cross_frozen_degradation_bound",
)
DECISION_THRESHOLDS = {
    "exact_control_matches_required": 12,
    "synthetic_output_differences_required": 12,
    "minimum_directional_fixture_passes_per_four": 3,
    "maximum_general_capability_sentinel_code_changes": 1,
    "maximum_nonfinite_outputs": 0,
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d6d_core_approach_design.md",
    "docs/self_model_v0_1_coupling_d6c_real_persistent_mechanism_observation.md",
    "scripts/verify_self_model_v0_1_coupling_d6d_core_approach_design.py",
    "src/psa/self_model/d6d_core_approach_design.py",
    "src/psa/self_model/d6c_persistent_mechanism.py",
    "src/psa/self_model/state.py",
    "src/psa/self_model/encoding.py",
    "tests/test_self_model_d6d_core_approach_design.py",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D6D design must be a JSON object")
    return value


def balanced_condition_rows() -> tuple[tuple[str, ...], ...]:
    """Eleven cyclic rows plus one repeated row for the twelfth fixture."""
    rows = tuple(
        tuple(CONDITIONS[(column + row) % len(CONDITIONS)] for column in range(11))
        for row in range(11)
    )
    return (*rows, rows[0])


def build_no_model_call_plan() -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []
    for fixture_index, row in enumerate(balanced_condition_rows(), start=1):
        family, sensitive_field = TASK_FAMILIES[(fixture_index - 1) // 4]
        fixture_id = f"d6d-{family}-{((fixture_index - 1) % 4) + 1:02d}"
        calls.append(
            {
                "call_index": len(calls) + 1,
                "fixture_id": fixture_id,
                "task_family": family,
                "sensitive_field": sensitive_field,
                "phase": "off_precondition_unscored",
                "condition": "wrapper_off",
                "order_position": 0,
            }
        )
        for position, condition in enumerate(row, start=1):
            calls.append(
                {
                    "call_index": len(calls) + 1,
                    "fixture_id": fixture_id,
                    "task_family": family,
                    "sensitive_field": sensitive_field,
                    "phase": "scored",
                    "condition": condition,
                    "order_position": position,
                }
            )
    return tuple(calls)


def condition_projection_plan(condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError("unknown D6D condition")
    values: dict[str, dict[str, str]] = {
        "wrapper_off": {
            "projection": "none", "identity": "none", "goal": "none"
        },
        "wrapper_zero": {
            "projection": "none", "identity": "none", "goal": "none"
        },
        "synthetic_positive": {
            "projection": "synthetic_positive", "identity": "none", "goal": "none"
        },
        "self_matched": {
            "projection": "frozen_self", "identity": "matched", "goal": "matched"
        },
        "self_identity_swap": {
            "projection": "frozen_self", "identity": "paired_swap", "goal": "matched"
        },
        "self_goal_swap": {
            "projection": "frozen_self", "identity": "matched", "goal": "paired_swap"
        },
        "self_identity_goal_swap": {
            "projection": "frozen_self", "identity": "paired_swap", "goal": "paired_swap"
        },
        "self_identity_mask": {
            "projection": "frozen_self", "identity": "zero_mask", "goal": "matched"
        },
        "self_goal_mask": {
            "projection": "frozen_self", "identity": "matched", "goal": "zero_mask"
        },
        "self_identity_goal_mask": {
            "projection": "frozen_self", "identity": "zero_mask", "goal": "zero_mask"
        },
        "self_identity_goal_norm_matched_random": {
            "projection": "frozen_self",
            "identity": "seeded_norm_matched_random",
            "goal": "seeded_norm_matched_random",
        },
    }
    return {"condition": condition, **values[condition]}


def validate_design(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("frozen_prerequisites", {})
    experiment = config.get("single_experiment_contract", {})
    wrapper = config.get("wrapper_owned_persistent_contract", {})
    projection = config.get("projection_contract", {})
    blueprint = config.get("noncore_pilot_blueprint", {})
    schedule = config.get("schedule", {})
    decision = config.get("decision_contract", {})
    authority = config.get("authority", {})
    closed_authority = (
        "wrapper_runtime_implementation_authorized",
        "projection_training_or_construction_authorized",
        "installed_source_probe_authorized",
        "rwkv_import_authorized",
        "torch_import_authorized",
        "weights_access_authorized",
        "model_load_authorized",
        "model_execution_authorized",
        "d5c_rerun_authorized",
        "p1_rerun_authorized",
        "p2_authorized",
        "d6c_rerun_authorized",
        "d6d_real_execution_authorized",
        "d6e_authorized",
        "formal_test_set_authorized",
        "self_effect_conclusion_authorized",
        "self_updater_authorized",
        "automatic_rerun_authorized",
    )
    checks = {
        "identity_and_status_exact": config.get("design_version") == DESIGN_VERSION
        and config.get("stage")
        == "Coupling-D6D_noncore_joint_mechanism_and_frozen_self_projection_design"
        and config.get("status")
        == "design_and_no_model_review_only_execution_not_authorized"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "d6c_failure_and_consumed_claim_frozen": prerequisites.get("d6c_status")
        == "d6c_execution_attempt_failed_claim_consumed"
        and prerequisites.get("d6c_actual_model_forward_calls") == 0
        and prerequisites.get("d6c_rerun_allowed") is False
        and prerequisites.get("d6c_execution_claim_sha256")
        == "82b94c33513da0137127ce44a85513c48c381d92b87e3a7c27916931821fe6a3"
        and prerequisites.get("d6c_failure_digest_sha256")
        == "25fdd26b5b34bcb4b2adf81b8fc784d5c471c86fbedae24054b091015dadf273",
        "historical_fake_not_promoted": prerequisites.get(
            "fake_encoder_or_projection_acceptable_as_self_evidence"
        ) is False,
        "single_joint_experiment_exact": experiment.get("one_model_process") is True
        and experiment.get("one_wrapper_instance") is True
        and experiment.get("one_authorization_and_single_use_claim") is True
        and experiment.get("synthetic_positive_control_interleaved_with_self_conditions")
        is True
        and experiment.get("separate_mechanism_execution_round_allowed") is False
        and experiment.get("raw_original_route_allowed") is False,
        "wrapper_owns_all_persistent_bindings": wrapper.get("owner")
        == "project_wrapper_not_real_rwkv_instance"
        and wrapper.get("wrapper_owns")
        == ["forward", "forward_one", "forward_seq", "fixed_dispatcher", "request_context"]
        and wrapper.get("instrumented_methods_bound_to") == "wrapper_only",
        "real_instance_dictionary_is_read_only": wrapper.get(
            "real_model_instance_dictionary_mutation_allowed"
        ) is False
        and wrapper.get("real_model_setattr_or_delattr_allowed") is False
        and wrapper.get("real_model_dictionary_identity_snapshot_required_before_and_after_every_call")
        is True,
        "same_wrapper_path_for_all_routes": wrapper.get(
            "same_method_and_dispatcher_identity_for_all_routes"
        ) is True
        and wrapper.get("off_and_zero_use_same_instrumented_wrapper") is True
        and wrapper.get("request_transport") == "wrapper_owned_contextvars_ContextVar",
        "real_projection_contract_not_fake": projection.get("kind")
        == "field_separated_learned_frozen_self_projection"
        and projection.get("synthetic_or_hash_fake") is False
        and projection.get("source_contract") == "validated_Self_State_v0.1"
        and projection.get("source_fields") == ["identity_anchors", "active_goals"],
        "projection_freeze_and_no_leakage_exact": projection.get(
            "projection_training_source"
        ) == "noncore_development_only_distinct_from_pilot"
        and projection.get("projection_training_manifest_digest_required") is True
        and projection.get("projection_parameter_digest_required_before_pilot") is True
        and projection.get("encoder_parameter_digest_required_before_pilot") is True
        and projection.get("pilot_data_may_select_projection_or_threshold") is False
        and projection.get("online_update_allowed") is False,
        "projection_shape_site_and_prompt_exact": projection.get(
            "output_hidden_dimension"
        ) == 2560
        and projection.get("target_layer_index_zero_based") == 15
        and projection.get("target_phase") == "post_ffn_residual"
        and projection.get("projection_bias_allowed") is False
        and projection.get("double_mask_projection_exact_zero") is True
        and projection.get("natural_language_prompt_serialization_allowed") is False,
        "real_projection_not_constructed_this_round": projection.get(
            "projection_constructed_this_round"
        ) is False,
        "eleven_causal_conditions_exact": config.get("causal_conditions")
        == list(CONDITIONS),
        "condition_semantics_exact": config.get("condition_contract")
        == CONDITION_CONTRACT,
        "three_balanced_noncore_families_exact": blueprint.get("fixture_count") == 12
        and blueprint.get("task_families")
        == [
            {"name": name, "fixtures": 4, "prespecified_sensitive_field": field}
            for name, field in TASK_FAMILIES
        ]
        and blueprint.get("formal_test_set") is False
        and blueprint.get("prompt_contains_self_state") is False
        and blueprint.get("pilot_manifest_values_blinded_until_projection_digest_frozen")
        is True,
        "schedule_counts_exact": schedule.get("unscored_off_precondition_per_fixture")
        == 1
        and schedule.get("scored_conditions_per_fixture") == 11
        and schedule.get("calls_per_fixture") == 12
        and schedule.get("model_forward_calls_total") == 144
        and schedule.get("scored_forward_calls_total") == 132
        and schedule.get("synthetic_positive_calls_total") == 12
        and schedule.get("real_self_projection_calls_total") == 96
        and schedule.get("target_layer_applications_total") == 108,
        "balanced_schedule_and_no_posthoc_rerun": schedule.get("condition_order")
        == "balanced_cyclic_latin_11_routes_first_row_repeated_for_fixture_12"
        and schedule.get("posthoc_route_removal_allowed") is False
        and schedule.get("automatic_rerun_allowed") is False,
        "engineering_decision_not_effect_claim": decision.get("engineering_checks")
        == list(ENGINEERING_CHECKS)
        and decision.get("thresholds") == DECISION_THRESHOLDS
        and decision.get("thresholds_frozen_in_this_design_before_any_model_execution")
        is True
        and decision.get("classification_only")
        == "noncore_engineering_pilot_not_self_effect_conclusion"
        and decision.get("self_effect_conclusion_allowed") is False
        and decision.get("d6e_opened_automatically") is False,
        "design_only_authority_exact": authority.get("d6d_design_authorized") is True
        and authority.get("no_model_review_authorized") is True
        and all(authority.get(name) is False for name in closed_authority),
        "next_gate_exact": config.get("required_next_owner_confirmation_text")
        == NEXT_CONFIRMATION
        and config.get("next_gate")
        == "separate_d6d_wrapper_and_projection_tooling_implementation_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6D design failed closed: " + ", ".join(failed))
    return checks


def _restricted_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return sorted(found & {"rwkv", "torch"})


def build_d6d_no_model_review(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D6D review requires the locked project config path")
    config = _object(config_file)
    config_checks = validate_design(config)

    prerequisites = config["frozen_prerequisites"]
    digest_checks = {
        "d6c_observation_digest_frozen": sha256_file(
            root / "docs/self_model_v0_1_coupling_d6c_real_persistent_mechanism_observation.md"
        ) == prerequisites["d6c_observation_source_sha256"],
        "d6c_runtime_digest_frozen": sha256_file(
            root / "src/psa/self_model/d6c_persistent_mechanism.py"
        ) == prerequisites["d6c_runtime_source_sha256"],
        "d6b_ast_digest_frozen": sha256_file(
            root / "src/psa/self_model/d6b_persistent_ast.py"
        ) == prerequisites["d6b_ast_source_sha256"],
        "self_state_digest_frozen": sha256_file(
            root / "src/psa/self_model/state.py"
        ) == prerequisites["self_state_source_sha256"],
        "fake_encoder_digest_frozen": sha256_file(
            root / "src/psa/self_model/encoding.py"
        ) == prerequisites["offline_fake_encoder_source_sha256"],
    }
    if not all(digest_checks.values()):
        failed = [name for name, valid in digest_checks.items() if not valid]
        raise RuntimeError("D6D prerequisite digest changed: " + ", ".join(failed))

    calls = build_no_model_call_plan()
    scored = [call for call in calls if call["phase"] == "scored"]
    condition_counts = {
        condition: sum(call["condition"] == condition for call in scored)
        for condition in CONDITIONS
    }
    condition_plans = [condition_projection_plan(value) for value in CONDITIONS]
    plan_checks = {
        "twelve_fixtures_and_144_calls": len(calls) == 144
        and len({call["fixture_id"] for call in calls}) == 12,
        "one_unscored_off_per_fixture": sum(
            call["phase"] == "off_precondition_unscored" for call in calls
        ) == 12,
        "all_eleven_conditions_twelve_times": set(condition_counts.values()) == {12}
        and set(condition_counts) == set(CONDITIONS),
        "three_families_four_fixtures_each": all(
            len({call["fixture_id"] for call in calls if call["task_family"] == family})
            == 4
            for family, _ in TASK_FAMILIES
        ),
        "synthetic_and_self_share_one_plan": any(
            call["condition"] == "synthetic_positive" for call in scored
        )
        and all(any(call["condition"] == route for call in scored) for route in SELF_CONDITIONS),
        "projection_routes_are_unambiguous": sum(
            item["projection"] == "frozen_self" for item in condition_plans
        ) == 8
        and sum(item["projection"] == "synthetic_positive" for item in condition_plans)
        == 1
        and sum(item["projection"] == "none" for item in condition_plans) == 2,
    }
    reviewed_python = (
        root / "src/psa/self_model/d6d_core_approach_design.py",
        root / "scripts/verify_self_model_v0_1_coupling_d6d_core_approach_design.py",
        root / "tests/test_self_model_d6d_core_approach_design.py",
    )
    restricted = {
        str(path.relative_to(root)).replace("\\", "/"): _restricted_imports(path)
        for path in reviewed_python
    }
    import_checks = {
        "d6d_review_has_no_rwkv_or_torch_import": all(
            not values for values in restricted.values()
        )
    }
    source_digests = {
        relative: sha256_file(root / relative) for relative in SOURCE_PATHS
    }
    checks = {**config_checks, **digest_checks, **plan_checks, **import_checks}
    report: dict[str, Any] = {
        "report_version": DESIGN_VERSION,
        "status": "d6d_core_approach_design_no_model_verified",
        "valid": all(checks.values()),
        "classification": CLASSIFICATION,
        "checks": checks,
        "counts": {
            "checks": len(checks),
            "fixtures": 12,
            "planned_model_forward_calls": 144,
            "planned_scored_calls": 132,
            "conditions": len(CONDITIONS),
            "planned_calls_per_condition": condition_counts,
            "planned_real_self_projection_calls": 96,
            "planned_synthetic_positive_calls": 12,
        },
        "design": {
            "single_joint_experiment": True,
            "separate_mechanism_execution_round": False,
            "persistent_owner": "project_wrapper_not_real_rwkv_instance",
            "real_model_instance_dictionary_mutation": False,
            "conditions": list(CONDITIONS),
            "projection_kind": config["projection_contract"]["kind"],
            "projection_is_fake": False,
            "projection_constructed": False,
            "self_effect_conclusion": False,
        },
        "condition_projection_plans": condition_plans,
        "restricted_import_audit": restricted,
        "safety": {
            "installed_source_probed": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_model_instance_mutated": False,
            "wrapper_runtime_implemented": False,
            "real_self_projection_constructed": False,
            "projection_training_run": False,
            "d5c_p1_p2_d6c_rerun": False,
            "d6d_real_execution_authorized": False,
            "d6e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "automatic_rerun_authorized": False,
        },
        "next_gate": config["next_gate"],
        "required_next_owner_confirmation_text": NEXT_CONFIRMATION,
        "source_digests": dict(sorted(source_digests.items())),
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
