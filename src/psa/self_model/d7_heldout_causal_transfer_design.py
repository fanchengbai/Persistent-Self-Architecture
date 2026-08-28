from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-self-model-d7-heldout-causal-transfer-draft"
CONFIG_RELATIVE_PATH = (
    "configs/preregistration/self_model_v0_1_d7_heldout_causal_transfer.draft.json"
)
D6D_TRAINING_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d6d_projection_training_manifest.json"
)
D6D_PILOT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d6d_blinded_pilot_manifest.json"
)
D6D_ENTRY_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6d_ii_real_entry.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D7 独立held-out causal transfer研究路线与无模型预注册设计；"
    "只允许定义新的研究问题、训练/held-out评估分离、因果对照、通过标准、真实协议兼容前置门和"
    "全新artifact/authorization/claim/output命名空间，不复用D6D训练或pilot fixtures、seed、"
    "claim和执行结果，不修改真实runner、不实现projection或执行模型；不授权D6D重跑、"
    "D7真实执行、正式测试集、Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
CLASSIFICATION = (
    "d7_independent_heldout_causal_transfer_preregistration_design_preserved_"
    "d7b_manifests_separately_materialized_unrun"
)
NEXT_GATE = "owner_reviews_d7_design_then_separate_d7b_no_model_implementation_confirmation"
D7B_NEXT_GATE = "remote_no_model_d7b_verification_then_separate_d7c_design_confirmation"
D7_IDENTITIES = ("falcon", "otter", "maple", "silver", "violet")
D7_GOALS = ("survey", "repair", "catalog", "mediate", "forecast")
D7_TASK_FAMILIES = (
    "role_policy_transfer",
    "goal_plan_transfer",
    "joint_constraint_transfer",
    "counterfactual_field_transfer",
)
D7_CONDITIONS = (
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
    "self_identity_random",
    "self_goal_random",
    "self_identity_goal_random",
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    D6D_TRAINING_RELATIVE_PATH,
    D6D_PILOT_RELATIVE_PATH,
    D6D_ENTRY_RELATIVE_PATH,
    "docs/self_model_v0_1_d7_heldout_causal_transfer_design.md",
    "scripts/verify_self_model_v0_1_d7_heldout_causal_transfer_design.py",
    "src/psa/self_model/d7_heldout_causal_transfer_design.py",
    "tests/test_self_model_d7_heldout_causal_transfer_design.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D7 {label} must be an object")
    return value


def validate_design(config: Mapping[str, Any]) -> dict[str, bool]:
    independence = config.get("independence_contract", {})
    model = config.get("model_contract", {})
    training = config.get("training_design", {})
    heldout = config.get("heldout_design", {})
    gates = config.get("gate_sequence")
    compatibility = config.get("compatibility_gate", {})
    capability = config.get("capability_gate", {})
    endpoints = config.get("causal_endpoints", {})
    primary = endpoints.get("primary_conjunctive", {})
    safety_endpoint = endpoints.get("mechanism_and_safety", {})
    authority = config.get("authority", {})
    checks = {
        "identity_exact": config.get("design_version") == DESIGN_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D7_independent_heldout_causal_transfer_design"
        and config.get("status") == "preregistration_design_frozen_unimplemented_unrun"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "research_question_is_heldout_field_specific_transfer": config.get(
            "research_question"
        )
        == (
            "Can a field-separated Self projection learned from a new non-Core calibration "
            "set cause identity-specific and goal-specific behavioral transfer on previously "
            "unseen task families while preserving general capability?"
        ),
        "independence_contract_exact": independence.get(
            "new_research_question_not_wrapper_repair"
        )
        is True
        and independence.get("historical_d6d_use")
        == "failure_boundary_and_nonreuse_constraint_only"
        and all(
            independence.get(field) is False
            for field in (
                "d6d_training_fixtures_reused",
                "d6d_pilot_fixtures_reused",
                "d6d_seed_reused",
                "d6d_claim_reused",
                "d6d_output_or_quantitative_result_reused",
                "d6d_execution_rerun",
            )
        ),
        "model_contract_frozen_without_training": model.get("model_id")
        == "rwkv7-g1h-2.9b-20260710"
        and model.get("base_weights_trainable") is False
        and model.get("fields") == ["identity_anchors", "active_goals"]
        and model.get("natural_language_self_state_serialization") is False
        and model.get("target_layer_selection")
        == "future_compatibility_gate_must_freeze_before_any_projection_training",
        "new_five_by_five_training_grid": training.get("identity_keys")
        == list(D7_IDENTITIES)
        and training.get("goal_keys") == list(D7_GOALS)
        and training.get("capture_count") == 25
        and training.get("seed") == 27082801
        and training.get("capture_split")
        == "calibration_only_excluded_from_capability_and_heldout_transfer"
        and training.get("projection_implementation_authorized") is False,
        "heldout_structure_and_counts_frozen": heldout.get("task_families")
        == list(D7_TASK_FAMILIES)
        and heldout.get("conditions") == list(D7_CONDITIONS)
        and heldout.get("semantic_cases_per_family") == 4
        and heldout.get("answer_code_rotations") == 4
        and heldout.get("fixture_count") == 64
        and heldout.get("scored_calls_per_fixture") == 13
        and heldout.get("off_precondition_calls_per_fixture") == 1
        and heldout.get("heldout_forward_calls") == 896
        and heldout.get("projection_training_plus_heldout_forward_calls") == 921
        and heldout.get("prompt_payload_overlap_with_training_allowed") is False
        and heldout.get("formal_test_set") is False,
        "gate_sequence_is_separate_and_ordered": isinstance(gates, list)
        and [gate.get("gate_id") for gate in gates] == ["D7-B", "D7-C", "D7-D", "D7-E"]
        and all(gate.get("separate_authorization_required") is True for gate in gates)
        and gates[0].get("model_execution") is False
        and all(gate.get("model_execution") is True for gate in gates[1:])
        and gates[1].get("future_forward_calls") == 18
        and gates[2].get("future_forward_calls") == 64
        and gates[3].get("future_forward_calls") == 921
        and gates[3].get("self_effect_conclusion_allowed") is False,
        "compatibility_gate_covers_real_none_semantics": compatibility.get(
            "equivalence_cells"
        )
        == 8
        and compatibility.get("equivalence_forward_calls") == 16
        and compatibility.get("synthetic_active_forward_calls") == 2
        and compatibility.get("total_forward_calls") == 18
        and compatibility.get("execution_paths") == ["forward_one", "forward_seq"]
        and compatibility.get("state_inputs") == ["none", "prebuilt"]
        and compatibility.get("full_output_values") == [False, True]
        and compatibility.get("public_state_none_initialization_required") is True
        and compatibility.get("off_and_zero_logits_torch_equal_required") is True
        and compatibility.get("off_and_zero_state_torch_equal_required") is True
        and compatibility.get("heldout_payload_accessed") is False
        and compatibility.get("failure_action")
        == "stop_without_projection_training_or_effect_pilot",
        "capability_gate_is_prompt_visible_and_projection_free": capability.get(
            "prompt_visible_only"
        )
        is True
        and capability.get("projection_used") is False
        and capability.get("record_count") == 64
        and capability.get("minimum_label_marginalized_joint_accuracy") == 0.8
        and capability.get("minimum_each_task_family_accuracy") == 0.75
        and capability.get("minimum_identity_marginal_accuracy") == 0.85
        and capability.get("minimum_goal_marginal_accuracy") == 0.85
        and capability.get("required_prefix_roundtrip_rate") == 1.0
        and capability.get("maximum_single_predicted_code_share") == 0.5
        and capability.get("failure_action")
        == "stop_without_projection_training_or_effect_pilot",
        "primary_causal_thresholds_frozen": primary
        == {
            "minimum_matched_joint_accuracy": 0.75,
            "minimum_identity_swap_injected_identity_accuracy": 0.75,
            "minimum_identity_swap_preserved_goal_accuracy": 0.75,
            "minimum_goal_swap_injected_goal_accuracy": 0.75,
            "minimum_goal_swap_preserved_identity_accuracy": 0.75,
            "minimum_dual_swap_injected_joint_accuracy": 0.7,
            "matched_vs_dual_mask_mean_log_margin_positive": True,
            "matched_vs_dual_mask_cluster_bootstrap_95_lower_bound_positive": True,
            "matched_vs_dual_random_mean_log_margin_positive": True,
        },
        "mechanism_safety_thresholds_frozen": safety_endpoint
        == {
            "off_zero_exact_fixture_rate": 1.0,
            "synthetic_positive_target_layer_application_rate": 1.0,
            "minimum_general_capability_sentinel_retention": 0.95,
            "maximum_sentinel_drop_from_off": 0.05,
            "source_self_state_unchanged_required": True,
            "base_instance_dictionary_unchanged_required": True,
        }
        and endpoints.get(
            "mask_and_random_are_preregistered_contrasts_not_posthoc_success_rules"
        )
        is True
        and endpoints.get("all_primary_thresholds_fixed_before_any_d7_model_execution")
        is True
        and endpoints.get("passing_d7e_is_noncore_engineering_evidence_only") is True
        and endpoints.get("self_effect_conclusion_allowed") is False,
        "design_only_authority_exact": authority.get(
            "d7_research_question_design_authorized"
        )
        is True
        and authority.get("d7_preregistration_design_authorized") is True
        and authority.get("offline_design_verification_authorized") is True,
        "implementation_execution_and_later_authority_closed": all(
            authority.get(field) is False
            for field in (
                "d6d_rerun_authorized",
                "d7_manifest_implementation_authorized",
                "real_runner_modification_authorized",
                "projection_implementation_authorized",
                "d7_compatibility_execution_authorized",
                "d7_capability_execution_authorized",
                "d7_effect_execution_authorized",
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
        raise PermissionError("D7 design changed: " + ", ".join(failed))
    return checks


def analyze_independence(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    d6d_training = _object(root / D6D_TRAINING_RELATIVE_PATH, "D6D training manifest")
    d6d_pilot = _object(root / D6D_PILOT_RELATIVE_PATH, "D6D pilot manifest")
    d6d_entry = _object(root / D6D_ENTRY_RELATIVE_PATH, "D6D entry config")
    training = config["training_design"]
    heldout = config["heldout_design"]
    namespaces = config["namespaces"]
    d6d_source = d6d_training["source_contract"]
    d6d_families = {
        fixture["task_family"] for fixture in d6d_pilot["fixtures"]
    }
    d7_paths = list(namespaces.values()) + [
        training["manifest_future_path"], heldout["manifest_future_path"]
    ]
    d7b_manifest_paths = (
        training["manifest_future_path"],
        heldout["manifest_future_path"],
    )
    d7b_manifests = [
        _object(root / path, "D7-B materialized manifest")
        for path in d7b_manifest_paths
    ]
    checks = {
        "identity_keys_disjoint_from_d6d": set(training["identity_keys"]).isdisjoint(
            d6d_source["identity_keys"]
        ),
        "goal_keys_disjoint_from_d6d": set(training["goal_keys"]).isdisjoint(
            d6d_source["goal_keys"]
        ),
        "task_families_disjoint_from_d6d": set(heldout["task_families"]).isdisjoint(
            d6d_families
        ),
        "all_d7_seeds_distinct_from_d6d_optimizer_seed": len(
            {
                training["seed"],
                heldout["fixture_seed"],
                heldout["schedule_seed"],
                config["capability_gate"]["qualification_seed"],
            }
        )
        == 4
        and d6d_training["training_algorithm"]["optimizer_seed"]
        not in {
            training["seed"],
            heldout["fixture_seed"],
            heldout["schedule_seed"],
            config["capability_gate"]["qualification_seed"],
        },
        "all_d7_paths_unique": len(d7_paths) == len(set(d7_paths)),
        "d7_paths_do_not_reuse_d6d_authorization_or_output": d6d_entry[
            "authorization_path"
        ]
        not in d7_paths
        and d6d_entry["output_dir"] not in d7_paths,
        "future_execution_namespaces_still_absent": all(
            not (root / path).exists() for path in namespaces.values()
        ),
        "separately_authorized_d7b_manifests_only_materialization": all(
            (root / path).is_file() for path in d7b_manifest_paths
        )
        and all(
            manifest.get("design_config_sha256")
            == "94687cc07f06a72e784e21338b554cb1b57fadeb35f7a052eb02bb1b580bb647"
            for manifest in d7b_manifests
        ),
        "training_and_heldout_namespaces_distinct": training["namespace"]
        != heldout["namespace"]
        and config["capability_gate"]["qualification_fixture_namespace"]
        not in {training["namespace"], heldout["namespace"]},
        "no_d6d_quantitative_evidence_dependency": config["independence_contract"][
            "d6d_output_or_quantitative_result_reused"
        ]
        is False,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "d6d_identity_keys": d6d_source["identity_keys"],
        "d7_identity_keys": training["identity_keys"],
        "d6d_goal_keys": d6d_source["goal_keys"],
        "d7_goal_keys": training["goal_keys"],
        "d6d_task_families": sorted(d6d_families),
        "d7_task_families": heldout["task_families"],
        "future_path_count": len(d7_paths),
    }


def build_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    expected = (root / CONFIG_RELATIVE_PATH).resolve()
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected:
        raise PermissionError("D7 design config path is not frozen")
    config = _object(supplied, "design config")
    config_checks = validate_design(config)
    independence = analyze_independence(config, root)
    heldout = config["heldout_design"]
    count_derivation = {
        "training_capture_calls": len(D7_IDENTITIES) * len(D7_GOALS),
        "heldout_fixture_count": len(D7_TASK_FAMILIES)
        * heldout["semantic_cases_per_family"]
        * heldout["answer_code_rotations"],
        "heldout_calls_per_fixture": heldout["off_precondition_calls_per_fixture"]
        + len(D7_CONDITIONS),
        "heldout_forward_calls": heldout["fixture_count"]
        * (
            heldout["off_precondition_calls_per_fixture"] + len(D7_CONDITIONS)
        ),
    }
    count_derivation["single_joint_future_forward_calls"] = (
        count_derivation["training_capture_calls"]
        + count_derivation["heldout_forward_calls"]
    )
    checks = {
        "config_valid": all(config_checks.values()),
        "independence_valid": independence["valid"],
        "training_count_derived": count_derivation["training_capture_calls"] == 25,
        "heldout_fixture_count_derived": count_derivation["heldout_fixture_count"] == 64,
        "heldout_call_count_derived": count_derivation["heldout_forward_calls"] == 896,
        "single_joint_count_derived": count_derivation[
            "single_joint_future_forward_calls"
        ]
        == 921,
        "compatibility_gate_precedes_capability_and_effect": [
            gate["gate_id"] for gate in config["gate_sequence"]
        ]
        == ["D7-B", "D7-C", "D7-D", "D7-E"],
        "effect_thresholds_frozen_before_execution": config["causal_endpoints"][
            "all_primary_thresholds_fixed_before_any_d7_model_execution"
        ]
        is True,
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D7 design verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": DESIGN_VERSION,
        "status": "d7_heldout_causal_transfer_preregistration_design_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "independence": independence,
        "count_derivation": count_derivation,
        "research_question": config["research_question"],
        "gate_sequence": config["gate_sequence"],
        "causal_endpoints": config["causal_endpoints"],
        "historical_design_next_gate": NEXT_GATE,
        "next_gate": D7B_NEXT_GATE,
        "safety": {
            "d6d_rerun": False,
            "d7_manifests_implemented": True,
            "real_runner_modified": False,
            "projection_implemented": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
