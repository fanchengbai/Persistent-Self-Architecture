from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-coupling-d5-active-design"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d5_active_design.json"
)
D4_REPORT_DIGEST = "39d4611a6d50791f1677f9eb27e6fb2ea702151a26236fa4094699b821ca721a"
D4A_REPORT_DIGEST = "d6b0602a85553fddae184e2accb3ef06ed280a925ebc4d90a9e13032726b2e88"
D4B_REPORT_DIGEST = "8befb5f4b2ce90241b66aff1f43bce59645d367c14f6594169e9c454fcf36a20"
D4B_EXECUTION_COMMIT = "949bfa0e10a984ceb139f20e6861bb320d3fd54d"
REQUIRED_NEXT_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D5A 离线active contract与fake projection实现；"
    "不授权Coupling-D5B/D5C/D5D/D5E、RWKV/Torch导入、权重访问、模型加载或执行、"
    "真实层选择、真实Self projection构造、Self效果实验、Self Updater或自动重跑。"
)
GATE_IDS = ["Coupling-D5A", "Coupling-D5B", "Coupling-D5C", "Coupling-D5D", "Coupling-D5E"]
REQUIRED_CONTROLS = [
    "original_uninstrumented",
    "coupling_off",
    "zero_scale",
    "active_correct_self",
    "identity_field_swap",
    "goal_field_swap",
    "field_mask",
    "encoded_norm_matched_random",
    "scale_dose",
    "prompt_visible_reference",
    "general_capability_side_effect",
]
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d5_active_design.md",
    "scripts/verify_self_model_v0_1_coupling_d5_active_design.py",
    "src/psa/self_model/d5_active_injection_design.py",
    "tests/test_self_model_d5_active_injection_design.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Coupling-D5 design must be a JSON object")
    return value


def validate_design(design: Mapping[str, Any]) -> dict[str, bool]:
    names = design.get("nomenclature")
    prerequisites = design.get("prerequisites")
    contract = design.get("active_contract")
    ladder = design.get("gate_ladder")
    smoke = design.get("d5c_mechanism_smoke_requirements")
    selection = design.get("selection_boundaries")
    authority = design.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (names, prerequisites, contract, smoke, selection, authority)
    ) or not isinstance(ladder, list):
        raise ValueError("Coupling-D5 design structure is incomplete")
    checks = {
        "identity_exact": design.get("design_version") == DESIGN_VERSION
        and design.get("stage") == "Coupling-D5_offline_active_injection_review"
        and design.get("status")
        == "design_only_active_implementation_and_execution_not_authorized"
        and design.get("development_only") is True,
        "d5_names_disambiguated": names
        == {
            "workflow_gate_id": "Coupling-D5",
            "workflow_gate_meaning": "active_injection_design_after_off_equivalence",
            "architecture_decision_id": "Architecture-D5-Self-Updater",
            "architecture_decision_meaning": "deterministic_constrained_self_update_in_stage4",
            "same_gate": False,
            "self_updater_in_scope": False,
        },
        "prerequisites_frozen": prerequisites.get("d4_status") == "failed_preserved"
        and prerequisites.get("d4_report_digest_sha256") == D4_REPORT_DIGEST
        and prerequisites.get("d4a_classification") == "within_route_instability_observed"
        and prerequisites.get("d4a_report_digest_sha256") == D4A_REPORT_DIGEST
        and prerequisites.get("d4b_status") == "d4b_real_off_equivalence_passed"
        and prerequisites.get("d4b_report_digest_sha256") == D4B_REPORT_DIGEST
        and prerequisites.get("d4b_execution_commit") == D4B_EXECUTION_COMMIT
        and prerequisites.get("d4_or_d4b_rerun_required") is False
        and prerequisites.get("d4_status_can_change") is False,
        "project_owned_post_ffn_contract": contract.get("implementation_location")
        == "project_owned_code_only"
        and contract.get("site_packages_modification_allowed") is False
        and contract.get("upstream_source_lock_required") is True
        and contract.get("coupling_phase") == "post_ffn_residual"
        and contract.get("execution_paths") == ["forward_one", "forward_seq"],
        "sequence_and_state_semantics_frozen": contract.get("sequence_policy")
        == "broadcast_same_self_residual_to_each_sequence_position"
        and contract.get("recurrent_state_semantics_may_change") is False
        and contract.get("source_self_state_immutable") is True
        and contract.get("source_recurrent_state_immutable_between_conditions") is True,
        "projection_contract_bounded": contract.get("residual_operation")
        == "x_out_equals_x_in_plus_gate_times_scale_times_projection"
        and contract.get("projection_output_hidden_dimension") == 2560
        and contract.get("projection_must_match_residual_shape_dtype_device") is True
        and contract.get("nonfinite_projection_action")
        == "fail_before_residual_addition",
        "off_contract_preserved": contract.get("off_or_zero_scale_calls_callback")
        is False
        and contract.get("off_or_zero_scale_must_remain_exact") is True,
        "real_choices_not_selected": all(
            contract.get(field) is False
            for field in (
                "real_layer_indices_selected",
                "real_projection_parameters_selected",
                "real_scale_values_selected",
                "self_encoder_kind_selected",
            )
        ),
        "gate_ladder_exact": [item.get("gate_id") for item in ladder] == GATE_IDS
        and [item.get("name") for item in ladder]
        == [
            "offline_active_contract_and_fake_projection",
            "project_owned_active_path_static_integration",
            "real_2_9b_noncore_mechanism_smoke",
            "noncore_self_semantic_effect_pilot",
            "formal_self_effect_experiment",
        ],
        "model_boundaries_separated": [item.get("model_allowed") for item in ladder]
        == [False, False, True, True, True]
        and [item.get("effect_claim_allowed") for item in ladder]
        == [False, False, False, "development_only", "preregistered_only"],
        "future_real_gates_require_new_authority": all(
            item.get("single_use_machine_claim_required") is True
            and item.get("separate_owner_authorization_required") is True
            for item in ladder[2:4]
        )
        and ladder[4].get("new_preregistration_and_authorization_required") is True,
        "mechanism_smoke_not_effect_test": smoke.get("formal_test_set_accessed") is False
        and smoke.get("synthetic_probe_not_self_evidence") is True
        and smoke.get("original_off_zero_exact_controls_required") is True
        and smoke.get("active_callback_count_exact") is True
        and smoke.get("active_output_finite") is True
        and smoke.get("repeat_determinism_required") is True
        and smoke.get("source_inputs_unchanged_required") is True
        and smoke.get("scale_ordering_recorded_without_behavior_claim") is True
        and smoke.get("failure_action")
        == "stop_without_automatic_rerun_or_effect_pilot",
        "future_controls_complete": design.get("future_noncore_effect_controls")
        == REQUIRED_CONTROLS,
        "selection_uses_noncore_only": selection
        == {
            "layer_scale_encoder_selection_uses_noncore_development_only": True,
            "formal_test_set_for_selection": False,
            "base_model_frozen": True,
            "inference_prompt_self_serialization_forbidden": True,
            "projection_training_data_and_seed_must_freeze_before_effect_evaluation": True,
            "no_posthoc_threshold_or_control_removal": True,
        },
        "only_offline_review_authorized": authority.get("offline_design_review_authorized")
        is True
        and all(
            authority.get(field) is False
            for field in authority
            if field != "offline_design_review_authorized"
        ),
        "next_confirmation_text_frozen": design.get(
            "required_next_owner_confirmation_text"
        )
        == REQUIRED_NEXT_CONFIRMATION,
        "round_exit_exact": design.get("current_round_exit")
        == "offline_design_verified_wait_for_explicit_coupling_d5a_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("Coupling-D5 design failed closed: " + ", ".join(failed))
    return checks


def build_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("Coupling-D5 config path is not frozen")
    checks = validate_design(_load_object(config_file))
    report = {
        "report_version": DESIGN_VERSION,
        "status": "coupling_d5_active_design_static_verified",
        "valid": True,
        "checks": checks,
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "next_gate": "Coupling-D5A_requires_explicit_confirmation",
        "safety": {
            "d4_or_d4b_rerun": False,
            "d4_status_changed": False,
            "active_injection_implemented": False,
            "active_injection_executed": False,
            "self_projection_constructed": False,
            "real_layers_selected": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "formal_test_set_accessed": False,
            "self_effect_experiment_run": False,
            "self_updater_implemented": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
