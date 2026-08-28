from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d7_heldout_causal_transfer_design import (
    D7_CONDITIONS,
    D7_GOALS,
    D7_IDENTITIES,
    D7_TASK_FAMILIES,
)


CONTRACT_VERSION = "0.1-self-model-d7b-manifest-runtime-contract"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d7b_manifest_runtime_contract.json"
)
DESIGN_RELATIVE_PATH = (
    "configs/preregistration/self_model_v0_1_d7_heldout_causal_transfer.draft.json"
)
CALIBRATION_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d7_projection_training_manifest.json"
)
HELDOUT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d7_heldout_transfer_manifest.json"
)
DESIGN_SHA256 = "94687cc07f06a72e784e21338b554cb1b57fadeb35f7a052eb02bb1b580bb647"
HISTORICAL_DESIGN_REPORT_SHA256 = (
    "142b52f226406781620ef0ba45422fd65760841b823f2079d2c9af25802e2536"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D7-B held-out causal transfer manifest与fake-first runtime "
    "contract无模型实现；只允许把已冻结的25条calibration、64条held-out fixture、13条件"
    "调度、D7-C/D7-D/D7-E分阶段门、通过阈值和11个全新命名空间转成确定性manifest、纯"
    "Python fixture、完整性校验与离线报告；不得复用或修改D6D fixtures、seed、"
    "authorization、claim或结果，不探测installed source、不修改真实runner、不实现或构造"
    "projection、不导入RWKV/Torch、不访问权重、不加载或执行模型；不授权D7-C/D7-D/D7-E"
    "真实执行、D6D重跑、正式测试集、Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
CLASSIFICATION = (
    "d7b_deterministic_manifests_and_symbolic_fake_runtime_verified_"
    "no_model_no_projection"
)
NEXT_GATE = "remote_no_model_d7b_verification_then_separate_d7c_design_confirmation"
CALIBRATION_RECORDS = 25
HELDOUT_FIXTURES = 64
OFF_PRECONDITION_CALLS = 64
SCORED_CALLS = 832
HELDOUT_FORWARD_CALLS = 896
FUTURE_JOINT_FORWARD_CALLS = 921
CHOICE_CODES = ("A", "B", "C", "D")
OPTION_PAIR_KINDS = (
    "matched",
    "identity_swap",
    "goal_swap",
    "identity_goal_swap",
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    HELDOUT_RELATIVE_PATH,
    "docs/self_model_v0_1_d7_heldout_causal_transfer_design.md",
    "docs/self_model_v0_1_d7b_manifest_runtime_contract.md",
    "scripts/verify_self_model_v0_1_d7b_manifest_runtime_contract.py",
    "src/psa/self_model/d7_heldout_causal_transfer_design.py",
    "src/psa/self_model/d7b_manifest_runtime_contract.py",
    "tests/test_self_model_d7_heldout_causal_transfer_design.py",
    "tests/test_self_model_d7b_manifest_runtime_contract.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D7-B {label} must be an object")
    return value


def validate_contract_config(config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config.get("frozen_design", {})
    manifests = config.get("manifests", {})
    counts = config.get("counts", {})
    runtime = config.get("fake_runtime_contract", {})
    gates = config.get("gate_sequence")
    authority = config.get("authority", {})
    namespaces = config.get("namespace_groups")
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D7-B_no_model_manifest_and_fake_first_runtime_contract"
        and config.get("status") == "offline_contract_implemented_unrun_no_model"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "frozen_design_exact": frozen.get("path") == DESIGN_RELATIVE_PATH
        and frozen.get("sha256") == DESIGN_SHA256
        and frozen.get("historical_design_report_sha256")
        == HISTORICAL_DESIGN_REPORT_SHA256
        and frozen.get("research_question_changed") is False,
        "manifest_paths_and_commitments_exact": manifests.get("calibration_path")
        == CALIBRATION_RELATIVE_PATH
        and manifests.get("heldout_path") == HELDOUT_RELATIVE_PATH
        and manifests.get("deterministic_expansion_required") is True
        and manifests.get("expanded_commitment_required") is True
        and manifests.get("prompt_overlap_allowed") is False,
        "counts_exact": counts
        == {
            "calibration_records": CALIBRATION_RECORDS,
            "heldout_fixtures": HELDOUT_FIXTURES,
            "conditions": len(D7_CONDITIONS),
            "off_precondition_calls": OFF_PRECONDITION_CALLS,
            "scored_calls": SCORED_CALLS,
            "heldout_forward_calls": HELDOUT_FORWARD_CALLS,
            "future_joint_forward_calls": FUTURE_JOINT_FORWARD_CALLS,
        },
        "symbolic_runtime_only": runtime.get("fixture_kind")
        == "d7b_symbolic_python_only"
        and runtime.get("symbolic_route_resolution_only") is True
        and runtime.get("numeric_projection_allowed") is False
        and runtime.get("tensor_construction_allowed") is False
        and runtime.get("model_object_allowed") is False
        and runtime.get("input_mutation_allowed") is False
        and runtime.get("unknown_condition_policy")
        == "reject_before_ledger_append"
        and runtime.get("wrong_phase_policy") == "reject_before_ledger_append"
        and runtime.get("output_claim")
        == "route_contract_only_not_model_or_effect_evidence",
        "eleven_namespace_groups_exact": isinstance(namespaces, list)
        and len(namespaces) == len(set(namespaces)) == 11,
        "gates_ordered_and_later_closed": isinstance(gates, list)
        and [gate.get("gate_id") for gate in gates]
        == ["D7-B", "D7-C", "D7-D", "D7-E"]
        and gates[0]
        == {"gate_id": "D7-B", "implemented": True, "model_execution": False}
        and [gate.get("future_forward_calls") for gate in gates[1:]]
        == [18, 64, 921]
        and all(gate.get("implemented") is False for gate in gates[1:])
        and all(gate.get("authorized") is False for gate in gates[1:]),
        "d7b_authority_exact": authority.get(
            "d7b_manifest_implementation_authorized"
        )
        is True
        and authority.get("pure_python_fixture_authorized") is True
        and authority.get("offline_integrity_report_authorized") is True,
        "model_projection_and_later_authority_closed": all(
            authority.get(field) is False
            for field in (
                "installed_source_probe_authorized",
                "real_runner_modification_authorized",
                "projection_implementation_authorized",
                "projection_construction_authorized",
                "rwkv_import_authorized",
                "torch_import_authorized",
                "weights_access_authorized",
                "model_load_authorized",
                "model_execution_authorized",
                "d7c_authorized",
                "d7d_authorized",
                "d7e_authorized",
                "d6d_rerun_authorized",
                "formal_test_set_authorized",
                "self_effect_conclusion_authorized",
                "self_updater_authorized",
                "raw_original_route_authorized",
                "automatic_rerun_authorized",
            )
        ),
        "classification_and_next_gate_exact": config.get(
            "required_classification"
        )
        == CLASSIFICATION
        and config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D7-B contract config failed closed: " + ", ".join(failed))
    return checks


def validate_calibration_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    source = manifest.get("source_contract", {})
    generator = manifest.get("record_generator", {})
    prompt = manifest.get("prompt_blueprint", {})
    capture = manifest.get("capture_contract", {})
    projection = manifest.get("projection_contract", {})
    separation = manifest.get("separation", {})
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d7b-calibration-manifest"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D7-projection-calibration-v01"
        and manifest.get("status")
        == "frozen_no_model_contract_projection_unimplemented",
        "development_noncore_only": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False,
        "design_and_namespace_frozen": manifest.get("design_config_sha256")
        == DESIGN_SHA256
        and manifest.get("namespace")
        == "psa-self-model-v0.1-d7-calibration-20260828"
        and manifest.get("seed") == 27082801,
        "keys_exact": source.get("self_state_version") == "0.1"
        and source.get("fields") == ["identity_anchors", "active_goals"]
        and source.get("identity_keys") == list(D7_IDENTITIES)
        and source.get("goal_keys") == list(D7_GOALS)
        and set(source.get("identity_semantics", {})) == set(D7_IDENTITIES)
        and set(source.get("goal_semantics", {})) == set(D7_GOALS)
        and source.get("natural_language_self_state_serialization") is False,
        "five_by_five_generator_exact": generator.get("order")
        == "identity_major_then_goal_major"
        and generator.get("identity_count") == 5
        and generator.get("goal_count") == 5
        and generator.get("record_count") == CALIBRATION_RECORDS
        and generator.get("record_id_template") == "d7-cal-{identity}-{goal}"
        and generator.get("deterministic_seed_is_commitment_only") is True,
        "prompt_separation_exact": isinstance(prompt.get("history_template"), str)
        and "{identity_semantic}" in prompt["history_template"]
        and "{goal_semantic}" in prompt["history_template"]
        and isinstance(prompt.get("capture_suffix"), str)
        and prompt.get("heldout_template_reuse_allowed") is False
        and prompt.get("capability_template_reuse_allowed") is False
        and prompt.get("self_state_object_rendered") is False,
        "capture_is_future_read_only_and_layer_unselected": capture.get("future_only")
        is True
        and capture.get("read_only") is True
        and capture.get("record_count") == CALIBRATION_RECORDS
        and capture.get("model_forward_calls") == CALIBRATION_RECORDS
        and capture.get("target_layer_selected") is False
        and capture.get("target_layer_selection_deferred_to_d7c") is True
        and capture.get("base_instance_dictionary_mutation_allowed") is False
        and capture.get("source_self_state_mutation_allowed") is False,
        "projection_unimplemented_unconstructed": projection.get("family")
        == "field_separated_additive_no_bias_frozen_after_calibration"
        and projection.get("implemented") is False
        and projection.get("constructed") is False
        and projection.get("training_authorized") is False,
        "split_and_d6d_separation_exact": separation.get(
            "heldout_payload_accessed_during_calibration"
        )
        is False
        and separation.get("capability_payload_accessed_during_calibration") is False
        and separation.get("d6d_fixture_or_result_dependency") is False
        and separation.get("failure_policy")
        == "stop_without_projection_or_heldout_execution",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D7-B calibration manifest failed closed: " + ", ".join(failed))
    return checks


def validate_heldout_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    qualification = manifest.get("capability_qualification", {})
    keys = manifest.get("keys", {})
    families = manifest.get("task_families")
    generator = manifest.get("fixture_generator", {})
    prompt = manifest.get("prompt_contract", {})
    schedule = manifest.get("schedule", {})
    gates = manifest.get("gates", {})
    thresholds = manifest.get("thresholds", {})
    separation = manifest.get("separation", {})
    family_ids = [item.get("family_id") for item in families] if isinstance(families, list) else []
    case_ids = [
        case.get("case_id")
        for family in families or []
        for case in family.get("semantic_cases", [])
        if isinstance(case, dict)
    ]
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d7b-heldout-transfer-manifest"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D7-heldout-causal-transfer-v01"
        and manifest.get("status") == "frozen_no_model_contract_unrun",
        "development_noncore_only": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False,
        "design_namespace_and_seeds_frozen": manifest.get("design_config_sha256")
        == DESIGN_SHA256
        and manifest.get("namespace")
        == "psa-self-model-v0.1-d7-heldout-transfer-20260828"
        and manifest.get("fixture_seed") == 27082802
        and manifest.get("schedule_seed") == 27082803,
        "qualification_separate_visible_and_projection_free": qualification
        == {
            "namespace": "psa-self-model-v0.1-d7-capability-qualification-20260828",
            "seed": 27082804,
            "prompt_visible": True,
            "projection_used": False,
            "record_count": HELDOUT_FIXTURES,
        },
        "keys_and_semantics_exact": keys.get("identity") == list(D7_IDENTITIES)
        and keys.get("goal") == list(D7_GOALS)
        and set(keys.get("identity_semantics", {})) == set(D7_IDENTITIES)
        and set(keys.get("goal_semantics", {})) == set(D7_GOALS),
        "four_families_and_sixteen_cases": family_ids == list(D7_TASK_FAMILIES)
        and len(case_ids) == len(set(case_ids)) == 16
        and all(len(family.get("semantic_cases", [])) == 4 for family in families or []),
        "fixture_generator_exact": generator.get("semantic_cases_per_family") == 4
        and generator.get("answer_code_rotations") == [0, 1, 2, 3]
        and generator.get("choice_codes") == list(CHOICE_CODES)
        and generator.get("option_pair_kinds") == list(OPTION_PAIR_KINDS)
        and generator.get("fixture_count") == HELDOUT_FIXTURES
        and generator.get("key_assignment") == "seeded_modular_v0.1"
        and generator.get("code_rotation")
        == "matched_target_moves_A_B_C_D_once_per_semantic_case"
        and generator.get("random_control_assignment")
        == "seeded_nonmatched_modular_v0.1",
        "prompt_contract_exact": isinstance(prompt.get("hidden_prompt_prefix"), str)
        and isinstance(prompt.get("qualification_prompt_prefix"), str)
        and "{identity_semantic}" in prompt["qualification_prompt_prefix"]
        and "{goal_semantic}" in prompt["qualification_prompt_prefix"]
        and prompt.get("hidden_prompt_contains_identity_or_goal_key") is False
        and prompt.get("qualification_prompt_visible_only") is True
        and prompt.get("calibration_prompt_overlap_allowed") is False,
        "conditions_exact": manifest.get("conditions") == list(D7_CONDITIONS),
        "schedule_counts_exact": schedule.get("order_algorithm")
        == "seeded_cyclic_rotation_v0.1"
        and schedule.get("off_precondition_calls_per_fixture") == 1
        and schedule.get("scored_calls_per_fixture") == len(D7_CONDITIONS)
        and schedule.get("calls_per_fixture") == 14
        and schedule.get("off_precondition_calls_total") == OFF_PRECONDITION_CALLS
        and schedule.get("scored_calls_total") == SCORED_CALLS
        and schedule.get("heldout_forward_calls_total") == HELDOUT_FORWARD_CALLS
        and schedule.get("projection_training_plus_heldout_forward_calls")
        == FUTURE_JOINT_FORWARD_CALLS
        and schedule.get("each_condition_calls_total") == HELDOUT_FIXTURES,
        "later_gates_closed_and_counts_frozen": gates.get("D7-C")
        == {"future_calls": 18, "heldout_payload_accessed": False, "authorized": False}
        and gates.get("D7-D")
        == {"future_calls": 64, "projection_used": False, "authorized": False}
        and gates.get("D7-E")
        == {"future_calls": 921, "single_joint_process": True, "authorized": False},
        "thresholds_match_preregistration": _thresholds_exact(thresholds),
        "separation_and_failure_policy_exact": separation
        == {
            "projection_training_eligible": False,
            "calibration_payload_accessed": False,
            "d6d_fixture_seed_claim_authorization_or_result_dependency": False,
            "failure_policy": "stop_without_rerun_or_later_gate_upgrade",
        },
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D7-B heldout manifest failed closed: " + ", ".join(failed))
    return checks


def _thresholds_exact(thresholds: Mapping[str, Any]) -> bool:
    return thresholds == {
        "capability": {
            "minimum_label_marginalized_joint_accuracy": 0.8,
            "minimum_each_task_family_accuracy": 0.75,
            "minimum_identity_marginal_accuracy": 0.85,
            "minimum_goal_marginal_accuracy": 0.85,
            "required_prefix_roundtrip_rate": 1.0,
            "maximum_single_predicted_code_share": 0.5,
        },
        "primary_causal": {
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
        "mechanism_safety": {
            "off_zero_exact_fixture_rate": 1.0,
            "synthetic_positive_target_layer_application_rate": 1.0,
            "minimum_general_capability_sentinel_retention": 0.95,
            "maximum_sentinel_drop_from_off": 0.05,
            "source_self_state_unchanged_required": True,
            "base_instance_dictionary_unchanged_required": True,
        },
        "passing_is_noncore_engineering_evidence_only": True,
        "self_effect_conclusion_allowed": False,
    }


def expand_calibration_records(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    validate_calibration_manifest(manifest)
    source = manifest["source_contract"]
    prompt = manifest["prompt_blueprint"]
    records: list[dict[str, Any]] = []
    for identity in D7_IDENTITIES:
        for goal in D7_GOALS:
            text = prompt["history_template"].format(
                identity_semantic=source["identity_semantics"][identity],
                goal_semantic=source["goal_semantics"][goal],
            ) + prompt["capture_suffix"]
            records.append(
                {
                    "record_index": len(records),
                    "record_id": f"d7-cal-{identity}-{goal}",
                    "identity_key": identity,
                    "goal_key": goal,
                    "prompt_text": text,
                    "prompt_sha256": sha256_json(text),
                    "calibration_only": True,
                    "future_capture": True,
                }
            )
    if len(records) != CALIBRATION_RECORDS:
        raise AssertionError("D7-B calibration expansion count changed")
    return tuple(records)


def _case_rows(manifest: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    rows = []
    for family in manifest["task_families"]:
        for case in family["semantic_cases"]:
            rows.append((family["family_id"], case["case_id"], case["context"]))
    return tuple(rows)


def _pair_values(
    *, matched_identity: str, paired_identity: str, matched_goal: str, paired_goal: str
) -> dict[str, tuple[str, str]]:
    return {
        "matched": (matched_identity, matched_goal),
        "identity_swap": (paired_identity, matched_goal),
        "goal_swap": (matched_identity, paired_goal),
        "identity_goal_swap": (paired_identity, paired_goal),
    }


def expand_heldout_fixtures(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    validate_heldout_manifest(manifest)
    seed = int(manifest["fixture_seed"])
    identity_semantics = manifest["keys"]["identity_semantics"]
    goal_semantics = manifest["keys"]["goal_semantics"]
    prompt = manifest["prompt_contract"]
    fixtures: list[dict[str, Any]] = []
    for semantic_index, (family, case_id, context) in enumerate(_case_rows(manifest)):
        identity_index = (seed + semantic_index * 2) % len(D7_IDENTITIES)
        goal_index = ((seed // 5) + semantic_index * 3) % len(D7_GOALS)
        identity_offset = 1 + semantic_index % (len(D7_IDENTITIES) - 1)
        goal_offset = 1 + (semantic_index + 1) % (len(D7_GOALS) - 1)
        random_identity_offset = 1 + (semantic_index + 2) % (len(D7_IDENTITIES) - 1)
        random_goal_offset = 1 + (semantic_index + 3) % (len(D7_GOALS) - 1)
        matched_identity = D7_IDENTITIES[identity_index]
        matched_goal = D7_GOALS[goal_index]
        paired_identity = D7_IDENTITIES[(identity_index + identity_offset) % 5]
        paired_goal = D7_GOALS[(goal_index + goal_offset) % 5]
        random_identity = D7_IDENTITIES[(identity_index + random_identity_offset) % 5]
        random_goal = D7_GOALS[(goal_index + random_goal_offset) % 5]
        pairs = _pair_values(
            matched_identity=matched_identity,
            paired_identity=paired_identity,
            matched_goal=matched_goal,
            paired_goal=paired_goal,
        )
        for rotation in range(4):
            option_pairs: dict[str, dict[str, str]] = {}
            option_lines = []
            for pair_index, pair_kind in enumerate(OPTION_PAIR_KINDS):
                code = CHOICE_CODES[(pair_index + rotation) % len(CHOICE_CODES)]
                identity, goal = pairs[pair_kind]
                option_pairs[code] = {
                    "pair_kind": pair_kind,
                    "identity_key": identity,
                    "goal_key": goal,
                }
            for code in CHOICE_CODES:
                values = option_pairs[code]
                option_lines.append(
                    f"{code}. {identity_semantics[values['identity_key']]}; "
                    f"{goal_semantics[values['goal_key']]}"
                )
            options = "\n".join(option_lines)
            hidden = (
                prompt["hidden_prompt_prefix"]
                + context
                + "\n"
                + options
                + prompt["answer_suffix"]
            )
            qualification = (
                prompt["qualification_prompt_prefix"].format(
                    identity_semantic=identity_semantics[matched_identity],
                    goal_semantic=goal_semantics[matched_goal],
                )
                + context
                + "\n"
                + options
                + prompt["answer_suffix"]
            )
            fixtures.append(
                {
                    "fixture_index": len(fixtures),
                    "fixture_id": f"d7-heldout-{family}-{case_id}-r{rotation}",
                    "task_family": family,
                    "semantic_case_id": case_id,
                    "answer_code_rotation": rotation,
                    "matched_identity": matched_identity,
                    "paired_identity": paired_identity,
                    "random_identity": random_identity,
                    "matched_goal": matched_goal,
                    "paired_goal": paired_goal,
                    "random_goal": random_goal,
                    "target_code": CHOICE_CODES[rotation],
                    "option_pairs": option_pairs,
                    "hidden_prompt_text": hidden,
                    "hidden_prompt_sha256": sha256_json(hidden),
                    "qualification_prompt_text": qualification,
                    "qualification_prompt_sha256": sha256_json(qualification),
                    "projection_training_eligible": False,
                }
            )
    if len(fixtures) != HELDOUT_FIXTURES:
        raise AssertionError("D7-B heldout fixture expansion count changed")
    return tuple(fixtures)


def build_heldout_call_plan(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    fixtures = expand_heldout_fixtures(manifest)
    schedule_seed = int(manifest["schedule_seed"])
    calls: list[dict[str, Any]] = []
    for fixture in fixtures:
        calls.append(
            {
                "call_index": len(calls) + 1,
                "fixture_index": fixture["fixture_index"],
                "fixture_id": fixture["fixture_id"],
                "phase": "off_precondition_unscored",
                "condition": "wrapper_off",
                "scored": False,
            }
        )
        offset = (schedule_seed + fixture["fixture_index"]) % len(D7_CONDITIONS)
        order = D7_CONDITIONS[offset:] + D7_CONDITIONS[:offset]
        for condition in order:
            calls.append(
                {
                    "call_index": len(calls) + 1,
                    "fixture_index": fixture["fixture_index"],
                    "fixture_id": fixture["fixture_id"],
                    "phase": "heldout_scored",
                    "condition": condition,
                    "scored": True,
                }
            )
    if len(calls) != HELDOUT_FORWARD_CALLS:
        raise AssertionError("D7-B heldout call plan count changed")
    return tuple(calls)


def resolve_symbolic_route(
    fixture: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    if condition not in D7_CONDITIONS:
        raise PermissionError("D7-B unknown condition rejected")
    matched_identity = fixture["matched_identity"]
    paired_identity = fixture["paired_identity"]
    random_identity = fixture["random_identity"]
    matched_goal = fixture["matched_goal"]
    paired_goal = fixture["paired_goal"]
    random_goal = fixture["random_goal"]
    routes: dict[str, tuple[str | None, str | None, str]] = {
        "wrapper_off": (None, None, "off_control"),
        "wrapper_zero": (None, None, "zero_control"),
        "synthetic_positive": (matched_identity, matched_goal, "synthetic_control"),
        "self_matched": (matched_identity, matched_goal, "future_real_self"),
        "self_identity_swap": (paired_identity, matched_goal, "future_real_self"),
        "self_goal_swap": (matched_identity, paired_goal, "future_real_self"),
        "self_identity_goal_swap": (paired_identity, paired_goal, "future_real_self"),
        "self_identity_mask": (None, matched_goal, "future_real_self"),
        "self_goal_mask": (matched_identity, None, "future_real_self"),
        "self_identity_goal_mask": (None, None, "future_real_self"),
        "self_identity_random": (random_identity, matched_goal, "future_real_self"),
        "self_goal_random": (matched_identity, random_goal, "future_real_self"),
        "self_identity_goal_random": (random_identity, random_goal, "future_real_self"),
    }
    identity, goal, route_kind = routes[condition]
    return {
        "condition": condition,
        "identity_key": identity,
        "goal_key": goal,
        "route_kind": route_kind,
        "numeric_projection_constructed": False,
        "model_output_created": False,
    }


@dataclass
class D7BSymbolicFakeRuntime:
    fixture_kind: str = "d7b_symbolic_python_only"

    def __post_init__(self) -> None:
        self.ledger: list[dict[str, Any]] = []

    def execute(
        self, *, fixture: Mapping[str, Any], call: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.fixture_kind != "d7b_symbolic_python_only":
            raise PermissionError("D7-B runtime only accepts the symbolic Python fixture")
        before_fixture = copy.deepcopy(fixture)
        before_call = copy.deepcopy(call)
        if call.get("fixture_id") != fixture.get("fixture_id"):
            raise PermissionError("D7-B call and fixture identity mismatch")
        phase = call.get("phase")
        condition = call.get("condition")
        if phase == "off_precondition_unscored":
            if condition != "wrapper_off" or call.get("scored") is not False:
                raise PermissionError("D7-B OFF precondition contract changed")
        elif phase == "heldout_scored":
            if condition not in D7_CONDITIONS or call.get("scored") is not True:
                raise PermissionError("D7-B scored call contract changed")
        else:
            raise PermissionError("D7-B unknown phase rejected")
        route = resolve_symbolic_route(fixture, str(condition))
        entry = {
            "call_index": call["call_index"],
            "fixture_id": fixture["fixture_id"],
            "phase": phase,
            "condition": condition,
            "route": route,
        }
        if fixture != before_fixture or call != before_call:
            raise RuntimeError("D7-B symbolic runtime mutated its input")
        self.ledger.append(entry)
        return copy.deepcopy(entry)


def run_fake_runtime_acceptance(
    calibration: Mapping[str, Any], heldout: Mapping[str, Any]
) -> dict[str, Any]:
    calibration_records = expand_calibration_records(calibration)
    fixtures = expand_heldout_fixtures(heldout)
    calls = build_heldout_call_plan(heldout)
    calibration_before = copy.deepcopy(calibration_records)
    fixtures_before = copy.deepcopy(fixtures)
    calls_before = copy.deepcopy(calls)
    runtime = D7BSymbolicFakeRuntime()
    fixture_by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}
    outputs = [
        runtime.execute(fixture=fixture_by_id[call["fixture_id"]], call=call)
        for call in calls
    ]
    condition_counts = {
        condition: sum(
            output["phase"] == "heldout_scored"
            and output["condition"] == condition
            for output in outputs
        )
        for condition in D7_CONDITIONS
    }
    target_codes = [fixture["target_code"] for fixture in fixtures]
    hidden_hashes = {fixture["hidden_prompt_sha256"] for fixture in fixtures}
    qualification_hashes = {
        fixture["qualification_prompt_sha256"] for fixture in fixtures
    }
    calibration_hashes = {record["prompt_sha256"] for record in calibration_records}
    hidden_key_free = all(
        all(key not in fixture["hidden_prompt_text"] for key in (*D7_IDENTITIES, *D7_GOALS))
        for fixture in fixtures
    )
    checks = {
        "twenty_five_calibration_records": len(calibration_records)
        == CALIBRATION_RECORDS,
        "sixty_four_unique_heldout_fixtures": len(fixtures)
        == len({fixture["fixture_id"] for fixture in fixtures})
        == HELDOUT_FIXTURES,
        "four_families_balanced": all(
            sum(fixture["task_family"] == family for fixture in fixtures) == 16
            for family in D7_TASK_FAMILIES
        ),
        "answer_codes_balanced": all(target_codes.count(code) == 16 for code in CHOICE_CODES),
        "calibration_hidden_and_qualification_prompts_disjoint": not (
            calibration_hashes & hidden_hashes
            or calibration_hashes & qualification_hashes
            or hidden_hashes & qualification_hashes
        ),
        "hidden_prompts_do_not_serialize_keys": hidden_key_free,
        "qualification_prompts_are_distinct_and_visible": len(qualification_hashes)
        == HELDOUT_FIXTURES
        and all(
            heldout["keys"]["identity_semantics"][fixture["matched_identity"]]
            in fixture["qualification_prompt_text"]
            and heldout["keys"]["goal_semantics"][fixture["matched_goal"]]
            in fixture["qualification_prompt_text"]
            for fixture in fixtures
        ),
        "eight_hundred_ninety_six_calls": len(calls) == len(outputs)
        == HELDOUT_FORWARD_CALLS,
        "one_off_precondition_per_fixture": sum(
            output["phase"] == "off_precondition_unscored" for output in outputs
        )
        == OFF_PRECONDITION_CALLS,
        "each_scored_condition_exactly_sixty_four": all(
            count == HELDOUT_FIXTURES for count in condition_counts.values()
        ),
        "symbolic_routes_never_construct_projection_or_output": all(
            output["route"]["numeric_projection_constructed"] is False
            and output["route"]["model_output_created"] is False
            for output in outputs
        ),
        "off_zero_and_masks_resolve_exact_fields": all(
            resolve_symbolic_route(fixture, "wrapper_off")["identity_key"] is None
            and resolve_symbolic_route(fixture, "wrapper_zero")["goal_key"] is None
            and resolve_symbolic_route(fixture, "self_identity_mask")["identity_key"]
            is None
            and resolve_symbolic_route(fixture, "self_goal_mask")["goal_key"] is None
            for fixture in fixtures
        ),
        "swap_routes_preserve_and_replace_correct_fields": all(
            resolve_symbolic_route(fixture, "self_identity_swap")["identity_key"]
            == fixture["paired_identity"]
            and resolve_symbolic_route(fixture, "self_identity_swap")["goal_key"]
            == fixture["matched_goal"]
            and resolve_symbolic_route(fixture, "self_goal_swap")["identity_key"]
            == fixture["matched_identity"]
            and resolve_symbolic_route(fixture, "self_goal_swap")["goal_key"]
            == fixture["paired_goal"]
            for fixture in fixtures
        ),
        "inputs_unchanged": calibration_records == calibration_before
        and fixtures == fixtures_before
        and calls == calls_before,
        "ledger_is_complete_route_only": len(runtime.ledger) == HELDOUT_FORWARD_CALLS
        and runtime.ledger == outputs,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "calibration_records": len(calibration_records),
            "heldout_fixtures": len(fixtures),
            "off_precondition_calls": OFF_PRECONDITION_CALLS,
            "scored_calls": SCORED_CALLS,
            "heldout_forward_calls": len(outputs),
            "future_joint_forward_calls": len(calibration_records) + len(outputs),
            "symbolic_ledger_entries": len(runtime.ledger),
        },
        "condition_counts": condition_counts,
        "commitments": {
            "expanded_calibration_sha256": sha256_json(calibration_records),
            "expanded_heldout_fixture_sha256": sha256_json(fixtures),
            "expanded_call_plan_sha256": sha256_json(calls),
            "symbolic_route_ledger_sha256": sha256_json(runtime.ledger),
        },
    }


def _future_execution_paths_absent(design: Mapping[str, Any], root: Path) -> bool:
    namespaces = design["namespaces"]
    paths = list(namespaces.values())
    return all(not (root / path).exists() for path in paths)


def build_d7b_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    expected = (root / CONFIG_RELATIVE_PATH).resolve()
    if supplied != expected:
        raise PermissionError("D7-B contract config path is not frozen")
    config = _object(supplied, "contract config")
    design_path = root / DESIGN_RELATIVE_PATH
    calibration_path = root / CALIBRATION_RELATIVE_PATH
    heldout_path = root / HELDOUT_RELATIVE_PATH
    if sha256_file(design_path) != DESIGN_SHA256:
        raise PermissionError("D7-B frozen design config digest changed")
    design = _object(design_path, "frozen design")
    calibration = _object(calibration_path, "calibration manifest")
    heldout = _object(heldout_path, "heldout manifest")
    config_checks = validate_contract_config(config)
    calibration_checks = validate_calibration_manifest(calibration)
    heldout_checks = validate_heldout_manifest(heldout)
    acceptance = run_fake_runtime_acceptance(calibration, heldout)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "contract_config_valid": all(config_checks.values()),
        "calibration_manifest_valid": all(calibration_checks.values()),
        "heldout_manifest_valid": all(heldout_checks.values()),
        "fake_runtime_acceptance_valid": acceptance["valid"],
        "future_joint_count_exact": acceptance["counts"]["future_joint_forward_calls"]
        == FUTURE_JOINT_FORWARD_CALLS,
        "frozen_design_digest_preserved": source_digests[DESIGN_RELATIVE_PATH]
        == DESIGN_SHA256,
        "manifest_digests_distinct": source_digests[CALIBRATION_RELATIVE_PATH]
        != source_digests[HELDOUT_RELATIVE_PATH],
        "future_execution_authorization_claim_and_outputs_absent":
        _future_execution_paths_absent(design, root),
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D7-B verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "status": "d7b_manifests_and_symbolic_fake_runtime_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "calibration_checks": calibration_checks,
        "heldout_checks": heldout_checks,
        "acceptance": acceptance,
        "decision": {
            "d7b_implemented": True,
            "d7c_d7d_d7e_authorized": False,
            "projection_implemented_or_constructed": False,
            "d6d_rerun": False,
            "self_effect_conclusion": False,
        },
        "source_digests": source_digests,
        "next_gate": NEXT_GATE,
        "safety": {
            "installed_source_probed": False,
            "real_runner_modified": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "projection_implemented": False,
            "projection_constructed": False,
            "d7c_executed": False,
            "d7d_executed": False,
            "d7e_executed": False,
            "d6d_rerun": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
