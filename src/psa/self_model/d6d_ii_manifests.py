from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d6d_core_approach_design import (
    CONDITIONS,
    TASK_FAMILIES,
    balanced_condition_rows,
)


MANIFEST_REPORT_VERSION = "0.1-coupling-d6d-ii-manifests"
TRAINING_MANIFEST_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d6d_projection_training_manifest.json"
)
PILOT_MANIFEST_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d6d_blinded_pilot_manifest.json"
)
IDENTITY_KEYS = ("saffron", "indigo", "amber", "cobalt")
GOAL_KEYS = ("spiral", "harbor", "orbit", "beacon")
CHOICE_TOKEN_IDS = {"A": 66, "B": 67, "C": 68, "D": 69}
FORCED_PREFIX_TOKEN_IDS = (63, 11)
TOKENIZER_DIGEST = "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
TRAINING_FORWARD_CALLS = 16
PILOT_FORWARD_CALLS = 144
TOTAL_FORWARD_CALLS = TRAINING_FORWARD_CALLS + PILOT_FORWARD_CALLS


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D6D-II {label} must be an object")
    return value


def load_training_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _object(path, "training manifest")
    validate_training_manifest(manifest)
    return manifest


def validate_training_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    source = manifest.get("source_contract", {})
    capture = manifest.get("capture_contract", {})
    prompt = manifest.get("prompt_blueprint", {})
    training = manifest.get("training_algorithm", {})
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-d6d-projection-training"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D6D-projection-training-v01"
        and manifest.get("status") == "frozen_unrun",
        "development_noncore_only": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False,
        "training_pilot_separation": manifest.get(
            "pilot_payload_accessed_during_training"
        ) is False,
        "base_and_updater_frozen": manifest.get("base_model_weights_trainable")
        is False
        and manifest.get("self_updater_allowed") is False,
        "self_source_exact": source.get("self_state_version") == "0.1"
        and source.get("fields") == ["identity_anchors", "active_goals"]
        and source.get("identity_keys") == list(IDENTITY_KEYS)
        and source.get("goal_keys") == list(GOAL_KEYS)
        and source.get("natural_language_self_state_serialization") is False,
        "capture_is_read_only_wrapper_owned": capture.get("wrapper_owned") is True
        and capture.get("condition") == "training_capture_observer_zero_delta"
        and capture.get("phase") == "post_ffn_residual"
        and capture.get("target_layer_index_zero_based") == 15
        and capture.get("sequence_position") == "last"
        and capture.get("residual_returned_unchanged") is True
        and capture.get("model_instance_dictionary_mutation_allowed") is False,
        "sixteen_training_calls_frozen": capture.get("record_order")
        == "identity_major_then_goal_major"
        and capture.get("record_count") == TRAINING_FORWARD_CALLS
        and capture.get("model_forward_calls") == TRAINING_FORWARD_CALLS,
        "prompt_blueprint_separate": isinstance(prompt.get("history_template"), str)
        and "{identity}" in prompt["history_template"]
        and "{goal}" in prompt["history_template"]
        and isinstance(prompt.get("capture_suffix"), str)
        and prompt.get("self_state_object_is_not_rendered") is True
        and prompt.get("pilot_template_reuse_allowed") is False,
        "closed_form_field_separated_projection": training.get("trainer_kind")
        == "two_way_centered_additive_branch_means_closed_form_v0.1"
        and training.get("grand_mean_handling")
        == "split_equally_between_identity_and_goal_branches_no_explicit_bias"
        and training.get("identity_branch")
        == "mean_by_identity_minus_half_grand_mean"
        and training.get("goal_branch") == "mean_by_goal_minus_half_grand_mean",
        "projection_shape_and_scale_frozen": training.get("output_dimension") == 2560
        and training.get("branch_target_rms_ratio_each") == 0.005
        and training.get("combined_target_rms_ratio_nominal") == 0.01
        and training.get("optimizer_seed") == 260825
        and training.get("bias_present") is False
        and training.get("double_mask_projection_exact_zero") is True,
        "digests_before_pilot_required": training.get("parameter_digest_required")
        is True
        and training.get("artifact_digest_required_before_pilot_load") is True,
        "failure_consumes_claim": manifest.get("failure_policy")
        == "persist_failure_consume_claim_stop_without_pilot_or_rerun",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6D-II training manifest failed closed: " + ", ".join(failed))
    return checks


def expand_training_records(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    validate_training_manifest(manifest)
    blueprint = manifest["prompt_blueprint"]
    records = []
    for identity in IDENTITY_KEYS:
        for goal in GOAL_KEYS:
            history = blueprint["history_template"].format(
                identity=identity, goal=goal
            )
            prompt = history + blueprint["capture_suffix"]
            records.append(
                {
                    "record_id": f"d6d-train-{identity}-{goal}",
                    "identity_key": identity,
                    "goal_key": goal,
                    "prompt_text": prompt,
                    "prompt_sha256": sha256_json(prompt),
                    "pilot_eligible": False,
                }
            )
    if len(records) != TRAINING_FORWARD_CALLS:
        raise AssertionError("D6D-II expanded training record count changed")
    return tuple(records)


def load_pilot_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _object(path, "pilot manifest")
    validate_pilot_manifest(manifest)
    return manifest


def _render_query(manifest: Mapping[str, Any], fixture: Mapping[str, Any]) -> str:
    template = manifest["query_templates"][fixture["task_family"]]
    sentinel = manifest["general_capability_sentinel"]
    sentinel_text = (
        sentinel["prefix_text"]
        + sentinel["expected_text"]
        + sentinel["suffix_text"]
    )
    return sentinel_text + template.format(**fixture["options"]) + "\n\nAssistant: <think></think>\n"


def validate_pilot_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    boundary = manifest.get("answer_boundary", {})
    sentinel = manifest.get("general_capability_sentinel", {})
    schedule = manifest.get("schedule", {})
    fixtures = manifest.get("fixtures")
    decision = manifest.get("decision_contract", {})
    if not isinstance(fixtures, list):
        raise PermissionError("D6D-II pilot fixtures must be a list")
    fixture_ids = [item.get("fixture_id") for item in fixtures if isinstance(item, dict)]
    families = [item.get("task_family") for item in fixtures if isinstance(item, dict)]
    targets = [item.get("target_code") for item in fixtures if isinstance(item, dict)]
    key_sets_valid = all(
        isinstance(item, dict)
        and item.get("matched_identity") in IDENTITY_KEYS
        and item.get("paired_identity") in IDENTITY_KEYS
        and item.get("matched_goal") in GOAL_KEYS
        and item.get("paired_goal") in GOAL_KEYS
        and set(item.get("options", {})) == set(CHOICE_TOKEN_IDS)
        for item in fixtures
    )
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-d6d-blinded-noncore-pilot"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D6D-blinded-pilot-v01"
        and manifest.get("status")
        == "frozen_unrun_load_only_after_projection_artifact_freeze",
        "development_noncore_only": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False,
        "pilot_excluded_from_training": manifest.get("projection_training_eligible")
        is False
        and manifest.get("load_order")
        == "after_projection_parameter_and_artifact_digests_persisted",
        "self_not_prompt_serialized": manifest.get("prompt_contains_current_self_state")
        is False
        and manifest.get("recurrent_state_reset_per_call") is True,
        "general_capability_sentinel_frozen_without_extra_calls": sentinel.get(
            "prefix_text"
        ) == "[NON-CORE D6D GENERAL SENTINEL] The stable neutral code is "
        and sentinel.get("expected_text") == "A"
        and sentinel.get("expected_token_id") == 66
        and sentinel.get("suffix_text") == ". "
        and sentinel.get("evaluation")
        == "teacher_forced_position_from_same_full_output_forward"
        and sentinel.get("same_sentinel_in_all_three_families") is True
        and sentinel.get("extra_model_forward_calls") == 0,
        "answer_tokens_frozen": boundary.get("forced_prefix_text") == ">\n"
        and boundary.get("forced_prefix_token_ids") == list(FORCED_PREFIX_TOKEN_IDS)
        and boundary.get("choice_token_ids") == CHOICE_TOKEN_IDS
        and boundary.get("tokenizer_sha256") == TOKENIZER_DIGEST
        and boundary.get("single_token_roundtrip_required") is True,
        "conditions_exact": manifest.get("conditions") == list(CONDITIONS),
        "schedule_counts_exact": schedule.get("fixture_count") == 12
        and schedule.get("unscored_off_precondition_per_fixture") == 1
        and schedule.get("scored_conditions_per_fixture") == 11
        and schedule.get("calls_per_fixture") == 12
        and schedule.get("model_forward_calls_total") == PILOT_FORWARD_CALLS
        and schedule.get("scored_forward_calls_total") == 132
        and schedule.get("real_self_projection_calls_total") == 96,
        "twelve_unique_fixtures": len(fixtures) == 12
        and len(fixture_ids) == len(set(fixture_ids)) == 12,
        "three_families_balanced": all(families.count(name) == 4 for name, _ in TASK_FAMILIES),
        "answer_codes_balanced": all(targets.count(code) == 3 for code in CHOICE_TOKEN_IDS),
        "fixture_keys_and_options_valid": key_sets_valid,
        "queries_render_without_self_statement": all(
            "current Self State is not written" in _render_query(manifest, item)
            for item in fixtures
        ),
        "decision_thresholds_frozen": decision.get("exact_control_matches_required") == 12
        and decision.get("synthetic_output_differences_required") == 12
        and decision.get("minimum_directional_fixture_passes_per_four") == 3
        and decision.get("maximum_general_capability_sentinel_code_changes") == 1
        and decision.get("maximum_nonfinite_outputs") == 0
        and decision.get("classification_only")
        == "noncore_engineering_pilot_not_self_effect_conclusion",
        "no_posthoc_or_rerun": decision.get("posthoc_route_removal_allowed") is False
        and decision.get("automatic_rerun_allowed") is False
        and manifest.get("failure_policy")
        == "persist_failure_consume_claim_stop_without_rerun_or_route_splitting",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6D-II pilot manifest failed closed: " + ", ".join(failed))
    return checks


def expand_pilot_fixtures(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    validate_pilot_manifest(manifest)
    values = []
    for fixture in manifest["fixtures"]:
        value = copy.deepcopy(fixture)
        value["query_text"] = _render_query(manifest, fixture)
        value["query_sha256"] = sha256_json(value["query_text"])
        values.append(value)
    return tuple(values)


def build_pilot_call_plan(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    fixtures = expand_pilot_fixtures(manifest)
    rows = balanced_condition_rows()
    calls = []
    for fixture_index, (fixture, row) in enumerate(zip(fixtures, rows)):
        calls.append(
            {
                "call_index": len(calls) + 1,
                "fixture_index": fixture_index,
                "fixture_id": fixture["fixture_id"],
                "phase": "off_precondition_unscored",
                "condition": "wrapper_off",
            }
        )
        for condition in row:
            calls.append(
                {
                    "call_index": len(calls) + 1,
                    "fixture_index": fixture_index,
                    "fixture_id": fixture["fixture_id"],
                    "phase": "scored",
                    "condition": condition,
                }
            )
    if len(calls) != PILOT_FORWARD_CALLS:
        raise AssertionError("D6D-II expanded pilot call count changed")
    return tuple(calls)


def build_manifest_report(
    *, training_path: str | Path, pilot_path: str | Path
) -> dict[str, Any]:
    training = load_training_manifest(training_path)
    pilot = load_pilot_manifest(pilot_path)
    training_checks = validate_training_manifest(training)
    pilot_checks = validate_pilot_manifest(pilot)
    training_records = expand_training_records(training)
    pilot_fixtures = expand_pilot_fixtures(pilot)
    pilot_calls = build_pilot_call_plan(pilot)
    training_digest = sha256_file(training_path)
    pilot_digest = sha256_file(pilot_path)
    checks = {
        "training_manifest_valid": True,
        "pilot_manifest_valid": True,
        "manifest_digests_distinct": training_digest != pilot_digest,
        "sixteen_training_records": len(training_records) == TRAINING_FORWARD_CALLS,
        "twelve_pilot_fixtures": len(pilot_fixtures) == 12,
        "one_hundred_forty_four_pilot_calls": len(pilot_calls) == PILOT_FORWARD_CALLS,
        "one_hundred_sixty_total_calls": TRAINING_FORWARD_CALLS
        + len(pilot_calls) == TOTAL_FORWARD_CALLS,
        "training_prompts_excluded_from_pilot": not (
            {item["prompt_sha256"] for item in training_records}
            & {item["query_sha256"] for item in pilot_fixtures}
        ),
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D-II manifest report failed: " + ", ".join(failed))
    report = {
        "report_version": MANIFEST_REPORT_VERSION,
        "status": "d6d_ii_training_and_blinded_pilot_manifests_frozen",
        "valid": True,
        "checks": checks,
        "training_checks": training_checks,
        "pilot_checks": pilot_checks,
        "training_manifest_sha256": training_digest,
        "pilot_manifest_sha256": pilot_digest,
        "expanded_training_commitment_sha256": sha256_json(training_records),
        "expanded_pilot_fixture_commitment_sha256": sha256_json(pilot_fixtures),
        "expanded_pilot_call_plan_commitment_sha256": sha256_json(pilot_calls),
        "counts": {
            "training_forward_calls": TRAINING_FORWARD_CALLS,
            "pilot_forward_calls": PILOT_FORWARD_CALLS,
            "total_forward_calls": TOTAL_FORWARD_CALLS,
            "training_manifest_checks": len(training_checks),
            "pilot_manifest_checks": len(pilot_checks),
        },
        "safety": {
            "formal_test_set_used": False,
            "model_loaded": False,
            "model_executed": False,
            "real_projection_constructed": False,
            "pilot_results_observed": False,
            "self_effect_conclusion_made": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
