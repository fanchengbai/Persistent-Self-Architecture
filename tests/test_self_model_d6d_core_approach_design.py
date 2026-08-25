from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d6d_core_approach_design import (
    CONDITIONS,
    CONFIG_RELATIVE_PATH,
    NEXT_CONFIRMATION,
    REQUIRED_CONFIRMATION,
    SELF_CONDITIONS,
    balanced_condition_rows,
    build_d6d_no_model_review,
    build_no_model_call_plan,
    condition_projection_plan,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D6DCoreApproachDesignTests(unittest.TestCase):
    def test_exact_owner_scope_and_next_gate_are_frozen(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(
            config["required_next_owner_confirmation_text"], NEXT_CONFIRMATION
        )
        self.assertTrue(all(validate_design(config).values()))

    def test_one_joint_plan_contains_all_controls_without_split(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        contract = config["single_experiment_contract"]
        self.assertTrue(contract["synthetic_positive_control_interleaved_with_self_conditions"])
        self.assertFalse(contract["separate_mechanism_execution_round_allowed"])
        self.assertFalse(contract["raw_original_route_allowed"])
        self.assertEqual(config["causal_conditions"], list(CONDITIONS))

    def test_balanced_plan_has_twelve_fixtures_and_144_calls(self):
        rows = balanced_condition_rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0], rows[-1])
        self.assertTrue(all(set(row) == set(CONDITIONS) for row in rows))
        calls = build_no_model_call_plan()
        self.assertEqual(len(calls), 144)
        self.assertEqual(len({call["fixture_id"] for call in calls}), 12)
        scored = [call for call in calls if call["phase"] == "scored"]
        for condition in CONDITIONS:
            self.assertEqual(
                sum(call["condition"] == condition for call in scored), 12
            )

    def test_projection_conditions_distinguish_none_synthetic_and_real_self(self):
        plans = {value: condition_projection_plan(value) for value in CONDITIONS}
        self.assertEqual(plans["wrapper_off"]["projection"], "none")
        self.assertEqual(plans["wrapper_zero"]["projection"], "none")
        self.assertEqual(
            plans["synthetic_positive"]["projection"], "synthetic_positive"
        )
        self.assertTrue(
            all(plans[condition]["projection"] == "frozen_self" for condition in SELF_CONDITIONS)
        )
        self.assertEqual(plans["self_identity_swap"]["identity"], "paired_swap")
        self.assertEqual(plans["self_identity_swap"]["goal"], "matched")
        self.assertEqual(plans["self_goal_mask"]["goal"], "zero_mask")
        self.assertEqual(
            plans["self_identity_goal_norm_matched_random"]["identity"],
            "seeded_norm_matched_random",
        )

    def test_wrapper_contract_forbids_real_instance_mutation(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        wrapper = config["wrapper_owned_persistent_contract"]
        self.assertEqual(wrapper["owner"], "project_wrapper_not_real_rwkv_instance")
        self.assertFalse(wrapper["real_model_instance_dictionary_mutation_allowed"])
        self.assertFalse(wrapper["real_model_setattr_or_delattr_allowed"])
        self.assertEqual(wrapper["instrumented_methods_bound_to"], "wrapper_only")

    def test_projection_is_real_design_but_not_constructed_or_authorized(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        projection = config["projection_contract"]
        authority = config["authority"]
        self.assertEqual(
            projection["kind"], "field_separated_learned_frozen_self_projection"
        )
        self.assertFalse(projection["synthetic_or_hash_fake"])
        self.assertFalse(projection["projection_bias_allowed"])
        self.assertTrue(projection["double_mask_projection_exact_zero"])
        self.assertFalse(projection["projection_constructed_this_round"])
        self.assertFalse(authority["projection_training_or_construction_authorized"])
        self.assertFalse(authority["model_execution_authorized"])
        self.assertFalse(authority["self_effect_conclusion_authorized"])

    def test_numeric_decision_thresholds_are_frozen_before_execution(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        decision = config["decision_contract"]
        self.assertEqual(
            decision["thresholds"],
            {
                "exact_control_matches_required": 12,
                "synthetic_output_differences_required": 12,
                "minimum_directional_fixture_passes_per_four": 3,
                "maximum_general_capability_sentinel_code_changes": 1,
                "maximum_nonfinite_outputs": 0,
            },
        )
        self.assertTrue(
            decision["thresholds_frozen_in_this_design_before_any_model_execution"]
        )

    def test_scope_or_schedule_expansion_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("model_execution_authorized", True),
            ("d6c_rerun_authorized", True),
            ("d6e_authorized", True),
            ("self_effect_conclusion_authorized", True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            for name, value in mutations:
                changed = copy.deepcopy(payload)
                changed["authority"][name] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(PermissionError):
                    validate_design(json.loads(path.read_text(encoding="utf-8")))
            changed = copy.deepcopy(payload)
            changed["schedule"]["model_forward_calls_total"] = 145
            with self.assertRaises(PermissionError):
                validate_design(changed)
            changed = copy.deepcopy(payload)
            changed["single_experiment_contract"][
                "separate_mechanism_execution_round_allowed"
            ] = True
            with self.assertRaises(PermissionError):
                validate_design(changed)

    def test_static_review_is_no_model_and_complete(self):
        report = build_d6d_no_model_review(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["counts"]["planned_model_forward_calls"], 144)
        self.assertEqual(report["counts"]["planned_real_self_projection_calls"], 96)
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["real_self_projection_constructed"])
        self.assertFalse(report["design"]["separate_mechanism_execution_round"])
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_copied_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d6d_no_model_review(config_path=path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
