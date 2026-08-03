from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preregistration/exp001b_supplemental_controls.draft.json"


class Exp001BSupplementalDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_draft_has_no_execution_authority(self) -> None:
        safety = self.config["safety_boundary"]
        self.assertEqual(
            self.config["status"],
            "design_confirmed_development_only",
        )
        self.assertTrue(self.config["design_review"]["b1_b7_confirmed"])
        self.assertTrue(
            self.config["design_review"]["does_not_authorize_formal_run"]
        )
        self.assertFalse(safety["supplemental_set_generated"])
        self.assertFalse(safety["supplemental_experiment_authorized"])
        self.assertFalse(safety["supplemental_experiment_run"])
        self.assertFalse(safety["supplemental_results_observed"])
        self.assertFalse(safety["automatic_rerun_authorized"])
        self.assertFalse(safety["modify_exp001_artifacts_authorized"])

    def test_parent_evidence_is_exact_and_read_only(self) -> None:
        parent = self.config["parent_evidence"]
        self.assertEqual(parent["experiment_id"], "EXP-001")
        self.assertEqual(
            parent["raw_group_payload_digest_sha256"],
            "db4ba70ed521b55f23c4fc0ddafd2fb09af3cbe0132c0f065358a96f858b5ba7",
        )
        self.assertTrue(parent["results_already_observed"])
        self.assertIn("rerun EXP-001 primary conditions", parent["forbidden_use"])

    def test_record_budget_is_exact(self) -> None:
        budget = self.config["record_budget"]
        self.assertEqual(budget["matched_context_records"], 5120)
        self.assertEqual(budget["formal_generation_records"], 5120)
        self.assertEqual(budget["general_control_condition_records"], 96 * 8)
        self.assertEqual(
            budget["total_new_records"],
            sum(value for key, value in budget.items() if key != "total_new_records"),
        )

    def test_controls_reuse_frozen_d5_design(self) -> None:
        controls = self.config["general_capability_controls"]
        self.assertTrue(controls["reuse_exact_exp001_d5_manifest"])
        self.assertEqual(controls["generator_seed"], 62511541)
        self.assertEqual(controls["trial_count"], 96)
        self.assertEqual(len(controls["conditions"]), 8)
        self.assertEqual(controls["new_record_count"], 768)
        self.assertEqual(controls["alerts"]["maximum_accuracy_drop_points"], 0.05)
        self.assertEqual(controls["alerts"]["maximum_format_drop_points"], 0.02)
        self.assertEqual(
            controls["alerts"]["maximum_mean_target_log_prob_drop"],
            0.25,
        )

    def test_matched_templates_are_unbound_and_balanced(self) -> None:
        matched = self.config["matched_context"]
        self.assertEqual(matched["factorial_group_count"], 320)
        self.assertEqual(matched["source_trial_count"], 5120)
        self.assertEqual(len(matched["templates"]), 4)
        for template in matched["templates"]:
            text = template["user_text"]
            self.assertEqual(text.count("{domain}"), 1)
            self.assertEqual(text.count("{operation}"), 1)
            for phrase in matched["forbidden_binding_phrases"]:
                self.assertNotIn(phrase, text)

    def test_formal_generation_thresholds_are_inherited(self) -> None:
        thresholds = self.config["formal_generation_readout"]["thresholds"]
        self.assertEqual(thresholds["forced_prefix_greedy_exact_rate"], 1.0)
        self.assertEqual(thresholds["minimum_format_valid_rate"], 0.99)
        self.assertEqual(thresholds["minimum_joint_accuracy_lower_bound"], 0.8)
        self.assertEqual(thresholds["minimum_identity_accuracy_lower_bound"], 0.9)
        self.assertEqual(thresholds["minimum_goal_accuracy_lower_bound"], 0.9)
        self.assertEqual(thresholds["maximum_answer_position_accuracy_gap"], 0.25)

    def test_state_norm_quantile_is_predeclared(self) -> None:
        calibration = self.config["state_norm_development_calibration"]
        self.assertEqual(calibration["group_count"], 64)
        self.assertIn("nearest-rank", calibration["threshold"])
        self.assertTrue(calibration["core_set_access_forbidden"])

    def test_named_seeds_match_public_derivation(self) -> None:
        seeds = self.config["seeds"]
        namespace = "PSA|EXP-001B|supplemental-v1|"
        names = {
            "matched_context_generator": "matched-context-generator",
            "control_assignment": "control-assignment",
            "bootstrap": "bootstrap",
            "permutation": "permutation",
        }
        for key, purpose in names.items():
            expected = int.from_bytes(
                hashlib.sha256((namespace + purpose).encode()).digest()[:4],
                "big",
            )
            self.assertEqual(seeds[key], expected)


if __name__ == "__main__":
    unittest.main()
