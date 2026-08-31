from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d9a_within_wrapper_causal_isolation import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    CONTRASTS,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    analyze_expansion_and_independence,
    build_design_report,
    evaluate_candidate,
    expand_fixtures,
    expand_schedule,
    run_synthetic_endpoint_review,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


class D9AWithinWrapperCausalIsolationTests(unittest.TestCase):
    def test_design_freezes_core_question_nonreuse_and_authority(self):
        payload = _load()
        checks = validate_design(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertIn("Within one persistent wrapper path", payload["research_question"])
        self.assertEqual(
            payload["historical_boundary"]["d8c_use"],
            "route_confound_rationale_and_nonreuse_audit_only",
        )
        self.assertFalse(
            payload["historical_boundary"][
                "d8c_result_used_as_new_experiment_data"
            ]
        )
        self.assertFalse(payload["authority"]["projection_implementation_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])

    def test_calibration_and_heldout_are_deterministic_and_separate(self):
        payload = _load()
        first = expand_fixtures(payload)
        second = expand_fixtures(payload)
        self.assertEqual(first, second)
        calibration = first["calibration_manifest"]["fixtures"]
        heldout = first["heldout_manifest"]["fixtures"]
        self.assertEqual(len(calibration), 32)
        self.assertEqual(len(heldout), 64)
        calibration_tokens = {
            token for fixture in calibration for token in fixture["token_ids"]
        }
        heldout_tokens = {
            token for fixture in heldout for token in fixture["token_ids"]
        }
        self.assertTrue(calibration_tokens.isdisjoint(heldout_tokens))
        for base_case in {fixture["base_case_id"] for fixture in heldout}:
            selected = [f for f in heldout if f["base_case_id"] == base_case]
            self.assertEqual(
                len({tuple(f["content_token_ids"]) for f in selected}), 1
            )
            self.assertEqual(
                len({tuple(f["rotation_code_token_ids"]) for f in selected}), 4
            )
        self.assertEqual(
            first["calibration_manifest"]["commitment_sha256"],
            payload["calibration_design"]["expected_commitment_sha256"],
        )
        self.assertEqual(
            first["heldout_manifest"]["commitment_sha256"],
            payload["heldout_design"]["expected_commitment_sha256"],
        )

    def test_heldout_has_sixteen_cases_and_four_rotations(self):
        payload = _load()
        fixtures = expand_fixtures(payload)["heldout_manifest"]["fixtures"]
        base_cases = {fixture["base_case_id"] for fixture in fixtures}
        self.assertEqual(len(base_cases), 16)
        for base_case in base_cases:
            selected = [f for f in fixtures if f["base_case_id"] == base_case]
            self.assertEqual(len(selected), 4)
            self.assertEqual({f["code_rotation"] for f in selected}, {0, 1, 2, 3})

    def test_schedule_is_within_wrapper_counterbalanced_and_exact(self):
        payload = _load()
        fixtures = expand_fixtures(payload)
        schedule = expand_schedule(payload, fixtures)
        blocks = schedule["heldout_pair_blocks"]
        self.assertEqual(len(schedule["calibration_calls"]), 32)
        self.assertEqual(len(blocks), 448)
        self.assertEqual(32 + len(blocks) * 2, 928)
        for contrast in CONTRASTS:
            selected = [block for block in blocks if block["contrast"] == contrast]
            self.assertEqual(len(selected), 64)
            self.assertEqual(
                sum(block["pair_order"] == "zero_first" for block in selected), 32
            )
            self.assertEqual(
                sum(block["pair_order"] == "condition_first" for block in selected),
                32,
            )
            position_counts = [
                sum(block["latin_position"] == position for block in selected)
                for position in range(1, 8)
            ]
            self.assertLessEqual(max(position_counts) - min(position_counts), 1)
        self.assertTrue(all(block["route"] == "persistent_wrapper" for block in blocks))
        self.assertTrue(
            all("public" not in block["condition_order"] for block in blocks)
        )
        self.assertEqual(
            schedule["commitment_sha256"],
            payload["schedule_design"]["expected_commitment_sha256"],
        )

    def test_expansion_audit_closes_d8_reuse_and_namespace_leakage(self):
        audit = analyze_expansion_and_independence(_load(), ROOT)
        self.assertTrue(audit["valid"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["future_forward_call_count"], 928)
        self.assertEqual(set(audit["contrast_counts"].values()), {64})
        self.assertTrue(audit["namespaces"]["valid"])

    def test_candidate_requires_every_causal_specificity_gate(self):
        supported = {
            "active_minus_zero_mean": 0.2,
            "active_minus_zero_lb99": 0.1,
            "positive_base_cases": 16,
            "identity_level_min_positive": 4,
            "goal_level_min_positive": 4,
            "true_minus_random_lb99": 0.05,
            "mask_identity_specific_count": 16,
            "mask_goal_specific_count": 16,
            "swap_identity_follow_count": 16,
            "swap_goal_follow_count": 16,
            "synthetic_active_changed_fixture_count": 64,
        }
        result = evaluate_candidate(supported)
        self.assertTrue(result["all_gates_pass"])
        self.assertFalse(result["self_effect_conclusion"])
        for field in supported:
            changed = copy.deepcopy(supported)
            if field in {
                "active_minus_zero_mean",
                "active_minus_zero_lb99",
                "true_minus_random_lb99",
            }:
                changed[field] = 0.0
            else:
                changed[field] = 0
            with self.subTest(field=field):
                self.assertFalse(evaluate_candidate(changed)["all_gates_pass"])

    def test_candidate_rejects_missing_and_nonfinite_metrics(self):
        with self.assertRaises(ValueError):
            evaluate_candidate({"active_minus_zero_mean": 1.0})
        invalid = {
            "active_minus_zero_mean": float("nan"),
            "active_minus_zero_lb99": 0.1,
            "positive_base_cases": 16,
            "identity_level_min_positive": 4,
            "goal_level_min_positive": 4,
            "true_minus_random_lb99": 0.1,
            "mask_identity_specific_count": 16,
            "mask_goal_specific_count": 16,
            "swap_identity_follow_count": 16,
            "swap_goal_follow_count": 16,
            "synthetic_active_changed_fixture_count": 64,
        }
        with self.assertRaises(ValueError):
            evaluate_candidate(invalid)

    def test_synthetic_review_rejects_route_only_and_nonspecific_effects(self):
        review = run_synthetic_endpoint_review()
        self.assertTrue(review["valid"])
        self.assertTrue(all(review["checks"].values()))
        self.assertTrue(review["cases"]["field_specific_candidate"]["all_gates_pass"])
        self.assertFalse(review["cases"]["wrapper_route_only"]["all_gates_pass"])
        self.assertFalse(
            review["cases"]["nonspecific_active_or_random"]["all_gates_pass"]
        )

    def test_scope_commitment_threshold_or_authority_mutation_fails_closed(self):
        payload = _load()
        mutations = (
            ("authority", "model_execution_authorized", True),
            ("authority", "projection_implementation_authorized", True),
            ("authority", "d8c_rerun_authorized", True),
            ("historical_boundary", "d8c_fixture_reused", True),
            ("calibration_design", "seed", "changed"),
            ("heldout_design", "expected_commitment_sha256", "0" * 64),
            ("schedule_design", "total_future_forward_calls", 927),
            ("causal_conditions", "public_route_scored_or_used", True),
            ("endpoint_contract", "all_gates_required", False),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_design(changed)

    def test_report_is_design_only_and_no_model(self):
        report = build_design_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertEqual(
            report["expansion_and_independence"]["future_forward_call_count"],
            928,
        )
        self.assertFalse(report["safety"]["projection_implemented"])
        self.assertFalse(report["safety"]["projection_constructed"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(
            report["frozen_decision_contract"]["self_effect_conclusion_allowed"]
        )

    def test_wrong_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied.json"
            copied.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_design_report(config_path=copied, project_root=ROOT)

    def test_no_model_modules_or_execution_locks_used(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("PSA_SELF_MODEL_D9_REAL", os.environ)


if __name__ == "__main__":
    unittest.main()
