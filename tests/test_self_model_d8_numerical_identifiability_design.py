from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d8_numerical_identifiability_design import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    FORBIDDEN_D7C_TOKENS,
    NEXT_GATE,
    PAIR_TYPES,
    REQUIRED_CONFIRMATION,
    analyze_expansion_and_independence,
    build_design_report,
    excess_drift_from_distances,
    expand_fixtures,
    expand_schedule,
    run_synthetic_endpoint_review,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D8NumericalIdentifiabilityDesignTests(unittest.TestCase):
    def test_design_freezes_question_nonreuse_and_authority(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_design(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertIn("within-public", payload["research_question"])
        self.assertFalse(
            payload["historical_boundary"][
                "d7c_quantitative_results_used_as_new_experiment_data"
            ]
        )
        self.assertFalse(payload["authority"]["d8b_authorized"])
        self.assertFalse(payload["authority"]["d8c_real_execution_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])

    def test_fixture_expansion_is_new_deterministic_and_committed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        first = expand_fixtures(payload)
        second = expand_fixtures(payload)
        self.assertEqual(first, second)
        self.assertEqual(len(first["conditioning_fixtures"]), 4)
        self.assertEqual(len(first["scored_fixtures"]), 24)
        tokens = [
            token
            for fixture in first["conditioning_fixtures"] + first["scored_fixtures"]
            for token in fixture["token_ids"]
        ]
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertTrue(set(tokens).isdisjoint(FORBIDDEN_D7C_TOKENS))
        self.assertEqual(
            first["fixture_commitment_sha256"],
            payload["fixture_design"]["expected_fixture_commitment_sha256"],
        )

    def test_schedule_has_four_pairs_three_replicates_and_exact_balance(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        fixtures = expand_fixtures(payload)
        schedule = expand_schedule(payload, fixtures)
        blocks = schedule["pair_blocks"]
        self.assertEqual(len(schedule["conditioning_calls"]), 8)
        self.assertEqual(len(blocks), 288)
        self.assertEqual(len(blocks) * 2 + len(schedule["conditioning_calls"]), 584)
        for pair_type in PAIR_TYPES:
            self.assertEqual(sum(block["pair_type"] == pair_type for block in blocks), 72)
            for position in range(1, 5):
                self.assertEqual(
                    sum(
                        block["pair_type"] == pair_type
                        and block["latin_position"] == position
                        for block in blocks
                    ),
                    18,
                )
        self.assertEqual(
            schedule["schedule_commitment_sha256"],
            payload["schedule_design"]["expected_schedule_commitment_sha256"],
        )

    def test_expansion_audit_closes_d7c_and_namespace_reuse(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        audit = analyze_expansion_and_independence(payload)
        self.assertTrue(audit["valid"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(set(audit["pair_counts"].values()), {72})
        self.assertEqual(set(audit["stratum_counts"].values()), {6})
        self.assertEqual(audit["total_future_forward_call_count"], 584)
        self.assertTrue(audit["namespaces"]["valid"])

    def test_conservative_endpoint_rejects_one_order_only_effect(self):
        route_effect = excess_drift_from_distances(
            {
                "public_public": 0.001,
                "wrapper_wrapper": 0.002,
                "public_wrapper": 0.02,
                "wrapper_public": 0.018,
            }
        )
        order_only = excess_drift_from_distances(
            {
                "public_public": 0.001,
                "wrapper_wrapper": 0.001,
                "public_wrapper": 0.02,
                "wrapper_public": 0.001,
            }
        )
        self.assertGreater(route_effect["excess_drift"], 0.0)
        self.assertEqual(order_only["excess_drift"], 0.0)
        self.assertGreater(order_only["order_interaction"], 0.0)

    def test_endpoint_rejects_missing_or_negative_distance(self):
        with self.assertRaises(ValueError):
            excess_drift_from_distances({"public_public": 0.0})
        changed = {name: 0.0 for name in PAIR_TYPES}
        changed["wrapper_public"] = -0.1
        with self.assertRaises(ValueError):
            excess_drift_from_distances(changed)

    def test_synthetic_endpoint_review_distinguishes_three_causes(self):
        review = run_synthetic_endpoint_review()
        self.assertTrue(review["valid"])
        self.assertTrue(all(review["checks"].values()))
        self.assertGreater(
            review["cases"]["route_specific_excess"]["excess_drift"], 0.0
        )
        self.assertEqual(
            review["cases"]["public_first_order_only"]["excess_drift"], 0.0
        )
        self.assertEqual(
            review["cases"]["shared_background_repeatability"]["excess_drift"],
            0.0,
        )

    def test_scope_seed_commitment_or_authority_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d8b_authorized", True),
            ("authority", "d7c_rerun_authorized", True),
            ("historical_boundary", "d7c_eight_cells_reused", True),
            ("fixture_design", "fixture_seed", "changed"),
            ("schedule_design", "schedule_seed", "changed"),
            ("schedule_design", "expected_schedule_commitment_sha256", "0" * 64),
            ("determinism_policy", "torch_use_deterministic_algorithms", False),
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
            report["expansion_and_independence"]["total_future_forward_call_count"],
            584,
        )
        self.assertFalse(report["safety"]["execution_entry_implemented"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d7c_rerun"])
        self.assertFalse(report["frozen_decision_contract"]["self_effect_conclusion_allowed"])

    def test_wrong_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_design_report(config_path=path, project_root=ROOT)

    def test_no_model_modules_or_execution_locks_used(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("PSA_SELF_MODEL_D8_REAL", os.environ)


if __name__ == "__main__":
    unittest.main()
