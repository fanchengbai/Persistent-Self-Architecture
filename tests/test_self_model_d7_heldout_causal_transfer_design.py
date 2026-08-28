from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d7_heldout_causal_transfer_design import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    D7_CONDITIONS,
    D7_GOALS,
    D7_IDENTITIES,
    D7_TASK_FAMILIES,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    analyze_independence,
    build_design_report,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D7HeldoutCausalTransferDesignTests(unittest.TestCase):
    def test_design_freezes_independent_question_and_authority(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_design(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertIn("previously unseen task families", payload["research_question"])
        self.assertFalse(payload["independence_contract"]["d6d_execution_rerun"])
        self.assertFalse(payload["authority"]["d7_manifest_implementation_authorized"])
        for field in (
            "d7_compatibility_execution_authorized",
            "d7_capability_execution_authorized",
            "d7_effect_execution_authorized",
        ):
            self.assertFalse(payload["authority"][field])

    def test_training_heldout_split_and_counts_are_exact(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        training = payload["training_design"]
        heldout = payload["heldout_design"]
        self.assertEqual(tuple(training["identity_keys"]), D7_IDENTITIES)
        self.assertEqual(tuple(training["goal_keys"]), D7_GOALS)
        self.assertEqual(training["capture_count"], 25)
        self.assertEqual(tuple(heldout["task_families"]), D7_TASK_FAMILIES)
        self.assertEqual(tuple(heldout["conditions"]), D7_CONDITIONS)
        self.assertEqual(heldout["fixture_count"], 64)
        self.assertEqual(heldout["heldout_forward_calls"], 896)
        self.assertEqual(heldout["projection_training_plus_heldout_forward_calls"], 921)

    def test_independence_is_checked_against_frozen_d6d_files(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        report = analyze_independence(payload, ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(set(report["d6d_identity_keys"]).isdisjoint(report["d7_identity_keys"]))
        self.assertTrue(set(report["d6d_goal_keys"]).isdisjoint(report["d7_goal_keys"]))
        self.assertTrue(set(report["d6d_task_families"]).isdisjoint(report["d7_task_families"]))

    def test_four_gates_have_separate_authority_and_stop_order(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        gates = payload["gate_sequence"]
        self.assertEqual([gate["gate_id"] for gate in gates], ["D7-B", "D7-C", "D7-D", "D7-E"])
        self.assertTrue(all(gate["separate_authorization_required"] for gate in gates))
        self.assertFalse(gates[0]["model_execution"])
        self.assertEqual([gate["future_forward_calls"] for gate in gates[1:]], [18, 64, 921])

    def test_compatibility_gate_explicitly_covers_none_state(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        gate = payload["compatibility_gate"]
        self.assertEqual(gate["state_inputs"], ["none", "prebuilt"])
        self.assertEqual(gate["execution_paths"], ["forward_one", "forward_seq"])
        self.assertEqual(gate["full_output_values"], [False, True])
        self.assertTrue(gate["public_state_none_initialization_required"])
        self.assertEqual(gate["total_forward_calls"], 18)
        self.assertFalse(gate["heldout_payload_accessed"])

    def test_causal_thresholds_are_fixed_and_noncore_only(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        endpoints = payload["causal_endpoints"]
        primary = endpoints["primary_conjunctive"]
        self.assertEqual(primary["minimum_matched_joint_accuracy"], 0.75)
        self.assertEqual(primary["minimum_dual_swap_injected_joint_accuracy"], 0.7)
        self.assertTrue(primary["matched_vs_dual_mask_cluster_bootstrap_95_lower_bound_positive"])
        self.assertTrue(endpoints["all_primary_thresholds_fixed_before_any_d7_model_execution"])
        self.assertTrue(endpoints["passing_d7e_is_noncore_engineering_evidence_only"])
        self.assertFalse(endpoints["self_effect_conclusion_allowed"])

    def test_scope_or_independence_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "d6d_rerun_authorized", True),
            ("authority", "d7_manifest_implementation_authorized", True),
            ("authority", "d7_effect_execution_authorized", True),
            ("independence_contract", "d6d_pilot_fixtures_reused", True),
            ("training_design", "capture_count", 16),
            ("heldout_design", "fixture_count", 12),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_design(changed)

    def test_report_is_no_model_and_unimplemented(self):
        report = build_design_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertEqual(report["count_derivation"]["single_joint_future_forward_calls"], 921)
        self.assertFalse(report["safety"]["d6d_rerun"])
        self.assertFalse(report["safety"]["d7_manifests_implemented"])
        self.assertFalse(report["safety"]["projection_implemented"])
        self.assertFalse(report["safety"]["model_executed"])

    def test_wrong_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_design_report(config_path=path, project_root=ROOT)

    def test_no_model_modules_or_execution_locks_used(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("PSA_SELF_MODEL_D6D_REAL_JOINT", os.environ)


if __name__ == "__main__":
    unittest.main()
