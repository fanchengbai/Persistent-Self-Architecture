from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_file
from psa.self_model.d7c_failure_difference_diagnostic import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    audit_frozen_source,
    build_diagnostic_report,
    observed_difference_fingerprint,
    run_synthetic_nonidentifiability_fixture,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D7CFailureDifferenceDiagnosticTests(unittest.TestCase):
    def test_config_freezes_failure_cells_and_authority(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_config(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(len(payload["cell_observations"]), 8)
        self.assertTrue(payload["frozen_failure_evidence"]["claim_consumed"])
        self.assertFalse(payload["authority"]["d7c_rerun_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])

    def test_scope_failure_or_route_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d7c_fix_implementation_authorized", True),
            ("authority", "d7c_rerun_authorized", True),
            ("frozen_failure_evidence", "claim_consumed", False),
            ("route_review", "candidate_execution_authorized", True),
            ("route_review", "d7c_causal_source_identifiable_from_existing_evidence", True),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_cell_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(payload)
        changed["cell_observations"][0]["first_nonexact_state_index"] = 5
        with self.assertRaises(PermissionError):
            validate_config(changed)

    def test_ast_audit_finds_order_and_missing_identification_controls(self):
        audit = audit_frozen_source(ROOT)
        self.assertTrue(audit["valid"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertLess(audit["public_call_line"], audit["wrapper_call_line"])
        self.assertEqual(
            audit["within_route_repeatability_calls_per_cell"],
            {"public": 1, "wrapper": 1},
        )
        self.assertFalse(audit["counterbalanced_order_present"])

    def test_observed_fingerprint_preserves_all_eight_cells(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        result = observed_difference_fingerprint(payload["cell_observations"])
        self.assertEqual(result["cell_count"], 8)
        self.assertTrue(result["all_logits_nonexact"])
        self.assertTrue(result["all_states_nonexact"])
        self.assertTrue(result["all_state_components_compatible"])
        self.assertEqual(result["common_exact_state_indices"], [0, 1, 2, 3])
        self.assertEqual(result["common_first_nonexact_state_index"], 4)
        self.assertEqual(result["common_max_error_state_index"], 94)
        self.assertTrue(result["state_input_not_unique_explanation"])
        self.assertTrue(result["full_output_not_unique_explanation"])

    def test_two_distinct_causes_have_the_same_d7c_fingerprint(self):
        result = run_synthetic_nonidentifiability_fixture()
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["identifiability_result"],
            "same_observation_two_distinct_causes",
        )
        left, right = result["mechanisms"]
        self.assertNotEqual(left["id"], right["id"])
        self.assertNotEqual(left["route_semantics_equal"], right["route_semantics_equal"])
        self.assertEqual(left["fingerprint"], right["fingerprint"])

    def test_report_closes_at_nonidentifiability_and_design_only_route(self):
        report = build_diagnostic_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertFalse(report["findings"]["unique_cause_identified"])
        self.assertTrue(
            report["findings"]["independent_design_candidate_established"]
        )
        self.assertFalse(report["findings"]["candidate_execution_authorized"])
        self.assertFalse(report["safety"]["d7c_fix_implemented"])
        self.assertFalse(report["safety"]["d7c_rerun"])
        self.assertFalse(report["safety"]["model_executed"])

    def test_source_locks_match_frozen_production_files(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        for relative, digest in payload["frozen_source_locks"].items():
            with self.subTest(relative=relative):
                self.assertEqual(sha256_file(ROOT / relative), digest)

    def test_wrong_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_diagnostic_report(config_path=path, project_root=ROOT)

    def test_no_model_modules_or_execution_lock_used(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("PSA_SELF_MODEL_D7C_REAL_COMPATIBILITY", os.environ)


if __name__ == "__main__":
    unittest.main()
