from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_file
from psa.self_model.d6d_failure_state_initialization_diagnostic import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    audit_frozen_source,
    build_diagnostic_report,
    run_synthetic_dispatch_triangle,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D6DFailureStateInitializationDiagnosticTests(unittest.TestCase):
    def test_config_freezes_failure_claim_and_authority(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_config(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        evidence = payload["frozen_failure_evidence"]
        self.assertTrue(evidence["claim_consumed"])
        self.assertEqual(evidence["training_captures_completed"], 0)
        self.assertFalse(evidence["projection_artifact_constructed"])
        self.assertEqual(evidence["pilot_forward_calls_completed"], 0)
        self.assertFalse(payload["authority"]["d6d_rerun_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])

    def test_scope_failure_or_route_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d6d_fix_implementation_authorized", True),
            ("authority", "d6d_rerun_authorized", True),
            ("frozen_failure_evidence", "claim_consumed", False),
            ("frozen_failure_evidence", "training_captures_completed", 1),
            ("route_review", "independent_new_experiment_route_established", True),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_ast_audit_converges_on_dispatch_and_test_gap(self):
        audit = audit_frozen_source(ROOT)
        self.assertTrue(audit["valid"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(
            set(audit["wrapper_direct_dispatch_lines"]),
            {"forward_one", "forward_seq"},
        )
        self.assertEqual(audit["wrapper_zero_state_call_count"], 0)
        self.assertEqual(audit["existing_wrapper_test_state_argument"], "_state()")

    def test_synthetic_triangle_reproduces_exact_failure(self):
        result = run_synthetic_dispatch_triangle()
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["public_case"]["zero_state_calls"], 1)
        self.assertEqual(result["none_case"]["error_type"], "TypeError")
        self.assertEqual(
            result["none_case"]["error"],
            "'NoneType' object is not subscriptable",
        )
        self.assertEqual(result["none_case"]["dispatcher_calls"], 0)
        self.assertEqual(result["prebuilt_case"]["dispatcher_calls"], 32)

    def test_report_closes_root_cause_without_opening_rerun(self):
        report = build_diagnostic_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertEqual(
            report["findings"]["root_cause"],
            "wrapper_direct_child_dispatch_skips_public_forward_zero_state_initialization",
        )
        self.assertFalse(report["findings"]["independent_successor_established"])
        self.assertFalse(report["safety"]["d6d_fix_implemented"])
        self.assertFalse(report["safety"]["d6d_rerun"])
        self.assertFalse(report["safety"]["d6e_authorized"])

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
        self.assertNotIn("PSA_SELF_MODEL_D6D_REAL_JOINT", os.environ)


if __name__ == "__main__":
    unittest.main()
