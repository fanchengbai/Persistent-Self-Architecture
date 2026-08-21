from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    REQUIRED_CONFIRMATION,
    _schedule_analysis,
    build_diagnostic_report,
    run_plain_python_lifecycle_fixture,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CFailureLifecycleDiagnosticTests(unittest.TestCase):
    def test_config_freezes_authority_and_failed_evidence(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_config(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertFalse(payload["d5c_frozen_evidence"]["real_report_valid"])
        self.assertEqual(
            payload["d5c_frozen_evidence"]["decision_effect"],
            "stop_without_rerun",
        )
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["d5c_rerun_authorized"])
        self.assertFalse(payload["authority"]["d5d_authorized"])

    def test_scope_or_evidence_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d5c_rerun_authorized", True),
            ("authority", "d5d_authorized", True),
            ("d5c_frozen_evidence", "real_report_valid", True),
            ("d5c_frozen_evidence", "probe_applications_observed", 10),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_schedule_confounds_every_scored_original_with_post_active_position(self):
        analysis = _schedule_analysis()
        self.assertEqual(analysis["per_fixture_scored_original_count"], 4)
        self.assertEqual(analysis["total_scored_original_count"], 8)
        self.assertTrue(analysis["all_scored_originals_immediately_follow_active"])
        self.assertTrue(analysis["extra_applications_match_post_active_originals"])
        self.assertTrue(analysis["extra_invocations_match_layers_times_originals"])
        self.assertFalse(analysis["schedule_can_separate_route_from_predecessor_effect"])

    def test_plain_python_forward_one_does_not_reproduce_leak(self):
        result = run_plain_python_lifecycle_fixture([3])
        self.assertTrue(result["valid"])
        self.assertEqual(result["execution_path"], "forward_one")
        self.assertTrue(result["checks"]["active_differs_from_baseline"])
        self.assertTrue(result["checks"]["post_active_original_returns_to_baseline"])
        self.assertTrue(result["checks"]["temporary_instance_bindings_absent"])
        self.assertEqual(result["counts_after_active"], {"invocations": 32, "applications": 1})
        self.assertEqual(result["counts_after_original"], result["counts_after_active"])

    def test_plain_python_forward_seq_does_not_reproduce_leak(self):
        result = run_plain_python_lifecycle_fixture([3, 5, 8])
        self.assertTrue(result["valid"])
        self.assertEqual(result["execution_path"], "forward_seq")
        self.assertTrue(result["checks"]["active_differs_from_baseline"])
        self.assertTrue(result["checks"]["post_active_original_returns_to_baseline"])
        self.assertTrue(result["checks"]["raw_original_does_not_advance_callback"])

    def test_report_is_no_model_and_does_not_overclaim_root_cause(self):
        report = build_diagnostic_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertIn(
            "one_proven_low_level_root_cause", report["findings"]["not_supported"]
        )
        self.assertIn(
            "real_upstream_dispatch_boundary",
            report["findings"]["unresolved_candidates"],
        )
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d5c_rerun"])
        self.assertFalse(report["safety"]["d5c_conclusion_changed"])
        self.assertFalse(report["safety"]["d5d_authorized"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["rwkv_model_imported"])

    def test_wrong_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_diagnostic_report(config_path=path, project_root=ROOT)

    def test_no_real_runtime_modules_loaded(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("PSA_SELF_MODEL_D5C_REAL_MECHANISM_SMOKE", os.environ)


if __name__ == "__main__":
    unittest.main()
