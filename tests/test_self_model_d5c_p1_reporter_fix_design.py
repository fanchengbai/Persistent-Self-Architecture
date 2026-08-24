from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_p1_reporter_fix_design import (
    ACCEPTANCE_CATEGORIES,
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    P1_CORE_DIGEST,
    REQUIRED_CONFIRMATION,
    build_reporter_fix_design_report,
    inspect_frozen_reporter,
    run_synthetic_dispatch_diagnostic,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CP1ReporterFixDesignTests(unittest.TestCase):
    def test_scope_is_exactly_offline_diagnosis_and_design(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertTrue(payload["authority"]["reporter_fix_design_authorized"])
        self.assertFalse(payload["authority"]["reporter_fix_implementation_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["p1_rerun_authorized"])

    def test_scope_evidence_or_strategy_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "reporter_fix_implementation_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "p1_rerun_authorized", True),
            ("frozen_failure_evidence", "patched_routes_reached", True),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)
        changed = copy.deepcopy(payload)
        changed["strategy_review"][0]["decision"] = "sufficient_fix"
        with self.assertRaises(PermissionError):
            validate_config(changed)

    def test_source_audit_preserves_history_and_detects_adapter_transition(self):
        audit = inspect_frozen_reporter(ROOT)
        self.assertEqual(audit["historical_source_sha256"], P1_CORE_DIGEST)
        self.assertNotEqual(audit["source_sha256"], P1_CORE_DIGEST)
        self.assertEqual(audit["hasattr_values_call_count"], 0)
        self.assertEqual(audit["sha256_json_values_call_count"], 0)
        self.assertTrue(audit["explicit_adapter_present"])
        self.assertFalse(audit["torch_import_present"])

    def test_synthetic_fixture_reproduces_collision_and_design_boundaries(self):
        diagnostic = run_synthetic_dispatch_diagnostic()
        self.assertTrue(diagnostic["valid"])
        self.assertTrue(all(diagnostic["checks"].values()))
        self.assertEqual(diagnostic["reproduced_error"]["type"], "TypeError")
        self.assertFalse(diagnostic["recommended_dispatch"]["attribute_name_inference"])

    def test_future_acceptance_matrix_is_complete(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(payload["future_fake_acceptance"], list(ACCEPTANCE_CATEGORIES))
        self.assertEqual(len(payload["future_fake_acceptance"]), 9)

    def test_report_selects_design_without_implementing_fix_or_rerun(self):
        report = build_reporter_fix_design_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(
            report["decision"]["root_cause_boundary"],
            "reporter_dispatch_attribute_name_collision",
        )
        self.assertFalse(report["decision"]["runtime_patch_evaluated_by_p1"])
        self.assertFalse(report["decision"]["reporter_fix_implemented"])
        self.assertTrue(report["decision"]["current_tree_reporter_fix_detected"])
        self.assertFalse(report["decision"]["real_rerun_authorized"])
        self.assertFalse(report["safety"]["model_executed"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_reporter_fix_design_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
