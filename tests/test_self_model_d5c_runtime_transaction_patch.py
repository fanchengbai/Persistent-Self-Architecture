from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_runtime_transaction_patch import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    build_runtime_transaction_patch_report,
    run_runtime_transaction_acceptance,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CRuntimeTransactionPatchTests(unittest.TestCase):
    def test_scope_is_exactly_patch_and_local_no_model_validation(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertTrue(payload["authority"]["real_runtime_transaction_patch_authorized"])
        self.assertTrue(payload["authority"]["remote_execution_delegated_to_owner"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["d5c_rerun_authorized"])

    def test_authority_or_history_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d5d_authorized", True),
            ("frozen_prerequisites", "d5c_status", "passed"),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_actual_runtime_passes_transaction_acceptance(self):
        acceptance = run_runtime_transaction_acceptance()
        self.assertTrue(acceptance["valid"])
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertGreaterEqual(len(acceptance["checks"]), 11)

    def test_report_claims_patch_but_not_real_model_validation(self):
        report = build_runtime_transaction_patch_report(
            config_path=CONFIG, project_root=ROOT
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertTrue(report["decision"]["real_runtime_patch_implemented"])
        self.assertTrue(report["decision"]["local_no_model_acceptance_passed"])
        self.assertFalse(report["decision"]["real_2_9b_validation_run"])
        self.assertFalse(report["decision"]["d5c_failure_conclusion_changed"])
        self.assertFalse(report["safety"]["model_executed"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_runtime_transaction_patch_report(
                    config_path=path, project_root=ROOT
                )
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
