from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_offline_fix_design import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    build_fix_design_report,
    inspect_current_wrapper,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5COfflineFixDesignTests(unittest.TestCase):
    def test_confirmation_is_bound_to_design_only_scope(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["owner_confirmation_text"], "确认")
        self.assertIn("pure-offline fix design", payload["confirmation_context"])
        self.assertFalse(payload["authority"]["fake_fix_implementation_authorized"])
        self.assertFalse(payload["authority"]["real_runtime_modification_authorized"])
        self.assertFalse(payload["authority"]["d5c_rerun_authorized"])

    def test_scope_evidence_or_strategy_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "fake_fix_implementation_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("frozen_prerequisites", "d5c_status", "passed"),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)
        changed = copy.deepcopy(payload)
        changed["strategy_review"][1]["decision"] = "verified_real_fix"
        with self.assertRaises(PermissionError):
            validate_config(changed)

    def test_current_runtime_is_observed_but_unchanged(self):
        observation = inspect_current_wrapper(ROOT)
        self.assertTrue(observation["runtime_unchanged_from_frozen_failure"])
        self.assertEqual(observation["setattr_call_count"], 2)
        self.assertEqual(observation["dict_pop_call_count"], 1)
        self.assertEqual(observation["delattr_call_count"], 0)
        self.assertFalse(observation["has_snapshot_helper"])
        self.assertFalse(observation["has_post_cleanup_identity_verification"])

    def test_design_does_not_treat_delattr_as_sufficient_fix(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        strategies = {item["strategy"]: item for item in payload["strategy_review"]}
        self.assertEqual(
            strategies["delattr_only"]["decision"],
            "insufficient_as_standalone_fix",
        )
        self.assertEqual(
            strategies["transactional_snapshot_restore_verify"]["decision"],
            "recommend_for_future_fake_first_implementation",
        )

    def test_design_requires_output_commit_after_cleanup_verification(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        transaction = payload["recommended_transaction"]
        self.assertIn("only after restoration verification", transaction["commit_rule"])
        self.assertTrue(transaction["failure_rule"].startswith("discard the forward output"))
        self.assertEqual(len(transaction["capture"]), 4)
        self.assertEqual(len(transaction["restore"]), 4)
        self.assertEqual(len(transaction["verify"]), 4)

    def test_fake_acceptance_requires_sticky_cache_and_exception_paths(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        matrix = payload["fake_acceptance_matrix"]
        self.assertEqual(len(matrix), 10)
        self.assertTrue(any(item.startswith("noncooperative sticky") for item in matrix))
        self.assertTrue(any(item.startswith("post-cleanup identity mismatch") for item in matrix))
        self.assertEqual(matrix[-1], "no extra real-model forward is used for cleanup verification")

    def test_report_is_design_only_and_does_not_claim_real_fix(self):
        report = build_fix_design_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertFalse(report["decision"]["real_fix_selected"])
        self.assertFalse(report["decision"]["real_runtime_change_authorized"])
        self.assertFalse(report["safety"]["fake_fix_implemented"])
        self.assertFalse(report["safety"]["real_runtime_modified"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d5c_rerun"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_fix_design_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
