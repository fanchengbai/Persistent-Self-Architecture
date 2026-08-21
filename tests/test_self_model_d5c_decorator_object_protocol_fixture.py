from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_decorator_object_protocol_fixture import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    REQUIRED_CONFIRMATION,
    build_fixture_report,
    run_fixture_case,
    run_fixture_matrix,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CDecoratorObjectProtocolFixtureTests(unittest.TestCase):
    def test_config_freezes_scope_and_failed_prerequisites(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(payload["frozen_prerequisites"]["d5c_status"], "d5c_mechanism_smoke_failed")
        self.assertFalse(payload["authority"]["real_runtime_modification_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["d5c_rerun_authorized"])

    def test_authority_or_evidence_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "fix_implementation_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "d5c_rerun_authorized", True),
            ("frozen_prerequisites", "d5c_status", "passed"),
            ("matrix", "total_case_count", 15),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_standard_decorator_matrix_always_restores(self):
        cases = [case for case in run_fixture_matrix() if not case["side_dispatch"]]
        self.assertEqual(len(cases), 12)
        self.assertTrue(all(case["valid"] for case in cases))
        self.assertTrue(all(case["observed"]["restored"] for case in cases))
        self.assertFalse(any(case["observed"]["contaminated"] for case in cases))

    def test_side_cache_direct_pop_reproduces_contamination_shape(self):
        for path in ("forward_one", "forward_seq"):
            case = run_fixture_case(
                decorator_kind="non_caching_descriptor",
                cleanup_mode="direct_instance_dict_pop",
                execution_path=path,
                side_dispatch=True,
            )
            self.assertTrue(case["valid"])
            self.assertTrue(case["observed"]["contaminated"])
            self.assertEqual(case["observed"]["instance_keys_after_cleanup"], [])
            self.assertEqual(len(case["observed"]["side_keys_after_cleanup"]), 3)
            self.assertEqual(case["observed"]["resolved_origin_after_cleanup"], "active")
            self.assertEqual(case["observed"]["callback_count_after_post_cleanup"], 2)

    def test_side_cache_delattr_restores_in_synthetic_protocol(self):
        for path in ("forward_one", "forward_seq"):
            case = run_fixture_case(
                decorator_kind="non_caching_descriptor",
                cleanup_mode="delattr",
                execution_path=path,
                side_dispatch=True,
            )
            self.assertTrue(case["valid"])
            self.assertTrue(case["observed"]["restored"])
            self.assertEqual(case["observed"]["side_keys_after_cleanup"], [])
            self.assertEqual(case["observed"]["resolved_origin_after_cleanup"], "original")
            delattr_events = [
                event for event in case["observed"]["protocol_events"]
                if event["operation"] == "delattr"
            ]
            self.assertEqual(len(delattr_events), 3)

    def test_invalid_matrix_parameters_fail_closed(self):
        with self.assertRaises(ValueError):
            run_fixture_case(
                decorator_kind="unknown", cleanup_mode="delattr",
                execution_path="forward_one", side_dispatch=False,
            )
        with self.assertRaises(ValueError):
            run_fixture_case(
                decorator_kind="plain", cleanup_mode="unknown",
                execution_path="forward_one", side_dispatch=False,
            )
        with self.assertRaises(ValueError):
            run_fixture_case(
                decorator_kind="plain", cleanup_mode="delattr",
                execution_path="forward_one", side_dispatch=True,
            )

    def test_report_marks_sufficient_mechanism_but_not_real_root_cause(self):
        report = build_fixture_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["matrix_summary"]["case_count"], 16)
        self.assertEqual(report["matrix_summary"]["standard_restored"], 12)
        self.assertEqual(report["matrix_summary"]["side_cache_direct_pop_contaminated"], 2)
        self.assertEqual(report["matrix_summary"]["side_cache_delattr_restored"], 2)
        self.assertIn(
            "direct_dict_pop_is_the_real_d5c_root_cause",
            report["findings"]["not_supported"],
        )
        self.assertFalse(report["safety"]["fix_implemented"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d5c_rerun"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_fixture_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
