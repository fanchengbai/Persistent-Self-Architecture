from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_failure_lifecycle_diagnostic import OfflineTensor
from psa.self_model.d5c_p1_engineering_validation import _tensor_payload
from psa.self_model.d5c_p1_reporter_adapter_fix import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    ExactOfflineTensorAdapter,
    REQUIRED_CONFIRMATION,
    RealLikeTensor,
    _SyntheticTorch,
    build_reporter_adapter_fix_report,
    run_nine_category_acceptance,
    validate_config,
)
from psa.self_model.d5c_p1_reporter_fix_design import ACCEPTANCE_CATEGORIES


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CP1ReporterAdapterFixTests(unittest.TestCase):
    def test_scope_is_exactly_fake_first_adapter_fix(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertTrue(payload["authority"]["reporter_fix_implementation_authorized"])
        self.assertTrue(payload["authority"]["synthetic_fixture_execution_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["p1_rerun_authorized"])

    def test_scope_or_contract_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "model_execution_authorized", True),
            ("authority", "p1_rerun_authorized", True),
            ("authority", "d5d_authorized", True),
            ("dispatch_contract", "unknown_object_policy", "guess"),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_explicit_adapter_accepts_only_exact_offline_tensor(self):
        adapter = ExactOfflineTensorAdapter()
        value = OfflineTensor((1.0, 2.0))
        self.assertTrue(adapter.accepts(value))
        self.assertFalse(adapter.accepts(RealLikeTensor()))
        payload = _tensor_payload(
            value, _SyntheticTorch(), offline_adapter=adapter
        )
        self.assertEqual(payload["kind"], "offline_tensor")

    def test_default_real_like_serializer_never_reads_values(self):
        value = RealLikeTensor()
        payload = _tensor_payload(value, _SyntheticTorch())
        self.assertEqual(payload["kind"], "tensor")
        self.assertEqual(value.values_member_reads, 0)

    def test_all_nine_frozen_categories_and_full_core_pass(self):
        acceptance = run_nine_category_acceptance(ROOT)
        self.assertTrue(acceptance["valid"])
        self.assertEqual(list(acceptance["checks"]), list(ACCEPTANCE_CATEGORIES))
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertTrue(acceptance["full_core_fixture"]["valid"])
        self.assertEqual(
            acceptance["full_core_fixture"]["counts"]["model_forward_calls"], 12
        )

    def test_report_is_no_model_no_rerun_and_real_runner_has_no_adapter(self):
        report = build_reporter_adapter_fix_report(
            config_path=CONFIG, project_root=ROOT
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertTrue(report["decision"]["reporter_fix_implemented"])
        self.assertIsNone(report["decision"]["real_runner_offline_adapter"])
        self.assertFalse(
            report["acceptance"]["reporter_ast"]["real_runner_passes_offline_adapter"]
        )
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["p1_rerun"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_reporter_adapter_fix_report(
                    config_path=path, project_root=ROOT
                )
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
