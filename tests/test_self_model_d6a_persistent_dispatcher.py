from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d6a_persistent_dispatcher import (
    ACCEPTANCE_CATEGORIES,
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    D6ACouplingRequest,
    PersistentInstrumentedRuntime,
    REQUIRED_CONFIRMATION,
    SyntheticPersistentFixture,
    build_d6a_report,
    run_fake_lifecycle_acceptance,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D6APersistentDispatcherTests(unittest.TestCase):
    def test_scope_is_exactly_pure_python_d6a(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertTrue(payload["authority"]["d6a_implementation_authorized"])
        self.assertTrue(payload["authority"]["synthetic_python_fixture_authorized"])
        self.assertFalse(payload["authority"]["d6b_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["p2_authorized"])

    def test_scope_or_lifecycle_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "d6b_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "p1_rerun_authorized", True),
            ("authority", "p2_authorized", True),
            ("contract", "forward_time_model_setattr_or_delattr_allowed", True),
            ("contract", "nested_request_policy", "allow"),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_constructor_installs_once_and_direct_unscoped_forward_fails(self):
        model = SyntheticPersistentFixture()
        runtime = PersistentInstrumentedRuntime(model)
        self.assertEqual(runtime.installation_count, 3)
        self.assertTrue(runtime.bindings_are_stable())
        with self.assertRaises(PermissionError):
            model.forward([1], None)
        self.assertTrue(runtime.context_is_empty())
        with self.assertRaises(RuntimeError):
            PersistentInstrumentedRuntime(model)
        self.assertTrue(runtime.bindings_are_stable())

    def test_invalid_requests_fail_before_inner_forward(self):
        model = SyntheticPersistentFixture()
        runtime = PersistentInstrumentedRuntime(model)
        before = model._ledger.inner_forward_calls
        invalid = (
            D6ACouplingRequest(False, 0.0, lambda residual, **_: residual),
            D6ACouplingRequest(True, 1.0, None),
            D6ACouplingRequest(True, 0.5, None),
        )
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(PermissionError):
                runtime.forward([1], None, coupling=request)
        self.assertEqual(model._ledger.inner_forward_calls, before)
        self.assertTrue(runtime.context_is_empty())

    def test_all_twelve_fake_lifecycle_categories_pass(self):
        acceptance = run_fake_lifecycle_acceptance(ROOT)
        self.assertTrue(acceptance["valid"])
        self.assertEqual(list(acceptance["checks"]), list(ACCEPTANCE_CATEGORIES))
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(acceptance["counts"]["installation_count"], 3)
        self.assertGreaterEqual(acceptance["counts"]["runtime_rejections"], 2)

    def test_report_is_no_model_and_keeps_d5_line_stopped(self):
        report = build_d6a_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertTrue(report["decision"]["d6a_implemented"])
        self.assertFalse(report["decision"]["d6b_or_later_authorized"])
        self.assertFalse(report["decision"]["d5c_p1_or_p2_rerun"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["torch_imported"])

    def test_wrong_config_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d6a_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
