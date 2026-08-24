from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    _namespace,
    _state,
)
from psa.self_model.d6b_persistent_ast import (
    ACCEPTANCE_CATEGORIES,
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    D6BASTRequest,
    PersistentASTRuntime,
    REQUIRED_CONFIRMATION,
    build_d6b_report,
    run_static_integration_acceptance,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D6BPersistentASTTests(unittest.TestCase):
    def test_scope_is_exactly_static_no_model_d6b(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertTrue(payload["authority"]["d6b_implementation_authorized"])
        self.assertTrue(payload["authority"]["locked_ast_reuse_authorized"])
        self.assertFalse(payload["authority"]["installed_source_probe_authorized"])
        self.assertFalse(payload["authority"]["d6c_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])

    def test_scope_or_persistent_contract_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "installed_source_probe_authorized", True),
            ("authority", "d6c_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "p2_authorized", True),
            ("integration_contract", "runtime_model_attribute_mutation_allowed", True),
            ("integration_contract", "installation_count", 6),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_locked_ast_installs_once_and_unscoped_forward_fails(self):
        namespace, fixture_type = _namespace()
        fixture = fixture_type()
        runtime = PersistentASTRuntime(
            base_model=fixture,
            exact_fixture_type=fixture_type,
            upstream_source=DIAGNOSTIC_SOURCE,
            upstream_globals=namespace,
        )
        self.assertEqual(runtime.installation_count, 3)
        self.assertEqual(runtime.injection_counts, {"forward_one": 1, "forward_seq": 1})
        self.assertTrue(runtime.bindings_are_stable())
        with self.assertRaises(PermissionError):
            fixture.forward([3], _state(), False)
        self.assertTrue(runtime.context_is_empty())
        with self.assertRaises(RuntimeError):
            PersistentASTRuntime(
                base_model=fixture,
                exact_fixture_type=fixture_type,
                upstream_source=DIAGNOSTIC_SOURCE,
                upstream_globals=namespace,
            )
        self.assertTrue(runtime.bindings_are_stable())

    def test_invalid_request_fails_before_fixture_forward(self):
        namespace, fixture_type = _namespace()
        fixture = fixture_type()
        runtime = PersistentASTRuntime(
            base_model=fixture,
            exact_fixture_type=fixture_type,
            upstream_source=DIAGNOSTIC_SOURCE,
            upstream_globals=namespace,
        )
        invalid = D6BASTRequest("off", True, 0.0, None)
        with self.assertRaises(PermissionError):
            runtime.forward([3], _state(), full_output=False, coupling=invalid)
        self.assertEqual(runtime.execution_count, 0)
        self.assertTrue(runtime.context_is_empty())

    def test_all_fourteen_static_integration_categories_pass(self):
        acceptance = run_static_integration_acceptance(ROOT)
        self.assertTrue(acceptance["valid"])
        self.assertEqual(list(acceptance["checks"]), list(ACCEPTANCE_CATEGORIES))
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(acceptance["counts"]["layers"], 32)
        self.assertEqual(acceptance["counts"]["hidden_dimension"], 2560)
        self.assertEqual(acceptance["counts"]["state_components"], 96)
        self.assertFalse(acceptance["installed_source_probed"])

    def test_report_keeps_installed_source_model_and_later_gates_closed(self):
        report = build_d6b_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertTrue(report["decision"]["d6b_implemented"])
        self.assertFalse(report["decision"]["d6c_or_later_authorized"])
        self.assertFalse(report["decision"]["installed_source_probed"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["rwkv_model_imported"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d6b_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
