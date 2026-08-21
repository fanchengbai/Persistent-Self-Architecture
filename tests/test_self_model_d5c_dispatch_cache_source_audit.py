from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d5c_dispatch_cache_source_audit import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    REQUIRED_CONFIRMATION,
    _source_boundary_analysis,
    build_source_audit_report,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CDispatchCacheSourceAuditTests(unittest.TestCase):
    def test_config_freezes_failure_and_read_only_authority(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["required_owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertTrue(payload["authority"]["source_audit_authorized"])
        self.assertFalse(payload["authority"]["fix_implementation_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["d5c_rerun_authorized"])

    def test_scope_or_evidence_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("authority", "fix_implementation_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "d5c_rerun_authorized", True),
            ("frozen_evidence", "extra_callback_invocations", 0),
            ("frozen_evidence", "rwkv_jit_on", "1"),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_ast_transform_is_fresh_and_decorators_are_removed(self):
        analysis = _source_boundary_analysis(ROOT)
        self.assertTrue(analysis["fresh_ast_parse_present"])
        self.assertTrue(analysis["independent_exec_compile_present"])
        self.assertTrue(analysis["decorator_lists_cleared"])
        self.assertFalse(
            analysis["source_level_implications"][
                "ast_transform_can_mutate_loaded_original_method_objects"
            ]
        )

    def test_wrapper_transition_closes_audited_protocol_asymmetry(self):
        analysis = _source_boundary_analysis(ROOT)
        self.assertEqual(analysis["wrapper_setattr_count"], 2)
        self.assertTrue(analysis["wrapper_methodtype_present"])
        self.assertEqual(analysis["wrapper_direct_dict_pop_count"], 0)
        self.assertEqual(analysis["wrapper_delattr_count"], 0)
        self.assertEqual(analysis["wrapper_getattr_count"], 0)
        self.assertEqual(analysis["restore_delattr_count"], 1)
        self.assertGreaterEqual(analysis["verify_getattr_count"], 2)
        self.assertTrue(analysis["restore_helper_called"])
        self.assertTrue(analysis["verify_helper_called"])
        self.assertTrue(
            analysis["source_level_implications"][
                "installation_and_cleanup_use_symmetric_object_protocol"
            ]
        )

    def test_existing_fake_does_not_cover_real_decorator_boundary(self):
        analysis = _source_boundary_analysis(ROOT)
        self.assertEqual(
            analysis["fake_method_decorators"],
            {"forward_one": [], "forward_seq": []},
        )
        self.assertFalse(analysis["myfunction_definition_in_audited_sources"])
        self.assertFalse(
            analysis["source_level_implications"][
                "existing_fake_covers_real_decorator_boundary"
            ]
        )

    def test_report_narrows_boundary_without_claiming_root_cause_or_fix(self):
        report = build_source_audit_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertIn(
            "one_proven_cache_or_dispatch_root_cause",
            report["findings"]["not_supported"],
        )
        self.assertIn(
            "current_wrapper_uses_protocol_restore_and_resolution_verification",
            report["findings"]["confirmed"],
        )
        self.assertFalse(report["safety"]["fix_implemented"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d5c_rerun"])
        self.assertFalse(report["safety"]["d5c_conclusion_changed"])

    def test_wrong_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_source_audit_report(config_path=path, project_root=ROOT)

    def test_no_model_modules_loaded(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
