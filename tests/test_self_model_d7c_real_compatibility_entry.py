from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from psa.self_model import d7c_real_compatibility_entry as entry
from psa.self_model.d7c_real_compatibility_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    IMPLEMENTATION_CONFIRMATION_TEXT,
    NEXT_GATE,
    OUTPUT_RELATIVE_DIR,
    build_d7c_authorization,
    build_d7c_entry_static_report,
    read_spec,
    run_d7c_real_compatibility,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D7CRealCompatibilityEntryTests(unittest.TestCase):
    def test_design_freezes_eight_cells_and_eighteen_calls(self):
        spec = read_spec(CONFIG)
        self.assertEqual(spec["implementation_confirmation_text"], IMPLEMENTATION_CONFIRMATION_TEXT)
        self.assertEqual(len(spec["compatibility_cells"]), 8)
        self.assertEqual(spec["counts"]["equivalence_forward_calls"], 16)
        self.assertEqual(spec["counts"]["synthetic_active_forward_calls"], 2)
        self.assertEqual(spec["counts"]["model_forward_calls_total"], 18)
        self.assertFalse(spec["payload_separation"]["heldout_payload_accessed"])
        self.assertFalse(spec["execution_authorized_at_implementation"])

    def test_scope_or_later_authority_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("installed_source_probe_authorized_at_implementation", True),
            ("execution_authorized_at_implementation", True),
            ("d7d_authorized", True),
            ("projection_implementation_authorized", True),
            ("d6d_rerun_authorized", True),
            ("automatic_rerun_authorized", True),
        )
        for field, value in mutations:
            changed = copy.deepcopy(payload)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_static_entry_is_complete_and_no_model(self):
        report = build_d7c_entry_static_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertFalse(report["safety"]["installed_source_probed"])
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["machine_authorization_created"])
        self.assertFalse(report["safety"]["execution_claim_created"])

    def test_future_authorization_is_exact_and_separate(self):
        git = {"commit": "a" * 40, "branch": "main", "status_porcelain": ""}
        authorization = build_d7c_authorization(
            config_path=CONFIG,
            project_root=ROOT,
            authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
            git_metadata=git,
        )
        self.assertTrue(authorization["model_execution_authorized"])
        self.assertTrue(authorization["installed_source_probe_authorized"])
        self.assertEqual(authorization["model_forward_calls"], 18)
        self.assertFalse(authorization["heldout_payload_accessed"])
        self.assertFalse(authorization["d7c_rerun_authorized"])
        self.assertFalse(authorization["d7d_authorized"])
        with self.assertRaises(PermissionError):
            build_d7c_authorization(
                config_path=CONFIG,
                project_root=ROOT,
                authorization_text=IMPLEMENTATION_CONFIRMATION_TEXT,
                git_metadata=git,
            )

    def test_existing_machine_authorization_validates_bound_pre_authorization_report(self):
        git = {"commit": "a" * 40, "branch": "main", "status_porcelain": ""}
        observed = {
            "machine_authorization_absent": False,
            "execution_claim_absent": True,
            "output_report_absent": True,
            "failure_report_absent": True,
        }
        with patch.object(entry, "_execution_artifacts_absent", return_value=observed):
            with self.assertRaises(RuntimeError):
                build_d7c_authorization(
                    config_path=CONFIG,
                    project_root=ROOT,
                    authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
                    git_metadata=git,
                )
            authorization = build_d7c_authorization(
                config_path=CONFIG,
                project_root=ROOT,
                authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
                git_metadata=git,
                verify_execution_artifacts_absent=False,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "authorization.json"
                path.write_text(json.dumps(authorization), encoding="utf-8")
                validated = entry.validate_d7c_authorization(
                    authorization_path=path,
                    config_path=CONFIG,
                    project_root=ROOT,
                    git=git,
                )
        self.assertEqual(
            validated["entry_static_report_sha256"],
            authorization["entry_static_report_sha256"],
        )

    def test_missing_lock_fails_before_git_source_claim_or_model(self):
        environment = dict(os.environ)
        environment.pop(entry.EXECUTION_LOCK_ENV, None)
        environment.pop("RWKV_DE_VERSION", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            entry, "_git_metadata"
        ) as git, patch.object(entry, "_probe_installed_source") as installed, patch.object(
            entry, "_create_claim"
        ) as claim, patch.object(entry, "_runtime_dependencies") as dependencies:
            with self.assertRaises(PermissionError):
                run_d7c_real_compatibility(
                    config_path=CONFIG,
                    authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
                    project_root=ROOT,
                    output_dir=ROOT / OUTPUT_RELATIVE_DIR,
                )
        git.assert_not_called()
        installed.assert_not_called()
        claim.assert_not_called()
        dependencies.assert_not_called()

    def test_wrong_config_path_creates_no_authority_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d7c_entry_static_report(config_path=path, project_root=ROOT)
        self.assertFalse((ROOT / AUTHORIZATION_RELATIVE_PATH).exists())
        self.assertFalse((ROOT / OUTPUT_RELATIVE_DIR / "execution_claim.json").exists())

    def test_no_model_modules_loaded_by_static_verification(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
