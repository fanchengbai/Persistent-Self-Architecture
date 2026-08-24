from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from psa.self_model import d5c_mechanism_runtime as runtime_module
from psa.self_model import d5c_p1_real_entry as entry
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    OfflineTorch,
    _namespace,
    _state,
)
from psa.self_model.d5c_mechanism_runtime import (
    D5CSyntheticProbe,
    RWKV7D5CActiveRuntime,
)
from psa.self_model.d5c_p1_engineering_validation import (
    ROUTE_ORDER,
    execute_d5c_p1_engineering_core,
)
from psa.self_model.d5c_p1_reporter_adapter_fix import ExactOfflineTensorAdapter
from psa.self_model.d5c_p1_real_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    IMPLEMENTATION_CONFIRMATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    build_p1_authorization,
    build_p1_entry_static_report,
    read_spec,
    run_p1_real_engineering_validation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CP1RealEntryTests(unittest.TestCase):
    def test_design_is_fixed_to_new_twelve_call_engineering_gate(self):
        spec = read_spec(CONFIG)
        self.assertEqual(spec["implementation_confirmation_text"], IMPLEMENTATION_CONFIRMATION_TEXT)
        self.assertEqual(spec["route_order"], list(ROUTE_ORDER))
        self.assertEqual(spec["counts"]["model_forward_calls_total"], 12)
        self.assertEqual(spec["counts"]["wrapped_forward_calls_total"], 8)
        self.assertFalse(spec["execution_authorized_at_implementation"])
        self.assertFalse(spec["historical_d5c_authorization_reusable"])
        self.assertFalse(spec["historical_d5c_claim_reusable"])
        self.assertFalse(spec["historical_d5c_rerun_authorized"])

    def test_scope_or_history_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("execution_authorized_at_implementation", True),
            ("historical_d5c_claim_reusable", True),
            ("d5d_authorized", True),
            ("automatic_rerun_authorized", True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            for field, value in changes:
                changed = copy.deepcopy(payload)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(PermissionError):
                    read_spec(path)
            changed = copy.deepcopy(payload)
            changed["counts"]["model_forward_calls_total"] = 42
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(PermissionError):
                read_spec(path)

    def test_offline_32_layer_fixture_passes_real_p1_core(self):
        namespace, fixture_class = _namespace()
        fixture = fixture_class()
        source_bytes = DIAGNOSTIC_SOURCE.encode("utf-8")
        digest = hashlib.sha256(source_bytes).hexdigest()
        with patch.object(runtime_module, "EXPECTED_RWKV_MODEL_SOURCE_SHA256", digest):
            runtime = RWKV7D5CActiveRuntime(
                base_model=fixture,
                upstream_source_bytes=source_bytes,
                upstream_globals=namespace,
                upstream_package_version="0.8.32",
                upstream_de_version=None,
                execution_claim_sha256="a" * 64,
                machine_authorization_sha256="b" * 64,
            )
        probe = D5CSyntheticProbe(
            torch=OfflineTorch(), execution_claim_sha256="a" * 64,
            machine_authorization_sha256="b" * 64,
        )
        report = execute_d5c_p1_engineering_core(
            base_model=fixture,
            active_runtime=runtime,
            probe=probe,
            torch=OfflineTorch(),
            state_factory=_state,
            offline_adapter=ExactOfflineTensorAdapter(),
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["counts"]["model_forward_calls"], 12)
        self.assertEqual(report["counts"]["wrapped_forward_calls"], 8)
        self.assertEqual(report["counts"]["callback_invocations"], 64)
        self.assertEqual(report["counts"]["probe_applications"], 2)
        self.assertFalse(report["interpretation"]["historical_d5c_result_changed"])

    def test_static_entry_is_complete_and_no_model(self):
        report = build_p1_entry_static_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["machine_authorization_created"])
        self.assertFalse(report["safety"]["execution_claim_created"])
        self.assertFalse(report["safety"]["historical_d5c_rerun"])

    def test_future_authorization_text_is_separate_and_exact(self):
        git = {"commit": "a" * 40, "branch": "main", "status_porcelain": ""}
        authorization = build_p1_authorization(
            config_path=CONFIG,
            project_root=ROOT,
            authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
            git_metadata=git,
        )
        self.assertTrue(authorization["model_execution_authorized"])
        self.assertTrue(authorization["engineering_validation_only"])
        self.assertFalse(authorization["historical_d5c_rerun_authorized"])
        self.assertFalse(authorization["historical_d5c_conclusion_change_authorized"])
        with self.assertRaises(PermissionError):
            build_p1_authorization(
                config_path=CONFIG, project_root=ROOT,
                authorization_text=IMPLEMENTATION_CONFIRMATION_TEXT,
                git_metadata=git,
            )

    def test_missing_lock_fails_before_git_source_claim_or_model(self):
        environment = dict(os.environ)
        environment.pop(entry.EXECUTION_LOCK_ENV, None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            entry, "_git_metadata"
        ) as git, patch.object(entry, "_installed_source") as installed, patch.object(
            entry, "_create_claim"
        ) as claim, patch.object(entry.RWKV7Adapter, "load") as load:
            with self.assertRaises(PermissionError):
                run_p1_real_engineering_validation(
                    config_path=CONFIG,
                    authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
                    project_root=ROOT,
                    output_dir=ROOT / OUTPUT_RELATIVE_DIR,
                )
        git.assert_not_called()
        installed.assert_not_called()
        claim.assert_not_called()
        load.assert_not_called()

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_p1_entry_static_report(config_path=path, project_root=ROOT)
        self.assertFalse((ROOT / AUTHORIZATION_RELATIVE_PATH).exists())
        self.assertFalse((ROOT / OUTPUT_RELATIVE_DIR / "execution_claim.json").exists())
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
