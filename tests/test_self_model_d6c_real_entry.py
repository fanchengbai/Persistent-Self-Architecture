from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from psa.self_model import d6c_persistent_mechanism as runtime_module
from psa.self_model import d6c_real_entry as entry
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    OfflineTorch,
    _namespace,
    _state,
)
from psa.self_model.d5c_mechanism_runtime import D5CSyntheticProbe
from psa.self_model.d6c_persistent_mechanism import (
    ACTIVE_CALLBACK_CALLS_TOTAL,
    ACTIVE_PROBE_APPLICATIONS_TOTAL,
    D6COfflineReporterAdapter,
    LATIN_ROUNDS,
    MODEL_FORWARD_CALLS_TOTAL,
    ROUTES,
    RWKV7D6CPersistentRuntime,
    execute_d6c_mechanism_core,
)
from psa.self_model.d6c_real_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    IMPLEMENTATION_CONFIRMATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    build_d6c_authorization,
    build_d6c_entry_static_report,
    read_spec,
    run_d6c_real_persistent_mechanism,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D6CRealEntryTests(unittest.TestCase):
    def test_design_freezes_two_shapes_and_twenty_six_calls(self):
        spec = read_spec(CONFIG)
        self.assertEqual(spec["implementation_confirmation_text"], IMPLEMENTATION_CONFIRMATION_TEXT)
        self.assertEqual(spec["routes"], list(ROUTES))
        self.assertEqual(spec["latin_rounds"], [list(value) for value in LATIN_ROUNDS])
        self.assertEqual(spec["counts"]["model_forward_calls_total"], 26)
        self.assertEqual(spec["counts"]["active_callback_calls_total"], 256)
        self.assertEqual(spec["counts"]["active_probe_applications_total"], 8)
        self.assertFalse(spec["execution_authorized_at_implementation"])
        self.assertFalse(spec["installed_source_probe_authorized_at_implementation"])
        self.assertFalse(spec["raw_original_route_authorized"])

    def test_scope_schedule_or_later_authority_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("execution_authorized_at_implementation", True),
            ("installed_source_probe_authorized_at_implementation", True),
            ("raw_original_route_authorized", True),
            ("d6d_authorized", True),
            ("automatic_rerun_authorized", True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            for field, value in mutations:
                changed = copy.deepcopy(payload)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(PermissionError):
                    read_spec(path)
            changed = copy.deepcopy(payload)
            changed["counts"]["model_forward_calls_total"] = 27
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(PermissionError):
                read_spec(path)

    def test_persistent_fake_core_passes_frozen_protocol(self):
        namespace, fixture_type = _namespace()
        fixture = fixture_type()
        source_bytes = DIAGNOSTIC_SOURCE.encode("utf-8")
        digest = hashlib.sha256(source_bytes).hexdigest()
        with patch.object(runtime_module, "EXPECTED_RWKV_MODEL_SOURCE_SHA256", digest):
            runtime = RWKV7D6CPersistentRuntime(
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
        report = execute_d6c_mechanism_core(
            runtime=runtime, probe=probe, torch=OfflineTorch(), state_factory=_state,
            offline_adapter=D6COfflineReporterAdapter(),
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["counts"]["model_forward_calls"], MODEL_FORWARD_CALLS_TOTAL)
        self.assertEqual(report["counts"]["callback_invocations"], ACTIVE_CALLBACK_CALLS_TOTAL)
        self.assertEqual(report["counts"]["probe_applications"], ACTIVE_PROBE_APPLICATIONS_TOTAL)
        self.assertEqual(runtime.installation_count, 3)
        self.assertTrue(runtime.bindings_are_stable())
        self.assertTrue(runtime.context_is_empty())

    def test_second_persistent_install_is_rejected_without_binding_change(self):
        namespace, fixture_type = _namespace()
        fixture = fixture_type()
        source_bytes = DIAGNOSTIC_SOURCE.encode("utf-8")
        digest = hashlib.sha256(source_bytes).hexdigest()
        with patch.object(runtime_module, "EXPECTED_RWKV_MODEL_SOURCE_SHA256", digest):
            runtime = RWKV7D6CPersistentRuntime(
                base_model=fixture, upstream_source_bytes=source_bytes,
                upstream_globals=namespace, upstream_package_version="0.8.32",
                upstream_de_version=None, execution_claim_sha256="a" * 64,
                machine_authorization_sha256="b" * 64,
            )
            with self.assertRaises(RuntimeError):
                RWKV7D6CPersistentRuntime(
                    base_model=fixture, upstream_source_bytes=source_bytes,
                    upstream_globals=namespace, upstream_package_version="0.8.32",
                    upstream_de_version=None, execution_claim_sha256="a" * 64,
                    machine_authorization_sha256="b" * 64,
                )
        self.assertTrue(runtime.bindings_are_stable())

    def test_static_entry_is_complete_and_no_model(self):
        report = build_d6c_entry_static_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["installed_source_probed"])
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["machine_authorization_created"])
        self.assertFalse(report["safety"]["execution_claim_created"])

    def test_future_authorization_is_exact_and_separate(self):
        git = {"commit": "a" * 40, "branch": "main", "status_porcelain": ""}
        authorization = build_d6c_authorization(
            config_path=CONFIG, project_root=ROOT,
            authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
            git_metadata=git,
        )
        self.assertTrue(authorization["installed_source_probe_authorized"])
        self.assertTrue(authorization["model_execution_authorized"])
        self.assertFalse(authorization["raw_original_route_authorized"])
        self.assertFalse(authorization["d6c_rerun_authorized"])
        with self.assertRaises(PermissionError):
            build_d6c_authorization(
                config_path=CONFIG, project_root=ROOT,
                authorization_text=IMPLEMENTATION_CONFIRMATION_TEXT,
                git_metadata=git,
            )

    def test_missing_lock_fails_before_git_source_claim_or_model(self):
        environment = dict(os.environ)
        environment.pop(entry.EXECUTION_LOCK_ENV, None)
        environment.pop("RWKV_DE_VERSION", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            entry, "_git_metadata"
        ) as git, patch.object(entry, "_installed_source") as installed, patch.object(
            entry, "_create_claim"
        ) as claim, patch.object(entry.RWKV7Adapter, "load") as load:
            with self.assertRaises(PermissionError):
                run_d6c_real_persistent_mechanism(
                    config_path=CONFIG,
                    authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
                    project_root=ROOT,
                    output_dir=ROOT / OUTPUT_RELATIVE_DIR,
                )
        git.assert_not_called()
        installed.assert_not_called()
        claim.assert_not_called()
        load.assert_not_called()

    def test_cli_missing_lock_creates_no_authorization_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(os.environ)
            environment.pop(entry.EXECUTION_LOCK_ENV, None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_self_model_v0_1_coupling_d6c_real_persistent_mechanism.py"),
                    "--project-root", str(root),
                ],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("single-use lock is absent", completed.stderr)
            self.assertFalse((root / AUTHORIZATION_RELATIVE_PATH).exists())
            self.assertFalse((root / OUTPUT_RELATIVE_DIR).exists())

    def test_wrong_config_path_creates_no_authority_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d6c_entry_static_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
