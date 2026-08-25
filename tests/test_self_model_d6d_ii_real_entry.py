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

from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    _namespace,
    _state,
)
from psa.self_model.d6d_ii_joint_runtime import (
    D6DIIWrapperOwnedRuntime,
    request_for_pilot_condition,
)
from psa.self_model.d6d_ii_manifests import (
    PILOT_FORWARD_CALLS,
    PILOT_MANIFEST_RELATIVE_PATH,
    TOTAL_FORWARD_CALLS,
    TRAINING_FORWARD_CALLS,
    TRAINING_MANIFEST_RELATIVE_PATH,
    build_manifest_report,
    build_pilot_call_plan,
    expand_pilot_fixtures,
    expand_training_records,
    load_pilot_manifest,
    load_training_manifest,
)
from psa.self_model.d6d_ii_real_entry import (
    AUTHORIZATION_FIELDS,
    AUTHORIZATION_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    IMPLEMENTATION_CONFIRMATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    build_d6d_ii_static_report,
    probe_installed_source_compatibility,
    read_spec,
)
from psa.self_model.rwkv7_instrumented_off_runtime import compile_instrumented_methods


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH
TRAINING = ROOT / TRAINING_MANIFEST_RELATIVE_PATH
PILOT = ROOT / PILOT_MANIFEST_RELATIVE_PATH


class D6DIIRealEntryTests(unittest.TestCase):
    def test_exact_confirmation_and_future_authorization_are_frozen(self):
        spec = read_spec(CONFIG, ROOT)
        self.assertEqual(
            spec["implementation_confirmation_text"],
            IMPLEMENTATION_CONFIRMATION_TEXT,
        )
        self.assertEqual(
            spec["future_execution_authorization_text"],
            FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        )
        self.assertFalse(spec["authority"]["model_execution_authorized"])
        self.assertFalse(spec["authority"]["real_projection_construction_authorized"])

    def test_training_manifest_expands_exact_four_by_four_grid(self):
        manifest = load_training_manifest(TRAINING)
        records = expand_training_records(manifest)
        self.assertEqual(len(records), TRAINING_FORWARD_CALLS)
        self.assertEqual(len({item["record_id"] for item in records}), 16)
        self.assertTrue(all(item["pilot_eligible"] is False for item in records))

    def test_pilot_manifest_expands_twelve_fixtures_and_144_calls(self):
        manifest = load_pilot_manifest(PILOT)
        fixtures = expand_pilot_fixtures(manifest)
        calls = build_pilot_call_plan(manifest)
        self.assertEqual(len(fixtures), 12)
        self.assertEqual(len(calls), PILOT_FORWARD_CALLS)
        self.assertEqual(sum(call["phase"] == "off_precondition_unscored" for call in calls), 12)
        self.assertEqual(sum(call["phase"] == "scored" for call in calls), 132)

    def test_manifest_report_binds_distinct_training_and_pilot_payloads(self):
        report = build_manifest_report(training_path=TRAINING, pilot_path=PILOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertNotEqual(
            report["training_manifest_sha256"], report["pilot_manifest_sha256"]
        )
        self.assertEqual(report["counts"]["total_forward_calls"], TOTAL_FORWARD_CALLS)
        self.assertTrue(all(report["training_checks"].values()))
        self.assertTrue(all(report["pilot_checks"].values()))
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["real_projection_constructed"])

    def test_manifest_scope_expansion_fails_closed(self):
        training = load_training_manifest(TRAINING)
        changed = copy.deepcopy(training)
        changed["capture_contract"]["model_forward_calls"] = 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_manifest_report(training_path=path, pilot_path=PILOT)

    def test_new_wrapper_keeps_base_dictionary_unchanged_on_off_forward(self):
        namespace, fixture_type = _namespace()
        fixture = fixture_type()
        before = dict(fixture.__dict__)
        methods, counts = compile_instrumented_methods(
            upstream_source=DIAGNOSTIC_SOURCE,
            upstream_globals=namespace,
            rwkv_de_version=None,
        )
        runtime = D6DIIWrapperOwnedRuntime(
            base_model=fixture,
            compiled_methods=methods,
            injection_counts=counts,
        )
        output = runtime.forward(
            [7], _state(), full_output=False,
            coupling=request_for_pilot_condition("wrapper_off"),
        )
        self.assertIsNotNone(output)
        self.assertEqual(before.keys(), fixture.__dict__.keys())
        self.assertTrue(all(before[key] is fixture.__dict__[key] for key in before))
        self.assertTrue(runtime.base_dictionary_is_stable())
        self.assertTrue(runtime.owned_bindings_are_stable())

    def test_static_report_is_valid_with_remote_installed_probe_pending(self):
        report = build_d6d_ii_static_report(
            config_path=CONFIG,
            project_root=ROOT,
            probe_installed_source=False,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["installed_source_report"]["status"],
            "remote_installed_source_probe_pending",
        )
        self.assertFalse(report["safety"]["installed_source_probed"])
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["weights_accessed"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["real_projection_constructed"])

    def test_installed_source_provider_digest_is_recomputed_and_rejected(self):
        source = DIAGNOSTIC_SOURCE.encode("utf-8")
        actual = hashlib.sha256(source).hexdigest()
        with self.assertRaises(PermissionError):
            probe_installed_source_compatibility(
                lambda: ("0.8.32", Path("synthetic/model.py"), source, "0" * 64)
            )
        self.assertNotEqual(actual, "0" * 64)

    def test_authorization_schema_and_unique_artifact_paths_are_exact(self):
        schema = json.loads(
            (ROOT / AUTHORIZATION_SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["properties"]), AUTHORIZATION_FIELDS)
        self.assertEqual(set(schema["required"]), AUTHORIZATION_FIELDS)
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse((ROOT / AUTHORIZATION_RELATIVE_PATH).exists())
        self.assertFalse((ROOT / OUTPUT_RELATIVE_DIR / "execution_claim.json").exists())
        self.assertFalse((ROOT / OUTPUT_RELATIVE_DIR / "projection_artifact.json").exists())

    def test_copied_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied.json"
            copied.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d6d_ii_static_report(
                    config_path=copied,
                    project_root=ROOT,
                    probe_installed_source=False,
                )

    def test_future_runner_requires_exact_execution_lock_before_side_effect(self):
        environment = dict(os.environ)
        environment.pop("PSA_SELF_MODEL_D6D_REAL_JOINT", None)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_self_model_v0_1_coupling_d6d_joint.py"),
                "--project-root",
                str(ROOT),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("single-use lock is absent", completed.stderr)
        self.assertFalse((ROOT / AUTHORIZATION_RELATIVE_PATH).exists())
        self.assertFalse((ROOT / OUTPUT_RELATIVE_DIR / "execution_claim.json").exists())

    def test_no_model_modules_are_imported_by_static_suite(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
