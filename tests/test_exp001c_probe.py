from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.cli import main
from psa.development.exp001c_probe import (
    PROBE_EXECUTION_LOCK,
    PROBE_SOURCE_FILES,
    build_exp001c_probe_pilot_authorization,
    build_exp001c_probe_manifest,
    run_exp001c_development_probe,
    verify_exp001c_probe_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/preregistration/exp001c_prefix_semantics.draft.json"
MODEL = ROOT / "configs/models/rwkv7_g1h_2.9b.candidate.json"


class _FakeBackend:
    def run_probe(self, manifest):
        return {
            "probe_result_version": "0.1-development",
            "development_only": True,
            "model_executed": True,
            "formal_test_set_accessed": False,
            "contains_confirmatory_decision": False,
            "record_count": 0,
            "records": [],
        }


class Exp001CProbeRunnerTests(unittest.TestCase):
    def test_manifest_build_and_verification_are_model_free(self) -> None:
        manifest = build_exp001c_probe_manifest(
            design_config_path=DESIGN,
            model_config_path=MODEL,
            project_root=ROOT,
        )
        self.assertFalse(manifest["model_executed"])
        self.assertFalse(manifest["formal_test_set_accessed"])
        self.assertFalse(manifest["design_config"]["noncore_pilot_authorized_at_build"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_probe_manifest(path, project_root=ROOT)
        self.assertTrue(verification["valid"])
        self.assertTrue(verification["source_inventory_complete"])

    def test_cli_builds_and_verifies_unrun_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            authorization_path = Path(temporary) / "authorization.json"
            code = main(
                [
                    "exp001c-probe-manifest",
                    "--design",
                    str(DESIGN),
                    "--model-config",
                    str(MODEL),
                    "--output",
                    str(path),
                    "--project-root",
                    str(ROOT),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(
                main(
                    [
                        "exp001c-probe-verify",
                        "--manifest",
                        str(path),
                        "--project-root",
                        str(ROOT),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "exp001c-probe-authorize",
                        "--manifest",
                        str(path),
                        "--output",
                        str(authorization_path),
                        "--project-root",
                        str(ROOT),
                    ]
                ),
                2,
            )
            self.assertFalse(authorization_path.exists())

    def test_execution_lock_fails_before_paths_or_backend_factory(self) -> None:
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            return _FakeBackend()

        with self.assertRaisesRegex(PermissionError, "execution lock"):
            run_exp001c_development_probe(
                manifest_path="missing-manifest.json",
                authorization_path="missing-authorization.json",
                output_dir="unused-output",
                backend_factory=factory,
                execution_lock="",
                project_root=ROOT,
            )
        self.assertFalse(factory_called)

    def test_locked_run_cli_fails_before_manifest_or_model_access(self) -> None:
        self.assertEqual(
            main(
                [
                    "exp001c-probe-run",
                    "--manifest",
                    "missing-manifest.json",
                    "--authorization",
                    "missing-authorization.json",
                    "--model-config",
                    "missing-model.json",
                    "--output-dir",
                    "unused-output",
                    "--project-root",
                    str(ROOT),
                ]
            ),
            2,
        )

    def test_current_design_closes_pilot_before_authorization_file(self) -> None:
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            return _FakeBackend()

        manifest = build_exp001c_probe_manifest(
            design_config_path=DESIGN,
            model_config_path=MODEL,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                PermissionError,
                "built without pilot authority",
            ):
                run_exp001c_development_probe(
                    manifest_path=manifest_path,
                    authorization_path=Path(temporary) / "missing.json",
                    output_dir=Path(temporary) / "output",
                    backend_factory=factory,
                    execution_lock=PROBE_EXECUTION_LOCK,
                    project_root=ROOT,
                )
        self.assertFalse(factory_called)

    def test_verifier_rejects_extra_or_traversing_inventory_entries(self) -> None:
        manifest = build_exp001c_probe_manifest(
            design_config_path=DESIGN,
            model_config_path=MODEL,
            project_root=ROOT,
        )
        manifest["locked_source_digests"]["../outside.json"] = "0" * 64
        payload = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_digest_sha256"
        }
        manifest["manifest_digest_sha256"] = sha256_json(payload)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_probe_manifest(path, project_root=ROOT)
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["source_inventory_complete"])

    def test_authorized_fixture_runner_calls_factory_only_after_all_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary) / "project"
            for relative in PROBE_SOURCE_FILES:
                source = ROOT / relative
                destination = sandbox / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            model_relative = "configs/models/rwkv7_g1h_2.9b.candidate.json"
            (sandbox / model_relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MODEL, sandbox / model_relative)
            design_path = sandbox / str(DESIGN.relative_to(ROOT))
            design = json.loads(design_path.read_text(encoding="utf-8"))
            design["authority"]["pilot_run_authorized"] = True
            design["development_authorization"]["noncore_pilot_authorized"] = True
            design["development_authorization"]["model_execution_authorized"] = True
            design_path.write_text(json.dumps(design), encoding="utf-8")
            manifest = build_exp001c_probe_manifest(
                design_config_path=design_path,
                model_config_path=sandbox / model_relative,
                project_root=sandbox,
            )
            manifest_path = sandbox / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            authorization = build_exp001c_probe_pilot_authorization(
                manifest_path=manifest_path,
                project_root=sandbox,
            )
            authorization_path = sandbox / "authorization.json"
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            factory_calls = 0

            def factory():
                nonlocal factory_calls
                factory_calls += 1
                return _FakeBackend()

            summary = run_exp001c_development_probe(
                manifest_path=manifest_path,
                authorization_path=authorization_path,
                output_dir=sandbox / "output",
                backend_factory=factory,
                execution_lock=PROBE_EXECUTION_LOCK,
                project_root=sandbox,
            )
            self.assertEqual(factory_calls, 1)
            self.assertTrue(summary["valid"])
            self.assertTrue(summary["model_executed"])
            self.assertFalse(summary["formal_test_set_accessed"])
            self.assertFalse(summary["contains_confirmatory_decision"])

    def test_authorization_builder_rejects_closed_current_design(self) -> None:
        manifest = build_exp001c_probe_manifest(
            design_config_path=DESIGN,
            model_config_path=MODEL,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                PermissionError,
                "not built with pilot authority",
            ):
                build_exp001c_probe_pilot_authorization(
                    manifest_path=manifest_path,
                    project_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
