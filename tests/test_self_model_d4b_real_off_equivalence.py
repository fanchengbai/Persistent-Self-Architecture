from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from psa.artifacts import sha256_json
from psa.self_model import d4b_real_off_equivalence as module
from psa.self_model.d4b_real_off_equivalence import (
    AUTHORIZATION_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    D4B_EXECUTION_LOCK_ENV,
    D4B_EXECUTION_LOCK_VALUE,
    D4B_OWNER_AUTHORIZATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    _create_claim,
    _read_spec,
    build_d4b_real_authorization,
    build_d4b_real_entry_static_report,
    run_d4b_real_off_equivalence,
    validate_d4b_real_authorization,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH
MODEL_CONFIG_RELATIVE = "configs/models/rwkv7_g1h_2.9b.candidate.json"
GIT = {"commit": "a" * 40, "branch": "main", "status_porcelain": ""}


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def reset_peak_memory_stats():
        return None

    @staticmethod
    def synchronize():
        return None

    @staticmethod
    def max_memory_allocated():
        return 4321


class FakeTorch:
    cuda = FakeCuda()


class FakeAdapter:
    def __init__(self):
        self.model = SimpleNamespace()
        self.torch = FakeTorch()

    def model_metadata(self):
        return {"model_id": "rwkv7-g1h-2.9b-20260710"}


class D4BRealOffEquivalenceTests(unittest.TestCase):
    def _root(self, directory: str) -> Path:
        root = Path(directory)
        for relative in (
            CONFIG_RELATIVE_PATH,
            MODEL_CONFIG_RELATIVE,
            AUTHORIZATION_SCHEMA_RELATIVE_PATH,
        ):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        return root

    def _authorization(self, root: Path) -> Path:
        authorization = build_d4b_real_authorization(
            config_path=root / CONFIG_RELATIVE_PATH,
            project_root=root,
            authorization_text=D4B_OWNER_AUTHORIZATION_TEXT,
            git_metadata=GIT,
        )
        path = root / AUTHORIZATION_RELATIVE_PATH
        write_json_exclusive(path, authorization)
        return path

    def test_config_freezes_schedule_paths_and_future_authority(self):
        spec = _read_spec(CONFIG)
        self.assertEqual(spec["model_forward_call_count"], 21)
        self.assertEqual(spec["within_route_comparison_count"], 24)
        self.assertEqual(spec["cross_route_comparison_count"], 96)
        self.assertEqual(spec["authorization_path"], AUTHORIZATION_RELATIVE_PATH)
        self.assertEqual(spec["output_dir"], OUTPUT_RELATIVE_DIR)
        self.assertFalse(spec["execution_authorized_at_implementation"])
        self.assertTrue(spec["future_exact_owner_authorization_required"])
        self.assertFalse(spec["d4b_rerun_authorized"])
        self.assertFalse(spec["d5_authorized"])

    def test_scope_changes_fail_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field, value in (
                ("execution_authorized_at_implementation", True),
                ("automatic_rerun_authorized", True),
                ("d5_authorized", True),
                ("model_forward_call_count", 22),
                ("comparison", "allclose"),
                ("output_dir", "results/development/other"),
            ):
                changed = copy.deepcopy(payload)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(PermissionError):
                    _read_spec(path)

    def test_authorization_schema_and_exact_text_are_frozen(self):
        schema = json.loads(
            (ROOT / AUTHORIZATION_SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["authorization_text"]["const"],
            D4B_OWNER_AUTHORIZATION_TEXT,
        )
        self.assertEqual(
            schema["properties"]["runtime_static_report_sha256"]["const"],
            module.D4B_RUNTIME_STATIC_REPORT_DIGEST,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            with self.assertRaises(PermissionError):
                build_d4b_real_authorization(
                    config_path=root / CONFIG_RELATIVE_PATH,
                    project_root=root,
                    authorization_text="下一轮确认",
                    git_metadata=GIT,
                )

    def test_authorization_binds_commit_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = build_d4b_real_authorization(
                config_path=root / CONFIG_RELATIVE_PATH,
                project_root=root,
                authorization_text=D4B_OWNER_AUTHORIZATION_TEXT,
                git_metadata=GIT,
            )
            self.assertEqual(authorization["git_commit"], GIT["commit"])
            self.assertTrue(authorization["single_use"])
            digest = authorization["authorization_digest_sha256"]
            digest_payload = {
                key: value
                for key, value in authorization.items()
                if key != "authorization_digest_sha256"
            }
            self.assertEqual(digest, sha256_json(digest_payload))
            path = root / AUTHORIZATION_RELATIVE_PATH
            write_json_exclusive(path, authorization)
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["d5_authorized"] = True
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(PermissionError):
                validate_d4b_real_authorization(
                    authorization_path=path,
                    config_path=root / CONFIG_RELATIVE_PATH,
                    project_root=root,
                    git=GIT,
                )

    def test_claim_is_single_use_and_binds_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = self._authorization(root)
            output = root / OUTPUT_RELATIVE_DIR
            claim = _create_claim(
                output_dir=output,
                config_path=root / CONFIG_RELATIVE_PATH,
                authorization_path=authorization,
                git=GIT,
            )
            payload = json.loads(claim.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["runtime_static_report_sha256"],
                module.D4B_RUNTIME_STATIC_REPORT_DIGEST,
            )
            self.assertEqual(payload["model_forward_call_count"], 21)
            with self.assertRaises(FileExistsError):
                _create_claim(
                    output_dir=output,
                    config_path=root / CONFIG_RELATIVE_PATH,
                    authorization_path=authorization,
                    git=GIT,
                )

    def test_missing_execution_lock_fails_before_git_or_model(self):
        environment = dict(os.environ)
        environment.pop(D4B_EXECUTION_LOCK_ENV, None)
        environment.pop("RWKV_DE_VERSION", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            module, "_git_metadata"
        ) as git, patch.object(module.RWKV7Adapter, "load") as load:
            with self.assertRaises(PermissionError):
                run_d4b_real_off_equivalence(
                    config_path=CONFIG,
                    authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
                    project_root=ROOT,
                    output_dir=ROOT / OUTPUT_RELATIVE_DIR,
                )
        git.assert_not_called()
        load.assert_not_called()

    def test_cli_missing_lock_creates_no_authorization_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            environment = dict(os.environ)
            environment.pop(D4B_EXECUTION_LOCK_ENV, None)
            environment.pop("RWKV_DE_VERSION", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_self_model_v0_1_d4b_real_off_equivalence.py"),
                    "--project-root",
                    str(root),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("single-use lock is absent", completed.stderr)
            self.assertFalse((root / AUTHORIZATION_RELATIVE_PATH).exists())
            self.assertFalse((root / OUTPUT_RELATIVE_DIR).exists())

    def test_cli_rejects_nonfrozen_paths_before_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            environment = dict(os.environ)
            environment[D4B_EXECUTION_LOCK_ENV] = D4B_EXECUTION_LOCK_VALUE
            environment.pop("RWKV_DE_VERSION", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_self_model_v0_1_d4b_real_off_equivalence.py"),
                    "--project-root",
                    str(root),
                    "--authorization",
                    "results/authorizations/other.json",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("authorization path is not frozen", completed.stderr)
            self.assertFalse((root / "results/authorizations/other.json").exists())
            self.assertFalse((root / OUTPUT_RELATIVE_DIR).exists())

    def test_fake_orchestration_claim_precedes_model_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = self._authorization(root)
            output = root / OUTPUT_RELATIVE_DIR
            model_config = SimpleNamespace(
                model_id="rwkv7-g1h-2.9b-20260710", environment={}
            )
            core = {
                "valid": True,
                "status": "fake",
                "safety": {
                    "real_model_entry_implemented": False,
                    "real_model_executed": False,
                },
            }

            def load_config(*args, **kwargs):
                self.assertTrue((output / "execution_claim.json").is_file())
                return model_config

            environment = dict(os.environ)
            environment[D4B_EXECUTION_LOCK_ENV] = D4B_EXECUTION_LOCK_VALUE
            environment.pop("RWKV_DE_VERSION", None)
            with patch.dict(os.environ, environment, clear=True), patch.object(
                module, "_git_metadata", return_value=GIT
            ), patch.object(
                module,
                "_installed_source",
                return_value=("0.8.32", Path("/fake/model.py"), b"source", "b" * 64),
            ), patch.object(
                module, "load_model_config", side_effect=load_config
            ), patch.object(
                module.RWKV7Adapter, "load", return_value=FakeAdapter()
            ), patch.object(
                module, "RWKV7CouplingOffAdapter", return_value=SimpleNamespace()
            ), patch.object(
                module,
                "RWKV7RecompiledUnmodifiedRuntime",
                return_value=SimpleNamespace(),
            ), patch.object(
                module, "RWKV7InstrumentedOffRuntime", return_value=SimpleNamespace()
            ), patch.object(
                module,
                "execute_d4b_fake_or_future_authorized_core",
                return_value=core,
            ), patch.dict(
                sys.modules, {"rwkv.model": SimpleNamespace()}, clear=False
            ):
                report = run_d4b_real_off_equivalence(
                    config_path=root / CONFIG_RELATIVE_PATH,
                    authorization_path=authorization,
                    project_root=root,
                    output_dir=output,
                )
            self.assertTrue(report["valid"])
            self.assertEqual(report["status"], "d4b_real_off_equivalence_passed")
            self.assertEqual(report["decision_effect"], "d5_review_candidate_only")
            self.assertEqual(
                report["runtime_core"]["execution_context"], "authorized_real_2_9b"
            )
            self.assertNotIn("safety", report["runtime_core"])
            self.assertFalse(
                report["runtime_core_template_safety"]["real_model_executed"]
            )
            self.assertTrue(report["safety"]["real_2_9b_model_executed"])
            self.assertFalse(report["safety"]["d5_authorized"])
            self.assertTrue((output / "report.json").is_file())
            self.assertFalse((output / "failure.json").exists())

    def test_post_claim_failure_is_persisted_and_blocks_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = self._authorization(root)
            output = root / OUTPUT_RELATIVE_DIR
            environment = dict(os.environ)
            environment[D4B_EXECUTION_LOCK_ENV] = D4B_EXECUTION_LOCK_VALUE
            environment.pop("RWKV_DE_VERSION", None)
            with patch.dict(os.environ, environment, clear=True), patch.object(
                module, "_git_metadata", return_value=GIT
            ), patch.object(
                module,
                "_installed_source",
                return_value=("0.8.32", Path("/fake/model.py"), b"source", "b" * 64),
            ), patch.object(
                module, "load_model_config", side_effect=RuntimeError("fake failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "fake failure"):
                    run_d4b_real_off_equivalence(
                        config_path=root / CONFIG_RELATIVE_PATH,
                        authorization_path=authorization,
                        project_root=root,
                        output_dir=output,
                    )
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(
                failure["status"], "d4b_execution_attempt_failed_claim_consumed"
            )
            self.assertFalse(failure["d4b_rerun_authorized"])
            self.assertFalse(failure["automatic_rerun_authorized"])
            with self.assertRaises(FileExistsError):
                _create_claim(
                    output_dir=output,
                    config_path=root / CONFIG_RELATIVE_PATH,
                    authorization_path=authorization,
                    git=GIT,
                )

    def test_static_report_is_no_model_and_complete(self):
        report = build_d4b_real_entry_static_report(
            config_path=CONFIG, project_root=ROOT
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["execution_claim_created"])
        self.assertFalse(report["safety"]["machine_authorization_created"])


if __name__ == "__main__":
    unittest.main()
