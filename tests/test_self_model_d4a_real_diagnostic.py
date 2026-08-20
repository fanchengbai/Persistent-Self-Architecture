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
from psa.self_model import d4a_real_diagnostic as module
from psa.self_model.d4a_real_diagnostic import (
    CONFIG_RELATIVE_PATH,
    D4A_EXECUTION_LOCK_ENV,
    D4A_EXECUTION_LOCK_VALUE,
    D4A_OWNER_AUTHORIZATION_TEXT,
    _create_claim,
    _read_spec,
    build_d4a_real_authorization,
    build_d4a_real_entry_static_report,
    run_d4a_real_diagnostic,
    validate_d4a_real_authorization,
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
        return 1234


class FakeTorch:
    cuda = FakeCuda()


class FakeAdapter:
    def __init__(self):
        self.model = SimpleNamespace()
        self.torch = FakeTorch()

    def model_metadata(self):
        return {"model_id": "rwkv7-g1h-2.9b-20260710"}


class D4ARealDiagnosticTests(unittest.TestCase):
    def _root(self, directory: str) -> Path:
        root = Path(directory)
        config = root / CONFIG_RELATIVE_PATH
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG, config)
        model_config = root / MODEL_CONFIG_RELATIVE
        model_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / MODEL_CONFIG_RELATIVE, model_config)
        return root

    def _authorization(self, root: Path) -> Path:
        authorization = build_d4a_real_authorization(
            config_path=root / CONFIG_RELATIVE_PATH,
            project_root=root,
            authorization_text=D4A_OWNER_AUTHORIZATION_TEXT,
            git_metadata=GIT,
        )
        path = root / "results/authorizations/d4a.json"
        write_json_exclusive(path, authorization)
        return path

    def test_config_freezes_failed_fixture_and_future_authority(self):
        spec = _read_spec(CONFIG)
        self.assertEqual(
            spec["fixture"],
            {"token_ids": [2764], "state_input": "none", "full_output": False},
        )
        self.assertEqual(spec["model_forward_call_count"], 9)
        self.assertEqual(spec["discarded_warmup_call_count"], 0)
        self.assertFalse(spec["execution_authorized_at_implementation"])
        self.assertTrue(spec["future_exact_owner_authorization_required"])
        self.assertFalse(spec["d4_rerun_authorized"])
        self.assertFalse(spec["d5_authorized"])

    def test_scope_changes_fail_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field, value in (
                ("execution_authorized_at_implementation", True),
                ("automatic_rerun_authorized", True),
                ("d5_authorized", True),
                ("model_forward_call_count", 10),
            ):
                changed = copy.deepcopy(payload)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(PermissionError):
                    _read_spec(path)

    def test_authorization_requires_exact_text_and_binds_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            with self.assertRaises(PermissionError):
                build_d4a_real_authorization(
                    config_path=root / CONFIG_RELATIVE_PATH,
                    project_root=root,
                    authorization_text="下一轮",
                    git_metadata=GIT,
                )
            authorization = build_d4a_real_authorization(
                config_path=root / CONFIG_RELATIVE_PATH,
                project_root=root,
                authorization_text=D4A_OWNER_AUTHORIZATION_TEXT,
                git_metadata=GIT,
            )
            self.assertEqual(authorization["git_commit"], GIT["commit"])
            self.assertTrue(authorization["single_use"])
            self.assertFalse(authorization["automatic_rerun_authorized"])
            digest = authorization.pop("authorization_digest_sha256")
            self.assertEqual(digest, sha256_json(authorization))

    def test_machine_authorization_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            path = self._authorization(root)
            authorization = json.loads(path.read_text(encoding="utf-8"))
            authorization["d5_authorized"] = True
            path.write_text(json.dumps(authorization), encoding="utf-8")
            with self.assertRaises(PermissionError):
                validate_d4a_real_authorization(
                    authorization_path=path,
                    config_path=root / CONFIG_RELATIVE_PATH,
                    git=GIT,
                )

    def test_claim_is_single_use_and_binds_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = self._authorization(root)
            output = root / "results/development/d4a"
            claim = _create_claim(
                output_dir=output,
                config_path=root / CONFIG_RELATIVE_PATH,
                authorization_path=authorization,
                git=GIT,
            )
            self.assertTrue(claim.is_file())
            with self.assertRaises(FileExistsError):
                _create_claim(
                    output_dir=output,
                    config_path=root / CONFIG_RELATIVE_PATH,
                    authorization_path=authorization,
                    git=GIT,
                )

    def test_missing_execution_lock_fails_before_git_or_model(self):
        environment = dict(os.environ)
        environment.pop(D4A_EXECUTION_LOCK_ENV, None)
        environment.pop("RWKV_DE_VERSION", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            module, "_git_metadata"
        ) as git, patch.object(module.RWKV7Adapter, "load") as load:
            with self.assertRaises(PermissionError):
                run_d4a_real_diagnostic(
                    config_path=CONFIG,
                    authorization_path="results/authorizations/missing.json",
                    project_root=ROOT,
                    output_dir="results/development/missing",
                )
        git.assert_not_called()
        load.assert_not_called()

    def test_cli_missing_lock_creates_no_authorization_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            environment = dict(os.environ)
            environment.pop(D4A_EXECUTION_LOCK_ENV, None)
            environment.pop("RWKV_DE_VERSION", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_self_model_v0_1_d4a_real_diagnostic.py"),
                    "--config",
                    str(root / CONFIG_RELATIVE_PATH),
                    "--project-root",
                    str(root),
                    "--authorization",
                    "results/authorizations/d4a.json",
                    "--output-dir",
                    "results/development/d4a",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("single-use lock is absent", completed.stderr)
            self.assertFalse((root / "results/authorizations/d4a.json").exists())
            self.assertFalse((root / "results/development/d4a").exists())

    def test_cli_rejects_paths_outside_controlled_results_before_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            environment = dict(os.environ)
            environment[D4A_EXECUTION_LOCK_ENV] = D4A_EXECUTION_LOCK_VALUE
            environment.pop("RWKV_DE_VERSION", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_self_model_v0_1_d4a_real_diagnostic.py"),
                    "--config",
                    str(root / CONFIG_RELATIVE_PATH),
                    "--project-root",
                    str(root),
                    "--authorization",
                    "outside.json",
                    "--output-dir",
                    "results/development/d4a",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must stay in results/authorizations", completed.stderr)
            self.assertFalse((root / "outside.json").exists())
            self.assertFalse((root / "results/development/d4a").exists())

    def test_fake_orchestration_writes_report_after_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = self._authorization(root)
            output = root / "results/development/d4a"
            model_config = SimpleNamespace(
                model_id="rwkv7-g1h-2.9b-20260710", environment={}
            )
            diagnostic = {
                "valid": True,
                "diagnostic_classification": "fake_classification",
            }
            environment = dict(os.environ)
            environment[D4A_EXECUTION_LOCK_ENV] = D4A_EXECUTION_LOCK_VALUE
            environment.pop("RWKV_DE_VERSION", None)
            with patch.dict(os.environ, environment, clear=True), patch.object(
                module, "_git_metadata", return_value=GIT
            ), patch.object(
                module,
                "_installed_source",
                return_value=(
                    "0.8.32",
                    Path("/fake/rwkv/model.py"),
                    b"source",
                    module.EXPECTED_RWKV_MODEL_SOURCE_SHA256,
                ),
            ), patch.object(
                module, "load_model_config", return_value=model_config
            ), patch.object(
                module.RWKV7Adapter, "load", return_value=FakeAdapter()
            ), patch.object(
                module, "RWKV7RecompiledUnmodifiedRuntime", return_value=SimpleNamespace()
            ), patch.object(
                module, "RWKV7InstrumentedOffRuntime", return_value=SimpleNamespace()
            ), patch.object(
                module,
                "execute_d4a_fake_or_authorized_diagnostic",
                return_value=diagnostic,
            ), patch.dict(
                sys.modules, {"rwkv.model": SimpleNamespace()}, clear=False
            ):
                report = run_d4a_real_diagnostic(
                    config_path=root / CONFIG_RELATIVE_PATH,
                    authorization_path=authorization,
                    project_root=root,
                    output_dir=output,
                )
            self.assertTrue(report["valid"])
            self.assertEqual(report["status"], "d4a_real_diagnostic_complete")
            self.assertTrue((output / "execution_claim.json").is_file())
            self.assertTrue((output / "report.json").is_file())
            self.assertFalse((output / "failure.json").exists())
            self.assertFalse(report["safety"]["d4_status_changed"])
            self.assertFalse(report["safety"]["d5_authorized"])

    def test_post_claim_failure_is_persisted_and_blocks_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            authorization = self._authorization(root)
            output = root / "results/development/d4a"
            environment = dict(os.environ)
            environment[D4A_EXECUTION_LOCK_ENV] = D4A_EXECUTION_LOCK_VALUE
            environment.pop("RWKV_DE_VERSION", None)
            with patch.dict(os.environ, environment, clear=True), patch.object(
                module, "_git_metadata", return_value=GIT
            ), patch.object(
                module,
                "_installed_source",
                return_value=(
                    "0.8.32",
                    Path("/fake/rwkv/model.py"),
                    b"source",
                    module.EXPECTED_RWKV_MODEL_SOURCE_SHA256,
                ),
            ), patch.object(
                module, "load_model_config", side_effect=RuntimeError("fake failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "fake failure"):
                    run_d4a_real_diagnostic(
                        config_path=root / CONFIG_RELATIVE_PATH,
                        authorization_path=authorization,
                        project_root=root,
                        output_dir=output,
                    )
            self.assertTrue((output / "execution_claim.json").is_file())
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(
                failure["status"], "d4a_execution_attempt_failed_claim_consumed"
            )
            self.assertFalse(failure["automatic_rerun_authorized"])
            with self.assertRaises(FileExistsError):
                _create_claim(
                    output_dir=output,
                    config_path=root / CONFIG_RELATIVE_PATH,
                    authorization_path=authorization,
                    git=GIT,
                )

    def test_static_report_is_no_model_and_complete(self):
        report = build_d4a_real_entry_static_report(
            config_path=CONFIG, project_root=ROOT
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["execution_claim_created"])


if __name__ == "__main__":
    unittest.main()
