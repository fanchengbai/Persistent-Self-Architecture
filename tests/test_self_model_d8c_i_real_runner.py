from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from psa.artifacts import sha256_json
from psa.self_model.d8c_i_real_runner import (
    AUTHORIZATION_FIELDS,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    CALL_IDS_SHA256,
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    IMPLEMENTATION_CONFIRMATION,
    NEXT_GATE,
    _entry_ast_audit,
    _launcher_ast_audit,
    _validate_launcher_environment,
    _write_json_exclusive,
    build_d8c_authorization,
    build_static_report,
    run_pure_python_acceptance,
    validate_authorization_payload,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _fake_git() -> dict[str, str]:
    return {
        "branch": "main",
        "commit": "a" * 40,
        "origin_main": "a" * 40,
        "status": "",
    }


class D8CIRealRunnerTests(unittest.TestCase):
    def test_config_freezes_runner_and_closes_execution(self):
        config = _load(CONFIG_RELATIVE_PATH)
        self.assertTrue(all(validate_config(config).values()))
        self.assertEqual(
            config["implementation_confirmation_text"], IMPLEMENTATION_CONFIRMATION
        )
        self.assertEqual(
            config["future_execution_authorization_text"],
            FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        )
        self.assertEqual(config["execution_plan"]["total_forward_calls"], 584)
        self.assertFalse(
            config["implementation_authority"]["model_execution_authorized_at_implementation"]
        )

    def test_authorization_schema_is_exact_and_single_use(self):
        schema = _load(AUTHORIZATION_SCHEMA_RELATIVE_PATH)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), AUTHORIZATION_FIELDS)
        self.assertEqual(set(schema["properties"]), AUTHORIZATION_FIELDS)
        self.assertEqual(schema["properties"]["model_forward_calls"]["const"], 584)
        self.assertTrue(schema["properties"]["single_use"]["const"])
        self.assertFalse(schema["properties"]["d8c_rerun_authorized"]["const"])

    def test_pure_python_acceptance_expands_exact_plan(self):
        acceptance = run_pure_python_acceptance(ROOT)
        self.assertTrue(acceptance["valid"])
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(acceptance["counts"]["conditioning_calls"], 8)
        self.assertEqual(acceptance["counts"]["pair_blocks"], 288)
        self.assertEqual(acceptance["counts"]["total_forward_calls"], 584)
        self.assertEqual(acceptance["call_ids_digest"], CALL_IDS_SHA256)

    def test_authorization_builder_is_bound_and_does_not_write(self):
        authorization = build_d8c_authorization(
            config_path=ROOT / CONFIG_RELATIVE_PATH,
            project_root=ROOT,
            authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
            git_metadata=_fake_git(),
        )
        self.assertEqual(set(authorization), AUTHORIZATION_FIELDS)
        self.assertEqual(authorization["git_commit"], "a" * 40)
        self.assertEqual(authorization["model_forward_calls"], 584)
        self.assertFalse(authorization["d8c_rerun_authorized"])
        self.assertFalse((ROOT / "results/authorizations/self_model_v0_1_d8c_real_v01.json").exists())

    def test_authorization_mutation_fails_even_with_recomputed_digest(self):
        authorization = build_d8c_authorization(
            config_path=ROOT / CONFIG_RELATIVE_PATH,
            project_root=ROOT,
            authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
            git_metadata=_fake_git(),
        )
        expected = copy.deepcopy(authorization)
        changed = copy.deepcopy(authorization)
        changed["model_forward_calls"] = 583
        payload = {
            key: value
            for key, value in changed.items()
            if key != "authorization_digest_sha256"
        }
        changed["authorization_digest_sha256"] = sha256_json(payload)
        with self.assertRaises(PermissionError):
            validate_authorization_payload(changed, expected)

    def test_wrong_authorization_text_rejected(self):
        with self.assertRaises(PermissionError):
            build_d8c_authorization(
                config_path=ROOT / CONFIG_RELATIVE_PATH,
                project_root=ROOT,
                authorization_text="not authorized",
                git_metadata=_fake_git(),
            )

    def test_static_report_is_no_model_and_all_artifacts_absent(self):
        report = build_static_report(
            config_path=ROOT / CONFIG_RELATIVE_PATH,
            project_root=ROOT,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertTrue(all(report["execution_artifacts"].values()))
        self.assertFalse(report["safety"]["installed_source_probed"])
        self.assertFalse(report["safety"]["execution_claim_created"])
        self.assertFalse(report["safety"]["model_executed"])

    def test_ast_orders_authorization_claim_import_and_load(self):
        lines = _entry_ast_audit()
        self.assertLess(lines["validate_d8c_authorization"], lines["_probe_installed_source"])
        self.assertLess(lines["_create_claim"], lines["_runtime_dependencies"])
        self.assertLess(lines["_runtime_dependencies"], lines["load_model_config"])
        self.assertLess(lines["_apply_runtime_determinism"], lines["load"])
        launcher = _launcher_ast_audit(ROOT)
        self.assertLess(
            launcher["_validate_launcher_environment"],
            launcher["build_d8c_authorization"],
        )
        self.assertLess(
            launcher["build_d8c_authorization"],
            launcher["_write_json_exclusive"],
        )

    def test_launcher_determinism_requires_exact_pre_python_environment(self):
        exact = {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "28083101",
        }
        with patch.dict(os.environ, exact, clear=True):
            self.assertTrue(all(_validate_launcher_environment().values()))
        with patch.dict(os.environ, {"PYTHONHASHSEED": "wrong"}, clear=True):
            with self.assertRaises(PermissionError):
                _validate_launcher_environment()

    def test_exclusive_json_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            _write_json_exclusive(path, {"valid": True})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(path, {"valid": False})

    def test_config_mutations_fail_closed(self):
        config = _load(CONFIG_RELATIVE_PATH)
        for section, field, value in (
            ("execution_plan", "total_forward_calls", 583),
            ("implementation_authority", "model_execution_authorized_at_implementation", True),
            ("artifact_contract", "single_use", False),
            ("frozen_prerequisites", "expanded_call_ids_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(config)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_wrong_config_path_and_model_modules_absent(self):
        with self.assertRaises(PermissionError):
            build_static_report(
                config_path=ROOT / "configs/development/self_model_v0_1_d8c_real_numerical_identifiability.json",
                project_root=ROOT,
            )
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
