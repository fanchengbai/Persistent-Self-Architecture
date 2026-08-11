from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from psa.artifacts import sha256_json
from psa.development.exp001c_v02_stage_b_design import (
    build_exp001c_v02_stage_b_design_manifest,
)
from psa.development.exp001c_v02_stage_b_preflight import (
    STAGE_B_AUTHORIZATION_TEXT,
    build_exp001c_v02_stage_b_authorization,
    build_exp001c_v02_stage_b_preflight,
    validate_exp001c_v02_stage_b_machine_authority,
    verify_exp001c_v02_stage_b_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_CONFIG = (
    ROOT / "configs" / "development" / "exp001c_v02_stage_b_design.draft.json"
)
MODEL_CONFIG = ROOT / "configs" / "models" / "rwkv7_g1h_2.9b.candidate.json"
MODEL_TARGET = "psa.development.exp001c_v02_stage_b_preflight.load_model_config"
VERIFY_TARGET = (
    "psa.development.exp001c_v02_stage_b_preflight."
    "verify_exp001c_v02_stage_b_preflight"
)


def _environment(*, dirty: bool = False):
    return {
        "valid": True,
        "git": {"commit": "1" * 40, "branch": "main", "dirty": dirty},
    }


def _model():
    return SimpleNamespace(
        model_id="RWKV-x070-World-2.9B-v3-20250211-ctx4096",
        weights_sha256="2" * 64,
        weights_size_bytes=5_915_347_128,
        tokenizer_sha256="3" * 64,
        tokenizer_size_bytes=1_091_359,
    )


def _write_design(directory: Path):
    design = build_exp001c_v02_stage_b_design_manifest(
        design_config_path=DESIGN_CONFIG,
        project_root=ROOT,
    )
    path = directory / "stage_b_design_manifest.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    return design, path


def _stage_a_report(design):
    return {
        "report_version": "0.1-development",
        "stage_a_result_path": (
            "results/development/exp001c_v02_stage_a_pilot_v01/"
            "stage_a_result.json"
        ),
        "stage_a_summary_path": (
            "results/development/exp001c_v02_stage_a_pilot_v01/summary.json"
        ),
        "stage_a_result_sha256": design["stage_a_pass_evidence"][
            "stage_a_result_sha256"
        ],
        "stage_a_summary_sha256": "4" * 64,
        "checks": {"injected_test_fixture_valid": True},
        "valid": True,
        "model_loaded": False,
        "model_executed": False,
    }


def _build(directory: Path, *, dirty: bool = False):
    design, design_path = _write_design(directory)
    report = _stage_a_report(design)
    with patch(MODEL_TARGET, return_value=_model()):
        preflight = build_exp001c_v02_stage_b_preflight(
            design_manifest_path=design_path,
            stage_a_result_path=report["stage_a_result_path"],
            stage_a_summary_path=report["stage_a_summary_path"],
            model_config_path=MODEL_CONFIG,
            output_dir=directory / "stage_b_output",
            project_root=ROOT,
            environment_report=_environment(dirty=dirty),
            stage_a_artifact_report=report,
        )
    return design, design_path, report, preflight


class Exp001CV02StageBPreflightTests(unittest.TestCase):
    def test_builds_read_only_commit_and_artifact_bound_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, _, _, preflight = _build(root)
            self.assertTrue(preflight["valid"])
            self.assertEqual(
                preflight["status"],
                "preflight_valid_authorization_still_required",
            )
            self.assertFalse(preflight["model_loaded"])
            self.assertFalse(preflight["model_executed"])
            self.assertFalse(preflight["stage_b_execution_authorized"])
            self.assertFalse(preflight["stage_b_result_observation_authorized"])
            plan = preflight["run_plan_candidate"]
            self.assertEqual(plan["git_commit"], "1" * 40)
            self.assertEqual(plan["record_count"], 224)
            self.assertEqual(plan["condition_count"], 7)
            self.assertEqual(
                plan["design_manifest_digest_sha256"],
                design["design_manifest_digest_sha256"],
            )
            self.assertEqual(
                preflight["preflight_digest_sha256"], sha256_json(plan)
            )
            self.assertEqual(
                preflight["authorization_boundary"][
                    "required_authorization_text"
                ],
                STAGE_B_AUTHORIZATION_TEXT,
            )

    def test_dirty_worktree_fails_preflight_without_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, preflight = _build(Path(directory), dirty=True)
            self.assertFalse(preflight["valid"])
            self.assertFalse(preflight["checks"]["git_clean"])
            self.assertFalse(preflight["model_executed"])

    def test_persisted_preflight_reverifies_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, design_path, report, preflight = _build(root)
            preflight_path = root / "preflight.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            with patch(MODEL_TARGET, return_value=_model()):
                verification = verify_exp001c_v02_stage_b_preflight(
                    preflight_path=preflight_path,
                    design_manifest_path=design_path,
                    model_config_path=MODEL_CONFIG,
                    project_root=ROOT,
                    environment_report=_environment(),
                    stage_a_artifact_report=report,
                )
            self.assertTrue(verification["valid"])

            tampered = copy.deepcopy(preflight)
            tampered["run_plan_candidate"]["record_count"] = 223
            preflight_path.write_text(json.dumps(tampered), encoding="utf-8")
            with patch(MODEL_TARGET, return_value=_model()):
                verification = verify_exp001c_v02_stage_b_preflight(
                    preflight_path=preflight_path,
                    design_manifest_path=design_path,
                    model_config_path=MODEL_CONFIG,
                    project_root=ROOT,
                    environment_report=_environment(),
                    stage_a_artifact_report=report,
                )
            self.assertFalse(verification["valid"])

    def test_authorization_builder_requires_exact_future_owner_text(self) -> None:
        with self.assertRaisesRegex(PermissionError, "text is not exact"):
            build_exp001c_v02_stage_b_authorization(
                design_manifest_path="missing-design.json",
                preflight_path="missing-preflight.json",
                model_config_path="missing-model.json",
                authorization_text="继续下一轮",
                project_root=ROOT,
            )

    def test_authorization_builder_binds_verified_preflight(self) -> None:
        verification = {
            "valid": True,
            "design_manifest_digest_sha256": "5" * 64,
            "preflight_digest_sha256": "6" * 64,
            "stage_a_result_sha256": "7" * 64,
        }
        with patch(VERIFY_TARGET, return_value=verification):
            authorization = build_exp001c_v02_stage_b_authorization(
                design_manifest_path="design.json",
                preflight_path="preflight.json",
                model_config_path="model.json",
                authorization_text=STAGE_B_AUTHORIZATION_TEXT,
                project_root=ROOT,
            )
        self.assertTrue(authorization["model_execution_authorized"])
        self.assertTrue(
            authorization["stage_b_result_observation_authorized"]
        )
        self.assertFalse(authorization["stage_a_rerun_authorized"])
        payload = {
            key: value
            for key, value in authorization.items()
            if key != "authorization_digest_sha256"
        }
        self.assertEqual(
            authorization["authorization_digest_sha256"], sha256_json(payload)
        )

    def test_machine_authority_accepts_only_digest_bound_authorization(self) -> None:
        verification = {
            "valid": True,
            "design_manifest_digest_sha256": "5" * 64,
            "preflight_digest_sha256": "6" * 64,
            "stage_a_result_sha256": "7" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(VERIFY_TARGET, return_value=verification):
                authorization = build_exp001c_v02_stage_b_authorization(
                    design_manifest_path="design.json",
                    preflight_path="preflight.json",
                    model_config_path="model.json",
                    authorization_text=STAGE_B_AUTHORIZATION_TEXT,
                    project_root=ROOT,
                )
                path = root / "authorization.json"
                path.write_text(json.dumps(authorization), encoding="utf-8")
                authority = validate_exp001c_v02_stage_b_machine_authority(
                    design_manifest_path="design.json",
                    preflight_path="preflight.json",
                    authorization_path=path,
                    model_config_path="model.json",
                    project_root=ROOT,
                )
            self.assertTrue(authority["valid"])

            authorization["stage_a_rerun_authorized"] = True
            path.write_text(json.dumps(authorization), encoding="utf-8")
            with patch(VERIFY_TARGET, return_value=verification):
                with self.assertRaisesRegex(PermissionError, "authorization"):
                    validate_exp001c_v02_stage_b_machine_authority(
                        design_manifest_path="design.json",
                        preflight_path="preflight.json",
                        authorization_path=path,
                        model_config_path="model.json",
                        project_root=ROOT,
                    )


if __name__ == "__main__":
    unittest.main()
