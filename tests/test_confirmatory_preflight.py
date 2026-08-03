from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes, sha256_file

from psa.confirmatory import (
    build_confirmatory_preflight,
    verify_confirmatory_run_authorization,
)


class ConfirmatoryPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.temporary = tempfile.TemporaryDirectory(dir=cls.root)
        cls.runner_evidence_path = (
            Path(cls.temporary.name) / "runner_summary.json"
        )
        cls.runner_evidence_path.write_bytes(
            canonical_json_bytes(
                {
                    "valid": True,
                    "gate": "impl5b_confirmatory_runner_development",
                    "development_only": True,
                    "fixture_kind": "non_core_confirmatory_runner_fixture",
                    "group_count": 1,
                    "trial_count": 16,
                    "condition_count": 8,
                    "raw_record_count": 128,
                    "runner_source_digests": {
                        relative: sha256_file(cls.root / relative)
                        for relative in (
                            "src/psa/confirmatory/runner.py",
                            "src/psa/confirmatory/rwkv_backend.py",
                            "src/psa/confirmatory/development.py",
                        )
                    },
                    "contains_derived_accuracy": False,
                    "formal_authorization_used": False,
                    "confirmatory_experiment_run": False,
                    "confirmatory_results_observed": False,
                }
            )
        )
        model_config = json.loads(
            (
                cls.root
                / "configs"
                / "models"
                / "rwkv7_g1h_2.9b.candidate.json"
            ).read_text(encoding="utf-8")
        )
        cls.environment = {
            "valid": True,
            "git": {
                "commit": "1" * 40,
                "branch": "main",
                "dirty": False,
            },
            "runtime_environment": {
                "RWKV_V7_ON": "1",
                "RWKV_JIT_ON": "0",
                "RWKV_CUDA_ON": "0",
            },
            "disk": {"free_bytes": 30 * 1024**3},
            "nvidia_smi": {
                "gpus": [
                    {
                        "index": 0,
                        "name": "test GPU",
                        "driver_version": "test",
                        "memory_mib": 32768,
                        "compute_capability": "12.0",
                    }
                ]
            },
            "torch": {
                "version": "2.12.0+cu132",
                "cuda_runtime": "13.2",
            },
        }
        cls.assets = {
            "valid": True,
            "assets": [
                {
                    "id": "rwkv7-g1h-2.9b-20260710",
                    "status": "valid",
                    "sha256": model_config["weights"]["sha256"],
                    "size_bytes": model_config["weights"]["size_bytes"],
                },
                {
                    "id": "rwkv-world-tokenizer-20230424",
                    "status": "valid",
                    "sha256": model_config["tokenizer"]["sha256"],
                    "size_bytes": model_config["tokenizer"]["size_bytes"],
                },
            ],
        }
        common = dict(
            project_root=cls.root,
            final_package_dir=(
                cls.root / "preregistration" / "exp001" / "final_v1"
            ),
            core_set_package_dir=(
                cls.root / "preregistration" / "exp001" / "core_set_v1"
            ),
            model_config_path=(
                cls.root
                / "configs"
                / "models"
                / "rwkv7_g1h_2.9b.candidate.json"
            ),
            asset_manifest_path=(
                cls.root
                / "configs"
                / "assets"
                / "exp001_rwkv7_g1h_2.9b_candidate.json"
            ),
            asset_root=cls.root / ".psa-assets",
            environment_report=cls.environment,
            asset_report=cls.assets,
        )
        cls.preflight_without_runner = build_confirmatory_preflight(**common)
        cls.preflight = build_confirmatory_preflight(
            **common,
            runner_evidence_path=cls.runner_evidence_path,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_preflight_is_valid_but_does_not_authorize_or_run(self) -> None:
        self.assertTrue(self.preflight["valid"])
        self.assertFalse(
            self.preflight["confirmatory_experiment_authorized"]
        )
        self.assertFalse(self.preflight["confirmatory_experiment_run"])
        self.assertFalse(self.preflight["confirmatory_results_observed"])
        self.assertEqual(
            self.preflight["status"],
            "preflight_valid_authorization_still_required",
        )
        self.assertTrue(
            self.preflight["runner_development_evidence"]["valid"]
        )
        self.assertEqual(
            self.preflight["run_plan_candidate"][
                "planned_trial_condition_count"
            ],
            40960,
        )

    def test_preflight_without_runner_evidence_is_not_authorization_ready(self) -> None:
        report = self.preflight_without_runner
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["status"],
            "preflight_valid_runner_evidence_required",
        )
        self.assertEqual(
            report["route_decision"],
            "run_non_core_runner_development_gate",
        )
        self.assertFalse(report["runner_development_evidence"]["valid"])

    def test_core_set_authorization_cannot_be_reused_for_formal_run(self) -> None:
        old_authorization = json.loads(
            (
                self.root
                / "preregistration"
                / "exp001"
                / "core_set_authorization.json"
            ).read_text(encoding="utf-8")
        )
        report = verify_confirmatory_run_authorization(
            old_authorization,
            preflight=self.preflight,
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["scope_exact"])
        self.assertFalse(report["checks"]["preflight_digest_bound"])

    def test_future_authorization_must_bind_exact_preflight(self) -> None:
        authorization = {
            "authorization_version": "1.0",
            "experiment_id": "EXP-001",
            "authorized_by_role": "project_owner",
            "authorized_at_utc": "2026-08-01T00:00:00Z",
            "authorization_text": (
                "I explicitly authorize the frozen EXP-001 confirmatory run."
            ),
            "preflight_digest_sha256": self.preflight[
                "preflight_digest_sha256"
            ],
            "final_preregistration_digest_sha256": (
                "0daf056dc6b38aa20fa69dd9e8df9b8065876529947cbc01353ffe604933d0c9"
            ),
            "core_set_digest_sha256": (
                "6ea2b6be15a7728c96d84dcc8e48da64e740438980f818e78c8ee8570a47eb9d"
            ),
            "core_set_package_digest_sha256": (
                "9659e286de4128b43226f2d6df27075eba60bd953c2330ee70c0ec3e677f1642"
            ),
            "model_id": "rwkv7-g1h-2.9b-20260710",
            "authorization": {
                "run_confirmatory_experiment": True,
                "observe_results_after_full_completion": True,
                "modify_frozen_design": False,
                "automatic_rerun_after_results": False,
            },
        }
        report = verify_confirmatory_run_authorization(
            authorization,
            preflight=self.preflight,
        )
        self.assertTrue(report["valid"])
        authorization["preflight_digest_sha256"] = "0" * 64
        changed = verify_confirmatory_run_authorization(
            authorization,
            preflight=self.preflight,
        )
        self.assertFalse(changed["valid"])
        self.assertFalse(changed["checks"]["preflight_digest_bound"])


if __name__ == "__main__":
    unittest.main()
