from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.cli import main


class CliTests(unittest.TestCase):
    def test_task_generate_writes_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dev.json"
            exit_code = main(
                [
                    "task-generate",
                    "--output",
                    str(output),
                    "--count",
                    "4",
                    "--base-seed",
                    "11",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["group_count"], 4)
            self.assertTrue(payload["validation"]["valid"])

    def test_task_generate_accepts_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = project_root / "configs" / "tasks" / "exp001_identity_goal.dev.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "configured.json"
            exit_code = main(
                [
                    "task-generate",
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["group_count"], 8)
            self.assertEqual(payload["base_seed"], 20260729)
            self.assertEqual(
                payload["source_config"]["filename"],
                "exp001_identity_goal.dev.json",
            )
            self.assertEqual(len(payload["source_config"]["sha256"]), 64)
            self.assertEqual(len(payload["dataset_digest_sha256"]), 64)

    def test_model_interface_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_interface_gate",
                side_effect=RuntimeError("Numpy is not available"),
            ):
                exit_code = main(
                    [
                        "model-interface-gate",
                        "--config",
                        "configs/models/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl1_model_interface")
            self.assertEqual(report["exception_type"], "RuntimeError")
            self.assertEqual(report["message"], "Numpy is not available")
            self.assertTrue(report["development_only"])

    def test_successful_model_interface_gate_removes_stale_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            output_dir.mkdir()
            failure_path = output_dir / "failure_report.json"
            failure_path.write_text('{"stale":true}', encoding="utf-8")
            with patch(
                "psa.cli.run_interface_gate",
                return_value={"valid": True},
            ):
                exit_code = main(
                    [
                        "model-interface-gate",
                        "--config",
                        "configs/models/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(failure_path.exists())

    def test_checkpoint_gate_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_checkpoint_roundtrip_gate",
                side_effect=RuntimeError("child probe failed"),
            ):
                exit_code = main(
                    [
                        "checkpoint-roundtrip-gate",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl2_checkpoint_roundtrip")
            self.assertEqual(report["message"], "child probe failed")

    def test_state_operations_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_state_operations_gate",
                side_effect=RuntimeError("state operation failed"),
            ):
                exit_code = main(
                    [
                        "state-operations-gate",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl2b_state_operations")
            self.assertEqual(report["message"], "state operation failed")

    def test_reset_stability_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_reset_stability_diagnostic",
                side_effect=RuntimeError("reset diagnostic failed"),
            ):
                exit_code = main(
                    [
                        "reset-stability-diagnostic",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["gate"], "impl3na_g1h_2_9b_reset_stability"
            )
            self.assertEqual(report["message"], "reset diagnostic failed")

    def test_random_state_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_random_state_gate",
                side_effect=RuntimeError("random state failed"),
            ):
                exit_code = main(
                    [
                        "random-state-gate",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl2c_random_matched")
            self.assertEqual(report["message"], "random state failed")

    def test_impl3_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_impl3_development_gate",
                side_effect=RuntimeError("Batch 0 evidence is missing"),
            ):
                exit_code = main(
                    [
                        "impl3-development-gate",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl3_development")
            self.assertEqual(report["message"], "Batch 0 evidence is missing")

    def test_capability_ladder_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_capability_ladder_gate",
                side_effect=RuntimeError("v0.2 evidence is missing"),
            ):
                exit_code = main(
                    [
                        "capability-ladder-gate",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl3b_capability_ladder")
            self.assertEqual(report["message"], "v0.2 evidence is missing")

    def test_g1_capability_ladder_failure_writes_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            with patch(
                "psa.cli.run_g1_capability_ladder_gate",
                side_effect=RuntimeError("G1 interface evidence is missing"),
            ):
                exit_code = main(
                    [
                        "g1-capability-ladder-gate",
                        "--config",
                        "configs/models/test.json",
                        "--gate-config",
                        "configs/gates/test.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            report = json.loads(
                (output_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["gate"], "impl3d_g1_capability_ladder")
            self.assertEqual(report["message"], "G1 interface evidence is missing")


if __name__ == "__main__":
    unittest.main()
