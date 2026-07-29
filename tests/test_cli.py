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


if __name__ == "__main__":
    unittest.main()
