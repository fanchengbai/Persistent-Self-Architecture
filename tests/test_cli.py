from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
