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
            self.assertEqual(len(payload["dataset_digest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

