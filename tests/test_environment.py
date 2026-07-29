from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.environment import collect_environment


class EnvironmentReportTests(unittest.TestCase):
    def test_missing_torch_produces_invalid_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "psa.environment.import_module",
                    side_effect=ImportError("torch is absent"),
                ),
                patch(
                    "psa.environment._package_version",
                    return_value=None,
                ),
                patch(
                    "psa.environment._run_text",
                    return_value=None,
                ),
            ):
                report = collect_environment(Path(directory))

        self.assertFalse(report["valid"])
        self.assertFalse(report["torch"]["available"])
        self.assertFalse(report["checks"]["cuda_available"])
        self.assertFalse(report["checks"]["rwkv_version_pinned"])

    def test_report_contains_no_environment_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "psa.environment.import_module",
                    side_effect=ImportError("torch is absent"),
                ),
                patch(
                    "psa.environment._package_version",
                    return_value=None,
                ),
                patch(
                    "psa.environment._run_text",
                    return_value=None,
                ),
                patch.dict(
                    "os.environ",
                    {"HF_TOKEN": "must-not-appear"},
                ),
            ):
                report = collect_environment(Path(directory))

        self.assertNotIn("HF_TOKEN", str(report))
        self.assertNotIn("must-not-appear", str(report))
