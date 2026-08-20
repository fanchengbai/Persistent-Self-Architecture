from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.self_model.d4a_failure_diagnostic_manifest import (
    IMPLEMENTATION_CONFIG,
    build_d4a_runtime_report,
    validate_d4a_runtime_config,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/development/self_model_v0_1_d4a_failure_diagnostic_design.json"
RUNTIME = ROOT / "src/psa/self_model/d4a_failure_diagnostic_runtime.py"


class D4AFailureDiagnosticManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / IMPLEMENTATION_CONFIG).read_text(encoding="utf-8"))
        self.design = json.loads(DESIGN.read_text(encoding="utf-8"))
        self.source = RUNTIME.read_text(encoding="utf-8")

    def test_manifest_is_valid_without_model_or_torch_import(self):
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_d4a_runtime_report(
            config_path=ROOT / IMPLEMENTATION_CONFIG, project_root=ROOT
        )
        stored = report.pop("report_digest_sha256")
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(stored, sha256_json(report))
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)
        self.assertTrue(report["safety"]["diagnostic_runtime_implemented"])
        self.assertFalse(report["safety"]["real_execution_entry_implemented"])
        self.assertFalse(report["safety"]["model_executed"])

    def test_authority_and_call_expansion_fail_closed(self):
        for field, value in (
            ("model_execution_authorized", True),
            ("active_injection_authorized", True),
            ("automatic_rerun_authorized", True),
        ):
            changed = copy.deepcopy(self.config)
            changed["authority"][field] = value
            with self.assertRaises(PermissionError):
                validate_d4a_runtime_config(
                    config=changed, design=self.design, runtime_source=self.source
                )
        changed = copy.deepcopy(self.config)
        changed["implementation"]["model_forward_call_count"] = 10
        with self.assertRaises(PermissionError):
            validate_d4a_runtime_config(
                config=changed, design=self.design, runtime_source=self.source
            )

    def test_alternate_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                (ROOT / IMPLEMENTATION_CONFIG).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaises(PermissionError):
                build_d4a_runtime_report(config_path=path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
