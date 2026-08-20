from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from psa.artifacts import sha256_json
from psa.self_model.d4a_cloud_static_verification import (
    VERIFICATION_CONFIG,
    build_d4a_cloud_static_report,
    inspect_d4a_installed_source,
    validate_d4a_cloud_static_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIGEST = "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
METHOD_ONE = """
        @MyFunction
        def forward_one(self, idx, state):
            x = idx
            for i in range(2):
                xx, state[i * 3 + 2] = RWKV_x070_CMix_one(x, state[i * 3 + 2], i)
                x = x + xx
            return x, state
"""
METHOD_SEQ = """
        @MyFunction
        def forward_seq(self, idx, state, full_output=False):
            x = idx
            for i in range(2):
                xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(x, state[i * 3 + 2], i)
                x = x + xx
            return x, state
"""
SYNTHETIC_SOURCE = (
    "class RWKV_x070:\n"
    "    if os.environ.get('RWKV_DE_VERSION') == '1':\n"
    + METHOD_ONE
    + "    else:\n"
    + METHOD_ONE
    + "    if os.environ.get('RWKV_DE_VERSION') == '1':\n"
    + METHOD_SEQ
    + "    else:\n"
    + METHOD_SEQ
)


def _locked_size_source() -> bytes:
    payload = SYNTHETIC_SOURCE.encode("utf-8")
    remaining = 85425 - len(payload)
    if remaining < 2:
        raise AssertionError("synthetic source unexpectedly exceeds lock size")
    return payload + ("#" + "x" * (remaining - 2) + "\n").encode("utf-8")


class D4ACloudStaticVerificationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / VERIFICATION_CONFIG).read_text(encoding="utf-8"))

    def test_real_shape_fixture_aligns_g0_and_g2_variants(self):
        inspection = inspect_d4a_installed_source(SYNTHETIC_SOURCE)
        self.assertTrue(inspection["valid"])
        self.assertTrue(all(inspection["checks"].values()))
        for name in ("forward_one", "forward_seq"):
            g0 = inspection["g0_variant_selection"][name]
            g2 = inspection["g2_variant_selection"][name]
            self.assertEqual(g0["candidate_count"], 2)
            self.assertEqual(g0["original_decorators"], ["MyFunction"])
            self.assertEqual(g0["compiled_decorators"], [])
            self.assertEqual(g0["selected_source_line"], g2["selected_source_line"])

    def test_static_report_passes_without_model_or_torch_import(self):
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        source_bytes = _locked_size_source()
        installed = {
            "package": "rwkv",
            "version": "0.8.32",
            "model_source_path": "rwkv/model.py",
            "model_source_sha256": EXPECTED_DIGEST,
            "model_source_size_bytes": len(source_bytes),
            "access_method": "importlib.metadata_and_read_bytes",
        }
        with mock.patch(
            "psa.self_model.d4a_cloud_static_verification._sha256_bytes",
            return_value=EXPECTED_DIGEST,
        ):
            report = build_d4a_cloud_static_report(
                config_path=ROOT / VERIFICATION_CONFIG,
                project_root=ROOT,
                installed_source=installed,
                upstream_source_bytes=source_bytes,
            )
        stored = report.pop("report_digest_sha256")
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(stored, sha256_json(report))
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["real_execution_entry_implemented"])

    def test_authority_or_scope_expansion_fails_closed(self):
        for section, field, value in (
            ("authority", "model_execution_authorized", True),
            ("authority", "automatic_rerun_authorized", True),
            ("verification", "model_execution_included", True),
        ):
            changed = copy.deepcopy(self.config)
            changed[section][field] = value
            with self.assertRaises(PermissionError):
                validate_d4a_cloud_static_config(changed)

    def test_alternate_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                (ROOT / VERIFICATION_CONFIG).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaises(PermissionError):
                build_d4a_cloud_static_report(
                    config_path=path,
                    project_root=ROOT,
                    installed_source={},
                    upstream_source_bytes=b"",
                )


if __name__ == "__main__":
    unittest.main()
