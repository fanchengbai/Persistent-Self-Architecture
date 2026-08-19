from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from psa.artifacts import sha256_json
from psa.self_model.d3_static_verification import (
    VERIFICATION_CONFIG_FILE,
    build_d3_static_report,
)
from psa.self_model.instrumented_off_manifest import (
    IMPLEMENTATION_CONFIG_FILE,
    IMPLEMENTATION_SOURCE_FILES,
    build_instrumented_off_report,
    probe_installed_rwkv_source,
    validate_instrumented_off_config,
)
from psa.self_model.off_only_adapter_manifest import (
    IMPLEMENTATION_CONFIG_FILE as D2_CONFIG_FILE,
    build_off_only_adapter_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / IMPLEMENTATION_CONFIG_FILE
EXPECTED_UPSTREAM_DIGEST = (
    "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
)
SYNTHETIC_UPSTREAM = b"""
class RWKV_x070:
    def forward_one(self, idx, state):
        x = idx
        for i in range(2):
            xx, state[i*3+2] = RWKV_x070_CMix(x, state[i*3+2], i)
            x = x + xx
        return x, state

    def forward_seq(self, idx, state, full_output=False):
        x = idx
        for i in range(2):
            xx, state[i*3+2] = RWKV_x070_CMix(x, state[i*3+2], i)
            x = x + xx
        return x, state
"""
SYNTHETIC_UPSTREAM += b"#" * (85425 - len(SYNTHETIC_UPSTREAM))
_REAL_SHA256 = hashlib.sha256


def _installed(**changes):
    result = {
        "package": "rwkv",
        "version": "0.8.32",
        "model_source_path": "rwkv/model.py",
        "model_source_sha256": EXPECTED_UPSTREAM_DIGEST,
        "model_source_size_bytes": 85425,
        "access_method": "importlib.metadata_and_read_bytes",
    }
    result.update(changes)
    return result


class _SyntheticDigest:
    def hexdigest(self):
        return EXPECTED_UPSTREAM_DIGEST


def _sha256_with_locked_synthetic(data=b""):
    if data == SYNTHETIC_UPSTREAM:
        return _SyntheticDigest()
    return _REAL_SHA256(data)


def _d3_report():
    d2 = build_off_only_adapter_report(
        config_path=ROOT / D2_CONFIG_FILE,
        project_root=ROOT,
    )
    report = build_d3_static_report(
        config_path=ROOT / VERIFICATION_CONFIG_FILE,
        project_root=ROOT,
        installed_source=_installed(),
        d2_report=d2,
    )
    if report["report_digest_sha256"] != (
        "fcb8dfeb58863c9bc5e6c02b8151fb581df5dd50c4158f164005560680c42918"
    ):
        raise AssertionError("local D3 prerequisite no longer matches cloud evidence")
    return report


def _report(*, installed=None, d3_report=None, source=SYNTHETIC_UPSTREAM):
    with mock.patch(
        "psa.self_model.instrumented_off_manifest.hashlib.sha256",
        side_effect=_sha256_with_locked_synthetic,
    ):
        return build_instrumented_off_report(
            config_path=CONFIG,
            project_root=ROOT,
            installed_source=installed or _installed(),
            upstream_source_bytes=source,
            d3_report=d3_report or _d3_report(),
        )


class InstrumentedOffManifestTests(unittest.TestCase):
    def test_report_verifies_implementation_without_model_import(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = _report()
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["status"], "instrumented_off_runtime_static_verified"
        )
        self.assertEqual(len(report["checks"]), 41)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(report["source_digests"]), 10)
        self.assertEqual(
            set(report["source_digests"]), set(IMPLEMENTATION_SOURCE_FILES)
        )
        payload = dict(report)
        payload.pop("report_digest_sha256")
        self.assertEqual(report["report_digest_sha256"], sha256_json(payload))
        self.assertEqual(
            report["transformation"]["injection_counts"],
            {"forward_one": 1, "forward_seq": 1},
        )
        self.assertTrue(report["safety"]["instrumented_runtime_implemented"])
        self.assertTrue(report["safety"]["off_g2_implemented"])
        self.assertFalse(report["safety"]["real_model_equivalence_executed"])
        self.assertFalse(report["safety"]["active_injection_implemented"])
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)

    def test_installed_fact_mismatch_fails_report(self) -> None:
        for change in (
            {"version": "0.8.33"},
            {"model_source_sha256": "0" * 64},
            {"model_source_size_bytes": 85424},
        ):
            with self.subTest(change=change):
                report = _report(installed=_installed(**change))
                self.assertFalse(report["valid"])

    def test_upstream_structure_change_fails_closed(self) -> None:
        dirty = SYNTHETIC_UPSTREAM.replace(b"x = x + xx", b"x = xx + x", 1)
        with self.assertRaises(RuntimeError):
            _report(source=dirty)

    def test_tampered_d3_report_fails_prerequisite(self) -> None:
        d3 = _d3_report()
        d3["checks"]["installed_version_matches"] = False
        report = _report(d3_report=d3)
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["d3_report_digest_self_valid"])
        self.assertFalse(report["checks"]["d3_checks_complete"])

    def test_authority_escalation_and_alternate_path_fail_closed(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["authority"]["model_execution_authorized"] = True
        with self.assertRaises(PermissionError):
            validate_instrumented_off_config(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            alternate = Path(temp_dir) / "config.json"
            alternate.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_instrumented_off_report(
                    config_path=alternate,
                    project_root=ROOT,
                    installed_source=_installed(),
                    upstream_source_bytes=SYNTHETIC_UPSTREAM,
                    d3_report=_d3_report(),
                )

    def test_probe_reads_bytes_without_importing_model(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "rwkv/model.py"
            source.parent.mkdir()
            source.write_bytes(SYNTHETIC_UPSTREAM)

            class Distribution:
                version = "0.8.32"

                @staticmethod
                def locate_file(relative):
                    return root / relative

            with mock.patch(
                "psa.self_model.instrumented_off_manifest.metadata.distribution",
                return_value=Distribution(),
            ):
                facts, source_bytes = probe_installed_rwkv_source()
        self.assertEqual(source_bytes, SYNTHETIC_UPSTREAM)
        self.assertEqual(
            facts["model_source_sha256"],
            _REAL_SHA256(SYNTHETIC_UPSTREAM).hexdigest(),
        )
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)


if __name__ == "__main__":
    unittest.main()
