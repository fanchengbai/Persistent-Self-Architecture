from __future__ import annotations

import copy
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
    VERIFICATION_SOURCE_FILES,
    build_d3_static_report,
    probe_installed_rwkv_source,
    validate_d3_config,
)
from psa.self_model.off_only_adapter_manifest import (
    IMPLEMENTATION_CONFIG_FILE,
    build_off_only_adapter_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / VERIFICATION_CONFIG_FILE


def _installed_source(**changes):
    result = {
        "package": "rwkv",
        "version": "0.8.32",
        "model_source_path": "rwkv/model.py",
        "model_source_sha256": (
            "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
        ),
        "model_source_size_bytes": 85425,
        "access_method": "importlib.metadata_and_read_bytes",
    }
    result.update(changes)
    return result


def _d2_report():
    return build_off_only_adapter_report(
        config_path=ROOT / IMPLEMENTATION_CONFIG_FILE,
        project_root=ROOT,
    )


def _report(*, installed_source=None, d2_report=None):
    return build_d3_static_report(
        config_path=CONFIG,
        project_root=ROOT,
        installed_source=installed_source or _installed_source(),
        d2_report=d2_report or _d2_report(),
    )


class D3StaticVerificationTests(unittest.TestCase):
    def test_report_verifies_static_gate_without_model_import(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = _report()
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "d3_cloud_static_verified")
        self.assertEqual(len(report["checks"]), 33)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(report["source_digests"]), 10)
        self.assertEqual(
            set(report["source_digests"]), set(VERIFICATION_SOURCE_FILES)
        )
        digest_payload = dict(report)
        digest_payload.pop("report_digest_sha256")
        self.assertEqual(report["report_digest_sha256"], sha256_json(digest_payload))
        self.assertTrue(report["safety"]["installed_rwkv_source_probed"])
        self.assertFalse(report["safety"]["model_loaded"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["off_g2_implemented"])
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)

    def test_installed_source_mismatches_fail_the_gate(self) -> None:
        cases = (
            {"version": "0.8.33"},
            {"model_source_sha256": "0" * 64},
            {"model_source_size_bytes": 85424},
            {"model_source_path": "rwkv/other.py"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                report = _report(installed_source=_installed_source(**changes))
                self.assertFalse(report["valid"])
                self.assertEqual(
                    report["status"], "d3_cloud_static_verification_failed"
                )

    def test_tampered_d2_report_digest_or_source_fails(self) -> None:
        d2 = _d2_report()
        d2["checks"]["off_g1_only"] = False
        report = _report(d2_report=d2)
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["d2_report_digest_self_valid"])
        self.assertFalse(report["checks"]["d2_checks_complete"])

        d2 = _d2_report()
        first = next(iter(d2["source_digests"]))
        d2["source_digests"][first] = "0" * 64
        payload = dict(d2)
        payload.pop("report_digest_sha256")
        d2["report_digest_sha256"] = sha256_json(payload)
        report = _report(d2_report=d2)
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["d2_report_digest_matches_frozen"])
        self.assertFalse(report["checks"]["d2_source_digests_current"])

    def test_config_escalation_and_alternate_path_fail_closed(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["authority"]["model_execution_authorized"] = True
        with self.assertRaises(PermissionError):
            validate_d3_config(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            alternate = Path(temp_dir) / "d3.json"
            alternate.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d3_static_report(
                    config_path=alternate,
                    project_root=ROOT,
                    installed_source=_installed_source(),
                    d2_report=_d2_report(),
                )

    def test_probe_reads_metadata_and_bytes_without_importing_model(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "rwkv/model.py"
            source.parent.mkdir()
            source_bytes = b"raise AssertionError('must not execute')\n"
            source.write_bytes(source_bytes)

            class Distribution:
                version = "0.8.32"

                @staticmethod
                def locate_file(relative):
                    self.assertEqual(relative, "rwkv/model.py")
                    return root / relative

            with mock.patch(
                "psa.self_model.d3_static_verification.metadata.distribution",
                return_value=Distribution(),
            ):
                facts = probe_installed_rwkv_source()
        self.assertEqual(
            facts["model_source_sha256"], hashlib.sha256(source_bytes).hexdigest()
        )
        self.assertEqual(facts["model_source_size_bytes"], len(source_bytes))
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)

    def test_report_is_deterministic_and_has_no_active_authority(self) -> None:
        first = _report()
        second = _report()
        self.assertEqual(first, second)
        self.assertEqual(
            first["off_gates"],
            {"off_g1_implemented": True, "off_g2_implemented": False},
        )
        for key in (
            "instrumented_runtime_implemented",
            "active_injection_implemented",
            "real_layers_selected",
            "self_effect_experiment_run",
            "automatic_rerun_authorized",
        ):
            self.assertFalse(first["safety"][key])


if __name__ == "__main__":
    unittest.main()
