from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from psa.self_model.real_adapter_off_design import (
    DESIGN_CONFIG_FILE,
    build_real_adapter_off_design_report,
    validate_real_adapter_off_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / DESIGN_CONFIG_FILE
AUDIT_CONFIG = (
    ROOT / "configs/development/self_model_v0_1_rwkv_interface_audit.json"
)
FAKE_CONFIG = (
    ROOT / "configs/development/self_model_v0_1_fake_callback.draft.json"
)
MODEL_CONFIG = ROOT / "configs/models/rwkv7_g1h_2.9b.candidate.json"
SOURCE_DIGEST = "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"


def _object(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _installed(**updates):
    return {
        "package_version": "0.8.32",
        "model_source_path": "/synthetic/site-packages/rwkv/model.py",
        "model_source_sha256": SOURCE_DIGEST,
        "source_size_bytes": 85425,
        **updates,
    }


class RealAdapterOffDesignTests(unittest.TestCase):
    def test_design_report_is_valid_and_no_real_adapter_exists(self) -> None:
        report = build_real_adapter_off_design_report(
            config_path=CONFIG,
            project_root=ROOT,
            installed_source=_installed(),
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(report["checks"]), 23)
        self.assertEqual(len(report["source_digests"]), 9)
        self.assertTrue(all(value is False for value in report["safety"].values()))
        self.assertFalse(
            (ROOT / "src/psa/self_model/rwkv7_coupling_adapter.py").exists()
        )

    def test_upstream_version_or_digest_mismatch_fails_report(self) -> None:
        for installed in (
            _installed(package_version="0.8.33"),
            _installed(model_source_sha256="0" * 64),
        ):
            report = build_real_adapter_off_design_report(
                config_path=CONFIG,
                project_root=ROOT,
                installed_source=installed,
            )
            self.assertFalse(report["valid"])

    def test_alternate_design_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_real_adapter_off_design_report(
                    config_path=path,
                    project_root=ROOT,
                    installed_source=_installed(),
                )

    def test_authority_escalation_fails_closed(self) -> None:
        design = _object(CONFIG)
        design["authority"]["real_adapter_implementation_authorized"] = True
        with self.assertRaises(PermissionError):
            validate_real_adapter_off_design(
                design=design,
                interface_audit=_object(AUDIT_CONFIG),
                fake_callback=_object(FAKE_CONFIG),
                model_config=_object(MODEL_CONFIG),
            )

    def test_real_layer_or_sequence_policy_selection_fails_closed(self) -> None:
        for field, value in (
            ("real_layer_mask", [15]),
            ("real_sequence_policy", "broadcast_all_tokens"),
        ):
            design = _object(CONFIG)
            design["adapter"][field] = value
            with self.assertRaises(PermissionError):
                validate_real_adapter_off_design(
                    design=design,
                    interface_audit=_object(AUDIT_CONFIG),
                    fake_callback=_object(FAKE_CONFIG),
                    model_config=_object(MODEL_CONFIG),
                )

    def test_tolerance_or_top1_cannot_replace_exact_checks(self) -> None:
        design = _object(CONFIG)
        design["future_model_test_protocol"]["required_checks"] = [
            "top1_equal",
            "max_abs_error_below_tolerance",
        ]
        with self.assertRaises(PermissionError):
            validate_real_adapter_off_design(
                design=design,
                interface_audit=_object(AUDIT_CONFIG),
                fake_callback=_object(FAKE_CONFIG),
                model_config=_object(MODEL_CONFIG),
            )

    def test_off_gate_order_and_active_boundary_are_frozen(self) -> None:
        design = _object(CONFIG)
        self.assertEqual(
            [gate["gate_id"] for gate in design["off_equivalence_gates"]],
            ["OFF-G1", "OFF-G2"],
        )
        self.assertFalse(design["adapter"]["active_injection_available"])
        self.assertFalse(
            design["future_model_test_protocol"]["currently_authorized"]
        )
        self.assertEqual(
            design["future_gates"][-1],
            "D5_new_design_and_authorization_before_active_injection",
        )

    def test_prerequisite_mutation_fails_closed(self) -> None:
        audit = copy.deepcopy(_object(AUDIT_CONFIG))
        audit["package"]["model_source_sha256"] = "f" * 64
        with self.assertRaises(PermissionError):
            validate_real_adapter_off_design(
                design=_object(CONFIG),
                interface_audit=audit,
                fake_callback=_object(FAKE_CONFIG),
                model_config=_object(MODEL_CONFIG),
            )


if __name__ == "__main__":
    unittest.main()
