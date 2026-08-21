from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.self_model.d5_active_injection_design import (
    CONFIG_RELATIVE_PATH,
    GATE_IDS,
    REQUIRED_NEXT_CONFIRMATION,
    REQUIRED_CONTROLS,
    build_design_report,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class CouplingD5ActiveDesignTests(unittest.TestCase):
    def test_design_disambiguates_coupling_gate_from_self_updater(self) -> None:
        design = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_design(design)
        self.assertTrue(all(checks.values()))
        names = design["nomenclature"]
        self.assertEqual(names["workflow_gate_id"], "Coupling-D5")
        self.assertEqual(names["architecture_decision_id"], "Architecture-D5-Self-Updater")
        self.assertFalse(names["same_gate"])
        self.assertFalse(names["self_updater_in_scope"])

    def test_gate_ladder_separates_mechanism_and_effect_claims(self) -> None:
        design = json.loads(CONFIG.read_text(encoding="utf-8"))
        ladder = design["gate_ladder"]
        self.assertEqual([item["gate_id"] for item in ladder], GATE_IDS)
        self.assertEqual(
            [item["effect_claim_allowed"] for item in ladder],
            [False, False, False, "development_only", "preregistered_only"],
        )
        self.assertEqual(design["future_noncore_effect_controls"], REQUIRED_CONTROLS)
        self.assertFalse(design["selection_boundaries"]["formal_test_set_for_selection"])
        self.assertEqual(
            design["required_next_owner_confirmation_text"],
            REQUIRED_NEXT_CONFIRMATION,
        )

    def test_report_is_static_and_digest_is_self_consistent(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_design_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(all(value is False for value in report["safety"].values()))
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)
        digest = report.pop("report_digest_sha256")
        self.assertEqual(digest, sha256_json(report))

    def test_authority_and_scope_changes_fail_closed(self) -> None:
        design = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "d5a_implementation_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "self_updater_authorized", True),
            ("active_contract", "real_layer_indices_selected", True),
            ("active_contract", "off_or_zero_scale_calls_callback", True),
            ("selection_boundaries", "formal_test_set_for_selection", True),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(design)
            changed[section][field] = value
            with self.assertRaises(PermissionError):
                validate_design(changed)

    def test_alternate_config_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_design_report(config_path=path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
