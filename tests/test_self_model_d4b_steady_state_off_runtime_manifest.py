from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from psa.self_model.d4b_steady_state_off_runtime_manifest import (
    RUNTIME_CONFIG,
    RUNTIME_SOURCE,
    SOURCE_FILES,
    build_d4b_runtime_report,
    validate_d4b_runtime_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / RUNTIME_CONFIG
DESIGN = ROOT / "configs/development/self_model_v0_1_d4b_steady_state_off_design.json"


class D4BSteadyStateOffRuntimeManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.design = json.loads(DESIGN.read_text(encoding="utf-8"))
        self.runtime_source = (ROOT / RUNTIME_SOURCE).read_text(encoding="utf-8")

    def test_static_report_locks_sources_and_keeps_real_entry_absent(self):
        report = build_d4b_runtime_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(set(report["source_digests"]), set(SOURCE_FILES))
        self.assertTrue(report["safety"]["runtime_core_implemented"])
        self.assertFalse(report["safety"]["real_execution_entry_implemented"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["execution_claim_created"])
        self.assertFalse(report["safety"]["d5_authorized"])

    def test_runtime_config_validates_exact_frozen_scope(self):
        checks = validate_d4b_runtime_config(
            config=self.config,
            design=self.design,
            runtime_source=self.runtime_source,
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(self.config["implementation"]["total_forward_call_count"], 21)
        self.assertEqual(
            self.config["implementation"]["cross_route_comparison_count"], 96
        )

    def test_authority_schedule_and_comparison_changes_fail_closed(self):
        changes = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d5_authorized", True),
            ("implementation", "adaptive_or_extra_calls_allowed", True),
            ("implementation", "comparison", "allclose"),
            ("implementation", "total_forward_call_count", 22),
            ("implementation", "real_model_entry_present", True),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(section=section, field=field):
                with self.assertRaises(PermissionError):
                    validate_d4b_runtime_config(
                        config=changed,
                        design=self.design,
                        runtime_source=self.runtime_source,
                    )


if __name__ == "__main__":
    unittest.main()
