from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

from psa.artifacts import sha256_json
from psa.self_model.d4a_failure_diagnostic_design import (
    DESIGN_CONFIG,
    build_d4a_design_report,
    validate_d4a_design,
)


ROOT = Path(__file__).resolve().parents[1]


class D4AFailureDiagnosticDesignTests(unittest.TestCase):
    def setUp(self):
        self.design = json.loads((ROOT / DESIGN_CONFIG).read_text(encoding="utf-8"))
        self.d4_source = (
            ROOT / "src/psa/self_model/d4_real_off_equivalence.py"
        ).read_text(encoding="utf-8")
        self.g2_source = (
            ROOT / "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
        ).read_text(encoding="utf-8")

    def test_static_design_report_is_valid_and_self_digest_matches(self):
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_d4a_design_report(
            config_path=ROOT / DESIGN_CONFIG, project_root=ROOT
        )
        stored = report.pop("report_digest_sha256")
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(stored, sha256_json(report))
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["diagnostic_runtime_implemented"])

    def test_latin_schedule_balances_routes_without_discarded_warmup(self):
        diagnostic = self.design["minimal_future_diagnostic"]
        rounds = diagnostic["recorded_rounds"]
        self.assertEqual(len(rounds), 3)
        for route in diagnostic["routes"]:
            self.assertEqual(sorted(round_.index(route) for round_ in rounds), [0, 1, 2])
        self.assertEqual(diagnostic["model_forward_call_count"], 9)
        self.assertEqual(diagnostic["discarded_warmup_call_count"], 0)

    def test_authority_or_scope_expansion_fails_closed(self):
        for mutate in (
            lambda value: value["authority"].__setitem__(
                "model_execution_authorized", True
            ),
            lambda value: value["minimal_future_diagnostic"].__setitem__(
                "model_forward_call_count", 10
            ),
            lambda value: value["future_stop_rules"].__setitem__(
                "no_automatic_rerun", False
            ),
        ):
            changed = copy.deepcopy(self.design)
            mutate(changed)
            with self.assertRaises(PermissionError):
                validate_d4a_design(
                    design=changed,
                    d4_source=self.d4_source,
                    off_g2_source=self.g2_source,
                )

    def test_locked_d4_failure_cannot_be_rewritten(self):
        changed = copy.deepcopy(self.design)
        changed["d4_failure_evidence"]["valid"] = True
        with self.assertRaises(PermissionError):
            validate_d4a_design(
                design=changed,
                d4_source=self.d4_source,
                off_g2_source=self.g2_source,
            )


if __name__ == "__main__":
    unittest.main()
