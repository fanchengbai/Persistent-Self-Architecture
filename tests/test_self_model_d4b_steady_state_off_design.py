from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from psa.self_model.d4b_steady_state_off_design import (
    CONFIG_RELATIVE_PATH,
    ROUTES,
    SCORED_ROUNDS,
    build_d4_call_trace,
    build_d4a_call_trace,
    build_design_report,
    validate_design,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D4BSteadyStateOffDesignTests(unittest.TestCase):
    def test_trace_closure_preserves_missing_d4_warmup_outputs(self):
        d4 = build_d4_call_trace()
        d4a = build_d4a_call_trace()
        self.assertEqual(len(d4), 37)
        self.assertEqual(len(d4a), 9)
        self.assertEqual(d4[0]["phase"], "prefix_snapshot")
        self.assertFalse(d4[0]["output_recorded"])
        failed_cell = [
            call
            for call in d4
            if call["cell_id"] == "forward_one__none__full_output_false"
        ]
        self.assertEqual(len(failed_cell), 6)
        self.assertEqual(sum(call["scored"] for call in failed_cell), 3)
        self.assertTrue(
            all(not call["output_recorded"] for call in failed_cell if not call["scored"])
        )
        self.assertNotIn(
            "recompiled_unmodified", {call["method_family"] for call in d4}
        )
        self.assertEqual(
            [call["observed_cluster"] for call in d4a[:3]],
            ["first_original_transient", "first_g0_transient", "shared_steady"],
        )
        self.assertTrue(
            all(call["observed_cluster"] == "shared_steady" for call in d4a[2:])
        )

    def test_design_freezes_recorded_precondition_and_four_by_four_latin_score(self):
        design = json.loads(CONFIG.read_text(encoding="utf-8"))
        checks = validate_design(design)
        self.assertTrue(all(checks.values()))
        closure = design["diagnostic_closure"]
        self.assertTrue(closure["d4_off_g2_warmup_was_present"])
        self.assertFalse(closure["d4a_reproduced_d4_prefix_and_schedule"])
        self.assertTrue(
            closure["d4b_preconditioning_is_prospective_control_not_causal_fix"]
        )
        self.assertEqual(design["routes"], ROUTES)
        self.assertEqual(design["scored_rounds"], SCORED_ROUNDS)
        for position in range(4):
            self.assertEqual(
                sorted(round_routes[position] for round_routes in SCORED_ROUNDS),
                sorted(ROUTES),
            )
        self.assertEqual(design["within_route_comparison_count"], 24)
        self.assertEqual(design["cross_route_comparison_count"], 96)
        self.assertFalse(design["adaptive_convergence_or_extra_warmup_allowed"])

    def test_design_report_is_offline_and_does_not_authorize_d5(self):
        report = build_design_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["runtime_implemented"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d4_status_changed"])
        self.assertFalse(report["safety"]["d5_authorized"])

    def test_scope_changes_fail_closed(self):
        design = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            for field, value in (
                ("adaptive_convergence_or_extra_warmup_allowed", True),
                ("runtime_implementation_authorized", True),
                ("model_execution_authorized", True),
                ("d5_authorized", True),
                ("comparison", "allclose"),
                ("scored_model_forward_call_count", 15),
            ):
                changed = copy.deepcopy(design)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(PermissionError):
                    validate_design(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
