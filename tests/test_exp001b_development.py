from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.cli import build_parser
from psa.supplemental.development import (
    _load_confirmed_design,
    empirical_quantile,
    evaluate_state_norms,
    fit_matched_context_history,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/preregistration/exp001b_supplemental_controls.draft.json"


class Exp001BDevelopmentTests(unittest.TestCase):
    def test_confirmed_design_still_has_no_formal_authority(self) -> None:
        design = _load_confirmed_design(DESIGN)
        self.assertEqual(design["status"], "design_confirmed_development_only")
        self.assertTrue(design["design_review"]["b1_b7_confirmed"])
        self.assertTrue(
            design["design_review"]["does_not_authorize_candidate_freeze"]
        )
        self.assertFalse(
            design["safety_boundary"]["supplemental_experiment_authorized"]
        )

    def test_confirmed_design_rejects_authority_escalation(self) -> None:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        design["safety_boundary"]["supplemental_experiment_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(design), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refuses formal authority"):
                _load_confirmed_design(path)

    def test_matched_context_fitter_reaches_exact_token_count(self) -> None:
        template = {
            "user_text": (
                "UNRELATED LOG. DOMAIN {domain}. OPERATION {operation}. "
                "This log is not a current-state record."
            ),
            "assistant_ack": "Unrelated log ignored.",
        }
        original = " ".join(["original"] * 100)
        filler = " ".join(["neutral"] * 20)
        fitted = fit_matched_context_history(
            original_history=original,
            template=template,
            domain="amber",
            operation="orbit",
            filler=filler,
            padding_fragments=(" x", " note"),
            token_counter=lambda text: len(text.split()),
        )
        self.assertTrue(fitted["token_count_exact"])
        self.assertEqual(fitted["matched_token_count"], 100)
        self.assertIn(filler, fitted["text"])

    def test_empirical_quantile_uses_declared_nearest_rank_rule(self) -> None:
        values = list(range(64))
        self.assertEqual(empirical_quantile(values, 0.0), 0.0)
        self.assertEqual(empirical_quantile(values, 0.5), 31.0)
        self.assertEqual(empirical_quantile(values, 0.999), 63.0)

    def test_state_norm_alert_path_accepts_and_rejects(self) -> None:
        thresholds = {
            "components": [
                {"path": "state[0]", "threshold_rms": 1.0},
                {"path": "state[1]", "threshold_rms": 2.0},
            ]
        }
        safe = evaluate_state_norms(
            [
                {"path": "state[0]", "rms": 1.0},
                {"path": "state[1]", "rms": 1.5},
            ],
            thresholds,
        )
        alert = evaluate_state_norms(
            [
                {"path": "state[0]", "rms": 1.01},
                {"path": "state[1]", "rms": 1.5},
            ],
            thresholds,
        )
        self.assertTrue(safe["valid"])
        self.assertFalse(alert["valid"])
        self.assertEqual(alert["alert_count"], 1)

    def test_cli_exposes_two_non_core_development_gates(self) -> None:
        parser = build_parser()
        bdev1 = parser.parse_args(
            [
                "exp001b-bdev1-gate",
                "--design",
                "design.json",
                "--model-config",
                "model.json",
                "--output-dir",
                "out",
            ]
        )
        bdev2 = parser.parse_args(
            [
                "exp001b-bdev2-gate",
                "--design",
                "design.json",
                "--model-config",
                "model.json",
                "--bdev1-summary",
                "summary.json",
                "--bdev1-thresholds",
                "thresholds.json",
                "--bdev1-matched-report",
                "matched.json",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(bdev1.command, "exp001b-bdev1-gate")
        self.assertEqual(bdev2.command, "exp001b-bdev2-gate")


if __name__ == "__main__":
    unittest.main()
