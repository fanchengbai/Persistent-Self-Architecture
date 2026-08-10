from __future__ import annotations

import json
from pathlib import Path
import unittest

from psa.artifacts import sha256_file
from psa.supplemental.analysis import (
    EXPECTED_ANALYSIS_CONFIG_SHA256,
    analyze_control_records,
    summarize_supplemental_analysis,
)


def _config() -> dict:
    return {
        "bootstrap": {
            "confidence": 0.95,
            "replicates": 200,
            "seed": 7,
            "method": "BCa_with_percentile_fallback",
        },
        "permutation": {
            "alternative": "greater",
            "replicates": 500,
            "seed": 11,
            "method": "paired_group_level_sign_flip",
        },
        "matched_context": {
            "mean_joint_margin_advantage_sesoi": 0.5,
            "confidence_interval_lower_must_exceed": 0.0,
            "one_sided_p_below": 0.05,
            "require_zero_state_norm_alerts": True,
        },
        "generation": {
            "minimum_format_valid_rate": 0.99,
            "minimum_joint_accuracy_ci_lower": 0.8,
            "minimum_identity_accuracy_ci_lower": 0.9,
            "minimum_goal_accuracy_ci_lower": 0.9,
            "maximum_answer_position_accuracy_gap": 0.25,
            "require_forced_prefix_greedy_exact_rate": 1.0,
        },
    }


def _prefix() -> dict:
    return {"text": ">\n", "greedy_exact": True, "roundtrip_exact": True}


class Exp001BSupplementalAnalysisTests(unittest.TestCase):
    def test_analysis_config_is_pinned_to_formal_preregistration(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "analysis" / "exp001b_supplemental_v1.json"
        candidate_path = (
            root
            / "preregistration"
            / "exp001b"
            / "final_v1"
            / "preregistration_candidate.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        statistics = candidate["locked_design"]["statistics"]
        parent_digest = candidate["parent_evidence"][
            "raw_group_payload_digest_sha256"
        ]
        self.assertEqual(sha256_file(config_path), EXPECTED_ANALYSIS_CONFIG_SHA256)
        self.assertEqual(config["bootstrap"], statistics["bootstrap"])
        self.assertEqual(config["permutation"], statistics["permutation"])
        self.assertEqual(
            config["expected_parent_raw_payload_digest_sha256"], parent_digest
        )

    def test_control_alerts_use_prompt_visible_reset_baseline(self) -> None:
        rows = []
        for task in ("copy", "single", "two"):
            for condition in (
                "continuous",
                "restored",
                "reset",
                "random_matched",
                "swapped_I",
                "swapped_G",
                "swapped_both",
                "prompt_visible_reset",
            ):
                for index in range(32):
                    target = "ABCD"[index % 4]
                    scores = {code: -2.0 for code in "ABCD"}
                    scores[target] = -0.1
                    rows.append(
                        (
                            {"condition": condition, "task_type": task, "target_code": target},
                            {"option_scores": scores, "metadata": {"forced_prefix": _prefix()}},
                        )
                    )
        report = analyze_control_records(
            rows,
            {
                "baseline_condition": "prompt_visible_reset",
                "maximum_accuracy_drop": 0.05,
                "maximum_prefix_format_drop": 0.02,
                "maximum_target_log_probability_drop": 0.25,
            },
        )
        self.assertTrue(report["measured_alerts_pass"])
        self.assertFalse(report["required_diagnostics_complete"])
        self.assertFalse(report["full_control_gate"])

    def test_missing_control_diagnostics_prevent_full_go(self) -> None:
        groups = []
        for index in range(12):
            groups.append(
                {
                    "matched_context": {
                        "matched_joint_margin": 0.0,
                        "continuous_minus_matched_joint_margin": 1.0 + index / 100,
                        "state_norm_alert_count": 0,
                    },
                    "generation": {
                        "format_valid": 1.0,
                        "prefix_valid": 1.0,
                        "joint_correct": 1.0,
                        "identity_correct": 1.0,
                        "goal_correct": 1.0,
                    },
                    "generation_position": {
                        code: {"count": 4, "accuracy": 1.0} for code in "ABCD"
                    },
                }
            )
        report = summarize_supplemental_analysis(
            groups,
            {
                "measured_alerts_pass": True,
                "required_diagnostics_complete": False,
            },
            _config(),
        )
        self.assertTrue(report["measured_supplemental_package_go"])
        self.assertEqual(
            report["gate_4_native_state_carrier_qualification"]["status"],
            "not_assessable_no_full_go",
        )
        self.assertEqual(
            report["route_decision"],
            "hold_phase_2_missing_frozen_control_diagnostics",
        )

    def test_missing_parent_reference_reports_partial_analysis(self) -> None:
        groups = []
        for index in range(12):
            groups.append(
                {
                    "matched_context": {
                        "matched_joint_margin": -0.5 + index / 100,
                        "continuous_minus_matched_joint_margin": None,
                        "state_norm_alert_count": 0,
                    },
                    "generation": {
                        "format_valid": 1.0,
                        "prefix_valid": 1.0,
                        "joint_correct": 1.0,
                        "identity_correct": 1.0,
                        "goal_correct": 1.0,
                    },
                    "generation_position": {
                        code: {"count": 4, "accuracy": 1.0} for code in "ABCD"
                    },
                }
            )
        report = summarize_supplemental_analysis(
            groups,
            {
                "measured_alerts_pass": True,
                "required_diagnostics_complete": False,
            },
            _config(),
        )
        self.assertFalse(report["matched_context_assessable"])
        self.assertFalse(report["measured_supplemental_package_go"])
        self.assertEqual(
            report["matched_context"]["status"],
            "not_assessable_missing_parent_reference",
        )
        self.assertEqual(
            report["route_decision"], "hold_phase_2_missing_parent_reference"
        )
        self.assertEqual(
            report["gate_2_single_variable_causal_transfer"]["status"],
            "not_assessable_no_full_go",
        )

    def test_measured_generation_failure_routes_to_review(self) -> None:
        groups = []
        for index in range(12):
            groups.append(
                {
                    "matched_context": {
                        "matched_joint_margin": 0.0,
                        "continuous_minus_matched_joint_margin": 1.0 + index / 100,
                        "state_norm_alert_count": 0,
                    },
                    "generation": {
                        "format_valid": 0.5,
                        "prefix_valid": 1.0,
                        "joint_correct": 0.5,
                        "identity_correct": 0.75,
                        "goal_correct": 0.75,
                    },
                    "generation_position": {
                        code: {"count": 4, "accuracy": 0.5} for code in "ABCD"
                    },
                }
            )
        report = summarize_supplemental_analysis(
            groups,
            {
                "measured_alerts_pass": True,
                "required_diagnostics_complete": False,
            },
            _config(),
        )
        self.assertFalse(report["measured_supplemental_package_go"])
        self.assertEqual(report["route_decision"], "review_frozen_failures_without_rerun")

    def test_observed_failure_precedes_missing_parent_reference(self) -> None:
        groups = []
        for index in range(12):
            groups.append(
                {
                    "matched_context": {
                        "matched_joint_margin": -0.5 + index / 100,
                        "continuous_minus_matched_joint_margin": None,
                        "state_norm_alert_count": 1 if index == 0 else 0,
                    },
                    "generation": {
                        "format_valid": 1.0,
                        "prefix_valid": 1.0,
                        "joint_correct": 1.0,
                        "identity_correct": 1.0,
                        "goal_correct": 1.0,
                    },
                    "generation_position": {
                        code: {"count": 4, "accuracy": 1.0} for code in "ABCD"
                    },
                }
            )
        report = summarize_supplemental_analysis(
            groups,
            {
                "measured_alerts_pass": True,
                "required_diagnostics_complete": False,
            },
            _config(),
        )
        self.assertFalse(report["matched_context_assessable"])
        self.assertFalse(report["measured_observed_components_go"])
        self.assertEqual(
            report["route_decision"], "review_frozen_failures_without_rerun"
        )
        self.assertEqual(
            report["gate_2_single_variable_causal_transfer"]["status"],
            "revise_or_stop_measured_supplemental_control_failure",
        )


if __name__ == "__main__":
    unittest.main()
