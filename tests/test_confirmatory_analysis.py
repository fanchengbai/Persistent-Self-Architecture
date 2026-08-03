from __future__ import annotations

import copy
import unittest

from psa.confirmatory.analysis import (
    analyze_confirmatory_group,
    summarize_confirmatory_groups,
)
from psa.confirmatory.runner import (
    CONDITIONS,
    build_condition_plan,
    build_non_core_development_fixture,
)
from psa.evaluation import group_contrasts


COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _scores_for_source(source: tuple[int, int] | None) -> dict[tuple[int, int], float]:
    if source is None:
        return {combo: 0.0 for combo in COMBOS}
    return {
        combo: 2.0 * float(combo[0] == source[0])
        + 2.0 * float(combo[1] == source[1])
        for combo in COMBOS
    }


def _clean_payload(group: dict, *, code_bias: bool = False) -> dict:
    records = []
    for trial in group["trials"]:
        mapping = {
            (option["identity"], option["goal"]): option["code"]
            for option in trial["option_mapping"]
        }
        for condition in CONDITIONS:
            plan = build_condition_plan(trial, condition)
            source_value = plan["state_source_combo"]
            source = tuple(source_value) if source_value is not None else None
            semantic_scores = _scores_for_source(source)
            scores = {
                mapping[combo]: value
                + ({"A": 0.9, "B": 0.3, "C": -0.3, "D": -0.9}[mapping[combo]] if code_bias else 0.0)
                for combo, value in semantic_scores.items()
            }
            records.append(
                {
                    "trial_id": trial["trial_id"],
                    **plan,
                    "option_scores": scores,
                    "metadata": {},
                }
            )
    return {
        "factorial_group_id": group["factorial_group_id"],
        "record_count": len(records),
        "records": records,
    }


def _analysis_config() -> dict:
    return {
        "primary_endpoints": {
            "E1_identity_transfer": "identity_transfer",
            "E2_goal_transfer": "goal_transfer",
            "E3_joint_binding": "mean_joint_margin",
        },
        "bootstrap": {
            "confidence": 0.95,
            "replicates": 200,
            "seed": 1969423784,
            "method": "BCa_with_percentile_fallback",
        },
        "permutation": {
            "alternative": "greater",
            "replicates": 200,
            "seed": 2721869949,
            "method": "paired_group_level_sign_flip",
        },
        "multiple_comparison": {"familywise_alpha": 0.05},
        "sesoi": {
            "E1_identity_transfer": 0.5,
            "E2_goal_transfer": 0.5,
            "E3_joint_binding": 0.5,
            "joint_accuracy_lower_bound": 0.6,
            "identity_specificity": 0.25,
            "goal_specificity": 0.25,
            "baseline_joint_margin_advantage": 0.5,
            "prompt_normalized_retention": 0.2,
        },
        "engineering_gates": {"restored_option_score_max_abs_error": 0.0625},
        "frozen_design_gaps": {
            "matched_context": "not_collected",
            "controls": "not_collected",
        },
    }


class ConfirmatoryAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.group = build_non_core_development_fixture()["groups"][0]

    def test_complete_rotation_removes_answer_code_bias(self) -> None:
        report = analyze_confirmatory_group(
            self.group,
            _clean_payload(self.group, code_bias=True),
        )
        self.assertGreater(report["primary"]["identity_transfer"], 3.0)
        self.assertGreater(report["primary"]["goal_transfer"], 3.0)
        self.assertEqual(
            report["condition_metrics"]["continuous"]["joint_accuracy"],
            1.0,
        )
        self.assertEqual(
            report["condition_metrics"]["reset"]["joint_accuracy"],
            0.25,
        )

    def test_swaps_follow_donor_and_restore_matches(self) -> None:
        report = analyze_confirmatory_group(self.group, _clean_payload(self.group))
        self.assertEqual(report["restore_fidelity"]["option_score_max_abs_error"], 0.0)
        self.assertEqual(report["restore_fidelity"]["semantic_argmax_match_rate"], 1.0)
        for condition in ("swapped_I", "swapped_G", "swapped_both"):
            self.assertEqual(report["swap"][condition]["donor_joint_accuracy"], 1.0)
            self.assertGreater(
                report["swap"][condition]["donor_over_query_joint_margin"],
                0.0,
            )

    def test_clean_synthetic_effects_do_not_hide_missing_frozen_controls(self) -> None:
        base = analyze_confirmatory_group(self.group, _clean_payload(self.group))
        prompt_scores = {
            combo: _scores_for_source(combo) for combo in COMBOS
        }
        base["prompt_visible_contrasts"] = group_contrasts(prompt_scores)
        groups = []
        for index in range(8):
            item = copy.deepcopy(base)
            item["factorial_group_id"] = f"synthetic-{index}"
            groups.append(item)
        summary = summarize_confirmatory_groups(groups, _analysis_config())
        self.assertTrue(summary["measured_decisions"]["primary_endpoints_all_supported"])
        self.assertTrue(
            summary["measured_decisions"][
                "joint_binding_measured_requirements_supported"
            ]
        )
        self.assertEqual(
            summary["gate_2_single_variable_causal_transfer"]["status"],
            "not_assessable_no_full_go",
        )
        self.assertEqual(
            summary["gate_4_native_state_carrier_qualification"]["status"],
            "not_assessable_no_full_go",
        )

    def test_incomplete_rotation_is_rejected(self) -> None:
        payload = _clean_payload(self.group)
        payload["records"] = payload["records"][:-1]
        with self.assertRaisesRegex(ValueError, "128 raw records"):
            analyze_confirmatory_group(self.group, payload)


if __name__ == "__main__":
    unittest.main()
