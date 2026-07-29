from __future__ import annotations

import unittest

from psa.evaluation.contrasts import (
    goal_log_odds,
    group_contrasts,
    identity_log_odds,
    joint_margin,
)
from psa.evaluation.resampling import (
    bca_mean_interval,
    equivalence_from_interval,
    holm_adjust,
    sign_flip_test,
)


COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))


def ideal_scores(target: tuple[int, int]) -> dict[tuple[int, int], float]:
    return {combo: (4.0 if combo == target else 0.0) for combo in COMBOS}


class ContrastTests(unittest.TestCase):
    def test_dimension_scores(self) -> None:
        scores = ideal_scores((1, 0))
        self.assertGreater(identity_log_odds(scores), 0.0)
        self.assertLess(goal_log_odds(scores), 0.0)
        self.assertGreater(joint_margin(scores, (1, 0)), 0.0)

    def test_ideal_group_has_directional_transfer(self) -> None:
        states = {combo: ideal_scores(combo) for combo in COMBOS}
        result = group_contrasts(states)
        self.assertGreater(result["identity_transfer"], 0.5)
        self.assertGreater(result["goal_transfer"], 0.5)
        self.assertGreater(result["mean_joint_margin"], 0.5)
        self.assertEqual(result["joint_accuracy"], 1.0)
        self.assertGreater(result["identity_specificity"], 0.0)
        self.assertGreater(result["goal_specificity"], 0.0)

    def test_identity_only_signal_does_not_create_goal_transfer(self) -> None:
        states = {}
        for identity, goal in COMBOS:
            states[(identity, goal)] = {
                combo: (3.0 if combo[0] == identity else 0.0)
                for combo in COMBOS
            }
        result = group_contrasts(states)
        self.assertGreater(result["identity_transfer"], 0.5)
        self.assertAlmostEqual(result["goal_transfer"], 0.0)
        self.assertLessEqual(result["joint_accuracy"], 0.5)


class ResamplingTests(unittest.TestCase):
    def test_holm_adjustment_is_monotonic_in_rank(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_sign_flip_detects_consistently_positive_values(self) -> None:
        p_value = sign_flip_test([1.0] * 10, alternative="greater")
        self.assertLess(p_value, 0.01)

    def test_bca_interval_is_deterministic(self) -> None:
        values = [0.1, 0.2, 0.4, 0.8, 1.6]
        first = bca_mean_interval(values, replicates=500, seed=9)
        second = bca_mean_interval(values, replicates=500, seed=9)
        self.assertEqual(first, second)
        self.assertLess(first[0], sum(values) / len(values))
        self.assertGreater(first[1], sum(values) / len(values))

    def test_equivalence_uses_strict_containment(self) -> None:
        self.assertTrue(equivalence_from_interval((-0.2, 0.3), (-0.5, 0.5)))
        self.assertFalse(equivalence_from_interval((-0.6, 0.3), (-0.5, 0.5)))


if __name__ == "__main__":
    unittest.main()

