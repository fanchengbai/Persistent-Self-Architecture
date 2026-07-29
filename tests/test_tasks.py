from __future__ import annotations

from dataclasses import replace
import unittest

from psa.tasks.identity_goal import generate_dataset, generate_factorial_group
from psa.validation import validate_dataset, validate_group


class IdentityGoalTaskTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_factorial_group(group_seed=17)
        second = generate_factorial_group(group_seed=17)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_group_is_complete_and_balanced(self) -> None:
        group = generate_factorial_group(group_seed=31)
        self.assertEqual(len(group.trajectories), 4)
        self.assertEqual(
            {(sample.identity, sample.goal) for sample in group.trajectories},
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertEqual(
            {sample.correct_code for sample in group.trajectories},
            {"A", "B", "C", "D"},
        )
        self.assertEqual(len({sample.query for sample in group.trajectories}), 1)
        self.assertTrue(validate_group(group).valid)

    def test_dataset_cycles_history_order(self) -> None:
        groups = generate_dataset(group_count=8, base_seed=42)
        counts: dict[str, int] = {}
        for group in groups:
            counts[group.history_order] = counts.get(group.history_order, 0) + 1
        self.assertEqual(set(counts.values()), {2})
        self.assertTrue(validate_dataset(groups).valid)

    def test_validator_detects_wrong_answer_mapping(self) -> None:
        group = generate_factorial_group(group_seed=5)
        sample = group.trajectories[0]
        wrong_code = next(
            code for code in ("A", "B", "C", "D") if code != sample.correct_code
        )
        bad_sample = replace(sample, correct_code=wrong_code)
        bad_group = replace(
            group, trajectories=(bad_sample, *group.trajectories[1:])
        )
        report = validate_group(bad_group)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("incorrect answer mapping" in error for error in report.errors)
        )

    def test_natural_track_contains_no_binding_in_query(self) -> None:
        group = generate_factorial_group(
            group_seed=7,
            track="natural",
            identity_labels=("amber", "cyan"),
            goal_labels=("inspect", "seal"),
        )
        report = validate_group(group)
        self.assertTrue(report.valid, report.errors)
        self.assertNotIn("authorized for", group.trajectories[0].query)


if __name__ == "__main__":
    unittest.main()

