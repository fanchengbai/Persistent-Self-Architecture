from __future__ import annotations

import unittest

from psa.development import (
    calibrate_standard_delay,
    evaluate_prompt_visible,
    inspect_dataset_tokenization,
    inspect_label_pairs,
)
from psa.tasks import generate_dataset, generate_factorial_group


class ByteTokenizerAdapter:
    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class Impl3DevelopmentTests(unittest.TestCase):
    def test_label_selection_uses_declared_order_and_token_rules(self) -> None:
        report = inspect_label_pairs(
            ByteTokenizerAdapter(),
            (("dax", "kel"), ("long", "tiny"), ("mip", "rov")),
            selected_pair_count=2,
            max_tokens_per_form=4,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["selected_pairs"],
            [["dax", "kel"], ["mip", "rov"]],
        )
        self.assertFalse(report["pairs"][1]["eligible"])

    def test_delay_calibration_depends_only_on_token_count(self) -> None:
        adapter = ByteTokenizerAdapter()
        two_units = generate_factorial_group(
            group_seed=0,
            delay_units=2,
        ).trajectories[0].common_suffix
        report = calibrate_standard_delay(
            adapter,
            track="synthetic",
            target_token_count=len(adapter.encode(two_units)),
            max_absolute_error=0,
            max_units=4,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["selected_delay_units"], 2)

    def test_prompt_visible_evaluation_passes_ideal_group_records(self) -> None:
        groups = generate_dataset(group_count=4, base_seed=17)
        records = []
        for group in groups:
            for sample in group.trajectories:
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "status": "success",
                        "argmax_choice": sample.correct_code,
                        "format_valid": True,
                    }
                )
        report = evaluate_prompt_visible(
            groups,
            records,
            bootstrap_replicates=500,
            bootstrap_seed=11,
            thresholds={
                "joint_lower_bound": 0.8,
                "marginal_lower_bound": 0.9,
                "format_valid_rate": 0.99,
                "max_answer_position_accuracy_gap": 0.25,
                "max_infrastructure_failure_rate": 0.01,
            },
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["metrics"]["joint_accuracy"], 1.0)
        self.assertEqual(
            report["metrics"]["joint_accuracy_interval"],
            [1.0, 1.0],
        )

    def test_dataset_tokenization_is_balanced_with_equal_length_labels(self) -> None:
        groups = generate_dataset(group_count=4, base_seed=19)
        report = inspect_dataset_tokenization(ByteTokenizerAdapter(), groups)
        self.assertTrue(report["valid"])
        self.assertTrue(
            all(group["prompt_token_balanced"] for group in report["groups"])
        )


if __name__ == "__main__":
    unittest.main()
