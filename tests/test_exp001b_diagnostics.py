from __future__ import annotations

import unittest

from psa.supplemental.diagnostics import (
    classify_control_prefix_failures,
    prefix_failure_flags,
    prefix_token_divergence,
    decode_control_greedy_tokens,
    summarize_control_concordance,
    summarize_control_greedy_tokens,
    summarize_nonrandom_failure_samples,
    summarize_matched_norms,
    summarize_prefix_cells,
)


def _output(*, greedy: bool, roundtrip: bool = True) -> dict:
    return {
        "metadata": {
            "forced_prefix": {
                "text": ">\n",
                "greedy_exact": greedy,
                "roundtrip_exact": roundtrip,
            }
        }
    }


class Exp001BPosthocDiagnosticsTests(unittest.TestCase):
    def test_classify_control_prefix_failures_covers_four_categories(self) -> None:
        cases = (
            ("A", [63, 65], "correct_answer_emitted_immediately"),
            ("B", [63, 65], "wrong_answer_emitted_immediately"),
            ("C", [42, 11], "first_token_corruption"),
            ("D", [63, 99], "other"),
        )
        rows = []
        decoded = {
            (63, 11): ">\n",
            (63, 65): ">A",
            (42, 11): "x\n",
            (63, 99): ">?",
        }
        for index, (target, greedy_ids, _) in enumerate(cases):
            output = _output(greedy=False)
            output["metadata"]["forced_prefix"]["token_ids"] = [63, 11]
            output["metadata"]["forced_prefix"]["greedy_token_ids"] = greedy_ids
            rows.append(
                (
                    {
                        "record_id": f"record-{index}",
                        "source_control_sample_id": f"sample-{index}",
                        "condition": "continuous",
                        "task_type": "copy",
                        "target_code": target,
                        "assigned_source_combo": [index % 2, index // 2],
                        "assigned_factorial_group_id": f"group-{index}",
                    },
                    output,
                )
            )
        report = classify_control_prefix_failures(
            rows, lambda ids: decoded[tuple(ids)]
        )
        self.assertEqual(report["greedy_mismatch_record_count"], 4)
        self.assertTrue(report["classification_complete"])
        self.assertEqual(report["semantic_preserving_format_only_count"], 1)
        self.assertEqual(
            [record["category"] for record in report["records"]],
            [expected for _, _, expected in cases],
        )
        self.assertEqual(len(report["by_assigned_source_combo"]), 4)

    def test_prefix_flags_separate_greedy_and_roundtrip_failures(self) -> None:
        greedy = prefix_failure_flags(_output(greedy=False))
        self.assertTrue(greedy["greedy_mismatch"])
        self.assertFalse(greedy["roundtrip_mismatch"])
        self.assertFalse(greedy["valid"])

    def test_prefix_token_divergence_reports_first_changed_token(self) -> None:
        output = _output(greedy=False)
        output["metadata"]["forced_prefix"]["token_ids"] = [63, 11]
        output["metadata"]["forced_prefix"]["greedy_token_ids"] = [42, 11]
        report = prefix_token_divergence(output)
        self.assertEqual(report["divergence_index"], 0)
        self.assertEqual(report["expected_divergent_token_id"], 63)
        self.assertEqual(report["greedy_divergent_token_id"], 42)

    def test_control_concordance_groups_all_conditions_by_source_sample(self) -> None:
        rows = []
        conditions = tuple(f"condition_{index}" for index in range(8))
        for sample_index in range(2):
            for condition_index, condition in enumerate(conditions):
                rows.append(
                    (
                        {
                            "source_control_sample_id": f"sample-{sample_index}",
                            "condition": condition,
                            "task_type": "copy",
                        },
                        _output(greedy=not (sample_index == 0 and condition_index < 2)),
                    )
                )
        report = summarize_control_concordance(rows)
        self.assertEqual(report["source_sample_count"], 2)
        self.assertEqual(report["samples_with_any_greedy_mismatch"], 1)
        self.assertEqual(report["failed_condition_count_distribution"], {"0": 1, "2": 1})
        self.assertEqual(report["pairwise_failure_overlaps"][0]["overlap_count"], 1)

    def test_control_greedy_tokens_count_repeated_patterns(self) -> None:
        rows = []
        for _ in range(3):
            output = _output(greedy=False)
            output["metadata"]["forced_prefix"]["token_ids"] = [63, 11]
            output["metadata"]["forced_prefix"]["greedy_token_ids"] = [42, 11]
            rows.append(({"condition": "reset", "task_type": "copy"}, output))
        report = summarize_control_greedy_tokens(rows)
        self.assertEqual(report["greedy_mismatch_record_count"], 3)
        self.assertEqual(report["token_patterns"][0]["record_count"], 3)
        self.assertEqual(report["first_divergence_patterns"][0]["greedy_token_id"], 42)

    def test_decode_control_tokens_adds_readable_text(self) -> None:
        report = {
            "greedy_mismatch_record_count": 1,
            "token_patterns": [
                {
                    "condition": "continuous",
                    "task_type": "copy",
                    "expected_token_ids": [63, 11],
                    "greedy_token_ids": [63, 66],
                    "record_count": 1,
                }
            ],
            "first_divergence_patterns": [
                {
                    "divergence_index": 1,
                    "expected_token_id": 11,
                    "greedy_token_id": 66,
                    "record_count": 1,
                }
            ],
        }
        decoded = decode_control_greedy_tokens(
            report, lambda ids: "/".join(str(item) for item in ids)
        )
        self.assertEqual(decoded["token_patterns"][0]["greedy_text"], "63/66")
        self.assertEqual(
            decoded["first_divergence_patterns"][0]["greedy_token_text"], "66"
        )

    def test_nonrandom_failure_samples_exclude_random_only_failures(self) -> None:
        rows = []
        conditions = (
            "continuous",
            "restored",
            "reset",
            "random_matched",
            "swapped_I",
            "swapped_G",
            "swapped_both",
            "prompt_visible_reset",
        )
        for condition in conditions:
            output = _output(greedy=condition not in {"continuous", "random_matched"})
            output["metadata"]["forced_prefix"]["token_ids"] = [63, 11]
            output["metadata"]["forced_prefix"]["greedy_token_ids"] = (
                [63, 66]
                if condition in {"continuous", "random_matched"}
                else [63, 11]
            )
            rows.append(
                (
                    {
                        "source_control_sample_id": "sample-1",
                        "condition": condition,
                        "task_type": "copy",
                        "target_code": "A",
                        "prompt": "CONTROL\nAnswer: A",
                        "prompt_digest_sha256": "a" * 64,
                    },
                    output,
                )
            )
        report = summarize_nonrandom_failure_samples(
            rows,
            lambda text: len(text.split()),
            lambda ids: "/".join(str(item) for item in ids),
        )
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(
            report["samples"][0]["failed_nonrandom_conditions"], ["continuous"]
        )
        self.assertEqual(report["samples"][0]["prompt_token_count"], 3)

    def test_prefix_cells_count_failure_reasons(self) -> None:
        rows = []
        for index in range(32):
            rows.append(
                (
                    {"condition": "continuous", "task_type": "copy"},
                    _output(greedy=index != 0),
                )
            )
        report = summarize_prefix_cells(rows)
        self.assertEqual(report["cell_count"], 1)
        self.assertEqual(report["cells_with_failures"], 1)
        self.assertEqual(report["invalid_record_count"], 1)
        self.assertEqual(report["cells"][0]["failure_counts"]["greedy_mismatch"], 1)

    def test_matched_norms_count_records_components_and_paths(self) -> None:
        outputs = []
        for index in range(4):
            output = _output(greedy=True)
            output["metadata"].update(
                {
                    "state_norm_alert_count": 2 if index < 2 else 0,
                    "state_norm_alert_paths": ["a", "b"] if index < 2 else [],
                    "state_norm_max_alert_ratio": 1.5 if index < 2 else 0.0,
                }
            )
            outputs.append(output)
        report = summarize_matched_norms(outputs)
        self.assertEqual(report["records_with_alerts"], 2)
        self.assertEqual(report["total_component_alerts"], 4)
        self.assertEqual(report["record_alert_rate"], 0.5)
        self.assertEqual(report["top_alert_paths"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
