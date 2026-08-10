from __future__ import annotations

import unittest

from psa.supplemental.diagnostics import (
    prefix_failure_flags,
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
    def test_prefix_flags_separate_greedy_and_roundtrip_failures(self) -> None:
        greedy = prefix_failure_flags(_output(greedy=False))
        self.assertTrue(greedy["greedy_mismatch"])
        self.assertFalse(greedy["roundtrip_mismatch"])
        self.assertFalse(greedy["valid"])

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
