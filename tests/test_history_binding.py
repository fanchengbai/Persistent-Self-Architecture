from __future__ import annotations

import json
from pathlib import Path
import unittest

from psa.development.history_binding import (
    SUPPORTED_HISTORY_MODES,
    evaluate_history_binding,
    generate_history_binding_manifest,
    render_history_binding,
)


class HistoryBindingTests(unittest.TestCase):
    def _manifest(self) -> dict:
        return generate_history_binding_manifest(
            answer_codes=("A", "B", "C", "D"),
            identity_label_pairs=(("baf", "zom"), ("niv", "teg")),
            goal_label_pairs=(("vam", "zep"), ("qir", "bok")),
            history_modes=SUPPORTED_HISTORY_MODES,
            repetitions=2,
            base_seed=20260731,
            delay_units=11,
            assistant_prefix="<think></think",
        )

    @staticmethod
    def _records(
        manifest: dict,
        *,
        failing_modes: set[str] | None = None,
    ) -> list[dict]:
        failing_modes = failing_modes or set()
        records = []
        for trial in manifest["trials"]:
            target_fields = trial["target_fields"]
            target_option = next(
                option
                for option in trial["option_mapping"]
                if (
                    option["domain"] == target_fields["domain"]
                    and option["operation"] == target_fields["operation"]
                )
            )
            selected_option = target_option
            if trial["history_mode"] in failing_modes:
                selected_option = next(
                    option
                    for option in trial["option_mapping"]
                    if (
                        option["domain"] != target_fields["domain"]
                        or option["operation"] != target_fields["operation"]
                    )
                )
            scores = {
                option["code"]: (
                    2.0 if option == selected_option else 0.0
                )
                for option in trial["option_mapping"]
            }
            records.append(
                {
                    "sample_id": trial["sample_id"],
                    "option_scores": scores,
                    "argmax_choice": selected_option["code"],
                    "format_valid": True,
                    "status": "success",
                }
            )
        return records

    def test_manifest_is_paired_and_code_rotated(self) -> None:
        manifest = self._manifest()
        self.assertEqual(manifest["semantic_case_count_per_mode"], 8)
        self.assertEqual(manifest["trial_count_per_mode"], 32)
        self.assertEqual(manifest["trial_count"], 96)
        self.assertTrue(manifest["general_rule_visible"])
        self.assertFalse(
            manifest[
                "current_state_values_visible_outside_balanced_options"
            ]
        )

        grouped: dict[tuple[str, str], list[dict]] = {}
        for trial in manifest["trials"]:
            grouped.setdefault(
                (trial["history_mode"], trial["semantic_case_id"]),
                [],
            ).append(trial)
            history = trial["history_text"]
            self.assertIn(trial["target_fields"]["domain"], history)
            self.assertIn(trial["target_fields"]["operation"], history)
            query_prefix = trial["query_text"].split("OPTIONS:", 1)[0]
            self.assertNotIn(
                trial["target_fields"]["domain"],
                query_prefix,
            )
            self.assertNotIn(
                trial["target_fields"]["operation"],
                query_prefix,
            )

        self.assertEqual(len(grouped), 24)
        for trials in grouped.values():
            self.assertEqual(len(trials), 4)
            self.assertEqual(
                {trial["target_code"] for trial in trials},
                {"A", "B", "C", "D"},
            )

    def test_committed_config_freezes_simplest_passing_rule(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (
                root
                / "configs"
                / "gates"
                / "impl3p_g1h_2.9b_history_binding.dev.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["selection"]["rule"],
            "first_passing_mode_in_predeclared_complexity_order",
        )
        self.assertEqual(
            config["selection"]["ordered_modes"],
            list(SUPPORTED_HISTORY_MODES),
        )
        self.assertEqual(
            config["selection"][
                "minimum_label_marginalized_accuracy"
            ],
            0.8,
        )
        self.assertEqual(config["delay_units"], 11)
        self.assertTrue(config["general_rule_visible"])

    def test_history_modes_have_predeclared_strength_order(self) -> None:
        rendered = {
            mode: render_history_binding(
                mode,
                domain="baf",
                operation="vam",
                delay_units=11,
            )
            for mode in SUPPORTED_HISTORY_MODES
        }
        self.assertEqual(
            rendered["single_statement"].count("CURRENT DOMAIN: baf"),
            1,
        )
        self.assertEqual(
            rendered["statement_plus_verification"].count(
                "CURRENT DOMAIN: baf"
            ),
            2,
        )
        self.assertEqual(
            rendered["repeated_consistent"].count("CURRENT DOMAIN: baf"),
            2,
        )
        self.assertIn(
            "Verify the saved state",
            rendered["statement_plus_verification"],
        )
        self.assertIn(
            "CONSISTENT STATE BINDING",
            rendered["repeated_consistent"],
        )

    def test_selection_chooses_first_passing_mode_not_best_later(self) -> None:
        manifest = self._manifest()
        report = evaluate_history_binding(
            manifest=manifest,
            records=self._records(manifest),
            selection_order=SUPPORTED_HISTORY_MODES,
            minimum_label_marginalized_accuracy=0.8,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["history_binding_gate_passed"])
        self.assertEqual(report["selected_mode"], "single_statement")
        self.assertEqual(
            report["route_decision"],
            "freeze_single_statement",
        )

    def test_selection_advances_only_when_simpler_mode_fails(self) -> None:
        manifest = self._manifest()
        report = evaluate_history_binding(
            manifest=manifest,
            records=self._records(
                manifest,
                failing_modes={"single_statement"},
            ),
            selection_order=SUPPORTED_HISTORY_MODES,
            minimum_label_marginalized_accuracy=0.8,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["mode_reports"]["single_statement"][
                "label_marginalized_accuracy"
            ],
            0.0,
        )
        self.assertEqual(
            report["selected_mode"],
            "statement_plus_verification",
        )

    def test_no_mode_passes_routes_to_revise(self) -> None:
        manifest = self._manifest()
        report = evaluate_history_binding(
            manifest=manifest,
            records=self._records(
                manifest,
                failing_modes=set(SUPPORTED_HISTORY_MODES),
            ),
            selection_order=SUPPORTED_HISTORY_MODES,
            minimum_label_marginalized_accuracy=0.8,
        )
        self.assertTrue(report["valid"])
        self.assertFalse(report["history_binding_gate_passed"])
        self.assertIsNone(report["selected_mode"])
        self.assertEqual(
            report["route_decision"],
            "revise_history_binding_protocol",
        )

    def test_missing_record_invalidates_diagnostic(self) -> None:
        manifest = self._manifest()
        records = self._records(manifest)
        report = evaluate_history_binding(
            manifest=manifest,
            records=records[:-1],
            selection_order=SUPPORTED_HISTORY_MODES,
            minimum_label_marginalized_accuracy=0.8,
        )
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
