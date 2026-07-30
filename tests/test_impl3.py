from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.development import (
    calibrate_standard_delay,
    classify_capability_route,
    evaluate_capability_level,
    evaluate_prompt_visible,
    generate_capability_manifest,
    generate_g1_capability_manifest,
    inspect_dataset_tokenization,
    inspect_label_pairs,
    render_prompt_visible,
    render_g1_chat_prompt,
    write_jsonl,
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
    def test_capability_manifest_balances_codes_at_each_level(self) -> None:
        manifest = generate_capability_manifest(
            answer_codes=("A", "B", "C", "D"),
            symbols=("baf", "zom", "niv", "teg"),
            repetitions=3,
            base_seed=29,
        )
        self.assertEqual(manifest["trial_count"], 24)
        for level in ("copy_code", "single_field"):
            counts = {}
            for trial in manifest["trials"]:
                if trial["task_level"] == level:
                    code = trial["target_code"]
                    counts[code] = counts.get(code, 0) + 1
            self.assertEqual(counts, {"A": 3, "B": 3, "C": 3, "D": 3})

    def test_g1_manifest_runs_all_levels_live_and_balanced(self) -> None:
        manifest = generate_g1_capability_manifest(
            answer_codes=("A", "B", "C", "D"),
            symbols=("baf", "zom", "niv", "teg"),
            identity_label_pairs=(("baf", "zom"), ("niv", "teg")),
            goal_label_pairs=(("vam", "zep"), ("qir", "bok")),
            repetitions=3,
            base_seed=41,
        )
        self.assertEqual(manifest["trial_count"], 36)
        self.assertEqual(manifest["prompt_format"], "rwkv7-g1-chat-v0.1")
        for level in ("copy_code", "single_field", "two_field"):
            counts = {}
            for trial in manifest["trials"]:
                if trial["task_level"] == level:
                    code = trial["target_code"]
                    counts[code] = counts.get(code, 0) + 1
                    self.assertTrue(trial["prompt"].startswith("User: "))
                    self.assertTrue(trial["prompt"].endswith("Assistant:"))
                    self.assertFalse(trial["prompt"].endswith(" "))
            self.assertEqual(counts, {"A": 3, "B": 3, "C": 3, "D": 3})

    def test_g1_prompt_cleans_round_separators_and_trailing_space(self) -> None:
        prompt = render_g1_chat_prompt(" first\r\n\r\nsecond \n")
        self.assertEqual(prompt, "User: first\nsecond\n\nAssistant:")

    def test_g1_fake_think_prompt_matches_official_incomplete_prefix(self) -> None:
        prompt = render_g1_chat_prompt(
            "choose one code",
            assistant_prefix="<think></think",
        )
        self.assertEqual(
            prompt,
            "User: choose one code\n\nAssistant: <think></think",
        )
        self.assertFalse(prompt.endswith(" "))

    def test_impl3e_changes_only_the_g1_answer_prefill(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        impl3d = json.loads(
            (
                project_root
                / "configs"
                / "gates"
                / "impl3d_g1h_1.5b_capability_ladder.dev.json"
            ).read_text(encoding="utf-8")
        )
        impl3e = json.loads(
            (
                project_root
                / "configs"
                / "gates"
                / "impl3e_g1h_1.5b_fake_think.dev.json"
            ).read_text(encoding="utf-8")
        )
        controlled_fields = (
            "interface_summary",
            "answer_codes",
            "answer_continuation_prefix",
            "single_field_symbols",
            "identity_label_pairs",
            "goal_label_pairs",
            "repetitions",
            "base_seed",
            "max_generation_tokens",
            "bootstrap_replicates",
            "bootstrap_seed",
            "thresholds",
        )
        for field in controlled_fields:
            self.assertEqual(impl3e[field], impl3d[field])
        self.assertEqual(impl3e["assistant_prefix"], "<think></think")
        self.assertEqual(impl3e["forced_answer_prefix"], ">")

    def test_impl3g_changes_only_the_model_interface_evidence(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        gate_dir = project_root / "configs" / "gates"
        impl3e = json.loads(
            (gate_dir / "impl3e_g1h_1.5b_fake_think.dev.json").read_text(
                encoding="utf-8"
            )
        )
        impl3g = json.loads(
            (gate_dir / "impl3g_g1h_2.9b_fake_think.dev.json").read_text(
                encoding="utf-8"
            )
        )
        controlled_fields = (
            "answer_codes",
            "answer_continuation_prefix",
            "assistant_prefix",
            "forced_answer_prefix",
            "single_field_symbols",
            "identity_label_pairs",
            "goal_label_pairs",
            "repetitions",
            "base_seed",
            "max_generation_tokens",
            "bootstrap_replicates",
            "bootstrap_seed",
            "thresholds",
        )
        for field in controlled_fields:
            self.assertEqual(impl3g[field], impl3e[field])
        self.assertEqual(
            impl3g["interface_summary"],
            "results/development/impl3f_g1h_2.9b_interface/summary.json",
        )

    def test_capability_level_evaluation_passes_ideal_records(self) -> None:
        manifest = generate_capability_manifest(
            answer_codes=("A", "B", "C", "D"),
            symbols=("baf", "zom", "niv", "teg"),
            repetitions=4,
            base_seed=31,
        )
        records = [
            {
                "sample_id": trial["sample_id"],
                "task_level": trial["task_level"],
                "status": "success",
                "argmax_choice": trial["target_code"],
                "format_valid": True,
            }
            for trial in manifest["trials"]
        ]
        report = evaluate_capability_level(
            manifest=manifest,
            records=records,
            task_level="copy_code",
            bootstrap_replicates=500,
            bootstrap_seed=37,
            thresholds={
                "accuracy_lower_bound": 0.8,
                "format_valid_rate": 0.99,
                "max_answer_position_accuracy_gap": 0.25,
                "max_infrastructure_failure_rate": 0.01,
            },
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["accuracy_interval"], [1.0, 1.0])

    def test_capability_route_identifies_first_failed_level(self) -> None:
        self.assertEqual(
            classify_capability_route(
                copy_valid=False,
                single_field_valid=False,
                two_field_valid=False,
            ),
            "revise_checkpoint_or_answer_interface",
        )
        self.assertEqual(
            classify_capability_route(
                copy_valid=True,
                single_field_valid=False,
                two_field_valid=False,
            ),
            "revise_single_field_matching",
        )
        self.assertEqual(
            classify_capability_route(
                copy_valid=True,
                single_field_valid=True,
                two_field_valid=False,
            ),
            "revise_compositional_matching",
        )
        self.assertEqual(
            classify_capability_route(
                copy_valid=True,
                single_field_valid=True,
                two_field_valid=True,
            ),
            "go_batch2",
        )

    def test_jsonl_writer_emits_exactly_one_json_object_per_line(self) -> None:
        records = [{"record": 1}, {"record": 2}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(path, records)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line) for line in lines], records)

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

    def test_explicit_match_template_keeps_bindings_next_to_query(self) -> None:
        group = generate_factorial_group(group_seed=23, delay_units=3)
        sample = group.trajectories[0]
        prompt = render_prompt_visible(
            group,
            sample,
            template_version="explicit-match-v0.2",
        )
        self.assertNotIn("NEUTRAL-FILLER", prompt)
        self.assertIn(
            f"CURRENT DOMAIN: {group.identity_labels[sample.identity]}",
            prompt,
        )
        self.assertIn(
            f"CURRENT OPERATION: {group.goal_labels[sample.goal]}",
            prompt,
        )
        self.assertEqual(prompt.count("DOMAIN:"), 5)
        self.assertEqual(prompt.count("OPERATION:"), 5)


if __name__ == "__main__":
    unittest.main()
