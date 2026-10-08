from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.self_model.semantic_goal_offline import (
    GOALS, SPLITS, GoalRequest, Option, Scenario, build_goal_state, clear_goal_state,
    generate_scenarios, load_config, prompt_visible, reference_answer,
    render_task, request_from_state, sample_size_review, score_response,
    summarize_responses, swap_goal_states, validate_dataset, write_offline_package,
)
from psa.self_model.state import SelfStore, self_state_digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/development/self_model_v0_1_semantic_goal_offline.json"


class SemanticGoalOfflineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.scenarios = generate_scenarios(self.config)

    def test_meaningful_tradeoff_and_label_reversal(self):
        scenario = Scenario("semgoal-v01-heldout-example", "heldout", 0, "tradeoff",
                            (Option(4, 9), Option(9, 4)))
        normal, reversed_task = render_task(scenario, "AB"), render_task(scenario, "BA")
        self.assertEqual(reference_answer(normal, GOALS[0]), "B")
        self.assertEqual(reference_answer(normal, GOALS[1]), "A")
        self.assertEqual(reference_answer(reversed_task, GOALS[0]), "A")
        self.assertEqual(reference_answer(reversed_task, GOALS[1]), "B")
        self.assertIn("耗时 4 分钟，能耗 9 单位", normal.prompt)

    def test_dominance_does_not_reverse_when_goal_swaps(self):
        scenario = Scenario("semgoal-v01-heldout-example", "heldout", 0, "dominance",
                            (Option(3, 2), Option(9, 8)))
        task = render_task(scenario, "AB")
        self.assertEqual([reference_answer(task, g) for g in GOALS], ["A", "A"])

    def test_determinism_and_semantic_balance_in_every_split(self):
        self.assertEqual(self.scenarios, generate_scenarios(self.config))
        report = validate_dataset(self.scenarios, self.config)
        self.assertEqual(report["scenario_counts"], self.config["scenario_counts"])
        for split in SPLITS:
            for goal in GOALS:
                counts = report["answer_balance"][split][goal]
                self.assertEqual(counts["A"], counts["B"])
            for kind in ("tradeoff", "dominance"):
                subset = [s for s in self.scenarios if s.split == split and s.kind == kind]
                choices = [reference_answer(render_task(s, "AB"), GOALS[0]) for s in subset]
                self.assertEqual(choices.count("A"), choices.count("B"))

    def test_duplicate_facts_across_splits_are_rejected_even_with_new_id(self):
        changed = list(self.scenarios)
        first_dev = next(i for i, s in enumerate(changed) if s.split == "development")
        changed[first_dev] = replace(changed[first_dev], kind=changed[0].kind,
                                     options=changed[0].options[::-1])
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset(tuple(changed), self.config)

    def test_missing_or_duplicate_scenario_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataset(self.scenarios[:-1], self.config)
        with self.assertRaises(ValueError):
            validate_dataset(self.scenarios + (self.scenarios[0],), self.config)

    def test_invalid_numbers_ties_and_kind_fail_closed(self):
        for bad in (True, 0, -1, float("nan"), float("inf"), 2.5):
            with self.assertRaises(ValueError):
                Option(bad, 3).validate()
        scenario = self.scenarios[0]
        with self.assertRaises(ValueError):
            replace(scenario, options=(Option(4, 3), Option(4, 9))).validate()
        with self.assertRaises(ValueError):
            replace(scenario, kind="dominance").validate()
        with self.assertRaises(ValueError):
            replace(scenario, template_index=True).validate()

    def test_public_view_has_no_goal_or_correct_answer(self):
        task = render_task(self.scenarios[0], "AB")
        payload = asdict(task)
        self.assertNotIn("reference_choice", payload)
        self.assertNotIn("goal", payload)
        self.assertNotIn("minimize_energy", task.prompt)
        self.assertNotIn("minimize_time", task.prompt)
        self.assertNotIn("优先选择", task.prompt)
        self.assertNotEqual(prompt_visible(task, GOALS[0]), prompt_visible(task, GOALS[1]))

    def test_format_failure_is_wrong_not_dropped(self):
        task = render_task(self.scenarios[0], "AB")
        choice = reference_answer(task, GOALS[0])
        self.assertTrue(score_response(task, GOALS[0], "\n" + choice + " ")["correct"])
        for response in ("", "A or B", "答案是 " + choice, "C", "AB"):
            score = score_response(task, GOALS[0], response)
            self.assertFalse(score["format_valid"])
            self.assertFalse(score["correct"])
        with self.assertRaises(TypeError):
            score_response(task, GOALS[0], None)

    def test_scoring_aggregates_at_scenario_level(self):
        scenarios = self.scenarios[:2]
        responses = []
        for s in scenarios:
            for order in ("AB", "BA"):
                task = render_task(s, order)
                for goal in GOALS:
                    responses.append({"variant_id": task.variant_id, "goal": goal,
                                      "response": reference_answer(task, goal)})
        responses[0]["response"] = "invalid"
        report = summarize_responses(scenarios, responses)
        self.assertEqual(report["scenario_count"], 2)
        self.assertEqual(report["response_count"], 8)
        self.assertEqual(report["format_failures"], 1)
        self.assertEqual(report["accuracy"], 7 / 8)
        self.assertEqual(report["scenario_cluster_accuracy"][scenarios[0].scenario_id], 3 / 4)
        self.assertFalse(report["scientific_decision_made"])
        for bad in (responses[:-1], responses + [responses[0]],
                    [dict(responses[0], reference_choice="A")] + responses[1:]):
            with self.assertRaises(ValueError):
                summarize_responses(scenarios, bad)

    def test_goal_swap_and_mask_are_task_independent_and_inputs_immutable(self):
        energy = build_goal_state(GOALS[0], state_id="energy")
        time = build_goal_state(GOALS[1], state_id="time")
        source = copy.deepcopy([energy, time])
        swapped = swap_goal_states(energy, time)
        self.assertEqual(request_from_state(swapped[0]), GoalRequest("active", GOALS[1]))
        self.assertEqual(request_from_state(swapped[1]), GoalRequest("active", GOALS[0]))
        for mode in ("zero", "mask", "random"):
            self.assertIsNone(request_from_state(energy, mode).goal)
        self.assertEqual(source, [energy, time])
        self.assertEqual({f.name for f in fields(GoalRequest)}, {"mode", "goal"})

    def test_state_save_restore_and_tamper_detection(self):
        state = build_goal_state(GOALS[0], state_id="energy")
        with tempfile.TemporaryDirectory() as directory:
            store = SelfStore(directory)
            path = store.save(state)
            self.assertEqual(store.load("energy"), state)
            with self.assertRaises(FileExistsError):
                store.save(state)
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["active_goals"][0]["value"] = GOALS[1]
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                store.load("energy")

    def test_explicit_clear_is_persistent_and_does_not_change_source(self):
        state = build_goal_state(GOALS[0], state_id="energy")
        original = copy.deepcopy(state)
        cleared = clear_goal_state(state, state_id="cleared")
        self.assertEqual(cleared["active_goals"], [])
        self.assertEqual(request_from_state(cleared, "mask"), GoalRequest("mask", None))
        with self.assertRaises(ValueError):
            request_from_state(cleared)
        with tempfile.TemporaryDirectory() as directory:
            store = SelfStore(directory)
            store.save(cleared)
            self.assertEqual(store.load("cleared"), cleared)
        self.assertEqual(state, original)

    def test_unknown_or_multiple_goals_and_extra_metadata_rejected(self):
        for bad in ("minimize_money", "A", None):
            with self.assertRaises(ValueError):
                build_goal_state(bad, state_id="bad")
        state = build_goal_state(GOALS[0], state_id="energy")
        for bad in (dict(state, correct_answer="A"), copy.deepcopy(state)):
            if "correct_answer" not in bad:
                bad["active_goals"][0]["value"] = "A"
                bad["integrity"]["payload_sha256"] = self_state_digest(bad)
            with self.assertRaises(ValueError):
                request_from_state(bad)
        with self.assertRaises(ValueError):
            GoalRequest("mask", GOALS[0])

    def test_config_cannot_upgrade_execution_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = copy.deepcopy(self.config)
            config["model_execution_authorized"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
            config = copy.deepcopy(self.config)
            config["scenario_counts"]["heldout"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_package_separates_evaluator_references_and_never_claims_model_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "package"
            report = write_offline_package(CONFIG, output)
            self.assertEqual(report["scenario_count"], 256)
            self.assertEqual(report["public_task_count"], 512)
            self.assertEqual(report["evaluation_reference_count"], 1024)
            self.assertTrue(all(report["interface_checks"].values()))
            self.assertFalse(report["model_accuracy_measured"])
            self.assertEqual(report["actual_model_forward_calls"], 0)
            public = json.loads((output / "public_tasks.json").read_text(encoding="utf-8"))
            self.assertTrue(all("goal" not in t and "reference_choice" not in t for t in public["tasks"]))
            self.assertEqual(report["payload_digests"]["public_tasks.json"], sha256_json(public))
            digest = report.pop("report_digest_sha256")
            self.assertEqual(digest, sha256_json(report))
            with self.assertRaises(FileExistsError):
                write_offline_package(CONFIG, output)

    def test_sample_size_is_only_a_planning_approximation(self):
        review = sample_size_review()
        self.assertFalse(review["final_sample_size_frozen"])
        self.assertFalse(review["real_call_budget_frozen"])
        low_variance = review["assumptions"][:3]
        self.assertGreater(low_variance[0]["approx_required_independent_scenarios"],
                           low_variance[2]["approx_required_independent_scenarios"])
        self.assertGreater(review["assumptions"][3]["approx_required_independent_scenarios"],
                           low_variance[0]["approx_required_independent_scenarios"])


if __name__ == "__main__":
    unittest.main()
