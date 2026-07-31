from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import payload_digest, sha256_file, sha256_json
from psa.preregistration import (
    derive_formal_seed,
    derive_template_holdout_seed,
    evaluate_control_records,
    evaluate_template_qualification,
    generate_control_manifest,
    generate_template_qualification_manifest,
    review_control_rotation,
    simulate_power,
    verify_preregistration_candidate,
)
from psa.preregistration.formal_freeze import _load_formal_config
from psa.preregistration.formal_review import _select_review_route


class FormalFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v1.json"
        )
        cls.config = json.loads(
            cls.config_path.read_text(encoding="utf-8")
        )

    def test_seed_derivation_matches_frozen_config(self) -> None:
        expected = {
            "core_generator": derive_formal_seed("core-generator"),
            "control_generator": derive_formal_seed("control-generator"),
            "bootstrap": derive_formal_seed("bootstrap"),
            "permutation": derive_formal_seed("permutation"),
            "simulation": derive_formal_seed("simulation"),
        }
        self.assertEqual(
            {
                field: self.config["seeds"][field]
                for field in expected
            },
            expected,
        )
        self.assertEqual(expected["core_generator"], 22217530)
        self.assertEqual(expected["simulation"], 4045556568)

    def test_template_manifest_is_balanced_and_does_not_create_core_set(
        self,
    ) -> None:
        manifest = generate_template_qualification_manifest(
            self.config,
            token_counter=lambda _: 131,
        )
        self.assertEqual(manifest["semantic_case_count"], 128)
        self.assertEqual(manifest["trial_count"], 512)
        self.assertTrue(manifest["development_only"])
        self.assertFalse(manifest["confirmatory_results_observed"])
        self.assertEqual(
            {item["token_count"] for item in manifest["filler_variants"]},
            {131},
        )
        self.assertEqual(
            len({item["text"] for item in manifest["filler_variants"]}),
            4,
        )

        cases_by_history: dict[str, set[str]] = {}
        cases_by_query: dict[str, set[str]] = {}
        trials_by_case: dict[str, list[dict]] = {}
        for trial in manifest["trials"]:
            cases_by_history.setdefault(
                trial["history_template_id"], set()
            ).add(trial["semantic_case_id"])
            cases_by_query.setdefault(
                trial["query_template_id"], set()
            ).add(trial["semantic_case_id"])
            trials_by_case.setdefault(
                trial["semantic_case_id"], []
            ).append(trial)
            query = trial["prompt"].rsplit("User:", maxsplit=1)[1]
            before_options = query.split("OPTIONS:", maxsplit=1)[0]
            self.assertNotIn(
                trial["target_fields"]["domain"],
                before_options,
            )
            self.assertNotIn(
                trial["target_fields"]["operation"],
                before_options,
            )

        self.assertEqual(
            {len(cases) for cases in cases_by_history.values()},
            {32},
        )
        self.assertEqual(
            {len(cases) for cases in cases_by_query.values()},
            {32},
        )
        self.assertEqual(len(trials_by_case), 128)
        self.assertTrue(
            all(len(trials) == 4 for trials in trials_by_case.values())
        )
        self.assertTrue(
            all(
                {trial["target_code"] for trial in trials}
                == {"A", "B", "C", "D"}
                for trials in trials_by_case.values()
            )
        )
        self.assertFalse(
            self.config["core_design"]["generate_or_unseal_core_set"]
        )

    def test_perfect_template_records_pass_frozen_thresholds(self) -> None:
        manifest = generate_template_qualification_manifest(
            self.config,
            token_counter=lambda _: 131,
        )
        records = []
        for trial in manifest["trials"]:
            target_option = next(
                option
                for option in trial["option_mapping"]
                if option["code"] == trial["target_code"]
            )
            scores = {
                option["code"]: 2.0 if option == target_option else 0.0
                for option in trial["option_mapping"]
            }
            records.append(
                {
                    "sample_id": trial["sample_id"],
                    "option_scores": scores,
                    "argmax_choice": trial["target_code"],
                    "format_valid": True,
                    "status": "success",
                }
            )
        thresholds = dict(self.config["template_qualification"])
        thresholds["bootstrap_replicates"] = 100
        report = evaluate_template_qualification(
            manifest=manifest,
            records=records,
            thresholds=thresholds,
            bootstrap_seed=self.config["seeds"]["bootstrap"],
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["template_qualification_passed"])
        self.assertEqual(report["metrics"]["joint"]["point"], 1.0)
        self.assertEqual(
            set(report["history_template_metrics"]),
            set(manifest["history_template_ids"]),
        )
        self.assertEqual(
            report["development_nuisance"]["factorial_group_count"],
            32,
        )
        self.assertEqual(
            set(
                report["development_nuisance"][
                    "standard_deviation"
                ]
            ),
            {
                "E1_identity_transfer",
                "E2_goal_transfer",
                "E3_joint_binding",
            },
        )

    def test_control_manifest_and_evaluator(self) -> None:
        manifest = generate_control_manifest(self.config)
        self.assertEqual(manifest["trial_count"], 96)
        counts = {}
        records = []
        for trial in manifest["trials"]:
            counts[trial["task_type"]] = (
                counts.get(trial["task_type"], 0) + 1
            )
            records.append(
                {
                    "sample_id": trial["sample_id"],
                    "argmax_choice": trial["target_code"],
                    "format_valid": True,
                    "status": "success",
                }
            )
        self.assertEqual(set(counts.values()), {32})
        report = evaluate_control_records(
            manifest=manifest,
            records=records,
            minimum_accuracy_per_task=0.9,
            minimum_format_valid_rate=0.99,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["control_baseline_passed"])

    def test_control_review_marginalizes_known_answer_code_bias(self) -> None:
        manifest = generate_control_manifest(self.config)
        records = []
        for trial in manifest["trials"]:
            scores = {
                option["code"]: (
                    2.0
                    if option["code"] == trial["target_code"]
                    else 0.0
                )
                for option in trial["option_mapping"]
            }
            predicted = trial["target_code"]
            if (
                trial["task_type"]
                == "unrelated_two_field_symbol_match"
                and trial["target_code"] == "D"
            ):
                predicted = next(
                    code for code in scores if code != trial["target_code"]
                )
                scores[trial["target_code"]] = 0.0
                scores[predicted] = 3.0
            records.append(
                {
                    "sample_id": trial["sample_id"],
                    "option_scores": scores,
                    "argmax_choice": predicted,
                    "format_valid": True,
                    "status": "success",
                }
            )
        review = review_control_rotation(
            manifest=manifest,
            records=records,
            minimum_accuracy=0.9,
        )
        two_field = review["task_reports"][
            "unrelated_two_field_symbol_match"
        ]
        self.assertEqual(two_field["code_level_accuracy"], 0.75)
        self.assertEqual(two_field["label_marginalized_accuracy"], 1.0)
        self.assertTrue(two_field["label_marginalized_pass_threshold"])
        self.assertEqual(
            review["route_decision"],
            "control_code_bias_controlled_by_rotation",
        )

    def test_v2_overlay_preserves_locked_design_and_revises_prompts(
        self,
    ) -> None:
        v2_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v2.json"
        )
        v2 = _load_formal_config(v2_path, self.root)
        self.assertEqual(
            v2["gate"],
            "impl3r_exp001_formal_freeze_candidate_v2",
        )
        for field in (
            "model_config",
            "labels",
            "answer_interface",
            "filler_protocol",
            "seeds",
            "core_design",
            "statistics",
            "power_simulation",
            "safety_boundary",
            "template_qualification",
        ):
            self.assertEqual(v2[field], self.config[field])
        self.assertEqual(
            v2["history_protocol"]["mode"],
            self.config["history_protocol"]["mode"],
        )
        self.assertTrue(
            all(
                "CURRENT DOMAIN:" in template["user_text"]
                and "CURRENT OPERATION:" in template["user_text"]
                for template in v2["history_protocol"]["templates"]
            )
        )
        self.assertTrue(
            all(
                "DOMAIN" in template["user_text"]
                and "OPERATION" in template["user_text"]
                for template in v2["query_protocol"]["templates"]
            )
        )
        self.assertEqual(
            v2["controls"]["vocabulary"]["two_field_names"],
            ["COLOR", "SHAPE"],
        )
        self.assertTrue(
            v2["controls"][
                "use_rotation_marginalized_semantic_controls"
            ]
        )
        self.assertIn(
            "configs/preregistration/exp001_track_s.formal_v2.json",
            v2["source_files"],
        )

    def test_v2_control_gate_uses_predeclared_rotation_readout(
        self,
    ) -> None:
        v2_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v2.json"
        )
        v2 = _load_formal_config(v2_path, self.root)
        manifest = generate_control_manifest(v2)
        two_field_prompts = [
            trial["prompt"]
            for trial in manifest["trials"]
            if trial["task_type"]
            == "unrelated_two_field_symbol_match"
        ]
        self.assertTrue(two_field_prompts)
        self.assertTrue(
            all(
                "TARGET COLOR:" in prompt
                and "TARGET SHAPE:" in prompt
                and "MARKER" not in prompt
                and "PATTERN" not in prompt
                for prompt in two_field_prompts
            )
        )

        records = []
        for trial in manifest["trials"]:
            scores = {
                option["code"]: (
                    2.0
                    if option["code"] == trial["target_code"]
                    else 0.0
                )
                for option in trial["option_mapping"]
            }
            predicted = trial["target_code"]
            if (
                trial["task_type"]
                == "unrelated_two_field_symbol_match"
                and trial["target_code"] == "D"
            ):
                predicted = next(
                    code for code in scores if code != trial["target_code"]
                )
                scores[trial["target_code"]] = 0.0
                scores[predicted] = 3.0
            records.append(
                {
                    "sample_id": trial["sample_id"],
                    "option_scores": scores,
                    "argmax_choice": predicted,
                    "format_valid": True,
                    "status": "success",
                }
            )
        report = evaluate_control_records(
            manifest=manifest,
            records=records,
            minimum_accuracy_per_task=0.9,
            minimum_format_valid_rate=0.99,
            use_rotation_marginalized_semantic_controls=True,
        )
        two_field = report["task_metrics"][
            "unrelated_two_field_symbol_match"
        ]
        self.assertEqual(two_field["code_level_accuracy"], 0.75)
        self.assertEqual(two_field["label_marginalized_accuracy"], 1.0)
        self.assertEqual(two_field["evaluation_accuracy"], 1.0)
        self.assertTrue(two_field["pass_threshold"])
        self.assertTrue(report["control_baseline_passed"])
        self.assertEqual(
            report["semantic_control_readout"],
            "rotation_marginalized",
        )

    def test_v2_control_gate_never_falls_back_when_scores_are_missing(
        self,
    ) -> None:
        v2_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v2.json"
        )
        v2 = _load_formal_config(v2_path, self.root)
        manifest = generate_control_manifest(v2)
        records = [
            {
                "sample_id": trial["sample_id"],
                "argmax_choice": trial["target_code"],
                "format_valid": True,
                "status": "success",
            }
            for trial in manifest["trials"]
        ]
        report = evaluate_control_records(
            manifest=manifest,
            records=records,
            minimum_accuracy_per_task=0.9,
            minimum_format_valid_rate=0.99,
            use_rotation_marginalized_semantic_controls=True,
        )
        self.assertFalse(report["control_baseline_passed"])
        for task_type in (
            "single_field_lexical_match",
            "unrelated_two_field_symbol_match",
        ):
            task = report["task_metrics"][task_type]
            self.assertEqual(task["code_level_accuracy"], 1.0)
            self.assertEqual(task["evaluation_accuracy"], 0.0)
            self.assertFalse(task["evaluation_readout_complete"])
            self.assertFalse(task["pass_threshold"])

    def test_v2_hold_review_revises_only_formal_templates(self) -> None:
        self.assertEqual(
            _select_review_route(
                template_passed=False,
                control_passed=True,
                control_rotation_route=(
                    "control_code_bias_controlled_by_rotation"
                ),
            ),
            "revise_formal_template_family_only",
        )

    def test_v3_changes_only_the_history_template_family(self) -> None:
        v2_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v2.json"
        )
        v3_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v3.json"
        )
        v2 = _load_formal_config(v2_path, self.root)
        v3 = _load_formal_config(v3_path, self.root)
        self.assertEqual(
            v3["gate"],
            "impl3s_exp001_formal_freeze_candidate_v3",
        )
        for field in (
            "model_config",
            "labels",
            "answer_interface",
            "filler_protocol",
            "seeds",
            "core_design",
            "statistics",
            "power_simulation",
            "safety_boundary",
            "template_qualification",
            "query_protocol",
            "controls",
        ):
            self.assertEqual(v3[field], v2[field])
        self.assertEqual(
            v3["history_protocol"]["mode"],
            v2["history_protocol"]["mode"],
        )
        self.assertNotEqual(
            v3["history_protocol"]["templates"],
            v2["history_protocol"]["templates"],
        )
        self.assertEqual(
            len(v3["history_protocol"]["templates"]),
            4,
        )
        for template in v3["history_protocol"]["templates"]:
            self.assertIn(
                "FIELD 1 - CURRENT DOMAIN:",
                template["user_text"],
            )
            self.assertIn(
                "FIELD 2 - CURRENT OPERATION:",
                template["user_text"],
            )
            self.assertIn(
                "Both CURRENT DOMAIN and CURRENT OPERATION",
                template["assistant_ack"],
            )
        self.assertFalse(
            v3["core_design"]["generate_or_unseal_core_set"]
        )
        self.assertIn(
            "configs/preregistration/exp001_track_s.formal_v3.json",
            v3["source_files"],
        )

    def test_v3_holdout_is_fresh_one_shot_and_prompt_frozen(
        self,
    ) -> None:
        v3_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v3.json"
        )
        holdout_path = (
            self.root
            / "configs"
            / "preregistration"
            / "exp001_track_s.formal_v3_holdout.json"
        )
        v3 = _load_formal_config(v3_path, self.root)
        holdout = _load_formal_config(holdout_path, self.root)
        self.assertEqual(
            holdout["gate"],
            "impl3t_exp001_formal_v3_holdout",
        )
        for field in (
            "model_config",
            "labels",
            "answer_interface",
            "history_protocol",
            "query_protocol",
            "filler_protocol",
            "controls",
            "seeds",
            "core_design",
            "statistics",
            "power_simulation",
            "safety_boundary",
        ):
            self.assertEqual(holdout[field], v3[field])
        qualification = holdout["template_qualification"]
        self.assertEqual(
            qualification["manifest_seed"],
            derive_template_holdout_seed(),
        )
        self.assertEqual(
            qualification["selection_role"],
            "one_shot_held_out_validation",
        )
        self.assertEqual(derive_template_holdout_seed(), 3061017642)

        v3_manifest = generate_template_qualification_manifest(
            v3,
            token_counter=lambda _: 131,
        )
        holdout_manifest = generate_template_qualification_manifest(
            holdout,
            token_counter=lambda _: 131,
        )
        repeated_holdout = generate_template_qualification_manifest(
            holdout,
            token_counter=lambda _: 131,
        )
        self.assertEqual(v3_manifest["trial_count"], 512)
        self.assertEqual(holdout_manifest["trial_count"], 512)
        self.assertNotEqual(
            v3_manifest["manifest_digest_sha256"],
            holdout_manifest["manifest_digest_sha256"],
        )
        self.assertEqual(
            holdout_manifest["manifest_digest_sha256"],
            repeated_holdout["manifest_digest_sha256"],
        )
        self.assertEqual(
            holdout_manifest["selection_role"],
            "one_shot_held_out_validation",
        )
        self.assertFalse(
            holdout["core_design"]["generate_or_unseal_core_set"]
        )

    def test_power_simulation_retains_320_groups(self) -> None:
        report = simulate_power(
            self.config,
            nuisance_standard_deviation={
                "E1_identity_transfer": 1.0,
                "E2_goal_transfer": 1.0,
                "E3_joint_binding": 1.0,
            },
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["power_gate_passed"])
        self.assertEqual(report["sample_size"], 320)
        self.assertEqual(report["route_decision"], "retain_n_320")
        self.assertTrue(
            all(
                value >= 0.9
                for value in report[
                    "standardized_endpoint_power"
                ].values()
            )
        )
        self.assertTrue(
            all(
                value >= 0.9
                for value in report[
                    "empirical_proxy_endpoint_power"
                ].values()
            )
        )

    def test_candidate_verification_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("locked source", encoding="utf-8")
            evidence = root / "evidence.json"
            evidence.write_text('{"valid":true}\n', encoding="utf-8")
            source_digests = {"source.txt": sha256_file(source)}
            evidence_digests = {"evidence.json": sha256_file(evidence)}
            locked = {
                "source:source.txt": source_digests["source.txt"],
                "evidence:evidence.json": evidence_digests[
                    "evidence.json"
                ],
            }
            candidate = {
                "source_file_digests": source_digests,
                "evidence_file_digests": evidence_digests,
                "payload_root_digest_sha256": payload_digest(locked),
                "core_set_generated": False,
                "core_set_unsealed": False,
                "formal_state_only_results_observed": False,
                "human_checksum_confirmation_required": True,
                "eligible_for_human_freeze": True,
            }
            candidate["candidate_digest_sha256"] = sha256_json(candidate)
            candidate_path = root / "candidate.json"
            candidate_path.write_text(
                json.dumps(candidate),
                encoding="utf-8",
            )
            report = verify_preregistration_candidate(
                candidate_path,
                project_root=root,
            )
            self.assertTrue(report["valid"])
            evidence.write_text('{"valid":false}\n', encoding="utf-8")
            tampered = verify_preregistration_candidate(
                candidate_path,
                project_root=root,
            )
            self.assertFalse(tampered["valid"])
            self.assertFalse(
                tampered["evidence_file_checks"]["evidence.json"]
            )
            evidence.write_text('{"valid":true}\n', encoding="utf-8")
            candidate["source_file_digests"] = {
                "../source.txt": sha256_file(source)
            }
            candidate["payload_root_digest_sha256"] = payload_digest(
                {
                    "source:../source.txt": sha256_file(source),
                    "evidence:evidence.json": sha256_file(evidence),
                }
            )
            candidate.pop("candidate_digest_sha256", None)
            candidate["candidate_digest_sha256"] = sha256_json(candidate)
            candidate_path.write_text(
                json.dumps(candidate),
                encoding="utf-8",
            )
            traversal = verify_preregistration_candidate(
                candidate_path,
                project_root=root,
            )
            self.assertFalse(traversal["valid"])
            self.assertFalse(
                traversal["source_file_checks"]["../source.txt"]
            )


if __name__ == "__main__":
    unittest.main()
