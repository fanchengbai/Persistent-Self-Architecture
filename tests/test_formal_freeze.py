from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import payload_digest, sha256_file, sha256_json
from psa.preregistration import (
    derive_formal_seed,
    evaluate_control_records,
    evaluate_template_qualification,
    generate_control_manifest,
    generate_template_qualification_manifest,
    review_control_rotation,
    simulate_power,
    verify_preregistration_candidate,
)


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
