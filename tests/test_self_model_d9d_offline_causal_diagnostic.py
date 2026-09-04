from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import unittest

from psa.artifacts import sha256_json
from psa.self_model.d9c_projection_contract import (
    CALIBRATION_COMMITMENT,
    CALIBRATION_SHA256,
    HELDOUT_COMMITMENT,
    HELDOUT_SHA256,
    SCHEDULE_COMMITMENT,
    CalibrationCapture,
    build_frozen_projection_artifact,
)
from psa.self_model.d9d_offline_causal_diagnostic import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    CONFIRMATION_TEXT,
    NEXT_GATE,
    _margins,
    _target_codes,
    analyze_calibration_capture_identity,
    analyze_causal_distribution,
    analyze_projection_geometry,
    build_static_report,
    validate_config,
    validate_evidence_bundle,
    validate_ledger_structure,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifests() -> tuple[dict, dict, dict]:
    calibration = json.loads(
        (ROOT / "configs/development/self_model_v0_1_d9_calibration_manifest.json")
        .read_text(encoding="utf-8")
    )
    heldout = json.loads(
        (ROOT / "configs/development/self_model_v0_1_d9_heldout_manifest.json")
        .read_text(encoding="utf-8")
    )
    schedule = json.loads(
        (ROOT / "configs/development/self_model_v0_1_d9_within_wrapper_schedule.json")
        .read_text(encoding="utf-8")
    )
    return calibration, heldout, schedule


def _projection() -> dict:
    captures = []
    for identity in range(4):
        for goal in range(4):
            for replicate in (1, 2):
                fixture_number = identity * 8 + goal * 2 + replicate
                vector = tuple(
                    1.0
                    + identity * 0.1
                    + goal * 0.2
                    + replicate * 0.01
                    + (index % 7) * 0.001
                    for index in range(2560)
                )
                captures.append(
                    CalibrationCapture(
                        fixture_id=f"d9cal-{fixture_number:03d}",
                        identity_index=identity,
                        goal_index=goal,
                        replicate=replicate,
                        vector=vector,
                    )
                )
    return build_frozen_projection_artifact(
        captures=captures,
        calibration_manifest_sha256=CALIBRATION_SHA256,
        calibration_commitment_sha256=CALIBRATION_COMMITMENT,
        heldout_manifest_sha256=HELDOUT_SHA256,
        heldout_commitment_sha256=HELDOUT_COMMITMENT,
        schedule_commitment_sha256=SCHEDULE_COMMITMENT,
        output_dimension=2560,
        fixture_only=False,
    )


def _scores(codes: dict[str, str], contrast: str, base_number: int) -> dict[str, float]:
    scores = {code: 0.0 for code in "ABCD"}
    scores[codes["true"]] = 1.0
    if contrast == "active_true":
        scores[codes["true"]] += 0.1 if base_number <= 10 else -0.05
    elif contrast == "matched_random":
        scores[codes["true"]] += 0.15
    elif contrast in ("mask_identity", "mask_goal"):
        scores[codes["true"]] += 0.02
    elif contrast == "swap_identity":
        if base_number <= 6:
            scores[codes["identity_swap"]] = 2.0
        else:
            scores[codes["true"]] += 0.02
    elif contrast == "swap_goal":
        if base_number <= 8:
            scores[codes["goal_swap"]] = 2.0
        else:
            scores[codes["true"]] += 0.02
    elif contrast == "synthetic_active":
        scores[codes["true"]] += 0.2
    elif contrast != "wrapper_zero":
        raise AssertionError(contrast)
    return scores


def _ledger(projection_digest: str) -> list[dict]:
    calibration, heldout, schedule = _manifests()
    records = [
        {
            "record_type": "calibration_capture",
            "call_id": f"{fixture['fixture_id']}-capture",
            "fixture_id": fixture["fixture_id"],
            "phase": "calibration",
            "route": "persistent_wrapper_capture",
            "capture_sha256": sha256_json([fixture["fixture_id"], "capture"]),
            "callback_invocations": 32,
            "target_layer_applications": 1,
            "heldout_scored": False,
        }
        for fixture in calibration["fixtures"]
    ]
    fixtures = {item["fixture_id"]: item for item in heldout["fixtures"]}
    for block in schedule["heldout_pair_blocks"]:
        fixture = fixtures[block["fixture_id"]]
        base_number = int(block["base_case_id"].split("-")[-1])
        codes = _target_codes(
            int(block["identity_index"]),
            int(block["goal_index"]),
            int(block["code_rotation"]),
        )
        observations = []
        for condition in block["condition_order"]:
            scores = _scores(codes, condition, base_number)
            observations.append(
                {
                    "condition": condition,
                    "choice_scores": scores,
                    "margins": _margins(scores, codes),
                    "logits_sha256": sha256_json(
                        [block["fixture_id"], condition, base_number]
                    ),
                    "state_component_count": 96,
                    "projection_artifact_sha256": projection_digest,
                }
            )
        by_condition = {item["condition"]: item for item in observations}
        contrast = block["contrast"]
        records.append(
            {
                "record_type": "heldout_pair",
                "pair_block_id": block["pair_block_id"],
                "fixture_id": block["fixture_id"],
                "base_case_id": block["base_case_id"],
                "identity_index": block["identity_index"],
                "goal_index": block["goal_index"],
                "code_rotation": block["code_rotation"],
                "contrast": contrast,
                "latin_position": block["latin_position"],
                "pair_order": block["pair_order"],
                "condition_order": block["condition_order"],
                "route": block["route"],
                "source_state_contract": block["source_state_contract"],
                "zero_margins": by_condition["wrapper_zero"]["margins"],
                "condition_margins": by_condition[contrast]["margins"],
                "observations": observations,
                "synthetic_output_changed": contrast == "synthetic_active",
                "projection_artifact_sha256": projection_digest,
            }
        )
    return records


class D9DOfflineCausalDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
        cls.calibration, cls.heldout, cls.schedule = _manifests()
        cls.projection = _projection()
        cls.ledger = _ledger(cls.projection["artifact_digest_sha256"])

    def test_config_and_static_report_close_all_execution_authority(self) -> None:
        checks = validate_config(self.config)
        self.assertTrue(all(checks.values()))
        self.assertEqual(self.config["implementation_confirmation_text"], CONFIRMATION_TEXT)
        report = build_static_report(project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertFalse(report["real_evidence_loaded"])
        self.assertFalse(report["model_executed"])
        self.assertTrue(all(value is False for value in report["safety"].values()))

    def test_source_has_no_rwkv_or_torch_import(self) -> None:
        source = (
            ROOT / "src/psa/self_model/d9d_offline_causal_diagnostic.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        self.assertNotIn("torch", imports)
        self.assertNotIn("rwkv", imports)

    def test_ledger_validation_and_causal_failure_decomposition(self) -> None:
        checks = validate_ledger_structure(
            records=self.ledger,
            calibration_manifest=self.calibration,
            heldout_manifest=self.heldout,
            schedule=self.schedule,
            projection_digest=self.projection["artifact_digest_sha256"],
        )
        self.assertTrue(all(checks.values()))
        analysis = analyze_causal_distribution(
            records=self.ledger, heldout_manifest=self.heldout
        )
        self.assertEqual(analysis["endpoint"]["metrics"]["positive_base_cases"], 10)
        self.assertTrue(
            analysis["endpoint"]["checks"]["synthetic_positive_control_passes"]
        )
        self.assertFalse(analysis["endpoint"]["all_gates_pass"])
        self.assertFalse(analysis["endpoint"]["self_effect_conclusion"])
        self.assertEqual(len(analysis["base_case_summaries"]), 16)
        self.assertEqual(
            set(analysis["contrast_rotation_level_distributions"]),
            {
                "active_true",
                "mask_identity",
                "mask_goal",
                "swap_identity",
                "swap_goal",
                "matched_random",
                "synthetic_active",
            },
        )

    def test_calibration_hashes_cannot_identify_numeric_replicate_stability(self) -> None:
        audit = analyze_calibration_capture_identity(self.ledger, self.calibration)
        self.assertEqual(len(audit["cells"]), 16)
        self.assertEqual(audit["distinct_capture_hash_pairs"], 16)
        self.assertFalse(audit["capture_vectors_stored"])
        self.assertFalse(audit["replicate_numeric_distance_identifiable"])
        self.assertFalse(audit["per_cell_replicate_magnitude_or_cosine_claim_allowed"])

    def test_projection_geometry_is_complete_and_descriptive_only(self) -> None:
        geometry = analyze_projection_geometry(self.projection)
        self.assertEqual(geometry["identity_branch_rms"]["count"], 4)
        self.assertEqual(geometry["goal_branch_rms"]["count"], 4)
        self.assertEqual(geometry["cross_field_cosine"]["count"], 16)
        self.assertEqual(geometry["active_sum_rms_across_sixteen_cells"]["count"], 16)
        self.assertTrue(geometry["geometry_is_descriptive_not_causal_evidence"])

    def test_missing_reordered_public_and_nonfinite_ledger_fail_closed(self) -> None:
        variants = []
        variants.append(self.ledger[:-1])
        reordered = copy.deepcopy(self.ledger)
        reordered[32], reordered[33] = reordered[33], reordered[32]
        variants.append(reordered)
        public = copy.deepcopy(self.ledger)
        public[32]["route"] = "public"
        variants.append(public)
        nonfinite = copy.deepcopy(self.ledger)
        nonfinite[32]["observations"][0]["choice_scores"]["A"] = float("nan")
        variants.append(nonfinite)
        for variant in variants:
            with self.subTest(length=len(variant)):
                with self.assertRaises((ValueError, TypeError)):
                    validate_ledger_structure(
                        records=variant,
                        calibration_manifest=self.calibration,
                        heldout_manifest=self.heldout,
                        schedule=self.schedule,
                        projection_digest=self.projection["artifact_digest_sha256"],
                    )

    def test_evidence_bundle_binds_consumed_claim_report_and_integrity(self) -> None:
        config = copy.deepcopy(self.config)
        expected = config["evidence"]
        expected["git_commit"] = "a" * 40
        expected["installed_source_sha256"] = "b" * 64
        expected["projection_artifact_digest_sha256"] = self.projection[
            "artifact_digest_sha256"
        ]
        expected["projection_parameter_digest_sha256"] = self.projection[
            "parameter_digest_sha256"
        ]
        authorization = {
            "authorized": True,
            "single_use": True,
            "model_forward_calls": 928,
            "git_commit": "a" * 40,
            "d9d_rerun_authorized": False,
            "d8c_rerun_authorized": False,
            "historical_rerun_authorized": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "formal_test_set_authorized": False,
            "self_effect_conclusion_authorized": False,
            "self_updater_authorized": False,
            "raw_original_route_authorized": False,
            "automatic_rerun_authorized": False,
        }
        authorization["authorization_digest_sha256"] = sha256_json(authorization)
        expected["authorization_digest_sha256"] = authorization[
            "authorization_digest_sha256"
        ]
        claim = {
            "status": "d9d_single_use_joint_execution_claim_consumed",
            "single_use": True,
            "git_commit": "a" * 40,
            "authorization_sha256": expected["authorization_file_sha256"],
            "entry_static_report_sha256": (
                "e9ad2903a5bf703b0eebcc61cdc8d5afb87f27b7838443a406df84e77fc5cc09"
            ),
            "installed_source_sha256": "b" * 64,
            "calibration_forward_calls": 32,
            "heldout_forward_calls": 896,
            "total_forward_calls": 928,
            "d9d_rerun_authorized": False,
            "automatic_rerun_authorized": False,
        }
        report = {
            "status": "d9d_real_within_wrapper_causal_isolation_completed_claim_consumed",
            "valid": True,
            "classification": "revise_or_stop_without_self_effect_claim_or_rerun",
            "self_effect_conclusion": False,
            "git_commit": "a" * 40,
            "execution_claim_sha256": expected["claim_file_sha256"],
            "authorization_digest_sha256": expected["authorization_digest_sha256"],
            "counts": {
                "calibration_forward_calls": 32,
                "heldout_forward_calls": 896,
                "heldout_pair_records": 448,
                "ledger_records": 480,
                "total_forward_calls": 928,
            },
            "installed_source": {"sha256": "b" * 64, "version": "0.8.32"},
            "projection": {
                "artifact_digest_sha256": self.projection["artifact_digest_sha256"],
                "parameter_digest_sha256": self.projection["parameter_digest_sha256"],
                "calibration_only": True,
                "frozen_before_heldout_access": True,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        expected["report_digest_sha256"] = report["report_digest_sha256"]
        integrity = {
            "status": "d9d_real_artifact_integrity_complete",
            "execution_claim_sha256": expected["claim_file_sha256"],
            "raw_ledger_sha256": expected["raw_ledger_file_sha256"],
            "projection_artifact_sha256": expected["projection_file_sha256"],
            "report_sha256": expected["report_file_sha256"],
            "d9d_rerun_authorized": False,
            "automatic_rerun_authorized": False,
        }
        integrity["integrity_digest_sha256"] = sha256_json(integrity)
        expected["integrity_digest_sha256"] = integrity["integrity_digest_sha256"]
        checks = validate_evidence_bundle(
            config=config,
            authorization=authorization,
            claim=claim,
            projection=self.projection,
            report=report,
            integrity=integrity,
        )
        self.assertTrue(all(checks.values()))
        tampered = copy.deepcopy(report)
        tampered["valid"] = False
        with self.assertRaises(ValueError):
            validate_evidence_bundle(
                config=config,
                authorization=authorization,
                claim=claim,
                projection=self.projection,
                report=tampered,
                integrity=integrity,
            )


if __name__ == "__main__":
    unittest.main()
