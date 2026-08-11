from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.development.exp001c_v02_stage_b_design import (
    build_exp001c_v02_stage_b_design_manifest,
)
from psa.development.exp001c_v02_stage_b_run import (
    STAGE_B_EXECUTION_LOCK,
    run_exp001c_v02_stage_b,
    verify_exp001c_v02_stage_b_result,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_CONFIG = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_v02_stage_b_design.draft.json"
)
AUTHORITY_TARGET = (
    "psa.development.exp001c_v02_stage_b_run."
    "validate_exp001c_v02_stage_b_authority"
)


def _write_design(directory: Path):
    design = build_exp001c_v02_stage_b_design_manifest(
        design_config_path=DESIGN_CONFIG,
        project_root=ROOT,
    )
    path = directory / "stage_b_design_manifest.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    return design, path


def _authority(design):
    return {
        "valid": True,
        "experiment_id": "EXP-001C",
        "scope": "v02_stage_b_recurrent_state_noncore_pilot_once",
        "design_manifest_digest_sha256": design[
            "design_manifest_digest_sha256"
        ],
        "preflight_digest_sha256": "1" * 64,
        "stage_a_result_sha256": "2" * 64,
        "model_execution_authorized": True,
        "stage_b_result_observation_authorized": False,
        "stage_a_rerun_authorized": False,
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
    }


def _prefix_evidence():
    return {
        "instrumentation_version": "0.1-development",
        "development_only": True,
        "text": ">\n",
        "token_ids": [4, 3],
        "greedy_token_ids": [4, 3],
        "greedy_exact": True,
        "roundtrip_exact": True,
        "top_k": 10,
        "positions": [{"position_index": 0}, {"position_index": 1}],
    }


def _valid_result(design):
    records = []
    for route in design["records"]:
        target = route["expected_state_semantic_target_code"]
        predicted = target or "A"
        scores = {code: -4.0 for code in "ABCD"}
        scores[predicted] = -0.1
        if target is None:
            boundary = None
        else:
            incorrect = [code for code in "ABCD" if code != target]
            best_incorrect = max(incorrect, key=lambda code: scores[code])
            boundary = {
                "target_code": target,
                "target_answer_log_probability": scores[target],
                "best_incorrect_code": best_incorrect,
                "best_incorrect_answer_log_probability": scores[
                    best_incorrect
                ],
                "target_margin_over_best_incorrect": (
                    scores[target] - scores[best_incorrect]
                ),
            }
        records.append(
            {
                "record_id": route["record_id"],
                "condition": route["condition"],
                "condition_role": route["condition_role"],
                "query_sample_id": route["query_sample_id"],
                "semantic_case_id": route["semantic_case_id"],
                "block_id": route["block_id"],
                "rotation_index": route["rotation_index"],
                "query_history_key": route["query_history_key"],
                "state_source_sample_id": route["state_source_sample_id"],
                "state_source_history_key": route[
                    "state_source_history_key"
                ],
                "state_source_fields": route["state_source_fields"],
                "reference_stage_a_target_code": route[
                    "reference_stage_a_target_code"
                ],
                "expected_state_semantic_target_code": target,
                "semantic_endpoint_role": route["semantic_endpoint_role"],
                "query_token_count": 10,
                "prefix_evidence": _prefix_evidence(),
                "option_log_probabilities": scores,
                "predicted_code": predicted,
                "answer_boundary_evidence": boundary,
            }
        )
    return {
        "result_version": "0.2-stage-b-development",
        "experiment_id": "EXP-001C",
        "status": "v02_stage_b_recurrent_state_complete",
        "development_only": True,
        "non_core": True,
        "model_executed": True,
        "recurrent_state_accessed": True,
        "source_states_cloned_per_route": True,
        "stage_a_rerun": False,
        "formal_test_set_accessed": False,
        "formal_run": False,
        "contains_confirmatory_decision": False,
        "automatic_rerun_authorized": False,
        "design_manifest_digest_sha256": design[
            "design_manifest_digest_sha256"
        ],
        "protocol_manifest_digest_sha256": design[
            "protocol_manifest_digest_sha256"
        ],
        "condition_count": 7,
        "record_count": 224,
        "warmup_token_lengths": [10, 20],
        "snapshot_roundtrip_reports": {
            "block-000": {},
            "block-001": {},
        },
        "records": records,
    }


class _FakeBackend:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def run_stage_b(self, design_manifest):
        del design_manifest
        self.calls += 1
        return copy.deepcopy(self.result)


class Exp001CV02StageBRunTests(unittest.TestCase):
    def test_runner_writes_claim_result_verification_and_summary_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, design_path = _write_design(root)
            backend = _FakeBackend(_valid_result(design))
            factory_authority = []

            def factory(authority_validated):
                factory_authority.append(authority_validated)
                return backend

            with patch(AUTHORITY_TARGET, return_value=_authority(design)):
                summary = run_exp001c_v02_stage_b(
                    design_manifest_path=design_path,
                    preflight_path="future-preflight.json",
                    authorization_path="future-authorization.json",
                    model_config_path="future-model.json",
                    output_dir=root / "output",
                    backend_factory=factory,
                    execution_lock=STAGE_B_EXECUTION_LOCK,
                    project_root=ROOT,
                )
            self.assertTrue(summary["valid"])
            self.assertEqual(factory_authority, [True])
            self.assertEqual(backend.calls, 1)
            self.assertFalse(summary["contains_derived_accuracy"])
            self.assertFalse(summary["contains_research_decision"])
            self.assertFalse(summary["stage_b_result_observation_authorized"])
            output = root / "output"
            self.assertTrue((output / "execution_claim.json").is_file())
            self.assertTrue((output / "stage_b_result.json").is_file())
            self.assertTrue((output / "result_verification.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            claim = json.loads(
                (output / "execution_claim.json").read_text(encoding="utf-8")
            )
            self.assertTrue(claim["single_use"])
            self.assertFalse(claim["automatic_rerun_authorized"])

            with patch(AUTHORITY_TARGET, return_value=_authority(design)):
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    run_exp001c_v02_stage_b(
                        design_manifest_path=design_path,
                        preflight_path="future-preflight.json",
                        authorization_path="future-authorization.json",
                        model_config_path="future-model.json",
                        output_dir=output,
                        backend_factory=factory,
                        execution_lock=STAGE_B_EXECUTION_LOCK,
                        project_root=ROOT,
                    )
            self.assertEqual(backend.calls, 1)

    def test_execution_lock_is_checked_before_paths_or_factory(self) -> None:
        factory_called = False

        def factory(authority_validated):
            del authority_validated
            nonlocal factory_called
            factory_called = True
            raise AssertionError("factory must remain unreachable")

        with self.assertRaisesRegex(PermissionError, "lock is absent"):
            run_exp001c_v02_stage_b(
                design_manifest_path="missing-design.json",
                preflight_path="missing-preflight.json",
                authorization_path="missing-authorization.json",
                model_config_path="missing-model.json",
                output_dir="missing-output",
                backend_factory=factory,
                execution_lock="",
                project_root=ROOT,
            )
        self.assertFalse(factory_called)

    def test_live_authority_boundary_remains_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, design_path = _write_design(root)
            factory_called = False

            def factory(authority_validated):
                del authority_validated
                nonlocal factory_called
                factory_called = True
                raise AssertionError("factory must remain unreachable")

            with self.assertRaisesRegex(PermissionError, "preflight"):
                run_exp001c_v02_stage_b(
                    design_manifest_path=design_path,
                    preflight_path="missing-preflight.json",
                    authorization_path="missing-authorization.json",
                    model_config_path="missing-model.json",
                    output_dir=root / "output",
                    backend_factory=factory,
                    execution_lock=STAGE_B_EXECUTION_LOCK,
                    project_root=ROOT,
                )
            self.assertFalse(factory_called)
            self.assertFalse((root / "output" / "execution_claim.json").exists())

    def test_invalid_backend_result_consumes_claim_and_blocks_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, design_path = _write_design(root)
            invalid = _valid_result(design)
            invalid["records"][0]["condition"] = "reset"
            backend = _FakeBackend(invalid)
            output = root / "output"
            with patch(AUTHORITY_TARGET, return_value=_authority(design)):
                with self.assertRaisesRegex(ValueError, "locked contract"):
                    run_exp001c_v02_stage_b(
                        design_manifest_path=design_path,
                        preflight_path="future-preflight.json",
                        authorization_path="future-authorization.json",
                        model_config_path="future-model.json",
                        output_dir=output,
                        backend_factory=lambda validated: backend,
                        execution_lock=STAGE_B_EXECUTION_LOCK,
                        project_root=ROOT,
                    )
            self.assertTrue((output / "execution_claim.json").is_file())
            self.assertFalse((output / "stage_b_result.json").exists())
            with patch(AUTHORITY_TARGET, return_value=_authority(design)):
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    run_exp001c_v02_stage_b(
                        design_manifest_path=design_path,
                        preflight_path="future-preflight.json",
                        authorization_path="future-authorization.json",
                        model_config_path="future-model.json",
                        output_dir=output,
                        backend_factory=lambda validated: backend,
                        execution_lock=STAGE_B_EXECUTION_LOCK,
                        project_root=ROOT,
                    )
            self.assertEqual(backend.calls, 1)

    def test_independent_verifier_rejects_tampered_route_without_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, design_path = _write_design(root)
            result_path = root / "result.json"
            result = _valid_result(design)
            result_path.write_text(json.dumps(result), encoding="utf-8")
            verification = verify_exp001c_v02_stage_b_result(
                result_path=result_path,
                design_manifest_path=design_path,
                project_root=ROOT,
            )
            self.assertTrue(verification["valid"])
            self.assertFalse(verification["contains_derived_accuracy"])
            self.assertFalse(verification["contains_research_decision"])

            result["records"][0]["state_source_history_key"] = "tampered"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            verification = verify_exp001c_v02_stage_b_result(
                result_path=result_path,
                design_manifest_path=design_path,
                project_root=ROOT,
            )
            self.assertFalse(verification["valid"])
            self.assertFalse(verification["record_inventory_valid"])

    def test_execution_claim_schema_is_valid_json(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "exp001c_v02_stage_b_execution_claim.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(schema["type"], "object")
        self.assertTrue(schema["properties"]["single_use"]["const"])


if __name__ == "__main__":
    unittest.main()
