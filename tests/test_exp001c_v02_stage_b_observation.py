from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import sha256_file
from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)
from psa.development.exp001c_v02_stage_b_design import (
    build_exp001c_v02_stage_b_design_manifest,
)
from psa.development.exp001c_v02_stage_b_observation import (
    DIAGNOSTIC_CONDITIONS,
    SEMANTIC_CONDITIONS,
    analyze_exp001c_v02_stage_b,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_CONFIG = (
    ROOT / "configs" / "development" / "exp001c_v02_stage_b_design.draft.json"
)
PROTOCOL_CONFIG = (
    ROOT / "configs" / "development" / "exp001c_noncore_protocol_v02.draft.json"
)
ANALYSIS_CONFIG = (
    ROOT / "configs" / "analysis" / "exp001c_v02_stage_b_observation_v01.json"
)


def _prefix():
    return {
        "instrumentation_version": "0.1-development",
        "development_only": True,
        "text": ">\n",
        "token_ids": [1, 2],
        "greedy_token_ids": [1, 2],
        "greedy_exact": True,
        "roundtrip_exact": True,
        "top_k": 10,
        "positions": [{"position_index": 0}, {"position_index": 1}],
    }


def _fixture(directory: Path):
    design = build_exp001c_v02_stage_b_design_manifest(
        design_config_path=DESIGN_CONFIG,
        project_root=ROOT,
    )
    protocol = build_exp001c_protocol_v02_manifest(
        config_path=PROTOCOL_CONFIG,
        project_root=ROOT,
    )
    trials = {trial["sample_id"]: trial for trial in protocol["trials"]}
    records = []
    for route in design["records"]:
        trial = trials[route["query_sample_id"]]
        selected = (
            route["expected_state_semantic_target_code"]
            if route["condition"] in SEMANTIC_CONDITIONS
            else trial["target_code"]
        )
        scores = {code: -5.0 for code in "ABCD"}
        scores[selected] = 0.0
        records.append(
            {
                key: route[key]
                for key in (
                    "record_id",
                    "condition",
                    "condition_role",
                    "query_sample_id",
                    "semantic_case_id",
                    "block_id",
                    "rotation_index",
                    "query_history_key",
                    "state_source_sample_id",
                    "state_source_history_key",
                    "state_source_fields",
                    "reference_stage_a_target_code",
                    "expected_state_semantic_target_code",
                    "semantic_endpoint_role",
                )
            }
            | {
                "query_token_count": 10,
                "prefix_evidence": _prefix(),
                "option_log_probabilities": scores,
                "predicted_code": selected,
                "answer_boundary_evidence": (
                    {
                        "target_code": selected,
                        "target_answer_log_probability": 0.0,
                        "best_incorrect_code": next(
                            code for code in "ABCD" if code != selected
                        ),
                        "best_incorrect_answer_log_probability": -5.0,
                        "target_margin_over_best_incorrect": 5.0,
                    }
                    if route["condition"] in SEMANTIC_CONDITIONS
                    else None
                ),
            }
        )
    result = {
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
        "design_manifest_digest_sha256": design["design_manifest_digest_sha256"],
        "protocol_manifest_digest_sha256": protocol["manifest_digest_sha256"],
        "condition_count": 7,
        "record_count": 224,
        "warmup_token_lengths": [10],
        "snapshot_roundtrip_reports": {"block-000": {}, "block-001": {}},
        "records": records,
    }
    design_path = directory / "design.json"
    result_path = directory / "result.json"
    summary_path = directory / "summary.json"
    config_path = directory / "analysis.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_sha = sha256_file(result_path)
    summary_path.write_text(
        json.dumps(
            {
                "valid": True,
                "status": "stage_b_raw_result_complete_verified_unobserved",
                "stage_b_result_observation_authorized": True,
                "stage_b_result_sha256": result_sha,
                "record_count": 224,
                "stage_a_rerun": False,
                "formal_test_set_accessed": False,
                "formal_run": False,
                "automatic_rerun_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    config = json.loads(ANALYSIS_CONFIG.read_text(encoding="utf-8"))
    config["expected_stage_b_result_sha256"] = result_sha
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, design_path, result_path, summary_path


class Exp001CV02StageBObservationTests(unittest.TestCase):
    def test_rotation_marginalized_observation_and_diagnostic_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            report = analyze_exp001c_v02_stage_b(
                analysis_config_path=paths[0],
                design_manifest_path=paths[1],
                stage_b_result_path=paths[2],
                stage_b_summary_path=paths[3],
                protocol_config_path=PROTOCOL_CONFIG,
                project_root=ROOT,
            )
        self.assertTrue(report["valid"])
        self.assertTrue(report["result_observed"])
        self.assertFalse(report["model_executed_by_analysis"])
        self.assertFalse(report["contains_confirmatory_decision"])
        for condition in SEMANTIC_CONDITIONS:
            values = report["condition_reports"][condition]
            self.assertEqual(values["label_marginalized_joint_accuracy"], 1.0)
            self.assertIsNone(values["diagnostic_reference_match_rate"])
        for condition in DIAGNOSTIC_CONDITIONS:
            values = report["condition_reports"][condition]
            self.assertIsNone(values["label_marginalized_joint_accuracy"])
            self.assertEqual(values["diagnostic_reference_match_rate"], 1.0)
        self.assertEqual(
            report["descriptive_contrasts"][
                "continuous_restored_prediction_agreement"
            ],
            1.0,
        )

    def test_result_digest_drift_fails_before_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = list(_fixture(Path(directory)))
            config = json.loads(paths[0].read_text(encoding="utf-8"))
            config["expected_stage_b_result_sha256"] = "0" * 64
            paths[0].write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                analyze_exp001c_v02_stage_b(
                    analysis_config_path=paths[0],
                    design_manifest_path=paths[1],
                    stage_b_result_path=paths[2],
                    stage_b_summary_path=paths[3],
                    protocol_config_path=PROTOCOL_CONFIG,
                    project_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
