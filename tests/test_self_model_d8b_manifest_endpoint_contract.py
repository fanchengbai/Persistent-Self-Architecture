from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_file
from psa.self_model.d8_numerical_identifiability_design import (
    expand_fixtures,
    expand_schedule,
)
from psa.self_model.d8b_manifest_endpoint_contract import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    DETERMINISM_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    SCHEDULE_RELATIVE_PATH,
    FakeForwardOutput,
    aggregate_fixture_excess,
    build_contract_report,
    decide_excess_drift,
    output_distance,
    run_fake_acceptance,
    tensor_distance,
    validate_contract_config,
    validate_determinism_manifest,
    validate_endpoint_manifest,
    validate_fixture_manifest,
    validate_pair_ledger,
    validate_schedule_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class D8BManifestEndpointContractTests(unittest.TestCase):
    def test_contract_freezes_design_manifests_counts_and_authority(self):
        payload = _load(CONFIG_RELATIVE_PATH)
        checks = validate_contract_config(payload)
        self.assertTrue(all(checks.values()))
        self.assertEqual(payload["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(payload["counts"]["scored_pairs"], 288)
        self.assertEqual(payload["counts"]["total_future_forward_calls"], 584)
        self.assertFalse(payload["frozen_design"]["thresholds_changed"])
        self.assertFalse(payload["authority"]["d8c_real_execution_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])

    def test_four_manifests_validate_and_file_hashes_are_bound(self):
        contract = _load(CONFIG_RELATIVE_PATH)
        fixtures = _load(FIXTURE_RELATIVE_PATH)
        schedule = _load(SCHEDULE_RELATIVE_PATH)
        determinism = _load(DETERMINISM_RELATIVE_PATH)
        endpoint = _load(ENDPOINT_RELATIVE_PATH)
        self.assertTrue(all(validate_fixture_manifest(fixtures).values()))
        self.assertTrue(all(validate_schedule_manifest(schedule).values()))
        self.assertTrue(all(validate_determinism_manifest(determinism).values()))
        self.assertTrue(all(validate_endpoint_manifest(endpoint).values()))
        for name in ("fixture", "schedule", "determinism", "endpoint"):
            self.assertEqual(
                sha256_file(ROOT / contract["manifests"][f"{name}_path"]),
                contract["manifests"][f"{name}_sha256"],
            )

    def test_manifest_expansion_materializes_exact_counts_and_commitments(self):
        design = _load(DESIGN_RELATIVE_PATH)
        fixtures = expand_fixtures(design)
        schedule = expand_schedule(design, fixtures)
        self.assertEqual(len(fixtures["conditioning_fixtures"]), 4)
        self.assertEqual(len(fixtures["scored_fixtures"]), 24)
        self.assertEqual(len(schedule["conditioning_calls"]), 8)
        self.assertEqual(len(schedule["pair_blocks"]), 288)
        self.assertEqual(
            fixtures["fixture_commitment_sha256"],
            _load(FIXTURE_RELATIVE_PATH)["expanded_fixture_commitment_sha256"],
        )
        self.assertEqual(
            schedule["schedule_commitment_sha256"],
            _load(SCHEDULE_RELATIVE_PATH)["expanded_schedule_commitment_sha256"],
        )

    def test_tensor_and_output_distance_follow_frozen_formula(self):
        self.assertEqual(tensor_distance([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertEqual(tensor_distance([0.0, 2.0], [0.0, 4.0]), 0.5)
        state = tuple((float(index + 1),) for index in range(96))
        changed = list(state)
        changed[94] = (state[94][0] + 5.0,)
        left = FakeForwardOutput((1.0, 2.0), state)
        right = FakeForwardOutput((1.0, 3.0), tuple(changed))
        report = output_distance(left, right)
        self.assertEqual(report["state_component_count"], 96)
        self.assertEqual(report["max_state_component_index"], 94)
        self.assertGreater(report["output_distance"], 0.0)

    def test_distance_rejects_nonfinite_shape_and_non_python_objects(self):
        with self.assertRaises(ValueError):
            tensor_distance([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            tensor_distance([[1.0, 2.0]], [1.0, 2.0])
        with self.assertRaises(ValueError):
            tensor_distance([[1.0], [2.0, 3.0]], [[1.0], [2.0]])
        with self.assertRaises(ValueError):
            tensor_distance([math.nan], [0.0])
        with self.assertRaises(TypeError):
            tensor_distance([True], [False])
        with self.assertRaises(TypeError):
            tensor_distance(object(), object())

    def test_complete_ledger_aggregates_and_decides_positive(self):
        design = _load(DESIGN_RELATIVE_PATH)
        fixtures = expand_fixtures(design)
        schedule = expand_schedule(design, fixtures)
        distances = {
            "public_public": 0.001,
            "wrapper_wrapper": 0.0015,
            "public_wrapper": 0.02,
            "wrapper_public": 0.018,
        }
        ledger = [
            {
                "pair_block_id": block["pair_block_id"],
                "pair_type": block["pair_type"],
                "output_distance": distances[block["pair_type"]],
            }
            for block in schedule["pair_blocks"]
        ]
        snapshot = copy.deepcopy(ledger)
        validated = validate_pair_ledger(schedule, ledger)
        results = aggregate_fixture_excess(schedule, ledger)
        decision = decide_excess_drift(results, _load(ENDPOINT_RELATIVE_PATH))
        self.assertEqual(validated["record_count"], 288)
        self.assertEqual(len(results), 24)
        self.assertTrue(decision["positive"])
        self.assertEqual(decision["positive_fixture_count"], 24)
        self.assertFalse(decision["route_equivalence_claim"])
        self.assertFalse(decision["self_effect_conclusion"])
        self.assertEqual(ledger, snapshot)

    def test_ledger_rejects_missing_duplicate_and_nonfinite(self):
        design = _load(DESIGN_RELATIVE_PATH)
        schedule = expand_schedule(design, expand_fixtures(design))
        ledger = [
            {
                "pair_block_id": block["pair_block_id"],
                "pair_type": block["pair_type"],
                "output_distance": 0.0,
            }
            for block in schedule["pair_blocks"]
        ]
        with self.assertRaises(ValueError):
            validate_pair_ledger(schedule, ledger[:-1])
        duplicate = copy.deepcopy(ledger)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaises(ValueError):
            validate_pair_ledger(schedule, duplicate)
        nonfinite = copy.deepcopy(ledger)
        nonfinite[0]["output_distance"] = float("inf")
        with self.assertRaises(ValueError):
            validate_pair_ledger(schedule, nonfinite)

    def test_fake_acceptance_separates_route_order_and_shared_drift(self):
        result = run_fake_acceptance(
            design=_load(DESIGN_RELATIVE_PATH),
            endpoint=_load(ENDPOINT_RELATIVE_PATH),
        )
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(result["rejection_checks"].values()))
        self.assertTrue(result["scenario_decisions"]["route_specific_excess"]["positive"])
        self.assertFalse(result["scenario_decisions"]["one_order_only"]["positive"])
        self.assertFalse(
            result["scenario_decisions"]["shared_background_repeatability"]["positive"]
        )

    def test_mutations_fail_closed(self):
        contract = _load(CONFIG_RELATIVE_PATH)
        contract_mutations = (
            ("authority", "model_execution_authorized", True),
            ("authority", "d8c_real_execution_authorized", True),
            ("frozen_design", "thresholds_changed", True),
            ("counts", "scored_pairs", 287),
        )
        for section, field, value in contract_mutations:
            changed = copy.deepcopy(contract)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_contract_config(changed)
        endpoint = _load(ENDPOINT_RELATIVE_PATH)
        changed_endpoint = copy.deepcopy(endpoint)
        changed_endpoint["primary"]["minimum_positive_fixtures"] = 20
        with self.assertRaises(PermissionError):
            validate_endpoint_manifest(changed_endpoint)

    def test_report_is_no_model_and_all_gates_closed(self):
        report = build_contract_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertFalse(report["safety"]["installed_source_probed"])
        self.assertFalse(report["safety"]["execution_entry_implemented"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d8c_real_execution_authorized"])
        self.assertFalse(report["safety"]["d7c_rerun"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_contract_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("PSA_SELF_MODEL_D8_REAL", os.environ)


if __name__ == "__main__":
    unittest.main()
