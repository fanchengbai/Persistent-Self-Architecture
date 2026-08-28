from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.d7b_manifest_runtime_contract import (
    CALIBRATION_RECORDS,
    CALIBRATION_RELATIVE_PATH,
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    D7BSymbolicFakeRuntime,
    DESIGN_SHA256,
    FUTURE_JOINT_FORWARD_CALLS,
    HELDOUT_FIXTURES,
    HELDOUT_FORWARD_CALLS,
    HELDOUT_RELATIVE_PATH,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    build_d7b_report,
    build_heldout_call_plan,
    expand_calibration_records,
    expand_heldout_fixtures,
    run_fake_runtime_acceptance,
    validate_calibration_manifest,
    validate_contract_config,
    validate_heldout_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH
CALIBRATION = ROOT / CALIBRATION_RELATIVE_PATH
HELDOUT = ROOT / HELDOUT_RELATIVE_PATH


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class D7BManifestRuntimeContractTests(unittest.TestCase):
    def test_authority_is_exactly_d7b_no_model(self):
        config = _load(CONFIG)
        self.assertTrue(all(validate_contract_config(config).values()))
        self.assertEqual(config["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(config["frozen_design"]["sha256"], DESIGN_SHA256)
        self.assertTrue(config["authority"]["d7b_manifest_implementation_authorized"])
        for field in (
            "installed_source_probe_authorized",
            "real_runner_modification_authorized",
            "projection_implementation_authorized",
            "projection_construction_authorized",
            "model_execution_authorized",
            "d7c_authorized",
            "d7d_authorized",
            "d7e_authorized",
            "d6d_rerun_authorized",
        ):
            self.assertFalse(config["authority"][field])

    def test_manifest_validation_and_scope_mutations_fail_closed(self):
        calibration = _load(CALIBRATION)
        heldout = _load(HELDOUT)
        self.assertTrue(all(validate_calibration_manifest(calibration).values()))
        self.assertTrue(all(validate_heldout_manifest(heldout).values()))
        mutations = (
            (calibration, ("projection_contract", "implemented"), True),
            (calibration, ("record_generator", "record_count"), 16),
            (heldout, ("gates", "D7-C", "authorized"), True),
            (heldout, ("schedule", "heldout_forward_calls_total"), 160),
            (heldout, ("thresholds", "self_effect_conclusion_allowed"), True),
        )
        for payload, path, value in mutations:
            changed = copy.deepcopy(payload)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(PermissionError):
                if payload is calibration:
                    validate_calibration_manifest(changed)
                else:
                    validate_heldout_manifest(changed)

    def test_expansions_are_deterministic_and_counted(self):
        calibration = _load(CALIBRATION)
        heldout = _load(HELDOUT)
        records_a = expand_calibration_records(calibration)
        records_b = expand_calibration_records(calibration)
        fixtures_a = expand_heldout_fixtures(heldout)
        fixtures_b = expand_heldout_fixtures(heldout)
        calls_a = build_heldout_call_plan(heldout)
        calls_b = build_heldout_call_plan(heldout)
        self.assertEqual(records_a, records_b)
        self.assertEqual(fixtures_a, fixtures_b)
        self.assertEqual(calls_a, calls_b)
        self.assertEqual(len(records_a), CALIBRATION_RECORDS)
        self.assertEqual(len(fixtures_a), HELDOUT_FIXTURES)
        self.assertEqual(len(calls_a), HELDOUT_FORWARD_CALLS)
        self.assertEqual(len(records_a) + len(calls_a), FUTURE_JOINT_FORWARD_CALLS)

    def test_training_heldout_and_qualification_prompts_are_separate(self):
        records = expand_calibration_records(_load(CALIBRATION))
        fixtures = expand_heldout_fixtures(_load(HELDOUT))
        calibration_hashes = {record["prompt_sha256"] for record in records}
        hidden_hashes = {fixture["hidden_prompt_sha256"] for fixture in fixtures}
        qualification_hashes = {
            fixture["qualification_prompt_sha256"] for fixture in fixtures
        }
        self.assertFalse(calibration_hashes & hidden_hashes)
        self.assertFalse(calibration_hashes & qualification_hashes)
        self.assertFalse(hidden_hashes & qualification_hashes)
        self.assertEqual(len(hidden_hashes), HELDOUT_FIXTURES)
        self.assertEqual(len(qualification_hashes), HELDOUT_FIXTURES)

    def test_symbolic_fake_runtime_covers_all_conditions_without_projection(self):
        acceptance = run_fake_runtime_acceptance(_load(CALIBRATION), _load(HELDOUT))
        self.assertTrue(acceptance["valid"])
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(acceptance["counts"]["heldout_forward_calls"], 896)
        self.assertEqual(acceptance["counts"]["future_joint_forward_calls"], 921)
        self.assertEqual(set(acceptance["condition_counts"].values()), {64})

    def test_invalid_fake_call_fails_before_ledger_append(self):
        heldout = _load(HELDOUT)
        fixture = expand_heldout_fixtures(heldout)[0]
        call = copy.deepcopy(build_heldout_call_plan(heldout)[0])
        runtime = D7BSymbolicFakeRuntime()
        call["condition"] = "not-a-condition"
        with self.assertRaises(PermissionError):
            runtime.execute(fixture=fixture, call=call)
        self.assertEqual(runtime.ledger, [])

    def test_report_is_offline_and_keeps_all_later_gates_closed(self):
        report = build_d7b_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertTrue(report["decision"]["d7b_implemented"])
        self.assertFalse(report["decision"]["d7c_d7d_d7e_authorized"])
        self.assertFalse(report["decision"]["projection_implemented_or_constructed"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["installed_source_probed"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d7b_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
