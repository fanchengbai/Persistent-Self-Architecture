from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

from psa.self_model.d9b_manifest_endpoint_contract import (
    CALIBRATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    HELDOUT_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    build_contract_report,
    evaluate_fake_ledger,
    make_fake_ledger,
    run_fake_acceptance,
    validate_calibration_manifest,
    validate_contract_config,
    validate_endpoint_manifest,
    validate_fake_ledger,
    validate_heldout_manifest,
    validate_schedule_manifest,
)
from psa.self_model.d9a_within_wrapper_causal_isolation import (
    expand_fixtures,
    expand_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class D9BManifestEndpointContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = load(DESIGN_RELATIVE_PATH)
        cls.fixtures = expand_fixtures(cls.design)
        cls.schedule_expanded = expand_schedule(cls.design, cls.fixtures)
        cls.schedule = load(SCHEDULE_RELATIVE_PATH)

    def test_contract_config_validates_and_rejects_threshold_change(self) -> None:
        config = load(CONFIG_RELATIVE_PATH)
        self.assertTrue(all(validate_contract_config(config).values()))
        changed = copy.deepcopy(config)
        changed["frozen_design"]["thresholds_changed"] = True
        with self.assertRaises(PermissionError):
            validate_contract_config(changed)

    def test_calibration_manifest_exact_and_heldout_unscored(self) -> None:
        manifest = load(CALIBRATION_RELATIVE_PATH)
        self.assertTrue(
            all(validate_calibration_manifest(manifest, self.fixtures).values())
        )
        self.assertEqual(len(manifest["fixtures"]), 32)
        self.assertTrue(all(not item["heldout_scored"] for item in manifest["fixtures"]))

    def test_heldout_manifest_exact_and_four_rotations(self) -> None:
        manifest = load(HELDOUT_RELATIVE_PATH)
        self.assertTrue(all(validate_heldout_manifest(manifest, self.fixtures).values()))
        groups: dict[str, list[dict]] = {}
        for item in manifest["fixtures"]:
            groups.setdefault(item["base_case_id"], []).append(item)
        self.assertEqual(len(groups), 16)
        self.assertTrue(all(len(items) == 4 for items in groups.values()))

    def test_schedule_is_exact_same_wrapper_and_balanced(self) -> None:
        self.assertTrue(
            all(validate_schedule_manifest(self.schedule, self.schedule_expanded).values())
        )
        blocks = self.schedule["heldout_pair_blocks"]
        self.assertEqual(len(blocks), 448)
        self.assertTrue(all(block["route"] == "persistent_wrapper" for block in blocks))
        for contrast in self.schedule["contrasts"]:
            selected = [block for block in blocks if block["contrast"] == contrast]
            self.assertEqual(len(selected), 64)
            self.assertEqual(
                {order: sum(block["pair_order"] == order for block in selected)
                 for order in ("zero_first", "condition_first")},
                {"zero_first": 32, "condition_first": 32},
            )

    def test_endpoint_manifest_copies_d9a_thresholds_exactly(self) -> None:
        endpoint = load(ENDPOINT_RELATIVE_PATH)
        self.assertTrue(all(validate_endpoint_manifest(endpoint, self.design).values()))
        self.assertEqual(endpoint["thresholds"], self.design["endpoint_contract"])

    def test_fake_ledger_has_480_records_representing_928_calls(self) -> None:
        ledger = make_fake_ledger(self.schedule, "field_specific_candidate")
        self.assertTrue(all(validate_fake_ledger(ledger, self.schedule).values()))
        self.assertEqual(len(ledger), 480)
        self.assertEqual(32 + 2 * 448, 928)

    def test_field_specific_candidate_passes_without_self_claim(self) -> None:
        result = evaluate_fake_ledger(
            make_fake_ledger(self.schedule, "field_specific_candidate"),
            self.schedule,
        )
        self.assertTrue(result["all_gates_pass"])
        self.assertFalse(result["self_effect_conclusion"])

    def test_route_only_and_nonspecific_cases_fail(self) -> None:
        for scenario in ("wrapper_route_only", "nonspecific_active_or_random"):
            with self.subTest(scenario=scenario):
                result = evaluate_fake_ledger(
                    make_fake_ledger(self.schedule, scenario), self.schedule
                )
                self.assertFalse(result["all_gates_pass"])
                self.assertFalse(result["self_effect_conclusion"])

    def test_missing_duplicate_reorder_public_nonfinite_and_leak_fail_closed(self) -> None:
        base = make_fake_ledger(self.schedule, "field_specific_candidate")
        variants = []
        variants.append(base[:-1])
        duplicate = copy.deepcopy(base)
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        variants.append(duplicate)
        reordered = copy.deepcopy(base)
        reordered[32], reordered[33] = reordered[33], reordered[32]
        variants.append(reordered)
        public = copy.deepcopy(base)
        public[32]["route"] = "public"
        variants.append(public)
        nonfinite = copy.deepcopy(base)
        nonfinite[32]["observations"][0]["target_alignment_margin"] = math.inf
        variants.append(nonfinite)
        leak = copy.deepcopy(base)
        leak[0]["fixture_id"] = leak[32]["fixture_id"]
        variants.append(leak)
        for variant in variants:
            with self.assertRaises((TypeError, ValueError)):
                evaluate_fake_ledger(variant, self.schedule)

    def test_fake_acceptance_all_categories_pass(self) -> None:
        result = run_fake_acceptance(self.schedule)
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["checks"]), 12)

    def test_report_is_valid_and_has_no_model_side_effects(self) -> None:
        report = build_contract_report(config_path=CONFIG_RELATIVE_PATH, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["fake_acceptance"]["valid"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertTrue(report["safety"]["projection_contract_implemented"])

    def test_wrong_config_path_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            build_contract_report(config_path=DESIGN_RELATIVE_PATH, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
