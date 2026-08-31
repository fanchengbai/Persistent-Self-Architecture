from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

from psa.self_model.d8c_real_numerical_identifiability import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    NEXT_GATE,
    REQUIRED_CONFIRMATION,
    build_call_plan,
    build_static_report,
    validate_authorization_schema,
    validate_call_plan,
    validate_config,
    validate_ledger_order,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class D8CRealEntryTests(unittest.TestCase):
    def test_config_freezes_protocol_and_closes_execution(self):
        config = _load(CONFIG_RELATIVE_PATH)
        self.assertTrue(all(validate_config(config).values()))
        self.assertEqual(config["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(config["future_execution_authorization_text"], FUTURE_EXECUTION_AUTHORIZATION_TEXT)
        self.assertEqual(config["protocol"]["future_forward_calls"], 584)
        self.assertFalse(config["authority"]["model_execution_authorized"])
        self.assertFalse(config["authority"]["d8c_real_execution_authorized"])

    def test_schema_is_new_single_use_future_namespace(self):
        schema = _load("schemas/self_model_v0_1_d8c_real_authorization.schema.json")
        self.assertTrue(all(validate_authorization_schema(schema).values()))
        self.assertTrue(schema["additionalProperties"] is False)
        self.assertEqual(
            schema["properties"]["authorization_text"]["const"],
            FUTURE_EXECUTION_AUTHORIZATION_TEXT,
        )
        self.assertEqual(
            schema["properties"]["scope"]["const"],
            "one_d8c_process_584_forward_calls_no_payload_access",
        )

    def test_call_plan_materializes_584_calls_without_payload(self):
        schedule = _load("configs/development/self_model_v0_1_d8_counterbalanced_schedule.json")
        plan = build_call_plan(schedule)
        self.assertEqual(len(plan), 584)
        checks = validate_call_plan(plan)
        self.assertTrue(all(checks.values()))
        self.assertEqual(sum(item["scored"] is False for item in plan), 8)
        self.assertEqual(sum(item["scored"] is True for item in plan), 576)

    def test_ledger_requires_exact_order_and_count(self):
        schedule = _load("configs/development/self_model_v0_1_d8_counterbalanced_schedule.json")
        plan = build_call_plan(schedule)
        ledger = [{"call_id": item["call_id"]} for item in plan]
        self.assertTrue(validate_ledger_order(plan, ledger)["valid"])
        with self.assertRaises(ValueError):
            validate_ledger_order(plan, ledger[:-1])
        with self.assertRaises(ValueError):
            validate_ledger_order(plan, ledger[:1] + ledger[2:3] + ledger[1:2] + ledger[3:])

    def test_report_is_static_no_model_and_no_artifacts(self):
        report = build_static_report(
            config_path=ROOT / CONFIG_RELATIVE_PATH,
            project_root=ROOT,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertEqual(report["next_gate"], NEXT_GATE)
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["execution_claim_created"])
        self.assertFalse(report["safety"]["payload_accessed"])
        self.assertTrue(all(report["namespace_checks"].values()))

    def test_config_mutations_fail_closed(self):
        config = _load(CONFIG_RELATIVE_PATH)
        for section, field, value in (
            ("authority", "model_execution_authorized", True),
            ("authority", "d8c_real_execution_authorized", True),
            ("protocol", "future_forward_calls", 583),
            ("historical_boundary", "thresholds_changed", True),
        ):
            changed = copy.deepcopy(config)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_plan_mutations_fail_closed(self):
        schedule = _load("configs/development/self_model_v0_1_d8_counterbalanced_schedule.json")
        plan = build_call_plan(schedule)
        changed = list(plan)
        changed[0] = dict(changed[0], scored=True)
        with self.assertRaises(ValueError):
            validate_call_plan(changed)
        changed = list(plan)
        changed[10] = dict(changed[10], call_id=changed[0]["call_id"])
        with self.assertRaises(ValueError):
            validate_call_plan(changed)

    def test_wrong_config_path_rejected(self):
        with self.assertRaises(PermissionError):
            build_static_report(
                config_path=ROOT / "configs/development/self_model_v0_1_d8b_manifest_endpoint_contract.json",
                project_root=ROOT,
            )

    def test_model_modules_not_imported(self):
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
