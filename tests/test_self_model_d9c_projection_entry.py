from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes, sha256_json
from psa.self_model.d9c_projection_contract import (
    CALIBRATION_COMMITMENT,
    CALIBRATION_SHA256,
    CONTRACT_RELATIVE_PATH,
    HELDOUT_COMMITMENT,
    HELDOUT_SHA256,
    SCHEDULE_COMMITMENT,
    CalibrationCapture,
    audit_frozen_projection_artifact,
    build_frozen_projection_artifact,
    project_condition,
    run_fake_projection_acceptance,
    validate_projection_contract,
    verify_projection_contract_files,
)
from psa.self_model.d9c_real_entry import (
    AUTHORIZATION_FIELDS,
    CONFIG_RELATIVE_PATH,
    EXECUTION_LOCK_ENV,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    _authorization_payload,
    _execution_artifacts_absent,
    _margins,
    _target_codes,
    build_call_plan,
    build_d9_authorization,
    build_static_report,
    run_d9d_real_causal_isolation,
    run_pure_python_acceptance,
    validate_call_plan,
    validate_config,
    validate_d9_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class D9CProjectionEntryTests(unittest.TestCase):
    def test_projection_contract_is_exact(self) -> None:
        contract = load(CONTRACT_RELATIVE_PATH)
        self.assertTrue(all(validate_projection_contract(contract).values()))

    def test_projection_contract_rejects_heldout_early_access(self) -> None:
        contract = load(CONTRACT_RELATIVE_PATH)
        changed = copy.deepcopy(contract)
        changed["capture_contract"]["heldout_payload_access_before_artifact_freeze"] = True
        with self.assertRaises(PermissionError):
            validate_projection_contract(changed)

    def test_fake_projection_acceptance_passes(self) -> None:
        result = run_fake_projection_acceptance()
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["real_projection_constructed"])

    def test_projection_files_bind_frozen_manifests(self) -> None:
        result = verify_projection_contract_files(ROOT)
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["source_checks"].values()))

    def test_real_dimension_rejects_fake_width(self) -> None:
        captures = [
            CalibrationCapture(
                fixture_id=f"d9cal-{index + 1:03d}",
                identity_index=(index // 8),
                goal_index=(index // 2) % 4,
                replicate=index % 2 + 1,
                vector=tuple(1.0 + index * 0.01 + j * 0.02 for j in range(8)),
            )
            for index in range(32)
        ]
        with self.assertRaises(ValueError):
            build_frozen_projection_artifact(
                captures=captures,
                calibration_manifest_sha256=CALIBRATION_SHA256,
                calibration_commitment_sha256=CALIBRATION_COMMITMENT,
                heldout_manifest_sha256=HELDOUT_SHA256,
                heldout_commitment_sha256=HELDOUT_COMMITMENT,
                schedule_commitment_sha256=SCHEDULE_COMMITMENT,
                output_dimension=8,
                fixture_only=False,
            )

    def test_entry_config_is_exact_and_later_authority_closed(self) -> None:
        config = load(CONFIG_RELATIVE_PATH)
        checks = validate_config(config)
        self.assertTrue(all(checks.values()))
        changed = copy.deepcopy(config)
        changed["implementation_authority"]["d9d_real_execution_authorized"] = True
        with self.assertRaises(PermissionError):
            validate_config(changed)

    def test_call_plan_is_928_and_same_wrapper_only(self) -> None:
        schedule = load("configs/development/self_model_v0_1_d9_within_wrapper_schedule.json")
        plan = build_call_plan(schedule)
        self.assertTrue(all(validate_call_plan(plan).values()))
        self.assertEqual(len(plan), 928)
        self.assertTrue(all("public" not in item["route"] for item in plan))

    def test_target_code_rotation_and_field_alternatives_are_predeclared(self) -> None:
        codes = _target_codes(1, 2, 3)
        self.assertEqual(codes, {"true": "A", "identity_swap": "D", "goal_swap": "B"})
        margins = _margins(
            {"A": 4.0, "B": 1.0, "C": 0.0, "D": 2.0}, codes
        )
        self.assertEqual(margins["target_alignment_margin"], 2.0)
        self.assertEqual(margins["identity_margin"], 2.0)
        self.assertEqual(margins["goal_margin"], 3.0)

    def test_call_plan_missing_duplicate_reorder_and_public_fail(self) -> None:
        schedule = load("configs/development/self_model_v0_1_d9_within_wrapper_schedule.json")
        plan = build_call_plan(schedule)
        variants = [plan[:-1]]
        duplicate = copy.deepcopy(plan)
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        variants.append(duplicate)
        reordered = copy.deepcopy(plan)
        reordered[32], reordered[33] = reordered[33], reordered[32]
        variants.append(reordered)
        public = copy.deepcopy(plan)
        public[32]["route"] = "public"
        variants.append(public)
        for value in variants:
            with self.assertRaises(ValueError):
                validate_call_plan(value)

    def test_authorization_payload_is_exact_and_tamper_evident(self) -> None:
        payload = _authorization_payload(
            git_commit="a" * 40,
            entry_static_report_sha256="b" * 64,
            authorized_at_utc="2026-09-04T00:00:00+00:00",
        )
        self.assertEqual(set(payload), AUTHORIZATION_FIELDS)
        stored = payload["authorization_digest_sha256"]
        unsigned = {key: value for key, value in payload.items()
                    if key != "authorization_digest_sha256"}
        self.assertEqual(stored, sha256_json(unsigned))
        payload["model_forward_calls"] = 927
        unsigned = {key: value for key, value in payload.items()
                    if key != "authorization_digest_sha256"}
        self.assertNotEqual(stored, sha256_json(unsigned))

    def test_build_authorization_requires_exact_future_text(self) -> None:
        fake_git = {"commit": "a" * 40, "branch": "main", "status": ""}
        with self.assertRaises(PermissionError):
            build_d9_authorization(
                project_root=ROOT,
                authorization_text="continue",
                entry_static_report_sha256="b" * 64,
                git=fake_git,
            )
        payload = build_d9_authorization(
            project_root=ROOT,
            authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
            entry_static_report_sha256="b" * 64,
            git=fake_git,
        )
        self.assertTrue(payload["authorized"])

    def test_authorization_validation_rejects_tamper(self) -> None:
        git = {"commit": "a" * 40, "branch": "main", "status": ""}
        payload = _authorization_payload(
            git_commit=git["commit"],
            entry_static_report_sha256="b" * 64,
            authorized_at_utc="2026-09-04T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "authorization.json"
            path.write_bytes(canonical_json_bytes(payload))
            self.assertEqual(
                validate_d9_authorization(
                    authorization_path=path, project_root=ROOT, git=git
                ),
                payload,
            )
            payload["model_forward_calls"] = 927
            path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaises(PermissionError):
                validate_d9_authorization(
                    authorization_path=path, project_root=ROOT, git=git
                )

    def test_pure_python_entry_acceptance_passes(self) -> None:
        result = run_pure_python_acceptance(ROOT)
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["counts"]["total_calls"], 928)

    def test_execution_artifacts_are_absent(self) -> None:
        self.assertTrue(all(_execution_artifacts_absent(ROOT).values()))
        self.assertFalse((ROOT / OUTPUT_RELATIVE_DIR / "projection.json").exists())

    def test_real_entry_stops_before_probe_without_future_lock(self) -> None:
        old = os.environ.pop(EXECUTION_LOCK_ENV, None)
        try:
            with self.assertRaises(PermissionError):
                run_d9d_real_causal_isolation(
                    config_path=CONFIG_RELATIVE_PATH,
                    authorization_path="results/authorizations/self_model_v0_1_d9_real_v01.json",
                    project_root=ROOT,
                    output_dir=OUTPUT_RELATIVE_DIR,
                )
        finally:
            if old is not None:
                os.environ[EXECUTION_LOCK_ENV] = old

    def test_entry_source_has_no_static_torch_or_rwkv_import(self) -> None:
        source = (ROOT / "src/psa/self_model/d9c_real_entry.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertNotIn("torch", imports)
        self.assertNotIn("rwkv.model", imports)

    def test_static_report_is_valid_and_no_model_effects(self) -> None:
        report = build_static_report(
            config_path=CONFIG_RELATIVE_PATH, project_root=ROOT
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["real_projection_constructed"])

    def test_wrong_config_path_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            build_static_report(config_path=CONTRACT_RELATIVE_PATH, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
