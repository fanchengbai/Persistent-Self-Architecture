from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.development.exp001c_v02_stage_b_design import (
    STAGE_B_CONDITIONS,
    STAGE_B_DESIGN_SOURCE_FILES,
    STATE_SEMANTIC_CONDITIONS,
    build_exp001c_v02_stage_b_design_manifest,
    verify_exp001c_v02_stage_b_design_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_v02_stage_b_design.draft.json"
)


class Exp001CV02StageBDesignTests(unittest.TestCase):
    def test_builds_balanced_offline_only_record_plan(self) -> None:
        manifest = build_exp001c_v02_stage_b_design_manifest(
            design_config_path=CONFIG,
            project_root=ROOT,
        )
        self.assertEqual(manifest["condition_count"], 7)
        self.assertEqual(manifest["record_count"], 224)
        self.assertEqual(manifest["conditions"], list(STAGE_B_CONDITIONS))
        self.assertEqual(
            Counter(record["condition"] for record in manifest["records"]),
            Counter({condition: 32 for condition in STAGE_B_CONDITIONS}),
        )
        self.assertFalse(manifest["stage_a_rerun_included"])
        self.assertNotIn("prompt_visible_reset", manifest["conditions"])
        self.assertFalse(manifest["model_executed"])
        self.assertFalse(manifest["execution_authorized"])
        self.assertFalse(manifest["result_observation_authorized"])
        self.assertFalse(manifest["formal_test_set_accessed"])
        self.assertFalse(manifest["formal_run_authorized"])
        self.assertFalse(manifest["automatic_rerun_authorized"])

    def test_state_sources_and_expected_codes_follow_each_intervention(self) -> None:
        manifest = build_exp001c_v02_stage_b_design_manifest(
            design_config_path=CONFIG,
            project_root=ROOT,
        )
        by_query = defaultdict(dict)
        for record in manifest["records"]:
            by_query[record["query_sample_id"]][record["condition"]] = record
        self.assertEqual(len(by_query), 32)
        for condition_records in by_query.values():
            self.assertEqual(set(condition_records), set(STAGE_B_CONDITIONS))
            continuous = condition_records["continuous"]
            restored = condition_records["restored"]
            self.assertEqual(
                continuous["state_source_fields"],
                restored["state_source_fields"],
            )
            self.assertEqual(
                continuous["expected_state_semantic_target_code"],
                continuous["reference_stage_a_target_code"],
            )
            for condition in STATE_SEMANTIC_CONDITIONS:
                record = condition_records[condition]
                self.assertIn(
                    record["expected_state_semantic_target_code"],
                    set("ABCD"),
                )
                self.assertIsNotNone(record["state_source_sample_id"])
                self.assertEqual(
                    record["semantic_endpoint_role"],
                    "state_faithful_primary",
                )
            for condition in ("reset", "random_matched"):
                record = condition_records[condition]
                self.assertIsNone(record["expected_state_semantic_target_code"])
                self.assertEqual(
                    record["semantic_endpoint_role"],
                    "diagnostic_control",
                )
            self.assertIsNone(condition_records["reset"]["state_source_sample_id"])
            self.assertIsNotNone(
                condition_records["random_matched"]["state_source_sample_id"]
            )

    def test_each_condition_retains_complete_code_rotation(self) -> None:
        manifest = build_exp001c_v02_stage_b_design_manifest(
            design_config_path=CONFIG,
            project_root=ROOT,
        )
        grouped = defaultdict(list)
        for record in manifest["records"]:
            grouped[(record["condition"], record["semantic_case_id"])].append(
                record
            )
        self.assertEqual(len(grouped), 56)
        for records in grouped.values():
            self.assertEqual(len(records), 4)
            self.assertEqual(
                Counter(record["rotation_index"] for record in records),
                Counter(range(4)),
            )

    def test_verifier_locks_sources_and_fails_closed_on_execution_flag(self) -> None:
        manifest = build_exp001c_v02_stage_b_design_manifest(
            design_config_path=CONFIG,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_b_design_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_v02_stage_b_design_manifest(
                path,
                project_root=ROOT,
            )
            self.assertTrue(verification["valid"])
            self.assertTrue(verification["deterministic_payload_valid"])
            self.assertEqual(
                set(verification["source_checks"]),
                set(STAGE_B_DESIGN_SOURCE_FILES),
            )

            manifest["execution_authorized"] = True
            manifest["design_manifest_digest_sha256"] = sha256_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "design_manifest_digest_sha256"
                }
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_v02_stage_b_design_manifest(
                path,
                project_root=ROOT,
            )
            self.assertFalse(verification["valid"])
            self.assertFalse(verification["safety_boundary_valid"])

    def test_new_schemas_are_valid_json(self) -> None:
        for name in (
            "exp001c_v02_stage_b_authorization.schema.json",
            "exp001c_v02_stage_b_design.schema.json",
        ):
            schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
            self.assertEqual(schema["type"], "object")


if __name__ == "__main__":
    unittest.main()
