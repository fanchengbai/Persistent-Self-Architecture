from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from psa.artifacts import canonical_json_bytes, payload_digest, sha256_file, sha256_json
from psa.cli import main
from psa.preregistration.formal_freeze import _load_formal_config, generate_control_manifest
from psa.supplemental.set_generation import (
    CONDITIONS,
    CONTROL_MANIFEST_DIGEST,
    EXPECTED_COUNTS,
    FINAL_PREREGISTRATION_DIGEST,
    PARENT_CORE_SET_DIGEST,
    PARENT_CORE_SET_PACKAGE_DIGEST,
    SET_STATUS,
    _balanced_assignments,
    _control_source_combo,
    _validate_authorization,
    _validate_payload,
    expected_set_authorization_text,
    generate_and_freeze_exp001b_supplemental_set,
    verify_exp001b_supplemental_set_package,
)


ROOT = Path(__file__).resolve().parents[1]


class Exp001BSetGenerationTests(unittest.TestCase):
    PREFLIGHT_DIGEST = "a" * 64

    def _synthetic_payload(self) -> dict:
        matched = [
            {
                "record_id": f"matched-{index}",
                "record_kind": "matched_context",
                "source_trial_id": f"source-{index}",
                "token_count_exact": True,
                "filler_copied_exactly": True,
            }
            for index in range(5120)
        ]
        generated = [
            {
                "record_id": f"generated-{index}",
                "record_kind": "formal_generation_readout",
                "source_trial_id": f"source-{index}",
            }
            for index in range(5120)
        ]
        assignments = _balanced_assignments(
            [f"coregrp-{index:03d}" for index in range(320)], 1006027012
        )
        controls = [
            {
                "record_id": f"control-{trial_index}-{condition}",
                "record_kind": "general_capability_control_condition",
                "source_control_sample_id": f"control-source-{trial_index}",
                "condition": condition,
                "assigned_factorial_group_id": assignment["factorial_group_id"],
                "assigned_source_combo": assignment["source_combo"],
            }
            for trial_index, assignment in enumerate(assignments)
            for condition in CONDITIONS
        ]
        payload = {
            "supplemental_set_version": "1.0",
            "experiment_id": "EXP-001B",
            "status": SET_STATUS,
            "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
            "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
            "set_preflight_digest_sha256": self.PREFLIGHT_DIGEST,
            "record_counts": dict(EXPECTED_COUNTS),
            "records": {
                "matched_context": matched,
                "formal_generation": generated,
                "controls": controls,
            },
            "supplemental_experiment_authorized": False,
            "supplemental_experiment_run": False,
            "supplemental_results_observed": False,
        }
        payload["supplemental_set_digest_sha256"] = sha256_json(payload)
        return payload

    def _authorization(self) -> dict:
        return {
            "authorization_version": "0.1",
            "experiment_id": "EXP-001B",
            "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
            "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
            "set_preflight_digest_sha256": self.PREFLIGHT_DIGEST,
            "authorized_by_role": "project_owner",
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization_text": expected_set_authorization_text(
                self.PREFLIGHT_DIGEST
            ),
            "authorization": {
                "generate_and_freeze_supplemental_set": True,
                "run_supplemental_experiment": False,
            },
            "total_record_count": 11008,
        }

    def test_authorization_is_exact_and_does_not_authorize_run(self) -> None:
        value = self._authorization()
        _validate_authorization(value)
        changed = dict(value)
        changed["authorization"] = {
            "generate_and_freeze_supplemental_set": True,
            "run_supplemental_experiment": True,
        }
        with self.assertRaisesRegex(ValueError, "authorization is invalid"):
            _validate_authorization(changed)

    def test_authorization_schema_requires_preflight_binding(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas/exp001b_supplemental_set_authorization.schema.json"
            ).read_text(encoding="utf-8")
        )
        properties = schema["properties"]
        self.assertEqual(
            properties["set_preflight_digest_sha256"]["pattern"],
            "^[0-9a-f]{64}$",
        )
        self.assertIn(
            self.PREFLIGHT_DIGEST,
            expected_set_authorization_text(self.PREFLIGHT_DIGEST),
        )
        self.assertEqual(
            properties["authorization"]["const"],
            self._authorization()["authorization"],
        )

    def test_execution_lock_fails_before_any_package_is_read(self) -> None:
        with self.assertRaisesRegex(PermissionError, "execution lock"):
            generate_and_freeze_exp001b_supplemental_set(
                final_package_dir="missing-final",
                core_set_package_dir="missing-core",
                authorization_path="missing-authorization",
                formal_config_path="missing-config",
                output_dir="missing-output",
                token_counter=len,
                tokenizer_provenance={},
                execution_lock="",
                project_root=ROOT,
            )

    def test_cli_lock_fails_before_tokenizer_or_package_loading(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                main(
                    [
                        "exp001b-set-generate",
                        "--final-package", "missing-final",
                        "--core-set-package", "missing-core",
                        "--authorization", "missing-authorization",
                        "--formal-config", "missing-formal-config",
                        "--model-config", "missing-model-config",
                        "--output-dir", "missing-output",
                    ]
                ),
                2,
            )

    def test_generation_rejects_authorization_for_an_old_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            authorization = Path(directory) / "authorization.json"
            authorization.write_bytes(canonical_json_bytes(self._authorization()))
            with mock.patch(
                "psa.supplemental.set_generation.build_exp001b_set_preflight",
                return_value={"valid": True, "preflight_digest_sha256": "b" * 64},
            ):
                with self.assertRaisesRegex(ValueError, "live preflight"):
                    generate_and_freeze_exp001b_supplemental_set(
                        final_package_dir="missing-final",
                        core_set_package_dir="missing-core",
                        authorization_path=authorization,
                        formal_config_path="missing-config",
                        output_dir=Path(directory) / "output",
                        token_counter=len,
                        tokenizer_provenance={},
                        execution_lock="AUTHORIZED_EXP001B_SET_GENERATION",
                        project_root=ROOT,
                    )

    def test_control_assignment_is_reproducible_unique_and_balanced(self) -> None:
        group_ids = [f"coregrp-{index:03d}" for index in range(320)]
        first = _balanced_assignments(group_ids, 1006027012)
        second = _balanced_assignments(list(reversed(group_ids)), 1006027012)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({item["factorial_group_id"] for item in first}), 96)
        self.assertEqual(
            Counter(tuple(item["source_combo"]) for item in first),
            Counter({(0, 0): 24, (0, 1): 24, (1, 0): 24, (1, 1): 24}),
        )

    def test_all_eight_condition_source_rules_are_explicit(self) -> None:
        observed = {
            condition: _control_source_combo((0, 1), condition)
            for condition in CONDITIONS
        }
        self.assertEqual(observed["continuous"], [0, 1])
        self.assertEqual(observed["restored"], [0, 1])
        self.assertEqual(observed["swapped_I"], [1, 1])
        self.assertEqual(observed["swapped_G"], [0, 0])
        self.assertEqual(observed["swapped_both"], [1, 0])
        self.assertIsNone(observed["reset"])
        self.assertIsNone(observed["random_matched"])
        self.assertIsNone(observed["prompt_visible_reset"])

    def test_exact_d5_control_manifest_is_reused(self) -> None:
        config = _load_formal_config(
            ROOT / "configs/preregistration/exp001_track_s.formal_v3_holdout.json",
            ROOT,
        )
        manifest = generate_control_manifest(config)
        self.assertEqual(manifest["trial_count"], 96)
        self.assertEqual(manifest["manifest_digest_sha256"], CONTROL_MANIFEST_DIGEST)

    def test_synthetic_payload_requires_all_11008_unique_records(self) -> None:
        payload = self._synthetic_payload()
        _validate_payload(payload)
        payload["records"]["controls"].pop()
        with self.assertRaisesRegex(ValueError, "payload is invalid"):
            _validate_payload(payload)

    def test_package_verifier_accepts_complete_fixture_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "supplemental_set.json": self._synthetic_payload(),
                "set_generation_authorization.json": self._authorization(),
                "exp001b_final_manifest.json": {
                    "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST
                },
                "parent_core_manifest.json": {
                    "core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
                    "core_set_package_digest_sha256": PARENT_CORE_SET_PACKAGE_DIGEST,
                },
            }
            for name, value in files.items():
                (root / name).write_bytes(canonical_json_bytes(value))
            locked = {name: sha256_file(root / name) for name in files}
            manifest = {
                "package_version": "1.0",
                "experiment_id": "EXP-001B",
                "status": SET_STATUS,
                "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
                "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
                "set_preflight_digest_sha256": self.PREFLIGHT_DIGEST,
                "supplemental_set_digest_sha256": files["supplemental_set.json"][
                    "supplemental_set_digest_sha256"
                ],
                "record_counts": dict(EXPECTED_COUNTS),
                "authorization": self._authorization()["authorization"],
                "locked_files": locked,
                "package_payload_root_digest_sha256": payload_digest(locked),
                "safety_boundary": {
                    "supplemental_set_generated": True,
                    "supplemental_set_frozen": True,
                    "supplemental_experiment_authorized": False,
                    "supplemental_experiment_run": False,
                    "supplemental_results_observed": False,
                },
            }
            manifest["supplemental_set_package_digest_sha256"] = sha256_json(manifest)
            (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            self.assertTrue(verify_exp001b_supplemental_set_package(root)["valid"])
            (root / "supplemental_set.json").write_text("{}\n", encoding="utf-8")
            report = verify_exp001b_supplemental_set_package(root)
            self.assertFalse(report["valid"])
            self.assertIn("supplemental_set.json", report["failed_locked_files"])


if __name__ == "__main__":
    unittest.main()
