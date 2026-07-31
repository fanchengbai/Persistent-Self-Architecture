from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes, payload_digest, sha256_json
from psa.cli import main
from psa.preregistration import (
    finalize_preregistration_package,
    verify_final_preregistration_package,
)


class PreregistrationFinalizeTests(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        source_digests = {"locked_source.py": "1" * 64}
        evidence_digests = {"locked_evidence.json": "2" * 64}
        candidate = {
            "candidate_version": "1.0",
            "created_at_utc": "2026-07-31T07:07:49+00:00",
            "status": (
                "frozen_candidate_awaiting_human_checksum_confirmation"
            ),
            "gate": "impl3t_exp001_formal_v3_holdout",
            "model_id": "rwkv7-g1h-2.9b-20260710",
            "confirmed_decision_ids": ["D4", "D5", "D6", "D7", "D8"],
            "history_mode": "single_statement",
            "formal_template_count": 4,
            "formal_query_template_count": 4,
            "filler_variant_count": 4,
            "control_trial_count": 96,
            "factorial_group_count": 320,
            "seeds": {"core_generator": 22217530},
            "statistics": {"familywise_alpha": 0.05},
            "conditions": [
                "continuous",
                "restored",
                "reset",
                "random_matched",
                "swapped_I",
                "swapped_G",
                "swapped_both",
                "prompt_visible",
            ],
            "qualification": {
                "prerequisites_valid": True,
                "template_qualification_passed": True,
                "control_baseline_passed": True,
                "power_gate_passed": True,
            },
            "source_config": {
                "path": "configs/preregistration/test.json",
                "sha256": "3" * 64,
            },
            "source_file_digests": source_digests,
            "evidence_file_digests": evidence_digests,
            "payload_root_digest_sha256": payload_digest(
                {
                    "source:locked_source.py": "1" * 64,
                    "evidence:locked_evidence.json": "2" * 64,
                }
            ),
            "core_set_generated": False,
            "core_set_unsealed": False,
            "formal_state_only_results_observed": False,
            "human_checksum_confirmation_required": True,
            "eligible_for_human_freeze": True,
        }
        candidate["candidate_digest_sha256"] = sha256_json(candidate)
        verification = {
            "report_version": "1.0",
            "candidate_digest_sha256": candidate[
                "candidate_digest_sha256"
            ],
            "self_digest_valid": True,
            "payload_root_valid": True,
            "source_file_checks": {"locked_source.py": True},
            "evidence_file_checks": {"locked_evidence.json": True},
            "safety_boundary_valid": True,
            "eligible_for_human_freeze": True,
            "valid": True,
        }
        confirmation = {
            "confirmation_version": "1.0",
            "experiment_id": "EXP-001",
            "candidate_digest_sha256": candidate[
                "candidate_digest_sha256"
            ],
            "confirmed_by_role": "project_owner",
            "confirmed_at_utc": "2026-07-31T07:42:05Z",
            "confirmation_text": (
                "I confirm checksum "
                f"{candidate['candidate_digest_sha256']}"
            ),
            "authorization": {
                "upgrade_to_final_preregistration_package": True,
                "generate_core_set": False,
                "run_confirmatory_experiment": False,
            },
        }
        paths = (
            root / "candidate.json",
            root / "verification.json",
            root / "confirmation.json",
        )
        for path, value in zip(
            paths,
            (candidate, verification, confirmation),
            strict=True,
        ):
            path.write_bytes(canonical_json_bytes(value))
        return paths

    def test_finalize_builds_locked_package_without_core_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            output = root / "final"
            result = finalize_preregistration_package(
                candidate_path=candidate,
                verification_path=verification,
                confirmation_path=confirmation,
                output_dir=output,
            )
            self.assertTrue(result["valid"])
            self.assertFalse(result["core_set_generated"])
            self.assertFalse(result["confirmatory_experiment_run"])
            self.assertEqual(
                set(result["locked_file_checks"]),
                {
                    "candidate.json",
                    "human_confirmation.json",
                    "verification.json",
                },
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["status"],
                "final_preregistration_frozen",
            )
            self.assertFalse(
                manifest["authorization"]["generate_core_set"]
            )
            self.assertFalse(
                manifest["authorization"][
                    "run_confirmatory_experiment"
                ]
            )
            verified = verify_final_preregistration_package(output)
            self.assertTrue(verified["package_content_valid"])
            self.assertTrue(verified["valid"])

    def test_finalize_is_idempotent_for_same_locked_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            arguments = {
                "candidate_path": candidate,
                "verification_path": verification,
                "confirmation_path": confirmation,
                "output_dir": root / "final",
            }
            first = finalize_preregistration_package(**arguments)
            second = finalize_preregistration_package(**arguments)
            self.assertEqual(
                first["final_preregistration_digest_sha256"],
                second["final_preregistration_digest_sha256"],
            )

    def test_finalize_rejects_different_confirmation_at_existing_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            output = root / "final"
            finalize_preregistration_package(
                candidate_path=candidate,
                verification_path=verification,
                confirmation_path=confirmation,
                output_dir=output,
            )
            value = json.loads(confirmation.read_text(encoding="utf-8"))
            value["confirmation_text"] += " additional text"
            confirmation.write_bytes(canonical_json_bytes(value))
            with self.assertRaisesRegex(ValueError, "already exists"):
                finalize_preregistration_package(
                    candidate_path=candidate,
                    verification_path=verification,
                    confirmation_path=confirmation,
                    output_dir=output,
                )

    def test_finalize_rejects_core_set_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            value = json.loads(confirmation.read_text(encoding="utf-8"))
            value["authorization"]["generate_core_set"] = True
            confirmation.write_bytes(canonical_json_bytes(value))
            with self.assertRaisesRegex(
                ValueError,
                "authorization scope",
            ):
                finalize_preregistration_package(
                    candidate_path=candidate,
                    verification_path=verification,
                    confirmation_path=confirmation,
                    output_dir=root / "final",
                )

    def test_finalize_rejects_wrong_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            value = json.loads(confirmation.read_text(encoding="utf-8"))
            value["candidate_digest_sha256"] = "0" * 64
            confirmation.write_bytes(canonical_json_bytes(value))
            with self.assertRaisesRegex(ValueError, "does not match"):
                finalize_preregistration_package(
                    candidate_path=candidate,
                    verification_path=verification,
                    confirmation_path=confirmation,
                    output_dir=root / "final",
                )

    def test_verify_rejects_manifest_candidate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            output = root / "final"
            finalize_preregistration_package(
                candidate_path=candidate,
                verification_path=verification,
                confirmation_path=confirmation,
                output_dir=output,
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["candidate_digest_sha256"] = "0" * 64
            manifest.pop("final_preregistration_digest_sha256")
            manifest["final_preregistration_digest_sha256"] = sha256_json(
                manifest
            )
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            result = verify_final_preregistration_package(output)
            self.assertFalse(result["package_content_valid"])
            self.assertFalse(result["valid"])

    def test_cli_finalizes_and_verifies_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, verification, confirmation = self._write_fixture(
                root
            )
            output = root / "final"
            exit_code = main(
                [
                    "preregistration-finalize",
                    "--candidate",
                    str(candidate),
                    "--verification",
                    str(verification),
                    "--confirmation",
                    str(confirmation),
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                main(
                    [
                        "preregistration-final-verify",
                        "--package-dir",
                        str(output),
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
