from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes, payload_digest, sha256_file, sha256_json
from psa.cli import main
from psa.supplemental.finalize import (
    finalize_exp001b_preregistration_package,
    verify_exp001b_final_preregistration_package,
)
from psa.supplemental.freeze import verify_exp001b_preregistration_candidate


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


class Exp001BFinalizeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, str]:
        source = root / "locked_source.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        candidate_dir = root / "candidate"
        evidence = candidate_dir / "evidence" / "development.json"
        _write(evidence, {"development_only": True, "valid": True})
        source_digests = {"locked_source.py": sha256_file(source)}
        evidence_digests = {
            "evidence/development.json": sha256_file(evidence)
        }
        candidate = {
            "candidate_version": "0.1",
            "experiment_id": "EXP-001B",
            "gate": "exp001b_preregistration_candidate_v1",
            "status": "candidate_awaiting_human_checksum_confirmation",
            "model_lock": {"model_id": "rwkv7-g1h-2.9b-20260710"},
            "source_config": {"path": "design.json", "sha256": "3" * 64},
            "source_file_digests": source_digests,
            "evidence_file_digests": evidence_digests,
            "payload_root_digest_sha256": payload_digest(
                {
                    f"source:{name}": digest
                    for name, digest in source_digests.items()
                }
                | {
                    f"evidence:{name}": digest
                    for name, digest in evidence_digests.items()
                }
            ),
            "eligible_for_human_confirmation": True,
            "safety_boundary": {
                "candidate_confirmed": False,
                "supplemental_set_generated": False,
                "supplemental_set_generation_authorized": False,
                "supplemental_experiment_authorized": False,
                "supplemental_experiment_run": False,
                "supplemental_results_observed": False,
                "automatic_rerun_authorized": False,
                "human_checksum_confirmation_required_before_generation": True,
            },
        }
        candidate["candidate_digest_sha256"] = sha256_json(candidate)
        candidate_path = candidate_dir / "preregistration_candidate.json"
        _write(candidate_path, candidate)
        verification = verify_exp001b_preregistration_candidate(
            candidate_path,
            project_root=root,
        )
        self.assertTrue(verification["valid"])
        _write(
            candidate_dir / "preregistration_verification.json",
            verification,
        )
        text = (
            "我确认 EXP-001B 预注册候选 checksum："
            f"{candidate['candidate_digest_sha256']}，"
            "授权将该候选升级为最终预注册包；"
            "暂不授权生成 EXP-001B 补充测试集，不授权运行正式实验。"
        )
        return candidate_dir, text

    def test_finalizes_self_contained_package_without_formal_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, text = self._fixture(root)
            output = root / "final"
            result = finalize_exp001b_preregistration_package(
                candidate_dir=candidate,
                confirmation_text=text,
                output_dir=output,
                project_root=root,
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["locked_file_count"], 4)
            self.assertEqual(result["failed_locked_files"], [])
            self.assertFalse(result["supplemental_set_generated"])
            self.assertFalse(result["supplemental_set_generation_authorized"])
            self.assertFalse(result["supplemental_experiment_authorized"])
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "final_preregistration_frozen")
            self.assertTrue(
                manifest["safety_boundary"]["final_preregistration_frozen"]
            )
            self.assertFalse(
                manifest["authorization"]["generate_supplemental_set"]
            )
            self.assertTrue((output / "evidence" / "development.json").is_file())

    def test_finalizer_is_idempotent_for_same_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, text = self._fixture(root)
            arguments = {
                "candidate_dir": candidate,
                "confirmation_text": text,
                "output_dir": root / "final",
                "project_root": root,
            }
            first = finalize_exp001b_preregistration_package(**arguments)
            second = finalize_exp001b_preregistration_package(**arguments)
            self.assertEqual(
                first["final_preregistration_digest_sha256"],
                second["final_preregistration_digest_sha256"],
            )

    def test_rejects_changed_confirmation_or_broader_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, text = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "does not exactly match"):
                finalize_exp001b_preregistration_package(
                    candidate_dir=candidate,
                    confirmation_text=text.replace(
                        "暂不授权生成",
                        "授权生成",
                    ),
                    output_dir=root / "final",
                    project_root=root,
                )

    def test_verifier_detects_evidence_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, text = self._fixture(root)
            output = root / "final"
            finalize_exp001b_preregistration_package(
                candidate_dir=candidate,
                confirmation_text=text,
                output_dir=output,
                project_root=root,
            )
            _write(
                output / "evidence" / "development.json",
                {"tampered": True},
            )
            report = verify_exp001b_final_preregistration_package(
                output,
                project_root=root,
            )
            self.assertFalse(report["candidate_verification_valid"])
            self.assertIn(
                "evidence/development.json",
                report["failed_locked_files"],
            )
            self.assertFalse(report["valid"])

    def test_cli_finalizes_exact_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, text = self._fixture(root)
            output = root / "final"
            self.assertEqual(
                main(
                    [
                        "exp001b-preregistration-finalize",
                        "--candidate-dir",
                        str(candidate),
                        "--confirmation-text",
                        text,
                        "--output-dir",
                        str(output),
                        "--project-root",
                        str(root),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "exp001b-preregistration-final-verify",
                        "--package-dir",
                        str(output),
                        "--project-root",
                        str(root),
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
