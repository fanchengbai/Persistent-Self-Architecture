from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes, sha256_file
from psa.cli import build_parser
from psa.supplemental.freeze import (
    build_exp001b_preregistration_candidate,
    verify_exp001b_preregistration_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/preregistration/exp001b_supplemental_controls.draft.json"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _unrun() -> dict[str, bool]:
    return {
        "core_set_accessed": False,
        "supplemental_set_generated": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
    }


class Exp001BFreezeTests(unittest.TestCase):
    def _evidence(self, root: Path) -> tuple[Path, Path, Path]:
        design_sha = sha256_file(DESIGN)
        bdev1 = root / "bdev1"
        _write(bdev1 / "matched_context_token_report.json", {"valid": True})
        _write(bdev1 / "state_norm_thresholds.json", {"valid": True})
        _write(
            bdev1 / "summary.json",
            {
                "gate": "exp001b_bdev1_non_core_calibration",
                "valid": True,
                "development_only": True,
                "design_sha256": design_sha,
                "model_id": "rwkv7-g1h-2.9b-20260710",
                "matched_context_case_count": 64,
                "matched_context_valid": True,
                "state_norm_case_count": 64,
                "state_norm_thresholds_valid": True,
                "state_component_count": 96,
                "reports": [
                    "matched_context_token_report.json",
                    "state_norm_thresholds.json",
                ],
                **_unrun(),
            },
        )

        v01 = root / "bdev2_v01"
        _write(v01 / "generation_probe.json", {"valid": False})
        _write(v01 / "state_norm_probe.json", {"valid": False})
        _write(
            v01 / "summary.json",
            {
                "gate": "exp001b_bdev2_non_core_runner",
                "valid": False,
                "development_only": True,
                "design_sha256": design_sha,
                "model_id": "rwkv7-g1h-2.9b-20260710",
                "condition_runner_valid": True,
                "condition_record_count": 128,
                "matched_context_probe_valid": True,
                "matched_context_record_count": 16,
                "generation_probe_valid": False,
                "generation_record_count": 16,
                "forced_prefix_greedy_exact_rate": 1.0,
                "format_valid_rate": 0.875,
                "state_norm_probe_valid": False,
                "reports": ["generation_probe.json", "state_norm_probe.json"],
                **_unrun(),
            },
        )

        v02 = root / "bdev2_v02"
        _write(v02 / "formal_probe_manifest.json", {"valid": True})
        _write(
            v02 / "summary.json",
            {
                "gate": "exp001b_bdev2_non_core_runner_v02",
                "revision_id": "formal-shaped-non-core-probes-v0.2",
                "valid": True,
                "development_only": True,
                "design_sha256": design_sha,
                "model_id": "rwkv7-g1h-2.9b-20260710",
                "bdev1_summary_sha256": sha256_file(bdev1 / "summary.json"),
                "bdev1_thresholds_sha256": sha256_file(
                    bdev1 / "state_norm_thresholds.json"
                ),
                "bdev1_matched_report_sha256": sha256_file(
                    bdev1 / "matched_context_token_report.json"
                ),
                "bdev1_valid": True,
                "condition_alias_valid": True,
                "condition_runner_valid": True,
                "condition_record_count": 128,
                "matched_context_probe_valid": True,
                "matched_context_record_count": 16,
                "formal_probe_manifest_valid": True,
                "formal_probe_manifest_digest_sha256": "4" * 64,
                "formal_probe_shape_warmup_excluded_from_scoring": True,
                "generation_probe_valid": True,
                "generation_record_count": 64,
                "forced_prefix_greedy_exact_rate": 1.0,
                "format_valid_rate": 1.0,
                "state_norm_probe_valid": True,
                "state_norm_record_count": 64,
                "reports": ["formal_probe_manifest.json"],
                **_unrun(),
            },
        )
        return bdev1, v01, v02

    def test_builds_verified_unconfirmed_candidate_without_formal_authority(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temp = Path(directory)
            bdev1, v01, v02 = self._evidence(temp)
            output = temp / "candidate"
            summary = build_exp001b_preregistration_candidate(
                design_path=DESIGN,
                bdev1_dir=bdev1,
                bdev2_v01_dir=v01,
                bdev2_v02_dir=v02,
                output_dir=output,
                project_root=ROOT,
            )
            candidate = json.loads(
                (output / "preregistration_candidate.json").read_text(
                    encoding="utf-8"
                )
            )
            verification = verify_exp001b_preregistration_candidate(
                output / "preregistration_candidate.json",
                project_root=ROOT,
            )
            self.assertTrue(summary["valid"])
            self.assertTrue(summary["candidate_ready_for_human_review"])
            self.assertTrue(verification["valid"])
            self.assertFalse(candidate["safety_boundary"]["candidate_confirmed"])
            self.assertFalse(
                candidate["safety_boundary"]["supplemental_set_generated"]
            )
            self.assertFalse(
                candidate["safety_boundary"][
                    "supplemental_set_generation_authorized"
                ]
            )
            self.assertFalse(
                candidate["safety_boundary"]["supplemental_experiment_authorized"]
            )
            self.assertTrue(
                candidate["development_qualification"][
                    "bdev2_v01_failure_preserved"
                ]
            )

    def test_verifier_rejects_candidate_tampering(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temp = Path(directory)
            bdev1, v01, v02 = self._evidence(temp)
            output = temp / "candidate"
            build_exp001b_preregistration_candidate(
                design_path=DESIGN,
                bdev1_dir=bdev1,
                bdev2_v01_dir=v01,
                bdev2_v02_dir=v02,
                output_dir=output,
                project_root=ROOT,
            )
            path = output / "preregistration_candidate.json"
            candidate = json.loads(path.read_text(encoding="utf-8"))
            candidate["safety_boundary"]["supplemental_experiment_authorized"] = True
            _write(path, candidate)
            report = verify_exp001b_preregistration_candidate(
                path,
                project_root=ROOT,
            )
            self.assertFalse(report["self_digest_valid"])
            self.assertFalse(report["safety_boundary_valid"])
            self.assertFalse(report["valid"])

    def test_builder_requires_preserved_v01_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temp = Path(directory)
            bdev1, v01, v02 = self._evidence(temp)
            path = v01 / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["valid"] = True
            _write(path, summary)
            with self.assertRaisesRegex(ValueError, "failure evidence"):
                build_exp001b_preregistration_candidate(
                    design_path=DESIGN,
                    bdev1_dir=bdev1,
                    bdev2_v01_dir=v01,
                    bdev2_v02_dir=v02,
                    output_dir=temp / "candidate",
                    project_root=ROOT,
                )

    def test_cli_exposes_candidate_build_without_run_authority(self) -> None:
        args = build_parser().parse_args(
            [
                "exp001b-candidate-build",
                "--design",
                "design.json",
                "--bdev1-dir",
                "bdev1",
                "--bdev2-v01-dir",
                "v01",
                "--bdev2-v02-dir",
                "v02",
                "--output-dir",
                "candidate",
            ]
        )
        self.assertEqual(args.command, "exp001b-candidate-build")
        self.assertFalse(hasattr(args, "authorization"))


if __name__ == "__main__":
    unittest.main()
