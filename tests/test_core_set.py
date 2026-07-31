from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes
from psa.preregistration import (
    generate_and_freeze_core_set,
    verify_core_set_package,
)


class CoreSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "core_set"
        cls.report = generate_and_freeze_core_set(
            final_package_dir=(
                cls.root / "preregistration" / "exp001" / "final_v1"
            ),
            authorization_path=(
                cls.root
                / "preregistration"
                / "exp001"
                / "core_set_authorization.json"
            ),
            config_path=(
                cls.root
                / "configs"
                / "preregistration"
                / "exp001_track_s.formal_v3_holdout.json"
            ),
            output_dir=cls.output,
            project_root=cls.root,
            token_counter=lambda _: 131,
            tokenizer_provenance={
                "path": "test-tokenizer",
                "revision": "test",
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_core_set_is_balanced_frozen_and_unrun(self) -> None:
        self.assertTrue(self.report["valid"])
        self.assertEqual(self.report["factorial_group_count"], 320)
        self.assertEqual(self.report["semantic_case_count"], 1280)
        self.assertEqual(self.report["trial_count"], 5120)
        self.assertFalse(self.report["confirmatory_experiment_run"])
        self.assertFalse(self.report["confirmatory_results_observed"])
        core_set = json.loads(
            (self.output / "core_set.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(core_set["groups"]), 320)
        self.assertTrue(
            all(group["trial_count"] == 16 for group in core_set["groups"])
        )
        self.assertFalse(core_set["confirmatory_experiment_run"])
        self.assertFalse(core_set["confirmatory_results_observed"])

    def test_core_set_generation_is_idempotent(self) -> None:
        repeated = generate_and_freeze_core_set(
            final_package_dir=(
                self.root / "preregistration" / "exp001" / "final_v1"
            ),
            authorization_path=(
                self.root
                / "preregistration"
                / "exp001"
                / "core_set_authorization.json"
            ),
            config_path=(
                self.root
                / "configs"
                / "preregistration"
                / "exp001_track_s.formal_v3_holdout.json"
            ),
            output_dir=self.output,
            project_root=self.root,
            token_counter=lambda _: 131,
            tokenizer_provenance={
                "path": "test-tokenizer",
                "revision": "test",
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
        )
        self.assertEqual(
            repeated["core_set_digest_sha256"],
            self.report["core_set_digest_sha256"],
        )

    def test_core_set_package_detects_tampering(self) -> None:
        tampered = Path(self.temporary.name) / "tampered"
        shutil.copytree(self.output, tampered)
        core_path = tampered / "core_set.json"
        core_set = json.loads(core_path.read_text(encoding="utf-8"))
        core_set["trial_count"] = 1
        core_path.write_bytes(canonical_json_bytes(core_set))
        report = verify_core_set_package(tampered)
        self.assertFalse(report["locked_file_checks"]["core_set.json"])
        self.assertFalse(report["content_valid"])
        self.assertFalse(report["valid"])

    def test_core_set_rejects_confirmatory_run_authorization(self) -> None:
        authorization_source = (
            self.root
            / "preregistration"
            / "exp001"
            / "core_set_authorization.json"
        )
        authorization = json.loads(
            authorization_source.read_text(encoding="utf-8")
        )
        authorization["authorization"][
            "run_confirmatory_experiment"
        ] = True
        invalid_path = Path(self.temporary.name) / "invalid_auth.json"
        invalid_path.write_bytes(canonical_json_bytes(authorization))
        with self.assertRaisesRegex(ValueError, "scope is invalid"):
            generate_and_freeze_core_set(
                final_package_dir=(
                    self.root / "preregistration" / "exp001" / "final_v1"
                ),
                authorization_path=invalid_path,
                config_path=(
                    self.root
                    / "configs"
                    / "preregistration"
                    / "exp001_track_s.formal_v3_holdout.json"
                ),
                output_dir=Path(self.temporary.name) / "invalid_output",
                project_root=self.root,
                token_counter=lambda _: 131,
            )


if __name__ == "__main__":
    unittest.main()
