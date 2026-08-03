from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
)
from psa.confirmatory.preflight import (
    EXPECTED_CORE_PACKAGE_DIGEST,
    EXPECTED_CORE_SET_DIGEST,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_FINAL_DIGEST,
    EXPECTED_MODEL_ID,
)
from psa.confirmatory.runner import (
    CONDITIONS,
    build_non_core_development_fixture,
    execute_group,
)
from psa.confirmatory.verification import (
    _verify_group_payload,
    verify_exp001_confirmatory_raw_package,
)


class FakeBackend:
    def score(self, *, group, trial, condition_plan):
        selected = condition_plan["evaluation_option_code"]
        return {
            "option_scores": {
                code: 1.0 if code == selected else -1.0
                for code in "ABCD"
            },
            "metadata": {"backend": "verification-test"},
        }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


class ConfirmatoryRawVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.group = deepcopy(
            build_non_core_development_fixture()["groups"][0]
        )
        self.group["factorial_group_id"] = "coregrp-verification-test"
        self.result = execute_group(self.group, FakeBackend())

    def test_group_verification_checks_structure_without_accuracy(self) -> None:
        checks = _verify_group_payload(
            self.result,
            expected_group=self.group,
        )
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()))
        self.assertNotIn("accuracy", checks)

    def _build_package(self, root: Path) -> dict[str, Path]:
        output = root / "run"
        core = root / "core"
        preflight_path = root / "preflight.json"
        authorization_path = root / "authorization.json"
        group_path = output / "groups" / "coregrp-verification-test.json"
        _write(group_path, self.result)
        group_digest = sha256_file(group_path)
        completed = {"coregrp-verification-test": group_digest}
        preflight_digest = "1" * 64
        authorization = {"test": "authorization"}
        _write(authorization_path, authorization)
        _write(
            preflight_path,
            {"preflight_digest_sha256": preflight_digest},
        )
        _write(
            core / "core_set.json",
            {
                "experiment_id": EXPECTED_EXPERIMENT_ID,
                "final_preregistration_digest_sha256": EXPECTED_FINAL_DIGEST,
                "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
                "factorial_group_count": 1,
                "trial_count": 16,
                "groups": [self.group],
            },
        )
        manifest = {
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "model_id": EXPECTED_MODEL_ID,
            "preflight_digest_sha256": preflight_digest,
            "authorization_file_sha256": sha256_file(authorization_path),
            "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
            "core_set_package_digest_sha256": EXPECTED_CORE_PACKAGE_DIGEST,
            "status": "confirmatory_raw_complete",
            "valid": True,
            "expected_group_count": 1,
            "completed_group_count": 1,
            "completed_group_files": completed,
            "raw_record_count": 128,
            "group_payload_digest_sha256": payload_digest(completed),
            "contains_derived_accuracy": False,
            "contains_interim_decision": False,
            "confirmatory_results_observed": False,
        }
        completion = {
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "preflight_digest_sha256": preflight_digest,
            "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
            "status": "confirmatory_raw_complete",
            "valid": True,
            "completed_group_count": 1,
            "raw_record_count": 128,
            "group_payload_digest_sha256": payload_digest(completed),
            "contains_derived_accuracy": False,
            "contains_interim_decision": False,
            "confirmatory_results_observed": False,
        }
        _write(output / "manifest.json", manifest)
        _write(output / "completion.json", completion)
        return {
            "output": output,
            "core": core,
            "preflight": preflight_path,
            "authorization": authorization_path,
            "group": group_path,
        }

    def _verify(self, paths: dict[str, Path]) -> dict:
        with (
            patch.multiple(
                "psa.confirmatory.verification",
                EXPECTED_GROUP_COUNT=1,
                EXPECTED_TRIAL_COUNT=16,
                EXPECTED_RAW_RECORD_COUNT=128,
            ),
            patch(
                "psa.confirmatory.verification.verify_core_set_package",
                return_value={"valid": True},
            ),
            patch(
                "psa.confirmatory.verification.verify_confirmatory_run_authorization",
                return_value={"valid": True},
            ),
        ):
            return verify_exp001_confirmatory_raw_package(
                output_dir=paths["output"],
                core_set_package_dir=paths["core"],
                preflight_path=paths["preflight"],
                authorization_path=paths["authorization"],
            )

    def test_complete_package_verifies_without_derived_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self._verify(self._build_package(Path(temporary)))
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "raw_package_verified_unanalyzed")
        self.assertEqual(report["verified_record_count"], 128)
        self.assertFalse(report["contains_derived_accuracy"])
        self.assertFalse(report["confirmatory_results_observed"])

    def test_tampered_group_fails_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_package(Path(temporary))
            payload = json.loads(paths["group"].read_text(encoding="utf-8"))
            payload["record_count"] = 1
            _write(paths["group"], payload)
            report = self._verify(paths)
        self.assertFalse(report["valid"])
        self.assertEqual(report["failed_group_count"], 1)
        self.assertIn("all_group_files_valid", report["failed_checks"])
        self.assertEqual(
            report["route_decision"],
            "hold_without_analysis_and_repair_integrity",
        )


if __name__ == "__main__":
    unittest.main()
