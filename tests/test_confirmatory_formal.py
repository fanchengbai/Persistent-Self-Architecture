from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.artifacts import canonical_json_bytes
from psa.confirmatory.formal import run_locked_confirmatory_groups
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
)


def _formal_fixture(group_count: int = 2) -> dict:
    base = build_non_core_development_fixture()["groups"][0]
    groups = []
    for group_index in range(group_count):
        group = deepcopy(base)
        group["factorial_group_id"] = f"coregrp-test-{group_index}"
        group["group_index"] = group_index
        for trial in group["trials"]:
            trial["trial_id"] = f"{trial['trial_id']}-g{group_index}"
            trial["semantic_case_id"] = (
                f"{trial['semantic_case_id']}-g{group_index}"
            )
        groups.append(group)
    return {
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": "core_set_frozen_unrun",
        "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
        "final_preregistration_digest_sha256": EXPECTED_FINAL_DIGEST,
        "factorial_group_count": group_count,
        "trial_count": group_count * 16,
        "conditions": list(CONDITIONS),
        "confirmatory_experiment_run": False,
        "confirmatory_results_observed": False,
        "groups": groups,
    }


def _launch_lock(dataset: dict) -> dict:
    return {
        "valid": True,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_id": EXPECTED_MODEL_ID,
        "preflight_digest_sha256": "1" * 64,
        "authorization_file_sha256": "2" * 64,
        "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
        "core_set_package_digest_sha256": EXPECTED_CORE_PACKAGE_DIGEST,
        "expected_group_ids": [
            group["factorial_group_id"] for group in dataset["groups"]
        ],
        "expected_group_count": len(dataset["groups"]),
        "expected_raw_record_count": len(dataset["groups"]) * 16 * 8,
    }


class FakeBackend:
    def __init__(self, fail_group: str | None = None) -> None:
        self.fail_group = fail_group
        self.calls: list[str] = []

    def score(self, *, group, trial, condition_plan):
        group_id = group["factorial_group_id"]
        self.calls.append(group_id)
        if group_id == self.fail_group:
            raise RuntimeError("injected formal interruption")
        selected = condition_plan["evaluation_option_code"]
        return {
            "option_scores": {
                code: 1.0 if code == selected else -1.0
                for code in "ABCD"
            },
            "metadata": {"backend": "fake-formal-test"},
        }


class ConfirmatoryFormalRunnerTests(unittest.TestCase):
    def _patch_counts(self, group_count: int):
        return patch.multiple(
            "psa.confirmatory.formal",
            EXPECTED_FACTORIAL_GROUP_COUNT=group_count,
            EXPECTED_ROTATION_TRIAL_COUNT=group_count * 16,
            EXPECTED_RAW_RECORD_COUNT=group_count * 16 * 8,
        )

    def test_invalid_launch_lock_is_rejected_before_output(self) -> None:
        dataset = _formal_fixture(2)
        lock = _launch_lock(dataset)
        lock["valid"] = False
        with tempfile.TemporaryDirectory() as temporary, self._patch_counts(2):
            with self.assertRaisesRegex(ValueError, "valid launch lock"):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(),
                    output_dir=temporary,
                    launch_lock=lock,
                )
            self.assertFalse((Path(temporary) / "manifest.json").exists())

    def test_new_run_rejects_nonempty_output_directory(self) -> None:
        dataset = _formal_fixture(2)
        lock = _launch_lock(dataset)
        with tempfile.TemporaryDirectory() as temporary, self._patch_counts(2):
            (Path(temporary) / "unrelated.txt").write_text(
                "preserve me",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be empty"):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(),
                    output_dir=temporary,
                    launch_lock=lock,
                )

    def test_interruption_requires_manual_resume_and_completion_cannot_rerun(self) -> None:
        dataset = _formal_fixture(2)
        lock = _launch_lock(dataset)
        with tempfile.TemporaryDirectory() as temporary, self._patch_counts(2):
            destination = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(fail_group="coregrp-test-1"),
                    output_dir=destination,
                    launch_lock=lock,
                )
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["completed_group_count"], 1)
            self.assertFalse(manifest["contains_derived_accuracy"])

            with self.assertRaisesRegex(ValueError, "explicit resume"):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(),
                    output_dir=destination,
                    launch_lock=lock,
                )

            resumed = FakeBackend()
            completion = run_locked_confirmatory_groups(
                dataset=dataset,
                backend=resumed,
                output_dir=destination,
                launch_lock=lock,
                resume=True,
            )
            self.assertTrue(completion["valid"])
            self.assertEqual(completion["completed_group_count"], 2)
            self.assertEqual(completion["raw_record_count"], 256)
            self.assertFalse(completion["contains_derived_accuracy"])
            self.assertEqual(set(resumed.calls), {"coregrp-test-1"})

            with self.assertRaisesRegex(ValueError, "cannot be rerun"):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(),
                    output_dir=destination,
                    launch_lock=lock,
                    resume=True,
                )

    def test_resume_rejects_tampered_completed_group(self) -> None:
        dataset = _formal_fixture(2)
        lock = _launch_lock(dataset)
        with tempfile.TemporaryDirectory() as temporary, self._patch_counts(2):
            destination = Path(temporary)
            with self.assertRaises(RuntimeError):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(fail_group="coregrp-test-1"),
                    output_dir=destination,
                    launch_lock=lock,
                )
            group_path = destination / "groups" / "coregrp-test-0.json"
            payload = json.loads(group_path.read_text(encoding="utf-8"))
            payload["record_count"] = 1
            group_path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(ValueError, "integrity failed"):
                run_locked_confirmatory_groups(
                    dataset=dataset,
                    backend=FakeBackend(),
                    output_dir=destination,
                    launch_lock=lock,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
