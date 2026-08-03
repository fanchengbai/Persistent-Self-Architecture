from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import canonical_json_bytes, sha256_json
from psa.confirmatory import (
    CONDITIONS,
    DEVELOPMENT_FIXTURE_KIND,
    build_group_execution_plan,
    build_non_core_development_fixture,
    condition_evaluation_combo,
    condition_source_combo,
    execute_group,
    run_development_fixture,
)


def _group(index: int) -> dict:
    combos = ((0, 0), (0, 1), (1, 0), (1, 1))
    trials = []
    codes = ("A", "B", "C", "D")
    for rotation in range(4):
        rotated = codes[rotation:] + codes[:rotation]
        option_mapping = [
            {
                "code": code,
                "identity": combo[0],
                "goal": combo[1],
                "domain": f"domain-{combo[0]}",
                "operation": f"goal-{combo[1]}",
            }
            for combo, code in zip(combos, rotated, strict=True)
        ]
        for identity, goal in combos:
            trials.append(
                {
                    "trial_id": f"devtrial-{index}-{rotation}-{identity}-{goal}",
                    "semantic_case_id": f"devcase-{index}-{identity}-{goal}",
                    "rotation_index": rotation,
                    "history_prompt": (
                        "NON-CORE DEVELOPMENT HISTORY "
                        f"IDENTITY {identity} GOAL {goal}"
                    ),
                    "query_prompt": "NON-CORE DEVELOPMENT QUERY",
                    "target_fields": {
                        "identity": identity,
                        "goal": goal,
                        "domain": f"domain-{identity}",
                        "operation": f"goal-{goal}",
                    },
                    "option_mapping": option_mapping,
                }
            )
    return {
        "factorial_group_id": f"devgrp-{index}",
        "group_index": index,
        "trial_count": 16,
        "trials": trials,
    }


def _fixture(group_count: int = 1) -> dict:
    fixture = {
        "fixture_version": "0.1",
        "fixture_kind": DEVELOPMENT_FIXTURE_KIND,
        "experiment_id": "DEV-IMPL5B-RUNNER",
        "groups": [_group(index) for index in range(group_count)],
    }
    fixture["fixture_digest_sha256"] = sha256_json(fixture)
    return fixture


class FakeBackend:
    def __init__(self, *, fail_group: str | None = None) -> None:
        self.fail_group = fail_group
        self.calls: list[tuple[str, str, str]] = []

    def score(self, *, group, trial, condition_plan):
        group_id = group["factorial_group_id"]
        condition = condition_plan["condition"]
        self.calls.append((group_id, trial["trial_id"], condition))
        if group_id == self.fail_group:
            raise RuntimeError("injected development backend failure")
        selected = condition_plan["evaluation_option_code"]
        return {
            "option_scores": {
                code: 1.0 if code == selected else -1.0 for code in "ABCD"
            },
            "metadata": {"backend": "fake", "formal_model_loaded": False},
        }


class ConfirmatoryRunnerTests(unittest.TestCase):
    def test_committed_fixture_builder_is_non_core_and_balanced(self) -> None:
        fixture = build_non_core_development_fixture()
        self.assertEqual(fixture["experiment_id"], "DEV-IMPL5B-RUNNER")
        self.assertEqual(fixture["fixture_kind"], DEVELOPMENT_FIXTURE_KIND)
        self.assertEqual(len(fixture["groups"]), 1)
        self.assertEqual(len(fixture["groups"][0]["trials"]), 16)
        self.assertFalse(
            fixture["groups"][0]["factorial_group_id"].startswith("coregrp-")
        )
        serialized = str(fixture)
        for frozen_label in ("baf", "zom", "niv", "teg", "vam", "zep"):
            self.assertNotIn(frozen_label, serialized)

    def test_all_condition_routes_are_explicit(self) -> None:
        target = (0, 1)
        expected_sources = {
            "continuous": (0, 1),
            "restored": (0, 1),
            "reset": None,
            "random_matched": None,
            "swapped_I": (1, 1),
            "swapped_G": (0, 0),
            "swapped_both": (1, 0),
            "prompt_visible": (0, 1),
        }
        self.assertEqual(set(expected_sources), set(CONDITIONS))
        for condition, expected in expected_sources.items():
            self.assertEqual(
                condition_source_combo(target, condition),
                expected,
            )
        self.assertEqual(
            condition_evaluation_combo(target, "random_matched"),
            target,
        )
        self.assertEqual(
            condition_evaluation_combo(target, "swapped_both"),
            (1, 0),
        )

    def test_group_plan_and_execution_cover_128_records(self) -> None:
        group = _group(0)
        plan = build_group_execution_plan(group)
        self.assertEqual(plan["trial_count"], 16)
        self.assertEqual(plan["condition_count"], 8)
        self.assertEqual(plan["record_count"], 128)
        backend = FakeBackend()
        result = execute_group(group, backend)
        self.assertEqual(result["record_count"], 128)
        self.assertEqual(len(backend.calls), 128)
        self.assertFalse(result["contains_derived_accuracy"])
        self.assertFalse(result["contains_interim_decision"])
        self.assertTrue(
            all("joint_correct" not in record for record in result["records"])
        )

    def test_interrupted_run_resumes_only_missing_groups(self) -> None:
        fixture = _fixture(2)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "run"
            failing = FakeBackend(fail_group="devgrp-1")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_development_fixture(
                    dataset=fixture,
                    backend=failing,
                    output_dir=destination,
                )
            interrupted = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["completed_group_count"], 1)
            self.assertEqual(
                set(interrupted["completed_group_files"]), {"devgrp-0"}
            )

            resumed = FakeBackend()
            report = run_development_fixture(
                dataset=fixture,
                backend=resumed,
                output_dir=destination,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["status"], "development_fixture_complete")
            self.assertEqual(report["completed_group_count"], 2)
            self.assertEqual(len(resumed.calls), 128)
            self.assertTrue(
                all(call[0] == "devgrp-1" for call in resumed.calls)
            )

    def test_resume_rejects_tampered_completed_group(self) -> None:
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "run"
            run_development_fixture(
                dataset=fixture,
                backend=FakeBackend(),
                output_dir=destination,
            )
            group_path = destination / "groups" / "devgrp-0.json"
            group_result = json.loads(group_path.read_text(encoding="utf-8"))
            group_result["record_count"] = 1
            group_path.write_bytes(canonical_json_bytes(group_result))
            with self.assertRaisesRegex(ValueError, "integrity failed"):
                run_development_fixture(
                    dataset=fixture,
                    backend=FakeBackend(),
                    output_dir=destination,
                )

    def test_development_path_rejects_core_identity(self) -> None:
        fixture = _fixture()
        fixture["experiment_id"] = "EXP-001"
        fixture["fixture_digest_sha256"] = sha256_json(
            {k: v for k, v in fixture.items() if k != "fixture_digest_sha256"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "cannot be used"):
                run_development_fixture(
                    dataset=fixture,
                    backend=FakeBackend(),
                    output_dir=temporary,
                )

    def test_development_path_rejects_duplicate_group_ids(self) -> None:
        fixture = _fixture(2)
        fixture["groups"][1]["factorial_group_id"] = "devgrp-0"
        fixture["fixture_digest_sha256"] = sha256_json(
            {k: v for k, v in fixture.items() if k != "fixture_digest_sha256"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must be unique"):
                run_development_fixture(
                    dataset=fixture,
                    backend=FakeBackend(),
                    output_dir=temporary,
                )


if __name__ == "__main__":
    unittest.main()
