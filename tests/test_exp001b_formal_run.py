from __future__ import annotations

from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.confirmatory.runner import build_non_core_development_fixture
from psa.supplemental.formal_run import (
    EXPERIMENT_ID,
    FINAL_PREREGISTRATION_DIGEST,
    MODEL_ID,
    PARENT_CORE_SET_DIGEST,
    PARENT_CORE_SET_PACKAGE_DIGEST,
    SUPPLEMENTAL_SET_DIGEST,
    SUPPLEMENTAL_SET_PACKAGE_DIGEST,
    build_supplemental_group_plan,
    execute_supplemental_group,
    run_locked_exp001b_groups,
    run_exp001b_supplemental,
    verify_exp001b_run_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture(group_count: int = 2) -> tuple[dict, dict]:
    base = build_non_core_development_fixture()["groups"][0]
    groups = []
    records = {"matched_context": [], "formal_generation": [], "controls": []}
    for group_index in range(group_count):
        group = deepcopy(base)
        group_id = f"coregrp-exp001b-test-{group_index}"
        group["factorial_group_id"] = group_id
        group["group_index"] = group_index
        for index, trial in enumerate(group["trials"]):
            trial["trial_id"] = f"trial-g{group_index}-{index}"
            common = {
                "source_factorial_group_id": group_id,
                "source_trial_id": trial["trial_id"],
                "target_fields": trial["target_fields"],
                "option_mapping": trial["option_mapping"],
                "query_prompt": trial["query_prompt"],
            }
            records["matched_context"].append(
                {
                    "record_id": f"matched-g{group_index}-{index}",
                    "record_kind": "matched_context",
                    **common,
                    "history_prompt": trial["history_prompt"],
                }
            )
            records["formal_generation"].append(
                {
                    "record_id": f"generation-g{group_index}-{index}",
                    "record_kind": "formal_generation_readout",
                    **common,
                    "prompt_visible_prompt": trial["history_prompt"] + trial["query_prompt"],
                    "forced_answer_prefix": ">\n",
                    "maximum_generated_tokens_after_prefix": 4,
                }
            )
        groups.append(group)
    for index, condition in enumerate(
        ("continuous", "restored", "reset", "random_matched", "swapped_I", "swapped_G", "swapped_both", "prompt_visible_reset")
    ):
        records["controls"].append(
            {
                "record_id": f"control-{index}",
                "record_kind": "general_capability_control_condition",
                "assigned_factorial_group_id": groups[0]["factorial_group_id"],
                "condition": condition,
                "prompt": "development control",
                "assigned_source_combo": [0, 0],
                "state_source_combo": None,
            }
        )
    return {"groups": groups}, {"records": records}


class FakeBackend:
    def __init__(self, fail_group: str | None = None) -> None:
        self.fail_group = fail_group
        self.calls: list[str] = []

    def start_group(self, core_group, records) -> None:
        self.active = core_group["factorial_group_id"]

    def score_record(self, *, core_group, record):
        group_id = core_group["factorial_group_id"]
        self.calls.append(group_id)
        if group_id == self.fail_group:
            raise RuntimeError("injected EXP-001B interruption")
        result = {
            "option_scores": {code: float(index) for index, code in enumerate("ABCD")},
            "metadata": {"fixture": True},
        }
        if record["record_kind"] == "formal_generation_readout":
            result.update(
                generated_text=" A",
                generated_token_ids=[300],
                generated_choice="A",
                format_valid=True,
            )
        return result

    def end_group(self, core_group) -> None:
        self.active = None


class Exp001BFormalRunTests(unittest.TestCase):
    def _patch_counts(self):
        return patch.multiple(
            "psa.supplemental.formal_run",
            EXPECTED_GROUP_COUNT=2,
            EXPECTED_RAW_RECORD_COUNT=72,
        )

    def test_frozen_package_maps_to_320_complete_groups(self) -> None:
        core = json.loads((ROOT / "preregistration/exp001/core_set_v1/core_set.json").read_text(encoding="utf-8"))
        supplemental = json.loads((ROOT / "preregistration/exp001b/supplemental_set_v1/supplemental_set.json").read_text(encoding="utf-8"))
        plans = build_supplemental_group_plan(supplemental, core)
        self.assertEqual(len(plans), 320)
        self.assertEqual(sum(item["record_count"] for item in plans), 11008)
        self.assertEqual(Counter(item["record_count"] for item in plans), Counter({32: 224, 40: 96}))

    def test_group_output_contains_raw_values_but_no_accuracy(self) -> None:
        core, supplemental = _fixture()
        with self._patch_counts():
            plan = build_supplemental_group_plan(supplemental, core)[0]
        result = execute_supplemental_group(core_group=core["groups"][0], plan=plan, backend=FakeBackend())
        self.assertEqual(result["record_count"], 40)
        self.assertFalse(result["contains_derived_accuracy"])
        self.assertFalse(result["contains_interim_decision"])
        generated = [item for item in result["records"] if item["record_kind"] == "formal_generation_readout"]
        self.assertEqual(len(generated), 16)
        self.assertTrue(all(item["format_valid"] for item in generated))

    def test_interruption_requires_resume_and_completed_run_cannot_rerun(self) -> None:
        core, supplemental = _fixture()
        lock = {
            "valid": True,
            "preflight_digest_sha256": "1" * 64,
            "authorization_file_sha256": "2" * 64,
            "expected_group_ids": [item["factorial_group_id"] for item in core["groups"]],
        }
        with tempfile.TemporaryDirectory() as temporary, self._patch_counts():
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_locked_exp001b_groups(core_set=core, supplemental_set=supplemental, backend=FakeBackend(fail_group=core["groups"][1]["factorial_group_id"]), output_dir=temporary, launch_lock=lock)
            manifest = json.loads((Path(temporary) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["completed_group_count"], 1)
            with self.assertRaisesRegex(ValueError, "explicit resume"):
                run_locked_exp001b_groups(core_set=core, supplemental_set=supplemental, backend=FakeBackend(), output_dir=temporary, launch_lock=lock)
            completion = run_locked_exp001b_groups(core_set=core, supplemental_set=supplemental, backend=FakeBackend(), output_dir=temporary, launch_lock=lock, resume=True)
            self.assertTrue(completion["valid"])
            self.assertEqual(completion["raw_record_count"], 72)
            with self.assertRaisesRegex(ValueError, "cannot be rerun"):
                run_locked_exp001b_groups(core_set=core, supplemental_set=supplemental, backend=FakeBackend(), output_dir=temporary, launch_lock=lock, resume=True)

    def test_run_authorization_requires_exact_scope_and_all_digests(self) -> None:
        preflight = {
            "valid": True,
            "status": "preflight_valid_authorization_still_required",
            "preflight_digest_sha256": "a" * 64,
            "runner_development_evidence": {"valid": True},
        }
        authorization = {
            "authorization_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "authorized_by_role": "project_owner",
            "authorized_at_utc": "2026-08-04T12:00:00+00:00",
            "authorization_text": "Explicit authorization for the complete frozen EXP-001B supplemental run.",
            "preflight_digest_sha256": "a" * 64,
            "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
            "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
            "parent_core_set_package_digest_sha256": PARENT_CORE_SET_PACKAGE_DIGEST,
            "supplemental_set_digest_sha256": SUPPLEMENTAL_SET_DIGEST,
            "supplemental_set_package_digest_sha256": SUPPLEMENTAL_SET_PACKAGE_DIGEST,
            "model_id": MODEL_ID,
            "authorization": {
                "run_supplemental_experiment": True,
                "observe_results_after_full_completion": True,
                "modify_frozen_design": False,
                "automatic_rerun_after_results": False,
                "rerun_exp001_primary_experiment": False,
            },
        }
        self.assertTrue(verify_exp001b_run_authorization(authorization, preflight=preflight)["valid"])
        authorization["authorization"]["automatic_rerun_after_results"] = True
        self.assertFalse(verify_exp001b_run_authorization(authorization, preflight=preflight)["valid"])

    def test_execution_lock_fails_before_preflight_or_package_access(self) -> None:
        with patch(
            "psa.supplemental.formal_run.prepare_exp001b_launch"
        ) as prepare:
            with self.assertRaisesRegex(PermissionError, "execution lock"):
                run_exp001b_supplemental(
                    project_root=".",
                    final_package_dir="missing-final",
                    core_set_package_dir="missing-core",
                    supplemental_set_package_dir="missing-set",
                    model_config_path="missing-model",
                    asset_manifest_path="missing-assets",
                    asset_root="missing-root",
                    runner_evidence_path="missing-evidence",
                    preflight_path="missing-preflight",
                    authorization_path="missing-authorization",
                    output_dir="missing-output",
                )
            prepare.assert_not_called()

    def test_gpu_entry_scripts_freeze_runtime_flags(self) -> None:
        for relative in (
            "scripts/run_exp001b_runner_development_gate.sh",
            "scripts/preflight_exp001b_supplemental_run.sh",
            "scripts/run_exp001b_supplemental.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("export RWKV_V7_ON=1", text)
            self.assertIn("export RWKV_JIT_ON=0", text)
            self.assertIn("export RWKV_CUDA_ON=0", text)


if __name__ == "__main__":
    unittest.main()
