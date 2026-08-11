from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.cli import main
from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)
from psa.development.exp001c_v02_stage_a import (
    STAGE_A_EXECUTION_ENV,
    STAGE_A_EXECUTION_LOCK,
    RWKVExp001CV02StageABackend,
    build_exp001c_v02_stage_a_backend,
    run_exp001c_v02_stage_a,
    validate_exp001c_v02_stage_a_authority,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "development" / "exp001c_noncore_protocol_v02.draft.json"
MODEL = ROOT / "configs" / "models" / "rwkv7_g1h_2.9b.candidate.json"


class _FakeAdapter:
    torch = object()

    def __init__(self) -> None:
        self.forward_calls: list[tuple[list[int], object]] = []

    def encode(self, text: str) -> list[int]:
        if text == ">\n":
            return [4, 3]
        if text in {"A", "B", "C", "D"}:
            return [6 + ord(text) - ord("A")]
        return [1] * max(1, len(text.split()))

    def decode(self, tokens) -> str:
        return ">\n" if list(tokens) == [4, 3] else "?"

    def forward(self, tokens, state=None):
        token_list = list(tokens)
        self.forward_calls.append((token_list, copy.deepcopy(state)))
        next_state = copy.deepcopy(state) if state is not None else {"steps": 0}
        next_state["steps"] = int(next_state.get("steps", 0)) + len(token_list)
        logits = [0.0] * 12
        if token_list == [4]:
            logits[2] = 4.0
            logits[3] = 2.0
        elif token_list == [3]:
            logits[6] = 4.0
        else:
            logits[4] = 4.0
        return logits, next_state


def _fake_option_scorer(adapter, *, logits, state, rendered_answers):
    del adapter, logits, state, rendered_answers
    return {"A": -0.1, "B": -1.0, "C": -2.0, "D": -3.0}


class Exp001CV02StageATests(unittest.TestCase):
    def test_fake_backend_runs_prompt_visible_only_with_full_rotation(self) -> None:
        manifest = build_exp001c_protocol_v02_manifest(
            config_path=CONFIG,
            project_root=ROOT,
        )
        adapter = _FakeAdapter()
        result = RWKVExp001CV02StageABackend(
            adapter=adapter,
            option_scorer=_fake_option_scorer,
        ).run_stage_a(manifest)

        self.assertEqual(result["record_count"], 32)
        self.assertTrue(result["prompt_visible_only"])
        self.assertFalse(result["stage_b_recurrent_state_accessed"])
        self.assertFalse(result["formal_test_set_accessed"])
        self.assertFalse(result["contains_confirmatory_decision"])
        self.assertEqual(
            Counter(record["target_code"] for record in result["records"]),
            Counter({code: 8 for code in "ABCD"}),
        )
        for record in result["records"]:
            self.assertEqual(record["prefix_evidence"]["token_ids"], [4, 3])
            self.assertTrue(record["prefix_evidence"]["roundtrip_exact"])
            self.assertIn(
                "target_margin_over_best_incorrect",
                record["answer_boundary_evidence"],
            )
        prompt_forwards = [
            call for call in adapter.forward_calls if call[1] is None
        ]
        self.assertGreaterEqual(len(prompt_forwards), 32)

    def test_execution_lock_is_checked_before_paths_or_factory(self) -> None:
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            raise AssertionError("backend factory must remain unreachable")

        with self.assertRaisesRegex(PermissionError, "lock is absent"):
            run_exp001c_v02_stage_a(
                manifest_path="missing-manifest.json",
                authorization_path="missing-authorization.json",
                model_config_path="missing-model.json",
                output_dir="missing-output",
                backend_factory=factory,
                execution_lock="",
                project_root=ROOT,
            )
        self.assertFalse(factory_called)

    def test_current_unrun_manifest_blocks_even_with_exact_execution_lock(self) -> None:
        manifest = build_exp001c_protocol_v02_manifest(
            config_path=CONFIG,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "not authorized by manifest"):
                validate_exp001c_v02_stage_a_authority(
                    manifest_path=manifest_path,
                    authorization_path=Path(directory) / "absent.json",
                    execution_lock=STAGE_A_EXECUTION_LOCK,
                    project_root=ROOT,
                )

    def test_cli_run_without_lock_cannot_load_model(self) -> None:
        with patch.dict(os.environ, {STAGE_A_EXECUTION_ENV: ""}):
            with patch(
                "psa.development.exp001c_v02_stage_a.RWKV7Adapter.load"
            ) as model_loader:
                self.assertEqual(
                    main(
                        [
                            "exp001c-v02-stage-a-run",
                            "--manifest",
                            "missing-manifest.json",
                            "--authorization",
                            "missing-authorization.json",
                            "--model-config",
                            str(MODEL),
                            "--output-dir",
                            "missing-output",
                            "--project-root",
                            str(ROOT),
                        ]
                    ),
                    2,
                )
                model_loader.assert_not_called()

    def test_factory_rejects_unlocked_model_before_model_load(self) -> None:
        manifest = build_exp001c_protocol_v02_manifest(
            config_path=CONFIG,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "psa.development.exp001c_v02_stage_a.RWKV7Adapter.load"
            ) as model_loader:
                with self.assertRaisesRegex(ValueError, "model config"):
                    build_exp001c_v02_stage_a_backend(
                        manifest_path=manifest_path,
                        model_config_path=CONFIG,
                        project_root=ROOT,
                    )
                model_loader.assert_not_called()

    def test_new_json_schemas_are_valid_json(self) -> None:
        for name in (
            "exp001c_v02_stage_a_authorization.schema.json",
            "exp001c_v02_stage_a_result.schema.json",
        ):
            value = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
            self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
