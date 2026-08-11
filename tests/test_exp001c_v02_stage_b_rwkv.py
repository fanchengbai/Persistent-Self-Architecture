from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)
from psa.development.exp001c_v02_stage_b_design import (
    STAGE_B_CONDITIONS,
    build_exp001c_v02_stage_b_design_manifest,
)
from psa.development.exp001c_v02_stage_b_rwkv import (
    RWKVExp001CV02StageBBackend,
    build_exp001c_v02_stage_b_rwkv_backend,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_CONFIG = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_v02_stage_b_design.draft.json"
)
PROTOCOL_CONFIG = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_noncore_protocol_v02.draft.json"
)
MODEL_CONFIG = ROOT / "configs" / "models" / "rwkv7_g1h_2.9b.candidate.json"


class _FakeRWKVAdapter:
    torch = object()

    def __init__(self) -> None:
        self.forward_calls: list[tuple[list[int], object]] = []

    def encode(self, text: str) -> list[int]:
        if text == ">\n":
            return [4, 3]
        if text in set("ABCD"):
            return [6 + ord(text) - ord("A")]
        size = max(1, len(text.split()))
        token = 1 + (sum(ord(character) for character in text) % 2)
        return [token] * size

    def decode(self, tokens) -> str:
        return ">\n" if list(tokens) == [4, 3] else "?"

    def forward(self, tokens, state=None):
        token_list = list(tokens)
        self.forward_calls.append((token_list, copy.deepcopy(state)))
        next_state = copy.deepcopy(state) if state is not None else {
            "origin": tuple(token_list),
            "steps": 0,
        }
        next_state["steps"] = int(next_state.get("steps", 0)) + len(token_list)
        logits = [0.0] * 16
        if token_list == [4]:
            logits[3] = 5.0
        elif token_list == [3]:
            logits[6] = 5.0
        else:
            logits[4] = 5.0
        return logits, next_state


def _fake_option_scorer(adapter, *, logits, state, rendered_answers):
    del adapter, logits, state, rendered_answers
    return {"A": -0.1, "B": -1.0, "C": -2.0, "D": -3.0}


class _DerivedStateHarness:
    def __init__(self) -> None:
        self.snapshot_calls: list[dict] = []
        self.random_seeds: list[int] = []

    def snapshot(self, states, *, adapter, directory):
        del adapter, directory
        self.snapshot_calls.append(copy.deepcopy(dict(states)))
        restored = copy.deepcopy(dict(states))
        for state in restored.values():
            state["restored"] = True
        reports = {
            f"{combo[0]},{combo[1]}": {
                "sha256": "0" * 64,
                "size_bytes": 1,
                "component_count": 1,
            }
            for combo in states
        }
        return restored, reports

    def randomize(self, state, torch, *, seed):
        del torch
        self.random_seeds.append(seed)
        randomized = copy.deepcopy(state)
        randomized["random_seed"] = seed
        return randomized


class Exp001CV02StageBRWKVTests(unittest.TestCase):
    def _build_inputs(self):
        design = build_exp001c_v02_stage_b_design_manifest(
            design_config_path=DESIGN_CONFIG,
            project_root=ROOT,
        )
        protocol = build_exp001c_protocol_v02_manifest(
            config_path=PROTOCOL_CONFIG,
            project_root=ROOT,
        )
        return design, protocol

    def test_fake_rwkv_adapter_exercises_all_state_routes(self) -> None:
        design, protocol = self._build_inputs()
        adapter = _FakeRWKVAdapter()
        harness = _DerivedStateHarness()
        result = RWKVExp001CV02StageBBackend(
            adapter=adapter,
            protocol_manifest=protocol,
            option_scorer=_fake_option_scorer,
            randomizer=harness.randomize,
            snapshot_roundtrip=harness.snapshot,
        ).run_stage_b(design)

        self.assertEqual(result["record_count"], 224)
        self.assertEqual(result["condition_count"], 7)
        self.assertTrue(result["model_executed"])
        self.assertTrue(result["recurrent_state_accessed"])
        self.assertFalse(result["stage_a_rerun"])
        self.assertFalse(result["formal_test_set_accessed"])
        self.assertFalse(result["contains_confirmatory_decision"])
        self.assertEqual(
            Counter(record["condition"] for record in result["records"]),
            Counter({condition: 32 for condition in STAGE_B_CONDITIONS}),
        )
        self.assertEqual(len(harness.snapshot_calls), 2)
        self.assertTrue(all(len(call) == 4 for call in harness.snapshot_calls))
        self.assertEqual(len(harness.random_seeds), 8)
        self.assertEqual(len(set(harness.random_seeds)), 8)
        self.assertEqual(set(result["snapshot_roundtrip_reports"]), {"block-000", "block-001"})

        by_id = {record["record_id"]: record for record in design["records"]}
        for record in result["records"]:
            route = by_id[record["record_id"]]
            self.assertEqual(
                record["state_source_history_key"],
                route["state_source_history_key"],
            )
            self.assertEqual(record["prefix_evidence"]["token_ids"], [4, 3])
            self.assertTrue(record["prefix_evidence"]["roundtrip_exact"])
            if record["condition"] in {"swapped_I", "swapped_G", "swapped_both"}:
                self.assertNotEqual(
                    record["state_source_history_key"],
                    record["query_history_key"],
                )
            if record["condition"] == "reset":
                self.assertIsNone(record["state_source_history_key"])
                self.assertIsNone(record["answer_boundary_evidence"])
            elif record["condition"] == "random_matched":
                self.assertEqual(
                    record["state_source_history_key"],
                    record["query_history_key"],
                )
                self.assertIsNone(record["answer_boundary_evidence"])
            else:
                self.assertEqual(
                    record["answer_boundary_evidence"]["target_code"],
                    record["expected_state_semantic_target_code"],
                )

    def test_protocol_digest_drift_is_rejected_before_scoring(self) -> None:
        design, protocol = self._build_inputs()
        design["protocol_manifest_digest_sha256"] = "0" * 64
        adapter = _FakeRWKVAdapter()
        with self.assertRaisesRegex(ValueError, "invalid design"):
            RWKVExp001CV02StageBBackend(
                adapter=adapter,
                protocol_manifest=protocol,
                option_scorer=_fake_option_scorer,
            ).run_stage_b(design)
        self.assertEqual(adapter.forward_calls, [])

    def test_factory_rejects_wrong_model_path_before_model_load(self) -> None:
        design, _ = self._build_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_b_design_manifest.json"
            path.write_text(json.dumps(design), encoding="utf-8")
            with patch(
                "psa.development.exp001c_v02_stage_b_rwkv.RWKV7Adapter.load"
            ) as model_loader:
                with self.assertRaisesRegex(ValueError, "model config"):
                    build_exp001c_v02_stage_b_rwkv_backend(
                        design_manifest_path=path,
                        model_config_path=DESIGN_CONFIG,
                        project_root=ROOT,
                        execution_authority_validated=True,
                    )
                model_loader.assert_not_called()

    def test_factory_requires_validated_authority_before_paths_or_load(self) -> None:
        with patch(
            "psa.development.exp001c_v02_stage_b_rwkv.RWKV7Adapter.load"
        ) as model_loader:
            with self.assertRaisesRegex(
                PermissionError,
                "validated execution authority",
            ):
                build_exp001c_v02_stage_b_rwkv_backend(
                    design_manifest_path="missing-design.json",
                    model_config_path="missing-model.json",
                    project_root=ROOT,
                )
            model_loader.assert_not_called()

    def test_real_result_schema_is_valid_json(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "exp001c_v02_stage_b_result.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["record_count"]["const"], 224)


if __name__ == "__main__":
    unittest.main()
