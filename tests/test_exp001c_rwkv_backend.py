from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.development.exp001c_probe import build_exp001c_probe_manifest
from psa.development.exp001c_rwkv_backend import (
    COMBOS,
    CONDITIONS,
    RWKVExp001CDevelopmentBackend,
    build_exp001c_rwkv_development_backend,
    load_exp001c_noncore_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_noncore_formal_shape_fixture.v0.1.json"
)
DESIGN = ROOT / "configs" / "preregistration" / "exp001c_prefix_semantics.draft.json"
MODEL = ROOT / "configs" / "models" / "rwkv7_g1h_2.9b.candidate.json"


class _FakeAdapter:
    torch = object()

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


def _fake_randomizer(state, torch, *, seed):
    del torch
    result = copy.deepcopy(state)
    result["random_seed"] = seed
    return result


def _fake_snapshot(states, *, adapter, directory):
    del adapter, directory
    restored = copy.deepcopy(dict(states))
    reports = {
        f"{combo[0]},{combo[1]}": {
            "sha256": "0" * 64,
            "size_bytes": 1,
            "component_count": 1,
        }
        for combo in COMBOS
    }
    return restored, reports


class Exp001CRWKVBackendTests(unittest.TestCase):
    def test_fixture_is_explicitly_noncore_and_formal_shape(self) -> None:
        fixture = load_exp001c_noncore_fixture(FIXTURE)
        self.assertTrue(fixture["development_only"])
        self.assertTrue(fixture["non_core"])
        self.assertFalse(fixture["formal_test_set_accessed"])
        self.assertEqual(set(fixture["histories"]), {"0,0", "0,1", "1,0", "1,1"})
        self.assertEqual(set(fixture["queries"]), {"0,0", "0,1", "1,0", "1,1"})

    def test_fake_adapter_exercises_all_conditions_and_prefix_evidence(self) -> None:
        fixture = load_exp001c_noncore_fixture(FIXTURE)
        backend = RWKVExp001CDevelopmentBackend(
            adapter=_FakeAdapter(),
            fixture=fixture,
            fixture_path=FIXTURE,
            option_scorer=_fake_option_scorer,
            randomizer=_fake_randomizer,
            snapshot_roundtrip=_fake_snapshot,
        )
        result = backend.run_probe(
            {
                "experiment_id": "EXP-001C",
                "development_only": True,
                "formal_test_set_accessed": False,
                "model_executed": False,
                "manifest_digest_sha256": "1" * 64,
            }
        )
        self.assertEqual(result["record_count"], 32)
        self.assertEqual(result["condition_count"], 8)
        self.assertTrue(result["model_executed"])
        self.assertFalse(result["formal_test_set_accessed"])
        self.assertFalse(result["contains_confirmatory_decision"])
        self.assertEqual(
            Counter(record["condition"] for record in result["records"]),
            Counter({condition: 4 for condition in CONDITIONS}),
        )
        self.assertEqual(
            {tuple(record["query_combo"]) for record in result["records"]},
            set(COMBOS),
        )
        for record in result["records"]:
            evidence = record["prefix_evidence"]
            self.assertEqual(evidence["token_ids"], [4, 3])
            self.assertEqual(evidence["greedy_token_ids"], [4, 2])
            self.assertFalse(evidence["greedy_exact"])
            self.assertTrue(evidence["roundtrip_exact"])
            self.assertEqual(len(evidence["positions"]), 2)
            self.assertIn("target_margin_over_best_incorrect", record["answer_boundary_evidence"])

    def test_fixture_loader_rejects_formal_or_core_marking(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["non_core"] = False
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "invalid.json"
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-Core"):
                load_exp001c_noncore_fixture(temporary)

    def test_factory_rejects_unlocked_model_path_before_model_load(self) -> None:
        manifest = build_exp001c_probe_manifest(
            design_config_path=DESIGN,
            model_config_path=MODEL,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "psa.development.exp001c_rwkv_backend.RWKV7Adapter.load"
            ) as model_loader:
                with self.assertRaisesRegex(ValueError, "model config"):
                    build_exp001c_rwkv_development_backend(
                        manifest_path=manifest_path,
                        model_config_path=FIXTURE,
                        fixture_path=FIXTURE,
                        project_root=ROOT,
                    )
                model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
