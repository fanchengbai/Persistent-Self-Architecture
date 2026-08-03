from __future__ import annotations

import unittest

from psa.confirmatory import (
    RWKVConfirmatoryBackend,
    derive_random_state_seed,
    execute_group,
)
from tests.test_confirmatory_runner import _group


class FakeAdapter:
    def __init__(self) -> None:
        self.torch = object()
        self.forward_calls = []

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 251 for character in text] or [1]

    def forward(self, tokens, state=None):
        token_list = list(tokens)
        self.forward_calls.append((len(token_list), state))
        return "fake-logits", ("history-state", tuple(token_list))


class RecordingScorers:
    def __init__(self) -> None:
        self.state_calls = []
        self.prompt_calls = []

    def state(self, adapter, **kwargs):
        self.state_calls.append(kwargs)
        return (
            {code: float(index) for index, code in enumerate("ABCD")},
            len(adapter.encode(kwargs["query_text"])),
            {"greedy_exact": True, "text": kwargs["forced_prefix"]},
        )

    def prompt(self, adapter, **kwargs):
        self.prompt_calls.append(kwargs)
        return (
            {code: float(index) for index, code in enumerate("ABCD")},
            len(adapter.encode(kwargs["prompt"])),
            {"greedy_exact": True, "text": kwargs["forced_prefix"]},
        )


def _randomizer(state, torch, *, seed):
    del torch
    return ("random-state", seed, state)


def _snapshot_roundtrip(states, *, adapter, directory):
    del adapter, directory
    restored = {
        combo: ("restored-state", state) for combo, state in states.items()
    }
    reports = {
        f"{combo[0]},{combo[1]}": {
            "sha256": f"snapshot-{combo[0]}-{combo[1]}",
            "size_bytes": 1,
            "component_count": 1,
        }
        for combo in states
    }
    return restored, reports


class ConfirmatoryRWKVBackendTests(unittest.TestCase):
    def test_random_seed_is_deterministic_and_combo_specific(self) -> None:
        first = derive_random_state_seed(group_id="devgrp-0", combo=(0, 0))
        repeated = derive_random_state_seed(group_id="devgrp-0", combo=(0, 0))
        different = derive_random_state_seed(group_id="devgrp-0", combo=(1, 0))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_backend_routes_all_conditions_without_real_model(self) -> None:
        adapter = FakeAdapter()
        scorers = RecordingScorers()
        backend = RWKVConfirmatoryBackend(
            adapter=adapter,
            state_scorer=scorers.state,
            prompt_scorer=scorers.prompt,
            randomizer=_randomizer,
            snapshot_roundtrip=_snapshot_roundtrip,
        )
        result = execute_group(_group(0), backend)
        self.assertEqual(result["record_count"], 128)
        self.assertEqual(len(scorers.prompt_calls), 16)
        self.assertEqual(len(scorers.state_calls), 112)
        metadata = result["backend_group_metadata"]
        self.assertEqual(metadata["state_combo_count"], 4)
        self.assertTrue(metadata["shape_warmup_excluded_from_scoring"])
        self.assertEqual(len(metadata["restored_snapshot_reports"]), 4)

        records = result["records"]
        target_trial = next(
            record
            for record in records
            if record["query_target_combo"] == [0, 1]
            and record["condition"] == "swapped_both"
        )
        self.assertEqual(target_trial["state_source_combo"], [1, 0])
        restored = next(
            record
            for record in records
            if record["query_target_combo"] == [0, 1]
            and record["condition"] == "restored"
        )
        self.assertEqual(
            restored["metadata"]["restored_snapshot_sha256"],
            "snapshot-0-1",
        )
        randomized = next(
            record
            for record in records
            if record["query_target_combo"] == [0, 1]
            and record["condition"] == "random_matched"
        )
        self.assertEqual(
            randomized["metadata"]["random_seed"],
            derive_random_state_seed(group_id="devgrp-0", combo=(0, 1)),
        )
        self.assertIsNone(backend._group_id)
        self.assertEqual(backend._states, {})

    def test_backend_cleans_up_after_scoring_failure(self) -> None:
        adapter = FakeAdapter()

        def fail(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("injected scoring failure")

        backend = RWKVConfirmatoryBackend(
            adapter=adapter,
            state_scorer=fail,
            prompt_scorer=fail,
            randomizer=_randomizer,
            snapshot_roundtrip=_snapshot_roundtrip,
        )
        with self.assertRaisesRegex(RuntimeError, "injected"):
            execute_group(_group(0), backend)
        self.assertIsNone(backend._group_id)
        self.assertEqual(backend._states, {})


if __name__ == "__main__":
    unittest.main()
