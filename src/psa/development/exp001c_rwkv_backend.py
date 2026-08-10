from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from psa.artifacts import sha256_file
from psa.confirmatory.rwkv_backend import (
    derive_random_state_seed,
    disk_roundtrip_states,
)
from psa.development.prefix_instrumentation import (
    answer_boundary_evidence,
    instrument_forced_prefix,
)
from psa.model import RWKV7Adapter, clone_state, load_model_config
from psa.state import randomize_state_matched


Combo = tuple[int, int]
COMBOS: tuple[Combo, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
CONDITIONS = (
    "continuous",
    "restored",
    "swapped_I",
    "swapped_G",
    "swapped_both",
    "reset",
    "prompt_visible_reset",
    "random_matched",
)
RANDOM_NAMESPACE = "PSA|EXP-001C|noncore-development-pilot|random-matched-v1"


def _combo_key(combo: Combo) -> str:
    return f"{combo[0]},{combo[1]}"


def _source_combo(condition: str, query_combo: Combo) -> Combo | None:
    identity, goal = query_combo
    if condition in {"continuous", "restored"}:
        return query_combo
    if condition == "swapped_I":
        return 1 - identity, goal
    if condition == "swapped_G":
        return identity, 1 - goal
    if condition == "swapped_both":
        return 1 - identity, 1 - goal
    if condition in {"reset", "prompt_visible_reset", "random_matched"}:
        return None
    raise ValueError(f"unsupported EXP-001C development condition: {condition}")


def load_exp001c_noncore_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("EXP-001C fixture must be a JSON object")
    if (
        fixture.get("fixture_version") != "0.1-development"
        or fixture.get("fixture_kind") != "noncore_formal_shape_prefix_probe"
        or fixture.get("development_only") is not True
        or fixture.get("non_core") is not True
        or fixture.get("formal_test_set_accessed") is not False
    ):
        raise ValueError("EXP-001C fixture violates the non-Core development boundary")
    histories = fixture.get("histories")
    queries = fixture.get("queries")
    expected_keys = {_combo_key(combo) for combo in COMBOS}
    if not isinstance(histories, Mapping) or set(histories) != expected_keys:
        raise ValueError("EXP-001C fixture must provide exactly four histories")
    if not isinstance(queries, Mapping) or set(queries) != expected_keys:
        raise ValueError("EXP-001C fixture must provide exactly four queries")
    for key in sorted(expected_keys):
        history = histories[key]
        query = queries[key]
        if not isinstance(history, str) or "NON-CORE DEVELOPMENT" not in history:
            raise ValueError(f"fixture history {key} lacks its development marker")
        if (
            not isinstance(query, Mapping)
            or not isinstance(query.get("prompt"), str)
            or "NON-CORE DEVELOPMENT" not in str(query.get("prompt"))
            or query.get("target_code") not in set("ABCD")
        ):
            raise ValueError(f"fixture query {key} is invalid")
    forced_prefix = fixture.get("forced_prefix")
    if not isinstance(forced_prefix, str) or not forced_prefix:
        raise ValueError("EXP-001C fixture requires a forced prefix")
    rendered_answers = fixture.get("rendered_answers")
    if (
        not isinstance(rendered_answers, Mapping)
        or set(rendered_answers) != set("ABCD")
        or not all(isinstance(value, str) and value for value in rendered_answers.values())
    ):
        raise ValueError("EXP-001C fixture requires non-empty A-D rendered answers")
    return fixture


def _default_option_scorer(
    adapter: Any,
    *,
    logits: Any,
    state: Any,
    rendered_answers: Mapping[str, str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for code in "ABCD":
        answer_tokens = adapter.encode(rendered_answers[code])
        answer_logits = logits
        answer_state = clone_state(state)
        score = 0.0
        for index, token in enumerate(answer_tokens):
            log_probabilities = adapter.torch.log_softmax(
                answer_logits.float(),
                dim=-1,
            )
            score += float(log_probabilities[token].item())
            if index + 1 < len(answer_tokens):
                answer_logits, answer_state = adapter.forward(
                    [token],
                    answer_state,
                )
        scores[code] = score
    return scores


class RWKVExp001CDevelopmentBackend:
    """Non-Core EXP-001C scorer instantiated only after execution authorization."""

    def __init__(
        self,
        *,
        adapter: Any,
        fixture: Mapping[str, Any],
        fixture_path: str | Path,
        option_scorer: Callable[..., dict[str, float]] = _default_option_scorer,
        randomizer: Callable[..., Any] = randomize_state_matched,
        snapshot_roundtrip: Callable[..., tuple[dict[Combo, Any], dict[str, Any]]] = disk_roundtrip_states,
    ) -> None:
        self.adapter = adapter
        self.fixture = dict(fixture)
        self.fixture_path = Path(fixture_path).resolve()
        self.option_scorer = option_scorer
        self.randomizer = randomizer
        self.snapshot_roundtrip = snapshot_roundtrip

    def _prewarm_shapes(self) -> list[int]:
        histories = self.fixture["histories"]
        queries = self.fixture["queries"]
        lengths = set()
        for combo in COMBOS:
            key = _combo_key(combo)
            history = histories[key]
            query = queries[key]["prompt"]
            lengths.add(len(self.adapter.encode(history)))
            lengths.add(len(self.adapter.encode(query)))
            lengths.add(len(self.adapter.encode(history + "\n\n" + query)))
        neutral_token = self.adapter.encode(" neutral")[0]
        for length in sorted(lengths):
            self.adapter.forward([neutral_token] * length, None)
        return sorted(lengths)

    def run_probe(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            manifest.get("experiment_id") != "EXP-001C"
            or manifest.get("development_only") is not True
            or manifest.get("formal_test_set_accessed") is not False
            or manifest.get("model_executed") is not False
        ):
            raise ValueError("EXP-001C backend received an invalid locked manifest")

        histories = self.fixture["histories"]
        queries = self.fixture["queries"]
        forced_prefix = str(self.fixture["forced_prefix"])
        prefix_tokens = self.adapter.encode(forced_prefix)
        if len(prefix_tokens) != 2:
            raise ValueError("EXP-001C forced prefix must tokenize to exactly two tokens")
        warmup_shapes = self._prewarm_shapes()
        states = {}
        for combo in COMBOS:
            _, state = self.adapter.forward(
                self.adapter.encode(histories[_combo_key(combo)]),
                None,
            )
            states[combo] = clone_state(state)

        records = []
        group_id = str(self.fixture["fixture_id"])
        with tempfile.TemporaryDirectory(prefix="psa-exp001c-noncore-") as temporary:
            restored, snapshot_reports = self.snapshot_roundtrip(
                states,
                adapter=self.adapter,
                directory=temporary,
            )
            random_states = {}
            for combo in COMBOS:
                random_states[combo] = self.randomizer(
                    states[combo],
                    self.adapter.torch,
                    seed=derive_random_state_seed(
                        group_id=group_id,
                        combo=combo,
                        namespace=RANDOM_NAMESPACE,
                    ),
                )

            for condition in CONDITIONS:
                for combo in COMBOS:
                    key = _combo_key(combo)
                    source_combo = _source_combo(condition, combo)
                    query_text = queries[key]["prompt"]
                    if condition == "prompt_visible_reset":
                        query_text = histories[key] + "\n\n" + query_text
                        source_state = None
                    elif condition == "reset":
                        source_state = None
                    elif condition == "random_matched":
                        source_state = clone_state(random_states[combo])
                    elif condition == "restored":
                        source_state = clone_state(restored[combo])
                    else:
                        if source_combo is None:
                            raise RuntimeError("missing state source combo")
                        source_state = clone_state(states[source_combo])

                    query_tokens = self.adapter.encode(query_text)
                    logits, query_state = self.adapter.forward(
                        query_tokens,
                        source_state,
                    )
                    prefix_evidence, answer_logits, answer_state = (
                        instrument_forced_prefix(
                            self.adapter,
                            logits=logits,
                            state=query_state,
                            prefix_token_ids=prefix_tokens,
                            forced_prefix_text=forced_prefix,
                            top_k=10,
                        )
                    )
                    option_scores = self.option_scorer(
                        self.adapter,
                        logits=answer_logits,
                        state=answer_state,
                        rendered_answers=self.fixture["rendered_answers"],
                    )
                    target_code = str(queries[key]["target_code"])
                    records.append(
                        {
                            "record_id": f"{group_id}-{condition}-{combo[0]}-{combo[1]}",
                            "condition": condition,
                            "query_combo": list(combo),
                            "state_source_combo": (
                                list(source_combo) if source_combo is not None else None
                            ),
                            "target_code": target_code,
                            "query_token_count": len(query_tokens),
                            "prefix_evidence": prefix_evidence,
                            "option_log_probabilities": {
                                code: float(option_scores[code]) for code in "ABCD"
                            },
                            "answer_boundary_evidence": answer_boundary_evidence(
                                option_scores,
                                target_code=target_code,
                            ),
                        }
                    )

        return {
            "probe_result_version": "0.1-development",
            "experiment_id": "EXP-001C",
            "status": "noncore_development_probe_complete",
            "development_only": True,
            "non_core_fixture": True,
            "model_executed": True,
            "formal_test_set_accessed": False,
            "contains_confirmatory_decision": False,
            "fixture_id": group_id,
            "fixture_sha256": sha256_file(self.fixture_path),
            "manifest_digest_sha256": manifest.get("manifest_digest_sha256"),
            "condition_count": len(CONDITIONS),
            "record_count": len(records),
            "warmup_token_lengths": warmup_shapes,
            "snapshot_roundtrip_reports": snapshot_reports,
            "records": records,
        }


def build_exp001c_rwkv_development_backend(
    *,
    manifest_path: str | Path,
    model_config_path: str | Path,
    fixture_path: str | Path,
    project_root: str | Path,
) -> RWKVExp001CDevelopmentBackend:
    """Load the real model only when an authorized runner invokes this factory."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    manifest = json.loads(manifest_file.resolve().read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("EXP-001C locked manifest must be a JSON object")
    model_entry = manifest.get("model_config")
    if not isinstance(model_entry, Mapping):
        raise ValueError("EXP-001C locked manifest lacks a model config")
    locked_model_path = (root / str(model_entry.get("path", ""))).resolve()
    requested_model_path = Path(model_config_path)
    if not requested_model_path.is_absolute():
        requested_model_path = root / requested_model_path
    requested_model_path = requested_model_path.resolve()
    if (
        requested_model_path != locked_model_path
        or not requested_model_path.is_file()
        or sha256_file(requested_model_path) != model_entry.get("sha256")
    ):
        raise ValueError("model config does not match the locked EXP-001C manifest")
    locked_fixture_path = (
        root
        / "configs"
        / "development"
        / "exp001c_noncore_formal_shape_fixture.v0.1.json"
    ).resolve()
    requested_fixture_path = Path(fixture_path)
    if not requested_fixture_path.is_absolute():
        requested_fixture_path = root / requested_fixture_path
    requested_fixture_path = requested_fixture_path.resolve()
    source_digests = manifest.get("locked_source_digests")
    fixture_relative = str(locked_fixture_path.relative_to(root)).replace("\\", "/")
    if (
        requested_fixture_path != locked_fixture_path
        or not requested_fixture_path.is_file()
        or not isinstance(source_digests, Mapping)
        or sha256_file(requested_fixture_path) != source_digests.get(fixture_relative)
    ):
        raise ValueError("fixture does not match the locked EXP-001C manifest")
    fixture = load_exp001c_noncore_fixture(requested_fixture_path)
    config = load_model_config(
        requested_model_path,
        project_root=root,
        verify_files=True,
    )
    adapter = RWKV7Adapter.load(config)
    return RWKVExp001CDevelopmentBackend(
        adapter=adapter,
        fixture=fixture,
        fixture_path=requested_fixture_path,
    )
