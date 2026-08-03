from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from psa.artifacts import sha256_file
from psa.development.history_binding import _score_from_state
from psa.development.impl3 import score_continuations_after_prefix
from psa.model import clone_state
from psa.model.rwkv7 import _flatten_state
from psa.state import randomize_state_matched


Combo = tuple[int, int]


def _combo(value: Any) -> Combo:
    combo = tuple(value) if isinstance(value, (list, tuple)) else ()
    if combo not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("RWKV backend requires a valid I x G combo")
    return int(combo[0]), int(combo[1])


def derive_random_state_seed(
    *,
    group_id: str,
    combo: Combo,
    namespace: str = "PSA|EXP-001|confirmatory|random-matched-v1",
) -> int:
    payload = f"{namespace}|{group_id}|{combo[0]}|{combo[1]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def disk_roundtrip_states(
    states: Mapping[Combo, Any],
    *,
    adapter: Any,
    directory: str | Path,
) -> tuple[dict[Combo, Any], dict[str, Any]]:
    """Round-trip flat RWKV states through safetensors in a temporary group dir."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    safetensors = import_module("safetensors.torch")
    restored: dict[Combo, Any] = {}
    reports: dict[str, Any] = {}
    for combo in sorted(states):
        source_items = list(_flatten_state(states[combo]))
        if [path for path, _ in source_items] != [
            f"state[{index}]" for index in range(len(source_items))
        ]:
            raise ValueError("formal disk roundtrip requires a flat RWKV state list")
        tensor_map = {
            f"state_{index:04d}": tensor.detach().contiguous().cpu()
            for index, (_, tensor) in enumerate(source_items)
        }
        path = destination / f"state-{combo[0]}-{combo[1]}.safetensors"
        safetensors.save_file(
            tensor_map,
            str(path),
            metadata={
                "format": "psa-exp001-confirmatory-temporary-state",
                "identity": str(combo[0]),
                "goal": str(combo[1]),
            },
        )
        loaded = safetensors.load_file(str(path), device="cpu")
        restored_components = []
        for index, (_, source_tensor) in enumerate(source_items):
            key = f"state_{index:04d}"
            tensor = loaded[key].to(
                device=source_tensor.device,
                dtype=source_tensor.dtype,
            )
            restored_components.append(tensor)
        restored[combo] = restored_components
        reports[f"{combo[0]},{combo[1]}"] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "component_count": len(restored_components),
        }
    return restored, reports


def _default_state_scorer(
    adapter: Any,
    *,
    query_text: str,
    source_state: Any,
    rendered_answers: dict[str, str],
    forced_prefix: str,
) -> tuple[dict[str, float], int, dict[str, Any]]:
    return _score_from_state(
        adapter,
        query_text=query_text,
        source_state=source_state,
        rendered_answers=rendered_answers,
        forced_prefix=forced_prefix,
    )


def _default_prompt_scorer(
    adapter: Any,
    *,
    prompt: str,
    rendered_answers: dict[str, str],
    forced_prefix: str,
) -> tuple[dict[str, float], int, dict[str, Any]]:
    scores, _, _, token_count, prefix = score_continuations_after_prefix(
        adapter,
        prompt,
        rendered_answers,
        forced_prefix=forced_prefix,
    )
    return scores, token_count, prefix


class RWKVConfirmatoryBackend:
    """State-aware scorer used by the grouped runner after separate authorization."""

    def __init__(
        self,
        *,
        adapter: Any,
        forced_prefix: str = ">\n",
        rendered_answers: Mapping[str, str] | None = None,
        state_scorer: Callable[..., tuple[dict[str, float], int, dict[str, Any]]] = _default_state_scorer,
        prompt_scorer: Callable[..., tuple[dict[str, float], int, dict[str, Any]]] = _default_prompt_scorer,
        randomizer: Callable[..., Any] = randomize_state_matched,
        snapshot_roundtrip: Callable[..., tuple[dict[Combo, Any], dict[str, Any]]] = disk_roundtrip_states,
    ) -> None:
        self.adapter = adapter
        self.forced_prefix = forced_prefix
        self.rendered_answers = dict(
            rendered_answers or {code: code for code in "ABCD"}
        )
        self.state_scorer = state_scorer
        self.prompt_scorer = prompt_scorer
        self.randomizer = randomizer
        self.snapshot_roundtrip = snapshot_roundtrip
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._group_id: str | None = None
        self._states: dict[Combo, Any] = {}
        self._restored: dict[Combo, Any] = {}
        self._random: dict[Combo, Any] = {}
        self._snapshot_reports: dict[str, Any] = {}
        self._warmup_shapes: list[int] = []

    def _representative_trials(
        self,
        group: Mapping[str, Any],
    ) -> dict[Combo, Mapping[str, Any]]:
        representatives: dict[Combo, Mapping[str, Any]] = {}
        for trial in group["trials"]:
            target = trial["target_fields"]
            combo = _combo((target["identity"], target["goal"]))
            representatives.setdefault(combo, trial)
        if set(representatives) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("group does not provide four state histories")
        return representatives

    def _prewarm_shapes(self, group: Mapping[str, Any]) -> list[int]:
        lengths = set()
        for trial in group["trials"]:
            lengths.add(len(self.adapter.encode(trial["history_prompt"])))
            lengths.add(len(self.adapter.encode(trial["query_prompt"])))
            lengths.add(
                len(
                    self.adapter.encode(
                        trial["history_prompt"] + "\n\n" + trial["query_prompt"]
                    )
                )
            )
        neutral_token = self.adapter.encode(" neutral")[0]
        for length in sorted(lengths):
            self.adapter.forward([neutral_token] * length, None)
        return sorted(lengths)

    def start_group(self, group: Mapping[str, Any]) -> None:
        if self._group_id is not None:
            raise RuntimeError("RWKV backend already has an active group")
        self._group_id = str(group["factorial_group_id"])
        self._warmup_shapes = self._prewarm_shapes(group)
        representatives = self._representative_trials(group)
        for combo, trial in representatives.items():
            _, state = self.adapter.forward(
                self.adapter.encode(trial["history_prompt"]),
                None,
            )
            self._states[combo] = clone_state(state)
        self._temporary = tempfile.TemporaryDirectory(
            prefix=f"psa-{self._group_id}-"
        )
        self._restored, self._snapshot_reports = self.snapshot_roundtrip(
            self._states,
            adapter=self.adapter,
            directory=self._temporary.name,
        )
        for combo, state in self._states.items():
            seed = derive_random_state_seed(
                group_id=self._group_id,
                combo=combo,
            )
            self._random[combo] = self.randomizer(
                state,
                self.adapter.torch,
                seed=seed,
            )

    def _source_state(
        self,
        condition: str,
        *,
        query_combo: Combo,
        source_combo: Combo | None,
    ) -> Any:
        if condition == "reset":
            return None
        if condition == "random_matched":
            return clone_state(self._random[query_combo])
        if condition == "restored":
            return clone_state(self._restored[query_combo])
        if source_combo is None:
            raise ValueError(f"condition {condition} requires a source combo")
        return clone_state(self._states[source_combo])

    def score(self, *, group, trial, condition_plan):
        if group["factorial_group_id"] != self._group_id:
            raise RuntimeError("RWKV backend group context mismatch")
        condition = str(condition_plan["condition"])
        query_combo = _combo(condition_plan["query_target_combo"])
        raw_source_combo = condition_plan["state_source_combo"]
        source_combo = (
            _combo(raw_source_combo) if raw_source_combo is not None else None
        )
        if condition == "prompt_visible":
            prompt = trial["history_prompt"] + "\n\n" + trial["query_prompt"]
            scores, token_count, prefix = self.prompt_scorer(
                self.adapter,
                prompt=prompt,
                rendered_answers=self.rendered_answers,
                forced_prefix=self.forced_prefix,
            )
            scoring_mode = "prompt_visible_from_reset"
        else:
            source_state = self._source_state(
                condition,
                query_combo=query_combo,
                source_combo=source_combo,
            )
            scores, token_count, prefix = self.state_scorer(
                self.adapter,
                query_text=trial["query_prompt"],
                source_state=source_state,
                rendered_answers=self.rendered_answers,
                forced_prefix=self.forced_prefix,
            )
            scoring_mode = "state_condition"
        return {
            "option_scores": scores,
            "metadata": {
                "scoring_mode": scoring_mode,
                "prompt_token_count": token_count,
                "forced_prefix": prefix,
                "random_seed": (
                    derive_random_state_seed(
                        group_id=self._group_id,
                        combo=query_combo,
                    )
                    if condition == "random_matched"
                    else None
                ),
                "restored_snapshot_sha256": (
                    self._snapshot_reports[f"{query_combo[0]},{query_combo[1]}"][
                        "sha256"
                    ]
                    if condition == "restored"
                    else None
                ),
            },
        }

    def group_metadata(self) -> dict[str, Any]:
        return {
            "backend": "rwkv7_state_conditions_v0.1",
            "shape_warmup_token_lengths": list(self._warmup_shapes),
            "shape_warmup_excluded_from_scoring": True,
            "state_combo_count": len(self._states),
            "restored_snapshot_reports": dict(self._snapshot_reports),
            "random_seed_rule": (
                "SHA-256 first 64 bits of "
                "PSA|EXP-001|confirmatory|random-matched-v1|group|I|G"
            ),
        }

    def end_group(self, group: Mapping[str, Any]) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = None
        self._group_id = None
        self._states = {}
        self._restored = {}
        self._random = {}
        self._snapshot_reports = {}
        self._warmup_shapes = []
