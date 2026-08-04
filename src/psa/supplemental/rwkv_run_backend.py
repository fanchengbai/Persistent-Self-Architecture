from __future__ import annotations

from typing import Any, Mapping, Sequence

from psa.confirmatory.rwkv_backend import RWKVConfirmatoryBackend, _combo
from psa.development.history_binding import _score_from_state
from psa.development.impl3 import greedy_format_probe, score_continuations_after_prefix
from psa.model import clone_state
from psa.supplemental.development import _state_rms, evaluate_state_norms


class RWKVSupplementalRunBackend:
    """Authorized RWKV scorer for the three frozen EXP-001B record families."""

    def __init__(
        self,
        *,
        adapter: Any,
        state_norm_thresholds: Mapping[str, Any],
        rendered_answers: Mapping[str, str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.rendered_answers = dict(
            rendered_answers or {code: code for code in "ABCD"}
        )
        self.thresholds = dict(state_norm_thresholds)
        self.condition_backend = RWKVConfirmatoryBackend(
            adapter=adapter,
            forced_prefix=">\n",
            rendered_answers=self.rendered_answers,
        )
        self._group_id: str | None = None
        self._warmup_shapes: list[int] = []
        self._kind_counts: dict[str, int] = {}

    def _prewarm(
        self,
        core_group: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        lengths: set[int] = set()
        for record in records:
            kind = record["record_kind"]
            if kind == "matched_context":
                lengths.add(len(self.adapter.encode(record["history_prompt"])))
                lengths.add(len(self.adapter.encode(record["query_prompt"])))
            elif kind == "formal_generation_readout":
                lengths.add(len(self.adapter.encode(record["prompt_visible_prompt"])))
            elif kind == "general_capability_control_condition":
                lengths.add(len(self.adapter.encode(record["prompt"])))
        neutral_token = self.adapter.encode(" neutral")[0]
        for length in sorted(lengths):
            self.adapter.forward([neutral_token] * length, None)
        return sorted(lengths)

    def start_group(
        self,
        core_group: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._group_id is not None:
            raise RuntimeError("EXP-001B backend already has an active group")
        self._group_id = str(core_group["factorial_group_id"])
        self._warmup_shapes = self._prewarm(core_group, records)
        self._kind_counts = {}
        for record in records:
            kind = str(record["record_kind"])
            self._kind_counts[kind] = self._kind_counts.get(kind, 0) + 1
        self.condition_backend.start_group(core_group)

    def _matched(self, record: Mapping[str, Any]) -> dict[str, Any]:
        _, state = self.adapter.forward(
            self.adapter.encode(record["history_prompt"]),
            None,
        )
        norm = evaluate_state_norms(
            _state_rms(state, self.adapter.torch),
            self.thresholds,
        )
        ratios = [
            (
                float(item["observed_rms"]) / float(item["threshold_rms"])
                if float(item["threshold_rms"]) > 0.0
                else float("inf")
            )
            for item in norm["alerts"]
        ]
        scores, token_count, prefix = _score_from_state(
            self.adapter,
            query_text=record["query_prompt"],
            source_state=state,
            rendered_answers=self.rendered_answers,
            forced_prefix=">\n",
        )
        return {
            "option_scores": scores,
            "metadata": {
                "scoring_mode": "token_matched_unbound_state",
                "query_token_count": token_count,
                "forced_prefix": prefix,
                "state_norm_component_count": norm["component_count"],
                "state_norm_alert_count": norm["alert_count"],
                "state_norm_alert_paths": [item["path"] for item in norm["alerts"]],
                "state_norm_max_alert_ratio": max(ratios, default=0.0),
            },
        }

    def _generation(self, record: Mapping[str, Any]) -> dict[str, Any]:
        scores, logits, state, token_count, prefix = score_continuations_after_prefix(
            self.adapter,
            record["prompt_visible_prompt"],
            self.rendered_answers,
            forced_prefix=record["forced_answer_prefix"],
        )
        generated = greedy_format_probe(
            self.adapter,
            logits,
            state,
            answer_codes=tuple("ABCD"),
            max_tokens=int(record["maximum_generated_tokens_after_prefix"]),
        )
        return {
            "option_scores": scores,
            "metadata": {
                "scoring_mode": "formal_prompt_visible_generation",
                "prompt_token_count": token_count,
                "forced_prefix": prefix,
            },
            **generated,
        }

    def _control(self, record: Mapping[str, Any]) -> dict[str, Any]:
        condition = str(record["condition"])
        query_combo = _combo(record["assigned_source_combo"])
        raw_source = record.get("state_source_combo")
        source_combo = _combo(raw_source) if raw_source is not None else None
        if condition == "prompt_visible_reset":
            scores, token_count, prefix = self.condition_backend.prompt_scorer(
                self.adapter,
                prompt=record["prompt"],
                rendered_answers=self.rendered_answers,
                forced_prefix=">\n",
            )
            mode = "prompt_visible_reset"
        else:
            source_state = self.condition_backend._source_state(
                condition,
                query_combo=query_combo,
                source_combo=source_combo,
            )
            scores, token_count, prefix = self.condition_backend.state_scorer(
                self.adapter,
                query_text=record["prompt"],
                source_state=source_state,
                rendered_answers=self.rendered_answers,
                forced_prefix=">\n",
            )
            mode = "state_condition_control"
        return {
            "option_scores": scores,
            "metadata": {
                "scoring_mode": mode,
                "condition": condition,
                "prompt_token_count": token_count,
                "forced_prefix": prefix,
            },
        }

    def score_record(self, *, core_group, record):
        if core_group["factorial_group_id"] != self._group_id:
            raise RuntimeError("EXP-001B backend group context mismatch")
        kind = record["record_kind"]
        if kind == "matched_context":
            return self._matched(record)
        if kind == "formal_generation_readout":
            return self._generation(record)
        if kind == "general_capability_control_condition":
            return self._control(record)
        raise ValueError(f"unsupported EXP-001B record kind: {kind}")

    def group_metadata(self) -> dict[str, Any]:
        return {
            "backend": "rwkv7_exp001b_supplemental_v0.1",
            "shape_warmup_token_lengths": list(self._warmup_shapes),
            "shape_warmup_excluded_from_scoring": True,
            "record_kind_counts": dict(self._kind_counts),
            "condition_backend": self.condition_backend.group_metadata(),
        }

    def end_group(self, core_group: Mapping[str, Any]) -> None:
        self.condition_backend.end_group(core_group)
        self._group_id = None
        self._warmup_shapes = []
        self._kind_counts = {}
