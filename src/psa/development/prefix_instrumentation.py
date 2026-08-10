from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


INSTRUMENTATION_VERSION = "0.1-development"
DEFAULT_PREFIX_POSITION_LABELS = (
    "greater_than_token",
    "newline_token",
)


def _float32_values(logits: Any) -> list[float]:
    value = logits
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 1:
        value = value.reshape(-1, shape[-1])[-1]
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("token logits must be a non-empty one-dimensional vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("token logits must be finite")
    return result


def token_evidence_from_logits(
    logits: Any,
    *,
    expected_token_id: int,
    top_k: int = 10,
) -> dict[str, Any]:
    values = _float32_values(logits)
    if not 0 <= expected_token_id < len(values):
        raise ValueError("expected token ID is outside the logit vocabulary")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ordered_ids = sorted(range(len(values)), key=lambda index: (-values[index], index))
    greedy_token_id = ordered_ids[0]
    selected_ids = ordered_ids[: min(top_k, len(ordered_ids))]
    maximum = values[greedy_token_id]
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    expected_logit = values[expected_token_id]
    greedy_logit = values[greedy_token_id]
    return {
        "expected_token_id": expected_token_id,
        "greedy_token_id": greedy_token_id,
        "expected_token_logit_float32": expected_logit,
        "expected_token_log_probability_float32": expected_logit - log_normalizer,
        "expected_token_rank": ordered_ids.index(expected_token_id) + 1,
        "greedy_token_logit_float32": greedy_logit,
        "greedy_token_log_probability_float32": greedy_logit - log_normalizer,
        "logit_margin_greedy_minus_expected_float32": (
            greedy_logit - expected_logit
        ),
        "top_k_token_ids": selected_ids,
        "top_k_logits_float32": [values[index] for index in selected_ids],
        "top_k_log_probabilities_float32": [
            values[index] - log_normalizer for index in selected_ids
        ],
    }


def answer_boundary_evidence(
    option_log_probabilities: Mapping[str, float],
    *,
    target_code: str,
) -> dict[str, Any]:
    if (
        set(option_log_probabilities) != set("ABCD")
        or target_code not in set("ABCD")
    ):
        raise ValueError("answer evidence requires A-D scores and an A-D target")
    scores = {code: float(option_log_probabilities[code]) for code in "ABCD"}
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("answer log probabilities must be finite")
    best_incorrect_code = max(
        (code for code in "ABCD" if code != target_code),
        key=lambda code: (scores[code], -ord(code)),
    )
    target_score = scores[target_code]
    best_incorrect_score = scores[best_incorrect_code]
    return {
        "target_code": target_code,
        "target_answer_log_probability": target_score,
        "best_incorrect_code": best_incorrect_code,
        "best_incorrect_answer_log_probability": best_incorrect_score,
        "target_margin_over_best_incorrect": target_score - best_incorrect_score,
    }


def instrument_forced_prefix(
    adapter: Any,
    *,
    logits: Any,
    state: Any,
    prefix_token_ids: Sequence[int],
    forced_prefix_text: str,
    position_labels: Sequence[str] | None = None,
    top_k: int = 10,
) -> tuple[dict[str, Any], Any, Any]:
    tokens = [int(token) for token in prefix_token_ids]
    if position_labels is None:
        labels = (
            list(DEFAULT_PREFIX_POSITION_LABELS)
            if len(tokens) == len(DEFAULT_PREFIX_POSITION_LABELS)
            else [f"prefix_position_{index}" for index in range(len(tokens))]
        )
    else:
        labels = [str(label) for label in position_labels]
    if len(labels) != len(tokens) or len(set(labels)) != len(labels):
        raise ValueError("prefix position labels must be unique and match token count")

    current_logits = logits
    current_state = state
    positions = []
    greedy_tokens = []
    for index, (label, expected_token_id) in enumerate(zip(labels, tokens)):
        evidence = token_evidence_from_logits(
            current_logits,
            expected_token_id=expected_token_id,
            top_k=top_k,
        )
        greedy_tokens.append(int(evidence["greedy_token_id"]))
        positions.append(
            {
                "position_index": index,
                "position_label": label,
                **evidence,
            }
        )
        current_logits, current_state = adapter.forward(
            [expected_token_id],
            current_state,
        )

    report = {
        "instrumentation_version": INSTRUMENTATION_VERSION,
        "development_only": True,
        "text": forced_prefix_text,
        "token_ids": tokens,
        "greedy_token_ids": greedy_tokens,
        "greedy_exact": greedy_tokens == tokens,
        "roundtrip_exact": adapter.decode(tokens) == forced_prefix_text,
        "top_k": top_k,
        "positions": positions,
    }
    return report, current_logits, current_state
