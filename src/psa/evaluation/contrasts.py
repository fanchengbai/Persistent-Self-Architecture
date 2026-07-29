from __future__ import annotations

import math
from typing import Mapping


Combo = tuple[int, int]
OptionScores = Mapping[Combo, float]


def _require_scores(scores: OptionScores) -> None:
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if set(scores) != expected:
        raise ValueError("scores must contain all four I×G combinations")
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be finite")


def logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def identity_log_odds(scores: OptionScores) -> float:
    _require_scores(scores)
    return logsumexp([scores[(1, 0)], scores[(1, 1)]]) - logsumexp(
        [scores[(0, 0)], scores[(0, 1)]]
    )


def goal_log_odds(scores: OptionScores) -> float:
    _require_scores(scores)
    return logsumexp([scores[(0, 1)], scores[(1, 1)]]) - logsumexp(
        [scores[(0, 0)], scores[(1, 0)]]
    )


def joint_margin(scores: OptionScores, target: Combo) -> float:
    _require_scores(scores)
    if target not in scores:
        raise ValueError(f"unknown target combo: {target}")
    competitors = [value for combo, value in scores.items() if combo != target]
    return scores[target] - logsumexp(competitors)


def argmax_combo(scores: OptionScores) -> Combo:
    _require_scores(scores)
    return max(scores, key=scores.__getitem__)


def group_contrasts(
    state_scores: Mapping[Combo, OptionScores],
) -> dict[str, float]:
    """Compute preregistered group-level E1/E2/E3 contrasts."""
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if set(state_scores) != expected:
        raise ValueError("state_scores must contain all four source states")
    for scores in state_scores.values():
        _require_scores(scores)

    d_i = 0.5 * sum(
        identity_log_odds(state_scores[(1, goal)])
        - identity_log_odds(state_scores[(0, goal)])
        for goal in (0, 1)
    )
    d_g = 0.5 * sum(
        goal_log_odds(state_scores[(identity, 1)])
        - goal_log_odds(state_scores[(identity, 0)])
        for identity in (0, 1)
    )
    non_target_i = 0.5 * sum(
        goal_log_odds(state_scores[(1, goal)])
        - goal_log_odds(state_scores[(0, goal)])
        for goal in (0, 1)
    )
    non_target_g = 0.5 * sum(
        identity_log_odds(state_scores[(identity, 1)])
        - identity_log_odds(state_scores[(identity, 0)])
        for identity in (0, 1)
    )

    margins = [
        joint_margin(state_scores[combo], combo) for combo in sorted(expected)
    ]
    correct = [
        float(argmax_combo(state_scores[combo]) == combo)
        for combo in sorted(expected)
    ]
    return {
        "identity_transfer": d_i,
        "goal_transfer": d_g,
        "identity_non_target_change": non_target_i,
        "goal_non_target_change": non_target_g,
        "identity_specificity": abs(d_i) - abs(non_target_i),
        "goal_specificity": abs(d_g) - abs(non_target_g),
        "mean_joint_margin": sum(margins) / len(margins),
        "joint_accuracy": sum(correct) / len(correct),
    }

