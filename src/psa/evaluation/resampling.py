from __future__ import annotations

import itertools
import math
import random
from statistics import NormalDist, mean
from typing import Iterable, Sequence


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty data")
    probability = min(1.0, max(0.0, probability))
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def bca_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval for a mean."""
    data = [float(value) for value in values]
    if len(data) < 2:
        raise ValueError("BCa interval requires at least two values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    if not all(math.isfinite(value) for value in data):
        raise ValueError("values must be finite")

    rng = random.Random(seed)
    observed = mean(data)
    n = len(data)
    boot = sorted(
        mean(data[rng.randrange(n)] for _ in range(n))
        for _ in range(replicates)
    )

    less = sum(value < observed for value in boot)
    proportion = min(1.0 - 1e-9, max(1e-9, less / replicates))
    normal = NormalDist()
    z0 = normal.inv_cdf(proportion)

    jackknife = [
        mean(data[:index] + data[index + 1 :]) for index in range(n)
    ]
    jack_mean = mean(jackknife)
    deviations = [jack_mean - value for value in jackknife]
    numerator = sum(value**3 for value in deviations)
    denominator_base = sum(value**2 for value in deviations)
    acceleration = (
        numerator / (6.0 * denominator_base**1.5)
        if denominator_base > 0.0
        else 0.0
    )

    alpha = (1.0 - confidence) / 2.0

    def adjusted_probability(tail_probability: float) -> float:
        z_alpha = normal.inv_cdf(tail_probability)
        denominator = 1.0 - acceleration * (z0 + z_alpha)
        if abs(denominator) < 1e-12:
            return tail_probability
        return normal.cdf(z0 + (z0 + z_alpha) / denominator)

    lower_p = adjusted_probability(alpha)
    upper_p = adjusted_probability(1.0 - alpha)
    return _quantile(boot, lower_p), _quantile(boot, upper_p)


def sign_flip_test(
    values: Sequence[float],
    *,
    alternative: str = "greater",
    replicates: int = 100_000,
    seed: int = 0,
    exact_limit: int = 20,
) -> float:
    """One-sample paired randomization test using sign flips."""
    data = [float(value) for value in values]
    if not data:
        raise ValueError("sign-flip test requires data")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("unsupported alternative")
    if not all(math.isfinite(value) for value in data):
        raise ValueError("values must be finite")

    observed = mean(data)

    def as_extreme(candidate: float) -> bool:
        if alternative == "greater":
            return candidate >= observed
        if alternative == "less":
            return candidate <= observed
        return abs(candidate) >= abs(observed)

    n = len(data)
    if n <= exact_limit:
        statistics = (
            mean(sign * value for sign, value in zip(signs, data, strict=True))
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        )
        extreme = 0
        total = 0
        for statistic in statistics:
            total += 1
            extreme += int(as_extreme(statistic))
        return extreme / total

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        statistic = mean(
            value if rng.getrandbits(1) else -value for value in data
        )
        extreme += int(as_extreme(statistic))
    return (extreme + 1) / (replicates + 1)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    count = len(values)
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def equivalence_from_interval(
    interval: tuple[float, float],
    bounds: tuple[float, float],
) -> bool:
    lower, upper = interval
    bound_lower, bound_upper = bounds
    if lower > upper or bound_lower >= bound_upper:
        raise ValueError("invalid interval or equivalence bounds")
    return lower > bound_lower and upper < bound_upper

