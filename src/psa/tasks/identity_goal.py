from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import random
from typing import Any, Iterable, Literal, Sequence


Track = Literal["synthetic", "natural"]
Combo = tuple[int, int]

_COMBOS: tuple[Combo, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
_HISTORY_ORDERS = ("I_G", "G_I", "I_NEUTRAL_G", "G_NEUTRAL_I")


def _opaque_id(kind: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{kind}_{digest}"


def _require_pair(name: str, values: Sequence[str]) -> tuple[str, str]:
    if len(values) != 2 or len(set(values)) != 2:
        raise ValueError(f"{name} must contain exactly two distinct labels")
    if any(not value.strip() for value in values):
        raise ValueError(f"{name} labels must be non-empty")
    return values[0], values[1]


@dataclass(frozen=True)
class Option:
    code: str
    identity: int
    goal: int
    text: str

    @property
    def combo(self) -> Combo:
        return self.identity, self.goal

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectorySample:
    sample_id: str
    trajectory_id: str
    factorial_group_id: str
    identity: int
    goal: int
    history: str
    common_suffix: str
    query: str
    correct_code: str
    correct_identity: int
    correct_goal: int
    generation_factors: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorialGroup:
    group_id: str
    generator_version: str
    group_seed: int
    track: Track
    identity_labels: tuple[str, str]
    goal_labels: tuple[str, str]
    history_order: str
    option_permutation: tuple[Combo, ...]
    options: tuple[Option, ...]
    trajectories: tuple[TrajectorySample, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["option_permutation"] = [list(combo) for combo in self.option_permutation]
        return payload


def _render_option(
    track: Track,
    code: str,
    combo: Combo,
    identity_labels: tuple[str, str],
    goal_labels: tuple[str, str],
) -> Option:
    identity, goal = combo
    if track == "synthetic":
        text = f"{identity_labels[identity]} / {goal_labels[goal]}"
    else:
        text = (
            f"Perform {goal_labels[goal]} in "
            f"the {identity_labels[identity]} domain"
        )
    return Option(code=code, identity=identity, goal=goal, text=text)


def _render_query(track: Track, options: Iterable[Option]) -> str:
    if track == "synthetic":
        header = (
            "RULE: choose the one action matching both this instance's "
            "constraint and active mission.\n"
            "Select one code:\n"
        )
    else:
        header = (
            "Choose the unique action that matches both this agent instance's "
            "authorized domain and its active mission.\n"
        )
    body = "\n".join(f"{option.code}. {option.text}" for option in options)
    return f"{header}{body}\nAnswer:"


def _render_history_statements(
    track: Track,
    identity_label: str,
    goal_label: str,
) -> tuple[str, str, str]:
    if track == "synthetic":
        identity = f"INSTANCE-CONSTRAINT: domain = {identity_label}"
        goal = f"ACTIVE-MISSION: operation = {goal_label}"
        neutral = "NEUTRAL-RECORD: system status = nominal"
    else:
        identity = (
            "This agent instance is authorized for the "
            f"{identity_label} domain until explicitly revoked."
        )
        goal = (
            "The active mission for this agent instance is "
            f"{goal_label} until completion or cancellation."
        )
        neutral = "An unrelated monitoring record reports normal operation."
    return identity, goal, neutral


def _render_history(
    track: Track,
    identity_label: str,
    goal_label: str,
    history_order: str,
) -> str:
    identity, goal, neutral = _render_history_statements(
        track, identity_label, goal_label
    )
    orders = {
        "I_G": (identity, goal),
        "G_I": (goal, identity),
        "I_NEUTRAL_G": (identity, neutral, goal),
        "G_NEUTRAL_I": (goal, neutral, identity),
    }
    try:
        statements = orders[history_order]
    except KeyError as exc:
        raise ValueError(f"unsupported history_order: {history_order}") from exc
    return "\n".join(statements)


def _render_common_suffix(track: Track, delay_units: int) -> str:
    if delay_units < 0:
        raise ValueError("delay_units must be non-negative")
    if delay_units == 0:
        return ""
    sentence = (
        "NEUTRAL-FILLER: unrelated system record."
        if track == "synthetic"
        else "An unrelated system record was reviewed and required no action."
    )
    return "\n".join(sentence for _ in range(delay_units))


def generate_factorial_group(
    *,
    group_seed: int,
    track: Track = "synthetic",
    identity_labels: Sequence[str] = ("dax", "kel"),
    goal_labels: Sequence[str] = ("mip", "rov"),
    answer_codes: Sequence[str] = ("A", "B", "C", "D"),
    delay_units: int = 1,
    generator_version: str = "0.1",
    history_order: str | None = None,
) -> FactorialGroup:
    """Generate one matched 2x2 identity-goal factorial group."""
    if track not in ("synthetic", "natural"):
        raise ValueError(f"unsupported track: {track}")
    identities = _require_pair("identity_labels", identity_labels)
    goals = _require_pair("goal_labels", goal_labels)
    if len(answer_codes) != 4 or len(set(answer_codes)) != 4:
        raise ValueError("answer_codes must contain four distinct values")

    rng = random.Random(group_seed)
    option_permutation = list(_COMBOS)
    rng.shuffle(option_permutation)
    selected_history_order = (
        history_order
        if history_order is not None
        else _HISTORY_ORDERS[group_seed % len(_HISTORY_ORDERS)]
    )
    if selected_history_order not in _HISTORY_ORDERS:
        raise ValueError(f"unsupported history_order: {selected_history_order}")

    options = tuple(
        _render_option(track, code, combo, identities, goals)
        for code, combo in zip(answer_codes, option_permutation, strict=True)
    )
    query = _render_query(track, options)
    common_suffix = _render_common_suffix(track, delay_units)
    group_id = _opaque_id(
        "grp",
        generator_version,
        group_seed,
        track,
        *identities,
        *goals,
        *answer_codes,
        delay_units,
        selected_history_order,
    )

    code_by_combo = {option.combo: option.code for option in options}
    trajectories: list[TrajectorySample] = []
    for identity, goal in _COMBOS:
        trajectory_id = _opaque_id("trj", group_id, identity, goal)
        sample_id = _opaque_id("smp", trajectory_id, "query")
        history = _render_history(
            track,
            identities[identity],
            goals[goal],
            selected_history_order,
        )
        factors = {
            "identity_value": identity,
            "goal_value": goal,
            "history_order": selected_history_order,
            "delay_units": delay_units,
            "track": track,
            "generator_version": generator_version,
            "group_seed": group_seed,
        }
        trajectories.append(
            TrajectorySample(
                sample_id=sample_id,
                trajectory_id=trajectory_id,
                factorial_group_id=group_id,
                identity=identity,
                goal=goal,
                history=history,
                common_suffix=common_suffix,
                query=query,
                correct_code=code_by_combo[(identity, goal)],
                correct_identity=identity,
                correct_goal=goal,
                generation_factors=factors,
            )
        )

    return FactorialGroup(
        group_id=group_id,
        generator_version=generator_version,
        group_seed=group_seed,
        track=track,
        identity_labels=identities,
        goal_labels=goals,
        history_order=selected_history_order,
        option_permutation=tuple(option_permutation),
        options=options,
        trajectories=tuple(trajectories),
    )


def generate_dataset(
    *,
    group_count: int,
    base_seed: int,
    track: Track = "synthetic",
    identity_label_pairs: Sequence[Sequence[str]] = (("dax", "kel"),),
    goal_label_pairs: Sequence[Sequence[str]] = (("mip", "rov"),),
    answer_codes: Sequence[str] = ("A", "B", "C", "D"),
    delay_units: int = 1,
) -> tuple[FactorialGroup, ...]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if not identity_label_pairs or not goal_label_pairs:
        raise ValueError("label pair pools must be non-empty")

    seed_rng = random.Random(base_seed)
    groups = []
    label_cycle = itertools.cycle(
        itertools.product(identity_label_pairs, goal_label_pairs)
    )
    for index in range(group_count):
        identity_labels, goal_labels = next(label_cycle)
        group_seed = seed_rng.getrandbits(63)
        groups.append(
            generate_factorial_group(
                group_seed=group_seed,
                track=track,
                identity_labels=identity_labels,
                goal_labels=goal_labels,
                answer_codes=answer_codes,
                delay_units=delay_units,
                history_order=_HISTORY_ORDERS[index % len(_HISTORY_ORDERS)],
            )
        )
    return tuple(groups)
