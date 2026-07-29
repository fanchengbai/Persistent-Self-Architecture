from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from psa.tasks.identity_goal import FactorialGroup


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    group_count: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "group_count": self.group_count,
        }


def validate_group(group: FactorialGroup) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    expected_combos = {(0, 0), (0, 1), (1, 0), (1, 1)}
    option_combos = {option.combo for option in group.options}
    if option_combos != expected_combos:
        errors.append("options must cover each I×G combination exactly once")
    if len(group.options) != 4:
        errors.append("group must contain exactly four options")
    if len({option.code for option in group.options}) != 4:
        errors.append("answer codes must be unique")
    if len(group.trajectories) != 4:
        errors.append("group must contain exactly four trajectories")

    queries = {sample.query for sample in group.trajectories}
    suffixes = {sample.common_suffix for sample in group.trajectories}
    if len(queries) != 1:
        errors.append("all trajectories in a group must share one query")
    if len(suffixes) != 1:
        errors.append("all trajectories in a group must share one common suffix")

    trajectory_combos = {
        (sample.identity, sample.goal) for sample in group.trajectories
    }
    if trajectory_combos != expected_combos:
        errors.append("trajectories must cover each I×G state exactly once")

    code_by_combo = {option.combo: option.code for option in group.options}
    for sample in group.trajectories:
        expected_code = code_by_combo.get((sample.identity, sample.goal))
        if sample.correct_code != expected_code:
            errors.append(f"incorrect answer mapping for sample {sample.sample_id}")
        if sample.factorial_group_id != group.group_id:
            errors.append(f"group ID mismatch for sample {sample.sample_id}")
        if group.identity_labels[sample.identity] not in sample.history:
            errors.append(f"identity binding missing in sample {sample.sample_id}")
        if group.goal_labels[sample.goal] not in sample.history:
            errors.append(f"goal binding missing in sample {sample.sample_id}")

    correct_codes = [sample.correct_code for sample in group.trajectories]
    if set(correct_codes) != {option.code for option in group.options}:
        errors.append("each answer position must be correct exactly once per group")

    forbidden_query_markers = (
        "INSTANCE-CONSTRAINT:",
        "ACTIVE-MISSION:",
        "This agent instance is authorized for",
        "The active mission for this agent instance is",
    )
    query = next(iter(queries), "")
    for marker in forbidden_query_markers:
        if marker in query:
            errors.append(f"query leaks a binding statement: {marker}")

    all_labels = (*group.identity_labels, *group.goal_labels)
    for opaque_id in (
        group.group_id,
        *(sample.sample_id for sample in group.trajectories),
        *(sample.trajectory_id for sample in group.trajectories),
    ):
        if any(label.lower() in opaque_id.lower() for label in all_labels):
            errors.append("opaque IDs must not contain literal task labels")

    history_lengths = [len(sample.history) for sample in group.trajectories]
    if history_lengths and max(history_lengths) != min(history_lengths):
        warnings.append(
            "history character lengths differ; tokenizer-level matching is required"
        )

    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_dataset(groups: Iterable[FactorialGroup]) -> ValidationReport:
    group_list = list(groups)
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    answer_counts: dict[str, int] = {}
    order_counts: dict[str, int] = {}

    for group in group_list:
        report = validate_group(group)
        errors.extend(f"{group.group_id}: {item}" for item in report.errors)
        warnings.extend(f"{group.group_id}: {item}" for item in report.warnings)
        if group.group_id in seen_ids:
            errors.append(f"duplicate group_id: {group.group_id}")
        seen_ids.add(group.group_id)
        order_counts[group.history_order] = order_counts.get(group.history_order, 0) + 1
        for sample in group.trajectories:
            answer_counts[sample.correct_code] = (
                answer_counts.get(sample.correct_code, 0) + 1
            )

    if not group_list:
        errors.append("dataset must contain at least one group")
    if answer_counts and len(set(answer_counts.values())) != 1:
        errors.append("correct answer codes are not exactly balanced")
    if len(group_list) >= 4 and len(order_counts) < 2:
        warnings.append("dataset uses fewer than two history orders")

    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        group_count=len(group_list),
    )

