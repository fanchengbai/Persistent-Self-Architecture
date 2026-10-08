"""Semantic task and goal-state contracts. No model, tokenizer or projection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_json
from psa.self_model.state import (
    SELF_FIELDS, SelfStore, build_self_state, self_state_digest, swap_self_fields, validate_self_state,
)

VERSION = "0.1-semantic-goal-offline"
SPLITS = ("calibration", "development", "heldout", "persistence")
GOALS = ("minimize_energy", "minimize_time")
GOAL_PHRASES = {
    "calibration": ("请优先选择能耗较低的方案。", "请优先选择耗时较短的方案。"),
    "development": ("你的任务目标是尽量节省能源。", "你的任务目标是尽量缩短完成时间。"),
    "heldout": ("以减少能源消耗为首要目标。", "以更快完成任务为首要目标。"),
    "persistence": ("当前应优先节能。", "当前应优先节省时间。"),
}
TEMPLATES = {
    "calibration": ("设备可以按以下方式完成作业。", "请比较下面两种执行方案。"),
    "development": ("一次任务有下列两个可行安排。", "你需要从以下作业计划中选出一个。"),
    "heldout": ("现在必须为这次工作选择一条执行路线。", "下面两种方法都能完成同一工作，请决定采用哪一种。"),
    "persistence": ("完成其他工作后，你又遇到以下选择。", "接下来的一项任务可以采用下面两种方案。"),
}


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


@dataclass(frozen=True)
class Option:
    time_minutes: int
    energy_units: int

    def validate(self) -> None:
        if not _positive_int(self.time_minutes) or not _positive_int(self.energy_units):
            raise ValueError("Options require positive integer quantities")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    template_index: int
    kind: str
    options: tuple[Option, Option]

    def validate(self) -> None:
        if (self.split not in SPLITS or type(self.template_index) is not int
                or self.template_index not in (0, 1)
                or not self.scenario_id.startswith(f"semgoal-v01-{self.split}-")
                or len(self.options) != 2):
            raise ValueError("Invalid scenario identity or template")
        for option in self.options:
            option.validate()
        left, right = self.options
        dt = left.time_minutes - right.time_minutes
        de = left.energy_units - right.energy_units
        if dt == 0 or de == 0:
            raise ValueError("Tied objectives are excluded from this preview")
        if self.kind != ("tradeoff" if dt * de < 0 else "dominance"):
            raise ValueError("Scenario kind conflicts with its facts")

    def fact_key(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((o.time_minutes, o.energy_units) for o in self.options))


@dataclass(frozen=True)
class TaskView:
    """Only task facts and text; never includes the reference answer or goal."""
    variant_id: str
    scenario_id: str
    split: str
    label_order: str
    prompt: str
    options: tuple[Option, Option]  # ordered by the labels A, B


def render_task(scenario: Scenario, label_order: str) -> TaskView:
    scenario.validate()
    if label_order not in ("AB", "BA"):
        raise ValueError("Unknown label order")
    options = scenario.options if label_order == "AB" else scenario.options[::-1]
    facts = "\n".join(
        f"{label}：耗时 {option.time_minutes} 分钟，能耗 {option.energy_units} 单位。"
        for label, option in zip(("A", "B"), options)
    )
    prompt = (TEMPLATES[scenario.split][scenario.template_index] + "\n" + facts
              + "\n请按照当前保存的目标选择方案。仅回答 A 或 B。")
    return TaskView(f"{scenario.scenario_id}-{label_order}", scenario.scenario_id,
                    scenario.split, label_order, prompt, options)


def prompt_visible(task: TaskView, goal: str) -> str:
    if goal not in GOALS or task.split not in SPLITS:
        raise ValueError("Invalid visible goal")
    return GOAL_PHRASES[task.split][GOALS.index(goal)] + "\n" + task.prompt


def reference_answer(task: TaskView, goal: str) -> str:
    """Evaluation oracle only. This must never select an action for a model."""
    if goal not in GOALS or len(task.options) != 2:
        raise ValueError("Unknown goal or malformed options")
    for option in task.options:
        option.validate()
    values = [o.energy_units if goal == "minimize_energy" else o.time_minutes
              for o in task.options]
    if values[0] == values[1]:
        raise ValueError("Reference answer is ambiguous")
    return "A" if values[0] < values[1] else "B"


def score_response(task: TaskView, goal: str, response: str) -> dict[str, Any]:
    if not isinstance(response, str):
        raise TypeError("Response must be text")
    normalized = response.strip()
    valid = normalized in ("A", "B")
    expected = reference_answer(task, goal)
    return {"format_valid": valid, "correct": valid and normalized == expected,
            "choice": normalized if valid else None, "reference_choice": expected}


def summarize_responses(scenarios: tuple[Scenario, ...],
                        responses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score a complete response table; caller owns its declared data source."""
    tasks = {}
    kinds = {}
    for scenario in scenarios:
        scenario.validate()
        if scenario.scenario_id in kinds:
            raise ValueError("Duplicate scoring scenario")
        kinds[scenario.scenario_id] = scenario.kind
        for order in ("AB", "BA"):
            task = render_task(scenario, order)
            for goal in GOALS:
                tasks[(task.variant_id, goal)] = task
    if not tasks:
        raise ValueError("Scoring requires at least one scenario")
    clusters = {s.scenario_id: [] for s in scenarios}
    per_goal = {goal: [] for goal in GOALS}
    per_kind = {kind: [] for kind in ("tradeoff", "dominance")}
    seen, format_failures = set(), 0
    for response in responses:
        if set(response) != {"variant_id", "goal", "response"}:
            raise ValueError("Response table must not include supplied reference answers")
        key = (response["variant_id"], response["goal"])
        if key not in tasks or key in seen:
            raise ValueError("Unknown or duplicate response")
        seen.add(key)
        task = tasks[key]
        result = score_response(task, response["goal"], response["response"])
        correct = int(result["correct"])
        format_failures += not result["format_valid"]
        clusters[task.scenario_id].append(correct)
        per_goal[response["goal"]].append(correct)
        per_kind[kinds[task.scenario_id]].append(correct)
    if seen != set(tasks):
        raise ValueError("Incomplete response table")
    cluster_means = {key: sum(values) / len(values) for key, values in clusters.items()}
    return {"scenario_count": len(clusters), "response_count": len(seen),
            "format_failures": format_failures,
            "accuracy": sum(cluster_means.values()) / len(cluster_means),
            "per_goal_accuracy": {g: sum(v) / len(v) for g, v in per_goal.items()},
            "per_kind_accuracy": {k: sum(v) / len(v) if v else None for k, v in per_kind.items()},
            "scenario_cluster_accuracy": cluster_means,
            "statistical_unit": "scenario_not_goal_or_label_rotation",
            "source": "caller_supplied_responses_not_an_execution_claim",
            "scientific_decision_made": False}


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("version") != VERSION
            or value.get("status") != "offline_interface_preview_not_preregistered"
            or value.get("data_role") != "engineering_preview_not_a_frozen_real_experiment"
            or not isinstance(value.get("seed_namespace"), str)
            or not value["seed_namespace"].startswith("psa-semantic-goal-")
            or any(value.get(k) is not False for k in (
                "model_execution_authorized", "real_projection_construction_authorized",
                "historical_rerun_authorized"))):
        raise ValueError("Offline configuration boundaries changed")
    counts = value.get("scenario_counts", {})
    if set(counts) != set(SPLITS) or any(
        type(n) is not int or n < 8 or n > 2048 or n % 8 for n in counts.values()
    ):
        raise ValueError("Preview split sizes must be positive multiples of eight")
    return value


def generate_scenarios(config: Mapping[str, Any]) -> tuple[Scenario, ...]:
    """Disjoint scenario facts and template families, not disjoint vocabularies."""
    counts = config["scenario_counts"]
    scenarios = []
    used = set()
    for split in SPLITS:
        seed = hashlib.sha256(f"{config['seed_namespace']}|{split}".encode()).digest()
        rng = random.Random(int.from_bytes(seed, "big"))
        per_kind = {"tradeoff": 0, "dominance": 0}
        for index in range(counts[split]):
            kind = "dominance" if index % 4 == 3 else "tradeoff"
            while True:
                t_low, t_high = sorted(rng.sample(range(2, 100), 2))
                e_low, e_high = sorted(rng.sample(range(2, 100), 2))
                pair = ((Option(t_low, e_high), Option(t_high, e_low))
                        if kind == "tradeoff" else
                        (Option(t_low, e_low), Option(t_high, e_high)))
                key = tuple(sorted((o.time_minutes, o.energy_units) for o in pair))
                if key not in used:
                    used.add(key)
                    break
            kind_index = per_kind[kind]
            if kind_index % 2:
                pair = pair[::-1]
            scenario = Scenario(f"semgoal-v01-{split}-{index + 1:04d}", split,
                                (kind_index // 2) % 2, kind, pair)
            scenario.validate()
            scenarios.append(scenario)
            per_kind[kind] += 1
    return tuple(scenarios)


def validate_dataset(scenarios: tuple[Scenario, ...], config: Mapping[str, Any]) -> dict[str, Any]:
    ids, facts = set(), set()
    counts = {s: 0 for s in SPLITS}
    kinds = {s: {"tradeoff": 0, "dominance": 0} for s in SPLITS}
    balances = {s: {g: {"A": 0, "B": 0} for g in GOALS} for s in SPLITS}
    templates = {s: set() for s in SPLITS}
    for scenario in scenarios:
        scenario.validate()
        if scenario.scenario_id in ids or scenario.fact_key() in facts:
            raise ValueError("Duplicate scenario or cross-split scenario leakage")
        ids.add(scenario.scenario_id)
        facts.add(scenario.fact_key())
        counts[scenario.split] += 1
        kinds[scenario.split][scenario.kind] += 1
        templates[scenario.split].add(scenario.template_index)
        for order in ("AB", "BA"):
            task = render_task(scenario, order)
            for goal in GOALS:
                balances[scenario.split][goal][reference_answer(task, goal)] += 1
    if counts != dict(config["scenario_counts"]):
        raise ValueError("Split counts differ from the preview configuration")
    if any(kinds[s]["tradeoff"] != counts[s] * 3 // 4
           or templates[s] != {0, 1} for s in SPLITS):
        raise ValueError("Missing scenario kind or template coverage")
    if any(balances[s][g]["A"] != balances[s][g]["B"] for s in SPLITS for g in GOALS):
        raise ValueError("Answer labels are not balanced")
    all_templates = [x for values in TEMPLATES.values() for x in values]
    all_phrases = [x for values in GOAL_PHRASES.values() for x in values]
    if len(set(all_templates)) != len(all_templates) or len(set(all_phrases)) != len(all_phrases):
        raise ValueError("Template families or goal expressions overlap splits")
    return {"scenario_counts": counts, "kind_counts": kinds, "answer_balance": balances,
            "scenario_facts_disjoint": True, "template_families_disjoint": True,
            "grouping_unit": "scenario_including_both_goals_and_label_orders"}


@dataclass(frozen=True)
class GoalRequest:
    mode: str
    goal: str | None

    def __post_init__(self) -> None:
        if self.mode not in ("active", "zero", "mask", "random"):
            raise ValueError("Unknown request mode")
        if (self.mode == "active" and self.goal not in GOALS
                or self.mode != "active" and self.goal is not None):
            raise ValueError("Only active requests may contain a semantic goal")


def build_goal_state(goal: str, *, state_id: str) -> dict[str, Any]:
    if goal not in GOALS:
        raise ValueError("Unknown goal")
    item = {"field_item_id": "semantic-active-goal", "value": goal, "value_type": "string",
            "confidence": 1.0, "created_step": 0, "updated_step": 0,
            "source_evidence_ids": ["owner-set-offline-contract"], "status": "active",
            "update_class": "fast"}
    return build_self_state(state_id=state_id, agent_instance_id="semantic-goal-offline",
        trajectory_id="semantic-goal-contract-only", step=0,
        model_id="offline-contract-unbound", tokenizer_id="offline-contract-unbound",
        fields={"active_goals": [item]})


def request_from_state(state: Mapping[str, Any], mode: str = "active") -> GoalRequest:
    validated = validate_self_state(state)
    if any(validated[field] for field in SELF_FIELDS if field != "active_goals"):
        raise ValueError("This minimal contract only accepts the active goal field")
    items = validated["active_goals"]
    if not items and mode in ("zero", "mask", "random"):
        return GoalRequest(mode, None)
    if (len(items) != 1 or items[0]["status"] != "active"
            or items[0]["value_type"] != "string" or items[0]["value"] not in GOALS):
        raise ValueError("Exactly one recognized active goal is required")
    return GoalRequest(mode, items[0]["value"] if mode == "active" else None)


def clear_goal_state(state: Mapping[str, Any], *, state_id: str) -> dict[str, Any]:
    request_from_state(state)
    cleared = copy.deepcopy(dict(state))
    cleared["parent_state_id"] = state["state_id"]
    cleared["state_id"] = state_id
    cleared["active_goals"] = []
    cleared["provenance_refs"].append("offline-explicit-goal-clear")
    cleared["integrity"]["payload_sha256"] = self_state_digest(cleared)
    return validate_self_state(cleared)


def swap_goal_states(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request_from_state(left)
    request_from_state(right)
    return swap_self_fields(left, right, fields=("active_goals",),
        left_state_id=str(left["state_id"]) + "-swapped",
        right_state_id=str(right["state_id"]) + "-swapped")


def sample_size_review() -> dict[str, Any]:
    """Prospective normal approximations, not proof of bootstrap endpoint power."""
    z_sum = NormalDist().inv_cdf(0.99) + NormalDist().inv_cdf(0.80)
    assumptions = []
    # At scenario level, difference in average correctness is bounded [-1, 1].
    for sigma in (0.5, 1.0):
        for improvement in (0.1, 0.15, 0.2):
            assumptions.append({"paired_improvement": improvement, "cluster_sd": sigma,
                "approx_required_independent_scenarios": math.ceil((z_sum * sigma / improvement) ** 2)})
    return {"method": "normal_approximation_for_scenario_level_paired_mean",
            "one_sided_alpha": 0.01, "target_power": 0.8, "assumptions": assumptions,
            "limitations": "Does not verify cluster-bootstrap coverage, other necessary gates, or dominance subsets",
            "final_sample_size_frozen": False, "real_call_budget_frozen": False,
            "next_review": "Prospective simulation of the complete joint decision before preregistration"}


def write_offline_package(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    scenarios = generate_scenarios(config)
    validation = validate_dataset(scenarios, config)
    public, references = [], []
    for scenario in scenarios:
        for order in ("AB", "BA"):
            task = render_task(scenario, order)
            public.append(asdict(task))
            for goal in GOALS:
                references.append({"variant_id": task.variant_id, "scenario_id": task.scenario_id,
                                   "goal": goal, "reference_choice": reference_answer(task, goal)})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    state_store = SelfStore(output / "goal_states")
    energy = build_goal_state(GOALS[0], state_id="semantic-goal-energy")
    time = build_goal_state(GOALS[1], state_id="semantic-goal-time")
    before = sha256_json([energy, time])
    swapped = swap_goal_states(energy, time)
    state_store.save(energy)
    state_store.save(time)
    restored = state_store.load(energy["state_id"])
    cleared = clear_goal_state(energy, state_id="semantic-goal-cleared")
    state_store.save(cleared)
    interface_checks = {
        "save_restore_snapshot_equal": restored == energy,
        "save_restore_request_equal": request_from_state(restored) == request_from_state(energy),
        "swap_changes_both_goal_requests": request_from_state(swapped[0]).goal == GOALS[1]
            and request_from_state(swapped[1]).goal == GOALS[0],
        "zero_and_mask_remove_goal_from_request": request_from_state(energy, "zero").goal is None
            and request_from_state(energy, "mask").goal is None,
        "source_snapshots_unchanged": sha256_json([energy, time]) == before,
        "cleared_snapshot_stays_empty_after_restore": not state_store.load(cleared["state_id"])["active_goals"],
    }
    if not all(interface_checks.values()):
        raise AssertionError("Offline goal-state contract failed")
    payloads = {"public_tasks.json": {"version": VERSION, "tasks": public},
                "evaluation_references.json": {"version": VERSION, "references": references}}
    for name, payload in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
    report = {"version": VERSION, "status": "semantic_goal_offline_data_and_state_contract_verified",
        "valid": True, "config_digest_sha256": sha256_json(config), "dataset_validation": validation,
        "scenario_count": len(scenarios), "public_task_count": len(public),
        "evaluation_reference_count": len(references), "interface_checks": interface_checks,
        "payload_digests": {name: sha256_json(value) for name, value in payloads.items()},
        "sample_size_review": sample_size_review(), "model_executed": False,
        "tokenizer_used": False, "real_projection_constructed": False,
        "actual_model_forward_calls": 0, "model_accuracy_measured": False,
        "real_execution_authorized": False, "historical_rerun_authorized": False,
        "data_role": config["data_role"]}
    report["report_digest_sha256"] = sha256_json(report)
    with (output / "report.json").open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    return report
