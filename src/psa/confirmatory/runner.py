from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json


CONDITIONS = (
    "continuous",
    "restored",
    "reset",
    "random_matched",
    "swapped_I",
    "swapped_G",
    "swapped_both",
    "prompt_visible",
)
EXPECTED_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))
DEVELOPMENT_FIXTURE_KIND = "non_core_confirmatory_runner_fixture"


class TrialBackend(Protocol):
    def score(
        self,
        *,
        group: Mapping[str, Any],
        trial: Mapping[str, Any],
        condition_plan: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _combo(value: Any, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or tuple(value) not in EXPECTED_COMBOS
    ):
        raise ValueError(f"{label} must be one of the four I x G combos")
    return int(value[0]), int(value[1])


def condition_source_combo(
    target_combo: tuple[int, int],
    condition: str,
) -> tuple[int, int] | None:
    identity, goal = _combo(target_combo, label="target_combo")
    if condition in {"continuous", "restored", "prompt_visible"}:
        return identity, goal
    if condition in {"reset", "random_matched"}:
        return None
    if condition == "swapped_I":
        return 1 - identity, goal
    if condition == "swapped_G":
        return identity, 1 - goal
    if condition == "swapped_both":
        return 1 - identity, 1 - goal
    raise ValueError(f"unsupported confirmatory condition: {condition}")


def condition_evaluation_combo(
    target_combo: tuple[int, int],
    condition: str,
) -> tuple[int, int]:
    source = condition_source_combo(target_combo, condition)
    return target_combo if source is None else source


def _trial_combo(trial: Mapping[str, Any]) -> tuple[int, int]:
    target = trial.get("target_fields")
    if not isinstance(target, Mapping):
        raise ValueError("trial target_fields are missing")
    return _combo(
        (target.get("identity"), target.get("goal")),
        label="trial target combo",
    )


def _option_code_for_combo(
    trial: Mapping[str, Any],
    combo: tuple[int, int],
) -> str:
    mapping = trial.get("option_mapping")
    if not isinstance(mapping, list) or len(mapping) != 4:
        raise ValueError("trial option_mapping must contain four options")
    matches = [
        option.get("code")
        for option in mapping
        if isinstance(option, Mapping)
        and (option.get("identity"), option.get("goal")) == combo
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError("option_mapping does not uniquely cover evaluation combo")
    return matches[0]


def build_condition_plan(
    trial: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    target_combo = _trial_combo(trial)
    source_combo = condition_source_combo(target_combo, condition)
    evaluation_combo = condition_evaluation_combo(target_combo, condition)
    return {
        "condition": condition,
        "query_target_combo": list(target_combo),
        "state_source_combo": (
            list(source_combo) if source_combo is not None else None
        ),
        "evaluation_combo": list(evaluation_combo),
        "evaluation_option_code": _option_code_for_combo(
            trial,
            evaluation_combo,
        ),
        "semantic_rule": (
            "donor_state_combo"
            if condition.startswith("swapped_")
            else "query_target_combo"
        ),
    }


def _validate_group(group: Mapping[str, Any]) -> None:
    group_id = group.get("factorial_group_id")
    trials = group.get("trials")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("factorial_group_id is required")
    if not isinstance(trials, list) or len(trials) != 16:
        raise ValueError("runner fixture groups require exactly 16 trials")
    trial_ids = [
        trial.get("trial_id") if isinstance(trial, Mapping) else None
        for trial in trials
    ]
    if any(not isinstance(item, str) or not item for item in trial_ids):
        raise ValueError("every trial requires a trial_id")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("trial_id values must be unique within a group")
    observed = {_trial_combo(trial) for trial in trials}
    if observed != set(EXPECTED_COMBOS):
        raise ValueError("group trials must cover all four semantic combos")


def build_group_execution_plan(
    group: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_group(group)
    records = []
    for trial in group["trials"]:
        for condition in CONDITIONS:
            records.append(
                {
                    "trial_id": trial["trial_id"],
                    **build_condition_plan(trial, condition),
                }
            )
    return {
        "plan_version": "0.1",
        "factorial_group_id": group["factorial_group_id"],
        "trial_count": len(group["trials"]),
        "condition_count": len(CONDITIONS),
        "record_count": len(records),
        "conditions": list(CONDITIONS),
        "records": records,
        "plan_digest_sha256": sha256_json(records),
    }


def _validate_backend_result(result: Mapping[str, Any]) -> dict[str, Any]:
    scores = result.get("option_scores")
    if (
        not isinstance(scores, Mapping)
        or set(scores) != {"A", "B", "C", "D"}
        or not all(isinstance(value, (int, float)) for value in scores.values())
    ):
        raise ValueError("backend must return finite A-D option_scores")
    normalized = {code: float(scores[code]) for code in "ABCD"}
    if not all(value == value and abs(value) != float("inf") for value in normalized.values()):
        raise ValueError("backend option_scores must be finite")
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("backend metadata must be an object")
    return {
        "option_scores": normalized,
        "metadata": dict(metadata),
    }


def execute_group(
    group: Mapping[str, Any],
    backend: TrialBackend,
) -> dict[str, Any]:
    plan = build_group_execution_plan(group)
    trials_by_id = {trial["trial_id"]: trial for trial in group["trials"]}
    records = []
    start_group = getattr(backend, "start_group", None)
    end_group = getattr(backend, "end_group", None)
    group_metadata = getattr(backend, "group_metadata", None)
    try:
        if callable(start_group):
            start_group(group)
        for planned in plan["records"]:
            trial = trials_by_id[planned["trial_id"]]
            result = _validate_backend_result(
                backend.score(
                    group=group,
                    trial=trial,
                    condition_plan=planned,
                )
            )
            records.append({**planned, **result})
        backend_metadata = (
            dict(group_metadata()) if callable(group_metadata) else {}
        )
    finally:
        if callable(end_group):
            end_group(group)
    return {
        "group_result_version": "0.1",
        "factorial_group_id": group["factorial_group_id"],
        "plan_digest_sha256": plan["plan_digest_sha256"],
        "record_count": len(records),
        "records": records,
        "contains_derived_accuracy": False,
        "contains_interim_decision": False,
        "backend_group_metadata": backend_metadata,
        "group_result_digest_sha256": sha256_json(records),
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        data = canonical_json_bytes(dict(payload))
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fixture_digest(dataset: Mapping[str, Any]) -> str:
    unsigned = dict(dataset)
    unsigned.pop("fixture_digest_sha256", None)
    return sha256_json(unsigned)


def build_non_core_development_fixture() -> dict[str, Any]:
    """Create one deterministic group that shares no EXP-001 labels or prompts."""
    combos = EXPECTED_COMBOS
    codes = ("A", "B", "C", "D")
    identity_labels = ("amber", "cobalt")
    goal_labels = ("orbit", "prism")
    trials = []
    for rotation_index in range(4):
        rotated_codes = codes[rotation_index:] + codes[:rotation_index]
        option_mapping = [
            {
                "code": code,
                "identity": combo[0],
                "goal": combo[1],
                "domain": identity_labels[combo[0]],
                "operation": goal_labels[combo[1]],
            }
            for combo, code in zip(combos, rotated_codes, strict=True)
        ]
        rendered_options = "\n".join(
            f"{item['code']}. DOMAIN: {item['domain']} | "
            f"OPERATION: {item['operation']}"
            for item in option_mapping
        )
        for identity, goal in combos:
            trial = {
                "trial_id": (
                    f"devtrial-r{rotation_index}-i{identity}-g{goal}"
                ),
                "semantic_case_id": f"devcase-i{identity}-g{goal}",
                "rotation_index": rotation_index,
                "history_prompt": (
                    "User: DEVELOPMENT FIXTURE ONLY. STORE TWO TEST FIELDS.\n"
                    f"TEST DOMAIN: {identity_labels[identity]}\n"
                    f"TEST OPERATION: {goal_labels[goal]}\n\n"
                    "Assistant: Development test fields stored."
                ),
                "query_prompt": (
                    "User: DEVELOPMENT FIXTURE ONLY. Select the option matching "
                    "both stored test fields.\nOPTIONS:\n"
                    f"{rendered_options}\nReturn one code.\n\n"
                    "Assistant: <think></think"
                ),
                "target_fields": {
                    "identity": identity,
                    "goal": goal,
                    "domain": identity_labels[identity],
                    "operation": goal_labels[goal],
                },
                "option_mapping": option_mapping,
            }
            trials.append(trial)
    group = {
        "factorial_group_id": "devgrp-impl5b-noncore-v1",
        "group_index": 0,
        "trial_count": len(trials),
        "trials": trials,
    }
    fixture = {
        "fixture_version": "0.1",
        "fixture_kind": DEVELOPMENT_FIXTURE_KIND,
        "experiment_id": "DEV-IMPL5B-RUNNER",
        "purpose": "exercise runner mechanics without Core Set inference",
        "groups": [group],
    }
    fixture["fixture_digest_sha256"] = _fixture_digest(fixture)
    return fixture


def _validate_development_fixture(dataset: Mapping[str, Any]) -> None:
    if dataset.get("fixture_kind") != DEVELOPMENT_FIXTURE_KIND:
        raise ValueError("runner development requires an explicit non-Core fixture")
    if dataset.get("experiment_id") == "EXP-001":
        raise ValueError("EXP-001 Core Set cannot be used as a development fixture")
    groups = dataset.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("development fixture requires at least one group")
    group_ids = [
        group.get("factorial_group_id")
        if isinstance(group, Mapping)
        else None
        for group in groups
    ]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("factorial_group_id values must be unique")
    if any(
        isinstance(group, Mapping)
        and str(group.get("factorial_group_id", "")).startswith("coregrp-")
        for group in groups
    ):
        raise ValueError("frozen Core Set group IDs are forbidden in development")
    expected = dataset.get("fixture_digest_sha256")
    if not isinstance(expected, str) or expected != _fixture_digest(dataset):
        raise ValueError("development fixture digest is invalid")
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("fixture groups must be objects")
        _validate_group(group)


def _new_manifest(dataset: Mapping[str, Any]) -> dict[str, Any]:
    group_ids = [group["factorial_group_id"] for group in dataset["groups"]]
    return {
        "run_manifest_version": "0.1",
        "run_kind": "non_core_runner_development",
        "fixture_digest_sha256": dataset["fixture_digest_sha256"],
        "expected_group_ids": group_ids,
        "expected_group_count": len(group_ids),
        "expected_records_per_group": 16 * len(CONDITIONS),
        "status": "running",
        "completed_group_files": {},
        "formal_authorization_used": False,
        "confirmatory_experiment_run": False,
        "confirmatory_results_observed": False,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runner manifest must be an object")
    return payload


def _validate_existing_manifest(
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    expected_ids = [group["factorial_group_id"] for group in dataset["groups"]]
    if (
        manifest.get("run_kind") != "non_core_runner_development"
        or manifest.get("fixture_digest_sha256")
        != dataset.get("fixture_digest_sha256")
        or manifest.get("expected_group_ids") != expected_ids
        or manifest.get("formal_authorization_used") is not False
        or manifest.get("confirmatory_experiment_run") is not False
        or manifest.get("confirmatory_results_observed") is not False
    ):
        raise ValueError("existing runner manifest does not match fixture")


def run_development_fixture(
    *,
    dataset: Mapping[str, Any],
    backend: TrialBackend,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Exercise runner mechanics without accepting any EXP-001 Core data."""
    _validate_development_fixture(dataset)
    destination = Path(output_dir).resolve()
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        _validate_existing_manifest(manifest, dataset)
    else:
        manifest = _new_manifest(dataset)
        _atomic_write(manifest_path, manifest)

    completed = manifest.get("completed_group_files")
    if not isinstance(completed, dict):
        raise ValueError("completed_group_files must be an object")
    groups_by_id = {
        group["factorial_group_id"]: group for group in dataset["groups"]
    }
    for group_id, expected_digest in completed.items():
        group_path = destination / "groups" / f"{group_id}.json"
        if (
            group_id not in groups_by_id
            or not isinstance(expected_digest, str)
            or not group_path.is_file()
            or sha256_file(group_path) != expected_digest
        ):
            raise ValueError(f"completed group integrity failed: {group_id}")

    try:
        for group_id in manifest["expected_group_ids"]:
            if group_id in completed:
                continue
            result = execute_group(groups_by_id[group_id], backend)
            group_path = destination / "groups" / f"{group_id}.json"
            _atomic_write(group_path, result)
            completed[group_id] = sha256_file(group_path)
            manifest["completed_group_files"] = dict(completed)
            manifest["completed_group_count"] = len(completed)
            manifest["status"] = "running"
            _atomic_write(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "interrupted"
        manifest["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "failed_at_utc": _utc_now(),
        }
        _atomic_write(manifest_path, manifest)
        raise

    manifest.pop("failure", None)
    manifest["status"] = "development_fixture_complete"
    manifest["completed_group_count"] = len(completed)
    manifest["completed_at_utc"] = _utc_now()
    manifest["valid"] = len(completed) == manifest["expected_group_count"]
    _atomic_write(manifest_path, manifest)
    return dict(manifest)
