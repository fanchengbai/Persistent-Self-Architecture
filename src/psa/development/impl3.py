from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.evaluation import bca_mean_interval
from psa.model import RWKV7Adapter, clone_state, load_model_config
from psa.tasks import generate_dataset, generate_factorial_group
from psa.validation import validate_dataset


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json_bytes(record).decode("utf-8") for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _require_int(payload: dict[str, Any], field: str, minimum: int = 0) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_number(
    payload: dict[str, Any],
    field: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    value = payload.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return numeric


def _require_pairs(value: Any, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(label, str) and label for label in item)
            or item[0] == item[1]
        ):
            raise ValueError(f"{field}[{index}] must contain two distinct labels")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def inspect_label_pairs(
    adapter: Any,
    candidates: Sequence[Sequence[str]],
    *,
    selected_pair_count: int,
    max_tokens_per_form: int,
) -> dict[str, Any]:
    """Select label pairs using tokenizer properties only."""
    if selected_pair_count <= 0:
        raise ValueError("selected_pair_count must be positive")
    if max_tokens_per_form <= 0:
        raise ValueError("max_tokens_per_form must be positive")

    records: list[dict[str, Any]] = []
    selected: list[list[str]] = []
    for raw_pair in candidates:
        if len(raw_pair) != 2:
            raise ValueError("each label candidate must contain two labels")
        pair = (str(raw_pair[0]), str(raw_pair[1]))
        if not pair[0] or not pair[1] or pair[0] == pair[1]:
            raise ValueError("label candidates must be non-empty and distinct")

        forms = []
        eligible = True
        for form_name, prefix in (("bare", ""), ("leading_space", " ")):
            rendered = [f"{prefix}{label}" for label in pair]
            token_lists = [adapter.encode(text) for text in rendered]
            exact = [
                adapter.decode(tokens) == text
                for tokens, text in zip(token_lists, rendered, strict=True)
            ]
            counts = [len(tokens) for tokens in token_lists]
            form_valid = (
                all(exact)
                and len(set(counts)) == 1
                and max(counts) <= max_tokens_per_form
            )
            eligible = eligible and form_valid
            forms.append(
                {
                    "form": form_name,
                    "rendered": rendered,
                    "token_ids": token_lists,
                    "token_counts": counts,
                    "roundtrip_exact": exact,
                    "valid": form_valid,
                }
            )
        record = {
            "labels": list(pair),
            "forms": forms,
            "eligible": eligible,
        }
        records.append(record)
        if eligible and len(selected) < selected_pair_count:
            selected.append(list(pair))

    return {
        "candidate_count": len(records),
        "eligible_count": sum(int(record["eligible"]) for record in records),
        "selected_pair_count": len(selected),
        "required_pair_count": selected_pair_count,
        "max_tokens_per_form": max_tokens_per_form,
        "selection_rule": (
            "first declared pairs with exact bare/leading-space roundtrip, "
            "equal within-pair token counts, and bounded token length"
        ),
        "selected_pairs": selected,
        "pairs": records,
        "valid": len(selected) == selected_pair_count,
    }


def inspect_answer_codes(
    adapter: Any,
    answer_codes: Sequence[str],
    *,
    continuation_prefix: str,
    require_equal_token_count: bool,
) -> dict[str, Any]:
    if len(answer_codes) != 4 or len(set(answer_codes)) != 4:
        raise ValueError("answer_codes must contain four distinct values")
    records = []
    token_sequences = []
    for code in answer_codes:
        rendered = f"{continuation_prefix}{code}"
        tokens = adapter.encode(rendered)
        token_sequences.append(tuple(tokens))
        records.append(
            {
                "code": code,
                "rendered": rendered,
                "token_ids": tokens,
                "token_count": len(tokens),
                "roundtrip_exact": adapter.decode(tokens) == rendered,
            }
        )
    counts = [record["token_count"] for record in records]
    valid = bool(
        all(record["roundtrip_exact"] for record in records)
        and len(set(token_sequences)) == 4
        and (not require_equal_token_count or len(set(counts)) == 1)
    )
    return {
        "continuation_prefix": continuation_prefix,
        "require_equal_token_count": require_equal_token_count,
        "codes": records,
        "valid": valid,
    }


def calibrate_standard_delay(
    adapter: Any,
    *,
    track: str,
    target_token_count: int,
    max_absolute_error: int,
    max_units: int,
) -> dict[str, Any]:
    """Choose delay by token distance only, before observing task behavior."""
    if target_token_count <= 0:
        raise ValueError("target_token_count must be positive")
    if max_absolute_error < 0:
        raise ValueError("max_absolute_error must be non-negative")
    if max_units <= 0:
        raise ValueError("max_units must be positive")

    candidates = []
    for delay_units in range(1, max_units + 1):
        group = generate_factorial_group(
            group_seed=0,
            track=track,
            delay_units=delay_units,
        )
        suffix = group.trajectories[0].common_suffix
        tokens = adapter.encode(suffix)
        candidates.append(
            {
                "delay_units": delay_units,
                "token_count": len(tokens),
                "absolute_error": abs(len(tokens) - target_token_count),
                "roundtrip_exact": adapter.decode(tokens) == suffix,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["absolute_error"],
            item["delay_units"],
        ),
    )
    return {
        "track": track,
        "target_token_count": target_token_count,
        "max_absolute_error": max_absolute_error,
        "selection_rule": (
            "minimum absolute tokenizer distance error; lower delay_units breaks ties"
        ),
        "selected_delay_units": selected["delay_units"],
        "selected_token_count": selected["token_count"],
        "selected_absolute_error": selected["absolute_error"],
        "candidates": candidates,
        "valid": bool(
            selected["roundtrip_exact"]
            and selected["absolute_error"] <= max_absolute_error
        ),
    }


def _score_continuations(
    adapter: Any,
    prompt: str,
    rendered_answers: dict[str, str],
) -> tuple[dict[str, float], Any, Any, int]:
    prompt_tokens = adapter.encode(prompt)
    prompt_logits, prompt_state = adapter.forward(prompt_tokens, None)
    scores: dict[str, float] = {}
    torch = adapter.torch
    for code, rendered in rendered_answers.items():
        answer_tokens = adapter.encode(rendered)
        logits = prompt_logits
        state = clone_state(prompt_state)
        score = 0.0
        for index, token in enumerate(answer_tokens):
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            score += float(log_probabilities[token].item())
            if index + 1 < len(answer_tokens):
                logits, state = adapter.forward([token], state)
        scores[code] = score
    return scores, prompt_logits, prompt_state, len(prompt_tokens)


def _normalized_probabilities(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    denominator = sum(weights.values())
    return {key: value / denominator for key, value in weights.items()}


def _greedy_format_probe(
    adapter: Any,
    logits: Any,
    state: Any,
    *,
    answer_codes: Sequence[str],
    max_tokens: int,
) -> dict[str, Any]:
    generated: list[int] = []
    current_logits = logits
    current_state = clone_state(state)
    for _ in range(max_tokens):
        token = int(adapter.torch.argmax(current_logits).item())
        generated.append(token)
        text = adapter.decode(generated)
        stripped = text.strip()
        if stripped in answer_codes:
            return {
                "generated_token_ids": generated,
                "generated_text": text,
                "generated_choice": stripped,
                "format_valid": True,
            }
        current_logits, current_state = adapter.forward([token], current_state)
    return {
        "generated_token_ids": generated,
        "generated_text": adapter.decode(generated),
        "generated_choice": None,
        "format_valid": False,
    }


def _render_prompt_visible(sample: Any) -> str:
    parts = []
    if sample.common_suffix:
        parts.append(sample.common_suffix)
    # T0 measures task comprehension, not persistence: the active bindings are
    # repeated immediately before the query. Delay material remains in the
    # prompt only as a tokenizer/resource calibration load.
    parts.append(sample.history)
    parts.append(sample.query)
    return "\n".join(parts)


def inspect_dataset_tokenization(
    adapter: Any,
    groups: Sequence[Any],
) -> dict[str, Any]:
    group_records = []
    for group in groups:
        history_counts = []
        prompt_counts = []
        roundtrip_exact = True
        for sample in group.trajectories:
            prompt = _render_prompt_visible(sample)
            for text in (sample.history, sample.common_suffix, sample.query, prompt):
                if text:
                    tokens = adapter.encode(text)
                    roundtrip_exact = (
                        roundtrip_exact and adapter.decode(tokens) == text
                    )
            history_counts.append(len(adapter.encode(sample.history)))
            prompt_counts.append(len(adapter.encode(prompt)))
        history_balanced = len(set(history_counts)) == 1
        prompt_balanced = len(set(prompt_counts)) == 1
        group_records.append(
            {
                "group_id": group.group_id,
                "history_token_counts": history_counts,
                "prompt_token_counts": prompt_counts,
                "history_token_balanced": history_balanced,
                "prompt_token_balanced": prompt_balanced,
                "roundtrip_exact": roundtrip_exact,
                "valid": bool(
                    history_balanced and prompt_balanced and roundtrip_exact
                ),
            }
        )
    return {
        "group_count": len(group_records),
        "groups": group_records,
        "valid": bool(group_records)
        and all(record["valid"] for record in group_records),
    }


def evaluate_prompt_visible(
    groups: Sequence[Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    samples = {
        sample.sample_id: (group, sample)
        for group in groups
        for sample in group.trajectories
    }
    outcomes = []
    failures = 0
    for record in records:
        if record.get("status") != "success":
            failures += 1
            continue
        group, sample = samples[record["sample_id"]]
        code_to_combo = {option.code: option.combo for option in group.options}
        predicted_code = record["argmax_choice"]
        predicted = code_to_combo[predicted_code]
        outcomes.append(
            {
                "sample_id": sample.sample_id,
                "group_id": group.group_id,
                "correct_code": sample.correct_code,
                "predicted_code": predicted_code,
                "identity_correct": predicted[0] == sample.identity,
                "goal_correct": predicted[1] == sample.goal,
                "joint_correct": predicted == (sample.identity, sample.goal),
                "format_valid": bool(record["format_valid"]),
            }
        )

    group_metrics = []
    for group in groups:
        group_outcomes = [
            item for item in outcomes if item["group_id"] == group.group_id
        ]
        if len(group_outcomes) != len(group.trajectories):
            continue
        group_metrics.append(
            {
                "group_id": group.group_id,
                "identity_accuracy": sum(
                    item["identity_correct"] for item in group_outcomes
                )
                / len(group_outcomes),
                "goal_accuracy": sum(
                    item["goal_correct"] for item in group_outcomes
                )
                / len(group_outcomes),
                "joint_accuracy": sum(
                    item["joint_correct"] for item in group_outcomes
                )
                / len(group_outcomes),
            }
        )

    def interval(field: str) -> tuple[float | None, float | None]:
        values = [float(group[field]) for group in group_metrics]
        if len(values) < 2:
            return None, None
        return bca_mean_interval(
            values,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )

    identity_interval = interval("identity_accuracy")
    goal_interval = interval("goal_accuracy")
    joint_interval = interval("joint_accuracy")
    denominator = len(records)
    format_valid_rate = (
        sum(item["format_valid"] for item in outcomes) / denominator
        if denominator
        else 0.0
    )
    infrastructure_failure_rate = failures / denominator if denominator else 1.0

    position_accuracy = {}
    answer_codes = sorted(
        {sample.correct_code for _, sample in samples.values()}
    )
    for code in answer_codes:
        items = [item for item in outcomes if item["correct_code"] == code]
        position_accuracy[code] = (
            sum(item["joint_correct"] for item in items) / len(items)
            if items
            else 0.0
        )
    position_gap = (
        max(position_accuracy.values()) - min(position_accuracy.values())
        if position_accuracy
        else 1.0
    )

    def average(field: str) -> float:
        return (
            sum(float(group[field]) for group in group_metrics) / len(group_metrics)
            if group_metrics
            else 0.0
        )

    checks = {
        "joint_lower_bound": bool(
            joint_interval[0] is not None
            and joint_interval[0] >= thresholds["joint_lower_bound"]
        ),
        "identity_lower_bound": bool(
            identity_interval[0] is not None
            and identity_interval[0] >= thresholds["marginal_lower_bound"]
        ),
        "goal_lower_bound": bool(
            goal_interval[0] is not None
            and goal_interval[0] >= thresholds["marginal_lower_bound"]
        ),
        "format_valid_rate": (
            format_valid_rate >= thresholds["format_valid_rate"]
        ),
        "answer_position_balance": (
            position_gap <= thresholds["max_answer_position_accuracy_gap"]
        ),
        "infrastructure_failure_rate": (
            infrastructure_failure_rate
            <= thresholds["max_infrastructure_failure_rate"]
        ),
    }
    return {
        "report_version": "0.1",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "scoring_method": "answer_sequence_log_likelihood",
        "cluster_unit": "factorial_group",
        "bootstrap": {
            "method": "BCa mean interval",
            "confidence": 0.95,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "planned_sample_count": len(samples),
        "successful_sample_count": len(outcomes),
        "failed_sample_count": failures,
        "complete_group_count": len(group_metrics),
        "metrics": {
            "identity_accuracy": average("identity_accuracy"),
            "identity_accuracy_interval": list(identity_interval),
            "goal_accuracy": average("goal_accuracy"),
            "goal_accuracy_interval": list(goal_interval),
            "joint_accuracy": average("joint_accuracy"),
            "joint_accuracy_interval": list(joint_interval),
            "format_valid_rate": format_valid_rate,
            "infrastructure_failure_rate": infrastructure_failure_rate,
            "answer_position_accuracy": position_accuracy,
            "max_answer_position_accuracy_gap": position_gap,
        },
        "thresholds": thresholds,
        "checks": checks,
        "group_metrics": group_metrics,
        "valid": all(checks.values()),
    }


def _verify_batch0(
    root: Path,
    prerequisites: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    records = []
    for item in prerequisites:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Batch 0 evidence paths must stay inside project root")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Batch 0 evidence is missing: {path}")
        payload = _load_object(path, "Batch 0 evidence")
        expected_gate = str(item["gate"])
        valid = payload.get("gate") == expected_gate and payload.get("valid") is True
        records.append(
            {
                "gate": expected_gate,
                "path": str(path),
                "sha256": sha256_file(path),
                "reported_gate": payload.get("gate"),
                "reported_valid": payload.get("valid"),
                "valid": valid,
                "summary": payload,
            }
        )
    return {
        "evidence_count": len(records),
        "evidence": records,
        "valid": bool(records) and all(record["valid"] for record in records),
    }


def _resource_estimate(
    *,
    output_dir: Path,
    batch0: dict[str, Any],
    record_count: int,
    total_forward_tokens: int,
    model_load_seconds: float,
    batch_seconds: float,
    peak_memory_bytes: int,
    planned_core_groups: int,
    planned_conditions_per_trajectory: int,
) -> dict[str, Any]:
    planned_trials = planned_core_groups * 4 * planned_conditions_per_trajectory
    per_trial_seconds = batch_seconds / record_count if record_count else 0.0
    per_trial_tokens = total_forward_tokens / record_count if record_count else 0.0
    checkpoint_bytes = 0
    for evidence in batch0["evidence"]:
        value = evidence["summary"].get("checkpoint_payload_size_bytes")
        if isinstance(value, int):
            checkpoint_bytes = max(checkpoint_bytes, value)
    raw_bytes = sum(
        path.stat().st_size for path in output_dir.glob("*") if path.is_file()
    )
    raw_bytes_per_trial = raw_bytes / record_count if record_count else 0.0
    planned_checkpoint_count = planned_core_groups * 4
    estimated_result_bytes = int(math.ceil(raw_bytes_per_trial * planned_trials))
    estimated_checkpoint_bytes = checkpoint_bytes * planned_checkpoint_count
    return {
        "report_version": "0.1",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "dry_run": {
            "trajectory_count": record_count,
            "forward_token_count": total_forward_tokens,
            "mean_forward_tokens_per_trajectory": per_trial_tokens,
            "model_load_seconds": model_load_seconds,
            "batch_seconds": batch_seconds,
            "mean_seconds_per_trajectory": per_trial_seconds,
            "cuda_peak_memory_bytes": peak_memory_bytes,
            "output_bytes_before_resource_report": raw_bytes,
            "native_state_bytes": checkpoint_bytes,
        },
        "core_projection": {
            "group_count": planned_core_groups,
            "trajectories_per_group": 4,
            "conditions_per_trajectory": planned_conditions_per_trajectory,
            "trial_count": planned_trials,
            "estimated_forward_token_count": int(
                math.ceil(per_trial_tokens * planned_trials)
            ),
            "estimated_gpu_seconds": per_trial_seconds * planned_trials,
            "estimated_gpu_hours": per_trial_seconds * planned_trials / 3600.0,
            "estimated_raw_result_bytes": estimated_result_bytes,
            "planned_checkpoint_count": planned_checkpoint_count,
            "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
            "estimated_total_disk_bytes": (
                estimated_result_bytes + estimated_checkpoint_bytes
            ),
        },
        "projection_note": (
            "Development Prompt-visible throughput extrapolation; state-condition "
            "runs may differ and Impl-4 must freeze the final budget."
        ),
        "valid": bool(
            record_count > 0
            and total_forward_tokens > 0
            and batch_seconds > 0.0
            and peak_memory_bytes > 0
            and checkpoint_bytes > 0
        ),
    }


def run_impl3_development_gate(
    *,
    config_path: str | Path,
    gate_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    gate_path = Path(gate_config_path).resolve()
    gate_config = _load_object(gate_path, "Impl-3 gate config")
    if gate_config.get("gate") != "impl3_development":
        raise ValueError("gate config is not for impl3_development")

    batch0 = _verify_batch0(root, gate_config.get("batch0_evidence", []))
    _write_json(destination / "batch0_evidence.json", batch0)
    if not batch0["valid"]:
        raise RuntimeError("Batch 0 evidence is incomplete or invalid")

    label_config = gate_config.get("label_pool")
    delay_config = gate_config.get("standard_delay")
    batch1_config = gate_config.get("batch1")
    thresholds_config = gate_config.get("thresholds")
    projection_config = gate_config.get("resource_projection")
    if not all(
        isinstance(item, dict)
        for item in (
            label_config,
            delay_config,
            batch1_config,
            thresholds_config,
            projection_config,
        )
    ):
        raise ValueError("Impl-3 config sections must be objects")

    model_config = load_model_config(config_path, root, verify_files=True)
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(model_config)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    pair_count = _require_int(label_config, "selected_pair_count", 1)
    max_label_tokens = _require_int(label_config, "max_tokens_per_form", 1)
    identity_candidates = _require_pairs(
        label_config.get("identity_candidates"),
        "label_pool.identity_candidates",
    )
    goal_candidates = _require_pairs(
        label_config.get("goal_candidates"),
        "label_pool.goal_candidates",
    )
    identity_report = inspect_label_pairs(
        adapter,
        identity_candidates,
        selected_pair_count=pair_count,
        max_tokens_per_form=max_label_tokens,
    )
    goal_report = inspect_label_pairs(
        adapter,
        goal_candidates,
        selected_pair_count=pair_count,
        max_tokens_per_form=max_label_tokens,
    )
    answer_codes = tuple(str(code) for code in batch1_config.get("answer_codes", []))
    answer_report = inspect_answer_codes(
        adapter,
        answer_codes,
        continuation_prefix=str(batch1_config.get("answer_continuation_prefix", "")),
        require_equal_token_count=bool(
            label_config.get("require_equal_answer_token_count", True)
        ),
    )
    selected_labels = {
        "report_version": "0.1",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "selection_uses_behavior": False,
        "identity": identity_report,
        "goal": goal_report,
        "answers": answer_report,
        "valid": bool(
            identity_report["valid"]
            and goal_report["valid"]
            and answer_report["valid"]
        ),
    }
    _write_json(destination / "label_pool_report.json", selected_labels)
    if not selected_labels["valid"]:
        raise RuntimeError("tokenizer label-pool qualification failed")

    track = str(batch1_config.get("track", "synthetic"))
    delay_report = calibrate_standard_delay(
        adapter,
        track=track,
        target_token_count=_require_int(
            delay_config, "target_token_count", 1
        ),
        max_absolute_error=_require_int(
            delay_config, "max_absolute_error", 0
        ),
        max_units=_require_int(delay_config, "max_units", 1),
    )
    delay_report.update(
        {
            "report_version": "0.1",
            "created_at_utc": _utc_now(),
            "development_only": True,
            "selection_uses_behavior": False,
        }
    )
    _write_json(destination / "delay_calibration.json", delay_report)
    if not delay_report["valid"]:
        raise RuntimeError("standard delay token calibration failed")

    groups = generate_dataset(
        group_count=_require_int(batch1_config, "group_count", 2),
        base_seed=_require_int(batch1_config, "base_seed", 0),
        track=track,
        identity_label_pairs=identity_report["selected_pairs"],
        goal_label_pairs=goal_report["selected_pairs"],
        answer_codes=answer_codes,
        delay_units=delay_report["selected_delay_units"],
        generator_version=str(batch1_config.get("generator_version", "0.1")),
    )
    validation = validate_dataset(groups)
    tokenizer_validation = inspect_dataset_tokenization(adapter, groups)
    group_payloads = [group.to_dict() for group in groups]
    dataset = {
        "dataset_version": "0.1",
        "development_only": True,
        "track": track,
        "group_count": len(groups),
        "base_seed": batch1_config["base_seed"],
        "generator_version": str(batch1_config.get("generator_version", "0.1")),
        "delay_units": delay_report["selected_delay_units"],
        "delay_token_count": delay_report["selected_token_count"],
        "label_pool_report_sha256": sha256_file(
            destination / "label_pool_report.json"
        ),
        "groups": group_payloads,
        "validation": validation.to_dict(),
        "dataset_digest_sha256": sha256_json(group_payloads),
    }
    _write_json(destination / "development_dataset.json", dataset)
    _write_json(
        destination / "task_validation.json",
        {
            "report_version": "0.1",
            "created_at_utc": _utc_now(),
            "development_only": True,
            "dataset_digest_sha256": dataset["dataset_digest_sha256"],
            "tokenizer": tokenizer_validation,
            **validation.to_dict(),
            "valid": bool(validation.valid and tokenizer_validation["valid"]),
        },
    )
    if not validation.valid or not tokenizer_validation["valid"]:
        raise RuntimeError(
            "development task leakage or tokenizer-balance validation failed"
        )

    continuation_prefix = str(batch1_config.get("answer_continuation_prefix", ""))
    rendered_answers = {
        code: f"{continuation_prefix}{code}" for code in answer_codes
    }
    max_generation_tokens = _require_int(
        batch1_config, "max_generation_tokens", 1
    )
    records: list[dict[str, Any]] = []
    total_forward_tokens = 0
    run_started = time.perf_counter()
    for group in groups:
        for sample in group.trajectories:
            prompt = _render_prompt_visible(sample)
            trial_started = time.perf_counter()
            base_record = {
                "record_version": "0.1",
                "experiment_id": "EXP-001",
                "batch_id": "batch1_prompt_visible_development",
                "run_id": "run-" + sha256_json(
                    {
                        "sample_id": sample.sample_id,
                        "gate_config_sha256": sha256_file(gate_path),
                    }
                )[:24],
                "factorial_group_id": group.group_id,
                "trajectory_id": sample.trajectory_id,
                "sample_id": sample.sample_id,
                "condition": "prompt_visible",
                "state_checkpoint_id": "none",
                "query_digest_sha256": sha256_json(prompt),
            }
            try:
                scores, logits, state, prompt_token_count = _score_continuations(
                    adapter,
                    prompt,
                    rendered_answers,
                )
                format_probe = _greedy_format_probe(
                    adapter,
                    logits,
                    state,
                    answer_codes=answer_codes,
                    max_tokens=max_generation_tokens,
                )
                predicted_code = max(scores, key=scores.__getitem__)
                answer_token_count = sum(
                    len(adapter.encode(value)) for value in rendered_answers.values()
                )
                generation_token_count = len(
                    format_probe["generated_token_ids"]
                )
                forward_tokens = (
                    prompt_token_count
                    + answer_token_count
                    + generation_token_count
                )
                total_forward_tokens += forward_tokens
                records.append(
                    {
                        **base_record,
                        "prompt_token_count": prompt_token_count,
                        "option_scores": scores,
                        "option_probabilities": _normalized_probabilities(scores),
                        "argmax_choice": predicted_code,
                        **format_probe,
                        "timing": {
                            "seconds": time.perf_counter() - trial_started,
                            "accounted_forward_tokens": forward_tokens,
                        },
                        "runtime": {
                            "model_id": model_config.model_id,
                            "strategy": model_config.strategy,
                        },
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        **base_record,
                        "prompt_token_count": None,
                        "option_scores": {},
                        "option_probabilities": {},
                        "argmax_choice": None,
                        "generated_token_ids": [],
                        "generated_text": "",
                        "generated_choice": None,
                        "format_valid": False,
                        "timing": {
                            "seconds": time.perf_counter() - trial_started,
                            "accounted_forward_tokens": 0,
                        },
                        "runtime": {
                            "model_id": model_config.model_id,
                            "strategy": model_config.strategy,
                        },
                        "status": "failed",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - run_started
    _write_jsonl(destination / "raw_prompt_visible.jsonl", records)

    thresholds = {
        "joint_lower_bound": _require_number(
            thresholds_config, "joint_lower_bound", 0.0, 1.0
        ),
        "marginal_lower_bound": _require_number(
            thresholds_config, "marginal_lower_bound", 0.0, 1.0
        ),
        "format_valid_rate": _require_number(
            thresholds_config, "format_valid_rate", 0.0, 1.0
        ),
        "max_answer_position_accuracy_gap": _require_number(
            thresholds_config,
            "max_answer_position_accuracy_gap",
            0.0,
            1.0,
        ),
        "max_infrastructure_failure_rate": _require_number(
            thresholds_config,
            "max_infrastructure_failure_rate",
            0.0,
            1.0,
        ),
    }
    capability = evaluate_prompt_visible(
        groups,
        records,
        bootstrap_replicates=_require_int(
            batch1_config, "bootstrap_replicates", 100
        ),
        bootstrap_seed=_require_int(batch1_config, "bootstrap_seed", 0),
        thresholds=thresholds,
    )
    _write_json(destination / "prompt_visible_report.json", capability)

    resource = _resource_estimate(
        output_dir=destination,
        batch0=batch0,
        record_count=len(records),
        total_forward_tokens=total_forward_tokens,
        model_load_seconds=model_load_seconds,
        batch_seconds=batch_seconds,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        planned_core_groups=_require_int(
            projection_config, "planned_core_groups", 1
        ),
        planned_conditions_per_trajectory=_require_int(
            projection_config, "conditions_per_trajectory", 1
        ),
    )
    _write_json(destination / "resource_estimate.json", resource)

    valid = bool(
        batch0["valid"]
        and selected_labels["valid"]
        and delay_report["valid"]
        and validation.valid
        and tokenizer_validation["valid"]
        and capability["valid"]
        and resource["valid"]
    )
    summary = {
        "gate": "impl3_development",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "development_only": True,
        "model_id": model_config.model_id,
        "gate_config_sha256": sha256_file(gate_path),
        "batch0_valid": batch0["valid"],
        "task_leakage_valid": validation.valid,
        "task_tokenization_valid": tokenizer_validation["valid"],
        "label_pool_valid": selected_labels["valid"],
        "selected_identity_pairs": identity_report["selected_pairs"],
        "selected_goal_pairs": goal_report["selected_pairs"],
        "answer_codes_valid": answer_report["valid"],
        "standard_delay_units": delay_report["selected_delay_units"],
        "standard_delay_token_count": delay_report["selected_token_count"],
        "prompt_visible_valid": capability["valid"],
        "prompt_visible_metrics": capability["metrics"],
        "resource_estimate_valid": resource["valid"],
        "cuda_peak_memory_bytes": resource["dry_run"]["cuda_peak_memory_bytes"],
        "decision": "go" if valid else "revise",
        "reports": [
            "batch0_evidence.json",
            "label_pool_report.json",
            "delay_calibration.json",
            "development_dataset.json",
            "task_validation.json",
            "raw_prompt_visible.jsonl",
            "prompt_visible_report.json",
            "resource_estimate.json",
        ],
        "valid": valid,
    }
    _write_json(destination / "summary.json", summary)
    return summary
