from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
import itertools
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.development.capability_ladder import evaluate_g1_code_rotation
from psa.development.impl3 import (
    inspect_answer_codes,
    normalized_probabilities,
    write_jsonl,
)
from psa.model import RWKV7Adapter, clone_state, load_model_config
from psa.model.rwkv7 import compare_states
from psa.tasks import generate_factorial_group


SUPPORTED_HISTORY_MODES = (
    "single_statement",
    "statement_plus_verification",
    "repeated_consistent",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _load_object(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _require_codes(value: Any) -> tuple[str, str, str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(code, str) and code for code in value)
        or len(set(value)) != 4
    ):
        raise ValueError("answer_codes must contain four distinct strings")
    return value[0], value[1], value[2], value[3]


def _require_label_pairs(
    value: Any,
    field: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    pairs = []
    for index, pair in enumerate(value):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(label, str) and label for label in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError(
                f"{field}[{index}] must contain two distinct labels"
            )
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _render_delay(delay_units: int) -> str:
    if delay_units < 0:
        raise ValueError("delay_units must be non-negative")
    if delay_units == 0:
        return ""
    return "\n".join(
        "NEUTRAL-FILLER: unrelated system record."
        for _ in range(delay_units)
    )


def render_history_binding(
    mode: str,
    *,
    domain: str,
    operation: str,
    delay_units: int,
) -> str:
    if mode not in SUPPORTED_HISTORY_MODES:
        raise ValueError(f"unsupported history binding mode: {mode}")
    declaration = (
        "User: STATE BINDING.\n"
        "Store the following current state until explicitly replaced.\n"
        f"CURRENT DOMAIN: {domain}\n"
        f"CURRENT OPERATION: {operation}\n\n"
        "Assistant: State recorded."
    )
    if mode == "single_statement":
        dialogue = declaration
    elif mode == "statement_plus_verification":
        dialogue = (
            f"{declaration}\n\n"
            "User: Verify the saved state by repeating both fields.\n\n"
            "Assistant: "
            f"CURRENT DOMAIN: {domain}\n"
            f"CURRENT OPERATION: {operation}"
        )
    else:
        dialogue = (
            f"{declaration}\n\n"
            "User: CONSISTENT STATE BINDING.\n"
            "Confirm the same current state remains active.\n"
            f"CURRENT DOMAIN: {domain}\n"
            f"CURRENT OPERATION: {operation}\n\n"
            "Assistant: The same state remains active."
        )
    delay = _render_delay(delay_units)
    if not delay:
        return dialogue
    return (
        f"{dialogue}\n\n"
        "User: Review these neutral records without changing the saved state.\n"
        f"{delay}\n\n"
        "Assistant: Neutral records reviewed; saved state unchanged."
    )


def render_state_only_query(
    options: Sequence[dict[str, str]],
    *,
    assistant_prefix: str,
) -> str:
    if assistant_prefix != "<think></think":
        raise ValueError("history binding gate requires fake-think prefix")
    if len(options) != 4:
        raise ValueError("state-only query requires four options")
    option_text = "\n".join(
        (
            f"{option['code']}. DOMAIN: {option['domain']} | "
            f"OPERATION: {option['operation']}"
        )
        for option in options
    )
    user_content = (
        "TASK: state-only exact two-field match.\n"
        "GENERAL RULE: choose the unique option matching both the saved "
        "CURRENT DOMAIN and saved CURRENT OPERATION.\n"
        "The saved values are intentionally not repeated outside the "
        "balanced options below.\n"
        "OPTIONS:\n"
        f"{option_text}\n"
        "Reply with the matching option code only."
    )
    return (
        f"\n\nUser: {user_content}\n\n"
        f"Assistant: {assistant_prefix}"
    )


def generate_history_binding_manifest(
    *,
    answer_codes: Sequence[str],
    identity_label_pairs: Sequence[Sequence[str]],
    goal_label_pairs: Sequence[Sequence[str]],
    history_modes: Sequence[str],
    repetitions: int,
    base_seed: int,
    delay_units: int,
    assistant_prefix: str,
) -> dict[str, Any]:
    codes = tuple(answer_codes)
    if len(codes) != 4 or len(set(codes)) != 4:
        raise ValueError("answer_codes must contain four distinct values")
    modes = tuple(history_modes)
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(mode not in SUPPORTED_HISTORY_MODES for mode in modes)
    ):
        raise ValueError("history_modes are missing, duplicated, or unsupported")
    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if delay_units < 0:
        raise ValueError("delay_units must be non-negative")
    pair_combinations = tuple(
        itertools.product(identity_label_pairs, goal_label_pairs)
    )
    if not pair_combinations:
        raise ValueError("history binding label pairs must be non-empty")

    trials = []
    rng = random.Random(base_seed ^ 0x48495354)
    for repetition in range(repetitions):
        block_id = f"block-{repetition:03d}"
        group_seed = rng.getrandbits(63)
        identity_labels, goal_labels = pair_combinations[
            repetition % len(pair_combinations)
        ]
        for rotation_index in range(4):
            rotated_codes = (
                codes[rotation_index:] + codes[:rotation_index]
            )
            group = generate_factorial_group(
                group_seed=group_seed,
                track="synthetic",
                identity_labels=identity_labels,
                goal_labels=goal_labels,
                answer_codes=rotated_codes,
                delay_units=0,
                generator_version="g1-history-binding-v0.1",
                history_order="I_G",
            )
            for sample in group.trajectories:
                domain = group.identity_labels[sample.identity]
                operation = group.goal_labels[sample.goal]
                semantic_case_id = "bindcase-" + sha256_json(
                    {
                        "block_id": block_id,
                        "group_seed": group_seed,
                        "identity": sample.identity,
                        "goal": sample.goal,
                        "identity_labels": list(identity_labels),
                        "goal_labels": list(goal_labels),
                    }
                )[:24]
                option_mapping = [
                    {
                        "code": option.code,
                        "domain": group.identity_labels[option.identity],
                        "operation": group.goal_labels[option.goal],
                    }
                    for option in group.options
                ]
                query_text = render_state_only_query(
                    option_mapping,
                    assistant_prefix=assistant_prefix,
                )
                for mode in modes:
                    history_text = render_history_binding(
                        mode,
                        domain=domain,
                        operation=operation,
                        delay_units=delay_units,
                    )
                    sample_id = "g1bind-" + sha256_json(
                        {
                            "mode": mode,
                            "semantic_case_id": semantic_case_id,
                            "rotation_index": rotation_index,
                            "target_code": sample.correct_code,
                            "history": history_text,
                            "query": query_text,
                        }
                    )[:24]
                    trials.append(
                        {
                            "sample_id": sample_id,
                            "semantic_case_id": semantic_case_id,
                            "history_key": f"{mode}:{semantic_case_id}",
                            "task_level": "state_only_history_binding",
                            "history_mode": mode,
                            "block_id": block_id,
                            "rotation_index": rotation_index,
                            "rotated_answer_codes": list(rotated_codes),
                            "history_text": history_text,
                            "history_digest_sha256": sha256_json(history_text),
                            "query_text": query_text,
                            "query_digest_sha256": sha256_json(query_text),
                            "target_code": sample.correct_code,
                            "target_fields": {
                                "domain": domain,
                                "operation": operation,
                            },
                            "option_mapping": option_mapping,
                        }
                    )

    semantic_case_count_per_mode = repetitions * 4
    return {
        "manifest_version": "0.1",
        "development_only": True,
        "diagnostic": "history_binding_protocol_comparison",
        "prompt_format": "rwkv7-g1-state-only-fake-think-v0.1",
        "general_rule_visible": True,
        "current_state_values_visible_outside_balanced_options": False,
        "assistant_prefix": assistant_prefix,
        "base_seed": base_seed,
        "repetitions": repetitions,
        "delay_units": delay_units,
        "rotation_count": 4,
        "answer_codes": list(codes),
        "identity_label_pairs": [
            list(pair) for pair in identity_label_pairs
        ],
        "goal_label_pairs": [list(pair) for pair in goal_label_pairs],
        "history_modes": list(modes),
        "semantic_case_count_per_mode": semantic_case_count_per_mode,
        "trial_count_per_mode": semantic_case_count_per_mode * 4,
        "trial_count": len(trials),
        "trials": trials,
        "manifest_digest_sha256": sha256_json(trials),
    }


def _score_from_state(
    adapter: Any,
    *,
    query_text: str,
    source_state: Any,
    rendered_answers: dict[str, str],
    forced_prefix: str,
) -> tuple[dict[str, float], int, dict[str, Any]]:
    query_tokens = adapter.encode(query_text)
    logits, query_state = adapter.forward(
        query_tokens,
        clone_state(source_state),
    )
    prefix_tokens = adapter.encode(forced_prefix) if forced_prefix else []
    greedy_tokens = []
    for token in prefix_tokens:
        greedy_tokens.append(int(adapter.torch.argmax(logits).item()))
        logits, query_state = adapter.forward([token], query_state)
    prefix_report = {
        "text": forced_prefix,
        "token_ids": prefix_tokens,
        "greedy_token_ids": greedy_tokens,
        "greedy_exact": greedy_tokens == prefix_tokens,
        "roundtrip_exact": (
            adapter.decode(prefix_tokens) == forced_prefix
            if prefix_tokens
            else True
        ),
    }
    scores = {}
    for code, rendered in rendered_answers.items():
        answer_tokens = adapter.encode(rendered)
        answer_logits = logits
        answer_state = clone_state(query_state)
        score = 0.0
        for index, token in enumerate(answer_tokens):
            log_probabilities = adapter.torch.log_softmax(
                answer_logits.float(),
                dim=-1,
            )
            score += float(log_probabilities[token].item())
            if index + 1 < len(answer_tokens):
                answer_logits, answer_state = adapter.forward(
                    [token],
                    answer_state,
                )
        scores[code] = score
    return scores, len(query_tokens), prefix_report


def _mode_manifest(
    manifest: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    trials = [
        trial
        for trial in manifest["trials"]
        if trial["history_mode"] == mode
    ]
    return {
        **manifest,
        "history_mode": mode,
        "trial_count": len(trials),
        "semantic_case_count": manifest["semantic_case_count_per_mode"],
        "trials": trials,
    }


def evaluate_history_binding(
    *,
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    selection_order: Sequence[str],
    minimum_label_marginalized_accuracy: float,
    source_states_immutable: dict[str, bool] | None = None,
) -> dict[str, Any]:
    if not 0.0 <= minimum_label_marginalized_accuracy <= 1.0:
        raise ValueError(
            "minimum_label_marginalized_accuracy must be in [0, 1]"
        )
    modes = tuple(manifest["history_modes"])
    order = tuple(selection_order)
    if set(order) != set(modes) or len(order) != len(modes):
        raise ValueError(
            "selection_order must contain every history mode exactly once"
        )
    immutable = source_states_immutable or {
        mode: True for mode in modes
    }
    mode_reports = {}
    for mode in modes:
        mode_manifest = _mode_manifest(manifest, mode)
        sample_ids = {
            trial["sample_id"] for trial in mode_manifest["trials"]
        }
        mode_records = [
            record for record in records if record["sample_id"] in sample_ids
        ]
        rotation_report = evaluate_g1_code_rotation(
            manifest=mode_manifest,
            records=mode_records,
        )
        diagnostic_complete = bool(
            len(mode_records) == mode_manifest["trial_count"]
            and rotation_report["valid"]
        )
        complete_case_count = sum(
            len(case["rotations"]) == manifest["rotation_count"]
            for case in rotation_report["case_reports"]
        )
        expected_case_count = manifest["semantic_case_count_per_mode"]
        pass_threshold = bool(
            diagnostic_complete
            and complete_case_count == expected_case_count
            and rotation_report["label_marginalized_accuracy"]
            >= minimum_label_marginalized_accuracy
            and immutable.get(mode, False)
        )
        mode_reports[mode] = {
            "history_mode": mode,
            "trial_count": rotation_report["trial_count"],
            "semantic_case_count": expected_case_count,
            "complete_rotation_case_count": complete_case_count,
            "code_level_accuracy": rotation_report["accuracy"],
            "label_marginalized_accuracy": rotation_report[
                "label_marginalized_accuracy"
            ],
            "label_marginalized_correct_case_count": rotation_report[
                "label_marginalized_correct_case_count"
            ],
            "label_marginalized_error_count": rotation_report[
                "label_marginalized_error_count"
            ],
            "source_states_immutable": bool(immutable.get(mode, False)),
            "minimum_label_marginalized_accuracy": (
                minimum_label_marginalized_accuracy
            ),
            "diagnostic_complete": diagnostic_complete,
            "pass_threshold": pass_threshold,
            "rotation_report": rotation_report,
        }
    selected_mode = next(
        (
            mode
            for mode in order
            if mode_reports[mode]["pass_threshold"]
        ),
        None,
    )
    return {
        "report_version": "0.1",
        "development_only": True,
        "selection_rule": (
            "first_passing_mode_in_predeclared_complexity_order"
        ),
        "selection_order": list(order),
        "minimum_label_marginalized_accuracy": (
            minimum_label_marginalized_accuracy
        ),
        "mode_reports": mode_reports,
        "selected_mode": selected_mode,
        "history_binding_gate_passed": selected_mode is not None,
        "route_decision": (
            f"freeze_{selected_mode}"
            if selected_mode is not None
            else "revise_history_binding_protocol"
        ),
        "valid": all(
            report["diagnostic_complete"]
            for report in mode_reports.values()
        ),
    }


def _verify_prerequisites(
    root: Path,
    prerequisites: Any,
    model_id: str,
) -> dict[str, Any]:
    if not isinstance(prerequisites, list) or not prerequisites:
        raise ValueError("prerequisites must be a non-empty list")
    reports = []
    for index, item in enumerate(prerequisites):
        if not isinstance(item, dict):
            raise ValueError(f"prerequisites[{index}] must be an object")
        relative_path = item.get("path")
        checks = item.get("checks")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"prerequisites[{index}].path is invalid")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"prerequisites[{index}].checks is invalid")
        path = (root / relative_path).resolve()
        payload = _load_object(path, f"prerequisite {relative_path}")
        check_results = {
            field: payload.get(field) == expected
            for field, expected in checks.items()
        }
        report_model_id = payload.get("model_id")
        model_valid = (
            report_model_id in (None, model_id)
        )
        reports.append(
            {
                "path": str(path),
                "checks": check_results,
                "model_compatible": model_valid,
                "valid": all(check_results.values()) and model_valid,
            }
        )
    return {
        "model_id": model_id,
        "reports": reports,
        "valid": all(report["valid"] for report in reports),
    }


def run_history_binding_gate(
    *,
    config_path: str | Path,
    gate_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    gate_path = Path(gate_config_path).resolve()
    gate_config = _load_object(gate_path, "history binding gate config")
    gate_name = "impl3p_g1h_2_9b_history_binding"
    if gate_config.get("gate") != gate_name:
        raise ValueError("gate config is not for Impl-3p history binding")
    started_at = _utc_now()

    model_config = load_model_config(config_path, root, verify_files=True)
    prerequisite_report = _verify_prerequisites(
        root,
        gate_config.get("prerequisites"),
        model_config.model_id,
    )
    _write_json(
        destination / "prerequisite_evidence.json",
        prerequisite_report,
    )
    if not prerequisite_report["valid"]:
        raise RuntimeError(
            "Impl-3p prerequisite evidence is incomplete or incompatible"
        )

    answer_codes = _require_codes(gate_config.get("answer_codes"))
    identity_pairs = _require_label_pairs(
        gate_config.get("identity_label_pairs"),
        "identity_label_pairs",
    )
    goal_pairs = _require_label_pairs(
        gate_config.get("goal_label_pairs"),
        "goal_label_pairs",
    )
    history_modes = gate_config.get("history_modes")
    if not isinstance(history_modes, list):
        raise ValueError("history_modes must be a list")
    selection = gate_config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    selection_order = selection.get("ordered_modes")
    minimum_accuracy = selection.get(
        "minimum_label_marginalized_accuracy"
    )
    if not isinstance(selection_order, list):
        raise ValueError("selection.ordered_modes must be a list")
    if not isinstance(minimum_accuracy, (int, float)) or isinstance(
        minimum_accuracy,
        bool,
    ):
        raise ValueError(
            "selection.minimum_label_marginalized_accuracy is invalid"
        )
    if selection.get("rule") != (
        "first_passing_mode_in_predeclared_complexity_order"
    ):
        raise ValueError("Impl-3p selection rule is not the frozen rule")
    if selection.get("require_complete_four_code_rotation") is not True:
        raise ValueError("Impl-3p requires complete four-code rotation")
    if selection.get("require_source_state_immutable") is not True:
        raise ValueError("Impl-3p requires immutable source states")
    repetitions = gate_config.get("repetitions")
    base_seed = gate_config.get("base_seed")
    delay_units = gate_config.get("delay_units")
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not isinstance(delay_units, int) or delay_units < 0:
        raise ValueError("delay_units must be a non-negative integer")
    assistant_prefix = str(gate_config.get("assistant_prefix", ""))
    forced_answer_prefix = str(
        gate_config.get("forced_answer_prefix", "")
    )
    continuation_prefix = str(
        gate_config.get("answer_continuation_prefix", "")
    )
    if (
        assistant_prefix != "<think></think"
        or forced_answer_prefix != ">\n"
        or continuation_prefix != ""
    ):
        raise ValueError(
            "Impl-3p requires the audited newline answer boundary"
        )
    if gate_config.get("general_rule_visible") is not True:
        raise ValueError("Impl-3p requires general_rule_visible=true")
    if gate_config.get("prewarm_each_token_shape") is not True:
        raise ValueError("Impl-3p requires prewarm_each_token_shape=true")

    manifest = generate_history_binding_manifest(
        answer_codes=answer_codes,
        identity_label_pairs=identity_pairs,
        goal_label_pairs=goal_pairs,
        history_modes=history_modes,
        repetitions=repetitions,
        base_seed=base_seed,
        delay_units=delay_units,
        assistant_prefix=assistant_prefix,
    )
    _write_json(destination / "history_binding_manifest.json", manifest)

    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(model_config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    answer_report = inspect_answer_codes(
        adapter,
        answer_codes,
        continuation_prefix=continuation_prefix,
        require_equal_token_count=True,
    )
    _write_json(destination / "answer_interface_report.json", answer_report)
    if not answer_report["valid"]:
        raise RuntimeError("answer interface tokenization is invalid")
    rendered_answers = {
        code: f"{continuation_prefix}{code}" for code in answer_codes
    }

    shape_representatives = {}
    for trial in manifest["trials"]:
        signature = (
            trial["history_mode"],
            len(adapter.encode(trial["history_text"])),
            len(adapter.encode(trial["query_text"])),
        )
        shape_representatives.setdefault(signature, trial)
    warmup_reports = []
    for signature, trial in sorted(shape_representatives.items()):
        _, history_state = adapter.forward(
            adapter.encode(trial["history_text"]),
            None,
        )
        _, _, prefix_report = _score_from_state(
            adapter,
            query_text=trial["query_text"],
            source_state=history_state,
            rendered_answers=rendered_answers,
            forced_prefix=forced_answer_prefix,
        )
        warmup_reports.append(
            {
                "history_mode": signature[0],
                "history_token_count": signature[1],
                "query_token_count": signature[2],
                "sample_id": trial["sample_id"],
                "forced_prefix_greedy_exact": prefix_report[
                    "greedy_exact"
                ],
                "excluded_from_scoring": True,
            }
        )
    _write_json(
        destination / "shape_warmup_report.json",
        {
            "report_version": "0.1",
            "prewarm_each_token_shape": True,
            "warmup_count": len(warmup_reports),
            "warmups": warmup_reports,
            "valid": all(
                report["forced_prefix_greedy_exact"]
                for report in warmup_reports
            ),
        },
    )

    trials_by_history = {}
    for trial in manifest["trials"]:
        trials_by_history.setdefault(trial["history_key"], []).append(trial)
    records = []
    source_immutable_by_mode = {
        mode: True for mode in history_modes
    }
    run_started = time.perf_counter()
    for history_key, trials in sorted(trials_by_history.items()):
        reference_trial = trials[0]
        history_tokens = adapter.encode(reference_trial["history_text"])
        _, history_state = adapter.forward(history_tokens, None)
        pristine_state = clone_state(history_state)
        for trial in sorted(
            trials,
            key=lambda item: item["rotation_index"],
        ):
            try:
                scores, query_token_count, prefix_report = _score_from_state(
                    adapter,
                    query_text=trial["query_text"],
                    source_state=history_state,
                    rendered_answers=rendered_answers,
                    forced_prefix=forced_answer_prefix,
                )
                records.append(
                    {
                        "record_version": "0.1",
                        "sample_id": trial["sample_id"],
                        "semantic_case_id": trial["semantic_case_id"],
                        "history_key": history_key,
                        "history_mode": trial["history_mode"],
                        "rotation_index": trial["rotation_index"],
                        "history_token_count": len(history_tokens),
                        "query_token_count": query_token_count,
                        "option_scores": scores,
                        "option_probabilities": normalized_probabilities(
                            scores
                        ),
                        "argmax_choice": max(
                            scores,
                            key=scores.__getitem__,
                        ),
                        "forced_prefix": prefix_report,
                        "format_valid": prefix_report["greedy_exact"],
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "record_version": "0.1",
                        "sample_id": trial["sample_id"],
                        "semantic_case_id": trial["semantic_case_id"],
                        "history_key": history_key,
                        "history_mode": trial["history_mode"],
                        "rotation_index": trial["rotation_index"],
                        "option_scores": {},
                        "argmax_choice": None,
                        "forced_prefix": None,
                        "format_valid": False,
                        "status": "failed",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
        state_comparison = compare_states(
            history_state,
            pristine_state,
            torch,
        )
        source_immutable_by_mode[reference_trial["history_mode"]] = bool(
            source_immutable_by_mode[reference_trial["history_mode"]]
            and state_comparison["exact"]
        )
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started
    write_jsonl(destination / "raw_history_binding.jsonl", records)

    report = evaluate_history_binding(
        manifest=manifest,
        records=records,
        selection_order=selection_order,
        minimum_label_marginalized_accuracy=float(minimum_accuracy),
        source_states_immutable=source_immutable_by_mode,
    )
    successful_records = [
        record for record in records if record["status"] == "success"
    ]
    forced_prefix_exact_rate = (
        sum(
            bool(record["forced_prefix"]["greedy_exact"])
            for record in successful_records
        )
        / len(successful_records)
        if successful_records
        else 0.0
    )
    warmup_valid = all(
        item["forced_prefix_greedy_exact"] for item in warmup_reports
    )
    report["forced_prefix_greedy_exact_rate"] = (
        forced_prefix_exact_rate
    )
    report["shape_warmup_count"] = len(warmup_reports)
    report["shape_warmup_valid"] = warmup_valid
    report["valid"] = bool(
        report["valid"]
        and prerequisite_report["valid"]
        and answer_report["valid"]
        and forced_prefix_exact_rate == 1.0
        and warmup_valid
    )
    _write_json(destination / "history_binding_report.json", report)

    mode_metrics = {
        mode: {
            "label_marginalized_accuracy": mode_report[
                "label_marginalized_accuracy"
            ],
            "code_level_accuracy": mode_report["code_level_accuracy"],
            "complete_rotation_case_count": mode_report[
                "complete_rotation_case_count"
            ],
            "source_states_immutable": mode_report[
                "source_states_immutable"
            ],
            "pass_threshold": mode_report["pass_threshold"],
        }
        for mode, mode_report in report["mode_reports"].items()
    }
    summary = {
        "gate": gate_name,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "development_only": True,
        "model_id": model_config.model_id,
        "gate_config_sha256": sha256_file(gate_path),
        "manifest_digest_sha256": manifest[
            "manifest_digest_sha256"
        ],
        "history_modes": list(history_modes),
        "selection_order": list(selection_order),
        "minimum_label_marginalized_accuracy": float(
            minimum_accuracy
        ),
        "semantic_case_count_per_mode": manifest[
            "semantic_case_count_per_mode"
        ],
        "trial_count": manifest["trial_count"],
        "shape_warmup_count": len(warmup_reports),
        "forced_prefix_greedy_exact_rate": forced_prefix_exact_rate,
        "mode_metrics": mode_metrics,
        "selected_mode": report["selected_mode"],
        "history_binding_gate_passed": report[
            "history_binding_gate_passed"
        ],
        "route_decision": report["route_decision"],
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "cuda_peak_memory_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "reports": [
            "prerequisite_evidence.json",
            "answer_interface_report.json",
            "history_binding_manifest.json",
            "shape_warmup_report.json",
            "raw_history_binding.jsonl",
            "history_binding_report.json",
        ],
        "valid": report["valid"],
    }
    _write_json(destination / "summary.json", summary)
    return summary
