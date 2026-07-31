from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import import_module
import itertools
import json
import math
from pathlib import Path
import random
from statistics import NormalDist, mean, stdev
import time
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)
from psa.development.capability_ladder import evaluate_g1_code_rotation
from psa.development.history_binding import _score_from_state
from psa.development.impl3 import (
    inspect_answer_codes,
    normalized_probabilities,
    write_jsonl,
)
from psa.evaluation import bca_mean_interval, group_contrasts
from psa.model import RWKV7Adapter, load_model_config
from psa.preregistration.formal_review import review_control_rotation
from psa.tasks import generate_factorial_group


FORMAL_GATE = "impl3q_exp001_formal_freeze_candidate"
FORMAL_GATE_V2 = "impl3r_exp001_formal_freeze_candidate_v2"
FORMAL_GATE_V3 = "impl3s_exp001_formal_freeze_candidate_v3"
SUPPORTED_FORMAL_GATES = (
    FORMAL_GATE,
    FORMAL_GATE_V2,
    FORMAL_GATE_V3,
)
SEED_PURPOSES = {
    "core_generator": "core-generator",
    "control_generator": "control-generator",
    "bootstrap": "bootstrap",
    "permutation": "permutation",
    "simulation": "simulation",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _load_object(path: str | Path, field: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_formal_config(path: Path, root: Path) -> dict[str, Any]:
    raw = _load_object(path, "formal freeze config")
    extends = raw.get("extends")
    if extends is None:
        return raw
    if not isinstance(extends, str):
        raise ValueError("formal config extends must be a relative path")
    base_path = _resolve_under(root, extends, "formal config extends")
    base = _load_object(base_path, "base formal freeze config")
    override = {
        key: value
        for key, value in raw.items()
        if key not in {"extends", "source_files_append"}
    }
    merged = _deep_merge(base, override)
    appended = raw.get("source_files_append", [])
    if not isinstance(appended, list) or not all(
        isinstance(item, str) and item for item in appended
    ):
        raise ValueError("source_files_append must contain relative paths")
    merged["source_files"] = list(
        dict.fromkeys([*base["source_files"], *appended])
    )
    merged["config_lineage"] = {
        "base": extends,
        "base_sha256": sha256_file(base_path),
        "overlay": str(path.relative_to(root)).replace("\\", "/"),
        "overlay_sha256": sha256_file(path),
    }
    return merged


def _resolve_under(root: Path, relative_path: str, field: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"{field} must be a non-empty relative path")
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes its locked root") from exc
    return resolved


def derive_formal_seed(purpose: str) -> int:
    """Derive a portable unsigned 32-bit seed from the frozen namespace."""
    if purpose not in SEED_PURPOSES.values():
        raise ValueError(f"unsupported formal seed purpose: {purpose}")
    text = f"PSA|EXP-001|formal-v1|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


def _validate_seed_lock(config: Mapping[str, Any]) -> dict[str, int]:
    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("seeds must be an object")
    expected = {
        field: derive_formal_seed(purpose)
        for field, purpose in SEED_PURPOSES.items()
    }
    observed = {field: seeds.get(field) for field in expected}
    if observed != expected:
        raise ValueError(
            f"formal seeds do not match the frozen SHA-256 derivation: "
            f"expected {expected}, observed {observed}"
        )
    return expected


def _require_templates(
    payload: Any,
    *,
    field: str,
    text_field: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError(f"{field} must contain exactly four templates")
    templates: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        identifier = item.get("id")
        text = item.get(text_field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{field}[{index}].id is invalid")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{field}[{index}].{text_field} is invalid")
        if identifier in identifiers:
            raise ValueError(f"{field} contains duplicate template IDs")
        identifiers.add(identifier)
        templates.append(
            {
                key: value
                for key, value in item.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        )
    return tuple(templates)


def _require_label_pairs(
    payload: Any,
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{field} must be a non-empty list")
    result = []
    for index, pair in enumerate(payload):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(value, str) and value for value in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError(f"{field}[{index}] must contain two labels")
        result.append((pair[0], pair[1]))
    return tuple(result)


def _render_chat(user_text: str, assistant_prefix: str) -> str:
    return f"User: {user_text.strip()}\n\nAssistant: {assistant_prefix}"


def _fit_filler(
    *,
    variant_index: int,
    filler_config: Mapping[str, Any],
    token_counter: Callable[[str], int],
) -> dict[str, Any]:
    units = filler_config.get("neutral_units")
    fragments = filler_config.get("padding_fragments")
    delay_units = filler_config.get("delay_units")
    target = filler_config.get("target_token_count")
    if (
        not isinstance(units, list)
        or len(units) != 4
        or not all(isinstance(item, str) and item for item in units)
    ):
        raise ValueError("filler_protocol.neutral_units must contain four strings")
    if (
        not isinstance(fragments, list)
        or not fragments
        or not all(isinstance(item, str) and item for item in fragments)
    ):
        raise ValueError("filler_protocol.padding_fragments is invalid")
    if not isinstance(delay_units, int) or delay_units != 11:
        raise ValueError("formal filler requires delay_units=11")
    if not isinstance(target, int) or target != 131:
        raise ValueError("formal filler requires target_token_count=131")

    ordered = [
        units[(position + variant_index) % len(units)]
        for position in range(delay_units)
    ]
    for kept_count in range(delay_units, 0, -1):
        base = "\n".join(ordered[:kept_count])
        base_count = token_counter(base)
        if base_count == target:
            return {
                "variant_id": f"formal-filler-{variant_index + 1:02d}",
                "text": base,
                "token_count": base_count,
                "neutral_unit_count": kept_count,
                "padding_fragment": None,
                "padding_repetitions": 0,
            }
        if base_count > target:
            continue
        for fragment in fragments:
            for repetitions in range(1, target * 3 + 1):
                candidate = (
                    f"{base}\nNEUTRAL-PADDING:"
                    f"{fragment * repetitions}."
                )
                candidate_count = token_counter(candidate)
                if candidate_count == target:
                    return {
                        "variant_id": (
                            f"formal-filler-{variant_index + 1:02d}"
                        ),
                        "text": candidate,
                        "token_count": candidate_count,
                        "neutral_unit_count": kept_count,
                        "padding_fragment": fragment,
                        "padding_repetitions": repetitions,
                    }
                if candidate_count > target + 8:
                    break
    raise RuntimeError(
        f"could not deterministically fit filler variant {variant_index} "
        f"to {target} tokens"
    )


def _render_history(
    template: Mapping[str, str],
    *,
    domain: str,
    operation: str,
    filler: str,
) -> str:
    user_text = template["user_text"].format(
        domain=domain,
        operation=operation,
    )
    acknowledgement = template["assistant_ack"]
    return (
        f"User: {user_text}\n\n"
        f"Assistant: {acknowledgement}\n\n"
        "User: Review these neutral records without changing the saved state.\n"
        f"{filler}\n\n"
        "Assistant: Neutral records reviewed; saved state unchanged."
    )


def _render_query(
    template: Mapping[str, str],
    *,
    option_mapping: Sequence[Mapping[str, str]],
    assistant_prefix: str,
) -> str:
    options = "\n".join(
        f"{item['code']}. DOMAIN: {item['domain']} | "
        f"OPERATION: {item['operation']}"
        for item in option_mapping
    )
    user_text = template["user_text"].format(options=options)
    return f"User: {user_text}\n\nAssistant: {assistant_prefix}"


def generate_template_qualification_manifest(
    config: Mapping[str, Any],
    *,
    token_counter: Callable[[str], int],
) -> dict[str, Any]:
    """Build prompt-visible formal-template qualification trials only."""
    _validate_seed_lock(config)
    answer_interface = config.get("answer_interface")
    labels = config.get("labels")
    history_protocol = config.get("history_protocol")
    query_protocol = config.get("query_protocol")
    filler_protocol = config.get("filler_protocol")
    qualification = config.get("template_qualification")
    if not all(
        isinstance(value, dict)
        for value in (
            answer_interface,
            labels,
            history_protocol,
            query_protocol,
            filler_protocol,
            qualification,
        )
    ):
        raise ValueError("formal template configuration is incomplete")
    if history_protocol.get("mode") != "single_statement":
        raise ValueError("formal history mode must be single_statement")
    if query_protocol.get("general_rule_visible") is not True:
        raise ValueError("formal query must keep the general rule visible")
    if (
        query_protocol.get(
            "current_values_visible_outside_balanced_options"
        )
        is not False
    ):
        raise ValueError("formal query must hide current values outside options")

    codes = tuple(answer_interface.get("answer_codes", ()))
    if len(codes) != 4 or len(set(codes)) != 4:
        raise ValueError("answer_codes must contain four distinct values")
    assistant_prefix = answer_interface.get("assistant_prefix")
    if assistant_prefix != "<think></think":
        raise ValueError("formal qualification requires fake-think prefix")
    histories = _require_templates(
        history_protocol.get("templates"),
        field="history_protocol.templates",
        text_field="user_text",
    )
    if any("assistant_ack" not in template for template in histories):
        raise ValueError("each formal history template needs assistant_ack")
    queries = _require_templates(
        query_protocol.get("templates"),
        field="query_protocol.templates",
        text_field="user_text",
    )
    identity_pairs = _require_label_pairs(
        labels.get("identity_label_pairs"),
        field="labels.identity_label_pairs",
    )
    goal_pairs = _require_label_pairs(
        labels.get("goal_label_pairs"),
        field="labels.goal_label_pairs",
    )
    semantic_per_pair = qualification.get(
        "semantic_cases_per_template_pair"
    )
    if (
        not isinstance(semantic_per_pair, int)
        or semantic_per_pair <= 0
        or semantic_per_pair % 4
    ):
        raise ValueError(
            "semantic_cases_per_template_pair must be a positive multiple of 4"
        )
    fillers = [
        _fit_filler(
            variant_index=index,
            filler_config=filler_protocol,
            token_counter=token_counter,
        )
        for index in range(4)
    ]
    if {item["token_count"] for item in fillers} != {131}:
        raise ValueError("all formal fillers must contain exactly 131 tokens")
    if len({item["text"] for item in fillers}) != 4:
        raise ValueError("formal filler variants must have distinct text")

    base_seed = int(config["seeds"]["core_generator"])
    rng = random.Random(base_seed ^ 0x514C4659)
    pair_pool = tuple(itertools.product(identity_pairs, goal_pairs))
    trials: list[dict[str, Any]] = []
    group_count_per_pair = semantic_per_pair // 4
    for history_index, history_template in enumerate(histories):
        for query_index, query_template in enumerate(queries):
            for group_index in range(group_count_per_pair):
                pool_index = (
                    history_index * len(queries)
                    + query_index
                    + group_index
                ) % len(pair_pool)
                identity_pair, goal_pair = pair_pool[pool_index]
                group_seed = rng.getrandbits(63)
                filler = fillers[
                    (
                        history_index
                        + query_index
                        + group_index
                    )
                    % len(fillers)
                ]
                for rotation_index in range(4):
                    rotated_codes = (
                        codes[rotation_index:] + codes[:rotation_index]
                    )
                    group = generate_factorial_group(
                        group_seed=group_seed,
                        track="synthetic",
                        identity_labels=identity_pair,
                        goal_labels=goal_pair,
                        answer_codes=rotated_codes,
                        delay_units=0,
                        generator_version="exp001-formal-qualification-v1",
                        history_order="I_G",
                    )
                    option_mapping = [
                        {
                            "code": option.code,
                            "domain": group.identity_labels[option.identity],
                            "operation": group.goal_labels[option.goal],
                            "identity": option.identity,
                            "goal": option.goal,
                        }
                        for option in group.options
                    ]
                    query_text = _render_query(
                        query_template,
                        option_mapping=option_mapping,
                        assistant_prefix=assistant_prefix,
                    )
                    query_before_options = query_text.split(
                        "OPTIONS:", maxsplit=1
                    )[0]
                    for sample in group.trajectories:
                        domain = group.identity_labels[sample.identity]
                        operation = group.goal_labels[sample.goal]
                        if (
                            domain in query_before_options
                            or operation in query_before_options
                        ):
                            raise ValueError(
                                "formal state-only query leaks current values"
                            )
                        history_text = _render_history(
                            history_template,
                            domain=domain,
                            operation=operation,
                            filler=filler["text"],
                        )
                        full_prompt = f"{history_text}\n\n{query_text}"
                        semantic_case_id = "formalcase-" + sha256_json(
                            {
                                "history_template_id": history_template["id"],
                                "query_template_id": query_template["id"],
                                "group_index": group_index,
                                "group_seed": group_seed,
                                "identity": sample.identity,
                                "goal": sample.goal,
                            }
                        )[:24]
                        sample_id = "formalqual-" + sha256_json(
                            {
                                "semantic_case_id": semantic_case_id,
                                "rotation_index": rotation_index,
                                "target_code": sample.correct_code,
                                "prompt": full_prompt,
                            }
                        )[:24]
                        trials.append(
                            {
                                "sample_id": sample_id,
                                "semantic_case_id": semantic_case_id,
                                "rotation_index": rotation_index,
                                "history_template_id": history_template["id"],
                                "query_template_id": query_template["id"],
                                "filler_variant_id": filler["variant_id"],
                                "factorial_group_key": (
                                    f"{history_template['id']}:"
                                    f"{query_template['id']}:"
                                    f"{group_index}:{group_seed}"
                                ),
                                "prompt": full_prompt,
                                "prompt_digest_sha256": sha256_json(full_prompt),
                                "target_code": sample.correct_code,
                                "target_fields": {
                                    "domain": domain,
                                    "operation": operation,
                                    "identity": sample.identity,
                                    "goal": sample.goal,
                                },
                                "option_mapping": option_mapping,
                            }
                        )

    expected_cases = int(qualification["expected_semantic_case_count"])
    expected_trials = int(qualification["expected_trial_count"])
    semantic_case_count = len(
        {trial["semantic_case_id"] for trial in trials}
    )
    if semantic_case_count != expected_cases or len(trials) != expected_trials:
        raise ValueError(
            "formal qualification manifest count does not match frozen config"
        )
    return {
        "manifest_version": "1.0",
        "development_only": True,
        "confirmatory_results_observed": False,
        "diagnostic": "prompt_visible_formal_template_qualification",
        "answer_codes": list(codes),
        "rotation_count": 4,
        "history_template_ids": [item["id"] for item in histories],
        "query_template_ids": [item["id"] for item in queries],
        "filler_variants": fillers,
        "semantic_case_count": semantic_case_count,
        "trial_count": len(trials),
        "trials": trials,
        "manifest_digest_sha256": sha256_json(trials),
    }


def generate_control_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _validate_seed_lock(config)
    answer = config["answer_interface"]
    control = config["controls"]
    codes = tuple(answer["answer_codes"])
    assistant_prefix = str(answer["assistant_prefix"])
    case_count = int(control["semantic_cases_per_task"])
    task_types = tuple(control["task_types"])
    if task_types != (
        "answer_code_copy",
        "single_field_lexical_match",
        "unrelated_two_field_symbol_match",
    ):
        raise ValueError("formal control task types differ from D5")
    if case_count != 8 or int(control["rotations_per_case"]) != 4:
        raise ValueError("formal controls require 8 cases x 4 rotations")

    rng = random.Random(int(config["seeds"]["control_generator"]))
    vocabulary = control.get("vocabulary", {})
    if not isinstance(vocabulary, dict):
        raise ValueError("controls.vocabulary must be an object")
    symbols = tuple(
        vocabulary.get(
            "single_field_symbols",
            ("luma", "sora", "pavi", "toke"),
        )
    )
    two_field_names = tuple(
        vocabulary.get("two_field_names", ("MARKER", "PATTERN"))
    )
    two_field_values = vocabulary.get(
        "two_field_values",
        (("cinder", "harbor"), ("trace", "fold")),
    )
    if (
        len(symbols) != 4
        or len(set(symbols)) != 4
        or not all(isinstance(item, str) and item for item in symbols)
    ):
        raise ValueError("control single-field symbols must be four values")
    if (
        len(two_field_names) != 2
        or not all(
            isinstance(item, str) and item for item in two_field_names
        )
    ):
        raise ValueError("control two-field names must contain two values")
    if (
        not isinstance(two_field_values, (list, tuple))
        or len(two_field_values) != 2
    ):
        raise ValueError("control two-field values must contain two pairs")
    domains = tuple(two_field_values[0])
    operations = tuple(two_field_values[1])
    if (
        len(domains) != 2
        or len(operations) != 2
        or len(set(domains)) != 2
        or len(set(operations)) != 2
        or not all(
            isinstance(item, str) and item
            for item in (*domains, *operations)
        )
    ):
        raise ValueError("each control two-field value set needs two labels")
    field_combos = tuple(itertools.product(domains, operations))
    trials: list[dict[str, Any]] = []
    for task_type in task_types:
        for case_index in range(case_count):
            semantic_case_id = f"control-{task_type}-" + sha256_json(
                {
                    "case_index": case_index,
                    "seed": config["seeds"]["control_generator"],
                }
            )[:20]
            base_rotation = rng.randrange(4)
            for rotation_index in range(4):
                rotated_codes = (
                    codes[
                        (base_rotation + rotation_index) % 4 :
                    ]
                    + codes[
                        : (base_rotation + rotation_index) % 4
                    ]
                )
                if task_type == "answer_code_copy":
                    target_code = rotated_codes[case_index % 4]
                    user_text = (
                        "CONTROL TASK: copy one answer code.\n"
                        f"TARGET CODE: {target_code}\n"
                        "Return exactly the target code and nothing else."
                    )
                    target_fields = {"control_value": target_code}
                    option_mapping = [
                        {"code": code, "control_value": code}
                        for code in codes
                    ]
                elif task_type == "single_field_lexical_match":
                    mapping = list(
                        zip(rotated_codes, symbols, strict=True)
                    )
                    target_symbol = symbols[case_index % 4]
                    target_code = next(
                        code
                        for code, symbol in mapping
                        if symbol == target_symbol
                    )
                    options = "\n".join(
                        f"{code}. SYMBOL: {symbol}"
                        for code, symbol in mapping
                    )
                    user_text = (
                        "CONTROL TASK: exact lexical match.\n"
                        f"CURRENT SYMBOL: {target_symbol}\n"
                        "Choose the option with the same symbol.\n"
                        f"OPTIONS:\n{options}\n"
                        "Return only the matching option code."
                    )
                    target_fields = {"symbol": target_symbol}
                    option_mapping = [
                        {"code": code, "symbol": symbol}
                        for code, symbol in mapping
                    ]
                else:
                    mapping = list(
                        zip(rotated_codes, field_combos, strict=True)
                    )
                    target_combo = field_combos[case_index % 4]
                    target_code = next(
                        code
                        for code, combo in mapping
                        if combo == target_combo
                    )
                    options = "\n".join(
                        f"{code}. {two_field_names[0]}: {combo[0]} | "
                        f"{two_field_names[1]}: {combo[1]}"
                        for code, combo in mapping
                    )
                    user_text = (
                        "CONTROL TASK: exact visible two-field match.\n"
                        f"TARGET {two_field_names[0]}: {target_combo[0]}\n"
                        f"TARGET {two_field_names[1]}: {target_combo[1]}\n"
                        "An option is correct only when both fields equal "
                        "their TARGET values.\n"
                        f"OPTIONS:\n{options}\n"
                        "Return only the matching option code."
                    )
                    target_fields = {
                        "marker": target_combo[0],
                        "pattern": target_combo[1],
                    }
                    option_mapping = [
                        {
                            "code": code,
                            "marker": combo[0],
                            "pattern": combo[1],
                        }
                        for code, combo in mapping
                    ]
                prompt = _render_chat(user_text, assistant_prefix)
                trials.append(
                    {
                        "sample_id": "formalctrl-" + sha256_json(
                            {
                                "semantic_case_id": semantic_case_id,
                                "rotation_index": rotation_index,
                                "prompt": prompt,
                            }
                        )[:24],
                        "semantic_case_id": semantic_case_id,
                        "task_type": task_type,
                        "rotation_index": rotation_index,
                        "prompt": prompt,
                        "prompt_digest_sha256": sha256_json(prompt),
                        "target_code": target_code,
                        "target_fields": target_fields,
                        "option_mapping": option_mapping,
                    }
                )
    if len(trials) != int(control["expected_trial_count"]):
        raise ValueError("formal control manifest count differs from D5")
    return {
        "manifest_version": "1.0",
        "development_only": True,
        "confirmatory_results_observed": False,
        "diagnostic": "formal_general_capability_controls",
        "answer_codes": list(codes),
        "task_types": list(task_types),
        "semantic_case_count_per_task": case_count,
        "rotation_count": 4,
        "trial_count": len(trials),
        "trials": trials,
        "manifest_digest_sha256": sha256_json(trials),
    }


def _percentile_interval(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    data = [float(value) for value in values]
    if len(data) < 2:
        raise ValueError("bootstrap interval requires at least two values")
    rng = random.Random(seed)
    estimates = sorted(
        mean(data[rng.randrange(len(data))] for _ in data)
        for _ in range(replicates)
    )
    alpha = (1.0 - confidence) / 2.0

    def pick(probability: float) -> float:
        index = round(probability * (len(estimates) - 1))
        return estimates[max(0, min(index, len(estimates) - 1))]

    return pick(alpha), pick(1.0 - alpha)


def _accuracy_interval(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[tuple[float, float], str]:
    try:
        interval = bca_mean_interval(
            values,
            replicates=replicates,
            seed=seed,
        )
        if all(math.isfinite(value) for value in interval):
            return interval, "BCa"
    except (ArithmeticError, ValueError):
        pass
    return (
        _percentile_interval(
            values,
            replicates=replicates,
            seed=seed,
        ),
        "percentile_fallback",
    )


def evaluate_template_qualification(
    *,
    manifest: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    thresholds: Mapping[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    rotation_report = evaluate_g1_code_rotation(
        manifest=dict(manifest),
        records=records,
    )
    trial_by_case: dict[str, dict[str, Any]] = {}
    for trial in manifest["trials"]:
        trial_by_case.setdefault(trial["semantic_case_id"], trial)
    joint_values = []
    identity_values = []
    goal_values = []
    enriched_cases = []
    for case in rotation_report["case_reports"]:
        trial = trial_by_case[case["semantic_case_id"]]
        prediction = case["label_marginalized_prediction"]
        target = case["target_fields"]
        joint_correct = bool(case["label_marginalized_correct"])
        identity_correct = bool(
            prediction is not None
            and prediction["domain"] == target["domain"]
        )
        goal_correct = bool(
            prediction is not None
            and prediction["operation"] == target["operation"]
        )
        joint_values.append(float(joint_correct))
        identity_values.append(float(identity_correct))
        goal_values.append(float(goal_correct))
        enriched_cases.append(
            {
                "semantic_case_id": case["semantic_case_id"],
                "history_template_id": trial["history_template_id"],
                "query_template_id": trial["query_template_id"],
                "joint_correct": joint_correct,
                "identity_correct": identity_correct,
                "goal_correct": goal_correct,
                "rotation_count": len(case["rotations"]),
            }
        )
    scores_by_group: dict[
        str,
        dict[tuple[int, int], dict[tuple[int, int], float]],
    ] = {}
    for case in rotation_report["case_reports"]:
        trial = trial_by_case[case["semantic_case_id"]]
        combo_by_labels = {
            (option["domain"], option["operation"]): (
                int(option["identity"]),
                int(option["goal"]),
            )
            for option in trial["option_mapping"]
        }
        semantic_scores = {
            combo_by_labels[
                (item["domain"], item["operation"])
            ]: float(item["mean_log_score"])
            for item in case["label_marginalized_scores"]
        }
        state_combo = (
            int(trial["target_fields"]["identity"]),
            int(trial["target_fields"]["goal"]),
        )
        scores_by_group.setdefault(
            trial["factorial_group_key"],
            {},
        )[state_combo] = semantic_scores
    group_level_contrasts = [
        group_contrasts(state_scores)
        for _, state_scores in sorted(scores_by_group.items())
        if len(state_scores) == 4
    ]
    nuisance_fields = {
        "E1_identity_transfer": "identity_transfer",
        "E2_goal_transfer": "goal_transfer",
        "E3_joint_binding": "mean_joint_margin",
    }
    nuisance_standard_deviation = {
        endpoint: stdev(
            [
                contrasts[field]
                for contrasts in group_level_contrasts
            ]
        )
        for endpoint, field in nuisance_fields.items()
        if len(group_level_contrasts) >= 2
    }
    replicates = int(thresholds["bootstrap_replicates"])
    intervals = {}
    for offset, (name, values) in enumerate(
        (
            ("joint", joint_values),
            ("identity", identity_values),
            ("goal", goal_values),
        )
    ):
        interval, method = _accuracy_interval(
            values,
            replicates=replicates,
            seed=bootstrap_seed + offset,
        )
        intervals[name] = {
            "point": mean(values) if values else 0.0,
            "interval": list(interval),
            "method": method,
        }

    def per_template(field: str) -> dict[str, Any]:
        identifiers = sorted({case[field] for case in enriched_cases})
        return {
            identifier: {
                "semantic_case_count": len(
                    [
                        case
                        for case in enriched_cases
                        if case[field] == identifier
                    ]
                ),
                "joint_accuracy": mean(
                    [
                        float(case["joint_correct"])
                        for case in enriched_cases
                        if case[field] == identifier
                    ]
                ),
            }
            for identifier in identifiers
        }

    history_metrics = per_template("history_template_id")
    query_metrics = per_template("query_template_id")
    complete = bool(
        len(records) == manifest["trial_count"]
        and rotation_report["valid"]
        and len(enriched_cases) == manifest["semantic_case_count"]
        and all(case["rotation_count"] == 4 for case in enriched_cases)
    )
    passed = bool(
        complete
        and intervals["joint"]["interval"][0]
        >= float(thresholds["minimum_joint_accuracy_lower_bound"])
        and intervals["identity"]["interval"][0]
        >= float(thresholds["minimum_identity_accuracy_lower_bound"])
        and intervals["goal"]["interval"][0]
        >= float(thresholds["minimum_goal_accuracy_lower_bound"])
        and rotation_report["format_valid_rate"]
        >= float(thresholds["minimum_format_valid_rate"])
        and min(
            item["joint_accuracy"] for item in history_metrics.values()
        )
        >= float(thresholds["minimum_per_history_template_accuracy"])
        and min(item["joint_accuracy"] for item in query_metrics.values())
        >= float(thresholds["minimum_per_query_template_accuracy"])
    )
    return {
        "report_version": "1.0",
        "development_only": True,
        "confirmatory_results_observed": False,
        "diagnostic_complete": complete,
        "template_qualification_passed": passed,
        "semantic_case_count": len(enriched_cases),
        "trial_count": len(records),
        "metrics": intervals,
        "format_valid_rate": rotation_report["format_valid_rate"],
        "history_template_metrics": history_metrics,
        "query_template_metrics": query_metrics,
        "development_nuisance": {
            "source": (
                "prompt_visible_formal_qualification_group_contrasts"
            ),
            "factorial_group_count": len(group_level_contrasts),
            "standard_deviation": nuisance_standard_deviation,
            "confirmatory_results_observed": False,
        },
        "thresholds": dict(thresholds),
        "rotation_report": rotation_report,
        "route_decision": (
            "freeze_formal_template_family"
            if passed
            else "revise_formal_template_family_without_state_only_results"
        ),
        "valid": complete,
    }


def evaluate_control_records(
    *,
    manifest: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    minimum_accuracy_per_task: float,
    minimum_format_valid_rate: float,
    use_rotation_marginalized_semantic_controls: bool = False,
) -> dict[str, Any]:
    trials = {trial["sample_id"]: trial for trial in manifest["trials"]}
    rotation_review = review_control_rotation(
        manifest=manifest,
        records=records,
        minimum_accuracy=minimum_accuracy_per_task,
    )
    metrics = {}
    complete = len(records) == manifest["trial_count"]
    for task_type in manifest["task_types"]:
        task_records = [
            record
            for record in records
            if trials[record["sample_id"]]["task_type"] == task_type
        ]
        successful = [
            record
            for record in task_records
            if record.get("status") == "success"
        ]
        accuracy = (
            sum(
                record.get("argmax_choice")
                == trials[record["sample_id"]]["target_code"]
                for record in successful
            )
            / len(task_records)
            if task_records
            else 0.0
        )
        format_rate = (
            sum(bool(record.get("format_valid")) for record in successful)
            / len(task_records)
            if task_records
            else 0.0
        )
        task_complete = bool(
            len(task_records)
            == manifest["semantic_case_count_per_task"] * 4
            and len(successful) == len(task_records)
        )
        rotation_task = rotation_review["task_reports"][task_type]
        marginalized_accuracy = rotation_task[
            "label_marginalized_accuracy"
        ]
        use_marginalized = bool(
            use_rotation_marginalized_semantic_controls
            and task_type != "answer_code_copy"
        )
        marginalized_complete = bool(
            marginalized_accuracy is not None
            and rotation_task["diagnostic_complete"]
        )
        if use_marginalized:
            evaluation_accuracy = (
                float(marginalized_accuracy)
                if marginalized_complete
                else 0.0
            )
            task_complete = bool(task_complete and marginalized_complete)
        else:
            evaluation_accuracy = accuracy
        metrics[task_type] = {
            "trial_count": len(task_records),
            "code_level_accuracy": accuracy,
            "label_marginalized_accuracy": marginalized_accuracy,
            "evaluation_readout": (
                "mean_candidate_log_score_across_four_code_rotations"
                if use_marginalized
                else "code_level_argmax_accuracy"
            ),
            "evaluation_accuracy": evaluation_accuracy,
            "evaluation_readout_complete": (
                marginalized_complete
                if use_marginalized
                else task_complete
            ),
            "accuracy": evaluation_accuracy,
            "format_valid_rate": format_rate,
            "diagnostic_complete": task_complete,
            "pass_threshold": bool(
                task_complete
                and evaluation_accuracy >= minimum_accuracy_per_task
                and format_rate >= minimum_format_valid_rate
            ),
        }
        complete = complete and task_complete
    passed = complete and all(
        item["pass_threshold"] for item in metrics.values()
    )
    return {
        "report_version": "1.0",
        "development_only": True,
        "confirmatory_results_observed": False,
        "diagnostic_complete": complete,
        "control_baseline_passed": passed,
        "task_metrics": metrics,
        "minimum_accuracy_per_task": minimum_accuracy_per_task,
        "minimum_format_valid_rate": minimum_format_valid_rate,
        "semantic_control_readout": (
            "rotation_marginalized"
            if use_rotation_marginalized_semantic_controls
            else "code_level"
        ),
        "rotation_review": rotation_review,
        "route_decision": (
            "freeze_general_capability_controls"
            if passed
            else "revise_controls_without_state_only_results"
        ),
        "valid": complete,
    }


def simulate_power(
    config: Mapping[str, Any],
    *,
    nuisance_standard_deviation: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    _validate_seed_lock(config)
    design = config["core_design"]
    simulation = config["power_simulation"]
    statistics = config["statistics"]
    sample_size = int(design["factorial_group_count"])
    replicates = int(simulation["replicates"])
    target_power = float(simulation["minimum_power"])
    effect = float(simulation["standardized_paired_effect_dz"])
    endpoint_count = len(statistics["primary_endpoints"])
    alpha = float(statistics["familywise_alpha"]) / endpoint_count
    critical_z = NormalDist().inv_cdf(1.0 - alpha)
    critical_mean = critical_z / math.sqrt(sample_size)
    rng = random.Random(int(config["seeds"]["simulation"]))
    standardized_power = {}
    for endpoint in statistics["primary_endpoints"]:
        rejections = 0
        for _ in range(replicates):
            simulated_mean = (
                sum(rng.gauss(effect, 1.0) for _ in range(sample_size))
                / sample_size
            )
            rejections += simulated_mean > critical_mean
        standardized_power[endpoint] = rejections / replicates

    empirical_power = None
    empirical_effects = {
        "E1_identity_transfer": float(
            statistics["sesoi"]["identity_log_odds"]
        ),
        "E2_goal_transfer": float(
            statistics["sesoi"]["goal_log_odds"]
        ),
        "E3_joint_binding": float(
            statistics["sesoi"]["joint_log_margin"]
        ),
    }
    if nuisance_standard_deviation is not None:
        if set(nuisance_standard_deviation) != set(empirical_effects):
            raise ValueError(
                "development nuisance SDs must cover E1, E2, and E3"
            )
        empirical_power = {}
        for endpoint, raw_effect in empirical_effects.items():
            nuisance_sd = float(nuisance_standard_deviation[endpoint])
            if not math.isfinite(nuisance_sd) or nuisance_sd < 0.0:
                raise ValueError("development nuisance SD must be finite")
            if nuisance_sd == 0.0:
                empirical_power[endpoint] = 1.0
                continue
            raw_critical_mean = (
                critical_z * nuisance_sd / math.sqrt(sample_size)
            )
            rejections = 0
            for _ in range(replicates):
                simulated_mean = (
                    sum(
                        rng.gauss(raw_effect, nuisance_sd)
                        for _ in range(sample_size)
                    )
                    / sample_size
                )
                rejections += simulated_mean > raw_critical_mean
            empirical_power[endpoint] = rejections / replicates
    require_both = bool(
        simulation.get("require_standardized_and_empirical_proxy")
    )
    passed = bool(
        all(power >= target_power for power in standardized_power.values())
        and (
            not require_both
            or (
                empirical_power is not None
                and all(
                    power >= target_power
                    for power in empirical_power.values()
                )
            )
        )
    )
    return {
        "report_version": "1.0",
        "development_only": True,
        "confirmatory_results_observed": False,
        "simulation_model": simulation["test"],
        "nuisance_source": simulation["nuisance_source"],
        "sample_size": sample_size,
        "replicates": replicates,
        "standardized_paired_effect_dz": effect,
        "familywise_alpha": statistics["familywise_alpha"],
        "conservative_per_endpoint_alpha": alpha,
        "critical_z": critical_z,
        "standardized_endpoint_power": standardized_power,
        "development_nuisance_standard_deviation": (
            dict(nuisance_standard_deviation)
            if nuisance_standard_deviation is not None
            else None
        ),
        "empirical_proxy_endpoint_power": empirical_power,
        "require_standardized_and_empirical_proxy": require_both,
        "minimum_power": target_power,
        "power_gate_passed": passed,
        "sample_size_may_decrease": False,
        "route_decision": (
            "retain_n_320"
            if passed
            else "increase_n_before_core_set_generation"
        ),
        "valid": True,
    }


def _verify_prerequisites(
    root: Path,
    prerequisites: Any,
    model_id: str,
) -> dict[str, Any]:
    if not isinstance(prerequisites, list) or not prerequisites:
        raise ValueError("formal prerequisites must be a non-empty list")
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
        path = _resolve_under(
            root,
            relative_path,
            f"prerequisites[{index}].path",
        )
        payload = _load_object(path, f"prerequisite {relative_path}")
        observed_model = payload.get("model_id")
        model_compatible = observed_model in (None, model_id)
        check_results = {
            field: payload.get(field) == expected
            for field, expected in checks.items()
        }
        reports.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "checks": check_results,
                "model_compatible": model_compatible,
                "valid": model_compatible and all(check_results.values()),
            }
        )
    return {
        "reports": reports,
        "valid": all(item["valid"] for item in reports),
    }


def _score_trials(
    adapter: Any,
    *,
    trials: Sequence[dict[str, Any]],
    rendered_answers: Mapping[str, str],
    forced_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shape_examples: dict[int, dict[str, Any]] = {}
    for trial in trials:
        shape_examples.setdefault(
            len(adapter.encode(trial["prompt"])),
            trial,
        )
    warmups = []
    for token_count, trial in sorted(shape_examples.items()):
        _, _, prefix = _score_from_state(
            adapter,
            query_text=trial["prompt"],
            source_state=None,
            rendered_answers=dict(rendered_answers),
            forced_prefix=forced_prefix,
        )
        warmups.append(
            {
                "prompt_token_count": token_count,
                "sample_id": trial["sample_id"],
                "forced_prefix_greedy_exact": prefix["greedy_exact"],
                "excluded_from_scoring": True,
            }
        )

    records = []
    for trial in trials:
        try:
            scores, token_count, prefix = _score_from_state(
                adapter,
                query_text=trial["prompt"],
                source_state=None,
                rendered_answers=dict(rendered_answers),
                forced_prefix=forced_prefix,
            )
            records.append(
                {
                    "record_version": "1.0",
                    "sample_id": trial["sample_id"],
                    "semantic_case_id": trial["semantic_case_id"],
                    "prompt_token_count": token_count,
                    "option_scores": scores,
                    "option_probabilities": normalized_probabilities(scores),
                    "argmax_choice": max(scores, key=scores.__getitem__),
                    "format_valid": prefix["greedy_exact"],
                    "forced_prefix": prefix,
                    "status": "success",
                    "error": None,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "record_version": "1.0",
                    "sample_id": trial["sample_id"],
                    "semantic_case_id": trial["semantic_case_id"],
                    "option_scores": {},
                    "argmax_choice": None,
                    "format_valid": False,
                    "forced_prefix": None,
                    "status": "failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
    return records, warmups


def _build_candidate(
    *,
    root: Path,
    destination: Path,
    config_path: Path,
    config: Mapping[str, Any],
    gate_name: str,
    model_id: str,
    prerequisite_report: Mapping[str, Any],
    template_report: Mapping[str, Any],
    control_report: Mapping[str, Any],
    power_report: Mapping[str, Any],
) -> dict[str, Any]:
    source_digests = {}
    for relative_path in config["source_files"]:
        path = _resolve_under(
            root,
            relative_path,
            "source_files entry",
        )
        source_digests[relative_path] = sha256_file(path)
    evidence_names = (
        "prerequisite_evidence.json",
        "answer_interface_report.json",
        "template_qualification_manifest.json",
        "raw_template_qualification.jsonl",
        "template_qualification_report.json",
        "control_manifest.json",
        "raw_control_baseline.jsonl",
        "control_baseline_report.json",
        "shape_warmup_report.json",
        "power_simulation_report.json",
    )
    evidence_digests = {
        name: sha256_file(destination / name) for name in evidence_names
    }
    ready = bool(
        prerequisite_report["valid"]
        and template_report["template_qualification_passed"]
        and control_report["control_baseline_passed"]
        and power_report["power_gate_passed"]
        and config["core_design"]["factorial_group_count"] >= 320
        and config["core_design"]["generate_or_unseal_core_set"] is False
        and config["safety_boundary"]["formal_state_only_results_allowed"]
        is False
        and config["safety_boundary"]["core_set_generation_allowed"] is False
    )
    locked_payload_digests = {
        **{f"source:{key}": value for key, value in source_digests.items()},
        **{
            f"evidence:{key}": value
            for key, value in evidence_digests.items()
        },
    }
    candidate = {
        "candidate_version": "1.0",
        "created_at_utc": _utc_now(),
        "status": (
            "frozen_candidate_awaiting_human_checksum_confirmation"
            if ready
            else "hold_not_eligible_for_human_freeze"
        ),
        "gate": gate_name,
        "model_id": model_id,
        "confirmed_decision_ids": config["confirmation"]["decision_ids"],
        "history_mode": config["history_protocol"]["mode"],
        "formal_template_count": len(
            config["history_protocol"]["templates"]
        ),
        "formal_query_template_count": len(
            config["query_protocol"]["templates"]
        ),
        "filler_variant_count": config["filler_protocol"]["variant_count"],
        "control_trial_count": config["controls"]["expected_trial_count"],
        "factorial_group_count": config["core_design"][
            "factorial_group_count"
        ],
        "seeds": config["seeds"],
        "statistics": config["statistics"],
        "conditions": config["core_design"]["conditions"],
        "qualification": {
            "prerequisites_valid": prerequisite_report["valid"],
            "template_qualification_passed": template_report[
                "template_qualification_passed"
            ],
            "control_baseline_passed": control_report[
                "control_baseline_passed"
            ],
            "power_gate_passed": power_report["power_gate_passed"],
        },
        "source_config": {
            "path": str(config_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(config_path),
        },
        "source_file_digests": source_digests,
        "evidence_file_digests": evidence_digests,
        "payload_root_digest_sha256": payload_digest(
            locked_payload_digests
        ),
        "core_set_generated": False,
        "core_set_unsealed": False,
        "formal_state_only_results_observed": False,
        "human_checksum_confirmation_required": True,
        "eligible_for_human_freeze": ready,
    }
    candidate["candidate_digest_sha256"] = sha256_json(candidate)
    return candidate


def verify_preregistration_candidate(
    candidate_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(candidate_path).resolve()
    candidate = _load_object(path, "preregistration candidate")
    expected_self_digest = candidate.get("candidate_digest_sha256")
    unsigned = dict(candidate)
    unsigned.pop("candidate_digest_sha256", None)
    self_digest_valid = (
        isinstance(expected_self_digest, str)
        and sha256_json(unsigned) == expected_self_digest
    )
    source_results = {}
    for relative_path, expected in candidate.get(
        "source_file_digests", {}
    ).items():
        try:
            source = _resolve_under(
                root,
                relative_path,
                "source_file_digests entry",
            )
            source_results[relative_path] = bool(
                source.is_file() and sha256_file(source) == expected
            )
        except (TypeError, ValueError):
            source_results[str(relative_path)] = False
    evidence_results = {}
    for filename, expected in candidate.get(
        "evidence_file_digests", {}
    ).items():
        try:
            evidence = _resolve_under(
                path.parent,
                filename,
                "evidence_file_digests entry",
            )
            evidence_results[filename] = bool(
                evidence.is_file() and sha256_file(evidence) == expected
            )
        except (TypeError, ValueError):
            evidence_results[str(filename)] = False
    observed_payload_root = payload_digest(
        {
            **{
                f"source:{key}": value
                for key, value in candidate.get(
                    "source_file_digests", {}
                ).items()
            },
            **{
                f"evidence:{key}": value
                for key, value in candidate.get(
                    "evidence_file_digests", {}
                ).items()
            },
        }
    )
    payload_root_valid = (
        observed_payload_root
        == candidate.get("payload_root_digest_sha256")
    )
    safety_valid = bool(
        candidate.get("core_set_generated") is False
        and candidate.get("core_set_unsealed") is False
        and candidate.get("formal_state_only_results_observed") is False
        and candidate.get("human_checksum_confirmation_required") is True
    )
    valid = bool(
        self_digest_valid
        and payload_root_valid
        and source_results
        and all(source_results.values())
        and evidence_results
        and all(evidence_results.values())
        and safety_valid
    )
    return {
        "report_version": "1.0",
        "candidate": str(path),
        "candidate_digest_sha256": expected_self_digest,
        "self_digest_valid": self_digest_valid,
        "payload_root_valid": payload_root_valid,
        "source_file_checks": source_results,
        "evidence_file_checks": evidence_results,
        "safety_boundary_valid": safety_valid,
        "eligible_for_human_freeze": candidate.get(
            "eligible_for_human_freeze"
        ),
        "valid": valid,
    }


def run_formal_freeze_candidate_gate(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    formal_config_path = Path(config_path).resolve()
    config = _load_formal_config(formal_config_path, root)
    gate_name = config.get("gate")
    if gate_name not in SUPPORTED_FORMAL_GATES:
        raise ValueError("config is not for a supported formal freeze gate")
    _validate_seed_lock(config)
    if config["confirmation"]["confirmed_by_project_owner"] is not True:
        raise ValueError("D4-D8 have not been confirmed by the project owner")
    if config["core_design"]["generate_or_unseal_core_set"] is not False:
        raise ValueError("Impl-3q must not generate or unseal the Core Set")
    started_at = _utc_now()

    model_config_path = _resolve_under(
        root,
        config["model_config"],
        "model_config",
    )
    model_config = load_model_config(
        model_config_path,
        root,
        verify_files=True,
    )
    prerequisite_report = _verify_prerequisites(
        root,
        config["prerequisites"],
        model_config.model_id,
    )
    _write_json(
        destination / "prerequisite_evidence.json",
        prerequisite_report,
    )
    if not prerequisite_report["valid"]:
        raise RuntimeError(
            "Impl-3q prerequisite evidence is incomplete or incompatible"
        )

    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(model_config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    answer = config["answer_interface"]
    answer_report = inspect_answer_codes(
        adapter,
        answer["answer_codes"],
        continuation_prefix=answer["continuation_prefix"],
        require_equal_token_count=True,
    )
    _write_json(destination / "answer_interface_report.json", answer_report)
    if not answer_report["valid"]:
        raise RuntimeError("formal answer interface is invalid")
    rendered_answers = {
        code: f"{answer['continuation_prefix']}{code}"
        for code in answer["answer_codes"]
    }

    template_manifest = generate_template_qualification_manifest(
        config,
        token_counter=lambda text: len(adapter.encode(text)),
    )
    control_manifest = generate_control_manifest(config)
    _write_json(
        destination / "template_qualification_manifest.json",
        template_manifest,
    )
    _write_json(destination / "control_manifest.json", control_manifest)

    all_trials = [
        *template_manifest["trials"],
        *control_manifest["trials"],
    ]
    run_started = time.perf_counter()
    records, warmups = _score_trials(
        adapter,
        trials=all_trials,
        rendered_answers=rendered_answers,
        forced_prefix=answer["forced_answer_prefix"],
    )
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started
    template_ids = {
        trial["sample_id"] for trial in template_manifest["trials"]
    }
    template_records = [
        record for record in records if record["sample_id"] in template_ids
    ]
    control_records = [
        record for record in records if record["sample_id"] not in template_ids
    ]
    write_jsonl(
        destination / "raw_template_qualification.jsonl",
        template_records,
    )
    write_jsonl(
        destination / "raw_control_baseline.jsonl",
        control_records,
    )
    _write_json(
        destination / "shape_warmup_report.json",
        {
            "report_version": "1.0",
            "development_only": True,
            "confirmatory_results_observed": False,
            "warmup_count": len(warmups),
            "warmups": warmups,
            "valid": all(
                item["forced_prefix_greedy_exact"] for item in warmups
            ),
        },
    )

    template_report = evaluate_template_qualification(
        manifest=template_manifest,
        records=template_records,
        thresholds=config["template_qualification"],
        bootstrap_seed=int(config["seeds"]["bootstrap"]),
    )
    control_report = evaluate_control_records(
        manifest=control_manifest,
        records=control_records,
        minimum_accuracy_per_task=float(
            config["controls"]["minimum_baseline_accuracy_per_task"]
        ),
        minimum_format_valid_rate=float(
            config["controls"]["minimum_baseline_format_valid_rate"]
        ),
        use_rotation_marginalized_semantic_controls=bool(
            config["controls"].get(
                "use_rotation_marginalized_semantic_controls",
                False,
            )
        ),
    )
    power_report = simulate_power(
        config,
        nuisance_standard_deviation=template_report[
            "development_nuisance"
        ]["standard_deviation"],
    )
    _write_json(
        destination / "template_qualification_report.json",
        template_report,
    )
    _write_json(
        destination / "control_baseline_report.json",
        control_report,
    )
    _write_json(
        destination / "power_simulation_report.json",
        power_report,
    )
    candidate = _build_candidate(
        root=root,
        destination=destination,
        config_path=formal_config_path,
        config=config,
        gate_name=gate_name,
        model_id=model_config.model_id,
        prerequisite_report=prerequisite_report,
        template_report=template_report,
        control_report=control_report,
        power_report=power_report,
    )
    candidate_path = destination / "preregistration_candidate.json"
    _write_json(candidate_path, candidate)
    verification = verify_preregistration_candidate(
        candidate_path,
        project_root=root,
    )
    _write_json(
        destination / "preregistration_verification.json",
        verification,
    )

    diagnostic_complete = bool(
        template_report["valid"]
        and control_report["valid"]
        and power_report["valid"]
        and verification["valid"]
    )
    summary = {
        "summary_version": "1.0",
        "gate": gate_name,
        "development_only": True,
        "confirmatory_results_observed": False,
        "core_set_generated": False,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "model_id": model_config.model_id,
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "formal_history_mode": config["history_protocol"]["mode"],
        "template_semantic_case_count": template_manifest[
            "semantic_case_count"
        ],
        "template_trial_count": template_manifest["trial_count"],
        "control_trial_count": control_manifest["trial_count"],
        "shape_warmup_count": len(warmups),
        "template_qualification_passed": template_report[
            "template_qualification_passed"
        ],
        "control_baseline_passed": control_report[
            "control_baseline_passed"
        ],
        "power_gate_passed": power_report["power_gate_passed"],
        "planned_factorial_group_count": config["core_design"][
            "factorial_group_count"
        ],
        "candidate_digest_sha256": candidate[
            "candidate_digest_sha256"
        ],
        "payload_root_digest_sha256": candidate[
            "payload_root_digest_sha256"
        ],
        "freeze_candidate_ready": candidate[
            "eligible_for_human_freeze"
        ],
        "human_checksum_confirmation_required": True,
        "route_decision": (
            "review_preregistration_checksum"
            if candidate["eligible_for_human_freeze"]
            else "hold_and_revise_without_confirmatory_results"
        ),
        "reports": [
            "prerequisite_evidence.json",
            "answer_interface_report.json",
            "template_qualification_manifest.json",
            "raw_template_qualification.jsonl",
            "template_qualification_report.json",
            "control_manifest.json",
            "raw_control_baseline.jsonl",
            "control_baseline_report.json",
            "shape_warmup_report.json",
            "power_simulation_report.json",
            "preregistration_candidate.json",
            "preregistration_verification.json",
        ],
        "valid": diagnostic_complete,
    }
    _write_json(destination / "summary.json", summary)
    return summary
