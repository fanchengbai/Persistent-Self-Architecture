from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from psa.artifacts import canonical_json_bytes, sha256_file
from psa.model.rwkv7 import (
    RWKV7Adapter,
    _flatten_state,
    clone_state,
    compare_states,
    compare_tensors,
    inventory_state,
    load_model_config,
)
from psa.state.checkpoint import (
    _apply_determinism_policy,
    _validate_acceptance_policy,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("state operations gate config must be an object")
    return payload


def official_reset_state() -> None:
    """Return the official RWKV package reset sentinel."""
    return None


def swap_full_state(donor_state: Any) -> Any:
    """Create a deep, independent full-state transplant from a donor."""
    return clone_state(donor_state)


def randomize_state_matched(
    source_state: Any,
    torch: Any,
    *,
    seed: int,
) -> list[Any]:
    """Generate per-component zero-mean Gaussian state matched on L2 scale."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("random state seed must be a non-negative integer")
    flattened = list(_flatten_state(source_state))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    randomized = []
    for index, (path, source_tensor) in enumerate(flattened):
        if path != f"state[{index}]":
            raise TypeError("matched random currently requires a flat state list")
        source_float = source_tensor.detach().float().cpu()
        noise = torch.randn(
            source_float.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        noise = noise - noise.mean()
        noise_norm = torch.linalg.vector_norm(noise)
        source_norm = torch.linalg.vector_norm(source_float)
        if float(source_norm.item()) == 0.0:
            matched = torch.zeros_like(source_float)
        elif float(noise_norm.item()) == 0.0:
            raise RuntimeError(f"random noise has zero norm at {path}")
        else:
            matched = noise * (source_norm / noise_norm)
        randomized.append(
            matched.to(dtype=source_tensor.dtype, device=source_tensor.device)
        )
    return randomized


def matched_scale_report(
    source_state: Any,
    randomized_state: Any,
    torch: Any,
) -> dict[str, Any]:
    source_items = list(_flatten_state(source_state))
    random_items = list(_flatten_state(randomized_state))
    if [path for path, _ in source_items] != [
        path for path, _ in random_items
    ]:
        return {
            "compatible": False,
            "component_count": 0,
            "all_finite": False,
            "max_relative_l2_error": None,
            "components": [],
        }
    components = []
    all_finite = True
    for (path, source), (_, randomized) in zip(source_items, random_items):
        compatible = bool(
            tuple(source.shape) == tuple(randomized.shape)
            and source.dtype == randomized.dtype
            and source.device == randomized.device
        )
        source_float = source.detach().float()
        random_float = randomized.detach().float()
        source_l2 = float(torch.linalg.vector_norm(source_float).item())
        random_l2 = float(torch.linalg.vector_norm(random_float).item())
        relative_error = (
            abs(random_l2 - source_l2) / source_l2
            if source_l2 > 0
            else (0.0 if random_l2 == 0 else None)
        )
        finite = bool(torch.isfinite(random_float).all().item())
        all_finite = all_finite and finite
        components.append(
            {
                "path": path,
                "compatible": compatible,
                "finite": finite,
                "source_l2_norm": source_l2,
                "random_l2_norm": random_l2,
                "relative_l2_error": relative_error,
                "random_mean": float(random_float.mean().item()),
            }
        )
    errors = [
        item["relative_l2_error"]
        for item in components
        if item["relative_l2_error"] is not None
    ]
    return {
        "compatible": all(item["compatible"] for item in components),
        "component_count": len(components),
        "all_finite": all_finite,
        "max_relative_l2_error": max(errors, default=0.0),
        "components": components,
    }


def diff_states(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    left_items = list(_flatten_state(left))
    right_items = list(_flatten_state(right))
    left_paths = [path for path, _ in left_items]
    right_paths = [path for path, _ in right_items]
    if left_paths != right_paths:
        return {
            "compatible": False,
            "all_finite": False,
            "component_count": 0,
            "different_component_count": 0,
            "components": [],
        }

    components = []
    all_finite = True
    for (path, left_tensor), (_, right_tensor) in zip(left_items, right_items):
        compatible = bool(
            tuple(left_tensor.shape) == tuple(right_tensor.shape)
            and left_tensor.dtype == right_tensor.dtype
        )
        if not compatible:
            components.append(
                {
                    "path": path,
                    "compatible": False,
                    "exact": False,
                    "l1_norm": None,
                    "l2_norm": None,
                    "linf_norm": None,
                    "cosine_similarity": None,
                    "rms_ratio_right_over_left": None,
                    "finite": False,
                }
            )
            all_finite = False
            continue

        left_float = left_tensor.detach().float().reshape(-1)
        right_float = right_tensor.detach().float().reshape(-1)
        delta = right_float - left_float
        absolute = delta.abs()
        left_l2 = torch.linalg.vector_norm(left_float)
        right_l2 = torch.linalg.vector_norm(right_float)
        left_rms = torch.sqrt(torch.mean(left_float.square()))
        right_rms = torch.sqrt(torch.mean(right_float.square()))
        finite = bool(
            torch.isfinite(left_float).all().item()
            and torch.isfinite(right_float).all().item()
            and torch.isfinite(delta).all().item()
        )
        all_finite = all_finite and finite
        cosine = None
        if float(left_l2.item()) > 0 and float(right_l2.item()) > 0:
            cosine = float(
                torch.dot(left_float, right_float).item()
                / (left_l2.item() * right_l2.item())
            )
        rms_ratio = None
        if float(left_rms.item()) > 0:
            rms_ratio = float(right_rms.item() / left_rms.item())
        components.append(
            {
                "path": path,
                "compatible": True,
                "exact": bool(torch.equal(left_tensor, right_tensor)),
                "l1_norm": float(absolute.sum().item()),
                "l2_norm": float(torch.linalg.vector_norm(delta).item()),
                "linf_norm": float(absolute.max().item()),
                "cosine_similarity": cosine,
                "rms_ratio_right_over_left": rms_ratio,
                "finite": finite,
            }
        )

    compatible = all(item["compatible"] for item in components)
    finite_linf = [
        item["linf_norm"]
        for item in components
        if item["linf_norm"] is not None
    ]
    return {
        "compatible": compatible,
        "all_finite": all_finite,
        "component_count": len(components),
        "different_component_count": sum(
            not item["exact"] for item in components
        ),
        "max_linf_norm": max(finite_linf, default=0.0),
        "components": components,
    }


def _comparison_passes(
    logits_comparison: dict[str, Any],
    state_comparison: dict[str, Any],
    *,
    top1_match: bool,
    acceptance: dict[str, Any],
) -> bool:
    logits_error = logits_comparison["max_abs_error"]
    state_error = state_comparison["max_abs_error"]
    return bool(
        logits_comparison["compatible"]
        and state_comparison["compatible"]
        and logits_error is not None
        and state_error is not None
        and logits_error <= acceptance["logits_max_abs_error"]
        and state_error <= acceptance["state_max_abs_error"]
        and top1_match
    )


def _top1_id(logits: Any) -> int:
    return int(logits.detach().float().argmax().item())


def _repeat_from_state(
    adapter: RWKV7Adapter,
    tokens: list[int],
    source_state: Any,
    *,
    repeat_count: int,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    baseline_logits, baseline_state = adapter.forward(
        tokens, clone_state(source_state)
    )
    baseline_logits = baseline_logits.detach().clone()
    baseline_state = clone_state(baseline_state)
    baseline_top1 = _top1_id(baseline_logits)
    trials = []
    exact_count = 0
    tolerance_count = 0
    worst_logits_error = 0.0
    worst_state_error = 0.0
    for index in range(repeat_count):
        logits, state = adapter.forward(tokens, clone_state(source_state))
        logits_comparison = compare_tensors(
            logits, baseline_logits, adapter.torch
        )
        state_comparison = compare_states(
            state, baseline_state, adapter.torch
        )
        top1_match = _top1_id(logits) == baseline_top1
        exact = bool(
            logits_comparison["exact"] and state_comparison["exact"]
        )
        passes = _comparison_passes(
            logits_comparison,
            state_comparison,
            top1_match=top1_match,
            acceptance=acceptance,
        )
        exact_count += int(exact)
        tolerance_count += int(passes)
        worst_logits_error = max(
            worst_logits_error,
            float(logits_comparison["max_abs_error"] or 0.0),
        )
        worst_state_error = max(
            worst_state_error,
            float(state_comparison["max_abs_error"] or 0.0),
        )
        trials.append(
            {
                "trial": index + 1,
                "exact": exact,
                "within_tolerance": passes,
                "top1_match": top1_match,
                "logits_max_abs_error": logits_comparison[
                    "max_abs_error"
                ],
                "state_max_abs_error": state_comparison[
                    "max_abs_error"
                ],
            }
        )
    return {
        "repeat_count": repeat_count,
        "exact_count": exact_count,
        "tolerance_pass_count": tolerance_count,
        "top1_match_count": sum(item["top1_match"] for item in trials),
        "worst_logits_max_abs_error": worst_logits_error,
        "worst_state_max_abs_error": worst_state_error,
        "trials": trials,
        "valid": tolerance_count == repeat_count,
    }


def run_state_operations_gate(
    *,
    config_path: str | Path,
    gate_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    started_at = _utc_now()
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    gate_config_path = Path(gate_config_path).resolve()
    gate_config = _read_config(gate_config_path)
    gate_name = gate_config.get("gate")
    if (
        gate_config.get("gate_version") != "0.1"
        or gate_name
        not in {
            "impl2b_state_operations",
            "impl3n_g1h_2_9b_state_operations",
        }
        or gate_config.get("development_only") is not True
    ):
        raise ValueError("unsupported state operations gate config")
    repeat_count = gate_config.get("repeat_count")
    if not isinstance(repeat_count, int) or repeat_count < 10:
        raise ValueError("repeat_count must be at least 10")
    _apply_determinism_policy(gate_config.get("determinism"))
    acceptance = _validate_acceptance_policy(
        gate_config.get("acceptance")
    )

    model_config = load_model_config(config_path, root, verify_files=True)
    adapter = RWKV7Adapter.load(model_config)
    base_text = gate_config.get("base_prefix_text")
    marker_a = gate_config.get("branch_a_marker")
    marker_b = gate_config.get("branch_b_marker")
    suffix_text = gate_config.get("suffix_text")
    if not all(
        isinstance(value, str) and value
        for value in (base_text, marker_a, marker_b, suffix_text)
    ):
        raise ValueError("gate probe text fields must be non-empty strings")
    base_tokens = adapter.encode(base_text)
    marker_a_tokens = adapter.encode(marker_a)
    marker_b_tokens = adapter.encode(marker_b)
    if len(marker_a_tokens) != len(marker_b_tokens):
        raise ValueError("branch markers must have equal token counts")
    prefix_a_tokens = base_tokens + marker_a_tokens
    prefix_b_tokens = base_tokens + marker_b_tokens
    suffix_tokens = adapter.encode(suffix_text)

    _, state_a = adapter.forward(prefix_a_tokens, official_reset_state())
    _, state_b = adapter.forward(prefix_b_tokens, official_reset_state())
    inventory_a_before = inventory_state(state_a, adapter.torch)
    inventory_b_before = inventory_state(state_b, adapter.torch)
    branch_diff = diff_states(state_a, state_b, adapter.torch)
    self_diff = diff_states(state_a, clone_state(state_a), adapter.torch)
    branch_diff.update(
        {
            "report_version": "0.1",
            "created_at_utc": _utc_now(),
            "development_only": True,
            "left_state_digest_sha256": inventory_a_before[
                "state_digest_sha256"
            ],
            "right_state_digest_sha256": inventory_b_before[
                "state_digest_sha256"
            ],
            "prefix_token_count": len(prefix_a_tokens),
            "self_diff_validation": {
                "compatible": self_diff["compatible"],
                "all_finite": self_diff["all_finite"],
                "different_component_count": self_diff[
                    "different_component_count"
                ],
                "max_linf_norm": self_diff["max_linf_norm"],
            },
        }
    )

    reset_probe = _repeat_from_state(
        adapter,
        suffix_tokens,
        official_reset_state(),
        repeat_count=repeat_count,
        acceptance=acceptance,
    )
    reset_probe.update(
        {
            "report_version": "0.1",
            "created_at_utc": _utc_now(),
            "development_only": True,
            "reset_representation": "None",
            "semantics": "official rwkv.model.RWKV.forward state=None",
        }
    )

    swap_a_into_b = _repeat_from_state(
        adapter,
        suffix_tokens,
        swap_full_state(state_a),
        repeat_count=repeat_count,
        acceptance=acceptance,
    )
    swap_b_into_a = _repeat_from_state(
        adapter,
        suffix_tokens,
        swap_full_state(state_b),
        repeat_count=repeat_count,
        acceptance=acceptance,
    )
    inventory_a_after = inventory_state(state_a, adapter.torch)
    inventory_b_after = inventory_state(state_b, adapter.torch)
    sources_immutable = bool(
        inventory_a_before["state_digest_sha256"]
        == inventory_a_after["state_digest_sha256"]
        and inventory_b_before["state_digest_sha256"]
        == inventory_b_after["state_digest_sha256"]
    )
    swap_probe = {
        "report_version": "0.1",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "operation": "full_native_state_swap",
        "source_states_immutable": sources_immutable,
        "donor_a_state_digest_sha256": inventory_a_before[
            "state_digest_sha256"
        ],
        "donor_b_state_digest_sha256": inventory_b_before[
            "state_digest_sha256"
        ],
        "a_into_b": swap_a_into_b,
        "b_into_a": swap_b_into_a,
        "valid": bool(
            sources_immutable
            and swap_a_into_b["valid"]
            and swap_b_into_a["valid"]
        ),
    }

    tokenizer_valid = bool(
        adapter.decode(base_tokens) == base_text
        and adapter.decode(marker_a_tokens) == marker_a
        and adapter.decode(marker_b_tokens) == marker_b
        and adapter.decode(suffix_tokens) == suffix_text
        and adapter.decode(prefix_a_tokens) == base_text + marker_a
        and adapter.decode(prefix_b_tokens) == base_text + marker_b
    )
    diff_valid = bool(
        branch_diff["compatible"]
        and branch_diff["all_finite"]
        and branch_diff["component_count"]
        == model_config.architecture_hint["n_layer"] * 3
        and branch_diff["different_component_count"] > 0
        and self_diff["compatible"]
        and self_diff["all_finite"]
        and self_diff["different_component_count"] == 0
        and self_diff["max_linf_norm"] == 0.0
    )
    reset_probe["valid"] = bool(reset_probe["valid"] and tokenizer_valid)
    summary = {
        "gate": gate_name,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "development_only": True,
        "model_id": model_config.model_id,
        "gate_config_sha256": sha256_file(gate_config_path),
        "tokenizer_roundtrip_valid": tokenizer_valid,
        "state_diff_valid": diff_valid,
        "reset_valid": reset_probe["valid"],
        "swap_valid": swap_probe["valid"],
        "source_states_immutable": sources_immutable,
        "component_count": branch_diff["component_count"],
        "different_component_count": branch_diff[
            "different_component_count"
        ],
        "valid": bool(
            tokenizer_valid
            and diff_valid
            and reset_probe["valid"]
            and swap_probe["valid"]
        ),
        "reports": [
            "state_diff_report.json",
            "reset_validation.json",
            "swap_validation.json",
        ],
    }
    _write_json(destination / "state_diff_report.json", branch_diff)
    _write_json(destination / "reset_validation.json", reset_probe)
    _write_json(destination / "swap_validation.json", swap_probe)
    _write_json(destination / "summary.json", summary)
    return summary


def run_random_state_gate(
    *,
    config_path: str | Path,
    gate_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    started_at = _utc_now()
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    gate_config_path = Path(gate_config_path).resolve()
    gate_config = _read_config(gate_config_path)
    gate_name = gate_config.get("gate")
    if (
        gate_config.get("gate_version") != "0.1"
        or gate_name
        not in {
            "impl2c_random_matched",
            "impl3o_g1h_2_9b_random_matched",
        }
        or gate_config.get("development_only") is not True
    ):
        raise ValueError("unsupported random state gate config")
    repeat_count = gate_config.get("repeat_count")
    if not isinstance(repeat_count, int) or repeat_count < 10:
        raise ValueError("repeat_count must be at least 10")
    base_seed = gate_config.get("base_seed")
    alternate_seed = gate_config.get("alternate_seed")
    if (
        not isinstance(base_seed, int)
        or not isinstance(alternate_seed, int)
        or base_seed < 0
        or alternate_seed < 0
        or base_seed == alternate_seed
    ):
        raise ValueError("random state seeds must be distinct non-negative integers")
    maximum_scale_error = gate_config.get("max_relative_l2_error")
    if (
        not isinstance(maximum_scale_error, (int, float))
        or maximum_scale_error <= 0
    ):
        raise ValueError("max_relative_l2_error must be positive")
    _apply_determinism_policy(gate_config.get("determinism"))
    acceptance = _validate_acceptance_policy(
        gate_config.get("acceptance")
    )

    model_config = load_model_config(config_path, root, verify_files=True)
    adapter = RWKV7Adapter.load(model_config)
    prefix_text = gate_config.get("prefix_text")
    suffix_text = gate_config.get("suffix_text")
    if not all(
        isinstance(value, str) and value
        for value in (prefix_text, suffix_text)
    ):
        raise ValueError("gate probe text fields must be non-empty strings")
    prefix_tokens = adapter.encode(prefix_text)
    suffix_tokens = adapter.encode(suffix_text)
    tokenizer_valid = bool(
        adapter.decode(prefix_tokens) == prefix_text
        and adapter.decode(suffix_tokens) == suffix_text
    )
    _, source_state = adapter.forward(prefix_tokens, official_reset_state())
    source_before = inventory_state(source_state, adapter.torch)

    random_state = randomize_state_matched(
        source_state, adapter.torch, seed=base_seed
    )
    same_seed_state = randomize_state_matched(
        source_state, adapter.torch, seed=base_seed
    )
    alternate_state = randomize_state_matched(
        source_state, adapter.torch, seed=alternate_seed
    )
    same_seed_comparison = compare_states(
        random_state, same_seed_state, adapter.torch
    )
    source_random_diff = diff_states(
        source_state, random_state, adapter.torch
    )
    seed_diff = diff_states(
        random_state, alternate_state, adapter.torch
    )
    scale = matched_scale_report(
        source_state, random_state, adapter.torch
    )
    random_inventory = inventory_state(random_state, adapter.torch)
    alternate_inventory = inventory_state(alternate_state, adapter.torch)
    continuation = _repeat_from_state(
        adapter,
        suffix_tokens,
        random_state,
        repeat_count=repeat_count,
        acceptance=acceptance,
    )
    source_after = inventory_state(source_state, adapter.torch)
    source_immutable = bool(
        source_before["state_digest_sha256"]
        == source_after["state_digest_sha256"]
    )
    reproducible = bool(
        same_seed_comparison["compatible"]
        and same_seed_comparison["exact"]
    )
    distinct_seed_valid = bool(
        seed_diff["compatible"]
        and seed_diff["all_finite"]
        and seed_diff["different_component_count"] > 0
        and random_inventory["state_digest_sha256"]
        != alternate_inventory["state_digest_sha256"]
    )
    scale_valid = bool(
        scale["compatible"]
        and scale["all_finite"]
        and scale["component_count"]
        == model_config.architecture_hint["n_layer"] * 3
        and scale["max_relative_l2_error"] is not None
        and scale["max_relative_l2_error"] <= maximum_scale_error
    )
    report = {
        "report_version": "0.1",
        "created_at_utc": _utc_now(),
        "development_only": True,
        "operation": "random_matched",
        "method": "per-component centered Gaussian matched to source L2 norm",
        "base_seed": base_seed,
        "alternate_seed": alternate_seed,
        "source_state_digest_sha256": source_before[
            "state_digest_sha256"
        ],
        "random_state_digest_sha256": random_inventory[
            "state_digest_sha256"
        ],
        "alternate_state_digest_sha256": alternate_inventory[
            "state_digest_sha256"
        ],
        "source_state_immutable": source_immutable,
        "same_seed_bitwise_reproducible": reproducible,
        "different_seed_distinct": distinct_seed_valid,
        "source_random_different_component_count": source_random_diff[
            "different_component_count"
        ],
        "scale": scale,
        "continuation": continuation,
        "valid": bool(
            tokenizer_valid
            and source_immutable
            and reproducible
            and distinct_seed_valid
            and scale_valid
            and continuation["valid"]
        ),
    }
    summary = {
        "gate": gate_name,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "development_only": True,
        "model_id": model_config.model_id,
        "gate_config_sha256": sha256_file(gate_config_path),
        "tokenizer_roundtrip_valid": tokenizer_valid,
        "component_count": scale["component_count"],
        "source_state_immutable": source_immutable,
        "same_seed_bitwise_reproducible": reproducible,
        "different_seed_distinct": distinct_seed_valid,
        "scale_match_valid": scale_valid,
        "max_relative_l2_error": scale["max_relative_l2_error"],
        "continuation_valid": continuation["valid"],
        "valid": report["valid"],
        "reports": ["random_state_validation.json"],
    }
    _write_json(destination / "random_state_validation.json", report)
    _write_json(destination / "summary.json", summary)
    return summary
