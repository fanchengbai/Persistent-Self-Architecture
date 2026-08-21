from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import sha256_json
from psa.self_model.d5c_mechanism_runtime import (
    D5CCouplingRequest,
    D5CSyntheticProbe,
    FIXTURES,
    TARGET_LAYER_INDEX,
    TARGET_RESIDUAL_RMS_RATIO,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
)


CORE_VERSION = "0.1-d5c-p1-engineering-core"
ROUTE_ORDER = (
    "original_before",
    "patched_off_before",
    "patched_active",
    "original_after_active",
    "patched_off_after",
    "patched_zero_after",
)
CONTROL_COMPARISONS = (
    ("original_before", "patched_off_before"),
    ("original_before", "original_after_active"),
    ("patched_off_before", "patched_off_after"),
    ("original_after_active", "patched_off_after"),
    ("patched_off_after", "patched_zero_after"),
)
ACTIVE_COMPARISONS = (
    ("patched_active", "original_before"),
    ("patched_active", "original_after_active"),
)
MANAGED_NAMES = (CALLBACK_ATTRIBUTE, *TARGET_METHODS)


def _tensor_payload(value: Any, torch: Any) -> dict[str, Any]:
    if hasattr(value, "values"):
        return {
            "kind": "offline_tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": sha256_json(value.values),
        }
    detached = value.detach().contiguous()
    byte_view = detached.cpu().view(torch.uint8)
    return {
        "kind": "tensor",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }


def _value_payload(value: Any, torch: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _tensor_payload(value, torch)
    if isinstance(value, (list, tuple)):
        return [_value_payload(item, torch) for item in value]
    if isinstance(value, dict):
        return {str(key): _value_payload(value[key], torch) for key in sorted(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported D5C-P1 output value {type(value).__name__}")


def _all_finite(value: Any, torch: Any) -> bool:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item, torch) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item, torch) for item in value.values())
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _output_record(output: tuple[Any, Any], torch: Any) -> dict[str, Any]:
    payload = {
        "logits": _value_payload(output[0], torch),
        "state": _value_payload(output[1], torch),
    }
    return {
        "sha256": sha256_json(payload),
        "finite": _all_finite(output, torch),
        "payload": payload,
    }


def _invoke(
    *,
    route: str,
    fixture: Mapping[str, Any],
    base_model: Any,
    runtime: Any,
    probe: D5CSyntheticProbe,
    torch: Any,
    state_factory: Callable[[], Any],
) -> tuple[Any, Any]:
    context_factory = getattr(torch, "inference_mode", None)
    context = context_factory() if callable(context_factory) else nullcontext()
    with context:
        tokens = list(fixture["token_ids"])
        state = state_factory()
        full_output = bool(fixture["full_output"])
        if route in {"original_before", "original_after_active"}:
            return base_model.forward(tokens, state, full_output)
        if route in {"patched_off_before", "patched_off_after"}:
            coupling = D5CCouplingRequest(False, 0.0, None)
        elif route == "patched_zero_after":
            coupling = D5CCouplingRequest(True, 0.0, None)
        elif route == "patched_active":
            coupling = D5CCouplingRequest(True, 1.0, probe)
        else:
            raise ValueError("unknown D5C-P1 route")
        return runtime.forward(
            tokens, state, full_output, coupling=coupling
        )


def execute_d5c_p1_engineering_core(
    *,
    base_model: Any,
    active_runtime: Any,
    probe: D5CSyntheticProbe,
    torch: Any,
    state_factory: Callable[[], Any] = lambda: None,
    fixtures: Sequence[Mapping[str, Any]] = FIXTURES,
) -> dict[str, Any]:
    fixture_reports: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    active_records: list[dict[str, Any]] = []
    cleanup_evidence_count = 0
    model_forward_calls = 0
    for fixture in fixtures:
        outputs: dict[str, tuple[Any, Any]] = {}
        calls: list[dict[str, Any]] = []
        for position, route in enumerate(ROUTE_ORDER, start=1):
            output = _invoke(
                route=route,
                fixture=fixture,
                base_model=base_model,
                runtime=active_runtime,
                probe=probe,
                torch=torch,
                state_factory=state_factory,
            )
            record = _output_record(output, torch)
            calls.append(
                {
                    "call_id": f"{fixture['fixture_id']}-{position}",
                    "route": route,
                    "position": position,
                    "output_sha256": record["sha256"],
                    "finite": record["finite"],
                }
            )
            outputs[route] = output
            model_forward_calls += 1
            if route.startswith("patched_"):
                remaining = sorted(
                    name for name in MANAGED_NAMES if name in base_model.__dict__
                )
                if remaining:
                    raise RuntimeError(
                        "D5C-P1 runtime returned with managed instance bindings"
                    )
                cleanup_evidence_count += 1

        fingerprints = {
            route: _output_record(output, torch)["sha256"]
            for route, output in outputs.items()
        }
        fixture_controls = [
            {
                "left": left,
                "right": right,
                "exact": fingerprints[left] == fingerprints[right],
            }
            for left, right in CONTROL_COMPARISONS
        ]
        fixture_active = [
            {
                "left": left,
                "right": right,
                "different": fingerprints[left] != fingerprints[right],
            }
            for left, right in ACTIVE_COMPARISONS
        ]
        control_records.extend(fixture_controls)
        active_records.extend(fixture_active)
        fixture_reports.append(
            {
                "fixture": dict(fixture),
                "calls": calls,
                "control_comparisons": fixture_controls,
                "active_comparisons": fixture_active,
            }
        )

    checks = {
        "model_forward_calls_exact": model_forward_calls == 12,
        "wrapped_forward_calls_exact": getattr(active_runtime, "execution_count", None) == 8,
        "control_comparison_count_exact": len(control_records) == 10,
        "active_comparison_count_exact": len(active_records) == 4,
        "all_outputs_finite": all(
            call["finite"] for report in fixture_reports for call in report["calls"]
        ),
        "original_before_after_active_exact": all(
            item["exact"]
            for item in control_records
            if {item["left"], item["right"]}
            == {"original_before", "original_after_active"}
        ),
        "all_control_comparisons_exact": all(item["exact"] for item in control_records),
        "active_differs_from_before_and_after": all(
            item["different"] for item in active_records
        ),
        "temporary_bindings_absent_after_every_wrapped_call": cleanup_evidence_count == 8,
        "callback_invocations_exact": probe.invocation_count == 64,
        "probe_applications_exact": probe.application_count == 2,
    }
    return {
        "core_version": CORE_VERSION,
        "status": (
            "d5c_p1_engineering_validation_passed"
            if all(checks.values())
            else "d5c_p1_engineering_validation_failed"
        ),
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "model_forward_calls": model_forward_calls,
            "wrapped_forward_calls": getattr(active_runtime, "execution_count", None),
            "control_comparisons": len(control_records),
            "active_comparisons": len(active_records),
            "cleanup_evidence_records": cleanup_evidence_count,
            "callback_invocations": probe.invocation_count,
            "probe_applications": probe.application_count,
        },
        "fixtures": fixture_reports,
        "interpretation": {
            "kind": "post_patch_engineering_validation_only",
            "historical_d5c_result_changed": False,
            "self_effect_claim": False,
            "real_self_projection": False,
            "target_layer_index_zero_based": TARGET_LAYER_INDEX,
            "target_residual_rms_ratio": TARGET_RESIDUAL_RMS_RATIO,
        },
    }
