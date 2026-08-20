from __future__ import annotations

from contextlib import nullcontext
import copy
from itertools import combinations, product
from typing import Any, Mapping

from psa.self_model.d4a_failure_diagnostic_runtime import (
    _flatten_tensors,
    _pair_record,
    _tensor_record,
)
from psa.self_model.d4b_steady_state_off_design import (
    PRECONDITION_ORDER,
    ROUTES,
    SCORED_ROUNDS,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
)


D4B_RUNTIME_VERSION = "0.1-d4b-steady-state-off-runtime"
D4B_PREFIX_TOKEN_IDS = [187, 931]
D4B_TARGET_TOKEN_IDS = [2764]


def _counter(route: Any, name: str) -> int:
    value = getattr(route, name, None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"D4B route must expose integer {name}")
    return value


def _validate_route_interfaces(
    *, base_model: Any, off_g1: Any, g0: Any, off_g2: Any
) -> None:
    for label, route in (
        ("original_baseline", base_model),
        ("off_g1_passthrough", off_g1),
        ("g0_recompiled_unmodified", g0),
        ("off_g2_instrumented", off_g2),
    ):
        if not callable(getattr(route, "forward", None)):
            raise TypeError(f"D4B route {label} must expose forward")
    _counter(off_g1, "delegation_count")
    _counter(g0, "execution_count")
    _counter(off_g2, "execution_count")
    for label, route in (
        ("off_g1_passthrough", off_g1),
        ("g0_recompiled_unmodified", g0),
        ("off_g2_instrumented", off_g2),
    ):
        if getattr(route, "callback_call_count", None) != 0:
            raise PermissionError(f"D4B route {label} has a callback")
        if getattr(route, "self_projection_constructed", None) is not False:
            raise PermissionError(f"D4B route {label} has a Self projection")
    instance_dict = getattr(base_model, "__dict__", None)
    if not isinstance(instance_dict, dict):
        raise TypeError("D4B base model must expose a mutable instance dictionary")
    if any(name in instance_dict for name in (*TARGET_METHODS, CALLBACK_ATTRIBUTE)):
        raise RuntimeError("D4B base model has conflicting instance overrides")


def _invoke(route: Any, token_ids: list[int], torch: Any) -> tuple[Any, Any]:
    inference_mode = getattr(torch, "inference_mode", None)
    context = inference_mode() if callable(inference_mode) else nullcontext()
    with context:
        output = route.forward(list(token_ids), None, False)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("D4B routes must return a logits/state pair")
    return output


def _call_record(
    *,
    call_index: int,
    call_id: str,
    phase: str,
    route: str,
    token_ids: list[int],
    scored: bool,
    logits: Any,
    state: Any,
    torch: Any,
    round_index: int | None = None,
    order_position: int | None = None,
) -> dict[str, Any]:
    components = [
        {"path": path, **_tensor_record(tensor, torch)}
        for path, tensor in _flatten_tensors(state)
    ]
    return {
        "call_index": call_index,
        "call_id": call_id,
        "phase": phase,
        "route": route,
        "round_index": round_index,
        "order_position": order_position,
        "token_ids": list(token_ids),
        "state_input": "none",
        "full_output": False,
        "scored": scored,
        "output_recorded": True,
        "logits": _tensor_record(logits, torch),
        "state": {
            "component_count": len(components),
            "components": components,
        },
    }


def _bindings_restored(base_model: Any) -> bool:
    instance_dict = getattr(base_model, "__dict__", {})
    return all(
        name not in instance_dict for name in (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
    )


def execute_d4b_fake_or_future_authorized_core(
    *,
    base_model: Any,
    off_g1: Any,
    g0: Any,
    off_g2: Any,
    torch: Any,
) -> dict[str, Any]:
    """Execute the fixed D4B core; this function never loads a model."""
    _validate_route_interfaces(
        base_model=base_model, off_g1=off_g1, g0=g0, off_g2=off_g2
    )
    routes = {
        "original_baseline": base_model,
        "off_g1_passthrough": off_g1,
        "g0_recompiled_unmodified": g0,
        "off_g2_instrumented": off_g2,
    }
    g1_before = _counter(off_g1, "delegation_count")
    g0_before = _counter(g0, "execution_count")
    g2_before = _counter(off_g2, "execution_count")
    calls: list[dict[str, Any]] = []
    outputs: list[tuple[Any, Any]] = []

    def record(
        *,
        phase: str,
        route: str,
        token_ids: list[int],
        scored: bool,
        round_index: int | None = None,
        order_position: int | None = None,
    ) -> None:
        output = _invoke(routes[route], token_ids, torch)
        call_index = len(calls) + 1
        call_id = f"call-{call_index:02d}-{phase}-{route}"
        calls.append(
            _call_record(
                call_index=call_index,
                call_id=call_id,
                phase=phase,
                route=route,
                token_ids=token_ids,
                scored=scored,
                logits=output[0],
                state=output[1],
                torch=torch,
                round_index=round_index,
                order_position=order_position,
            )
        )
        outputs.append(output)

    record(
        phase="prefix_snapshot",
        route="original_baseline",
        token_ids=D4B_PREFIX_TOKEN_IDS,
        scored=False,
    )
    for position, route in enumerate(PRECONDITION_ORDER, start=1):
        record(
            phase="fixed_preconditioning",
            route=route,
            token_ids=D4B_TARGET_TOKEN_IDS,
            scored=False,
            order_position=position,
        )
    for round_index, round_routes in enumerate(SCORED_ROUNDS, start=1):
        for order_position, route in enumerate(round_routes, start=1):
            record(
                phase="scored_latin",
                route=route,
                token_ids=D4B_TARGET_TOKEN_IDS,
                scored=True,
                round_index=round_index,
                order_position=order_position,
            )

    scored_indexes_by_route = {
        route: [
            index
            for index, call in enumerate(calls)
            if call["scored"] and call["route"] == route
        ]
        for route in ROUTES
    }
    within_route = []
    for indexes in scored_indexes_by_route.values():
        for left, right in combinations(indexes, 2):
            within_route.append(
                _pair_record(calls[left], calls[right], outputs[left], outputs[right], torch)
            )
    cross_route = []
    for left_route, right_route in combinations(ROUTES, 2):
        for left, right in product(
            scored_indexes_by_route[left_route], scored_indexes_by_route[right_route]
        ):
            cross_route.append(
                _pair_record(calls[left], calls[right], outputs[left], outputs[right], torch)
            )
    comparisons = within_route + cross_route
    all_compatible = all(
        item["logits"]["valid"]
        and item["state"]["shape_dtype_device_compatible"]
        for item in comparisons
    )
    all_exact = all(item["all_torch_equal"] for item in comparisons)
    calls_by_phase = {
        phase: [call for call in calls if call["phase"] == phase]
        for phase in ("prefix_snapshot", "fixed_preconditioning", "scored_latin")
    }
    checks = {
        "twenty_one_calls_recorded": len(calls) == 21,
        "prefix_recorded_not_scored": len(calls_by_phase["prefix_snapshot"]) == 1
        and calls[0]["route"] == "original_baseline"
        and calls[0]["token_ids"] == D4B_PREFIX_TOKEN_IDS
        and calls[0]["scored"] is False,
        "fixed_preconditioning_recorded_not_scored": [
            call["route"] for call in calls_by_phase["fixed_preconditioning"]
        ]
        == PRECONDITION_ORDER
        and all(not call["scored"] for call in calls_by_phase["fixed_preconditioning"]),
        "sixteen_scored_calls_recorded": len(calls_by_phase["scored_latin"]) == 16,
        "four_scored_calls_per_route": all(
            len(indexes) == 4 for indexes in scored_indexes_by_route.values()
        ),
        "each_route_occupies_each_scored_position_once": all(
            sorted(calls[index]["order_position"] for index in indexes)
            == [1, 2, 3, 4]
            for indexes in scored_indexes_by_route.values()
        ),
        "all_target_calls_use_frozen_fixture": all(
            call["token_ids"] == D4B_TARGET_TOKEN_IDS
            and call["state_input"] == "none"
            and call["full_output"] is False
            for call in calls[1:]
        ),
        "all_outputs_inventoried": all(
            call["output_recorded"]
            and call["logits"]["sha256"]
            and call["state"]["component_count"] == len(call["state"]["components"])
            and all(component["sha256"] for component in call["state"]["components"])
            for call in calls
        ),
        "all_within_route_pairs_recorded": len(within_route) == 24,
        "all_cross_route_pairs_recorded": len(cross_route) == 96,
        "all_pair_tensors_compatible": all_compatible,
        "all_scored_pairs_exact": all_exact,
        "off_g1_executed_exactly_five_times": (
            _counter(off_g1, "delegation_count") - g1_before == 5
        ),
        "g0_executed_exactly_five_times": _counter(g0, "execution_count") - g0_before
        == 5,
        "off_g2_executed_exactly_five_times": (
            _counter(off_g2, "execution_count") - g2_before == 5
        ),
        "callbacks_remain_zero": all(
            getattr(route, "callback_call_count", None) == 0
            for route in (off_g1, g0, off_g2)
        ),
        "self_projection_not_constructed": all(
            getattr(route, "self_projection_constructed", None) is False
            for route in (off_g1, g0, off_g2)
        ),
        "temporary_bindings_restored": _bindings_restored(base_model),
    }
    valid = all(checks.values())
    return {
        "runtime_version": D4B_RUNTIME_VERSION,
        "status": (
            "d4b_fake_core_exact_verified" if valid else "d4b_fake_core_failed_stop"
        ),
        "valid": valid,
        "development_only": True,
        "fake_first": True,
        "d4_status_changed": False,
        "d5_authorized": False,
        "pass_effect": (
            "runtime_core_verification_only" if valid else "stop_without_rerun"
        ),
        "future_authorized_real_d4b_pass_effect": "d5_review_candidate_only",
        "schedule": {
            "prefix": list(D4B_PREFIX_TOKEN_IDS),
            "preconditioning": list(PRECONDITION_ORDER),
            "scored_rounds": copy.deepcopy(SCORED_ROUNDS),
            "adaptive_or_extra_calls_allowed": False,
        },
        "calls": calls,
        "comparisons": {
            "within_route": within_route,
            "cross_route": cross_route,
        },
        "checks": checks,
        "safety": {
            "real_model_entry_implemented": False,
            "real_model_executed": False,
            "machine_authorization_created": False,
            "execution_claim_created": False,
            "active_injection_executed": False,
            "self_projection_constructed": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
