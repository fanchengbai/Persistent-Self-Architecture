from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
import threading
import types
from typing import Any, Callable, Mapping, Sequence

from psa.self_model.d4a_failure_diagnostic_runtime import _pair_record
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    OfflineTensor,
    _output_digest,
)
from psa.self_model.d5c_mechanism_runtime import (
    D5CSyntheticProbe,
    FIXTURES,
    _NullContext,
    _call_record,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
    compile_instrumented_methods,
)


RUNTIME_VERSION = "0.1-coupling-d6c-persistent-mechanism"
ROUTES = ("persistent_off", "persistent_zero", "persistent_active_synthetic")
LATIN_ROUNDS = (
    ROUTES,
    (ROUTES[1], ROUTES[2], ROUTES[0]),
    (ROUTES[2], ROUTES[0], ROUTES[1]),
    ROUTES,
)
N_LAYER = 32
HIDDEN_DIMENSION = 2560
STATE_COMPONENTS = 96
TARGET_LAYER_INDEX = 15
MODEL_FORWARD_CALLS_PER_FIXTURE = 13
MODEL_FORWARD_CALLS_TOTAL = 26
ACTIVE_FORWARD_CALLS_TOTAL = 8
ACTIVE_CALLBACK_CALLS_TOTAL = 256
ACTIVE_PROBE_APPLICATIONS_TOTAL = 8
MANAGED_NAMES = (CALLBACK_ATTRIBUTE, *TARGET_METHODS)
_NO_REQUEST = object()


@dataclass(frozen=True)
class D6CCouplingRequest:
    mode: str
    enabled: bool
    scale: float
    callback: D5CSyntheticProbe | None

    def validate(self) -> None:
        expected = {
            ROUTES[0]: (False, 0.0, False),
            ROUTES[1]: (True, 0.0, False),
            ROUTES[2]: (True, 1.0, True),
        }
        if self.mode not in expected:
            raise PermissionError("D6C request route is outside the frozen protocol")
        if type(self.enabled) is not bool or type(self.scale) is not float:
            raise PermissionError("D6C request flags must be exact bool and float")
        observed = (
            self.enabled,
            self.scale,
            type(self.callback) is D5CSyntheticProbe,
        )
        if observed != expected[self.mode]:
            raise PermissionError("D6C request does not match its frozen route")


class FixedPersistentDispatcher:
    def __init__(self, request_context: ContextVar[Any]) -> None:
        self._request_context = request_context
        self.dispatch_count = 0
        self.callback_count = 0

    def __call__(self, **payload: Any) -> Any:
        request = self._request_context.get()
        if request is _NO_REQUEST:
            raise PermissionError("D6C dispatcher requires a scoped request")
        if type(request) is not D6CCouplingRequest:
            raise PermissionError("D6C dispatcher rejects non-exact requests")
        residual = payload.get("residual_x")
        if residual is None:
            raise TypeError("D6C dispatcher requires a residual")
        self.dispatch_count += 1
        if not request.enabled or request.scale == 0.0:
            return residual
        callback = request.callback
        if type(callback) is not D5CSyntheticProbe:
            raise PermissionError("D6C active synthetic callback is unavailable")
        output = callback(**payload)
        self.callback_count += 1
        return output


class D6COfflineReporterAdapter:
    """Explicit pure-Python reporter used only by no-model fixture tests."""

    def call_record(
        self, *, call_id: str, fixture: Mapping[str, Any], route: str,
        phase: str, round_index: int, order_position: int,
        output: tuple[Any, Any],
    ) -> dict[str, Any]:
        logits, state = output
        if type(logits) is not OfflineTensor or not isinstance(state, list):
            raise TypeError("D6C offline reporter requires the exact fixture output")
        if not all(type(value) is int for value in state):
            raise TypeError("D6C offline reporter state must contain exact integers")
        def scalars(value: Any):
            if isinstance(value, tuple):
                for item in value:
                    yield from scalars(item)
            else:
                yield value
        return {
            "call_id": call_id, "fixture_id": fixture["fixture_id"],
            "route": route, "phase": phase, "round_index": round_index,
            "order_position": order_position, "token_ids": list(fixture["token_ids"]),
            "state_input": "offline_fixture", "full_output": bool(fixture["full_output"]),
            "finite": all(
                value == value and abs(value) != float("inf")
                for value in scalars(logits.values)
            ),
            "output_digest_sha256": _output_digest(output),
        }

    def pair_record(
        self, left_record: Mapping[str, Any], right_record: Mapping[str, Any],
        left_output: tuple[Any, Any], right_output: tuple[Any, Any],
    ) -> dict[str, Any]:
        equal = _output_digest(left_output) == _output_digest(right_output)
        return {
            "left_call_id": left_record["call_id"],
            "right_call_id": right_record["call_id"],
            "all_torch_equal": equal,
            "logits": {"valid": True},
            "state": {"shape_dtype_device_compatible": True},
        }


def _managed_snapshot(model: Any) -> dict[str, Any]:
    instance = getattr(model, "__dict__", None)
    if not isinstance(instance, dict):
        raise TypeError("D6C model must expose a mutable instance dictionary")
    if not all(name in instance for name in MANAGED_NAMES):
        raise RuntimeError("D6C persistent bindings are incomplete")
    return {name: instance[name] for name in MANAGED_NAMES}


def _snapshots_identical(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(left[name] is right[name] for name in left)


class RWKV7D6CPersistentRuntime:
    """One-time persistent AST installation with context-local route requests."""

    def __init__(
        self,
        *,
        base_model: Any,
        upstream_source_bytes: bytes,
        upstream_globals: Mapping[str, Any],
        upstream_package_version: str,
        upstream_de_version: str | None,
        execution_claim_sha256: str,
        machine_authorization_sha256: str,
    ) -> None:
        if upstream_package_version != EXPECTED_RWKV_PACKAGE_VERSION:
            raise PermissionError("D6C upstream package version changed")
        import hashlib

        if hashlib.sha256(upstream_source_bytes).hexdigest() != EXPECTED_RWKV_MODEL_SOURCE_SHA256:
            raise PermissionError("D6C upstream source digest changed")
        if upstream_de_version is not None:
            raise PermissionError("D6C requires RWKV_DE_VERSION to be unset")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (execution_claim_sha256, machine_authorization_sha256)
        ):
            raise PermissionError("D6C runtime requires bound execution evidence")
        instance = getattr(base_model, "__dict__", None)
        if not isinstance(instance, dict):
            raise TypeError("D6C base model must expose an instance dictionary")
        if any(name in instance for name in MANAGED_NAMES):
            raise RuntimeError("D6C persistent bindings already exist")
        source = upstream_source_bytes.decode("utf-8")
        methods, counts = compile_instrumented_methods(
            upstream_source=source,
            upstream_globals=upstream_globals,
            rwkv_de_version=upstream_de_version,
        )
        if counts != {"forward_one": 1, "forward_seq": 1}:
            raise RuntimeError("D6C AST injection counts changed")
        self._base_model = base_model
        self._request_context: ContextVar[Any] = ContextVar(
            f"psa_d6c_request_{id(self)}", default=_NO_REQUEST
        )
        self._dispatcher = FixedPersistentDispatcher(self._request_context)
        self._call_lock = threading.Lock()
        self.execution_count = 0
        self.rejection_count = 0
        setattr(base_model, CALLBACK_ATTRIBUTE, self._dispatcher)
        for name in TARGET_METHODS:
            setattr(base_model, name, types.MethodType(methods[name], base_model))
        self.installation_count = 3
        self.injection_counts = dict(counts)
        self._installed_snapshot = _managed_snapshot(base_model)

    @property
    def dispatcher(self) -> FixedPersistentDispatcher:
        return self._dispatcher

    def context_is_empty(self) -> bool:
        return self._request_context.get() is _NO_REQUEST

    def bindings_are_stable(self) -> bool:
        return _snapshots_identical(
            self._installed_snapshot, _managed_snapshot(self._base_model)
        )

    def forward(
        self,
        tokens: Sequence[int],
        state: Any,
        full_output: bool,
        *,
        coupling: D6CCouplingRequest,
    ) -> tuple[Any, Any]:
        if type(coupling) is not D6CCouplingRequest:
            raise PermissionError("D6C runtime rejects non-exact requests")
        coupling.validate()
        if not self._call_lock.acquire(blocking=False):
            self.rejection_count += 1
            raise RuntimeError("D6C rejects nested or concurrent requests")
        token: Token[Any] | None = None
        try:
            if not self.bindings_are_stable():
                raise RuntimeError("D6C persistent bindings changed before forward")
            token = self._request_context.set(coupling)
            self.execution_count += 1
            output = self._base_model.forward(list(tokens), state, bool(full_output))
            if not self.bindings_are_stable():
                raise RuntimeError("D6C persistent bindings changed during forward")
            return output
        finally:
            if token is not None:
                self._request_context.reset(token)
            self._call_lock.release()


def _request(route: str, probe: D5CSyntheticProbe) -> D6CCouplingRequest:
    if route == ROUTES[0]:
        return D6CCouplingRequest(route, False, 0.0, None)
    if route == ROUTES[1]:
        return D6CCouplingRequest(route, True, 0.0, None)
    if route == ROUTES[2]:
        return D6CCouplingRequest(route, True, 1.0, probe)
    raise ValueError("unknown D6C route")


def _invoke(
    *, fixture: Mapping[str, Any], route: str, runtime: Any,
    probe: D5CSyntheticProbe, torch: Any,
    state_factory: Callable[[], Any] | None = None,
) -> tuple[Any, Any]:
    context_factory = getattr(torch, "inference_mode", None)
    context = context_factory() if callable(context_factory) else _NullContext()
    with context:
        return runtime.forward(
            list(fixture["token_ids"]),
            state_factory() if callable(state_factory) else None,
            bool(fixture["full_output"]),
            coupling=_request(route, probe),
        )


def _record(
    *, call_id: str, fixture: Mapping[str, Any], route: str, phase: str,
    round_index: int, order_position: int, output: tuple[Any, Any],
    torch: Any, offline_adapter: D6COfflineReporterAdapter | None,
) -> dict[str, Any]:
    if offline_adapter is not None:
        if type(offline_adapter) is not D6COfflineReporterAdapter:
            raise TypeError("D6C rejects a non-exact offline reporter adapter")
        return offline_adapter.call_record(
            call_id=call_id, fixture=fixture, route=route, phase=phase,
            round_index=round_index, order_position=order_position, output=output,
        )
    return _call_record(
        call_id=call_id, fixture=fixture, route=route, phase=phase,
        round_index=round_index, order_position=order_position,
        output=output, torch=torch,
    )


def _pair(
    left_record: Mapping[str, Any], right_record: Mapping[str, Any],
    left_output: tuple[Any, Any], right_output: tuple[Any, Any],
    torch: Any, offline_adapter: D6COfflineReporterAdapter | None,
) -> dict[str, Any]:
    if offline_adapter is not None:
        if type(offline_adapter) is not D6COfflineReporterAdapter:
            raise TypeError("D6C rejects a non-exact offline reporter adapter")
        return offline_adapter.pair_record(
            left_record, right_record, left_output, right_output
        )
    return _pair_record(
        left_record, right_record, left_output, right_output, torch
    )


def execute_d6c_mechanism_core(
    *, runtime: Any, probe: D5CSyntheticProbe, torch: Any,
    state_factory: Callable[[], Any] | None = None,
    offline_adapter: D6COfflineReporterAdapter | None = None,
) -> dict[str, Any]:
    fixture_reports: list[dict[str, Any]] = []
    within: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    total_calls = 0
    for fixture in FIXTURES:
        calls: list[dict[str, Any]] = []
        outputs: dict[str, list[tuple[dict[str, Any], tuple[Any, Any]]]] = {
            route: [] for route in ROUTES
        }
        precondition = _invoke(
            fixture=fixture, route=ROUTES[0], runtime=runtime, probe=probe,
            torch=torch, state_factory=state_factory,
        )
        calls.append(_record(
            call_id=f"{fixture['fixture_id']}-pre-off", fixture=fixture,
            route=ROUTES[0], phase="off_precondition_unscored", round_index=0,
            order_position=1, output=precondition, torch=torch,
            offline_adapter=offline_adapter,
        ))
        total_calls += 1
        for round_index, round_routes in enumerate(LATIN_ROUNDS, start=1):
            per_round: dict[str, tuple[dict[str, Any], tuple[Any, Any]]] = {}
            for position, route in enumerate(round_routes, start=1):
                output = _invoke(
                    fixture=fixture, route=route, runtime=runtime,
                    probe=probe, torch=torch, state_factory=state_factory,
                )
                record = _record(
                    call_id=f"{fixture['fixture_id']}-r{round_index}-p{position}",
                    fixture=fixture, route=route, phase="scored",
                    round_index=round_index, order_position=position,
                    output=output, torch=torch, offline_adapter=offline_adapter,
                )
                calls.append(record)
                outputs[route].append((record, output))
                per_round[route] = (record, output)
                total_calls += 1
            off_record, off_output = per_round[ROUTES[0]]
            zero_record, zero_output = per_round[ROUTES[1]]
            active_record, active_output = per_round[ROUTES[2]]
            controls.append(_pair(
                off_record, zero_record, off_output, zero_output, torch,
                offline_adapter,
            ))
            active.append(_pair(
                active_record, off_record, active_output, off_output, torch,
                offline_adapter,
            ))
            active.append(_pair(
                active_record, zero_record, active_output, zero_output, torch,
                offline_adapter,
            ))
        for route in ROUTES:
            first_record, first_output = outputs[route][0]
            for record, output in outputs[route][1:]:
                within.append(_pair(
                    first_record, record, first_output, output, torch,
                    offline_adapter,
                ))
        fixture_reports.append({"fixture": dict(fixture), "calls": calls})

    checks = {
        "model_forward_calls_exact": total_calls == MODEL_FORWARD_CALLS_TOTAL,
        "runtime_execution_count_exact": runtime.execution_count == MODEL_FORWARD_CALLS_TOTAL,
        "no_raw_original_route": set(ROUTES) == {
            "persistent_off", "persistent_zero", "persistent_active_synthetic"
        },
        "within_route_comparisons_exact": len(within) == 18,
        "control_comparisons_exact": len(controls) == 8,
        "active_comparisons_exact": len(active) == 16,
        "all_outputs_finite": all(
            call["finite"] for fixture in fixture_reports for call in fixture["calls"]
        ),
        "within_route_all_exact": all(item["all_torch_equal"] for item in within),
        "off_zero_all_exact": all(item["all_torch_equal"] for item in controls),
        "active_differs_from_controls": all(
            not item["all_torch_equal"] for item in active
        ),
        "callback_invocations_exact": probe.invocation_count
        == ACTIVE_CALLBACK_CALLS_TOTAL,
        "probe_applications_exact": probe.application_count
        == ACTIVE_PROBE_APPLICATIONS_TOTAL,
        "persistent_bindings_stable": runtime.installation_count == 3
        and runtime.bindings_are_stable() and runtime.context_is_empty(),
    }
    return {
        "runtime_version": RUNTIME_VERSION,
        "status": (
            "d6c_persistent_mechanism_passed"
            if all(checks.values()) else "d6c_persistent_mechanism_failed"
        ),
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "model_forward_calls": total_calls,
            "active_forward_calls": ACTIVE_FORWARD_CALLS_TOTAL,
            "callback_invocations": probe.invocation_count,
            "probe_applications": probe.application_count,
            "within_route_comparisons": len(within),
            "control_comparisons": len(controls),
            "active_comparisons": len(active),
        },
        "fixtures": fixture_reports,
        "probe": {
            "kind": "deterministic_synthetic_not_self_projection",
            "target_layer_index_zero_based": TARGET_LAYER_INDEX,
            "real_layer_selected": False,
            "real_self_projection": False,
            "self_effect_claim": False,
        },
    }
