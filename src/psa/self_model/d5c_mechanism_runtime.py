from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
import threading
import types
from typing import Any, Mapping, Sequence

from psa.self_model.d4a_failure_diagnostic_runtime import (
    _flatten_tensors,
    _pair_record,
    _tensor_record,
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


RUNTIME_VERSION = "0.1-coupling-d5c-real-mechanism"
ROUTES = (
    "original_uninstrumented",
    "active_wrapper_off",
    "active_wrapper_zero",
    "active_synthetic_probe",
)
PRECONDITION_ORDER = ROUTES
SCORED_ROUNDS = (
    ROUTES,
    (ROUTES[1], ROUTES[2], ROUTES[3], ROUTES[0]),
    (ROUTES[2], ROUTES[3], ROUTES[0], ROUTES[1]),
    (ROUTES[3], ROUTES[0], ROUTES[1], ROUTES[2]),
)
CONTROL_ROUTES = ROUTES[:3]
FIXTURES = (
    {
        "fixture_id": "single_noncore",
        "token_ids": [2764],
        "state_input": "none",
        "full_output": False,
    },
    {
        "fixture_id": "sequence_noncore",
        "token_ids": [187, 931, 2764],
        "state_input": "none",
        "full_output": True,
    },
)
N_LAYER = 32
HIDDEN_DIMENSION = 2560
TARGET_LAYER_INDEX = (N_LAYER - 1) // 2
TARGET_RESIDUAL_RMS_RATIO = 0.01
MANAGED_BINDING_NAMES = (CALLBACK_ATTRIBUTE, *TARGET_METHODS)
_ABSENT_BINDING = object()
_ACTIVE_BINDING_MODEL_IDS: set[int] = set()
_ACTIVE_BINDING_LOCK = threading.RLock()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def deterministic_unit_rms_vector() -> tuple[float, ...]:
    raw = tuple(
        math.sin((index + 1) * 0.6180339887498948)
        + 0.5 * math.cos((index + 1) * 0.4142135623730950)
        for index in range(HIDDEN_DIMENSION)
    )
    rms = math.sqrt(sum(value * value for value in raw) / len(raw))
    vector = tuple(value / rms for value in raw)
    observed = math.sqrt(sum(value * value for value in vector) / len(vector))
    if not math.isfinite(observed) or abs(observed - 1.0) > 1e-12:
        raise RuntimeError("D5C deterministic vector normalization failed")
    return vector


class D5CSyntheticProbe:
    """Authorized mechanism-only synthetic residual; it is not a Self projection."""

    mechanism_only = True
    real_self_projection = False
    projection_trained = False
    effect_layer_selected = False
    phase = "post_ffn_residual"

    def __init__(
        self,
        *,
        torch: Any,
        execution_claim_sha256: str,
        machine_authorization_sha256: str,
    ) -> None:
        if not _is_sha256(execution_claim_sha256) or not _is_sha256(
            machine_authorization_sha256
        ):
            raise PermissionError("D5C probe requires bound execution evidence")
        self._torch = torch
        self._vector = deterministic_unit_rms_vector()
        self.execution_claim_sha256 = execution_claim_sha256
        self.machine_authorization_sha256 = machine_authorization_sha256
        self.invocation_count = 0
        self.application_count = 0
        self.applications: list[dict[str, Any]] = []

    def _finite(self, value: Any) -> bool:
        return bool(self._torch.isfinite(value).all().item())

    def __call__(
        self,
        *,
        phase: str,
        layer_index: int,
        execution_path: str,
        residual_x: Any,
    ) -> Any:
        if phase != self.phase or execution_path not in {"forward_one", "forward_seq"}:
            raise PermissionError("D5C callback phase or execution path changed")
        if type(layer_index) is not int or not 0 <= layer_index < N_LAYER:
            raise PermissionError("D5C callback layer index is outside the frozen model")
        shape = tuple(residual_x.shape)
        if not shape or shape[-1] != HIDDEN_DIMENSION or len(shape) not in {1, 2}:
            raise RuntimeError("D5C residual shape must end in 2560")
        if not self._finite(residual_x):
            raise RuntimeError("D5C residual contains non-finite values")
        self.invocation_count += 1
        if layer_index != TARGET_LAYER_INDEX:
            return residual_x

        residual_float = residual_x.detach().float()
        rms = residual_float.square().mean().sqrt()
        if not self._finite(rms) or float(rms.item()) <= 0.0:
            raise RuntimeError("D5C residual RMS is not finite and positive")
        vector = self._torch.tensor(
            self._vector,
            device=residual_x.device,
            dtype=self._torch.float32,
        )
        delta_float = vector * (rms * TARGET_RESIDUAL_RMS_RATIO)
        delta = delta_float.to(dtype=residual_x.dtype)
        output = residual_x + delta
        if (
            tuple(output.shape) != shape
            or output.dtype != residual_x.dtype
            or str(output.device) != str(residual_x.device)
            or not self._finite(delta)
            or not self._finite(output)
        ):
            raise RuntimeError("D5C synthetic probe failed the residual invariant")
        self.application_count += 1
        self.applications.append(
            {
                "layer_index": layer_index,
                "execution_path": execution_path,
                "shape": list(shape),
                "dtype": str(output.dtype),
                "device": str(output.device),
                "residual_rms": float(rms.item()),
                "target_residual_rms_ratio": TARGET_RESIDUAL_RMS_RATIO,
            }
        )
        return output


@dataclass(frozen=True)
class D5CCouplingRequest:
    enabled: bool
    scale: float
    callback: D5CSyntheticProbe | None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or isinstance(self.scale, bool):
            raise ValueError("D5C coupling request flags are invalid")
        if float(self.scale) not in {0.0, 1.0}:
            raise PermissionError("D5C coupling scale must be exactly zero or one")
        active = self.enabled and float(self.scale) == 1.0
        if active and type(self.callback) is not D5CSyntheticProbe:
            raise PermissionError("D5C active route requires the exact synthetic probe")
        if not active and self.callback is not None:
            raise PermissionError("D5C OFF and zero routes cannot bind a callback")


class D5CCleanupTransactionError(RuntimeError):
    """The model call cannot commit because binding restoration was not verified."""

    def __init__(
        self,
        *,
        primary_exception: BaseException | None,
        cleanup_failures: Sequence[str],
        verification_failures: Sequence[str],
        output_was_produced: bool,
    ) -> None:
        super().__init__("D5C temporary binding cleanup failed closed")
        self.primary_exception = primary_exception
        self.cleanup_failures = tuple(cleanup_failures)
        self.verification_failures = tuple(verification_failures)
        self.output_was_produced = output_was_produced


@dataclass(frozen=True)
class _D5CBindingSnapshot:
    instance_values: Mapping[str, Any]
    static_descriptors: Mapping[str, Any]
    resolved_functions: Mapping[str, Any]
    callback_value: Any


def _resolved_function(value: Any) -> Any:
    return getattr(value, "__func__", value)


def _capture_binding_snapshot(base_model: Any) -> _D5CBindingSnapshot:
    instance_dict = getattr(base_model, "__dict__", None)
    if not isinstance(instance_dict, dict):
        raise TypeError("D5C base model must expose a mutable instance dictionary")
    if any(name in instance_dict for name in MANAGED_BINDING_NAMES):
        raise RuntimeError("D5C model has conflicting temporary overrides")
    return _D5CBindingSnapshot(
        instance_values={
            name: instance_dict.get(name, _ABSENT_BINDING)
            for name in MANAGED_BINDING_NAMES
        },
        static_descriptors={
            name: inspect.getattr_static(base_model, name) for name in TARGET_METHODS
        },
        resolved_functions={
            name: _resolved_function(getattr(base_model, name)) for name in TARGET_METHODS
        },
        callback_value=getattr(base_model, CALLBACK_ATTRIBUTE, _ABSENT_BINDING),
    )


def _restore_bindings(
    base_model: Any, snapshot: _D5CBindingSnapshot
) -> list[str]:
    failures: list[str] = []
    instance_dict = base_model.__dict__
    for name in reversed(MANAGED_BINDING_NAMES):
        try:
            previous = snapshot.instance_values[name]
            if previous is _ABSENT_BINDING:
                if name in instance_dict:
                    delattr(base_model, name)
            else:
                setattr(base_model, name, previous)
        except BaseException as error:
            failures.append(f"{name}:{type(error).__name__}:{error}")
    return failures


def _verify_restored_bindings(
    base_model: Any, snapshot: _D5CBindingSnapshot
) -> list[str]:
    failures: list[str] = []
    try:
        instance_dict = base_model.__dict__
    except BaseException as error:
        failures.append(f"instance_dictionary:{type(error).__name__}:{error}")
        instance_dict = {}
    for name in MANAGED_BINDING_NAMES:
        try:
            before = snapshot.instance_values[name]
            after = instance_dict.get(name, _ABSENT_BINDING)
            if before is _ABSENT_BINDING:
                if after is not _ABSENT_BINDING:
                    failures.append(f"instance_ownership:{name}")
            elif after is not before:
                failures.append(f"instance_value:{name}")
        except BaseException as error:
            failures.append(f"instance_verification:{name}:{type(error).__name__}:{error}")
    for name in TARGET_METHODS:
        try:
            if inspect.getattr_static(base_model, name) is not snapshot.static_descriptors[name]:
                failures.append(f"static_descriptor:{name}")
        except BaseException as error:
            failures.append(f"static_descriptor:{name}:{type(error).__name__}:{error}")
        try:
            if _resolved_function(getattr(base_model, name)) is not snapshot.resolved_functions[name]:
                failures.append(f"resolved_function:{name}")
        except BaseException as error:
            failures.append(f"resolved_function:{name}:{type(error).__name__}:{error}")
    try:
        callback = getattr(base_model, CALLBACK_ATTRIBUTE, _ABSENT_BINDING)
        if callback is not snapshot.callback_value:
            failures.append("callback_resolution")
    except BaseException as error:
        failures.append(f"callback_resolution:{type(error).__name__}:{error}")
    return failures


class RWKV7D5CActiveRuntime:
    """Temporarily binds the locked project AST transform around one model call."""

    runtime_version = RUNTIME_VERSION

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
            raise RuntimeError("D5C RWKV package version differs from the source lock")
        if hashlib.sha256(upstream_source_bytes).hexdigest() != EXPECTED_RWKV_MODEL_SOURCE_SHA256:
            raise RuntimeError("D5C RWKV source differs from the source lock")
        if upstream_de_version is not None:
            raise PermissionError("D5C requires RWKV_DE_VERSION to be unset")
        if not _is_sha256(execution_claim_sha256) or not _is_sha256(
            machine_authorization_sha256
        ):
            raise PermissionError("D5C runtime requires bound execution evidence")
        if not callable(getattr(base_model, "forward", None)):
            raise TypeError("D5C base model must expose forward")
        instance_dict = getattr(base_model, "__dict__", None)
        if not isinstance(instance_dict, dict) or CALLBACK_ATTRIBUTE in instance_dict:
            raise RuntimeError("D5C base model cannot accept temporary callback binding")
        methods, counts = compile_instrumented_methods(
            upstream_source=upstream_source_bytes.decode("utf-8"),
            upstream_globals=upstream_globals,
            rwkv_de_version=upstream_de_version,
        )
        if counts != {"forward_one": 1, "forward_seq": 1}:
            raise RuntimeError("D5C injection counts differ from the source lock")
        self._base_model = base_model
        self._methods = methods
        self.execution_count = 0

    def forward(
        self,
        tokens: Sequence[int],
        state: Any,
        full_output: bool = False,
        *,
        coupling: D5CCouplingRequest,
    ) -> tuple[Any, Any]:
        if type(coupling) is not D5CCouplingRequest:
            raise PermissionError("D5C runtime rejects non-exact requests")
        callback = coupling.callback if coupling.enabled and coupling.scale == 1.0 else None
        model_id = id(self._base_model)
        with _ACTIVE_BINDING_LOCK:
            if model_id in _ACTIVE_BINDING_MODEL_IDS:
                raise RuntimeError("D5C rejects nested or concurrent temporary binding")
            snapshot = _capture_binding_snapshot(self._base_model)
            _ACTIVE_BINDING_MODEL_IDS.add(model_id)
        primary: BaseException | None = None
        output: Any = _ABSENT_BINDING
        try:
            try:
                setattr(self._base_model, CALLBACK_ATTRIBUTE, callback)
                for name, function in self._methods.items():
                    setattr(
                        self._base_model,
                        name,
                        types.MethodType(function, self._base_model),
                    )
                self.execution_count += 1
                output = self._base_model.forward(list(tokens), state, full_output)
            except BaseException as error:
                primary = error
            cleanup_failures = _restore_bindings(self._base_model, snapshot)
            verification_failures = _verify_restored_bindings(
                self._base_model, snapshot
            )
            if cleanup_failures or verification_failures:
                failure = D5CCleanupTransactionError(
                    primary_exception=primary,
                    cleanup_failures=cleanup_failures,
                    verification_failures=verification_failures,
                    output_was_produced=output is not _ABSENT_BINDING,
                )
                if primary is not None:
                    raise failure from primary
                raise failure
            if primary is not None:
                raise primary
            if output is _ABSENT_BINDING:
                raise AssertionError("D5C model call produced no output")
            return output
        finally:
            with _ACTIVE_BINDING_LOCK:
                _ACTIVE_BINDING_MODEL_IDS.discard(model_id)


def _invoke(
    *,
    route: str,
    fixture: Mapping[str, Any],
    base_model: Any,
    runtime: Any,
    probe: Any,
    torch: Any,
) -> tuple[Any, Any]:
    context_factory = getattr(torch, "inference_mode", None)
    context = context_factory() if callable(context_factory) else _NullContext()
    with context:
        if route == "original_uninstrumented":
            return base_model.forward(
                list(fixture["token_ids"]), None, bool(fixture["full_output"])
            )
        if route == "active_wrapper_off":
            request = D5CCouplingRequest(enabled=False, scale=0.0, callback=None)
        elif route == "active_wrapper_zero":
            request = D5CCouplingRequest(enabled=True, scale=0.0, callback=None)
        elif route == "active_synthetic_probe":
            request = D5CCouplingRequest(enabled=True, scale=1.0, callback=probe)
        else:
            raise ValueError("unknown D5C route")
        return runtime.forward(
            list(fixture["token_ids"]),
            None,
            bool(fixture["full_output"]),
            coupling=request,
        )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> None:
        return None


def _all_finite(output: tuple[Any, Any], torch: Any) -> bool:
    tensors = [output[0], *(tensor for _, tensor in _flatten_tensors(output[1]))]
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _call_record(
    *,
    call_id: str,
    fixture: Mapping[str, Any],
    route: str,
    phase: str,
    round_index: int,
    order_position: int,
    output: tuple[Any, Any],
    torch: Any,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "fixture_id": fixture["fixture_id"],
        "route": route,
        "phase": phase,
        "round_index": round_index,
        "order_position": order_position,
        "token_ids": list(fixture["token_ids"]),
        "state_input": "none",
        "full_output": bool(fixture["full_output"]),
        "finite": _all_finite(output, torch),
        "logits": _tensor_record(output[0], torch),
        "state": {
            "component_count": len(list(_flatten_tensors(output[1]))),
            "components": [
                {"path": path, **_tensor_record(tensor, torch)}
                for path, tensor in _flatten_tensors(output[1])
            ],
        },
    }


def execute_d5c_mechanism_core(
    *,
    base_model: Any,
    active_runtime: Any,
    probe: Any,
    torch: Any,
) -> dict[str, Any]:
    fixture_reports = []
    all_within: list[dict[str, Any]] = []
    all_controls: list[dict[str, Any]] = []
    all_active: list[dict[str, Any]] = []
    total_calls = 0
    for fixture in FIXTURES:
        calls: list[dict[str, Any]] = []
        outputs_by_route: dict[str, list[tuple[dict[str, Any], tuple[Any, Any]]]] = {
            route: [] for route in ROUTES
        }
        prefix_output = _invoke(
            route=ROUTES[0], fixture=fixture, base_model=base_model,
            runtime=active_runtime, probe=probe, torch=torch,
        )
        calls.append(
            _call_record(
                call_id=f"{fixture['fixture_id']}-prefix", fixture=fixture,
                route=ROUTES[0], phase="prefix_unscored", round_index=0,
                order_position=0, output=prefix_output, torch=torch,
            )
        )
        total_calls += 1
        for position, route in enumerate(PRECONDITION_ORDER, start=1):
            output = _invoke(
                route=route, fixture=fixture, base_model=base_model,
                runtime=active_runtime, probe=probe, torch=torch,
            )
            calls.append(
                _call_record(
                    call_id=f"{fixture['fixture_id']}-pre-{position}", fixture=fixture,
                    route=route, phase="precondition_unscored", round_index=0,
                    order_position=position, output=output, torch=torch,
                )
            )
            total_calls += 1
        for round_index, round_routes in enumerate(SCORED_ROUNDS, start=1):
            for position, route in enumerate(round_routes, start=1):
                output = _invoke(
                    route=route, fixture=fixture, base_model=base_model,
                    runtime=active_runtime, probe=probe, torch=torch,
                )
                record = _call_record(
                    call_id=f"{fixture['fixture_id']}-r{round_index}-p{position}",
                    fixture=fixture, route=route, phase="scored",
                    round_index=round_index, order_position=position,
                    output=output, torch=torch,
                )
                calls.append(record)
                outputs_by_route[route].append((record, output))
                total_calls += 1

        within = []
        for route in ROUTES:
            first_record, first_output = outputs_by_route[route][0]
            for record, output in outputs_by_route[route][1:]:
                within.append(_pair_record(first_record, record, first_output, output, torch))
        controls = []
        active = []
        for round_offset in range(4):
            per_route = {
                route: outputs_by_route[route][round_offset] for route in ROUTES
            }
            for left_index in range(len(CONTROL_ROUTES)):
                for right_index in range(left_index + 1, len(CONTROL_ROUTES)):
                    left_record, left_output = per_route[CONTROL_ROUTES[left_index]]
                    right_record, right_output = per_route[CONTROL_ROUTES[right_index]]
                    controls.append(
                        _pair_record(left_record, right_record, left_output, right_output, torch)
                    )
            active_record, active_output = per_route[ROUTES[3]]
            for control_route in CONTROL_ROUTES:
                control_record, control_output = per_route[control_route]
                active.append(
                    _pair_record(active_record, control_record, active_output, control_output, torch)
                )
        all_within.extend(within)
        all_controls.extend(controls)
        all_active.extend(active)
        fixture_reports.append(
            {
                "fixture": dict(fixture),
                "calls": calls,
                "within_route_comparisons": within,
                "control_cross_route_comparisons": controls,
                "active_control_comparisons": active,
            }
        )

    checks = {
        "model_forward_calls_exact": total_calls == 42,
        "within_comparison_count_exact": len(all_within) == 24,
        "control_comparison_count_exact": len(all_controls) == 24,
        "active_comparison_count_exact": len(all_active) == 24,
        "all_outputs_finite": all(
            call["finite"] for fixture in fixture_reports for call in fixture["calls"]
        ),
        "within_route_all_exact": all(item["all_torch_equal"] for item in all_within),
        "all_control_pairs_exact": all(item["all_torch_equal"] for item in all_controls),
        "active_differs_from_each_control": all(
            not item["all_torch_equal"] for item in all_active
        ),
        "all_comparisons_compatible": all(
            item["logits"]["valid"]
            and item["state"]["shape_dtype_device_compatible"]
            for item in (*all_within, *all_controls, *all_active)
        ),
        "callback_invocations_exact": getattr(probe, "invocation_count", None) == 320,
        "probe_applications_exact": getattr(probe, "application_count", None) == 10,
    }
    return {
        "runtime_version": RUNTIME_VERSION,
        "status": "d5c_mechanism_smoke_passed" if all(checks.values()) else "d5c_mechanism_smoke_failed",
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "model_forward_calls": total_calls,
            "within_route_comparisons": len(all_within),
            "control_cross_route_comparisons": len(all_controls),
            "active_control_comparisons": len(all_active),
            "callback_invocations": getattr(probe, "invocation_count", None),
            "probe_applications": getattr(probe, "application_count", None),
        },
        "fixtures": fixture_reports,
        "probe": {
            "kind": "deterministic_synthetic_unit_rms_vector_not_self_representation",
            "target_layer_index_zero_based": TARGET_LAYER_INDEX,
            "target_layer_rule": "floor((n_layer-1)/2)",
            "target_residual_rms_ratio": TARGET_RESIDUAL_RMS_RATIO,
            "real_self_projection": False,
            "self_effect_claim": False,
        },
    }
