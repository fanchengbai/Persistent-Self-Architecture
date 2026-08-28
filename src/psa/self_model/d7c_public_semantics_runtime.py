from __future__ import annotations

import copy
from contextvars import ContextVar, Token
from dataclasses import dataclass
import types
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import sha256_json
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
    compile_instrumented_methods,
)


RUNTIME_VERSION = "0.1-self-model-d7c-public-semantics-runtime"
N_LAYER = 32
STATE_COMPONENTS = N_LAYER * 3
TARGET_LAYER_RULE_ID = "d7_lower_half_terminal_layer_v01"
TARGET_LAYER_INDEX = N_LAYER // 2 - 1
SINGLE_TOKEN_IDS = (2764,)
SEQUENCE_TOKEN_IDS = (187, 931, 2764)
EXECUTION_PATHS = ("forward_one", "forward_seq")
STATE_INPUTS = ("none", "prebuilt")
FULL_OUTPUT_VALUES = (False, True)
_NO_REQUEST = object()


@dataclass(frozen=True)
class D7CCompatibilityRequest:
    mode: str
    enabled: bool
    scale: float
    callback: Callable[..., Any] | None

    def validate(self) -> None:
        if self.mode == "zero":
            expected = (True, 0.0, None)
        elif self.mode == "active_synthetic":
            expected = (True, 1.0, "callable")
        else:
            raise PermissionError("D7-C request mode is not frozen")
        actual_callback = "callable" if callable(self.callback) else None
        if (self.enabled, self.scale, actual_callback) != expected:
            raise PermissionError("D7-C request fields do not match the mode")


class D7CFixedDispatcher:
    def __init__(self, context: ContextVar[Any]) -> None:
        self._context = context
        self.invocation_count = 0
        self.active_callback_count = 0

    def __call__(self, **payload: Any) -> Any:
        request = self._context.get()
        if type(request) is not D7CCompatibilityRequest:
            raise PermissionError("D7-C dispatcher requires an exact scoped request")
        residual = payload.get("residual_x")
        if residual is None:
            raise TypeError("D7-C dispatcher requires residual_x")
        self.invocation_count += 1
        if request.scale == 0.0:
            return residual
        if not callable(request.callback):
            raise PermissionError("D7-C active request requires a callback")
        output = request.callback(**payload)
        if output is None:
            raise TypeError("D7-C callback returned no residual")
        self.active_callback_count += 1
        return output


def _instance_snapshot(value: Any) -> dict[str, Any]:
    instance = getattr(value, "__dict__", None)
    if not isinstance(instance, dict):
        raise TypeError("D7-C base object must expose an instance dictionary")
    return dict(instance)


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(left[key] is right[key] for key in left)


class D7CPublicSemanticsWrapper:
    """External wrapper preserving the base public forward dispatch semantics."""

    def __init__(
        self,
        *,
        base_model: Any,
        compiled_methods: Mapping[str, Callable[..., Any]],
        injection_counts: Mapping[str, int],
    ) -> None:
        if set(compiled_methods) != set(TARGET_METHODS) or not all(
            callable(compiled_methods[name]) for name in TARGET_METHODS
        ):
            raise TypeError("D7-C wrapper requires two compiled child methods")
        if dict(injection_counts) != {"forward_one": 1, "forward_seq": 1}:
            raise RuntimeError("D7-C child method injection counts changed")
        base_snapshot = _instance_snapshot(base_model)
        context: ContextVar[Any] = ContextVar(
            f"psa_d7c_request_{id(self)}", default=_NO_REQUEST
        )
        dispatcher = D7CFixedDispatcher(context)
        object.__setattr__(self, "_base_model", base_model)
        object.__setattr__(self, "_base_snapshot", base_snapshot)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_dispatcher", dispatcher)
        object.__setattr__(self, "execution_count", 0)
        object.__setattr__(self, "zero_state_initialization_count", 0)
        object.__setattr__(self, CALLBACK_ATTRIBUTE, dispatcher)
        for name in TARGET_METHODS:
            object.__setattr__(self, name, types.MethodType(compiled_methods[name], self))
        object.__setattr__(self, "_owned_snapshot", self._owned_snapshot_now())
        if not self.base_dictionary_is_stable():
            raise RuntimeError("D7-C wrapper constructor changed the base instance")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base_model"), name)

    @property
    def dispatcher(self) -> D7CFixedDispatcher:
        return self._dispatcher

    def _owned_snapshot_now(self) -> dict[str, Any]:
        return {
            CALLBACK_ATTRIBUTE: object.__getattribute__(self, CALLBACK_ATTRIBUTE),
            "forward_one": object.__getattribute__(self, "forward_one"),
            "forward_seq": object.__getattribute__(self, "forward_seq"),
            "forward_function": type(self).forward,
        }

    def owned_bindings_are_stable(self) -> bool:
        return _identity_equal(self._owned_snapshot, self._owned_snapshot_now())

    def base_dictionary_is_stable(self) -> bool:
        return _identity_equal(
            self._base_snapshot, _instance_snapshot(self._base_model)
        )

    def context_is_empty(self) -> bool:
        return self._context.get() is _NO_REQUEST

    def forward(
        self,
        tokens: Sequence[int],
        state: Any,
        full_output: bool = False,
        *,
        request: D7CCompatibilityRequest,
    ) -> Any:
        if type(request) is not D7CCompatibilityRequest:
            raise PermissionError("D7-C wrapper rejects non-exact requests")
        request.validate()
        if not self.base_dictionary_is_stable() or not self.owned_bindings_are_stable():
            raise RuntimeError("D7-C binding identity changed before forward")
        token_copy = list(tokens)
        if not token_copy:
            raise ValueError("D7-C tokens must not be empty")
        state_copy = copy.deepcopy(state)
        context_token: Token[Any] | None = None
        try:
            context_token = self._context.set(request)
            if state_copy is None:
                state_copy = self.generate_zero_state()
                object.__setattr__(
                    self,
                    "zero_state_initialization_count",
                    self.zero_state_initialization_count + 1,
                )
            object.__setattr__(self, "execution_count", self.execution_count + 1)
            if len(token_copy) > 1:
                output = self.forward_seq(token_copy, state_copy, bool(full_output))
            else:
                output = self.forward_one(token_copy[0], state_copy)
            if not self.base_dictionary_is_stable() or not self.owned_bindings_are_stable():
                raise RuntimeError("D7-C binding identity changed during forward")
            return output
        finally:
            if context_token is not None:
                self._context.reset(context_token)


def zero_request() -> D7CCompatibilityRequest:
    return D7CCompatibilityRequest("zero", True, 0.0, None)


def active_request(callback: Callable[..., Any]) -> D7CCompatibilityRequest:
    return D7CCompatibilityRequest("active_synthetic", True, 1.0, callback)


SYNTHETIC_UPSTREAM_SOURCE = """
class RWKV_x070:
    def __init__(self):
        self.n_layer = 32

    def generate_zero_state(self):
        return [0 for _ in range(self.n_layer * 3)]

    def forward(self, idx, state, full_output=False):
        if state == None:
            state = self.generate_zero_state()
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)

    def forward_one(self, idx, state):
        x = idx
        for i in range(self.n_layer):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_one(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state

    def forward_seq(self, idx, state, full_output=False):
        x = sum(idx)
        values = []
        for i in range(self.n_layer):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(x, state[i * 3 + 2], i)
            x = x + xx
            values.append(x)
        return (values if full_output else x), state
""".strip()


def _synthetic_namespace() -> tuple[dict[str, Any], type]:
    def cmix(x: int, state_value: int, layer_index: int) -> tuple[int, int]:
        del x, layer_index
        return 1, state_value + 1

    namespace: dict[str, Any] = {
        "RWKV_x070_CMix_one": cmix,
        "RWKV_x070_CMix_seq": cmix,
    }
    exec(SYNTHETIC_UPSTREAM_SOURCE, namespace)
    return namespace, namespace["RWKV_x070"]


class SyntheticActiveProbe:
    def __init__(self) -> None:
        self.invocation_count = 0
        self.application_count = 0
        self.layer_counts = {layer: 0 for layer in range(N_LAYER)}

    def __call__(
        self,
        *,
        phase: str,
        layer_index: int,
        execution_path: str,
        residual_x: int | list[int],
    ) -> int | list[int]:
        if phase != "post_ffn_residual" or execution_path not in EXECUTION_PATHS:
            raise PermissionError("D7-C synthetic callback metadata changed")
        self.invocation_count += 1
        self.layer_counts[layer_index] += 1
        if layer_index != TARGET_LAYER_INDEX:
            return residual_x
        self.application_count += 1
        if isinstance(residual_x, list):
            return [value + 7 for value in residual_x]
        return residual_x + 7


def compatibility_cells() -> tuple[dict[str, Any], ...]:
    cells = []
    for execution_path in EXECUTION_PATHS:
        for state_input in STATE_INPUTS:
            for full_output in FULL_OUTPUT_VALUES:
                cells.append(
                    {
                        "cell_id": f"d7c-cell-{len(cells) + 1:02d}",
                        "execution_path": execution_path,
                        "state_input": state_input,
                        "full_output": full_output,
                    }
                )
    return tuple(cells)


def run_synthetic_compatibility_acceptance() -> dict[str, Any]:
    namespace, fixture_type = _synthetic_namespace()
    cell_reports = []
    public_calls = 0
    wrapper_zero_calls = 0
    total_wrapper_initializations = 0
    all_base_stable = True
    zero_outputs: dict[tuple[str, str, bool], Any] = {}
    for cell in compatibility_cells():
        tokens = SINGLE_TOKEN_IDS if cell["execution_path"] == "forward_one" else SEQUENCE_TOKEN_IDS
        state = None if cell["state_input"] == "none" else [0] * STATE_COMPONENTS
        public_fixture = fixture_type()
        public_output = public_fixture.forward(
            list(tokens), copy.deepcopy(state), cell["full_output"]
        )
        public_calls += 1
        wrapper_fixture = fixture_type()
        methods, counts = compile_instrumented_methods(
            upstream_source=SYNTHETIC_UPSTREAM_SOURCE,
            upstream_globals=namespace,
            rwkv_de_version=None,
        )
        wrapper = D7CPublicSemanticsWrapper(
            base_model=wrapper_fixture,
            compiled_methods=methods,
            injection_counts=counts,
        )
        wrapper_output = wrapper.forward(
            list(tokens), copy.deepcopy(state), cell["full_output"], request=zero_request()
        )
        wrapper_zero_calls += 1
        zero_outputs[
            (cell["execution_path"], cell["state_input"], cell["full_output"])
        ] = copy.deepcopy(wrapper_output)
        total_wrapper_initializations += wrapper.zero_state_initialization_count
        all_base_stable = all_base_stable and wrapper.base_dictionary_is_stable()
        cell_reports.append(
            {
                **cell,
                "output_exact": public_output == wrapper_output,
                "logits_exact": public_output[0] == wrapper_output[0],
                "state_exact": public_output[1] == wrapper_output[1],
                "state_component_count": len(wrapper_output[1]),
                "zero_state_initializations": wrapper.zero_state_initialization_count,
                "base_dictionary_unchanged": wrapper.base_dictionary_is_stable(),
                "owned_bindings_stable": wrapper.owned_bindings_are_stable(),
                "context_restored": wrapper.context_is_empty(),
            }
        )

    active_reports = []
    probe = SyntheticActiveProbe()
    for execution_path, full_output in (("forward_one", False), ("forward_seq", True)):
        tokens = SINGLE_TOKEN_IDS if execution_path == "forward_one" else SEQUENCE_TOKEN_IDS
        fixture = fixture_type()
        methods, counts = compile_instrumented_methods(
            upstream_source=SYNTHETIC_UPSTREAM_SOURCE,
            upstream_globals=namespace,
            rwkv_de_version=None,
        )
        wrapper = D7CPublicSemanticsWrapper(
            base_model=fixture,
            compiled_methods=methods,
            injection_counts=counts,
        )
        zero_output = zero_outputs[(execution_path, "none", full_output)]
        active_output = wrapper.forward(
            list(tokens), None, full_output, request=active_request(probe)
        )
        active_reports.append(
            {
                "execution_path": execution_path,
                "full_output": full_output,
                "active_differs_from_zero": active_output != zero_output,
                "base_dictionary_unchanged": wrapper.base_dictionary_is_stable(),
                "context_restored": wrapper.context_is_empty(),
            }
        )
        all_base_stable = all_base_stable and wrapper.base_dictionary_is_stable()

    checks = {
        "eight_cells_exact": len(cell_reports) == 8,
        "sixteen_equivalence_forwards": public_calls == wrapper_zero_calls == 8,
        "all_cell_outputs_exact": all(item["output_exact"] for item in cell_reports),
        "all_cell_logits_exact": all(item["logits_exact"] for item in cell_reports),
        "all_cell_states_exact": all(item["state_exact"] for item in cell_reports),
        "all_states_have_ninety_six_components": all(
            item["state_component_count"] == STATE_COMPONENTS for item in cell_reports
        ),
        "none_cells_initialize_once": all(
            item["zero_state_initializations"] == (1 if item["state_input"] == "none" else 0)
            for item in cell_reports
        ),
        "two_active_forwards": len(active_reports) == 2,
        "active_outputs_differ_from_zero": all(
            item["active_differs_from_zero"] for item in active_reports
        ),
        "active_callbacks_cover_all_layers_twice": probe.invocation_count == 64
        and set(probe.layer_counts.values()) == {2},
        "target_layer_applied_once_per_active_forward": probe.application_count == 2,
        "base_instance_dictionaries_unchanged": all_base_stable,
        "owned_bindings_and_contexts_stable": all(
            item["owned_bindings_stable"] and item["context_restored"]
            for item in cell_reports
        )
        and all(item["context_restored"] for item in active_reports),
        "total_forward_plan_is_eighteen": public_calls + wrapper_zero_calls + len(active_reports)
        == 18,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "cell_reports": cell_reports,
        "active_reports": active_reports,
        "counts": {
            "equivalence_cells": len(cell_reports),
            "public_off_calls": public_calls,
            "wrapper_zero_calls": wrapper_zero_calls,
            "equivalence_forward_calls": public_calls + wrapper_zero_calls,
            "synthetic_active_forward_calls": len(active_reports),
            "total_forward_plan": public_calls + wrapper_zero_calls + len(active_reports),
            "synthetic_wrapper_forward_calls": wrapper_zero_calls + len(active_reports),
            "active_callback_invocations": probe.invocation_count,
            "active_target_layer_applications": probe.application_count,
            "wrapper_zero_state_initializations_in_equivalence_cells": total_wrapper_initializations,
        },
        "commitments": {
            "compatibility_cells_sha256": sha256_json(compatibility_cells()),
            "cell_reports_sha256": sha256_json(cell_reports),
            "active_reports_sha256": sha256_json(active_reports),
        },
    }
