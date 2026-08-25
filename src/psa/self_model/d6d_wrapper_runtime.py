from __future__ import annotations

import copy
from contextvars import ContextVar, Token
from dataclasses import dataclass
import threading
import types
from typing import Any, Callable, Mapping, Sequence

from psa.self_model.d6d_core_approach_design import CONDITIONS, SELF_CONDITIONS
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
)


_NO_REQUEST = object()


@dataclass(frozen=True)
class D6DIResidualCallback:
    kind: str
    function: Callable[..., Any]

    def __post_init__(self) -> None:
        if self.kind not in {"synthetic_positive", "frozen_self"}:
            raise PermissionError("D6D-I callback kind is outside the joint contract")
        if not callable(self.function):
            raise TypeError("D6D-I callback function must be callable")

    def __call__(self, **payload: Any) -> Any:
        return self.function(**payload)


@dataclass(frozen=True)
class D6DIRequest:
    condition: str
    enabled: bool
    scale: float
    callback: D6DIResidualCallback | None

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise PermissionError("D6D-I request condition is not frozen")
        if type(self.enabled) is not bool or type(self.scale) is not float:
            raise PermissionError("D6D-I request flags must be exact bool and float")
        if self.condition == "wrapper_off":
            expected = (False, 0.0, None)
        elif self.condition == "wrapper_zero":
            expected = (True, 0.0, None)
        elif self.condition == "synthetic_positive":
            expected = (True, 1.0, "synthetic_positive")
        else:
            expected = (True, 1.0, "frozen_self")
        callback_kind = self.callback.kind if type(self.callback) is D6DIResidualCallback else None
        if (self.enabled, self.scale, callback_kind) != expected:
            raise PermissionError("D6D-I request does not match its frozen condition")


class D6DIFixedDispatcher:
    def __init__(self, request_context: ContextVar[Any]) -> None:
        self._request_context = request_context
        self.dispatch_count = 0
        self.callback_count = 0

    def __call__(self, **payload: Any) -> Any:
        request = self._request_context.get()
        if request is _NO_REQUEST:
            raise PermissionError("D6D-I dispatcher requires a scoped request")
        if type(request) is not D6DIRequest:
            raise PermissionError("D6D-I dispatcher rejects non-exact requests")
        residual = payload.get("residual_x")
        if residual is None:
            raise TypeError("D6D-I dispatcher requires residual_x")
        self.dispatch_count += 1
        if not request.enabled or request.scale == 0.0:
            return residual
        callback = request.callback
        if type(callback) is not D6DIResidualCallback:
            raise PermissionError("D6D-I active callback is unavailable")
        output = callback(**payload)
        if output is None:
            raise TypeError("D6D-I callback returned no residual")
        self.callback_count += 1
        return output


def _instance_identity_snapshot(model: Any) -> dict[str, Any]:
    instance = getattr(model, "__dict__", None)
    if not isinstance(instance, dict):
        raise TypeError("D6D-I base object must expose an instance dictionary")
    return dict(instance)


def _identity_snapshots_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return left.keys() == right.keys() and all(left[name] is right[name] for name in left)


class D6DIWrapperOwnedRuntime:
    """No-model wrapper fixture for the future real wrapper-owned execution path."""

    offline_fixture_only = True
    model_loaded = False
    model_executed = False

    def __init__(
        self,
        *,
        base_model: Any,
        compiled_methods: Mapping[str, Callable[..., Any]],
        injection_counts: Mapping[str, int],
    ) -> None:
        if getattr(base_model, "model_loaded", None) is not False or getattr(
            base_model, "model_executed", None
        ) is not False:
            raise PermissionError("D6D-I implementation accepts only an unloaded fixture")
        if set(compiled_methods) != set(TARGET_METHODS) or not all(
            callable(compiled_methods[name]) for name in TARGET_METHODS
        ):
            raise TypeError("D6D-I requires the two compiled instrumented methods")
        if dict(injection_counts) != {"forward_one": 1, "forward_seq": 1}:
            raise RuntimeError("D6D-I injection counts changed")
        base_snapshot = _instance_identity_snapshot(base_model)
        request_context: ContextVar[Any] = ContextVar(
            f"psa_d6di_request_{id(self)}", default=_NO_REQUEST
        )
        dispatcher = D6DIFixedDispatcher(request_context)
        object.__setattr__(self, "_base_model", base_model)
        object.__setattr__(self, "_base_snapshot", base_snapshot)
        object.__setattr__(self, "_request_context", request_context)
        object.__setattr__(self, "_dispatcher", dispatcher)
        object.__setattr__(self, "_call_lock", threading.Lock())
        object.__setattr__(self, "execution_count", 0)
        object.__setattr__(self, "rejection_count", 0)
        object.__setattr__(self, "injection_counts", dict(injection_counts))
        object.__setattr__(self, CALLBACK_ATTRIBUTE, dispatcher)
        for name in TARGET_METHODS:
            object.__setattr__(self, name, types.MethodType(compiled_methods[name], self))
        object.__setattr__(self, "installation_count", 3)
        object.__setattr__(self, "_owned_snapshot", self._owned_identity_snapshot())
        if not self.base_dictionary_is_stable():
            raise RuntimeError("D6D-I constructor changed the base instance dictionary")

    def __getattr__(self, name: str) -> Any:
        base_model = object.__getattribute__(self, "_base_model")
        return getattr(base_model, name)

    @property
    def dispatcher(self) -> D6DIFixedDispatcher:
        return self._dispatcher

    def _owned_identity_snapshot(self) -> dict[str, Any]:
        return {
            CALLBACK_ATTRIBUTE: object.__getattribute__(self, CALLBACK_ATTRIBUTE),
            "forward_one": object.__getattribute__(self, "forward_one"),
            "forward_seq": object.__getattribute__(self, "forward_seq"),
            "forward_function": type(self).forward,
        }

    def owned_bindings_are_stable(self) -> bool:
        return _identity_snapshots_equal(
            self._owned_snapshot, self._owned_identity_snapshot()
        )

    def base_dictionary_is_stable(self) -> bool:
        return _identity_snapshots_equal(
            self._base_snapshot, _instance_identity_snapshot(self._base_model)
        )

    def context_is_empty(self) -> bool:
        return self._request_context.get() is _NO_REQUEST

    def forward(
        self,
        tokens: Sequence[int],
        state: Any,
        *,
        full_output: bool,
        coupling: D6DIRequest,
    ) -> tuple[Any, Any]:
        if type(coupling) is not D6DIRequest:
            raise PermissionError("D6D-I runtime rejects non-exact requests")
        coupling.validate()
        if not self._call_lock.acquire(blocking=False):
            object.__setattr__(self, "rejection_count", self.rejection_count + 1)
            raise RuntimeError("D6D-I rejects nested or concurrent requests")
        context_token: Token[Any] | None = None
        try:
            if not self.base_dictionary_is_stable() or not self.owned_bindings_are_stable():
                raise RuntimeError("D6D-I binding identity changed before forward")
            context_token = self._request_context.set(coupling)
            object.__setattr__(self, "execution_count", self.execution_count + 1)
            token_copy = list(tokens)
            state_copy = copy.deepcopy(state)
            if len(token_copy) > 1:
                output = self.forward_seq(token_copy, state_copy, bool(full_output))
            elif len(token_copy) == 1:
                output = self.forward_one(token_copy[0], state_copy)
            else:
                raise ValueError("D6D-I tokens must not be empty")
            if not self.base_dictionary_is_stable() or not self.owned_bindings_are_stable():
                raise RuntimeError("D6D-I binding identity changed during forward")
            return output
        finally:
            if context_token is not None:
                self._request_context.reset(context_token)
            self._call_lock.release()


def request_for_condition(
    condition: str, callback: D6DIResidualCallback | None = None
) -> D6DIRequest:
    if condition == "wrapper_off":
        return D6DIRequest(condition, False, 0.0, None)
    if condition == "wrapper_zero":
        return D6DIRequest(condition, True, 0.0, None)
    if condition == "synthetic_positive":
        return D6DIRequest(condition, True, 1.0, callback)
    if condition in SELF_CONDITIONS:
        return D6DIRequest(condition, True, 1.0, callback)
    raise ValueError("unknown D6D-I condition")
