from __future__ import annotations

import ast
import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from itertools import combinations, product
import types
from typing import Any, Iterable, Mapping

from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    RWKV_DE_VERSION_CONDITION,
    TARGET_CLASS,
    TARGET_METHODS,
)


D4A_RUNTIME_VERSION = "0.1-d4a-diagnostic-runtime"
D4A_TOKEN_IDS = [2764]
D4A_RECORDED_ROUNDS = [
    ["original_baseline", "g0_recompiled_unmodified", "off_g2_instrumented"],
    ["g0_recompiled_unmodified", "off_g2_instrumented", "original_baseline"],
    ["off_g2_instrumented", "original_baseline", "g0_recompiled_unmodified"],
]


@dataclass(frozen=True)
class D4ADiagnosticOffRequest:
    mode: str = "off"
    enabled: bool = False
    scale: float = 0.0

    def __post_init__(self) -> None:
        if self.mode != "off" or self.enabled is not False or self.scale != 0.0:
            raise PermissionError("D4A accepts only an exact diagnostic-off request")


def _select_unmodified_method_asts(
    upstream_source: str, *, rwkv_de_version: str | None
) -> tuple[dict[str, ast.FunctionDef], dict[str, dict[str, Any]]]:
    if rwkv_de_version is not None:
        raise PermissionError("D4A G0 requires RWKV_DE_VERSION to be unset")
    tree = ast.parse(upstream_source)
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == TARGET_CLASS
    ]
    if len(classes) != 1:
        raise RuntimeError("locked upstream source must contain one RWKV_x070 class")
    target_class = classes[0]
    parent_by_node = {
        child: parent
        for parent in ast.walk(target_class)
        for child in ast.iter_child_nodes(parent)
    }
    selected_methods: dict[str, ast.FunctionDef] = {}
    selections: dict[str, dict[str, Any]] = {}
    for method_name in TARGET_METHODS:
        matches = [
            node
            for node in ast.walk(target_class)
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        ]
        if len(matches) not in {1, 2}:
            raise RuntimeError(f"locked upstream source must contain {method_name}")
        selected = matches[0]
        selected_branch = "only_definition"
        condition = None
        if len(matches) == 2:
            parents = [parent_by_node.get(method) for method in matches]
            if (
                not all(isinstance(parent, ast.If) for parent in parents)
                or parents[0] is not parents[1]
            ):
                raise RuntimeError(
                    f"{method_name} variants must share one source condition"
                )
            variant_if = parents[0]
            if not isinstance(variant_if, ast.If):
                raise AssertionError("variant parent must be an if statement")
            condition = ast.unparse(variant_if.test)
            if condition != RWKV_DE_VERSION_CONDITION:
                raise RuntimeError(
                    f"{method_name} variants use an unexpected condition"
                )
            body_matches = [item for item in variant_if.body if item in matches]
            else_matches = [item for item in variant_if.orelse if item in matches]
            if len(body_matches) != 1 or len(else_matches) != 1:
                raise RuntimeError(
                    f"{method_name} variants must occupy body and else branches"
                )
            selected = else_matches[0]
            selected_branch = "else_rwkv_de_version_unset"
        cloned = copy.deepcopy(selected)
        original_decorators = [ast.unparse(item) for item in cloned.decorator_list]
        cloned.decorator_list = []
        selected_methods[method_name] = cloned
        selections[method_name] = {
            "candidate_count": len(matches),
            "condition": condition,
            "selected_branch": selected_branch,
            "selected_source_line": selected.lineno,
            "original_decorators": original_decorators,
            "compiled_decorators": [],
        }
    return selected_methods, selections


def compile_unmodified_methods(
    *,
    upstream_source: str,
    upstream_globals: Mapping[str, Any],
    rwkv_de_version: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    methods, selections = _select_unmodified_method_asts(
        upstream_source, rwkv_de_version=rwkv_de_version
    )
    module = ast.Module(body=list(methods.values()), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(upstream_globals)
    exec(compile(module, "<psa-rwkv7-recompiled-unmodified>", "exec"), namespace)
    compiled = {name: namespace[name] for name in TARGET_METHODS}
    if not all(callable(value) for value in compiled.values()):
        raise RuntimeError("G0 methods did not compile as callables")
    return compiled, selections


class RWKV7RecompiledUnmodifiedRuntime:
    """G0: reproduce OFF-G2's compile/bind boundary without instrumentation."""

    runtime_version = D4A_RUNTIME_VERSION
    development_only = True
    diagnostic_only = True
    active_injection_available = False

    def __init__(
        self,
        *,
        base_model: Any,
        upstream_source_bytes: bytes,
        upstream_globals: Mapping[str, Any],
        upstream_package_version: str,
        upstream_de_version: str | None,
    ) -> None:
        if upstream_package_version != EXPECTED_RWKV_PACKAGE_VERSION:
            raise RuntimeError("RWKV package version differs from the D4A lock")
        source_digest = hashlib.sha256(upstream_source_bytes).hexdigest()
        if source_digest != EXPECTED_RWKV_MODEL_SOURCE_SHA256:
            raise RuntimeError("RWKV model source differs from the D4A lock")
        if upstream_de_version is not None:
            raise PermissionError("D4A G0 requires RWKV_DE_VERSION to be unset")
        if not callable(getattr(base_model, "forward", None)):
            raise TypeError("base_model must expose a callable forward")
        source = upstream_source_bytes.decode("utf-8")
        methods, selections = compile_unmodified_methods(
            upstream_source=source,
            upstream_globals=upstream_globals,
            rwkv_de_version=upstream_de_version,
        )
        self._base_model = base_model
        self._methods = methods
        self._selections = selections
        self._execution_count = 0

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def callback_call_count(self) -> int:
        return 0

    @property
    def self_projection_constructed(self) -> bool:
        return False

    @property
    def variant_selection(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._selections)

    def forward(
        self,
        tokens: Any,
        state: Any,
        full_output: bool = False,
        *,
        coupling: D4ADiagnosticOffRequest | None = None,
    ) -> Any:
        request = D4ADiagnosticOffRequest() if coupling is None else coupling
        if type(request) is not D4ADiagnosticOffRequest:
            raise PermissionError("D4A G0 rejects non-off requests")
        instance_dict = getattr(self._base_model, "__dict__", None)
        if not isinstance(instance_dict, dict):
            raise TypeError("base_model must expose a mutable instance dictionary")
        if CALLBACK_ATTRIBUTE in instance_dict or any(
            name in instance_dict for name in TARGET_METHODS
        ):
            raise RuntimeError("base_model has conflicting instance overrides")
        try:
            for name, function in self._methods.items():
                setattr(self._base_model, name, types.MethodType(function, self._base_model))
            self._execution_count += 1
            return self._base_model.forward(tokens, state, full_output)
        finally:
            for name in TARGET_METHODS:
                instance_dict.pop(name, None)

    def forward_active(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("active injection is unavailable in D4A G0")


def _flatten_tensors(value: Any, path: str = "state") -> Iterable[tuple[str, Any]]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten_tensors(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_tensors(value[key], f"{path}.{key}")
    elif value is not None:
        raise TypeError(f"unsupported state component at {path}")


def _tensor_digest(tensor: Any, torch: Any) -> str:
    byte_view = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _tensor_record(tensor: Any, torch: Any) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "sha256": _tensor_digest(detached, torch),
    }


def _tensor_comparison(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    device_equal = str(left.device) == str(right.device)
    compatible = shape_equal and dtype_equal and device_equal
    exact = bool(torch.equal(left, right)) if compatible else False
    if compatible:
        difference = (left.detach().float() - right.detach().float()).abs()
        unequal = int(torch.count_nonzero(left != right).item())
        max_abs = float(difference.max().item()) if difference.numel() else 0.0
        mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
        sum_abs = float(difference.sum().item()) if difference.numel() else 0.0
        numel = int(difference.numel())
    else:
        unequal = None
        max_abs = None
        mean_abs = None
        sum_abs = None
        numel = 0
    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "device_equal": device_equal,
        "torch_equal": exact,
        "unequal_element_count": unequal,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "_sum_abs_error": sum_abs,
        "_numel": numel,
        "valid": compatible,
    }


def _state_comparison(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    left_items = list(_flatten_tensors(left))
    right_items = list(_flatten_tensors(right))
    paths_equal = [path for path, _ in left_items] == [path for path, _ in right_items]
    first_mismatch = None
    unequal_total = 0
    max_abs = 0.0
    sum_abs = 0.0
    numel = 0
    all_equal = paths_equal
    compatible = paths_equal
    if paths_equal:
        for (path, left_tensor), (_, right_tensor) in zip(left_items, right_items):
            item = _tensor_comparison(left_tensor, right_tensor, torch)
            compatible = compatible and item["valid"]
            all_equal = all_equal and item["torch_equal"]
            if not item["torch_equal"] and first_mismatch is None:
                first_mismatch = path
            if item["unequal_element_count"] is not None:
                unequal_total += item["unequal_element_count"]
            if item["max_abs_error"] is not None:
                max_abs = max(max_abs, item["max_abs_error"])
                sum_abs += item["_sum_abs_error"]
                numel += item["_numel"]
    return {
        "paths_equal": paths_equal,
        "component_count": len(left_items) if paths_equal else 0,
        "shape_dtype_device_compatible": compatible,
        "all_tensors_torch_equal": all_equal,
        "unequal_element_count": unequal_total if paths_equal else None,
        "max_abs_error": max_abs if paths_equal else None,
        "mean_abs_error": (sum_abs / numel) if paths_equal and numel else 0.0,
        "first_mismatch_component": first_mismatch,
    }


def _invoke(route: Any, torch: Any) -> tuple[Any, Any]:
    inference_mode = getattr(torch, "inference_mode", None)
    context = inference_mode() if callable(inference_mode) else nullcontext()
    with context:
        return route.forward(list(D4A_TOKEN_IDS), None, False)


def _call_record(
    *,
    call_id: str,
    route: str,
    round_index: int,
    order_position: int,
    logits: Any,
    state: Any,
    torch: Any,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "route": route,
        "round_index": round_index,
        "order_position": order_position,
        "token_ids": list(D4A_TOKEN_IDS),
        "state_input": "none",
        "full_output": False,
        "logits": _tensor_record(logits, torch),
        "state": {
            "component_count": len(list(_flatten_tensors(state))),
            "components": [
                {"path": path, **_tensor_record(tensor, torch)}
                for path, tensor in _flatten_tensors(state)
            ],
        },
    }


def _pair_record(
    left_call: dict[str, Any],
    right_call: dict[str, Any],
    left_output: tuple[Any, Any],
    right_output: tuple[Any, Any],
    torch: Any,
) -> dict[str, Any]:
    logits_raw = _tensor_comparison(left_output[0], right_output[0], torch)
    logits = {key: value for key, value in logits_raw.items() if not key.startswith("_")}
    state = _state_comparison(left_output[1], right_output[1], torch)
    return {
        "left_call_id": left_call["call_id"],
        "right_call_id": right_call["call_id"],
        "left_route": left_call["route"],
        "right_route": right_call["route"],
        "logits": logits,
        "state": state,
        "all_torch_equal": logits["torch_equal"] and state["all_tensors_torch_equal"],
    }


def _diagnostic_classification(
    within_route: list[dict[str, Any]], cross_route: list[dict[str, Any]]
) -> str:
    within_stable = all(item["all_torch_equal"] for item in within_route)
    if not within_stable:
        return "within_route_instability_observed"

    def route_pair_exact(left: str, right: str) -> bool:
        relevant = [
            item
            for item in cross_route
            if {item["left_route"], item["right_route"]} == {left, right}
        ]
        return bool(relevant) and all(item["all_torch_equal"] for item in relevant)

    original_g0 = route_pair_exact("original_baseline", "g0_recompiled_unmodified")
    original_g2 = route_pair_exact("original_baseline", "off_g2_instrumented")
    g0_g2 = route_pair_exact("g0_recompiled_unmodified", "off_g2_instrumented")
    if original_g0 and original_g2 and g0_g2:
        return "d4_failure_not_reproduced_in_balanced_diagnostic"
    if not original_g0 and g0_g2:
        return "recompile_or_binding_boundary_difference"
    if original_g0 and not original_g2:
        return "none_guarded_instrumentation_difference"
    return "mixed_or_unresolved_route_difference"


def execute_d4a_fake_or_authorized_diagnostic(
    *, base_model: Any, g0: Any, off_g2: Any, torch: Any
) -> dict[str, Any]:
    """Execute the frozen diagnostic core; this function never loads a model."""
    routes = {
        "original_baseline": base_model,
        "g0_recompiled_unmodified": g0,
        "off_g2_instrumented": off_g2,
    }
    g0_before = g0.execution_count
    g2_before = off_g2.execution_count
    calls: list[dict[str, Any]] = []
    outputs: list[tuple[Any, Any]] = []
    for round_index, round_routes in enumerate(D4A_RECORDED_ROUNDS, start=1):
        for order_position, route_name in enumerate(round_routes, start=1):
            output = _invoke(routes[route_name], torch)
            call_id = f"round-{round_index}-position-{order_position}-{route_name}"
            calls.append(
                _call_record(
                    call_id=call_id,
                    route=route_name,
                    round_index=round_index,
                    order_position=order_position,
                    logits=output[0],
                    state=output[1],
                    torch=torch,
                )
            )
            outputs.append(output)

    indexes_by_route = {
        route: [index for index, call in enumerate(calls) if call["route"] == route]
        for route in routes
    }
    within_route = []
    for indexes in indexes_by_route.values():
        for left, right in combinations(indexes, 2):
            within_route.append(
                _pair_record(calls[left], calls[right], outputs[left], outputs[right], torch)
            )
    cross_route = []
    for left_route, right_route in combinations(routes, 2):
        for left, right in product(indexes_by_route[left_route], indexes_by_route[right_route]):
            cross_route.append(
                _pair_record(calls[left], calls[right], outputs[left], outputs[right], torch)
            )

    instance_dict = getattr(base_model, "__dict__", {})
    bindings_restored = all(
        name not in instance_dict for name in (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
    )
    checks = {
        "nine_calls_recorded": len(calls) == 9,
        "three_calls_per_route": all(len(indexes) == 3 for indexes in indexes_by_route.values()),
        "each_route_occupies_each_position_once": all(
            sorted(calls[index]["order_position"] for index in indexes) == [1, 2, 3]
            for indexes in indexes_by_route.values()
        ),
        "all_calls_use_frozen_failed_fixture": all(
            call["token_ids"] == [2764]
            and call["state_input"] == "none"
            and call["full_output"] is False
            for call in calls
        ),
        "all_call_tensors_inventoried": all(
            call["logits"]["sha256"]
            and call["state"]["component_count"] == len(call["state"]["components"])
            and all(component["sha256"] for component in call["state"]["components"])
            for call in calls
        ),
        "all_within_route_pairs_recorded": len(within_route) == 9,
        "all_cross_route_pairs_recorded": len(cross_route) == 27,
        "g0_executed_exactly_three_times": g0.execution_count - g0_before == 3,
        "off_g2_executed_exactly_three_times": off_g2.execution_count - g2_before == 3,
        "g0_callback_count_zero": g0.callback_call_count == 0,
        "off_g2_callback_count_zero": off_g2.callback_call_count == 0,
        "g0_projection_not_constructed": not g0.self_projection_constructed,
        "off_g2_projection_not_constructed": not off_g2.self_projection_constructed,
        "temporary_bindings_restored": bindings_restored,
    }
    return {
        "runtime_version": D4A_RUNTIME_VERSION,
        "status": "d4a_diagnostic_core_complete",
        "valid": all(checks.values()),
        "development_only": True,
        "diagnostic_only": True,
        "d4_status_changed": False,
        "d5_authorized": False,
        "schedule": copy.deepcopy(D4A_RECORDED_ROUNDS),
        "calls": calls,
        "comparisons": {
            "within_route": within_route,
            "cross_route": cross_route,
        },
        "diagnostic_classification": _diagnostic_classification(
            within_route, cross_route
        ),
        "checks": checks,
        "safety": {
            "active_injection_executed": False,
            "self_projection_constructed": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
