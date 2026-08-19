from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import types
from typing import Any, Mapping

from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)


CALLBACK_ATTRIBUTE = "_psa_post_ffn_residual_callback"
TARGET_CLASS = "RWKV_x070"
TARGET_METHODS = ("forward_one", "forward_seq")
RWKV_DE_VERSION_CONDITION = "os.environ.get('RWKV_DE_VERSION') == '1'"


@dataclass(frozen=True)
class InstrumentedOffRequest:
    mode: str = "off"
    enabled: bool = False
    scale: float = 0.0

    def __post_init__(self) -> None:
        if self.mode != "off" or self.enabled is not False or self.scale != 0.0:
            raise PermissionError("OFF-G2 accepts only an exact off request")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_cmix_assignment(node: ast.stmt) -> bool:
    return bool(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == "RWKV_x070_CMix"
    )


def _is_x_plus_xx_assignment(node: ast.stmt) -> bool:
    return bool(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "x"
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Add)
        and isinstance(node.value.left, ast.Name)
        and node.value.left.id == "x"
        and isinstance(node.value.right, ast.Name)
        and node.value.right.id == "xx"
    )


def _callback_statement(execution_path: str) -> ast.If:
    source = f"""
if self.{CALLBACK_ATTRIBUTE} is not None:
    x = self.{CALLBACK_ATTRIBUTE}(
        phase="post_ffn_residual",
        layer_index=i,
        execution_path="{execution_path}",
        residual_x=x,
    )
"""
    statement = ast.parse(source).body[0]
    if not isinstance(statement, ast.If):
        raise AssertionError("callback template must produce an if statement")
    return statement


class _PostFFNInjector(ast.NodeTransformer):
    def __init__(self, execution_path: str) -> None:
        self.execution_path = execution_path
        self.injection_count = 0

    def _instrument_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        index = 0
        while index < len(body):
            current = body[index]
            rewritten.append(current)
            if _is_cmix_assignment(current):
                if index + 1 >= len(body) or not _is_x_plus_xx_assignment(
                    body[index + 1]
                ):
                    raise RuntimeError(
                        f"{self.execution_path} CMix is not followed by x = x + xx"
                    )
                residual_add = body[index + 1]
                rewritten.append(residual_add)
                callback = ast.copy_location(
                    _callback_statement(self.execution_path), residual_add
                )
                rewritten.append(callback)
                self.injection_count += 1
                index += 2
                continue
            index += 1
        return rewritten

    def generic_visit(self, node: ast.AST) -> ast.AST:
        node = super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list) and all(
                isinstance(item, ast.stmt) for item in value
            ):
                setattr(node, field, self._instrument_body(value))
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                handler.body = self._instrument_body(handler.body)
        return node


def _build_instrumented_method_plan(
    upstream_source: str,
    *,
    rwkv_de_version: str | None,
) -> tuple[
    dict[str, ast.FunctionDef],
    dict[str, int],
    dict[str, dict[str, Any]],
]:
    if rwkv_de_version is not None:
        raise PermissionError("OFF-G2 requires RWKV_DE_VERSION to be unset")
    tree = ast.parse(upstream_source)
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == TARGET_CLASS
    ]
    if len(classes) != 1:
        raise RuntimeError("locked upstream source must contain one RWKV_x070 class")
    class_node = classes[0]
    parent_by_node = {
        child: parent
        for parent in ast.walk(class_node)
        for child in ast.iter_child_nodes(parent)
    }
    transformed: dict[str, ast.FunctionDef] = {}
    counts: dict[str, int] = {}
    variants: dict[str, dict[str, Any]] = {}
    for method_name in TARGET_METHODS:
        matches = [
            node
            for node in ast.walk(class_node)
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

        injection_counts = []
        selected_method = None
        selected_count = None
        source_lines = []
        for method in matches:
            source_lines.append(method.lineno)
            method.decorator_list = []
            injector = _PostFFNInjector(method_name)
            transformed_method = injector.visit(method)
            if not isinstance(transformed_method, ast.FunctionDef):
                raise AssertionError("instrumented method must remain a function")
            if injector.injection_count != 1:
                raise RuntimeError(
                    f"{method_name} variants must each have one post-FFN site"
                )
            injection_counts.append(injector.injection_count)
            if method is selected:
                selected_method = transformed_method
                selected_count = injector.injection_count
        if selected_method is None or selected_count is None:
            raise AssertionError("selected method variant was not transformed")
        transformed[method_name] = selected_method
        counts[method_name] = selected_count
        variants[method_name] = {
            "candidate_count": len(matches),
            "condition": condition,
            "selected_branch": selected_branch,
            "source_lines": sorted(source_lines),
            "injection_counts": injection_counts,
            "selected_source_line": selected.lineno,
        }
    return transformed, counts, variants


def build_instrumented_method_asts(
    upstream_source: str,
    *,
    rwkv_de_version: str | None = None,
) -> tuple[dict[str, ast.FunctionDef], dict[str, int]]:
    methods, counts, _ = _build_instrumented_method_plan(
        upstream_source, rwkv_de_version=rwkv_de_version
    )
    return methods, counts


def compile_instrumented_methods(
    *,
    upstream_source: str,
    upstream_globals: Mapping[str, Any],
    rwkv_de_version: str | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    methods, counts = build_instrumented_method_asts(
        upstream_source, rwkv_de_version=rwkv_de_version
    )
    module = ast.Module(body=list(methods.values()), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(upstream_globals)
    exec(compile(module, "<psa-rwkv7-instrumented-off>", "exec"), namespace)
    compiled = {name: namespace[name] for name in TARGET_METHODS}
    if not all(callable(value) for value in compiled.values()):
        raise RuntimeError("instrumented methods did not compile as callables")
    return compiled, counts


def inspect_instrumented_source(upstream_source: str) -> dict[str, Any]:
    methods, counts, variants = _build_instrumented_method_plan(
        upstream_source, rwkv_de_version=None
    )
    rendered = {}
    for name, method in methods.items():
        module = ast.Module(body=[method], type_ignores=[])
        ast.fix_missing_locations(module)
        rendered[name] = ast.unparse(module)
    checks = {
        "target_class_is_rwkv_x070": TARGET_CLASS == "RWKV_x070",
        "both_execution_paths_transformed": set(methods) == set(TARGET_METHODS),
        "one_injection_per_execution_path": counts
        == {"forward_one": 1, "forward_seq": 1},
        "callback_attribute_is_project_namespaced": CALLBACK_ATTRIBUTE.startswith(
            "_psa_"
        ),
        "forward_one_phase_is_post_ffn": (
            'phase="post_ffn_residual"' in rendered["forward_one"]
            or "phase='post_ffn_residual'" in rendered["forward_one"]
        ),
        "forward_seq_phase_is_post_ffn": (
            'phase="post_ffn_residual"' in rendered["forward_seq"]
            or "phase='post_ffn_residual'" in rendered["forward_seq"]
        ),
        "forward_one_path_named": "execution_path='forward_one'"
        in rendered["forward_one"],
        "forward_seq_path_named": "execution_path='forward_seq'"
        in rendered["forward_seq"],
        "callback_branch_is_none_guarded": all(
            f"self.{CALLBACK_ATTRIBUTE} is not None" in source
            for source in rendered.values()
        ),
        "rwkv_de_version_is_unset": True,
        "variant_selection_is_consistent": all(
            details["selected_branch"]
            in {"only_definition", "else_rwkv_de_version_unset"}
            for details in variants.values()
        ),
        "all_variants_have_one_injection": all(
            all(count == 1 for count in details["injection_counts"])
            for details in variants.values()
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "injection_counts": counts,
        "variant_selection": variants,
        "method_source_sha256": {
            name: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for name, source in rendered.items()
        },
    }


class RWKV7InstrumentedOffRuntime:
    """OFF-G2 runtime: instrumented methods with a strictly absent callback."""

    runtime_version = "0.1-instrumented-off"
    development_only = True
    off_g1_implemented = True
    off_g2_implemented = True
    active_injection_available = False
    real_layers_selected = False
    real_sequence_policy_frozen = False

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
            raise RuntimeError("RWKV package version differs from the OFF-G2 lock")
        source_digest = hashlib.sha256(upstream_source_bytes).hexdigest()
        if source_digest != EXPECTED_RWKV_MODEL_SOURCE_SHA256:
            raise RuntimeError("RWKV model source differs from the OFF-G2 lock")
        if upstream_de_version is not None:
            raise PermissionError("OFF-G2 requires RWKV_DE_VERSION to be unset")
        if not callable(getattr(base_model, "forward", None)):
            raise TypeError("base_model must expose a callable forward")
        if CALLBACK_ATTRIBUTE in getattr(base_model, "__dict__", {}):
            raise RuntimeError("base_model already owns the PSA callback attribute")
        source = upstream_source_bytes.decode("utf-8")
        methods, counts = compile_instrumented_methods(
            upstream_source=source,
            upstream_globals=upstream_globals,
            rwkv_de_version=upstream_de_version,
        )
        self._base_model = base_model
        self._methods = methods
        self._injection_counts = counts
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
    def injection_counts(self) -> dict[str, int]:
        return dict(self._injection_counts)

    def forward(
        self,
        tokens: Any,
        state: Any,
        full_output: bool = False,
        *,
        coupling: InstrumentedOffRequest | None = None,
    ) -> Any:
        request = InstrumentedOffRequest() if coupling is None else coupling
        if type(request) is not InstrumentedOffRequest:
            raise PermissionError("OFF-G2 rejects non-off coupling requests")
        if request.mode != "off" or request.enabled or request.scale != 0.0:
            raise PermissionError("OFF-G2 active coupling is unavailable")

        instance_dict = getattr(self._base_model, "__dict__", None)
        if not isinstance(instance_dict, dict):
            raise TypeError("base_model must expose a mutable instance dictionary")
        managed_names = (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
        if any(name in instance_dict for name in managed_names):
            raise RuntimeError("base_model has conflicting instance overrides")
        try:
            setattr(self._base_model, CALLBACK_ATTRIBUTE, None)
            for name, function in self._methods.items():
                setattr(self._base_model, name, types.MethodType(function, self._base_model))
            self._execution_count += 1
            return self._base_model.forward(tokens, state, full_output)
        finally:
            for name in managed_names:
                instance_dict.pop(name, None)

    def forward_active(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError(
            "active injection is not implemented or authorized in OFF-G2"
        )
