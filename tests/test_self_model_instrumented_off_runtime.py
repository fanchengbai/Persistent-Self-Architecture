from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import textwrap
import unittest
from unittest import mock

from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    InstrumentedOffRequest,
    RWKV7InstrumentedOffRuntime,
    build_instrumented_method_asts,
    inspect_instrumented_source,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = ROOT / "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
SYNTHETIC_SOURCE = """
class RWKV_x070:
    def forward(self, idx, state, full_output=False):
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)

    def forward_one(self, idx, state):
        x = float(idx)
        for i in range(2):
            xx, state[i * 3 + 2] = RWKV_x070_CMix(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state

    def forward_seq(self, idx, state, full_output=False):
        x = [float(value) for value in idx]
        for i in range(2):
            xx, state[i * 3 + 2] = RWKV_x070_CMix(x, state[i * 3 + 2], i)
            x = x + xx
        return (x if full_output else x[-1]), state
"""


class FakeVector(list):
    def __add__(self, other):
        if isinstance(other, list):
            return FakeVector(a + b for a, b in zip(self, other))
        return NotImplemented


def RWKV_x070_CMix(x, state_value, layer_index):
    if isinstance(x, list):
        delta = FakeVector((layer_index + 1) / 10.0 for _ in x)
    else:
        delta = (layer_index + 1) / 10.0
    return delta, state_value + layer_index + 1


def _base_class():
    source = SYNTHETIC_SOURCE.replace(
        "x = [float(value) for value in idx]",
        "x = FakeVector(float(value) for value in idx)",
    )
    namespace = {"FakeVector": FakeVector, "RWKV_x070_CMix": RWKV_x070_CMix}
    exec(source, namespace)
    return source, namespace, namespace["RWKV_x070"]


def _runtime(base=None):
    source, namespace, base_class = _base_class()
    model = base or base_class()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with mock.patch(
        "psa.self_model.rwkv7_instrumented_off_runtime."
        "EXPECTED_RWKV_MODEL_SOURCE_SHA256",
        digest,
    ):
        runtime = RWKV7InstrumentedOffRuntime(
            base_model=model,
            upstream_source_bytes=source.encode("utf-8"),
            upstream_globals=namespace,
            upstream_package_version="0.8.32",
        )
    return runtime, model


def _state():
    return [0 for _ in range(6)]


class InstrumentedOffRuntimeTests(unittest.TestCase):
    def test_ast_transform_finds_one_post_ffn_site_per_path(self) -> None:
        source, _, _ = _base_class()
        methods, counts = build_instrumented_method_asts(source)
        self.assertEqual(set(methods), {"forward_one", "forward_seq"})
        self.assertEqual(counts, {"forward_one": 1, "forward_seq": 1})
        inspection = inspect_instrumented_source(source)
        self.assertTrue(inspection["valid"])
        self.assertTrue(all(inspection["checks"].values()))

    def test_ast_transform_finds_class_inside_upstream_feature_guard(self) -> None:
        source, _, _ = _base_class()
        guarded_source = "if FEATURE_ENABLED:\n" + textwrap.indent(source, "    ")
        methods, counts = build_instrumented_method_asts(guarded_source)
        self.assertEqual(set(methods), {"forward_one", "forward_seq"})
        self.assertEqual(counts, {"forward_one": 1, "forward_seq": 1})

    def test_ast_transform_finds_methods_inside_class_feature_guard(self) -> None:
        source, _, _ = _base_class()
        tree = ast.parse(source)
        class_node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef)
        )
        methods = [
            node for node in class_node.body if isinstance(node, ast.FunctionDef)
        ]
        class_node.body = [
            ast.If(
                test=ast.Name(id="FEATURE_ENABLED", ctx=ast.Load()),
                body=methods,
                orelse=[],
            )
        ]
        guarded_source = ast.unparse(ast.fix_missing_locations(tree))
        transformed, counts = build_instrumented_method_asts(guarded_source)
        self.assertEqual(set(transformed), {"forward_one", "forward_seq"})
        self.assertEqual(counts, {"forward_one": 1, "forward_seq": 1})

    def test_off_runtime_matches_original_for_both_dispatch_paths(self) -> None:
        runtime, model = _runtime()
        baseline_one = model.forward([3], _state(), False)
        off_one = runtime.forward([3], _state(), False)
        baseline_seq_last = model.forward([3, 5, 8], _state(), False)
        off_seq_last = runtime.forward([3, 5, 8], _state(), False)
        baseline_seq_full = model.forward([3, 5, 8], _state(), True)
        off_seq_full = runtime.forward([3, 5, 8], _state(), True)
        self.assertEqual(off_one, baseline_one)
        self.assertEqual(off_seq_last, baseline_seq_last)
        self.assertEqual(off_seq_full, baseline_seq_full)
        self.assertEqual(runtime.execution_count, 3)
        self.assertEqual(runtime.callback_call_count, 0)
        self.assertFalse(runtime.self_projection_constructed)

    def test_runtime_restores_base_model_after_each_call(self) -> None:
        runtime, model = _runtime()
        before = dict(model.__dict__)
        runtime.forward([3], _state())
        self.assertEqual(model.__dict__, before)
        self.assertNotIn(CALLBACK_ATTRIBUTE, model.__dict__)
        self.assertNotIn("forward_one", model.__dict__)
        self.assertNotIn("forward_seq", model.__dict__)

    def test_runtime_restores_base_model_after_upstream_exception(self) -> None:
        runtime, model = _runtime()
        with self.assertRaises(TypeError):
            runtime.forward(None, _state())
        self.assertNotIn(CALLBACK_ATTRIBUTE, model.__dict__)
        self.assertNotIn("forward_one", model.__dict__)
        self.assertNotIn("forward_seq", model.__dict__)

    def test_active_malformed_and_conflicting_paths_fail_closed(self) -> None:
        runtime, model = _runtime()

        class OffSubclass(InstrumentedOffRequest):
            pass

        for request in (
            {"enabled": True},
            {"scale": 1.0},
            {"mode": "active"},
        ):
            with self.assertRaises(PermissionError):
                InstrumentedOffRequest(**request)
        with self.assertRaises(PermissionError):
            runtime.forward([3], _state(), coupling={"enabled": True})
        with self.assertRaises(PermissionError):
            runtime.forward([3], _state(), coupling=OffSubclass())
        with self.assertRaises(PermissionError):
            runtime.forward_active([3], _state())
        self.assertEqual(runtime.execution_count, 0)

        setattr(model, CALLBACK_ATTRIBUTE, object())
        with self.assertRaises(RuntimeError):
            runtime.forward([3], _state())
        delattr(model, CALLBACK_ATTRIBUTE)

    def test_wrong_version_digest_or_source_shape_fails_before_execution(self) -> None:
        source, namespace, base_class = _base_class()
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        with mock.patch(
            "psa.self_model.rwkv7_instrumented_off_runtime."
            "EXPECTED_RWKV_MODEL_SOURCE_SHA256",
            digest,
        ):
            with self.assertRaises(RuntimeError):
                RWKV7InstrumentedOffRuntime(
                    base_model=base_class(),
                    upstream_source_bytes=source.encode("utf-8"),
                    upstream_globals=namespace,
                    upstream_package_version="0.8.33",
                )
        with self.assertRaises(RuntimeError):
            build_instrumented_method_asts(
                source.replace("x = x + xx", "x = xx + x", 1)
            )

    def test_runtime_module_has_no_rwkv_or_torch_import(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertNotIn("rwkv", imported)
        self.assertNotIn("torch", imported)
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)


if __name__ == "__main__":
    unittest.main()
