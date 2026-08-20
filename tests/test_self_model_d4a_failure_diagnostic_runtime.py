from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

from psa.self_model.d4a_failure_diagnostic_runtime import (
    D4ADiagnosticOffRequest,
    D4A_RECORDED_ROUNDS,
    RWKV7RecompiledUnmodifiedRuntime,
    execute_d4a_fake_or_authorized_diagnostic,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    RWKV7InstrumentedOffRuntime,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = ROOT / "src/psa/self_model/d4a_failure_diagnostic_runtime.py"
SYNTHETIC_SOURCE = """
class RWKV_x070:
    def forward(self, idx, state, full_output=False):
        if state is None:
            state = [FakeTensor([0.0]) for _ in range(6)]
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)

    @identity
    def forward_one(self, idx, state):
        x = FakeTensor([float(idx)])
        for i in range(2):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_one(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state

    @identity
    def forward_seq(self, idx, state, full_output=False):
        x = FakeTensor([float(value) for value in idx])
        for i in range(2):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(x, state[i * 3 + 2], i)
            x = x + xx
        return (x if full_output else FakeTensor([x.values[-1]])), state
"""


class FakeBytes:
    def __init__(self, payload):
        self.payload = payload

    def numpy(self):
        return self

    def tobytes(self):
        return self.payload


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, values, *, dtype="fake.float16", device="cuda:0"):
        self.values = [float(value) for value in values]
        self.shape = (len(self.values),)
        self.dtype = dtype
        self.device = device

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype, device=self.device)

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def view(self, dtype):
        return FakeBytes(b"".join(struct.pack("<d", value) for value in self.values))

    def numel(self):
        return len(self.values)

    def float(self):
        return self

    def abs(self):
        return FakeTensor([abs(value) for value in self.values])

    def max(self):
        return FakeScalar(max(self.values))

    def mean(self):
        return FakeScalar(sum(self.values) / len(self.values))

    def sum(self):
        return FakeScalar(sum(self.values))

    def __add__(self, other):
        return FakeTensor(a + b for a, b in zip(self.values, other.values))

    def __sub__(self, other):
        return FakeTensor(a - b for a, b in zip(self.values, other.values))

    def __ne__(self, other):
        return [a != b for a, b in zip(self.values, other.values)]


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    uint8 = object()

    @staticmethod
    def equal(left, right):
        return (
            left.values == right.values
            and left.dtype == right.dtype
            and left.device == right.device
        )

    @staticmethod
    def count_nonzero(values):
        return FakeScalar(sum(bool(value) for value in values))

    @staticmethod
    def inference_mode():
        return _InferenceMode()


def identity(function):
    return function


def fake_cmix(x, state_value, layer_index):
    delta = FakeTensor((layer_index + 1) / 10.0 for _ in x.values)
    state_delta = FakeTensor([layer_index + 1.0])
    return delta, state_value + state_delta


def _base_class():
    namespace = {
        "FakeTensor": FakeTensor,
        "identity": identity,
        "RWKV_x070_CMix_one": fake_cmix,
        "RWKV_x070_CMix_seq": fake_cmix,
    }
    exec(SYNTHETIC_SOURCE, namespace)
    return namespace, namespace["RWKV_x070"]


def _runtimes():
    namespace, base_class = _base_class()
    base = base_class()
    source_bytes = SYNTHETIC_SOURCE.encode("utf-8")
    digest = hashlib.sha256(source_bytes).hexdigest()
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                "psa.self_model.d4a_failure_diagnostic_runtime."
                "EXPECTED_RWKV_MODEL_SOURCE_SHA256",
                digest,
            )
        )
        stack.enter_context(
            mock.patch(
                "psa.self_model.rwkv7_instrumented_off_runtime."
                "EXPECTED_RWKV_MODEL_SOURCE_SHA256",
                digest,
            )
        )
        g0 = RWKV7RecompiledUnmodifiedRuntime(
            base_model=base,
            upstream_source_bytes=source_bytes,
            upstream_globals=namespace,
            upstream_package_version="0.8.32",
            upstream_de_version=None,
        )
        g2 = RWKV7InstrumentedOffRuntime(
            base_model=base,
            upstream_source_bytes=source_bytes,
            upstream_globals=namespace,
            upstream_package_version="0.8.32",
            upstream_de_version=None,
        )
    return base, g0, g2


class PerturbingG2:
    def __init__(self, inner):
        self.inner = inner

    @property
    def execution_count(self):
        return self.inner.execution_count

    @property
    def callback_call_count(self):
        return self.inner.callback_call_count

    @property
    def self_projection_constructed(self):
        return self.inner.self_projection_constructed

    def forward(self, tokens, state, full_output=False):
        logits, next_state = self.inner.forward(tokens, state, full_output)
        return logits + FakeTensor([1.0]), next_state


class D4AFailureDiagnosticRuntimeTests(unittest.TestCase):
    def test_g0_recompiles_unmodified_methods_and_restores_bindings(self):
        base, g0, _ = _runtimes()
        baseline = base.forward([2764], None, False)
        observed = g0.forward([2764], None, False)
        self.assertEqual(baseline[0].values, observed[0].values)
        self.assertEqual(
            [tensor.values for tensor in baseline[1]],
            [tensor.values for tensor in observed[1]],
        )
        self.assertEqual(g0.execution_count, 1)
        self.assertEqual(g0.callback_call_count, 0)
        self.assertFalse(g0.self_projection_constructed)
        self.assertEqual(
            g0.variant_selection["forward_one"]["original_decorators"],
            ["identity"],
        )
        for name in ("forward_one", "forward_seq", CALLBACK_ATTRIBUTE):
            self.assertNotIn(name, base.__dict__)

    def test_balanced_nine_call_fake_diagnostic_records_every_tensor(self):
        base, g0, g2 = _runtimes()
        report = execute_d4a_fake_or_authorized_diagnostic(
            base_model=base, g0=g0, off_g2=g2, torch=FakeTorch
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["schedule"], D4A_RECORDED_ROUNDS)
        self.assertEqual(len(report["calls"]), 9)
        self.assertEqual(len(report["comparisons"]["within_route"]), 9)
        self.assertEqual(len(report["comparisons"]["cross_route"]), 27)
        self.assertEqual(
            report["diagnostic_classification"],
            "d4_failure_not_reproduced_in_balanced_diagnostic",
        )
        self.assertTrue(
            all(len(call["state"]["components"]) == 6 for call in report["calls"])
        )
        self.assertFalse(report["d4_status_changed"])
        self.assertFalse(report["d5_authorized"])

    def test_g2_only_difference_is_classified_without_changing_d4(self):
        base, g0, g2 = _runtimes()
        report = execute_d4a_fake_or_authorized_diagnostic(
            base_model=base,
            g0=g0,
            off_g2=PerturbingG2(g2),
            torch=FakeTorch,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["diagnostic_classification"],
            "none_guarded_instrumentation_difference",
        )
        self.assertFalse(report["d4_status_changed"])
        self.assertFalse(report["d5_authorized"])
        mismatches = [
            item
            for item in report["comparisons"]["cross_route"]
            if not item["all_torch_equal"]
        ]
        self.assertTrue(mismatches)
        self.assertTrue(all(item["logits"]["max_abs_error"] == 1.0 for item in mismatches))

    def test_active_malformed_conflict_and_source_locks_fail_closed(self):
        base, g0, _ = _runtimes()
        for values in ({"mode": "active"}, {"enabled": True}, {"scale": 1.0}):
            with self.assertRaises(PermissionError):
                D4ADiagnosticOffRequest(**values)
        with self.assertRaises(PermissionError):
            g0.forward([2764], None, coupling={"enabled": True})
        with self.assertRaises(PermissionError):
            g0.forward_active([2764], None)
        setattr(base, CALLBACK_ATTRIBUTE, None)
        with self.assertRaises(RuntimeError):
            g0.forward([2764], None)
        delattr(base, CALLBACK_ATTRIBUTE)

        namespace, base_class = _base_class()
        with self.assertRaises(RuntimeError):
            RWKV7RecompiledUnmodifiedRuntime(
                base_model=base_class(),
                upstream_source_bytes=SYNTHETIC_SOURCE.encode("utf-8"),
                upstream_globals=namespace,
                upstream_package_version="0.8.33",
                upstream_de_version=None,
            )

    def test_runtime_module_does_not_import_rwkv_or_torch(self):
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import rwkv", source)
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)


if __name__ == "__main__":
    unittest.main()
