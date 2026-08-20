from __future__ import annotations

from pathlib import Path
import sys
import unittest

from psa.self_model.d4b_steady_state_off_runtime import (
    D4B_PREFIX_TOKEN_IDS,
    D4B_TARGET_TOKEN_IDS,
    execute_d4b_fake_or_future_authorized_core,
)
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    RWKV7CouplingOffAdapter,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
)
from tests.test_self_model_d4a_failure_diagnostic_runtime import (
    FakeTensor,
    FakeTorch,
    _runtimes,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = ROOT / "src/psa/self_model/d4b_steady_state_off_runtime.py"


def _routes():
    base, g0, g2 = _runtimes()
    g1 = RWKV7CouplingOffAdapter(
        base_model=base,
        upstream_package_version="0.8.32",
        upstream_model_source_sha256=EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    )
    return base, g1, g0, g2


class PerturbingRoute:
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


class RaisingRoute:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

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
        self.calls += 1
        self.inner.forward(tokens, state, full_output)
        raise RuntimeError("synthetic D4B stop")


class D4BSteadyStateOffRuntimeTests(unittest.TestCase):
    def test_exact_fake_core_records_fixed_21_calls_and_120_pairs(self):
        base, g1, g0, g2 = _routes()
        report = execute_d4b_fake_or_future_authorized_core(
            base_model=base, off_g1=g1, g0=g0, off_g2=g2, torch=FakeTorch
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(report["calls"]), 21)
        self.assertEqual(
            len(report["comparisons"]["within_route"]),
            24,
        )
        self.assertEqual(len(report["comparisons"]["cross_route"]), 96)
        self.assertEqual(report["calls"][0]["token_ids"], D4B_PREFIX_TOKEN_IDS)
        self.assertFalse(report["calls"][0]["scored"])
        self.assertTrue(
            all(call["token_ids"] == D4B_TARGET_TOKEN_IDS for call in report["calls"][1:])
        )
        self.assertEqual(g1.delegation_count, 5)
        self.assertEqual(g0.execution_count, 5)
        self.assertEqual(g2.execution_count, 5)
        self.assertEqual(report["pass_effect"], "runtime_core_verification_only")
        self.assertEqual(
            report["future_authorized_real_d4b_pass_effect"],
            "d5_review_candidate_only",
        )
        self.assertFalse(report["d4_status_changed"])
        self.assertFalse(report["d5_authorized"])

    def test_preconditioning_outputs_are_recorded_but_excluded_from_pairs(self):
        base, g1, g0, g2 = _routes()
        report = execute_d4b_fake_or_future_authorized_core(
            base_model=base, off_g1=g1, g0=g0, off_g2=g2, torch=FakeTorch
        )
        precondition = [
            call for call in report["calls"] if call["phase"] == "fixed_preconditioning"
        ]
        self.assertEqual(len(precondition), 4)
        self.assertTrue(all(call["output_recorded"] for call in precondition))
        self.assertTrue(all(not call["scored"] for call in precondition))
        compared_ids = {
            item[key]
            for group in report["comparisons"].values()
            for item in group
            for key in ("left_call_id", "right_call_id")
        }
        self.assertTrue(all(call["call_id"] not in compared_ids for call in precondition))
        self.assertNotIn(report["calls"][0]["call_id"], compared_ids)

    def test_one_route_difference_fails_exactly_without_adaptation(self):
        base, g1, g0, g2 = _routes()
        report = execute_d4b_fake_or_future_authorized_core(
            base_model=base,
            off_g1=g1,
            g0=g0,
            off_g2=PerturbingRoute(g2),
            torch=FakeTorch,
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["all_scored_pairs_exact"])
        self.assertEqual(len(report["calls"]), 21)
        self.assertEqual(g2.execution_count, 5)
        self.assertEqual(report["pass_effect"], "stop_without_rerun")
        self.assertFalse(report["safety"]["automatic_rerun_authorized"])

    def test_route_exception_propagates_once_and_restores_bindings(self):
        base, g1, g0, g2 = _routes()
        raising = RaisingRoute(g2)
        with self.assertRaisesRegex(RuntimeError, "synthetic D4B stop"):
            execute_d4b_fake_or_future_authorized_core(
                base_model=base,
                off_g1=g1,
                g0=g0,
                off_g2=raising,
                torch=FakeTorch,
            )
        self.assertEqual(raising.calls, 1)
        for name in (*TARGET_METHODS, CALLBACK_ATTRIBUTE):
            self.assertNotIn(name, base.__dict__)

    def test_conflicting_binding_fails_before_any_forward_call(self):
        base, g1, g0, g2 = _routes()
        setattr(base, CALLBACK_ATTRIBUTE, None)
        with self.assertRaisesRegex(RuntimeError, "conflicting instance overrides"):
            execute_d4b_fake_or_future_authorized_core(
                base_model=base,
                off_g1=g1,
                g0=g0,
                off_g2=g2,
                torch=FakeTorch,
            )
        self.assertEqual(g1.delegation_count, 0)
        self.assertEqual(g0.execution_count, 0)
        self.assertEqual(g2.execution_count, 0)

    def test_runtime_source_does_not_import_rwkv_or_torch(self):
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("import rwkv", source)
        self.assertNotIn("import torch", source)
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)


if __name__ == "__main__":
    unittest.main()
