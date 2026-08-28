from __future__ import annotations

import copy
import unittest

from psa.self_model.d7c_public_semantics_runtime import (
    D7CCompatibilityRequest,
    D7CPublicSemanticsWrapper,
    FULL_OUTPUT_VALUES,
    N_LAYER,
    SYNTHETIC_UPSTREAM_SOURCE,
    STATE_INPUTS,
    TARGET_LAYER_INDEX,
    TARGET_LAYER_RULE_ID,
    _synthetic_namespace,
    compatibility_cells,
    run_synthetic_compatibility_acceptance,
    zero_request,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    compile_instrumented_methods,
)


class D7CPublicSemanticsRuntimeTests(unittest.TestCase):
    def test_independent_target_layer_rule_is_architecture_derived(self):
        self.assertEqual(N_LAYER, 32)
        self.assertEqual(TARGET_LAYER_INDEX, N_LAYER // 2 - 1)
        self.assertEqual(TARGET_LAYER_INDEX, 15)
        self.assertEqual(TARGET_LAYER_RULE_ID, "d7_lower_half_terminal_layer_v01")

    def test_eight_cells_cover_exact_cartesian_product(self):
        cells = compatibility_cells()
        self.assertEqual(len(cells), 8)
        product = {
            (cell["execution_path"], cell["state_input"], cell["full_output"])
            for cell in cells
        }
        self.assertEqual(
            product,
            {
                (path, state, full)
                for path in ("forward_one", "forward_seq")
                for state in STATE_INPUTS
                for full in FULL_OUTPUT_VALUES
            },
        )

    def test_synthetic_acceptance_covers_eighteen_call_contract(self):
        acceptance = run_synthetic_compatibility_acceptance()
        self.assertTrue(acceptance["valid"])
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(acceptance["counts"]["equivalence_forward_calls"], 16)
        self.assertEqual(acceptance["counts"]["synthetic_active_forward_calls"], 2)
        self.assertEqual(acceptance["counts"]["total_forward_plan"], 18)
        self.assertEqual(acceptance["counts"]["synthetic_wrapper_forward_calls"], 10)
        self.assertEqual(acceptance["counts"]["active_callback_invocations"], 64)
        self.assertEqual(acceptance["counts"]["active_target_layer_applications"], 2)

    def test_none_and_prebuilt_initialization_counts_are_exact(self):
        acceptance = run_synthetic_compatibility_acceptance()
        for cell in acceptance["cell_reports"]:
            expected = 1 if cell["state_input"] == "none" else 0
            self.assertEqual(cell["zero_state_initializations"], expected)

    def test_invalid_request_fails_before_child_dispatch(self):
        namespace, fixture_type = _synthetic_namespace()
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
        invalid = D7CCompatibilityRequest("zero", True, 1.0, None)
        before = wrapper.execution_count
        with self.assertRaises(PermissionError):
            wrapper.forward([1], None, request=invalid)
        self.assertEqual(wrapper.execution_count, before)
        self.assertTrue(wrapper.base_dictionary_is_stable())

    def test_wrapper_does_not_mutate_source_tokens_or_state(self):
        acceptance = run_synthetic_compatibility_acceptance()
        snapshot = copy.deepcopy(acceptance["cell_reports"])
        self.assertEqual(acceptance["cell_reports"], snapshot)


if __name__ == "__main__":
    unittest.main()
