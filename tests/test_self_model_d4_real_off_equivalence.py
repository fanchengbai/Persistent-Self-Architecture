from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from psa.self_model.d4_real_off_equivalence import (
    D4_EXECUTION_LOCK_ENV,
    D4_EXECUTION_LOCK_VALUE,
    _create_claim,
    _matrix_cells,
    _read_spec,
    execute_equivalence_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/development/self_model_v0_1_d4_real_off_equivalence.json"


class FakeTensor:
    def __init__(self, values, *, dtype="fake.float16", device="cuda:0"):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = dtype
        self.device = device

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype, device=self.device)


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    @staticmethod
    def equal(left, right):
        return (
            left.values == right.values
            and left.dtype == right.dtype
            and left.device == right.device
        )

    @staticmethod
    def inference_mode():
        return _InferenceMode()


class FakeModel:
    def forward(self, tokens, state, full_output=False):
        current = [FakeTensor([0]), FakeTensor([10])] if state is None else state
        delta = sum(tokens)
        current[0].values[0] += delta
        logits = (
            FakeTensor([value + current[0].values[0] for value in tokens])
            if full_output
            else FakeTensor([current[0].values[0], current[1].values[0]])
        )
        return logits, current


class FakeOffRoute:
    callback_call_count = 0
    self_projection_constructed = False

    def __init__(self, base, *, perturb=False, g2=False):
        self.base = base
        self.perturb = perturb
        self.delegation_count = 0
        self.execution_count = 0
        self.g2 = g2

    def forward(self, tokens, state, full_output=False):
        if self.g2:
            self.execution_count += 1
        else:
            self.delegation_count += 1
        logits, next_state = self.base.forward(tokens, state, full_output)
        if self.perturb:
            logits.values[0] += 1
        return logits, next_state


class D4RealOffEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.spec = _read_spec(CONFIG)

    def test_config_freezes_six_cell_exact_protocol(self):
        cells = _matrix_cells(self.spec)
        self.assertEqual(len(cells), 6)
        self.assertEqual(
            {cell["execution_path"] for cell in cells},
            {"forward_one", "forward_seq"},
        )
        self.assertEqual(
            {cell["state_input"] for cell in cells},
            {"none", "cloned_restored_snapshot"},
        )
        self.assertEqual(
            {cell["full_output"] for cell in cells if cell["execution_path"] == "forward_seq"},
            {False, True},
        )
        self.assertEqual(self.spec["comparison"], "torch.equal")
        self.assertFalse(self.spec["automatic_rerun_authorized"])
        self.assertFalse(self.spec["active_injection_authorized"])

    def test_exact_off_matrix_passes_and_preserves_snapshot(self):
        base = FakeModel()
        g1 = FakeOffRoute(base)
        g2 = FakeOffRoute(base, g2=True)
        report = execute_equivalence_matrix(
            base_model=base,
            off_g1=g1,
            off_g2=g2,
            torch=FakeTorch,
            spec=self.spec,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(g1.delegation_count, 12)
        self.assertEqual(g2.execution_count, 12)
        for cell in report["cells"]:
            self.assertEqual(cell["warmup_count_per_route"], 1)
            self.assertTrue(cell["valid"])

    def test_one_tensor_difference_fails_without_tolerance(self):
        base = FakeModel()
        g1 = FakeOffRoute(base)
        g2 = FakeOffRoute(base, perturb=True, g2=True)
        report = execute_equivalence_matrix(
            base_model=base,
            off_g1=g1,
            off_g2=g2,
            torch=FakeTorch,
            spec=self.spec,
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["all_cells_exact"])
        self.assertTrue(
            all(
                not cell["comparisons"]["off_g2_instrumented"]["logits"]["torch_equal"]
                for cell in report["cells"]
            )
        )

    def test_scope_changes_fail_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field, value in (
                ("automatic_rerun_authorized", True),
                ("active_injection_authorized", True),
                ("comparison", "allclose"),
                ("warmup_count_per_route_per_cell", 2),
            ):
                changed = copy.deepcopy(payload)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(PermissionError):
                    _read_spec(path)

    def test_claim_is_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            git = {"commit": "a" * 40}
            claim = _create_claim(output, CONFIG, git)
            self.assertTrue(claim.is_file())
            with self.assertRaises(FileExistsError):
                _create_claim(output, CONFIG, git)

    def test_execution_lock_constants_match_config(self):
        self.assertEqual(self.spec["execution_lock_env"], D4_EXECUTION_LOCK_ENV)
        self.assertEqual(self.spec["execution_lock_value"], D4_EXECUTION_LOCK_VALUE)


if __name__ == "__main__":
    unittest.main()
