from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.fake_callback_runtime import (
    FAKE_CALLBACK_CONFIG_FILE,
    FakePostFFNResidualCallback,
    FakeRWKV7ResidualRuntime,
    FakeResidualTensor,
    ResidualCallbackRequest,
    build_fake_callback_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / FAKE_CALLBACK_CONFIG_FILE


def _runtime() -> FakeRWKV7ResidualRuntime:
    return FakeRWKV7ResidualRuntime(
        n_layer=3,
        hidden_dimension=4,
        dtype="float16",
        device="fake-cuda:0",
    )


def _callback(*, enabled: bool = True, scale: float = 1.0):
    return FakePostFFNResidualCallback(
        hidden_dimension=4,
        layer_mask=["fake-layer-01"],
        enabled=enabled,
        scale=scale,
        gate=0.5,
    )


class SelfModelFakeCallbackTests(unittest.TestCase):
    def test_report_passes_without_importing_model_or_torch(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_fake_callback_report(
            config_path=CONFIG,
            project_root=ROOT,
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(all(value is False for value in report["safety"].values()))
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)

    def test_off_and_zero_scale_are_exact_and_do_not_call_callback(self) -> None:
        runtime = _runtime()
        state = runtime.zero_state()
        vector = (0.25, -0.5, 0.75, -1.0)
        baseline_one = runtime.forward_one(3, state, self_vector=vector)
        baseline_seq = runtime.forward_seq((3, 5, 8), state, self_vector=vector)
        for callback in (_callback(enabled=False), _callback(scale=0.0)):
            self.assertEqual(
                baseline_one,
                runtime.forward_one(3, state, self_vector=vector, callback=callback),
            )
            self.assertEqual(
                baseline_seq,
                runtime.forward_seq(
                    (3, 5, 8), state, self_vector=vector, callback=callback
                ),
            )
            self.assertEqual(callback.calls, [])

    def test_active_callback_covers_both_paths_and_preserves_metadata(self) -> None:
        runtime = _runtime()
        state = runtime.zero_state()
        vector = (0.25, -0.5, 0.75, -1.0)
        callback = _callback()
        one, _ = runtime.forward_one(
            3, state, self_vector=vector, callback=callback
        )
        seq, _ = runtime.forward_seq(
            (3, 5, 8), state, self_vector=vector, callback=callback
        )
        self.assertEqual(one.shape, (4,))
        self.assertEqual(seq.shape, (3, 4))
        self.assertEqual(one.dtype, "float16")
        self.assertEqual(seq.dtype, "float16")
        self.assertEqual(one.device, "fake-cuda:0")
        self.assertEqual(seq.device, "fake-cuda:0")
        self.assertEqual(
            {call["execution_path"] for call in callback.calls},
            {"forward_one", "forward_seq"},
        )
        self.assertTrue(
            all(call["phase"] == "post_ffn_residual" for call in callback.calls)
        )

    def test_source_state_is_cloned_and_returned_state_is_independent(self) -> None:
        runtime = _runtime()
        state = runtime.zero_state()
        snapshot = copy.deepcopy(state)
        _, next_state = runtime.forward_seq(
            (3, 5, 8),
            state,
            self_vector=(0.25, -0.5, 0.75, -1.0),
            callback=_callback(),
        )
        self.assertEqual(state, snapshot)
        self.assertIsNot(next_state, state)
        next_state[0][0] = 999.0
        self.assertEqual(state, snapshot)

    def test_callback_rejects_wrong_dimension_phase_layer_and_path(self) -> None:
        residual = FakeResidualTensor(
            values=(0.0, 0.0, 0.0, 0.0),
            dtype="float16",
            device="fake-cuda:0",
        )
        base = {
            "phase": "post_ffn_residual",
            "layer_index": 1,
            "layer_name": "fake-layer-01",
            "execution_path": "forward_one",
            "residual_x": residual,
            "self_vector": (0.25, -0.5, 0.75, -1.0),
        }
        for update in (
            {"phase": "post_attention_residual"},
            {"layer_name": "fake-layer-02"},
            {"layer_index": 0},
            {"execution_path": "forward_unknown"},
            {"self_vector": (1.0, 2.0)},
        ):
            request = ResidualCallbackRequest(**{**base, **update})
            with self.assertRaises((RuntimeError, ValueError)):
                _callback().apply(request)
        with self.assertRaises(RuntimeError):
            _callback(enabled=False).apply(ResidualCallbackRequest(**base))

    def test_residual_rejects_real_device_label_and_ragged_sequence(self) -> None:
        with self.assertRaises(ValueError):
            FakeResidualTensor(
                values=(0.0, 0.0), dtype="float16", device="cuda:0"
            )
        with self.assertRaises(ValueError):
            FakeResidualTensor(
                values=((0.0, 1.0), (2.0,)),
                dtype="float16",
                device="fake-cuda:0",
            )

    def test_alternate_config_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_fake_callback_report(config_path=path, project_root=ROOT)

    def test_authority_cannot_enable_model_work(self) -> None:
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(original["authority"]["model_execution_authorized"])
        self.assertFalse(original["authority"]["rwkv_model_import_authorized"])
        self.assertFalse(original["authority"]["weights_access_authorized"])


if __name__ == "__main__":
    unittest.main()
