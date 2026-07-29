from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest

from psa.model.rwkv7 import RWKV7Adapter, clone_state, load_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = (
    PROJECT_ROOT / "configs" / "models" / "rwkv7_world_0.4b.impl1.json"
)


class Cloneable:
    def __init__(self, value: int) -> None:
        self.value = value

    def detach(self) -> "Cloneable":
        return self

    def clone(self) -> "Cloneable":
        return Cloneable(self.value)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class FakeModel:
    def forward(
        self, tokens: list[int], state: object
    ) -> tuple[list[int], object]:
        return [sum(tokens)], state


class FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class ModelAdapterTests(unittest.TestCase):
    def test_model_config_loads_without_local_assets(self) -> None:
        config = load_model_config(
            MODEL_CONFIG,
            project_root=PROJECT_ROOT,
            verify_files=False,
        )
        self.assertEqual(config.model_id, "rwkv7-world-0.4b-v2.9")
        self.assertEqual(config.strategy, "cuda fp16")
        self.assertEqual(config.architecture_hint["n_layer"], 24)
        self.assertEqual(len(config.weights_sha256), 64)
        self.assertEqual(len(config.tokenizer_sha256), 64)

    def test_clone_state_is_deep_for_tensor_like_values(self) -> None:
        original = [Cloneable(1), {"nested": Cloneable(2)}]
        cloned = clone_state(original)
        self.assertIsNot(cloned, original)
        self.assertIsNot(cloned[0], original[0])
        self.assertIsNot(cloned[1]["nested"], original[1]["nested"])
        self.assertEqual(cloned[1]["nested"].value, 2)

    def test_adapter_tokenizer_and_forward_contract(self) -> None:
        config = load_model_config(
            MODEL_CONFIG,
            project_root=PROJECT_ROOT,
            verify_files=False,
        )
        adapter = RWKV7Adapter(
            config=config,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            torch=FakeTorch(),
        )
        tokens = adapter.encode("abc")
        self.assertEqual(adapter.decode(tokens), "abc")
        logits, state = adapter.forward(tokens, {"state": 1})
        self.assertEqual(logits, [294])
        self.assertEqual(state, {"state": 1})
