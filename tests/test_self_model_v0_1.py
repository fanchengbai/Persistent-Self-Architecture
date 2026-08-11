from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from psa.self_model import (
    DeterministicHashFakeSelfEncoder,
    FakeGatedResidualAdapter,
    SelfStore,
    apply_offline_gated_injection,
    build_self_model_v0_1_offline_manifest,
    build_self_state,
    encoded_self_digest,
    randomize_encoded_fields,
    swap_self_fields,
    verify_self_model_v0_1_offline_manifest,
    validate_self_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "development" / "self_model_v0_1_offline.draft.json"


def _item(
    item_id: str,
    value,
    value_type: str,
    update_class: str,
):
    return {
        "field_item_id": item_id,
        "value": value,
        "value_type": value_type,
        "confidence": 1.0,
        "update_class": update_class,
        "created_step": 0,
        "updated_step": 0,
        "source_evidence_ids": ["fixture:owner-set"],
        "status": "active",
    }


def _state(state_id: str, identity: str, goal: str):
    return build_self_state(
        state_id=state_id,
        agent_instance_id="agent-offline-fixture",
        trajectory_id="trajectory-offline-fixture",
        step=0,
        model_id="offline-no-model-loaded",
        tokenizer_id="offline-no-tokenizer-loaded",
        fields={
            "identity_anchors": [
                _item("identity", identity, "string", "protected")
            ],
            "active_goals": [_item("goal", goal, "string", "fast")],
        },
        provenance_refs=["fixture:initialization"],
    )


class SelfModelV01Tests(unittest.TestCase):
    def test_offline_manifest_deterministically_locks_all_sources(self) -> None:
        manifest = build_self_model_v0_1_offline_manifest(
            config_path=CONTRACT,
            project_root=ROOT,
        )
        self.assertTrue(manifest["valid"])
        self.assertFalse(manifest["model_loaded"])
        self.assertFalse(manifest["model_executed"])
        self.assertFalse(manifest["real_rwkv_coupling_implemented"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_self_model_v0_1_offline_manifest(
                manifest_path=path,
                config_path=CONTRACT,
                project_root=ROOT,
            )
            self.assertTrue(verification["valid"])

            manifest["model_loaded"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_self_model_v0_1_offline_manifest(
                manifest_path=path,
                config_path=CONTRACT,
                project_root=ROOT,
            )
            self.assertFalse(verification["valid"])

    def test_static_state_store_is_immutable_and_checksum_verified(self) -> None:
        state = _state("self-a", "saffron", "spiral")
        with tempfile.TemporaryDirectory() as directory:
            store = SelfStore(directory)
            path = store.save(state)
            self.assertEqual(store.load("self-a"), state)
            self.assertEqual(path.name, "self-a.json")
            with self.assertRaises(FileExistsError):
                store.save(state)

    def test_tamper_and_wrong_update_class_are_rejected(self) -> None:
        state = _state("self-a", "saffron", "spiral")
        tampered = copy.deepcopy(state)
        tampered["active_goals"][0]["value"] = "harbor"
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_self_state(tampered)
        invalid_class = copy.deepcopy(state)
        invalid_class["identity_anchors"][0]["update_class"] = "fast"
        invalid_class["integrity"]["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "violates"):
            validate_self_state(invalid_class)

    def test_field_swap_is_specific_and_sources_remain_immutable(self) -> None:
        left = _state("self-a", "saffron", "spiral")
        right = _state("self-b", "indigo", "harbor")
        left_before = copy.deepcopy(left)
        right_before = copy.deepcopy(right)
        swapped_left, swapped_right = swap_self_fields(
            left,
            right,
            fields=["identity_anchors"],
            left_state_id="self-a-swap-i",
            right_state_id="self-b-swap-i",
        )
        self.assertEqual(left, left_before)
        self.assertEqual(right, right_before)
        self.assertEqual(
            swapped_left["identity_anchors"], right["identity_anchors"]
        )
        self.assertEqual(swapped_left["active_goals"], left["active_goals"])
        self.assertEqual(
            swapped_right["identity_anchors"], left["identity_anchors"]
        )
        validate_self_state(swapped_left)
        validate_self_state(swapped_right)

    def test_fake_encoder_is_deterministic_masked_and_non_prompt(self) -> None:
        state = _state("self-a", "saffron", "spiral")
        encoder = DeterministicHashFakeSelfEncoder(dimension=16)
        identity = encoder.encode(state, field_mask=["identity_anchors"])
        repeated = encoder.encode(state, field_mask=["identity_anchors"])
        goal = encoder.encode(state, field_mask=["active_goals"])
        self.assertEqual(encoded_self_digest(identity), encoded_self_digest(repeated))
        self.assertNotEqual(identity.aggregate_vector, goal.aggregate_vector)
        self.assertEqual(identity.active_fields, ("identity_anchors",))
        self.assertFalse(identity.prompt_serialization_used)
        self.assertFalse(identity.model_loaded)

    def test_encoded_field_randomization_is_seeded_and_norm_matched(self) -> None:
        encoded = DeterministicHashFakeSelfEncoder(16).encode(
            _state("self-a", "saffron", "spiral"),
            field_mask=["identity_anchors", "active_goals"],
        )
        first = randomize_encoded_fields(
            encoded, fields=["identity_anchors"], seed=20260811
        )
        repeated = randomize_encoded_fields(
            encoded, fields=["identity_anchors"], seed=20260811
        )
        alternate = randomize_encoded_fields(
            encoded, fields=["identity_anchors"], seed=20260812
        )
        self.assertEqual(first.field_vectors, repeated.field_vectors)
        self.assertNotEqual(
            first.field_vectors["identity_anchors"],
            alternate.field_vectors["identity_anchors"],
        )
        self.assertEqual(
            first.field_vectors["active_goals"],
            encoded.field_vectors["active_goals"],
        )
        source_norm = math.sqrt(
            sum(value * value for value in encoded.field_vectors["identity_anchors"])
        )
        random_norm = math.sqrt(
            sum(value * value for value in first.field_vectors["identity_anchors"])
        )
        self.assertAlmostEqual(source_norm, random_norm, places=12)

    def test_coupling_off_and_zero_scale_never_call_adapter(self) -> None:
        encoded = DeterministicHashFakeSelfEncoder(16).encode(
            _state("self-a", "saffron", "spiral"),
            field_mask=["identity_anchors", "active_goals"],
        )
        adapter = FakeGatedResidualAdapter(hidden_dimension=16)
        disabled = apply_offline_gated_injection(
            adapter,
            encoded,
            enabled=False,
            scale=1.0,
            layer_mask=["fake-layer-00"],
        )
        zero = apply_offline_gated_injection(
            adapter,
            encoded,
            enabled=True,
            scale=0.0,
            layer_mask=["fake-layer-01"],
        )
        self.assertFalse(disabled["applied"])
        self.assertFalse(zero["applied"])
        self.assertEqual(adapter.calls, [])

    def test_gated_injection_honors_scale_and_layer_mask(self) -> None:
        encoded = DeterministicHashFakeSelfEncoder(16).encode(
            _state("self-a", "saffron", "spiral"),
            field_mask=["identity_anchors", "active_goals"],
        )
        adapter = FakeGatedResidualAdapter(hidden_dimension=16, gate=0.5)
        report = apply_offline_gated_injection(
            adapter,
            encoded,
            enabled=True,
            scale=1.5,
            layer_mask=["fake-layer-01"],
        )
        self.assertTrue(report["applied"])
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(adapter.calls[0]["layer"], "fake-layer-01")
        expected = tuple(0.75 * value for value in encoded.aggregate_vector)
        self.assertEqual(adapter.calls[0]["residual"], expected)
        self.assertFalse(report["model_executed"])
        self.assertFalse(report["real_rwkv_coupling_implemented"])

    def test_offline_coupling_rejects_loaded_or_nonfake_adapter(self) -> None:
        encoded = DeterministicHashFakeSelfEncoder(16).encode(
            _state("self-a", "saffron", "spiral"),
            field_mask=["identity_anchors"],
        )

        class InvalidAdapter:
            offline_fake_adapter = False
            model_loaded = True

        with self.assertRaisesRegex(PermissionError, "unloaded fake"):
            apply_offline_gated_injection(
                InvalidAdapter(),
                encoded,
                enabled=True,
                scale=1.0,
                layer_mask=["fake-layer-00"],
            )


if __name__ == "__main__":
    unittest.main()
