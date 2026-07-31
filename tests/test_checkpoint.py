from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.artifacts import canonical_json_bytes, payload_digest, sha256_file
from psa.state.checkpoint import (
    CheckpointError,
    _apply_determinism_policy,
    _checksum_text,
    _validate_acceptance_policy,
    component_name,
    verify_native_checkpoint,
)
from psa.state.operations import (
    _classify_reset_stability,
    official_reset_state,
    randomize_state_matched,
    swap_full_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CloneableValue:
    def __init__(self, value: int) -> None:
        self.value = value

    def detach(self) -> "CloneableValue":
        return self

    def clone(self) -> "CloneableValue":
        return CloneableValue(self.value)


class CheckpointTests(unittest.TestCase):
    def _write_minimal_checkpoint(self, root: Path) -> Path:
        checkpoint = root / "ckpt-test0001"
        (checkpoint / "native_state").mkdir(parents=True)
        (checkpoint / "provenance").mkdir()
        (checkpoint / "validation").mkdir()
        payloads = {
            "native_state/tensors.safetensors": b"development tensor fixture\n",
            "native_state/inventory.json": canonical_json_bytes(
                {"components": []}
            ),
            "provenance/events.jsonl": canonical_json_bytes(
                {"event_type": "capture"}
            ),
        }
        for relative, content in payloads.items():
            (checkpoint / relative).write_bytes(content)
        digests = {
            relative: sha256_file(checkpoint / relative)
            for relative in payloads
        }
        (checkpoint / "validation" / "checksums.sha256").write_bytes(
            _checksum_text(digests)
        )
        manifest = {
            "format_version": "0.1",
            "checkpoint_id": "ckpt-test0001",
            "status": "complete",
            "model": {"layer_count": 1},
            "tokenizer": {},
            "state_components": [
                {"name": "layers.0.att_x_prev"},
                {"name": "layers.0.att_kv"},
                {"name": "layers.0.ffn_x_prev"},
            ],
            "integrity": {
                "checksums_file": "validation/checksums.sha256",
                "payload_root_digest_sha256": payload_digest(digests),
            },
        }
        (checkpoint / "manifest.json").write_bytes(
            canonical_json_bytes(manifest)
        )
        return checkpoint

    def test_rwkv7_component_names_follow_observed_three_tensor_pattern(self) -> None:
        self.assertEqual(component_name(0, 24), "layers.0.att_x_prev")
        self.assertEqual(component_name(1, 24), "layers.0.att_kv")
        self.assertEqual(component_name(2, 24), "layers.0.ffn_x_prev")
        self.assertEqual(component_name(71, 24), "layers.23.ffn_x_prev")
        with self.assertRaises(ValueError):
            component_name(72, 24)

    def test_determinism_policy_sets_pre_import_environment(self) -> None:
        policy = {
            "enabled": True,
            "seed": 17,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "float32_matmul_precision": "highest",
            "allow_tf32": False,
        }
        with patch.dict("os.environ", {}, clear=True):
            observed = _apply_determinism_policy(policy)
            self.assertEqual(observed, policy)
            self.assertEqual(
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
                ":4096:8",
            )
            self.assertEqual(
                os.environ["PSA_DETERMINISTIC_SEED"],
                "17",
            )

    def test_acceptance_policy_requires_positive_bounds(self) -> None:
        policy = {
            "require_shape_dtype_compatibility": True,
            "require_top1_match": True,
            "logits_max_abs_error": 0.0625,
            "state_max_abs_error": 0.125,
        }
        self.assertEqual(_validate_acceptance_policy(policy), policy)
        with self.assertRaises(ValueError):
            _validate_acceptance_policy(
                {**policy, "state_max_abs_error": 0.0}
            )

    def test_official_reset_uses_rwkv_none_sentinel(self) -> None:
        self.assertIsNone(official_reset_state())

    def test_full_state_swap_is_a_deep_clone(self) -> None:
        donor = [CloneableValue(3), {"nested": CloneableValue(5)}]
        swapped = swap_full_state(donor)
        self.assertIsNot(swapped, donor)
        self.assertIsNot(swapped[0], donor[0])
        self.assertIsNot(swapped[1]["nested"], donor[1]["nested"])
        self.assertEqual(swapped[1]["nested"].value, 5)

    def test_random_state_rejects_negative_seed_before_tensor_work(self) -> None:
        with self.assertRaises(ValueError):
            randomize_state_matched([], object(), seed=-1)

    def test_reset_stability_routes_distinguish_first_call_outlier(self) -> None:
        passed = {"within_tolerance": True}
        failed = {"within_tolerance": False}
        self.assertEqual(
            _classify_reset_stability([failed] * 10, [passed] * 9),
            "first_shape_call_outlier",
        )
        self.assertEqual(
            _classify_reset_stability([passed] * 10, [passed] * 9),
            "no_reset_instability_detected",
        )
        self.assertEqual(
            _classify_reset_stability([failed] * 10, [failed] + [passed] * 8),
            "persistent_reset_instability",
        )

    def test_g1h_state_replay_configs_preserve_frozen_controls(self) -> None:
        pairs = (
            (
                "impl2_checkpoint_roundtrip.dev.json",
                "impl3m_g1h_2.9b_checkpoint_roundtrip.dev.json",
            ),
            (
                "impl2b_state_operations.dev.json",
                "impl3n_g1h_2.9b_state_operations.dev.json",
            ),
            (
                "impl2c_random_matched.dev.json",
                "impl3o_g1h_2.9b_random_matched.dev.json",
            ),
        )
        for baseline_name, replay_name in pairs:
            with self.subTest(replay=replay_name):
                baseline = json.loads(
                    (PROJECT_ROOT / "configs" / "gates" / baseline_name)
                    .read_text(encoding="utf-8")
                )
                replay = json.loads(
                    (PROJECT_ROOT / "configs" / "gates" / replay_name)
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(replay["gate_version"], "0.1")
                self.assertTrue(replay["development_only"])
                self.assertEqual(
                    replay["repeat_count"], baseline["repeat_count"]
                )
                self.assertEqual(
                    replay["determinism"], baseline["determinism"]
                )
                self.assertEqual(replay["acceptance"], baseline["acceptance"])
        random_baseline = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "gates"
                / "impl2c_random_matched.dev.json"
            ).read_text(encoding="utf-8")
        )
        random_replay = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "gates"
                / "impl3o_g1h_2.9b_random_matched.dev.json"
            ).read_text(encoding="utf-8")
        )
        for key in ("base_seed", "alternate_seed", "max_relative_l2_error"):
            self.assertEqual(random_replay[key], random_baseline[key])

    def test_code_marginalized_readout_is_frozen_without_fitted_offsets(
        self,
    ) -> None:
        readout = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "readouts"
                / "exp001_g1h_2.9b_code_marginalized.dev.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(readout["answer_codes"], ["A", "B", "C", "D"])
        self.assertEqual(readout["rotation_count"], 4)
        self.assertEqual(
            readout["aggregation"]["operation"],
            "arithmetic_mean_across_answer_code_rotations",
        )
        self.assertEqual(readout["calibration"]["fitted_parameters"], [])
        self.assertFalse(
            readout["calibration"]["uses_post_hoc_label_offsets"]
        )
        self.assertEqual(
            readout["development_qualification"]["correct_case_count"], 32
        )

    def test_l1_verification_accepts_intact_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_minimal_checkpoint(Path(directory))
            report = verify_native_checkpoint(
                checkpoint,
                load_tensors=False,
            )
        self.assertTrue(report["valid"])
        self.assertEqual(report["achieved_level"], "L1")

    def test_l1_verification_rejects_payload_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_minimal_checkpoint(Path(directory))
            (checkpoint / "provenance" / "events.jsonl").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(CheckpointError, "E_CHECKSUM"):
                verify_native_checkpoint(checkpoint, load_tensors=False)

    def test_l1_verification_rejects_checksum_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_minimal_checkpoint(Path(directory))
            checksum_path = checkpoint / "validation" / "checksums.sha256"
            checksum_path.write_text(f"{'0' * 64}  ../outside\n", encoding="utf-8")
            with self.assertRaisesRegex(CheckpointError, "E_CHECKSUM"):
                verify_native_checkpoint(checkpoint, load_tensors=False)

    def test_l1_verification_rejects_manifest_checksum_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_minimal_checkpoint(Path(directory))
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["integrity"]["checksums_file"] = "../outside.sha256"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(CheckpointError, "E_CHECKSUM"):
                verify_native_checkpoint(checkpoint, load_tensors=False)

    def test_l1_verification_rejects_component_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_minimal_checkpoint(Path(directory))
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["state_components"].reverse()
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(CheckpointError, "E_SHAPE_MISMATCH"):
                verify_native_checkpoint(checkpoint, load_tensors=False)


if __name__ == "__main__":
    unittest.main()
