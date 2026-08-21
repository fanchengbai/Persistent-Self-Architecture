from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.self_model.d5a_offline_active import (
    CONFIG_RELATIVE_PATH,
    DeterministicHashMatrixFakeProjection,
    REQUIRED_NEXT_CONFIRMATION,
    build_d5a_report,
    validate_contract,
)
from psa.self_model.encoding import DeterministicHashFakeSelfEncoder
from psa.self_model.state import build_self_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


def _minimal_state() -> dict:
    item = {
        "field_item_id": "identity",
        "value": "saffron",
        "value_type": "string",
        "confidence": 1.0,
        "update_class": "protected",
        "created_step": 0,
        "updated_step": 0,
        "source_evidence_ids": ["fixture:test"],
        "status": "active",
    }
    return build_self_state(
        state_id="d5a-test",
        agent_instance_id="agent-d5a-test",
        trajectory_id="trajectory-d5a-test",
        step=0,
        model_id="offline-no-model",
        tokenizer_id="offline-no-tokenizer",
        fields={"identity_anchors": [item]},
    )


class CouplingD5AOfflineActiveTests(unittest.TestCase):
    def test_report_closes_fake_projection_and_callback_contract(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_d5a_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["contract_checks"].values()))
        self.assertTrue(all(report["runtime_checks"].values()))
        self.assertTrue(report["safety"]["d5a_offline_active_contract_implemented"])
        self.assertTrue(report["safety"]["fake_projection_constructed"])
        for field in (
            "d5b_real_path_implemented",
            "rwkv_model_imported",
            "torch_imported",
            "weights_accessed",
            "model_loaded",
            "model_executed",
            "real_layers_selected",
            "real_self_projection_constructed",
            "formal_test_set_accessed",
            "self_effect_experiment_run",
            "self_updater_implemented",
            "automatic_rerun_authorized",
        ):
            self.assertFalse(report["safety"][field])
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            config["required_next_owner_confirmation_text"],
            REQUIRED_NEXT_CONFIRMATION,
        )

    def test_projection_is_deterministic_fake_and_input_immutable(self) -> None:
        state = _minimal_state()
        encoded = DeterministicHashFakeSelfEncoder(16).encode(
            state, field_mask=["identity_anchors"]
        )
        snapshot = copy.deepcopy(encoded)
        projection = DeterministicHashMatrixFakeProjection(
            input_dimension=16,
            output_dimension=8,
            seed_namespace="PSA|Self-v0.1|Coupling-D5A|fake-projection",
        )
        first = projection.project(encoded)
        repeated = projection.project(encoded)
        self.assertEqual(first, repeated)
        self.assertEqual(encoded, snapshot)
        self.assertFalse(first.trained)
        self.assertFalse(first.real_self_projection)
        self.assertEqual(len(first.vector), 8)

    def test_projection_rejects_dimension_or_nonfake_encoding(self) -> None:
        encoded = DeterministicHashFakeSelfEncoder(16).encode(
            _minimal_state(), field_mask=["identity_anchors"]
        )
        wrong_dimension = DeterministicHashMatrixFakeProjection(
            input_dimension=8,
            output_dimension=8,
            seed_namespace="fake",
        )
        with self.assertRaises(PermissionError):
            wrong_dimension.project(encoded)
        changed = copy.copy(encoded)
        object.__setattr__(changed, "model_loaded", True)
        valid = DeterministicHashMatrixFakeProjection(
            input_dimension=16,
            output_dimension=8,
            seed_namespace="fake",
        )
        with self.assertRaises(PermissionError):
            valid.project(changed)

    def test_contract_scope_changes_fail_closed(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("authority", "d5b_real_path_static_implementation_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "self_updater_authorized", True),
            ("fake_projection", "real_self_projection", True),
            ("fake_runtime", "device", "cuda:0"),
            ("fake_callback", "phase", "post_attention_residual"),
        )
        for section, field, value in changes:
            changed = copy.deepcopy(config)
            changed[section][field] = value
            with self.assertRaises(PermissionError):
                validate_contract(changed)

    def test_report_digest_is_self_consistent(self) -> None:
        report = build_d5a_report(config_path=CONFIG, project_root=ROOT)
        digest = report.pop("report_digest_sha256")
        self.assertEqual(digest, sha256_json(report))

    def test_alternate_config_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d5a_report(config_path=path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
