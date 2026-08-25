from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.self_model.d6d_i_tooling import (
    ACCEPTANCE_CATEGORIES,
    CONFIG_RELATIVE_PATH,
    NEXT_CONFIRMATION,
    REQUIRED_CONFIRMATION,
    build_d6d_i_report,
    run_joint_pure_python_acceptance,
    validate_config,
)
from psa.self_model.d6d_projection_artifact import (
    FrozenSelfProjection,
    ProjectionTrainingRecord,
    audit_frozen_projection_artifact,
    build_frozen_projection_artifact,
    projection_vector_digest,
)
from psa.self_model.d6d_wrapper_runtime import (
    D6DIRequest,
    request_for_condition,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


def _small_artifact() -> dict:
    records = (
        ProjectionTrainingRecord("amber", "orbit", (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ProjectionTrainingRecord("cobalt", "harbor", (-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0)),
    )
    return build_frozen_projection_artifact(
        records=records,
        output_dimension=4,
        training_manifest_sha256=sha256_json({"training": 1}),
        pilot_manifest_commitment_sha256=sha256_json({"pilot": 1}),
        optimizer_seed=7,
        fixture_only=True,
    )


class D6DIToolingTests(unittest.TestCase):
    def test_exact_confirmation_and_next_gate_are_frozen(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["owner_confirmation_text"], REQUIRED_CONFIRMATION)
        self.assertEqual(config["required_next_owner_confirmation_text"], NEXT_CONFIRMATION)
        self.assertTrue(all(validate_config(config).values()))

    def test_wrapper_contract_places_all_bindings_outside_base_instance(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        wrapper = config["wrapper_tooling"]
        self.assertTrue(wrapper["wrapper_owns_forward"])
        self.assertTrue(wrapper["instrumented_methods_bound_to_wrapper_only"])
        self.assertFalse(wrapper["base_model_instance_setattr_or_delattr_allowed"])
        self.assertEqual(wrapper["attribute_delegation"], "read_only_getattr")

    def test_projection_artifact_is_frozen_bias_free_and_digest_bound(self):
        artifact = _small_artifact()
        report = audit_frozen_projection_artifact(artifact)
        self.assertTrue(report["valid"])
        self.assertFalse(artifact["bias_present"])
        self.assertTrue(artifact["double_mask_projection_exact_zero"])
        self.assertFalse(artifact["research_evidence_eligible"])
        self.assertEqual(FrozenSelfProjection(artifact).dimension, 4)

    def test_projection_artifact_tampering_fails_closed(self):
        artifact = _small_artifact()
        for mutation in ("parameter", "metadata"):
            changed = copy.deepcopy(artifact)
            if mutation == "parameter":
                changed["parameters"]["identity_weights"]["amber"][0] = 9.0
            else:
                changed["target_layer_index_zero_based"] = 16
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                audit_frozen_projection_artifact(changed)

    def test_projection_vector_digest_canonicalizes_platform_last_bits(self):
        vector = (0.000123456789012345, -0.000987654321098765)
        adjacent = tuple(math.nextafter(value, math.inf) for value in vector)
        materially_changed = (vector[0] + 1e-8, vector[1])
        self.assertEqual(
            projection_vector_digest(vector),
            projection_vector_digest(adjacent),
        )
        self.assertNotEqual(
            projection_vector_digest(vector),
            projection_vector_digest(materially_changed),
        )

    def test_training_and_pilot_commitments_must_differ(self):
        digest = sha256_json({"same": True})
        with self.assertRaises(PermissionError):
            build_frozen_projection_artifact(
                records=(
                    ProjectionTrainingRecord("a", "g", (1.0,) * 4, (1.0,) * 4),
                ),
                output_dimension=4,
                training_manifest_sha256=digest,
                pilot_manifest_commitment_sha256=digest,
                optimizer_seed=1,
                fixture_only=True,
            )

    def test_request_conditions_are_exact_and_fail_closed(self):
        off = request_for_condition("wrapper_off")
        zero = request_for_condition("wrapper_zero")
        off.validate()
        zero.validate()
        self.assertFalse(off.enabled)
        self.assertTrue(zero.enabled)
        with self.assertRaises(PermissionError):
            D6DIRequest("wrapper_off", True, 0.0, None).validate()
        with self.assertRaises(ValueError):
            request_for_condition("raw_original")

    def test_joint_pure_python_acceptance_passes_all_twenty_categories(self):
        acceptance = run_joint_pure_python_acceptance()
        self.assertTrue(acceptance["valid"])
        self.assertEqual(tuple(acceptance["checks"]), ACCEPTANCE_CATEGORIES)
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(acceptance["counts"]["joint_conditions"], 11)
        self.assertEqual(acceptance["counts"]["hidden_dimension"], 2560)
        self.assertEqual(
            acceptance["base_instance_dictionary_before_keys"],
            acceptance["base_instance_dictionary_after_keys"],
        )

    def test_scope_expansion_or_separate_mechanism_run_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changes = (
            ("model_execution_authorized", True),
            ("real_projection_artifact_construction_authorized", True),
            ("d6d_real_execution_authorized", True),
            ("d6c_rerun_authorized", True),
        )
        for name, value in changes:
            changed = copy.deepcopy(payload)
            changed["authority"][name] = value
            with self.subTest(name=name), self.assertRaises(PermissionError):
                validate_config(changed)
        changed = copy.deepcopy(payload)
        changed["joint_acceptance"]["separate_mechanism_acceptance_run"] = True
        with self.assertRaises(PermissionError):
            validate_config(changed)

    def test_static_report_is_no_model_and_inventory_complete(self):
        report = build_d6d_i_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["checks"]["d6d_design_config_frozen"])
        self.assertTrue(report["checks"]["d6d_design_document_frozen"])
        self.assertTrue(report["checks"]["d6d_design_source_frozen"])
        self.assertFalse(report["safety"]["installed_source_probed"])
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["real_projection_artifact_constructed"])
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_copied_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied.json"
            copied.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d6d_i_report(config_path=copied, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
