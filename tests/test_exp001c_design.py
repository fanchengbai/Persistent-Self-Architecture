from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "preregistration" / "exp001c_prefix_semantics.draft.json"
CLOSURE = ROOT / "docs" / "exp001b_posthoc_diagnostic_closure.md"
DESIGN = ROOT / "docs" / "exp001c_prospective_design.md"


class Exp001CProspectiveDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_draft_authorizes_only_offline_development(self) -> None:
        self.assertEqual(
            self.config["status"],
            "offline_development_authorized_unfrozen_formal_unapproved",
        )
        authority = self.config["authority"]
        self.assertTrue(authority["development_implementation_authorized"])
        self.assertTrue(
            all(
                value is False
                for key, value in authority.items()
                if key != "development_implementation_authorized"
            )
        )
        self.assertFalse(
            self.config["development_authorization"]["model_execution_authorized"]
        )
        self.assertFalse(self.config["scope"]["changes_exp001b_decision"])
        self.assertTrue(
            self.config["scope"]["requires_independent_preregistration_freeze"]
        )

    def test_semantic_format_and_random_control_roles_are_separate(self) -> None:
        endpoints = self.config["endpoints"]
        self.assertTrue(
            endpoints["secondary_format"]["independent_from_primary_semantic"]
        )
        self.assertTrue(
            endpoints["assay_sensitivity"]["not_a_natural_state_no_harm_cell"]
        )
        self.assertNotIn(
            "random_matched",
            self.config["conditions"]["natural_state_no_harm"],
        )
        self.assertEqual(
            self.config["conditions"]["assay_sensitivity_controls"],
            ["random_matched"],
        )

    def test_prefix_instrumentation_preserves_quantitative_evidence(self) -> None:
        instrumentation = self.config["prefix_instrumentation"]
        required = set(instrumentation["required_per_position_fields"])
        self.assertEqual(instrumentation["top_k"], 10)
        self.assertIn("expected_token_logit_float32", required)
        self.assertIn("expected_token_rank", required)
        self.assertIn("logit_margin_greedy_minus_expected_float32", required)
        self.assertIn("top_k_logits_float32", required)
        self.assertEqual(
            instrumentation["missing_required_field_policy"],
            "formal_record_invalid_no_imputation",
        )

    def test_sampling_requires_new_items_and_unfrozen_power_review(self) -> None:
        sampling = self.config["sampling"]
        self.assertEqual(sampling["formal_sample_count"], "not_yet_frozen")
        self.assertTrue(sampling["power_analysis_required"])
        self.assertTrue(sampling["exp001b_items_excluded_from_formal_test_set"])
        self.assertTrue(self.config["scope"]["requires_new_blinded_items"])

    def test_docs_preserve_closure_and_design_boundaries(self) -> None:
        closure = CLOSURE.read_text(encoding="utf-8")
        design = DESIGN.read_text(encoding="utf-8")
        self.assertIn("posthoc_diagnostics_closed_information_limit", closure)
        self.assertIn("revise_or_stop_measured_supplemental_control_failure", closure)
        self.assertIn("未授权重跑", closure)
        self.assertIn("仅离线 instrumentation 开发获批；未冻结", design)
        self.assertIn("不是 EXP-001B 重跑", design)
        self.assertIn("独立授权", design)


if __name__ == "__main__":
    unittest.main()
