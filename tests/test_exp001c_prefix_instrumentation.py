from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from psa.development.prefix_instrumentation import (
    DEFAULT_PREFIX_POSITION_LABELS,
    answer_boundary_evidence,
    instrument_forced_prefix,
    token_evidence_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "exp001c_prefix_logits_fixture.json"
SCHEMA = ROOT / "schemas" / "exp001c_prefix_evidence.schema.json"


class _SyntheticAdapter:
    def __init__(self, fixture: dict) -> None:
        self._fixture = fixture
        self._position = 1
        self.asserted_tokens: list[list[int]] = []

    def forward(self, token_ids: list[int], state: dict) -> tuple[list[float], dict]:
        self.asserted_tokens.append(list(token_ids))
        logits = (
            self._fixture["position_logits"][self._position]
            if self._position < len(self._fixture["position_logits"])
            else self._fixture["post_prefix_logits"]
        )
        self._position += 1
        return logits, {"position": self._position}

    def decode(self, token_ids: list[int]) -> str:
        return (
            self._fixture["forced_prefix_text"]
            if token_ids == self._fixture["prefix_token_ids"]
            else "invalid"
        )

class Exp001CPrefixInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_token_evidence_preserves_rank_margin_and_top_k(self) -> None:
        expected = self.fixture["expected"]
        for index, logits in enumerate(self.fixture["position_logits"]):
            evidence = token_evidence_from_logits(
                logits,
                expected_token_id=self.fixture["prefix_token_ids"][index],
                top_k=3,
            )
            self.assertEqual(
                evidence["greedy_token_id"], expected["greedy_token_ids"][index]
            )
            self.assertEqual(
                evidence["expected_token_rank"],
                expected["expected_token_ranks"][index],
            )
            self.assertAlmostEqual(
                evidence["logit_margin_greedy_minus_expected_float32"],
                expected["greedy_minus_expected_margins"][index],
            )
            self.assertEqual(
                evidence["top_k_token_ids"], expected["top_3_token_ids"][index]
            )
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for value in evidence["top_k_log_probabilities_float32"]
                )
            )

    def test_instrumented_prefix_records_and_advances_expected(self) -> None:
        adapter = _SyntheticAdapter(self.fixture)
        report, final_logits, final_state = instrument_forced_prefix(
            adapter,
            logits=self.fixture["position_logits"][0],
            state={"position": 0},
            prefix_token_ids=self.fixture["prefix_token_ids"],
            forced_prefix_text=self.fixture["forced_prefix_text"],
            top_k=3,
        )
        self.assertEqual(report["greedy_token_ids"], [4, 2])
        self.assertFalse(report["greedy_exact"])
        self.assertTrue(report["roundtrip_exact"])
        self.assertEqual(
            [item["position_label"] for item in report["positions"]],
            list(DEFAULT_PREFIX_POSITION_LABELS),
        )
        self.assertEqual(adapter.asserted_tokens, [[4], [3]])
        self.assertEqual(final_logits, self.fixture["post_prefix_logits"])
        self.assertEqual(final_state, {"position": 3})
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(set(report), set(schema["required"]))
        position_required = set(schema["$defs"]["positionEvidence"]["required"])
        self.assertTrue(
            all(set(item) == position_required for item in report["positions"])
        )

    def test_answer_boundary_evidence_preserves_semantic_margin(self) -> None:
        evidence = answer_boundary_evidence(
            self.fixture["answer_log_probabilities"],
            target_code=self.fixture["target_code"],
        )
        self.assertEqual(evidence["best_incorrect_code"], "B")
        self.assertAlmostEqual(
            evidence["target_margin_over_best_incorrect"],
            self.fixture["expected_target_margin"],
        )

    def test_schema_requires_every_quantitative_position_field(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        required = set(schema["$defs"]["positionEvidence"]["required"])
        for field in (
            "expected_token_logit_float32",
            "expected_token_log_probability_float32",
            "expected_token_rank",
            "greedy_token_logit_float32",
            "greedy_token_log_probability_float32",
            "logit_margin_greedy_minus_expected_float32",
            "top_k_token_ids",
            "top_k_logits_float32",
            "top_k_log_probabilities_float32",
        ):
            self.assertIn(field, required)
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
