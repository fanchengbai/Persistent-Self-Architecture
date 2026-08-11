from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from psa.artifacts import sha256_file
from psa.confirmatory.rwkv_backend import (
    derive_random_state_seed,
    disk_roundtrip_states,
)
from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)
from psa.development.exp001c_v02_stage_b_design import (
    EXPERIMENT_ID,
    STAGE_B_CONDITIONS,
    verify_exp001c_v02_stage_b_design_manifest,
)
from psa.development.prefix_instrumentation import (
    answer_boundary_evidence,
    instrument_forced_prefix,
)
from psa.model import RWKV7Adapter, clone_state, load_model_config
from psa.state import randomize_state_matched


STAGE_B_RESULT_VERSION = "0.2-stage-b-development"
RANDOM_NAMESPACE = "PSA|EXP-001C|v02-stage-b|random-matched-v1"
Combo = tuple[int, int]


def _load_object(
    path: str | Path,
    label: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    candidate = Path(path)
    if root is not None and not candidate.is_absolute():
        candidate = root / candidate
    value = json.loads(candidate.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _default_option_scorer(
    adapter: Any,
    *,
    logits: Any,
    state: Any,
    rendered_answers: Mapping[str, str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for code in "ABCD":
        answer_tokens = adapter.encode(rendered_answers[code])
        if not answer_tokens:
            raise ValueError("Stage B answer rendering must tokenize")
        answer_logits = logits
        answer_state = clone_state(state)
        score = 0.0
        for index, token in enumerate(answer_tokens):
            log_probabilities = adapter.torch.log_softmax(
                answer_logits.float(),
                dim=-1,
            )
            score += float(log_probabilities[token].item())
            if index + 1 < len(answer_tokens):
                answer_logits, answer_state = adapter.forward(
                    [token],
                    answer_state,
                )
        scores[code] = score
    return scores


def _factorial_layout(
    trials: list[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    tuple[str, str],
    tuple[str, str],
]:
    by_fields = {
        (
            str(trial["target_fields"]["domain"]),
            str(trial["target_fields"]["operation"]),
        ): trial
        for trial in trials
    }
    domains = tuple(sorted({fields[0] for fields in by_fields}))
    operations = tuple(sorted({fields[1] for fields in by_fields}))
    if (
        len(by_fields) != 4
        or len(domains) != 2
        or len(operations) != 2
        or set(by_fields) != set(
            (domain, operation)
            for domain in domains
            for operation in operations
        )
    ):
        raise ValueError("Stage B protocol block is not a complete 2x2 factorial")
    return by_fields, domains, operations


class RWKVExp001CV02StageBBackend:
    """State-aware Stage B scorer; instantiated only by a future authorized runner."""

    def __init__(
        self,
        *,
        adapter: Any,
        protocol_manifest: Mapping[str, Any],
        option_scorer: Callable[..., dict[str, float]] = _default_option_scorer,
        randomizer: Callable[..., Any] = randomize_state_matched,
        snapshot_roundtrip: Callable[..., tuple[dict[Combo, Any], dict[str, Any]]] = disk_roundtrip_states,
    ) -> None:
        self.adapter = adapter
        self.protocol_manifest = dict(protocol_manifest)
        self.option_scorer = option_scorer
        self.randomizer = randomizer
        self.snapshot_roundtrip = snapshot_roundtrip

    def _prewarm_shapes(self, trials: list[Mapping[str, Any]]) -> list[int]:
        lengths = {
            len(self.adapter.encode(str(trial[field])))
            for trial in trials
            for field in ("history_text", "query_text")
        }
        neutral_tokens = self.adapter.encode(" neutral")
        if not neutral_tokens:
            raise ValueError("Stage B neutral warmup token is unavailable")
        for length in sorted(lengths):
            self.adapter.forward([neutral_tokens[0]] * length, None)
        return sorted(lengths)

    def run_stage_b(
        self,
        design_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        design_records = design_manifest.get("records")
        protocol_trials = self.protocol_manifest.get("trials")
        if (
            design_manifest.get("experiment_id") != EXPERIMENT_ID
            or design_manifest.get("status")
            != "offline_stage_b_design_verified_execution_unapproved"
            or design_manifest.get("model_executed") is not False
            or design_manifest.get("execution_authorized") is not False
            or design_manifest.get("formal_test_set_accessed") is not False
            or design_manifest.get("stage_a_rerun_included") is not False
            or design_manifest.get("conditions") != list(STAGE_B_CONDITIONS)
            or not isinstance(design_records, list)
            or len(design_records) != 224
            or self.protocol_manifest.get("experiment_id") != EXPERIMENT_ID
            or self.protocol_manifest.get("model_executed") is not False
            or self.protocol_manifest.get("formal_test_set_accessed") is not False
            or self.protocol_manifest.get("manifest_digest_sha256")
            != design_manifest.get("protocol_manifest_digest_sha256")
            or not isinstance(protocol_trials, list)
            or len(protocol_trials) != 32
        ):
            raise ValueError("Stage B RWKV backend received an invalid design")

        trials_by_id = {
            str(trial["sample_id"]): trial for trial in protocol_trials
        }
        if len(trials_by_id) != 32:
            raise ValueError("Stage B protocol trial IDs are not unique")
        trials_by_block: dict[str, list[Mapping[str, Any]]] = {}
        history_trials: dict[str, Mapping[str, Any]] = {}
        for trial in protocol_trials:
            trials_by_block.setdefault(str(trial["block_id"]), []).append(trial)
            history_trials.setdefault(str(trial["history_key"]), trial)
        if len(trials_by_block) != 2 or len(history_trials) != 8:
            raise ValueError("Stage B protocol history inventory is invalid")

        warmup_token_lengths = self._prewarm_shapes(protocol_trials)
        source_states = {}
        for history_key, trial in sorted(history_trials.items()):
            history_tokens = self.adapter.encode(str(trial["history_text"]))
            _, state = self.adapter.forward(history_tokens, None)
            source_states[history_key] = clone_state(state)

        restored_states = {}
        random_states = {}
        snapshot_reports = {}
        with tempfile.TemporaryDirectory(prefix="psa-exp001c-v02-stage-b-") as temporary:
            for block_id, block_trials in sorted(trials_by_block.items()):
                by_fields, domains, operations = _factorial_layout(block_trials)
                combo_states = {}
                combo_history_keys = {}
                for identity, domain in enumerate(domains):
                    for goal, operation in enumerate(operations):
                        combo = (identity, goal)
                        trial = by_fields[(domain, operation)]
                        history_key = str(trial["history_key"])
                        combo_history_keys[combo] = history_key
                        combo_states[combo] = clone_state(
                            source_states[history_key]
                        )
                restored, reports = self.snapshot_roundtrip(
                    combo_states,
                    adapter=self.adapter,
                    directory=Path(temporary) / block_id,
                )
                for combo, history_key in combo_history_keys.items():
                    restored_states[history_key] = clone_state(restored[combo])
                    random_states[history_key] = self.randomizer(
                        source_states[history_key],
                        self.adapter.torch,
                        seed=derive_random_state_seed(
                            group_id=block_id,
                            combo=combo,
                            namespace=RANDOM_NAMESPACE,
                        ),
                    )
                snapshot_reports[block_id] = reports

        if (
            set(restored_states) != set(source_states)
            or set(random_states) != set(source_states)
        ):
            raise ValueError("Stage B derived state inventory is incomplete")

        forced_prefix = str(self.protocol_manifest.get("forced_answer_prefix", ""))
        prefix_tokens = self.adapter.encode(forced_prefix)
        if len(prefix_tokens) != 2:
            raise ValueError("Stage B forced prefix must contain exactly two tokens")
        rendered_answers = {code: code for code in "ABCD"}
        result_records = []
        for route in design_records:
            if not isinstance(route, Mapping):
                raise ValueError("Stage B route must be an object")
            query_id = str(route.get("query_sample_id", ""))
            query_trial = trials_by_id.get(query_id)
            if (
                query_trial is None
                or query_trial.get("query_digest_sha256")
                != route.get("query_digest_sha256")
            ):
                raise ValueError("Stage B route query is absent or drifted")
            condition = str(route.get("condition", ""))
            source_history_key = route.get("state_source_history_key")
            if condition == "reset":
                source_state = None
            elif not isinstance(source_history_key, str):
                raise ValueError("Stage B state route lacks a source history")
            elif condition == "restored":
                source_state = clone_state(restored_states[source_history_key])
            elif condition == "random_matched":
                source_state = clone_state(random_states[source_history_key])
            else:
                source_state = clone_state(source_states[source_history_key])

            query_tokens = self.adapter.encode(str(query_trial["query_text"]))
            logits, query_state = self.adapter.forward(
                query_tokens,
                source_state,
            )
            prefix_evidence, answer_logits, answer_state = instrument_forced_prefix(
                self.adapter,
                logits=logits,
                state=query_state,
                prefix_token_ids=prefix_tokens,
                forced_prefix_text=forced_prefix,
                top_k=10,
            )
            option_scores = self.option_scorer(
                self.adapter,
                logits=answer_logits,
                state=answer_state,
                rendered_answers=rendered_answers,
            )
            if set(option_scores) != set("ABCD"):
                raise ValueError("Stage B scorer must return exactly A-D")
            scores = {code: float(option_scores[code]) for code in "ABCD"}
            target_code = route.get("expected_state_semantic_target_code")
            boundary = (
                answer_boundary_evidence(scores, target_code=str(target_code))
                if target_code is not None
                else None
            )
            result_records.append(
                {
                    "record_id": route["record_id"],
                    "condition": condition,
                    "condition_role": route["condition_role"],
                    "query_sample_id": query_id,
                    "semantic_case_id": route["semantic_case_id"],
                    "block_id": route["block_id"],
                    "rotation_index": route["rotation_index"],
                    "query_history_key": route["query_history_key"],
                    "state_source_sample_id": route[
                        "state_source_sample_id"
                    ],
                    "state_source_history_key": source_history_key,
                    "state_source_fields": route["state_source_fields"],
                    "reference_stage_a_target_code": route[
                        "reference_stage_a_target_code"
                    ],
                    "expected_state_semantic_target_code": target_code,
                    "semantic_endpoint_role": route[
                        "semantic_endpoint_role"
                    ],
                    "query_token_count": len(query_tokens),
                    "prefix_evidence": prefix_evidence,
                    "option_log_probabilities": scores,
                    "predicted_code": max(scores, key=scores.__getitem__),
                    "answer_boundary_evidence": boundary,
                }
            )

        if [record["record_id"] for record in result_records] != [
            route["record_id"] for route in design_records
        ]:
            raise ValueError("Stage B result ordering drifted from the design")
        return {
            "result_version": STAGE_B_RESULT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "v02_stage_b_recurrent_state_complete",
            "development_only": True,
            "non_core": True,
            "model_executed": True,
            "recurrent_state_accessed": True,
            "source_states_cloned_per_route": True,
            "stage_a_rerun": False,
            "formal_test_set_accessed": False,
            "formal_run": False,
            "contains_confirmatory_decision": False,
            "automatic_rerun_authorized": False,
            "design_manifest_digest_sha256": design_manifest.get(
                "design_manifest_digest_sha256"
            ),
            "protocol_manifest_digest_sha256": self.protocol_manifest.get(
                "manifest_digest_sha256"
            ),
            "condition_count": 7,
            "record_count": len(result_records),
            "warmup_token_lengths": warmup_token_lengths,
            "snapshot_roundtrip_reports": snapshot_reports,
            "records": result_records,
        }


def build_exp001c_v02_stage_b_rwkv_backend(
    *,
    design_manifest_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
    execution_authority_validated: bool = False,
) -> RWKVExp001CV02StageBBackend:
    """Load the real model only when a future authorized runner calls this factory."""
    if execution_authority_validated is not True:
        raise PermissionError(
            "Stage B backend factory requires validated execution authority"
        )
    root = Path(project_root).resolve()
    verification = verify_exp001c_v02_stage_b_design_manifest(
        design_manifest_path,
        project_root=root,
    )
    if verification.get("valid") is not True:
        raise ValueError("Stage B design manifest verification failed")
    design = _load_object(
        design_manifest_path,
        "EXP-001C v02 Stage B design manifest",
        root=root,
    )
    design_entry = design.get("design_config")
    if not isinstance(design_entry, Mapping):
        raise ValueError("Stage B design config entry is missing")
    design_config = _load_object(
        str(design_entry.get("path", "")),
        "EXP-001C v02 Stage B design config",
        root=root,
    )
    protocol_manifest = build_exp001c_protocol_v02_manifest(
        config_path=root / str(design_config.get("protocol_config_path", "")),
        project_root=root,
    )
    if (
        protocol_manifest.get("manifest_digest_sha256")
        != design.get("protocol_manifest_digest_sha256")
    ):
        raise ValueError("Stage B protocol manifest digest drifted")
    model_entry = protocol_manifest.get("model_config")
    requested_model = Path(model_config_path)
    if not requested_model.is_absolute():
        requested_model = root / requested_model
    requested_model = requested_model.resolve()
    if not isinstance(model_entry, Mapping):
        raise ValueError("Stage B protocol lacks a model config")
    locked_model = (root / str(model_entry.get("path", ""))).resolve()
    if (
        requested_model != locked_model
        or not requested_model.is_file()
        or sha256_file(requested_model) != model_entry.get("sha256")
    ):
        raise ValueError("model config does not match the locked Stage B design")
    model_config = load_model_config(
        requested_model,
        project_root=root,
        verify_files=True,
    )
    return RWKVExp001CV02StageBBackend(
        adapter=RWKV7Adapter.load(model_config),
        protocol_manifest=protocol_manifest,
    )
