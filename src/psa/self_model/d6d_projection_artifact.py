from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_json
from psa.self_model.d6d_core_approach_design import SELF_CONDITIONS
from psa.self_model.state import validate_self_state


ARTIFACT_VERSION = "0.1-d6d-field-separated-frozen-projection"
TRAINER_KIND = "categorical_branch_mean_pure_python_v0.1"
SOURCE_FIELDS = ("identity_anchors", "active_goals")
PROJECTION_VECTOR_DIGEST_FORMAT = ".12e"


@dataclass(frozen=True)
class ProjectionTrainingRecord:
    identity_key: str
    goal_key: str
    identity_target: tuple[float, ...]
    goal_target: tuple[float, ...]


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite_vector(value: Sequence[Any], dimension: int) -> tuple[float, ...]:
    if len(value) != dimension:
        raise ValueError("D6D-I projection vector dimension changed")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError("D6D-I projection vector must be finite")
    return converted


def _mean(vectors: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("D6D-I projection branch has no training record")
    return tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0])))


def _payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(artifact))
    value.pop("artifact_digest_sha256", None)
    return value


def build_frozen_projection_artifact(
    *,
    records: Sequence[ProjectionTrainingRecord],
    output_dimension: int,
    training_manifest_sha256: str,
    pilot_manifest_commitment_sha256: str,
    optimizer_seed: int,
    fixture_only: bool,
) -> dict[str, Any]:
    if not isinstance(output_dimension, int) or output_dimension < 4:
        raise ValueError("D6D-I output dimension must be at least four")
    if not _hex_digest(training_manifest_sha256) or not _hex_digest(
        pilot_manifest_commitment_sha256
    ) or training_manifest_sha256 == pilot_manifest_commitment_sha256:
        raise PermissionError("D6D-I training and blinded pilot commitments must differ")
    if not isinstance(optimizer_seed, int) or optimizer_seed < 0:
        raise ValueError("D6D-I optimizer seed must be non-negative")
    if not records:
        raise ValueError("D6D-I projection training records are empty")
    identity_groups: dict[str, list[tuple[float, ...]]] = {}
    goal_groups: dict[str, list[tuple[float, ...]]] = {}
    for record in records:
        if type(record) is not ProjectionTrainingRecord:
            raise TypeError("D6D-I training record type changed")
        if not record.identity_key or not record.goal_key:
            raise ValueError("D6D-I categorical projection keys must be non-empty")
        identity_groups.setdefault(record.identity_key, []).append(
            _finite_vector(record.identity_target, output_dimension)
        )
        goal_groups.setdefault(record.goal_key, []).append(
            _finite_vector(record.goal_target, output_dimension)
        )
    identity_weights = {
        key: list(_mean(identity_groups[key])) for key in sorted(identity_groups)
    }
    goal_weights = {key: list(_mean(goal_groups[key])) for key in sorted(goal_groups)}
    parameters = {
        "identity_weights": identity_weights,
        "goal_weights": goal_weights,
    }
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_kind": "field_separated_learned_frozen_self_projection",
        "trainer_kind": TRAINER_KIND,
        "status": "frozen",
        "fixture_only": bool(fixture_only),
        "research_evidence_eligible": not bool(fixture_only),
        "synthetic_or_hash_fake": False,
        "source_contract": "validated_Self_State_v0.1",
        "source_fields": list(SOURCE_FIELDS),
        "output_dimension": output_dimension,
        "target_layer_index_zero_based": 15,
        "target_phase": "post_ffn_residual",
        "bias_present": False,
        "double_mask_projection_exact_zero": True,
        "natural_language_prompt_serialization_used": False,
        "base_model_parameters_present": False,
        "online_update_available": False,
        "training_manifest_sha256": training_manifest_sha256,
        "pilot_manifest_commitment_sha256": pilot_manifest_commitment_sha256,
        "optimizer": {
            "algorithm": "branch_category_arithmetic_mean",
            "seed": optimizer_seed,
            "posthoc_pilot_selection": False,
        },
        "vocabularies": {
            "identity": sorted(identity_weights),
            "goal": sorted(goal_weights),
        },
        "parameters": parameters,
        "parameter_digest_sha256": sha256_json(parameters),
    }
    artifact["artifact_digest_sha256"] = sha256_json(artifact)
    return audit_frozen_projection_artifact(artifact)["artifact"]


def audit_frozen_projection_artifact(
    artifact: Mapping[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(dict(artifact))
    parameters = value.get("parameters", {})
    identity = parameters.get("identity_weights", {}) if isinstance(parameters, Mapping) else {}
    goal = parameters.get("goal_weights", {}) if isinstance(parameters, Mapping) else {}
    dimension = value.get("output_dimension")
    vectors = [*identity.values(), *goal.values()] if isinstance(identity, Mapping) and isinstance(goal, Mapping) else []
    checks = {
        "identity_exact": value.get("artifact_version") == ARTIFACT_VERSION
        and value.get("artifact_kind")
        == "field_separated_learned_frozen_self_projection"
        and value.get("trainer_kind") == TRAINER_KIND
        and value.get("status") == "frozen",
        "field_and_site_contract_exact": value.get("source_fields")
        == list(SOURCE_FIELDS)
        and value.get("target_layer_index_zero_based") == 15
        and value.get("target_phase") == "post_ffn_residual",
        "no_bias_or_prompt_or_base_parameters": value.get("bias_present") is False
        and value.get("double_mask_projection_exact_zero") is True
        and value.get("natural_language_prompt_serialization_used") is False
        and value.get("base_model_parameters_present") is False,
        "training_and_pilot_commitments_valid": _hex_digest(
            value.get("training_manifest_sha256")
        )
        and _hex_digest(value.get("pilot_manifest_commitment_sha256"))
        and value.get("training_manifest_sha256")
        != value.get("pilot_manifest_commitment_sha256"),
        "parameters_nonempty_finite_and_shaped": isinstance(dimension, int)
        and dimension >= 4
        and bool(identity)
        and bool(goal)
        and all(len(vector) == dimension for vector in vectors)
        and all(math.isfinite(float(item)) for vector in vectors for item in vector),
        "vocabularies_match_parameters": value.get("vocabularies")
        == {"identity": sorted(identity), "goal": sorted(goal)},
        "parameter_digest_valid": value.get("parameter_digest_sha256")
        == sha256_json(parameters),
        "artifact_digest_valid": value.get("artifact_digest_sha256")
        == sha256_json(_payload(value)),
        "no_online_update": value.get("online_update_available") is False,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D-I frozen projection artifact failed: " + ", ".join(failed))
    return {
        "valid": True,
        "checks": checks,
        "artifact": value,
        "artifact_digest_sha256": value["artifact_digest_sha256"],
        "parameter_digest_sha256": value["parameter_digest_sha256"],
    }


def _active_string_key(state: Mapping[str, Any], field: str) -> str:
    active = [
        item for item in state[field]
        if item["status"] == "active" and item["value_type"] == "string"
    ]
    if len(active) != 1 or not isinstance(active[0]["value"], str) or not active[0]["value"]:
        raise ValueError(f"D6D-I {field} requires one active string key")
    return active[0]["value"]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def projection_vector_digest(value: Sequence[float]) -> str:
    """Return a cross-platform evidence digest for a finite projection vector."""
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError("D6D-I projection digest vector must be finite")
    return sha256_json(
        [format(item, PROJECTION_VECTOR_DIGEST_FORMAT) for item in converted]
    )


def _randomize_norm_matched(
    source: tuple[float, ...], *, seed: int, branch: str
) -> tuple[float, ...]:
    generator = random.Random(f"PSA|D6D-I|{seed}|{branch}")
    values = [generator.gauss(0.0, 1.0) for _ in source]
    center = sum(values) / len(values)
    centered = [item - center for item in values]
    norm = _norm(centered)
    source_norm = _norm(source)
    if norm == 0.0:
        raise RuntimeError("D6D-I random branch norm is zero")
    return tuple(item * source_norm / norm for item in centered)


class FrozenSelfProjection:
    def __init__(self, artifact: Mapping[str, Any]) -> None:
        audited = audit_frozen_projection_artifact(artifact)
        self.artifact = audited["artifact"]
        self.dimension = self.artifact["output_dimension"]

    def _weight(self, branch: str, key: str) -> tuple[float, ...]:
        values = self.artifact["parameters"][f"{branch}_weights"].get(key)
        if values is None:
            raise KeyError(f"D6D-I frozen {branch} key is absent")
        return tuple(float(item) for item in values)

    def project_condition(
        self,
        *,
        matched_state: Mapping[str, Any],
        paired_state: Mapping[str, Any],
        condition: str,
        random_seed: int,
    ) -> dict[str, Any]:
        if condition not in SELF_CONDITIONS:
            raise PermissionError("D6D-I projection accepts only frozen Self conditions")
        matched = validate_self_state(matched_state)
        paired = validate_self_state(paired_state)
        matched_identity = _active_string_key(matched, "identity_anchors")
        matched_goal = _active_string_key(matched, "active_goals")
        paired_identity = _active_string_key(paired, "identity_anchors")
        paired_goal = _active_string_key(paired, "active_goals")
        identity_key = paired_identity if condition in {
            "self_identity_swap", "self_identity_goal_swap"
        } else matched_identity
        goal_key = paired_goal if condition in {
            "self_goal_swap", "self_identity_goal_swap"
        } else matched_goal
        identity = self._weight("identity", identity_key)
        goal = self._weight("goal", goal_key)
        if condition in {"self_identity_mask", "self_identity_goal_mask"}:
            identity = tuple(0.0 for _ in identity)
        if condition in {"self_goal_mask", "self_identity_goal_mask"}:
            goal = tuple(0.0 for _ in goal)
        if condition == "self_identity_goal_norm_matched_random":
            identity = _randomize_norm_matched(identity, seed=random_seed, branch="identity")
            goal = _randomize_norm_matched(goal, seed=random_seed, branch="goal")
        aggregate = tuple(left + right for left, right in zip(identity, goal))
        return {
            "condition": condition,
            "identity_key": identity_key,
            "goal_key": goal_key,
            "identity_vector": identity,
            "goal_vector": goal,
            "aggregate_vector": aggregate,
            "identity_l2_norm": _norm(identity),
            "goal_l2_norm": _norm(goal),
            "aggregate_digest_sha256": projection_vector_digest(aggregate),
            "aggregate_digest_canonicalization": (
                "finite_float_scientific_12_decimal_places"
            ),
            "artifact_digest_sha256": self.artifact["artifact_digest_sha256"],
            "fixture_only": self.artifact["fixture_only"],
            "research_evidence_eligible": self.artifact["research_evidence_eligible"],
            "model_loaded": False,
            "model_executed": False,
        }
