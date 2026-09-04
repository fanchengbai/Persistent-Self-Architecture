from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json


CONTRACT_VERSION = "0.1-self-model-d9c-calibration-only-projection"
ARTIFACT_VERSION = "0.1-self-model-d9c-frozen-projection-artifact"
CONTRACT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9_projection_contract.json"
)
CALIBRATION_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9_calibration_manifest.json"
)
HELDOUT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9_heldout_manifest.json"
)
SCHEDULE_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9_within_wrapper_schedule.json"
)
CALIBRATION_SHA256 = (
    "0da7c885d9ffae14e097eb73241cc8b56b9e15beb587c1c4d10913c054b6d07b"
)
HELDOUT_SHA256 = (
    "3f70265716623ccfac264f44f6a7e900e90dd0f589e8269cb9869a960b629e4c"
)
SCHEDULE_SHA256 = (
    "04a359738166b386154aa13434902cd00d3552a4edbb9acc033d0dc8e11333d2"
)
CALIBRATION_COMMITMENT = (
    "2e8d555efdd81bdbee3ca13a56513d9c9bb66bf53a72f5ce424f324dc1c4fc39"
)
HELDOUT_COMMITMENT = (
    "02d33c92f3da78ca259b5fad9c3af7bcab14ecf36a90cad830a03f9361b315e4"
)
SCHEDULE_COMMITMENT = (
    "a6b34ef7eb9912632f65b245f17b5a7583be9fa874f6f95fbc970b0abc6085b9"
)
IDENTITY_KEYS = tuple(f"identity_{index}" for index in range(4))
GOAL_KEYS = tuple(f"goal_{index}" for index in range(4))


@dataclass(frozen=True)
class CalibrationCapture:
    fixture_id: str
    identity_index: int
    goal_index: int
    replicate: int
    vector: tuple[float, ...]


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D9-C {label} must be an object")
    return value


def _hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite_vector(value: Sequence[Any], dimension: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != dimension:
        raise ValueError("D9-C projection vector dimension changed")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("D9-C projection vector must be finite")
    return result


def _mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("D9-C cannot average an empty vector group")
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    )


def _rms(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector) / len(vector))


def _scaled(vector: Sequence[float], target_rms: float) -> tuple[float, ...]:
    current = _rms(vector)
    if current == 0.0 or not math.isfinite(current):
        raise ValueError("D9-C projection branch has invalid RMS")
    return tuple(float(value) * target_rms / current for value in vector)


def _without_digest(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(artifact))
    value.pop("artifact_digest_sha256", None)
    return value


def validate_projection_contract(contract: Mapping[str, Any]) -> dict[str, bool]:
    sources = contract.get("source_manifests", {})
    capture = contract.get("capture_contract", {})
    fitting = contract.get("fitting_contract", {})
    schema = contract.get("artifact_schema", {})
    conditions = contract.get("condition_projection_contract", {})
    scoring = contract.get("heldout_scoring_contract", {})
    checks = {
        "identity_exact": contract.get("contract_version") == CONTRACT_VERSION
        and contract.get("contract_id")
        == "Self-Model-v0.1-D9-calibration-only-field-separated-projection-v01"
        and contract.get("status")
        == "projection_contract_frozen_unrun_no_real_artifact",
        "source_manifests_exact": sources
        == {
            "calibration_path": CALIBRATION_RELATIVE_PATH,
            "calibration_sha256": CALIBRATION_SHA256,
            "calibration_commitment_sha256": CALIBRATION_COMMITMENT,
            "heldout_path": HELDOUT_RELATIVE_PATH,
            "heldout_sha256": HELDOUT_SHA256,
            "heldout_commitment_sha256": HELDOUT_COMMITMENT,
            "schedule_path": SCHEDULE_RELATIVE_PATH,
            "schedule_sha256": SCHEDULE_SHA256,
            "schedule_commitment_sha256": SCHEDULE_COMMITMENT,
        },
        "calibration_only_capture_exact": capture.get("fixture_count") == 32
        and capture.get("identity_levels") == 4
        and capture.get("goal_levels") == 4
        and capture.get("replicates_per_identity_goal_cell") == 2
        and capture.get("route") == "persistent_wrapper_capture"
        and capture.get("target_layer_index_zero_based") == 15
        and capture.get("target_phase") == "post_ffn_residual"
        and capture.get("residual_returned_unchanged") is True
        and capture.get("base_model_parameters_trainable") is False
        and capture.get("heldout_payload_access_before_artifact_freeze") is False,
        "fitting_exact": fitting.get("algorithm")
        == "replicate_mean_then_two_way_centered_additive_branch_means_closed_form_v0.1"
        and fitting.get("replicate_reduction")
        == "arithmetic_mean_within_each_identity_goal_cell"
        and fitting.get("branch_target_rms_ratio_each") == 0.005
        and fitting.get("optimizer_kind")
        == "closed_form_no_gradient_optimizer"
        and fitting.get("optimizer_seed") == 29083104
        and fitting.get("output_dimension") == 2560
        and fitting.get("bias_present") is False
        and fitting.get("double_mask_projection_exact_zero") is True
        and fitting.get("heldout_labels_or_outputs_used") is False,
        "artifact_schema_exact": schema.get("artifact_version") == ARTIFACT_VERSION
        and schema.get("artifact_kind")
        == "calibration_only_field_separated_frozen_self_projection"
        and schema.get("status") == "frozen_before_heldout_access"
        and schema.get("identity_vocabulary") == list(IDENTITY_KEYS)
        and schema.get("goal_vocabulary") == list(GOAL_KEYS)
        and schema.get("parameter_groups")
        == ["identity_weights", "goal_weights"]
        and schema.get("vectors_per_group") == 4
        and schema.get("vector_dimension") == 2560
        and schema.get("finite_values_required") is True
        and schema.get("online_update_available") is False
        and schema.get("base_model_parameters_present") is False
        and schema.get("training_records_or_heldout_payload_embedded") is False,
        "seven_conditions_exact": conditions
        == {
            "wrapper_zero": "exact_zero_vector_without_artifact_lookup",
            "active_true": "identity_true_plus_goal_true",
            "mask_identity": "goal_true_only",
            "mask_goal": "identity_true_only",
            "swap_identity": "identity_next_cyclic_plus_goal_true",
            "swap_goal": "identity_true_plus_goal_next_cyclic",
            "matched_random": (
                "deterministic_norm_matched_random_vector_seeded_by_artifact_and_fixture"
            ),
            "synthetic_active": (
                "separate_fixed_positive_control_not_from_projection_artifact"
            ),
        },
        "heldout_scoring_predeclared": scoring
        == {
            "choice_token_ids": {"A": 66, "B": 67, "C": 68, "D": 69},
            "semantic_target_index": (
                "identity_index_times_3_plus_goal_index_modulo_4"
            ),
            "rotation_target_code": (
                "semantic_target_index_plus_code_rotation_modulo_4"
            ),
            "identity_swap_target_code": (
                "identity_index_plus_1_cyclic_then_same_goal_and_rotation"
            ),
            "goal_swap_target_code": (
                "same_identity_then_goal_index_plus_1_cyclic_and_rotation"
            ),
            "target_alignment_margin": (
                "true_target_logit_minus_max_other_choice_logit"
            ),
            "identity_margin": (
                "true_target_logit_minus_identity_swap_target_logit"
            ),
            "goal_margin": "true_target_logit_minus_goal_swap_target_logit",
            "mask_identity_specific": (
                "mask_identity_identity_margin_minus_zero_less_than_zero_and_"
                "mask_identity_goal_margin_minus_zero_greater_than_zero"
            ),
            "mask_goal_specific": (
                "mask_goal_goal_margin_minus_zero_less_than_zero_and_mask_goal_"
                "identity_margin_minus_zero_greater_than_zero"
            ),
            "swap_identity_follows": (
                "swap_identity_target_logit_strictly_greater_than_true_target_"
                "logit_after_rotation_marginalization"
            ),
            "swap_goal_follows": (
                "swap_goal_target_logit_strictly_greater_than_true_target_logit_"
                "after_rotation_marginalization"
            ),
            "posthoc_scoring_change_allowed": False,
        },
        "freeze_precedes_heldout": contract.get("freeze_and_access_order", [])[-1]
        == "only_then_load_64_heldout_fixtures_and_448_pair_schedule"
        and contract.get("freeze_and_access_order", []).index(
            "exclusive_create_projection_artifact"
        )
        < contract.get("freeze_and_access_order", []).index(
            "only_then_load_64_heldout_fixtures_and_448_pair_schedule"
        ),
        "unrun_no_artifact_or_model": contract.get("real_artifact_created") is False
        and contract.get("model_executed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            "D9-C projection contract failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def build_frozen_projection_artifact(
    *,
    captures: Sequence[CalibrationCapture],
    calibration_manifest_sha256: str,
    calibration_commitment_sha256: str,
    heldout_manifest_sha256: str,
    heldout_commitment_sha256: str,
    schedule_commitment_sha256: str,
    output_dimension: int = 2560,
    fixture_only: bool = False,
) -> dict[str, Any]:
    if len(captures) != 32:
        raise ValueError("D9-C requires exactly 32 calibration captures")
    if not all(
        _hex(value)
        for value in (
            calibration_manifest_sha256,
            calibration_commitment_sha256,
            heldout_manifest_sha256,
            heldout_commitment_sha256,
            schedule_commitment_sha256,
        )
    ):
        raise ValueError("D9-C source digest is invalid")
    if calibration_manifest_sha256 == heldout_manifest_sha256:
        raise PermissionError("D9-C calibration and heldout sources must differ")
    if not fixture_only and output_dimension != 2560:
        raise ValueError("D9-C real projection output dimension must be 2560")
    if fixture_only and output_dimension < 4:
        raise ValueError("D9-C fake projection dimension must be at least four")
    by_cell: dict[tuple[int, int], list[tuple[float, ...]]] = {}
    fixture_ids: set[str] = set()
    for capture in captures:
        if type(capture) is not CalibrationCapture:
            raise TypeError("D9-C capture type changed")
        if capture.fixture_id in fixture_ids or not capture.fixture_id.startswith("d9cal-"):
            raise ValueError("D9-C calibration fixture ID is duplicate or out of phase")
        fixture_ids.add(capture.fixture_id)
        if capture.identity_index not in range(4) or capture.goal_index not in range(4):
            raise ValueError("D9-C calibration field index is invalid")
        if capture.replicate not in (1, 2):
            raise ValueError("D9-C replicate must be one or two")
        by_cell.setdefault((capture.identity_index, capture.goal_index), []).append(
            _finite_vector(capture.vector, output_dimension)
        )
    expected = {(identity, goal) for identity in range(4) for goal in range(4)}
    if set(by_cell) != expected or any(len(vectors) != 2 for vectors in by_cell.values()):
        raise ValueError("D9-C calibration grid or replicate count is incomplete")
    cell_means = {key: _mean(vectors) for key, vectors in by_cell.items()}
    grand = _mean(list(cell_means.values()))
    grand_rms = _rms(grand)
    if grand_rms == 0.0 or not math.isfinite(grand_rms):
        raise ValueError("D9-C calibration grand mean RMS is invalid")
    identity_weights: dict[str, list[float]] = {}
    for identity in range(4):
        mean = _mean([cell_means[(identity, goal)] for goal in range(4)])
        raw = tuple(value - grand[index] / 2.0 for index, value in enumerate(mean))
        identity_weights[IDENTITY_KEYS[identity]] = list(
            _scaled(raw, grand_rms * 0.005)
        )
    goal_weights: dict[str, list[float]] = {}
    for goal in range(4):
        mean = _mean([cell_means[(identity, goal)] for identity in range(4)])
        raw = tuple(value - grand[index] / 2.0 for index, value in enumerate(mean))
        goal_weights[GOAL_KEYS[goal]] = list(_scaled(raw, grand_rms * 0.005))
    parameters = {
        "identity_weights": identity_weights,
        "goal_weights": goal_weights,
    }
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_kind": "calibration_only_field_separated_frozen_self_projection",
        "status": "frozen_before_heldout_access",
        "fixture_only": bool(fixture_only),
        "research_evidence_eligible": not bool(fixture_only),
        "source": {
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "calibration_commitment_sha256": calibration_commitment_sha256,
            "calibration_capture_count": 32,
            "heldout_manifest_sha256": heldout_manifest_sha256,
            "heldout_commitment_sha256": heldout_commitment_sha256,
            "schedule_commitment_sha256": schedule_commitment_sha256,
            "heldout_payload_accessed_during_fit": False,
        },
        "site": {
            "target_layer_index_zero_based": 15,
            "target_phase": "post_ffn_residual",
            "output_dimension": output_dimension,
        },
        "fitting": {
            "algorithm": (
                "replicate_mean_then_two_way_centered_additive_branch_means_closed_form_v0.1"
            ),
            "optimizer_seed": 29083104,
            "branch_target_rms_ratio_each": 0.005,
            "bias_present": False,
            "base_model_parameters_trained": False,
            "online_update_available": False,
        },
        "vocabularies": {
            "identity": list(IDENTITY_KEYS),
            "goal": list(GOAL_KEYS),
        },
        "parameters": parameters,
        "parameter_digest_sha256": sha256_json(parameters),
    }
    artifact["artifact_digest_sha256"] = sha256_json(artifact)
    return audit_frozen_projection_artifact(
        artifact, expected_dimension=output_dimension, fixture_only=fixture_only
    )["artifact"]


def audit_frozen_projection_artifact(
    artifact: Mapping[str, Any], *, expected_dimension: int, fixture_only: bool
) -> dict[str, Any]:
    value = copy.deepcopy(dict(artifact))
    source = value.get("source", {})
    site = value.get("site", {})
    fitting = value.get("fitting", {})
    parameters = value.get("parameters", {})
    identity = parameters.get("identity_weights", {}) if isinstance(parameters, Mapping) else {}
    goal = parameters.get("goal_weights", {}) if isinstance(parameters, Mapping) else {}
    vectors = list(identity.values()) + list(goal.values()) if identity and goal else []
    checks = {
        "identity_and_freeze_exact": value.get("artifact_version") == ARTIFACT_VERSION
        and value.get("artifact_kind")
        == "calibration_only_field_separated_frozen_self_projection"
        and value.get("status") == "frozen_before_heldout_access",
        "fixture_and_evidence_boundary_exact": value.get("fixture_only")
        is fixture_only
        and value.get("research_evidence_eligible") is (not fixture_only),
        "calibration_only_source_exact": source.get("calibration_capture_count") == 32
        and source.get("heldout_payload_accessed_during_fit") is False
        and all(
            _hex(source.get(field))
            for field in (
                "calibration_manifest_sha256",
                "calibration_commitment_sha256",
                "heldout_manifest_sha256",
                "heldout_commitment_sha256",
                "schedule_commitment_sha256",
            )
        ),
        "site_exact": site
        == {
            "target_layer_index_zero_based": 15,
            "target_phase": "post_ffn_residual",
            "output_dimension": expected_dimension,
        },
        "fitting_is_frozen_closed_form_no_base_training": fitting.get("algorithm")
        == "replicate_mean_then_two_way_centered_additive_branch_means_closed_form_v0.1"
        and fitting.get("optimizer_seed") == 29083104
        and fitting.get("branch_target_rms_ratio_each") == 0.005
        and fitting.get("bias_present") is False
        and fitting.get("base_model_parameters_trained") is False
        and fitting.get("online_update_available") is False,
        "vocabularies_exact": value.get("vocabularies")
        == {"identity": list(IDENTITY_KEYS), "goal": list(GOAL_KEYS)},
        "parameters_complete_finite_and_shaped": set(identity) == set(IDENTITY_KEYS)
        and set(goal) == set(GOAL_KEYS)
        and len(vectors) == 8
        and all(len(vector) == expected_dimension for vector in vectors)
        and all(math.isfinite(float(item)) for vector in vectors for item in vector),
        "parameter_digest_valid": value.get("parameter_digest_sha256")
        == sha256_json(parameters),
        "artifact_digest_valid": value.get("artifact_digest_sha256")
        == sha256_json(_without_digest(value)),
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-C frozen projection artifact failed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return {
        "valid": True,
        "checks": checks,
        "artifact": value,
        "parameter_digest_sha256": value["parameter_digest_sha256"],
        "artifact_digest_sha256": value["artifact_digest_sha256"],
    }


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def project_condition(
    artifact: Mapping[str, Any], *, condition: str, identity_index: int,
    goal_index: int, fixture_id: str,
) -> tuple[float, ...]:
    dimension = int(artifact["site"]["output_dimension"])
    audit_frozen_projection_artifact(
        artifact, expected_dimension=dimension, fixture_only=bool(artifact["fixture_only"])
    )
    if identity_index not in range(4) or goal_index not in range(4):
        raise ValueError("D9-C projection field index is invalid")
    identity = artifact["parameters"]["identity_weights"]
    goal = artifact["parameters"]["goal_weights"]
    i_true = identity[IDENTITY_KEYS[identity_index]]
    g_true = goal[GOAL_KEYS[goal_index]]
    if condition == "wrapper_zero":
        return tuple(0.0 for _ in range(dimension))
    if condition == "active_true":
        return _add(i_true, g_true)
    if condition == "mask_identity":
        return tuple(g_true)
    if condition == "mask_goal":
        return tuple(i_true)
    if condition == "swap_identity":
        return _add(identity[IDENTITY_KEYS[(identity_index + 1) % 4]], g_true)
    if condition == "swap_goal":
        return _add(i_true, goal[GOAL_KEYS[(goal_index + 1) % 4]])
    if condition == "matched_random":
        target = _rms(_add(i_true, g_true))
        seed = int(
            hashlib.sha256(
                f"{artifact['artifact_digest_sha256']}|{fixture_id}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        rng = random.Random(seed)
        raw = tuple(rng.gauss(0.0, 1.0) for _ in range(dimension))
        return _scaled(raw, target)
    if condition == "synthetic_active":
        raise PermissionError("D9-C synthetic active is separate from projection artifact")
    raise ValueError("D9-C projection condition is unknown")


def _fake_captures(dimension: int = 8) -> list[CalibrationCapture]:
    captures: list[CalibrationCapture] = []
    index = 0
    for identity in range(4):
        for goal in range(4):
            for replicate in (1, 2):
                index += 1
                vector = tuple(
                    1.0
                    + identity * 0.3
                    + goal * 0.07
                    + replicate * 0.005
                    + position * 0.011
                    for position in range(dimension)
                )
                captures.append(
                    CalibrationCapture(
                        fixture_id=f"d9cal-{index:03d}",
                        identity_index=identity,
                        goal_index=goal,
                        replicate=replicate,
                        vector=vector,
                    )
                )
    return captures


def _fake_artifact() -> dict[str, Any]:
    return build_frozen_projection_artifact(
        captures=_fake_captures(),
        calibration_manifest_sha256=CALIBRATION_SHA256,
        calibration_commitment_sha256=CALIBRATION_COMMITMENT,
        heldout_manifest_sha256=HELDOUT_SHA256,
        heldout_commitment_sha256=HELDOUT_COMMITMENT,
        schedule_commitment_sha256=SCHEDULE_COMMITMENT,
        output_dimension=8,
        fixture_only=True,
    )


def _fails(action: Any) -> bool:
    try:
        action()
    except (TypeError, ValueError, PermissionError):
        return True
    return False


def run_fake_projection_acceptance() -> dict[str, Any]:
    captures = _fake_captures()
    captures_before = copy.deepcopy(captures)
    artifact = _fake_artifact()
    artifact_before = copy.deepcopy(artifact)
    active = project_condition(
        artifact, condition="active_true", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    zero = project_condition(
        artifact, condition="wrapper_zero", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    mask_identity = project_condition(
        artifact, condition="mask_identity", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    mask_goal = project_condition(
        artifact, condition="mask_goal", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    swapped_identity = project_condition(
        artifact, condition="swap_identity", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    swapped_goal = project_condition(
        artifact, condition="swap_goal", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    random_one = project_condition(
        artifact, condition="matched_random", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    random_two = project_condition(
        artifact, condition="matched_random", identity_index=1, goal_index=2,
        fixture_id="d9hold-001",
    )
    tampered = copy.deepcopy(artifact)
    tampered["parameters"]["identity_weights"]["identity_0"][0] += 1.0
    missing = captures[:-1]
    duplicate = copy.deepcopy(captures)
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    heldout_leak = copy.deepcopy(captures)
    heldout_leak[0] = CalibrationCapture(
        fixture_id="d9hold-001",
        identity_index=0,
        goal_index=0,
        replicate=1,
        vector=heldout_leak[0].vector,
    )
    nonfinite = copy.deepcopy(captures)
    nonfinite[0] = CalibrationCapture(
        fixture_id=nonfinite[0].fixture_id,
        identity_index=0,
        goal_index=0,
        replicate=1,
        vector=(math.nan,) + nonfinite[0].vector[1:],
    )
    checks = {
        "fake_artifact_audits": audit_frozen_projection_artifact(
            artifact, expected_dimension=8, fixture_only=True
        )["valid"],
        "fake_artifact_not_research_evidence": artifact[
            "research_evidence_eligible"
        ]
        is False,
        "zero_exact": all(value == 0.0 for value in zero),
        "active_is_sum_of_masks": active == _add(mask_identity, mask_goal),
        "identity_and_goal_swaps_change_distinct_vectors": swapped_identity != active
        and swapped_goal != active
        and swapped_identity != swapped_goal,
        "matched_random_deterministic_and_norm_matched": random_one == random_two
        and math.isclose(_rms(random_one), _rms(active), rel_tol=1e-12),
        "synthetic_projection_lookup_rejected": _fails(
            lambda: project_condition(
                artifact, condition="synthetic_active", identity_index=1,
                goal_index=2, fixture_id="d9hold-001"
            )
        ),
        "missing_capture_rejected": _fails(
            lambda: build_frozen_projection_artifact(
                captures=missing,
                calibration_manifest_sha256=CALIBRATION_SHA256,
                calibration_commitment_sha256=CALIBRATION_COMMITMENT,
                heldout_manifest_sha256=HELDOUT_SHA256,
                heldout_commitment_sha256=HELDOUT_COMMITMENT,
                schedule_commitment_sha256=SCHEDULE_COMMITMENT,
                output_dimension=8,
                fixture_only=True,
            )
        ),
        "duplicate_capture_rejected": _fails(
            lambda: build_frozen_projection_artifact(
                captures=duplicate,
                calibration_manifest_sha256=CALIBRATION_SHA256,
                calibration_commitment_sha256=CALIBRATION_COMMITMENT,
                heldout_manifest_sha256=HELDOUT_SHA256,
                heldout_commitment_sha256=HELDOUT_COMMITMENT,
                schedule_commitment_sha256=SCHEDULE_COMMITMENT,
                output_dimension=8,
                fixture_only=True,
            )
        ),
        "heldout_fixture_leak_rejected": _fails(
            lambda: build_frozen_projection_artifact(
                captures=heldout_leak,
                calibration_manifest_sha256=CALIBRATION_SHA256,
                calibration_commitment_sha256=CALIBRATION_COMMITMENT,
                heldout_manifest_sha256=HELDOUT_SHA256,
                heldout_commitment_sha256=HELDOUT_COMMITMENT,
                schedule_commitment_sha256=SCHEDULE_COMMITMENT,
                output_dimension=8,
                fixture_only=True,
            )
        ),
        "nonfinite_capture_rejected": _fails(
            lambda: build_frozen_projection_artifact(
                captures=nonfinite,
                calibration_manifest_sha256=CALIBRATION_SHA256,
                calibration_commitment_sha256=CALIBRATION_COMMITMENT,
                heldout_manifest_sha256=HELDOUT_SHA256,
                heldout_commitment_sha256=HELDOUT_COMMITMENT,
                schedule_commitment_sha256=SCHEDULE_COMMITMENT,
                output_dimension=8,
                fixture_only=True,
            )
        ),
        "tampered_parameter_rejected": _fails(
            lambda: audit_frozen_projection_artifact(
                tampered, expected_dimension=8, fixture_only=True
            )
        ),
        "captures_and_artifact_inputs_unchanged": captures == captures_before
        and artifact == artifact_before,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "artifact_digest_sha256": artifact["artifact_digest_sha256"],
        "parameter_digest_sha256": artifact["parameter_digest_sha256"],
        "real_projection_constructed": False,
    }


def verify_projection_contract_files(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract = _object(root / CONTRACT_RELATIVE_PATH, "projection contract")
    checks = validate_projection_contract(contract)
    source_checks = {
        CALIBRATION_RELATIVE_PATH: sha256_file(root / CALIBRATION_RELATIVE_PATH)
        == CALIBRATION_SHA256,
        HELDOUT_RELATIVE_PATH: sha256_file(root / HELDOUT_RELATIVE_PATH)
        == HELDOUT_SHA256,
        SCHEDULE_RELATIVE_PATH: sha256_file(root / SCHEDULE_RELATIVE_PATH)
        == SCHEDULE_SHA256,
    }
    if not all(source_checks.values()):
        raise RuntimeError("D9-C projection source manifest changed")
    fake = run_fake_projection_acceptance()
    if not fake["valid"]:
        raise RuntimeError("D9-C fake projection acceptance failed")
    return {
        "valid": True,
        "checks": checks,
        "source_checks": source_checks,
        "fake_acceptance": fake,
    }
