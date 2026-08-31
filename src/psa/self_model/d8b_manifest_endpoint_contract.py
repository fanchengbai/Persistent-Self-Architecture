from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d8_numerical_identifiability_design import (
    PAIR_TYPES,
    STRATA,
    excess_drift_from_distances,
    expand_fixtures,
    expand_schedule,
    validate_design,
)


CONTRACT_VERSION = "0.1-self-model-d8b-manifest-endpoint-contract"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8b_manifest_endpoint_contract.json"
)
DESIGN_RELATIVE_PATH = (
    "configs/preregistration/self_model_v0_1_d8_numerical_identifiability.draft.json"
)
FIXTURE_RELATIVE_PATH = "configs/development/self_model_v0_1_d8_fixture_manifest.json"
SCHEDULE_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8_counterbalanced_schedule.json"
)
DETERMINISM_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8_determinism_policy.json"
)
ENDPOINT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d8_excess_drift_endpoint.json"
)
DESIGN_SHA256 = "04f42ab2f42c7c0247165d4e42bdd0fdd83d986dc384c419767ff99e10c29bea"
DESIGN_REPORT_SHA256 = "3c59a1c65f0e9a101fbc1ede132c70fb7becf066bc487a04f217907e58cc8015"
FIXTURE_SHA256 = "d0b9c2e67eff48f2e9fd3cf9b5244cdcec165091bb4b079acecd8fd5b3323c2a"
SCHEDULE_SHA256 = "b1ecd2c027bd6aed8834147ec333c5a2ac1ba3df6550ee3ab221612e4805943f"
DETERMINISM_SHA256 = "81d98e24a61f29b9e5e5fed77612b2ffa56953c80282ee811192bbd7417dff83"
ENDPOINT_SHA256 = "96517ab9314ac2e8f68fd520bfb860626e39b1b69e3403f7033781a69983458c"
FIXTURE_COMMITMENT = "8976ac9f3f0b042e92ba146e58cc1df8c2d05e5a4635ccb0de558fb36161499e"
SCHEDULE_COMMITMENT = "a53cf5edc4f132c3fc773d63f7686f91f485b5c08c52f82beec983af36816465"
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D8-B deterministic manifest与fake endpoint contract纯离线实现；"
    "只允许将D8-A冻结的24个全新fixture、4个conditioning fixture、288个平衡配对、584次"
    "未来调用计划、launcher/运行期确定性清单、output-distance与excess-drift主要端点物化为"
    "确定性manifest、纯Python数据结构、完整性校验和合成验收；不得修改D8-A阈值或复用D7-C"
    "的cell、token、seed、claim和结果。本轮不探测installed source、不修改真实runner、不实现"
    "真实执行入口、不导入RWKV/Torch、不访问权重、不加载或执行模型；不授权D8-C真实执行、"
    "D7-C修复或重跑、D7-D/D7-E、projection、正式测试集、Self效果结论、Self Updater、"
    "raw-original或自动重跑。"
)
CLASSIFICATION = (
    "d8b_deterministic_manifests_and_fake_excess_drift_endpoint_verified_no_model"
)
NEXT_GATE = "remote_no_model_d8b_verification_then_separate_d8c_design_confirmation"
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    DETERMINISM_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    "docs/self_model_v0_1_d8_numerical_identifiability_design.md",
    "docs/self_model_v0_1_d8b_manifest_endpoint_contract.md",
    "scripts/verify_self_model_v0_1_d8b_manifest_endpoint_contract.py",
    "src/psa/self_model/d8_numerical_identifiability_design.py",
    "src/psa/self_model/d8b_manifest_endpoint_contract.py",
    "tests/test_self_model_d8_numerical_identifiability_design.py",
    "tests/test_self_model_d8b_manifest_endpoint_contract.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D8-B {label} must be an object")
    return value


def validate_contract_config(config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config.get("frozen_design", {})
    manifests = config.get("manifests", {})
    counts = config.get("counts", {})
    runtime = config.get("fake_runtime_contract", {})
    gates = config.get("gate_sequence", [])
    authority = config.get("authority", {})
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D8-B_deterministic_manifests_and_fake_endpoint_contract"
        and config.get("status") == "offline_contract_implemented_unrun_no_model"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "frozen_design_exact": frozen
        == {
            "path": DESIGN_RELATIVE_PATH,
            "sha256": DESIGN_SHA256,
            "historical_design_report_sha256": DESIGN_REPORT_SHA256,
            "research_question_changed": False,
            "thresholds_changed": False,
        },
        "four_manifest_paths_and_hashes_exact": manifests
        == {
            "fixture_path": FIXTURE_RELATIVE_PATH,
            "fixture_sha256": FIXTURE_SHA256,
            "schedule_path": SCHEDULE_RELATIVE_PATH,
            "schedule_sha256": SCHEDULE_SHA256,
            "determinism_path": DETERMINISM_RELATIVE_PATH,
            "determinism_sha256": DETERMINISM_SHA256,
            "endpoint_path": ENDPOINT_RELATIVE_PATH,
            "endpoint_sha256": ENDPOINT_SHA256,
        },
        "counts_exact": counts
        == {
            "conditioning_fixtures": 4,
            "scored_fixtures": 24,
            "pair_types": 4,
            "replicates_per_pair_type": 3,
            "scored_pairs": 288,
            "conditioning_forward_calls": 8,
            "scored_forward_calls": 576,
            "total_future_forward_calls": 584,
        },
        "pure_python_runtime_exact": runtime
        == {
            "fixture_kind": "d8b_pure_python_numeric_sequences_only",
            "tensor_or_model_object_allowed": False,
            "input_mutation_allowed": False,
            "complete_pair_ledger_required": True,
            "finite_values_required": True,
            "ninety_six_state_components_required": True,
            "duplicate_or_missing_pair_policy": "reject_before_decision",
            "output_claim": "endpoint_contract_only_not_real_route_or_self_effect_evidence",
        },
        "gates_ordered_and_d8c_closed": gates
        == [
            {"gate_id": "D8-A", "implemented": True, "model_execution": False},
            {"gate_id": "D8-B", "implemented": True, "model_execution": False},
            {
                "gate_id": "D8-C",
                "implemented": False,
                "authorized": False,
                "future_forward_calls": 584,
            },
        ],
        "d8b_authority_exact": authority.get("d8b_manifest_implementation_authorized")
        is True
        and authority.get("pure_python_endpoint_fixture_authorized") is True
        and authority.get("offline_integrity_report_authorized") is True,
        "model_execution_and_later_authority_closed": all(
            authority.get(field) is False
            for field in (
                "installed_source_probe_authorized",
                "real_runner_modification_authorized",
                "execution_entry_implementation_authorized",
                "rwkv_import_authorized",
                "torch_import_authorized",
                "weights_access_authorized",
                "model_load_authorized",
                "model_execution_authorized",
                "d8c_real_execution_authorized",
                "d7c_fix_authorized",
                "d7c_rerun_authorized",
                "d7d_authorized",
                "d7e_authorized",
                "projection_authorized",
                "formal_test_set_authorized",
                "self_effect_conclusion_authorized",
                "self_updater_authorized",
                "raw_original_route_authorized",
                "automatic_rerun_authorized",
            )
        ),
        "classification_and_next_gate_exact": config.get("required_classification")
        == CLASSIFICATION
        and config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D8-B contract config failed closed: " + ", ".join(failed))
    return checks


def validate_fixture_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    conditioning = manifest.get("conditioning_contract", {})
    scored = manifest.get("scored_contract", {})
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d8b-fixture-manifest"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D8-numerical-identifiability-fixtures-v01"
        and manifest.get("status") == "deterministic_fixture_manifest_frozen_unrun",
        "development_noncore_unrun": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("model_executed") is False,
        "design_namespace_and_seed_exact": manifest.get("design_path")
        == DESIGN_RELATIVE_PATH
        and manifest.get("design_sha256") == DESIGN_SHA256
        and manifest.get("namespace")
        == "psa-self-model-v0.1-d8a-numerical-identifiability-fixtures-20260831"
        and manifest.get("fixture_seed") == "d8a-fixtures-b0876e51-20260831",
        "token_contract_exact": manifest.get("token_derivation")
        == "sha256_seed_fixture_position_nonce_mod_58977_plus_1024_global_unique"
        and manifest.get("token_range") == [1024, 60000]
        and manifest.get("forbidden_d7c_token_ids") == [187, 931, 2764],
        "strata_exact": manifest.get("strata")
        == [
            {
                "stratum": name,
                "execution_path": path,
                "token_count": count,
                "full_output": full_output,
                "fixture_count": 6,
            }
            for name, path, count, full_output in STRATA
        ],
        "conditioning_exact": conditioning
        == {
            "fixture_count": 4,
            "one_per_stratum": True,
            "route_order": ["public", "wrapper_zero"],
            "forward_calls": 8,
            "outputs_scored": False,
        },
        "scored_exact": scored
        == {
            "fixture_count": 24,
            "six_per_stratum": True,
            "state_input": "fresh_clone_of_fixture_prebuilt_zero_state_for_every_call",
            "state_none_used": False,
        },
        "commitment_and_d7c_nonreuse_exact": manifest.get(
            "expanded_fixture_commitment_sha256"
        )
        == FIXTURE_COMMITMENT
        and manifest.get("d7c_cells_claim_seed_or_results_reused") is False,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D8-B fixture manifest failed closed: " + ", ".join(failed))
    return checks


def validate_schedule_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    latin = manifest.get("latin_contract", {})
    pair = manifest.get("pair_contract", {})
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d8b-counterbalanced-schedule"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D8-counterbalanced-pair-schedule-v01"
        and manifest.get("status")
        == "deterministic_schedule_manifest_frozen_unrun",
        "development_noncore_unrun": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("execution_authorized") is False
        and manifest.get("model_executed") is False,
        "design_fixture_namespace_seed_exact": manifest.get("design_sha256")
        == DESIGN_SHA256
        and manifest.get("fixture_manifest_path") == FIXTURE_RELATIVE_PATH
        and manifest.get("schedule_namespace")
        == "psa-self-model-v0.1-d8a-counterbalanced-schedule-20260831"
        and manifest.get("schedule_seed") == "d8a-schedule-0f409cf2-20260831",
        "pair_order_exact": manifest.get("pair_types") == list(PAIR_TYPES)
        and manifest.get("seeded_pair_order")
        == ["public_wrapper", "wrapper_wrapper", "public_public", "wrapper_public"],
        "latin_exact": latin
        == {
            "rotation": "four_by_four_latin_rotation_balanced_across_fixture_and_replicate",
            "replicates_per_pair_type_per_fixture": 3,
            "latin_positions": 4,
            "each_pair_type_per_position": 18,
            "cross_route_order_counterbalanced": True,
        },
        "pair_counts_exact": pair
        == {
            "calls_per_pair": 2,
            "same_source_state_clone_for_both_calls": True,
            "pairs_per_fixture": 12,
            "scored_pair_count": 288,
            "scored_forward_calls": 576,
        },
        "total_and_commitment_exact": manifest.get("conditioning_forward_calls") == 8
        and manifest.get("total_future_forward_calls") == 584
        and manifest.get("expanded_schedule_commitment_sha256")
        == SCHEDULE_COMMITMENT,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D8-B schedule manifest failed closed: " + ", ".join(failed))
    return checks


def validate_determinism_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    launcher = manifest.get("launcher_environment_before_python_start", {})
    runtime = manifest.get("runtime_before_model_load", {})
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d8b-determinism-policy"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D8-strict-determinism-policy-v01"
        and manifest.get("status") == "determinism_policy_frozen_unapplied",
        "design_policy_exact": manifest.get("development_only") is True
        and manifest.get("design_sha256") == DESIGN_SHA256
        and manifest.get("policy_namespace")
        == "psa-self-model-v0.1-d8a-determinism-policy-20260831"
        and manifest.get("policy_seed") == 28083101,
        "launcher_environment_exact": launcher
        == {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "28083101",
            "RWKV_DE_VERSION": "unset",
        },
        "runtime_flags_exact": runtime
        == {
            "python_random_seed": 28083101,
            "torch_manual_seed": 28083101,
            "torch_cuda_manual_seed_all": 28083101,
            "torch_use_deterministic_algorithms": True,
            "torch_deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
        "failure_policy_exact": manifest.get("failure_action_if_unavailable")
        == "stop_without_relaxing_policy_or_scoring_results",
        "policy_unapplied_and_no_model": manifest.get("policy_applied") is False
        and manifest.get("torch_imported") is False
        and manifest.get("rwkv_imported") is False
        and manifest.get("model_executed") is False,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D8-B determinism manifest failed closed: " + ", ".join(failed))
    return checks


def validate_endpoint_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    distance = manifest.get("distance", {})
    aggregation = manifest.get("pair_aggregation", {})
    primary = manifest.get("primary", {})
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d8b-excess-drift-endpoint"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D8-conservative-excess-drift-endpoint-v01"
        and manifest.get("status") == "endpoint_manifest_frozen_unrun",
        "design_unrun_exact": manifest.get("development_only") is True
        and manifest.get("design_sha256") == DESIGN_SHA256
        and manifest.get("model_executed") is False,
        "distance_exact": distance
        == {
            "tensor": "max_abs_difference_divided_by_max_of_both_max_abs_values_and_1e_minus_12",
            "state": "maximum_tensor_distance_across_all_96_compatible_state_components",
            "output": "maximum_of_logits_distance_and_state_distance",
            "nonfinite_missing_or_incompatible_action": "invalidate_and_stop_without_rerun",
        },
        "aggregation_exact": aggregation
        == {
            "within_route_envelope": "max(distance_public_public,distance_wrapper_wrapper)",
            "symmetric_cross_route_floor": "min(distance_public_wrapper,distance_wrapper_public)",
            "replicate_excess_drift": "symmetric_cross_route_floor_minus_within_route_envelope",
            "fixture_excess_drift": "median_of_three_replicate_excess_drift_values",
        },
        "primary_thresholds_exact": primary
        == {
            "estimand": "mean_fixture_excess_drift_across_24_fixtures",
            "test": "one_sided_cluster_bootstrap_99_percent_lower_bound_greater_than_zero",
            "bootstrap_seed": 28083102,
            "bootstrap_resamples": 100000,
            "minimum_positive_fixtures": 21,
            "minimum_positive_fixtures_per_stratum": 5,
            "positive_decision": "route_specific_excess_drift_detected_non_self_engineering_evidence_only",
            "otherwise_decision": "inconclusive_no_route_equivalence_claim",
        },
        "secondary_and_claims_closed": manifest.get("secondary_descriptive_only")
        == [
            "logits_excess_drift",
            "state_excess_drift",
            "absolute_public_wrapper_vs_wrapper_public_order_interaction",
            "public_vs_wrapper_within_route_repeatability_asymmetry",
        ]
        and manifest.get("thresholds_changed_from_d8a") is False
        and manifest.get("route_equivalence_claim_allowed") is False
        and manifest.get("self_effect_conclusion_allowed") is False,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D8-B endpoint manifest failed closed: " + ", ".join(failed))
    return checks


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, bool):
        raise TypeError("D8-B fake numeric sequence rejects booleans")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("D8-B fake numeric sequence must be finite")
        return [numeric]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError("D8-B fake numeric sequence must not be empty")
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    raise TypeError("D8-B accepts only pure Python numeric sequences")


def _numeric_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("D8-B fake numeric sequence rejects booleans")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("D8-B fake numeric sequence must be finite")
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError("D8-B fake numeric sequence must not be empty")
        child_shapes = [_numeric_shape(item) for item in value]
        if len(set(child_shapes)) != 1:
            raise ValueError("D8-B fake numeric sequence must not be ragged")
        return (len(value), *child_shapes[0])
    raise TypeError("D8-B accepts only pure Python numeric sequences")


def tensor_distance(left: Any, right: Any) -> float:
    if _numeric_shape(left) != _numeric_shape(right):
        raise ValueError("D8-B fake tensor shapes are incompatible")
    left_values = _flatten_numeric(left)
    right_values = _flatten_numeric(right)
    numerator = max(abs(a - b) for a, b in zip(left_values, right_values))
    denominator = max(
        max(abs(value) for value in left_values),
        max(abs(value) for value in right_values),
        1e-12,
    )
    return numerator / denominator


@dataclass(frozen=True)
class FakeForwardOutput:
    logits: tuple[float, ...]
    state: tuple[tuple[float, ...], ...]


def output_distance(left: FakeForwardOutput, right: FakeForwardOutput) -> dict[str, Any]:
    if type(left) is not FakeForwardOutput or type(right) is not FakeForwardOutput:
        raise TypeError("D8-B output distance requires exact fake output objects")
    if len(left.state) != len(right.state) or len(left.state) != 96:
        raise ValueError("D8-B output distance requires 96 compatible state components")
    logits = tensor_distance(left.logits, right.logits)
    state_components = [
        tensor_distance(left_component, right_component)
        for left_component, right_component in zip(left.state, right.state)
    ]
    state = max(state_components)
    return {
        "logits_distance": logits,
        "state_distance": state,
        "output_distance": max(logits, state),
        "max_state_component_index": state_components.index(state),
        "state_component_count": len(state_components),
    }


def validate_pair_ledger(
    schedule: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_blocks = schedule["pair_blocks"]
    expected = {block["pair_block_id"]: block for block in expected_blocks}
    if len(expected) != 288:
        raise RuntimeError("D8-B expanded schedule must contain 288 unique pair blocks")
    if len(ledger) != 288:
        raise ValueError("D8-B pair ledger must contain exactly 288 records")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in ledger:
        record_id = record.get("pair_block_id")
        if not isinstance(record_id, str) or record_id in seen or record_id not in expected:
            raise ValueError("D8-B pair ledger has an unknown or duplicate pair block")
        seen.add(record_id)
        expected_block = expected[record_id]
        if record.get("pair_type") != expected_block["pair_type"]:
            raise ValueError("D8-B pair ledger pair type changed")
        distance = float(record.get("output_distance"))
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("D8-B pair ledger distance must be finite and nonnegative")
        normalized.append(
            {
                "pair_block_id": record_id,
                "fixture_id": expected_block["fixture_id"],
                "stratum": expected_block["stratum"],
                "replicate": expected_block["replicate"],
                "pair_type": expected_block["pair_type"],
                "output_distance": distance,
            }
        )
    if seen != set(expected):
        raise ValueError("D8-B pair ledger is incomplete")
    return {"valid": True, "records": normalized, "record_count": len(normalized)}


def aggregate_fixture_excess(
    schedule: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated = validate_pair_ledger(schedule, ledger)
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    metadata: dict[str, str] = {}
    for record in validated["records"]:
        fixture_id = record["fixture_id"]
        metadata[fixture_id] = record["stratum"]
        key = (fixture_id, record["replicate"])
        distances = grouped.setdefault(key, {})
        if record["pair_type"] in distances:
            raise ValueError("D8-B fixture replicate contains a duplicate pair type")
        distances[record["pair_type"]] = record["output_distance"]
    replicate_results: dict[str, list[dict[str, float]]] = {}
    for (fixture_id, replicate), distances in grouped.items():
        result = excess_drift_from_distances(distances)
        replicate_results.setdefault(fixture_id, []).append(
            {"replicate": float(replicate), **result}
        )
    fixtures: list[dict[str, Any]] = []
    for fixture_id in sorted(replicate_results):
        values = sorted(replicate_results[fixture_id], key=lambda item: item["replicate"])
        if [int(item["replicate"]) for item in values] != [1, 2, 3]:
            raise ValueError("D8-B fixture must contain replicates 1, 2, and 3")
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "stratum": metadata[fixture_id],
                "replicate_excess_drift": [item["excess_drift"] for item in values],
                "fixture_excess_drift": statistics.median(
                    item["excess_drift"] for item in values
                ),
            }
        )
    if len(fixtures) != 24:
        raise ValueError("D8-B aggregation requires exactly 24 fixtures")
    return fixtures


def _bootstrap_lower_bound(values: Sequence[float], *, seed: int, resamples: int) -> float:
    if len(values) != 24 or resamples != 100000:
        raise ValueError("D8-B bootstrap inputs do not match the frozen endpoint")
    if len(set(values)) == 1:
        return float(values[0])
    generator = random.Random(seed)
    count = len(values)
    means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    means.sort()
    return means[int(0.01 * resamples)]


def decide_excess_drift(
    fixture_results: Sequence[Mapping[str, Any]], endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    if len(fixture_results) != 24:
        raise ValueError("D8-B decision requires 24 fixture results")
    primary = endpoint["primary"]
    values = [float(item["fixture_excess_drift"]) for item in fixture_results]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("D8-B fixture excess values must be finite")
    stratum_positive = {
        stratum: sum(
            item["stratum"] == stratum and float(item["fixture_excess_drift"]) > 0.0
            for item in fixture_results
        )
        for stratum, _, _, _ in STRATA
    }
    lower = _bootstrap_lower_bound(
        values,
        seed=primary["bootstrap_seed"],
        resamples=primary["bootstrap_resamples"],
    )
    positive_count = sum(value > 0.0 for value in values)
    positive = (
        lower > 0.0
        and positive_count >= primary["minimum_positive_fixtures"]
        and all(
            count >= primary["minimum_positive_fixtures_per_stratum"]
            for count in stratum_positive.values()
        )
    )
    return {
        "decision": (
            primary["positive_decision"]
            if positive
            else primary["otherwise_decision"]
        ),
        "positive": positive,
        "mean_fixture_excess_drift": statistics.mean(values),
        "bootstrap_99_percent_lower_bound": lower,
        "positive_fixture_count": positive_count,
        "positive_by_stratum": stratum_positive,
        "route_equivalence_claim": False,
        "self_effect_conclusion": False,
    }


def _fake_ledger(schedule: Mapping[str, Any], scenario: str) -> list[dict[str, Any]]:
    distances_by_scenario = {
        "route_specific_excess": {
            "public_public": 0.001,
            "wrapper_wrapper": 0.0015,
            "public_wrapper": 0.02,
            "wrapper_public": 0.018,
        },
        "one_order_only": {
            "public_public": 0.001,
            "wrapper_wrapper": 0.001,
            "public_wrapper": 0.02,
            "wrapper_public": 0.001,
        },
        "shared_background_repeatability": {
            "public_public": 0.01,
            "wrapper_wrapper": 0.01,
            "public_wrapper": 0.01,
            "wrapper_public": 0.01,
        },
    }
    if scenario not in distances_by_scenario:
        raise ValueError("D8-B fake scenario is not frozen")
    distances = distances_by_scenario[scenario]
    return [
        {
            "pair_block_id": block["pair_block_id"],
            "pair_type": block["pair_type"],
            "output_distance": distances[block["pair_type"]],
        }
        for block in schedule["pair_blocks"]
    ]


def _fake_output(*, changed_logits: bool = False, changed_state: bool = False) -> FakeForwardOutput:
    logits = [1.0, 2.0, 4.0]
    state = [[float(index + 1), float(index + 2)] for index in range(96)]
    if changed_logits:
        logits[-1] += 1.0
    if changed_state:
        state[94][-1] += 2.0
    return FakeForwardOutput(
        logits=tuple(logits),
        state=tuple(tuple(component) for component in state),
    )


def run_fake_acceptance(
    *, design: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    fixtures = expand_fixtures(design)
    schedule = expand_schedule(design, fixtures)
    base = _fake_output()
    logits_changed = _fake_output(changed_logits=True)
    state_changed = _fake_output(changed_state=True)
    exact = output_distance(base, base)
    logits_report = output_distance(base, logits_changed)
    state_report = output_distance(base, state_changed)
    positive_source = _fake_ledger(schedule, "route_specific_excess")
    positive_snapshot = copy.deepcopy(positive_source)
    positive_fixtures = aggregate_fixture_excess(schedule, positive_source)
    positive_decision = decide_excess_drift(positive_fixtures, endpoint)
    order_fixtures = aggregate_fixture_excess(
        schedule, _fake_ledger(schedule, "one_order_only")
    )
    order_decision = decide_excess_drift(order_fixtures, endpoint)
    shared_fixtures = aggregate_fixture_excess(
        schedule, _fake_ledger(schedule, "shared_background_repeatability")
    )
    shared_decision = decide_excess_drift(shared_fixtures, endpoint)
    rejection_checks: dict[str, bool] = {}
    missing = positive_source[:-1]
    try:
        validate_pair_ledger(schedule, missing)
        rejection_checks["missing_pair_rejected"] = False
    except ValueError:
        rejection_checks["missing_pair_rejected"] = True
    duplicate = copy.deepcopy(positive_source)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    try:
        validate_pair_ledger(schedule, duplicate)
        rejection_checks["duplicate_pair_rejected"] = False
    except ValueError:
        rejection_checks["duplicate_pair_rejected"] = True
    nonfinite = copy.deepcopy(positive_source)
    nonfinite[0]["output_distance"] = float("nan")
    try:
        validate_pair_ledger(schedule, nonfinite)
        rejection_checks["nonfinite_pair_rejected"] = False
    except ValueError:
        rejection_checks["nonfinite_pair_rejected"] = True
    try:
        output_distance(
            base,
            FakeForwardOutput(logits=base.logits, state=base.state[:-1]),
        )
        rejection_checks["wrong_state_count_rejected"] = False
    except ValueError:
        rejection_checks["wrong_state_count_rejected"] = True
    checks = {
        "fixture_and_schedule_commitments_exact": fixtures[
            "fixture_commitment_sha256"
        ]
        == FIXTURE_COMMITMENT
        and schedule["schedule_commitment_sha256"] == SCHEDULE_COMMITMENT,
        "expanded_counts_exact": len(fixtures["scored_fixtures"]) == 24
        and len(fixtures["conditioning_fixtures"]) == 4
        and len(schedule["pair_blocks"]) == 288
        and len(schedule["conditioning_calls"]) == 8,
        "exact_output_distance_is_zero": exact["output_distance"] == 0.0,
        "logits_change_selects_logits_distance": logits_report["output_distance"]
        == logits_report["logits_distance"]
        and logits_report["logits_distance"] > 0.0,
        "state_change_selects_state_distance_and_component": state_report[
            "output_distance"
        ]
        == state_report["state_distance"]
        and state_report["max_state_component_index"] == 94,
        "positive_route_specific_scenario_detected": positive_decision["positive"]
        and positive_decision["positive_fixture_count"] == 24,
        "one_order_only_scenario_is_inconclusive": not order_decision["positive"]
        and order_decision["decision"] == "inconclusive_no_route_equivalence_claim",
        "shared_repeatability_scenario_is_inconclusive": not shared_decision["positive"]
        and shared_decision["decision"]
        == "inconclusive_no_route_equivalence_claim",
        "no_equivalence_or_self_claim_created": not positive_decision[
            "route_equivalence_claim"
        ]
        and not positive_decision["self_effect_conclusion"]
        and not order_decision["route_equivalence_claim"]
        and not shared_decision["self_effect_conclusion"],
        "fake_ledger_input_unchanged": positive_source == positive_snapshot,
        "all_fail_closed_rejections_pass": all(rejection_checks.values()),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "rejection_checks": rejection_checks,
        "distance_examples": {
            "exact": exact,
            "logits_changed": logits_report,
            "state_changed": state_report,
        },
        "scenario_decisions": {
            "route_specific_excess": positive_decision,
            "one_order_only": order_decision,
            "shared_background_repeatability": shared_decision,
        },
        "counts": {
            "conditioning_fixtures": len(fixtures["conditioning_fixtures"]),
            "scored_fixtures": len(fixtures["scored_fixtures"]),
            "pair_blocks": len(schedule["pair_blocks"]),
            "future_forward_calls": len(schedule["conditioning_calls"])
            + len(schedule["pair_blocks"]) * 2,
        },
    }


def build_contract_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    expected_config = (root / CONFIG_RELATIVE_PATH).resolve()
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected_config:
        raise PermissionError("D8-B contract config path is not frozen")
    config = _object(supplied, "contract config")
    design = _object(root / DESIGN_RELATIVE_PATH, "D8-A design")
    fixture = _object(root / FIXTURE_RELATIVE_PATH, "fixture manifest")
    schedule = _object(root / SCHEDULE_RELATIVE_PATH, "schedule manifest")
    determinism = _object(root / DETERMINISM_RELATIVE_PATH, "determinism manifest")
    endpoint = _object(root / ENDPOINT_RELATIVE_PATH, "endpoint manifest")
    config_checks = validate_contract_config(config)
    design_checks = validate_design(design)
    fixture_checks = validate_fixture_manifest(fixture)
    schedule_checks = validate_schedule_manifest(schedule)
    determinism_checks = validate_determinism_manifest(determinism)
    endpoint_checks = validate_endpoint_manifest(endpoint)
    manifest_hash_checks = {
        FIXTURE_RELATIVE_PATH: sha256_file(root / FIXTURE_RELATIVE_PATH) == FIXTURE_SHA256,
        SCHEDULE_RELATIVE_PATH: sha256_file(root / SCHEDULE_RELATIVE_PATH)
        == SCHEDULE_SHA256,
        DETERMINISM_RELATIVE_PATH: sha256_file(root / DETERMINISM_RELATIVE_PATH)
        == DETERMINISM_SHA256,
        ENDPOINT_RELATIVE_PATH: sha256_file(root / ENDPOINT_RELATIVE_PATH)
        == ENDPOINT_SHA256,
    }
    acceptance = run_fake_acceptance(design=design, endpoint=endpoint)
    checks = {
        "contract_config_valid": all(config_checks.values()),
        "d8a_design_still_valid": all(design_checks.values()),
        "fixture_manifest_valid": all(fixture_checks.values()),
        "schedule_manifest_valid": all(schedule_checks.values()),
        "determinism_manifest_valid": all(determinism_checks.values()),
        "endpoint_manifest_valid": all(endpoint_checks.values()),
        "manifest_hashes_valid": all(manifest_hash_checks.values()),
        "fake_acceptance_valid": acceptance["valid"],
        "d8a_thresholds_unchanged": config["frozen_design"]["thresholds_changed"]
        is False
        and endpoint["thresholds_changed_from_d8a"] is False,
        "d7c_nonreuse_preserved": fixture["d7c_cells_claim_seed_or_results_reused"]
        is False,
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D8-B contract verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": CONTRACT_VERSION,
        "status": "d8b_deterministic_manifests_and_fake_endpoint_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "design_checks": design_checks,
        "fixture_manifest_checks": fixture_checks,
        "schedule_manifest_checks": schedule_checks,
        "determinism_manifest_checks": determinism_checks,
        "endpoint_manifest_checks": endpoint_checks,
        "manifest_hash_checks": manifest_hash_checks,
        "fake_acceptance": acceptance,
        "next_gate": NEXT_GATE,
        "safety": {
            "installed_source_probed": False,
            "real_runner_modified": False,
            "execution_entry_implemented": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d8c_real_execution_authorized": False,
            "d7c_fixed": False,
            "d7c_rerun": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
            "projection_constructed": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "raw_original_route_used": False,
            "automatic_rerun_authorized": False,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
