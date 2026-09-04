"""Pure-offline diagnostics for the consumed D9-D causal-isolation run."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d9a_within_wrapper_causal_isolation import evaluate_candidate
from psa.self_model.d9c_projection_contract import audit_frozen_projection_artifact


DIAGNOSTIC_VERSION = "0.1-self-model-d9d-offline-causal-diagnostic"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9d_offline_causal_diagnostic.json"
)
CLASSIFICATION = (
    "d9d_mechanism_positive_projection_causal_specificity_failed_"
    "calibration_numeric_stability_not_identifiable"
)
NEXT_GATE = (
    "remote_offline_diagnostic_on_consumed_d9d_artifacts_then_separate_route_"
    "confirmation"
)
CONFIRMATION_TEXT = (
    "确认进入 Self Model v0.1 D9-D失败纯离线projection/ledger因果结构诊断与研究路线审查；"
    "仅使用现有authorization、claim、真实projection artifact、raw ledger、report、integrity、"
    "冻结manifest和源码，分析16个identity×goal基础组合及四代码轮换的active-zero、"
    "true-random、mask、swap分布，审计calibration replicate稳定性、projection分支几何与字段"
    "可分离性，并判断后续是否存在科学独立且不构成D9-D重跑的新研究路线；不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型、不修改真实runner或冻结阈值、不实现或授权修复后重跑，"
    "也不授权D7-D/D7-E、正式测试集、Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
CONTRASTS = (
    "active_true",
    "mask_identity",
    "mask_goal",
    "swap_identity",
    "swap_goal",
    "matched_random",
    "synthetic_active",
)
CODE_LABELS = ("A", "B", "C", "D")
MARGIN_FIELDS = {
    "target_alignment_margin",
    "identity_margin",
    "goal_margin",
    "identity_swap_advantage",
    "goal_swap_advantage",
}
DIAGNOSTIC_SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "scripts/run_self_model_v0_1_d9d_offline_causal_diagnostic.py",
    "scripts/verify_self_model_v0_1_d9d_offline_causal_diagnostic.py",
    "src/psa/self_model/d9d_offline_causal_diagnostic.py",
    "tests/test_self_model_d9d_offline_causal_diagnostic.py",
    "docs/self_model_v0_1_d9d_offline_causal_diagnostic.md",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D9-D offline {label} must be an object")
    return value


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"D9-D offline ledger line {line_number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"D9-D offline ledger line {line_number} is not an object")
        records.append(value)
    return records


def _hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"D9-D offline {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"D9-D offline {label} is nonfinite")
    return result


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    return result


def _same_float_mapping(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        math.isclose(
            _number(left[key], f"left {key}"),
            _number(right[key], f"right {key}"),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key in left
    )


def _target_codes(identity: int, goal: int, rotation: int) -> dict[str, str]:
    def code(identity_value: int, goal_value: int) -> str:
        return CODE_LABELS[
            ((identity_value * 3 + goal_value) % 4 + rotation) % 4
        ]

    return {
        "true": code(identity, goal),
        "identity_swap": code((identity + 1) % 4, goal),
        "goal_swap": code(identity, (goal + 1) % 4),
    }


def _margins(scores: Mapping[str, Any], codes: Mapping[str, str]) -> dict[str, float]:
    if set(scores) != set(CODE_LABELS):
        raise ValueError("D9-D offline choice-score labels changed")
    values = {code: _number(scores[code], f"choice score {code}") for code in CODE_LABELS}
    true = codes["true"]
    return {
        "target_alignment_margin": values[true]
        - max(value for code, value in values.items() if code != true),
        "identity_margin": values[true] - values[codes["identity_swap"]],
        "goal_margin": values[true] - values[codes["goal_swap"]],
        "identity_swap_advantage": values[codes["identity_swap"]] - values[true],
        "goal_swap_advantage": values[codes["goal_swap"]] - values[true],
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [_number(value, "summary value") for value in values]
    if not finite:
        raise ValueError("D9-D offline cannot summarize an empty sequence")
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "population_stdev": statistics.pstdev(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "positive_count": sum(value > 0.0 for value in finite),
        "negative_count": sum(value < 0.0 for value in finite),
        "zero_count": sum(value == 0.0 for value in finite),
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    evidence = config.get("evidence", {})
    frozen = config.get("frozen_inputs", {})
    analysis = config.get("analysis_contract", {})
    route = config.get("route_review", {})
    safety = config.get("safety", {})
    checks = {
        "identity_exact": config.get("diagnostic_version") == DIAGNOSTIC_VERSION
        and config.get("diagnostic_id")
        == "Self-Model-v0.1-D9-D-failure-offline-projection-ledger-causal-diagnostic-v01"
        and config.get("status") == "authorized_offline_implementation_no_model",
        "confirmation_exact": config.get("implementation_confirmation_text")
        == CONFIRMATION_TEXT,
        "evidence_paths_unique": len(
            {
                evidence.get("authorization_path"),
                evidence.get("claim_path"),
                evidence.get("projection_path"),
                evidence.get("raw_ledger_path"),
                evidence.get("report_path"),
                evidence.get("integrity_path"),
            }
        )
        == 6,
        "evidence_digests_complete": all(
            _hex(evidence.get(field))
            for field in (
                "authorization_file_sha256",
                "claim_file_sha256",
                "projection_file_sha256",
                "raw_ledger_file_sha256",
                "report_file_sha256",
                "authorization_digest_sha256",
                "projection_artifact_digest_sha256",
                "projection_parameter_digest_sha256",
                "report_digest_sha256",
                "integrity_digest_sha256",
                "installed_source_sha256",
            )
        )
        and isinstance(evidence.get("git_commit"), str)
        and len(evidence["git_commit"]) == 40,
        "frozen_inputs_complete": all(
            isinstance(frozen.get(path_field), str)
            and _hex(frozen.get(digest_field))
            for path_field, digest_field in (
                ("calibration_manifest_path", "calibration_manifest_sha256"),
                ("heldout_manifest_path", "heldout_manifest_sha256"),
                ("schedule_manifest_path", "schedule_manifest_sha256"),
                ("endpoint_manifest_path", "endpoint_manifest_sha256"),
                ("projection_contract_path", "projection_contract_sha256"),
                ("real_entry_source_path", "real_entry_source_sha256"),
            )
        ),
        "analysis_counts_exact": analysis.get("calibration_records") == 32
        and analysis.get("calibration_cells") == 16
        and analysis.get("replicates_per_cell") == 2
        and analysis.get("heldout_fixtures") == 64
        and analysis.get("base_cases") == 16
        and analysis.get("rotations_per_base_case") == 4
        and analysis.get("contrasts") == list(CONTRASTS)
        and analysis.get("pair_records") == 448
        and analysis.get("ledger_records") == 480
        and analysis.get("forward_calls") == 928,
        "projection_and_identifiability_exact": analysis.get("projection_dimension")
        == 2560
        and analysis.get("projection_groups")
        == ["identity_weights", "goal_weights"]
        and analysis.get("projection_vectors_per_group") == 4
        and analysis.get("projection_geometry_required") is True
        and analysis.get("capture_vectors_present_in_ledger") is False
        and analysis.get("replicate_numeric_distance_identifiable") is False
        and analysis.get("capture_digest_identity_only") is True,
        "route_review_is_independent_candidate_only": route.get("d9d_rerun_allowed")
        is False
        and route.get("posthoc_gain_layer_or_threshold_search_on_d9d_allowed")
        is False
        and route.get(
            "reuse_d9d_fixture_claim_or_result_as_new_experiment_data_allowed"
        )
        is False
        and route.get("candidate_status")
        == "preregistration_candidate_only_not_implemented_or_authorized"
        and route.get("candidate_must_use_new_tokens_fixtures_seeds_claim_and_output")
        is True
        and route.get(
            "candidate_must_measure_cross_replicate_and_heldout_field_decodability_before_intervention"
        )
        is True,
        "classification_and_next_gate_exact": config.get("classification")
        == CLASSIFICATION
        and config.get("next_gate") == NEXT_GATE,
        "all_execution_and_claim_authority_closed": set(safety)
        == {
            "rwkv_imported",
            "torch_imported",
            "weights_accessed",
            "model_loaded",
            "model_executed",
            "real_runner_modified",
            "frozen_thresholds_modified",
            "d9d_rerun",
            "new_real_experiment_authorized",
            "formal_test_set_used",
            "self_effect_conclusion_made",
            "self_updater_used",
            "raw_original_route_used",
            "automatic_rerun_authorized",
        }
        and all(value is False for value in safety.values()),
    }
    if not all(checks.values()):
        raise PermissionError(
            "D9-D offline diagnostic config failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def validate_source_locks(root: Path, config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config["frozen_inputs"]
    checks: dict[str, bool] = {}
    for prefix in (
        "calibration_manifest",
        "heldout_manifest",
        "schedule_manifest",
        "endpoint_manifest",
        "projection_contract",
        "real_entry_source",
    ):
        relative = frozen[f"{prefix}_path"]
        checks[relative] = sha256_file(root / relative) == frozen[f"{prefix}_sha256"]
    if not all(checks.values()):
        raise RuntimeError(
            "D9-D offline frozen source differs: "
            + ", ".join(path for path, valid in checks.items() if not valid)
        )
    return checks


def validate_evidence_bundle(
    *, config: Mapping[str, Any], authorization: Mapping[str, Any],
    claim: Mapping[str, Any], projection: Mapping[str, Any],
    report: Mapping[str, Any], integrity: Mapping[str, Any],
) -> dict[str, bool]:
    expected = config["evidence"]
    projection_audit = audit_frozen_projection_artifact(
        projection, expected_dimension=2560, fixture_only=False
    )
    checks = {
        "authorization_internal_digest_valid": authorization.get(
            "authorization_digest_sha256"
        )
        == sha256_json(_without(authorization, "authorization_digest_sha256"))
        == expected["authorization_digest_sha256"],
        "authorization_scope_and_commit_exact": authorization.get("authorized") is True
        and authorization.get("single_use") is True
        and authorization.get("model_forward_calls") == 928
        and authorization.get("git_commit") == expected["git_commit"],
        "authorization_all_excluded_authority_false": all(
            authorization.get(field) is False
            for field in (
                "d9d_rerun_authorized",
                "d8c_rerun_authorized",
                "historical_rerun_authorized",
                "d7d_authorized",
                "d7e_authorized",
                "formal_test_set_authorized",
                "self_effect_conclusion_authorized",
                "self_updater_authorized",
                "raw_original_route_authorized",
                "automatic_rerun_authorized",
            )
        ),
        "claim_consumed_and_bound": claim.get("status")
        == "d9d_single_use_joint_execution_claim_consumed"
        and claim.get("single_use") is True
        and claim.get("git_commit") == expected["git_commit"]
        and claim.get("authorization_sha256")
        == expected["authorization_file_sha256"]
        and claim.get("entry_static_report_sha256")
        == "e9ad2903a5bf703b0eebcc61cdc8d5afb87f27b7838443a406df84e77fc5cc09"
        and claim.get("installed_source_sha256")
        == expected["installed_source_sha256"]
        and claim.get("calibration_forward_calls") == 32
        and claim.get("heldout_forward_calls") == 896
        and claim.get("total_forward_calls") == 928
        and claim.get("d9d_rerun_authorized") is False
        and claim.get("automatic_rerun_authorized") is False,
        "projection_audit_valid": projection_audit["valid"] is True
        and projection_audit["artifact_digest_sha256"]
        == expected["projection_artifact_digest_sha256"]
        and projection_audit["parameter_digest_sha256"]
        == expected["projection_parameter_digest_sha256"],
        "report_internal_digest_valid": report.get("report_digest_sha256")
        == sha256_json(_without(report, "report_digest_sha256"))
        == expected["report_digest_sha256"],
        "report_completed_without_self_claim": report.get("status")
        == "d9d_real_within_wrapper_causal_isolation_completed_claim_consumed"
        and report.get("valid") is True
        and report.get("classification")
        == "revise_or_stop_without_self_effect_claim_or_rerun"
        and report.get("self_effect_conclusion") is False
        and report.get("git_commit") == expected["git_commit"]
        and report.get("execution_claim_sha256")
        == expected["claim_file_sha256"]
        and report.get("authorization_digest_sha256")
        == expected["authorization_digest_sha256"],
        "report_counts_exact": report.get("counts")
        == {
            "calibration_forward_calls": 32,
            "heldout_forward_calls": 896,
            "heldout_pair_records": 448,
            "ledger_records": 480,
            "total_forward_calls": 928,
        },
        "report_installed_source_bound": report.get("installed_source", {}).get(
            "sha256"
        )
        == expected["installed_source_sha256"]
        and report.get("installed_source", {}).get("version") == "0.8.32",
        "report_projection_bound": report.get("projection", {}).get(
            "artifact_digest_sha256"
        )
        == expected["projection_artifact_digest_sha256"]
        and report.get("projection", {}).get("parameter_digest_sha256")
        == expected["projection_parameter_digest_sha256"]
        and report.get("projection", {}).get("calibration_only") is True
        and report.get("projection", {}).get("frozen_before_heldout_access") is True,
        "integrity_internal_digest_valid": integrity.get("integrity_digest_sha256")
        == sha256_json(_without(integrity, "integrity_digest_sha256"))
        == expected["integrity_digest_sha256"],
        "integrity_binds_all_primary_artifacts": integrity.get("status")
        == "d9d_real_artifact_integrity_complete"
        and integrity.get("execution_claim_sha256")
        == expected["claim_file_sha256"]
        and integrity.get("raw_ledger_sha256")
        == expected["raw_ledger_file_sha256"]
        and integrity.get("projection_artifact_sha256")
        == expected["projection_file_sha256"]
        and integrity.get("report_sha256") == expected["report_file_sha256"]
        and integrity.get("d9d_rerun_authorized") is False
        and integrity.get("automatic_rerun_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-D offline evidence bundle failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def validate_ledger_structure(
    *, records: Sequence[Mapping[str, Any]],
    calibration_manifest: Mapping[str, Any], heldout_manifest: Mapping[str, Any],
    schedule: Mapping[str, Any], projection_digest: str,
) -> dict[str, bool]:
    if len(records) != 480:
        raise ValueError("D9-D offline ledger must contain exactly 480 records")
    calibration_records = records[:32]
    pair_records = records[32:]
    calibration_fixtures = calibration_manifest.get("fixtures", [])
    heldout_fixtures = {
        item["fixture_id"]: item for item in heldout_manifest.get("fixtures", [])
    }
    blocks = schedule.get("heldout_pair_blocks", [])
    calibration_ok = len(calibration_fixtures) == 32
    capture_ids: list[str] = []
    for record, fixture in zip(calibration_records, calibration_fixtures):
        capture_ids.append(str(record.get("call_id")))
        calibration_ok = calibration_ok and (
            record.get("record_type") == "calibration_capture"
            and record.get("call_id") == f"{fixture['fixture_id']}-capture"
            and record.get("fixture_id") == fixture["fixture_id"]
            and record.get("phase") == "calibration"
            and record.get("route") == "persistent_wrapper_capture"
            and _hex(record.get("capture_sha256"))
            and record.get("callback_invocations") == 32
            and record.get("target_layer_applications") == 1
            and record.get("heldout_scored") is False
        )
    pair_ok = len(blocks) == 448 and len(heldout_fixtures) == 64
    pair_ids: list[str] = []
    for record, block in zip(pair_records, blocks):
        pair_ids.append(str(record.get("pair_block_id")))
        fixture = heldout_fixtures.get(str(block.get("fixture_id")))
        if fixture is None:
            pair_ok = False
            continue
        identity = int(block["identity_index"])
        goal = int(block["goal_index"])
        rotation = int(block["code_rotation"])
        codes = _target_codes(identity, goal, rotation)
        observations = record.get("observations", [])
        pair_ok = pair_ok and all(
            record.get(field) == block.get(field)
            for field in (
                "pair_block_id",
                "fixture_id",
                "base_case_id",
                "identity_index",
                "goal_index",
                "code_rotation",
                "contrast",
                "latin_position",
                "pair_order",
                "condition_order",
                "route",
                "source_state_contract",
            )
        )
        pair_ok = pair_ok and (
            record.get("record_type") == "heldout_pair"
            and record.get("route") == "persistent_wrapper"
            and record.get("projection_artifact_sha256") == projection_digest
            and isinstance(observations, list)
            and len(observations) == 2
            and [item.get("condition") for item in observations]
            == block.get("condition_order")
        )
        by_condition: dict[str, Mapping[str, Any]] = {}
        for observation in observations if isinstance(observations, list) else []:
            condition = str(observation.get("condition"))
            margins = observation.get("margins", {})
            calculated = _margins(observation.get("choice_scores", {}), codes)
            pair_ok = pair_ok and (
                set(margins) == MARGIN_FIELDS
                and _same_float_mapping(margins, calculated)
                and _hex(observation.get("logits_sha256"))
                and observation.get("state_component_count") == 96
                and observation.get("projection_artifact_sha256")
                == projection_digest
            )
            by_condition[condition] = observation
        contrast = str(block["contrast"])
        if "wrapper_zero" not in by_condition or contrast not in by_condition:
            pair_ok = False
        else:
            pair_ok = pair_ok and _same_float_mapping(
                record.get("zero_margins", {}),
                by_condition["wrapper_zero"].get("margins", {}),
            )
            pair_ok = pair_ok and _same_float_mapping(
                record.get("condition_margins", {}),
                by_condition[contrast].get("margins", {}),
            )
            if contrast == "synthetic_active":
                pair_ok = pair_ok and (
                    record.get("synthetic_output_changed")
                    is (
                        by_condition[contrast].get("logits_sha256")
                        != by_condition["wrapper_zero"].get("logits_sha256")
                    )
                )
            else:
                pair_ok = pair_ok and record.get("synthetic_output_changed") is False
    checks = {
        "record_counts_exact": len(calibration_records) == 32
        and len(pair_records) == 448,
        "calibration_order_and_capture_metadata_exact": calibration_ok,
        "pair_schedule_observations_and_margins_exact": pair_ok,
        "record_ids_unique": len(set(capture_ids)) == 32
        and len(set(pair_ids)) == 448,
        "all_routes_persistent_and_no_public": all(
            record.get("route") == "persistent_wrapper" for record in pair_records
        )
        and all(
            "public" not in json.dumps(record, sort_keys=True)
            for record in pair_records
        ),
        "forward_count_exact": len(calibration_records) + 2 * len(pair_records)
        == 928,
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-D offline ledger structure failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def analyze_calibration_capture_identity(
    records: Sequence[Mapping[str, Any]], calibration_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    fixtures = calibration_manifest["fixtures"]
    hashes = {
        fixture["fixture_id"]: str(record["capture_sha256"])
        for fixture, record in zip(fixtures, records[:32])
    }
    by_cell: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for fixture in fixtures:
        key = (int(fixture["identity_index"]), int(fixture["goal_index"]))
        by_cell.setdefault(key, []).append(
            (int(fixture["replicate"]), hashes[fixture["fixture_id"]])
        )
    if len(by_cell) != 16 or any(len(items) != 2 for items in by_cell.values()):
        raise ValueError("D9-D offline calibration cell structure changed")
    cells = []
    for (identity, goal), items in sorted(by_cell.items()):
        ordered = sorted(items)
        cells.append(
            {
                "identity_index": identity,
                "goal_index": goal,
                "replicate_1_capture_sha256": ordered[0][1],
                "replicate_2_capture_sha256": ordered[1][1],
                "capture_hashes_identical": ordered[0][1] == ordered[1][1],
            }
        )
    return {
        "cells": cells,
        "identical_capture_hash_pairs": sum(
            item["capture_hashes_identical"] for item in cells
        ),
        "distinct_capture_hash_pairs": sum(
            not item["capture_hashes_identical"] for item in cells
        ),
        "capture_vectors_stored": False,
        "replicate_numeric_distance_identifiable": False,
        "identifiability_reason": (
            "raw_ledger_contains_capture_sha256_only_not_the_2560_value_capture_vectors"
        ),
        "per_cell_replicate_magnitude_or_cosine_claim_allowed": False,
    }


def _rms(vector: Sequence[Any]) -> float:
    values = [_number(item, "projection value") for item in vector]
    return math.sqrt(sum(value * value for value in values) / len(values))


def _cosine(left: Sequence[Any], right: Sequence[Any]) -> float:
    a = [_number(item, "projection left") for item in left]
    b = [_number(item, "projection right") for item in right]
    denominator = math.sqrt(sum(value * value for value in a)) * math.sqrt(
        sum(value * value for value in b)
    )
    if denominator == 0.0:
        raise ValueError("D9-D offline projection contains a zero vector")
    return sum(x * y for x, y in zip(a, b)) / denominator


def _matrix_rank(rows: Sequence[Sequence[Any]]) -> int:
    matrix = [[_number(value, "rank value") for value in row] for row in rows]
    if not matrix:
        return 0
    scale = max(abs(value) for row in matrix for value in row)
    if scale == 0.0:
        return 0
    tolerance = scale * 1e-10
    rank = 0
    columns = len(matrix[0])
    for column in range(columns):
        pivot = max(range(rank, len(matrix)), key=lambda index: abs(matrix[index][column]))
        if abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        for row_index in range(rank + 1, len(matrix)):
            factor = matrix[row_index][column] / pivot_value
            if factor == 0.0:
                continue
            matrix[row_index][column:] = [
                value - factor * pivot_component
                for value, pivot_component in zip(
                    matrix[row_index][column:], matrix[rank][column:]
                )
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _pairwise_cosines(vectors: Sequence[Sequence[Any]]) -> list[float]:
    return [
        _cosine(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]


def analyze_projection_geometry(projection: Mapping[str, Any]) -> dict[str, Any]:
    audited = audit_frozen_projection_artifact(
        projection, expected_dimension=2560, fixture_only=False
    )["artifact"]
    identity_map = audited["parameters"]["identity_weights"]
    goal_map = audited["parameters"]["goal_weights"]
    identity = [identity_map[f"identity_{index}"] for index in range(4)]
    goal = [goal_map[f"goal_{index}"] for index in range(4)]
    identity_rms = [_rms(vector) for vector in identity]
    goal_rms = [_rms(vector) for vector in goal]
    cross_cosines = [_cosine(i_vector, g_vector) for i_vector in identity for g_vector in goal]
    active_vectors = [
        [i_value + g_value for i_value, g_value in zip(i_vector, g_vector)]
        for i_vector in identity
        for g_vector in goal
    ]
    active_rms = [_rms(vector) for vector in active_vectors]
    return {
        "identity_branch_rms": _summary(identity_rms),
        "goal_branch_rms": _summary(goal_rms),
        "identity_within_group_cosine": _summary(_pairwise_cosines(identity)),
        "goal_within_group_cosine": _summary(_pairwise_cosines(goal)),
        "cross_field_cosine": _summary(cross_cosines),
        "cross_field_maximum_absolute_cosine": max(abs(value) for value in cross_cosines),
        "active_sum_rms_across_sixteen_cells": _summary(active_rms),
        "identity_vector_rank": _matrix_rank(identity),
        "goal_vector_rank": _matrix_rank(goal),
        "joint_vector_rank": _matrix_rank(identity + goal),
        "geometry_is_descriptive_not_causal_evidence": True,
    }


def _bootstrap_lower_bound(values: Sequence[float], seed: int) -> float:
    import random

    finite = [_number(value, "bootstrap value") for value in values]
    if len(finite) != 16:
        raise ValueError("D9-D offline bootstrap requires sixteen values")
    rng = random.Random(seed)
    means = [
        sum(finite[rng.randrange(16)] for _ in range(16)) / 16
        for _ in range(100000)
    ]
    means.sort()
    return means[999]


def analyze_causal_distribution(
    *, records: Sequence[Mapping[str, Any]], heldout_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    pairs = records[32:]
    fixtures = {item["fixture_id"]: item for item in heldout_manifest["fixtures"]}
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    deltas_by_contrast: dict[str, list[float]] = {name: [] for name in CONTRASTS}
    deltas_by_order: dict[str, dict[str, list[float]]] = {
        name: {"zero_first": [], "condition_first": []} for name in CONTRASTS
    }
    active_by_rotation: dict[int, list[float]] = {index: [] for index in range(4)}
    active_by_full_output: dict[bool, list[float]] = {False: [], True: []}
    output_changed_by_contrast: dict[str, int] = {name: 0 for name in CONTRASTS}
    for record in pairs:
        base_case = str(record["base_case_id"])
        contrast = str(record["contrast"])
        grouped.setdefault(base_case, {}).setdefault(contrast, []).append(record)
        delta = _number(
            record["condition_margins"]["target_alignment_margin"],
            "condition target margin",
        ) - _number(record["zero_margins"]["target_alignment_margin"], "zero margin")
        deltas_by_contrast[contrast].append(delta)
        deltas_by_order[contrast][str(record["pair_order"])].append(delta)
        observations = {item["condition"]: item for item in record["observations"]}
        if (
            observations[contrast]["logits_sha256"]
            != observations["wrapper_zero"]["logits_sha256"]
        ):
            output_changed_by_contrast[contrast] += 1
        if contrast == "active_true":
            rotation = int(record["code_rotation"])
            active_by_rotation[rotation].append(delta)
            active_by_full_output[bool(fixtures[record["fixture_id"]]["full_output"])].append(
                delta
            )
    if len(grouped) != 16 or any(
        set(contrasts) != set(CONTRASTS)
        or any(len(items) != 4 for items in contrasts.values())
        for contrasts in grouped.values()
    ):
        raise ValueError("D9-D offline base-case rotation groups are incomplete")
    base_case_summaries: list[dict[str, Any]] = []
    active_values: list[float] = []
    true_random_values: list[float] = []
    identity_levels: dict[int, list[float]] = {index: [] for index in range(4)}
    goal_levels: dict[int, list[float]] = {index: [] for index in range(4)}
    mask_identity_count = 0
    mask_goal_count = 0
    swap_identity_count = 0
    swap_goal_count = 0
    synthetic_count = 0
    for base_case in sorted(grouped):
        contrasts = grouped[base_case]
        contrast_means = {
            name: statistics.fmean(
                _number(item["condition_margins"]["target_alignment_margin"], "condition")
                - _number(item["zero_margins"]["target_alignment_margin"], "zero")
                for item in items
            )
            for name, items in contrasts.items()
        }
        active = contrast_means["active_true"]
        true_random = active - contrast_means["matched_random"]
        identity = int(contrasts["active_true"][0]["identity_index"])
        goal = int(contrasts["active_true"][0]["goal_index"])
        identity_levels[identity].append(active)
        goal_levels[goal].append(active)
        mask_i_identity = statistics.fmean(
            _number(item["condition_margins"]["identity_margin"], "mask identity")
            - _number(item["zero_margins"]["identity_margin"], "zero identity")
            for item in contrasts["mask_identity"]
        )
        mask_i_goal = statistics.fmean(
            _number(item["condition_margins"]["goal_margin"], "mask goal")
            - _number(item["zero_margins"]["goal_margin"], "zero goal")
            for item in contrasts["mask_identity"]
        )
        mask_g_goal = statistics.fmean(
            _number(item["condition_margins"]["goal_margin"], "mask goal")
            - _number(item["zero_margins"]["goal_margin"], "zero goal")
            for item in contrasts["mask_goal"]
        )
        mask_g_identity = statistics.fmean(
            _number(item["condition_margins"]["identity_margin"], "mask identity")
            - _number(item["zero_margins"]["identity_margin"], "zero identity")
            for item in contrasts["mask_goal"]
        )
        mask_i_specific = mask_i_identity < 0.0 and mask_i_goal > 0.0
        mask_g_specific = mask_g_goal < 0.0 and mask_g_identity > 0.0
        swap_i_advantage = statistics.fmean(
            _number(
                item["condition_margins"]["identity_swap_advantage"],
                "identity swap advantage",
            )
            for item in contrasts["swap_identity"]
        )
        swap_g_advantage = statistics.fmean(
            _number(
                item["condition_margins"]["goal_swap_advantage"],
                "goal swap advantage",
            )
            for item in contrasts["swap_goal"]
        )
        mask_identity_count += int(mask_i_specific)
        mask_goal_count += int(mask_g_specific)
        swap_identity_count += int(swap_i_advantage > 0.0)
        swap_goal_count += int(swap_g_advantage > 0.0)
        synthetic_changed = sum(
            bool(item["synthetic_output_changed"])
            for item in contrasts["synthetic_active"]
        )
        synthetic_count += synthetic_changed
        active_values.append(active)
        true_random_values.append(true_random)
        base_case_summaries.append(
            {
                "base_case_id": base_case,
                "identity_index": identity,
                "goal_index": goal,
                "contrast_target_margin_deltas": contrast_means,
                "active_minus_zero_positive": active > 0.0,
                "active_minus_matched_random": true_random,
                "mask_identity_identity_margin_delta": mask_i_identity,
                "mask_identity_goal_margin_delta": mask_i_goal,
                "mask_identity_specific": mask_i_specific,
                "mask_goal_goal_margin_delta": mask_g_goal,
                "mask_goal_identity_margin_delta": mask_g_identity,
                "mask_goal_specific": mask_g_specific,
                "swap_identity_advantage": swap_i_advantage,
                "swap_identity_follows": swap_i_advantage > 0.0,
                "swap_goal_advantage": swap_g_advantage,
                "swap_goal_follows": swap_g_advantage > 0.0,
                "synthetic_changed_rotations": synthetic_changed,
            }
        )
    metrics = {
        "active_minus_zero_mean": statistics.fmean(active_values),
        "active_minus_zero_lb99": _bootstrap_lower_bound(active_values, 29083102),
        "positive_base_cases": sum(value > 0.0 for value in active_values),
        "identity_level_min_positive": min(
            sum(value > 0.0 for value in values)
            for values in identity_levels.values()
        ),
        "goal_level_min_positive": min(
            sum(value > 0.0 for value in values) for values in goal_levels.values()
        ),
        "true_minus_random_lb99": _bootstrap_lower_bound(
            true_random_values, 29083103
        ),
        "mask_identity_specific_count": mask_identity_count,
        "mask_goal_specific_count": mask_goal_count,
        "swap_identity_follow_count": swap_identity_count,
        "swap_goal_follow_count": swap_goal_count,
        "synthetic_active_changed_fixture_count": synthetic_count,
    }
    endpoint = evaluate_candidate(metrics)
    return {
        "endpoint": {
            "metrics": metrics,
            "checks": endpoint["checks"],
            "all_gates_pass": endpoint["all_gates_pass"],
            "decision": endpoint["decision"],
            "self_effect_conclusion": False,
        },
        "base_case_summaries": base_case_summaries,
        "contrast_rotation_level_distributions": {
            name: _summary(values) for name, values in deltas_by_contrast.items()
        },
        "active_by_code_rotation": {
            str(rotation): _summary(values)
            for rotation, values in active_by_rotation.items()
        },
        "active_by_full_output": {
            str(full_output).lower(): _summary(values)
            for full_output, values in active_by_full_output.items()
        },
        "pair_order_distributions": {
            name: {
                order: _summary(values)
                for order, values in orders.items()
                if values
            }
            for name, orders in deltas_by_order.items()
        },
        "output_changed_fixture_counts_by_contrast": output_changed_by_contrast,
        "identity_level_active": {
            str(index): _summary(values) for index, values in identity_levels.items()
        },
        "goal_level_active": {
            str(index): _summary(values) for index, values in goal_levels.items()
        },
    }


def _static_ast_audit(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])
    return {
        "imported_roots": sorted(set(imported_roots)),
        "torch_not_imported": "torch" not in imported_roots,
        "rwkv_not_imported": "rwkv" not in imported_roots,
    }


def build_static_report(
    *, config_path: str | Path = CONFIG_RELATIVE_PATH,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = (root / config_path).resolve()
    config = _object(config_file, "config")
    config_checks = validate_config(config)
    source_locks = validate_source_locks(root, config)
    diagnostic_source_inventory = {
        relative: (root / relative).is_file()
        for relative in DIAGNOSTIC_SOURCE_PATHS
    }
    ast_audit = _static_ast_audit(
        root / "src/psa/self_model/d9d_offline_causal_diagnostic.py"
    )
    evidence = config["evidence"]
    checks = {
        "config_valid": all(config_checks.values()),
        "frozen_sources_valid": all(source_locks.values()),
        "diagnostic_source_inventory_complete": all(
            diagnostic_source_inventory.values()
        ),
        "diagnostic_has_no_rwkv_or_torch_import": ast_audit["rwkv_not_imported"]
        and ast_audit["torch_not_imported"],
        "real_evidence_is_read_only_input": all(
            str(evidence[field]).startswith("results/")
            for field in (
                "authorization_path",
                "claim_path",
                "projection_path",
                "raw_ledger_path",
                "report_path",
                "integrity_path",
            )
        ),
        "d9d_and_all_model_authority_closed": all(
            value is False for value in config["safety"].values()
        ),
        "replicate_numeric_limit_declared": config["analysis_contract"][
            "replicate_numeric_distance_identifiable"
        ]
        is False,
        "independent_route_is_candidate_only": config["route_review"][
            "candidate_status"
        ]
        == "preregistration_candidate_only_not_implemented_or_authorized",
    }
    if not all(checks.values()):
        raise RuntimeError("D9-D offline static verification failed")
    report: dict[str, Any] = {
        "report_version": DIAGNOSTIC_VERSION,
        "status": "d9d_offline_causal_diagnostic_static_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "source_locks": source_locks,
        "diagnostic_source_inventory": diagnostic_source_inventory,
        "diagnostic_source_digests": {
            relative: sha256_file(root / relative)
            for relative in DIAGNOSTIC_SOURCE_PATHS
        },
        "ast_audit": ast_audit,
        "real_evidence_loaded": False,
        "model_executed": False,
        "next_gate": NEXT_GATE,
        "safety": copy.deepcopy(config["safety"]),
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report


def build_offline_diagnostic_report(
    *, config_path: str | Path = CONFIG_RELATIVE_PATH,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = _object(root / config_path, "config")
    config_checks = validate_config(config)
    source_locks = validate_source_locks(root, config)
    diagnostic_source_inventory = {
        relative: (root / relative).is_file()
        for relative in DIAGNOSTIC_SOURCE_PATHS
    }
    if not all(diagnostic_source_inventory.values()):
        raise RuntimeError("D9-D offline diagnostic source inventory is incomplete")
    expected = config["evidence"]
    evidence_paths = {
        name: root / expected[f"{name}_path"]
        for name in (
            "authorization",
            "claim",
            "projection",
            "raw_ledger",
            "report",
            "integrity",
        )
    }
    file_hash_checks = {
        f"{name}_file_sha256_exact": sha256_file(path)
        == expected[f"{name}_file_sha256"]
        for name, path in evidence_paths.items()
        if name != "integrity"
    }
    if not all(file_hash_checks.values()):
        raise ValueError(
            "D9-D offline evidence file hash differs: "
            + ", ".join(name for name, valid in file_hash_checks.items() if not valid)
        )
    failure_absent = not (evidence_paths["report"].parent / "failure.json").exists()
    if not failure_absent:
        raise ValueError("D9-D offline completed evidence unexpectedly has failure.json")
    authorization = _object(evidence_paths["authorization"], "authorization")
    claim = _object(evidence_paths["claim"], "claim")
    projection = _object(evidence_paths["projection"], "projection")
    report = _object(evidence_paths["report"], "report")
    integrity = _object(evidence_paths["integrity"], "integrity")
    records = _jsonl(evidence_paths["raw_ledger"])
    bundle_checks = validate_evidence_bundle(
        config=config,
        authorization=authorization,
        claim=claim,
        projection=projection,
        report=report,
        integrity=integrity,
    )
    frozen = config["frozen_inputs"]
    calibration_manifest = _object(
        root / frozen["calibration_manifest_path"], "calibration manifest"
    )
    heldout_manifest = _object(
        root / frozen["heldout_manifest_path"], "heldout manifest"
    )
    schedule = _object(root / frozen["schedule_manifest_path"], "schedule")
    ledger_checks = validate_ledger_structure(
        records=records,
        calibration_manifest=calibration_manifest,
        heldout_manifest=heldout_manifest,
        schedule=schedule,
        projection_digest=expected["projection_artifact_digest_sha256"],
    )
    calibration = analyze_calibration_capture_identity(records, calibration_manifest)
    geometry = analyze_projection_geometry(projection)
    causal = analyze_causal_distribution(
        records=records, heldout_manifest=heldout_manifest
    )
    endpoint_matches = causal["endpoint"] == report["endpoint"]
    if not endpoint_matches:
        raise ValueError("D9-D offline recomputed endpoint differs from frozen report")
    route_review = {
        "current_d9d_projection_supported": False,
        "mechanism_positive_control_supported": causal["endpoint"]["checks"][
            "synthetic_positive_control_passes"
        ],
        "field_specific_causal_signal_supported": False,
        "d9d_rerun_allowed": False,
        "posthoc_gain_layer_or_threshold_search_on_d9d_allowed": False,
        "calibration_numeric_replicate_stability_identifiable": False,
        "scientifically_independent_route_candidate_exists": True,
        "candidate": config["route_review"]["independent_candidate"],
        "candidate_status": config["route_review"]["candidate_status"],
        "candidate_requirements": {
            "new_tokens_fixtures_seeds_claim_and_output": True,
            "semantically_structured_calibration": True,
            "disjoint_representation_validation_before_intervention": True,
            "cross_replicate_field_decodability_gate": True,
            "cannot_reuse_d9d_result_as_new_experiment_data": True,
        },
        "interpretation": (
            "the_intervention_mechanism_changes_outputs_but_the_frozen_real_projection_"
            "does_not_show_reliable_field_specific_heldout_causality"
        ),
    }
    checks = {
        "config_valid": all(config_checks.values()),
        "frozen_sources_valid": all(source_locks.values()),
        "evidence_file_hashes_valid": all(file_hash_checks.values()),
        "evidence_bundle_valid": all(bundle_checks.values()),
        "ledger_structure_valid": all(ledger_checks.values()),
        "endpoint_exactly_recomputed": endpoint_matches,
        "mechanism_positive_control_passed": route_review[
            "mechanism_positive_control_supported"
        ],
        "all_real_projection_causal_gates_did_not_pass": causal["endpoint"][
            "all_gates_pass"
        ]
        is False,
        "self_effect_conclusion_remains_false": causal["endpoint"][
            "self_effect_conclusion"
        ]
        is False,
        "replicate_numeric_stability_not_overclaimed": calibration[
            "replicate_numeric_distance_identifiable"
        ]
        is False,
        "d9d_rerun_remains_closed": route_review["d9d_rerun_allowed"] is False,
        "new_route_is_candidate_only": route_review["candidate_status"]
        == "preregistration_candidate_only_not_implemented_or_authorized",
        "failure_artifact_absent_for_completed_run": failure_absent,
    }
    if not all(checks.values()):
        raise RuntimeError("D9-D offline diagnostic decision failed closed")
    result: dict[str, Any] = {
        "report_version": DIAGNOSTIC_VERSION,
        "status": "d9d_failure_offline_projection_ledger_diagnostic_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "source_locks": source_locks,
        "diagnostic_source_digests": {
            relative: sha256_file(root / relative)
            for relative in DIAGNOSTIC_SOURCE_PATHS
        },
        "evidence_file_hash_checks": file_hash_checks,
        "evidence_bundle_checks": bundle_checks,
        "ledger_checks": ledger_checks,
        "calibration_replicate_audit": calibration,
        "projection_geometry": geometry,
        "causal_distribution": causal,
        "route_review": route_review,
        "safety": copy.deepcopy(config["safety"]),
        "model_executed": False,
        "next_gate": NEXT_GATE,
    }
    result["report_digest_sha256"] = sha256_json(result)
    return result
