from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d9a_within_wrapper_causal_isolation import (
    CONTRASTS,
    evaluate_candidate,
    expand_fixtures,
    expand_schedule,
    validate_design,
)


CONTRACT_VERSION = "0.1-self-model-d9b-manifest-endpoint-contract"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9b_manifest_endpoint_contract.json"
)
DESIGN_RELATIVE_PATH = (
    "configs/preregistration/"
    "self_model_v0_1_d9a_within_wrapper_causal_isolation.draft.json"
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
ENDPOINT_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d9_causal_endpoint.json"
)
DESIGN_SHA256 = "430926dbf6eafad2246e25e695ca9783ea73df2f91b9c427f9105de44e9858a8"
DESIGN_REPORT_SHA256 = (
    "df3a203a8b9d483e9ddda0fc434f629bcda3c021655f9dee841b8000221eb714"
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
ENDPOINT_SHA256 = (
    "1323b15f269f2bd4123992e213ec4d6dd1262376ad5c7b9563279f6b2543f562"
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
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D9-B deterministic manifests与fake-first causal endpoint "
    "contract纯离线实现；只允许将D9-A冻结的32个calibration fixture、64个held-out "
    "fixture、七类同wrapper成对对照、448个pair/928次未来forward、训练与held-out隔离、"
    "配对反平衡、通过阈值和14个全新命名空间物化为deterministic calibration/held-out/"
    "schedule/endpoint manifests、纯Python数据结构、完整性校验、失败关闭与合成验收；不得"
    "修改D9-A阈值、引入public计分路线，或复用D8-C的fixture、token、seed、claim和结果作为"
    "新实验数据。本轮不实现projection contract或真实projection、不探测installed source、"
    "不修改真实runner、不实现执行入口、不导入RWKV/Torch、不访问权重、不加载或执行模型、"
    "不创建authorization/claim/output；不授权D9-C/D9-D真实执行、D8-C或历史重跑、"
    "D7-D/D7-E、正式测试集、Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
CLASSIFICATION = (
    "d9b_deterministic_manifests_and_fake_within_wrapper_causal_endpoint_"
    "verified_no_model"
)
NEXT_GATE = "remote_no_model_d9b_verification_then_separate_d9c_confirmation"
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    DESIGN_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    HELDOUT_RELATIVE_PATH,
    SCHEDULE_RELATIVE_PATH,
    ENDPOINT_RELATIVE_PATH,
    "docs/self_model_v0_1_d9a_within_wrapper_causal_isolation.md",
    "docs/self_model_v0_1_d9b_manifest_endpoint_contract.md",
    "scripts/verify_self_model_v0_1_d9a_within_wrapper_causal_isolation.py",
    "scripts/verify_self_model_v0_1_d9b_manifest_endpoint_contract.py",
    "src/psa/self_model/d9a_within_wrapper_causal_isolation.py",
    "src/psa/self_model/d9b_manifest_endpoint_contract.py",
    "tests/test_self_model_d9a_within_wrapper_causal_isolation.py",
    "tests/test_self_model_d9b_manifest_endpoint_contract.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D9-B {label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"D9-B {label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"D9-B {label} must be finite")
    return result


def validate_contract_config(config: Mapping[str, Any]) -> dict[str, bool]:
    frozen = config.get("frozen_design", {})
    manifests = config.get("manifests", {})
    counts = config.get("counts", {})
    runtime = config.get("fake_runtime_contract", {})
    authority = config.get("authority", {})
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D9-B_deterministic_manifests_and_fake_causal_endpoint_contract"
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
            "calibration_path": CALIBRATION_RELATIVE_PATH,
            "calibration_sha256": CALIBRATION_SHA256,
            "heldout_path": HELDOUT_RELATIVE_PATH,
            "heldout_sha256": HELDOUT_SHA256,
            "schedule_path": SCHEDULE_RELATIVE_PATH,
            "schedule_sha256": SCHEDULE_SHA256,
            "endpoint_path": ENDPOINT_RELATIVE_PATH,
            "endpoint_sha256": ENDPOINT_SHA256,
        },
        "counts_exact": counts
        == {
            "calibration_fixtures": 32,
            "heldout_fixtures": 64,
            "base_cases": 16,
            "contrasts": 7,
            "heldout_pairs": 448,
            "calibration_forward_calls": 32,
            "heldout_forward_calls": 896,
            "total_future_forward_calls": 928,
            "future_namespaces": 14,
        },
        "fake_runtime_exact": runtime
        == {
            "fixture_kind": "d9b_pure_python_scalar_margin_records_only",
            "tensor_projection_or_model_object_allowed": False,
            "public_scoring_route_allowed": False,
            "input_mutation_allowed": False,
            "complete_ordered_ledger_required": True,
            "finite_values_required": True,
            "calibration_heldout_leakage_policy": "reject_before_decision",
            "duplicate_missing_reordered_or_route_mismatch_policy": (
                "reject_before_decision"
            ),
            "output_claim": (
                "endpoint_contract_only_not_real_projection_or_self_effect_evidence"
            ),
        },
        "gates_ordered_and_later_closed": config.get("gate_sequence")
        == [
            {"gate_id": "D9-A", "implemented": True, "model_execution": False},
            {"gate_id": "D9-B", "implemented": True, "model_execution": False},
            {"gate_id": "D9-C", "implemented": False, "authorized": False},
            {
                "gate_id": "D9-D",
                "implemented": False,
                "authorized": False,
                "future_forward_calls": 928,
            },
        ],
        "d9b_authority_exact": authority.get(
            "d9b_manifest_implementation_authorized"
        )
        is True
        and authority.get("pure_python_fake_endpoint_authorized") is True
        and authority.get("offline_integrity_report_authorized") is True,
        "all_model_projection_execution_and_later_authority_closed": all(
            authority.get(name) is False
            for name in (
                "projection_contract_implementation_authorized",
                "real_projection_construction_authorized",
                "installed_source_probe_authorized",
                "real_runner_modification_authorized",
                "execution_entry_implementation_authorized",
                "authorization_claim_or_output_creation_authorized",
                "rwkv_import_authorized",
                "torch_import_authorized",
                "weights_access_authorized",
                "model_load_authorized",
                "model_execution_authorized",
                "d9c_authorized",
                "d9d_real_execution_authorized",
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
        "classification_and_next_gate_exact": config.get(
            "required_classification"
        )
        == CLASSIFICATION
        and config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D9-B contract config failed closed: " + ", ".join(failed))
    return checks


def validate_calibration_manifest(
    manifest: Mapping[str, Any], expanded: Mapping[str, Any]
) -> dict[str, bool]:
    fixtures = manifest.get("fixtures", [])
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d9b-calibration-manifest"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D9-calibration-fixtures-v01"
        and manifest.get("status")
        == "deterministic_calibration_manifest_frozen_unrun",
        "design_and_namespace_exact": manifest.get("design_path")
        == DESIGN_RELATIVE_PATH
        and manifest.get("design_sha256") == DESIGN_SHA256
        and manifest.get("namespace")
        == "psa-self-model-v0.1-d9a-calibration-fixtures-20260831"
        and manifest.get("seed") == "d9a-calibration-4f73b2d1-20260831",
        "counts_and_separation_exact": manifest.get("fixture_count") == 32
        and manifest.get("identity_levels") == 4
        and manifest.get("goal_levels") == 4
        and manifest.get("replicates_per_cell") == 2
        and manifest.get("capture_calls_per_fixture") == 1
        and manifest.get("heldout_endpoint_scored") is False
        and manifest.get("heldout_access_before_future_projection_freeze") is False,
        "fixtures_exact": fixtures == expanded["calibration_manifest"]["fixtures"],
        "commitment_exact": manifest.get("expanded_commitment_sha256")
        == CALIBRATION_COMMITMENT
        == expanded["calibration_manifest"]["commitment_sha256"],
        "unrun_noncore_exact": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("model_executed") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-B calibration manifest failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def validate_heldout_manifest(
    manifest: Mapping[str, Any], expanded: Mapping[str, Any]
) -> dict[str, bool]:
    fixtures = manifest.get("fixtures", [])
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d9b-heldout-manifest"
        and manifest.get("manifest_id")
        == "Self-Model-v0.1-D9-heldout-fixtures-v01"
        and manifest.get("status")
        == "deterministic_heldout_manifest_frozen_unrun",
        "design_and_namespace_exact": manifest.get("design_path")
        == DESIGN_RELATIVE_PATH
        and manifest.get("design_sha256") == DESIGN_SHA256
        and manifest.get("namespace")
        == "psa-self-model-v0.1-d9a-heldout-fixtures-20260831"
        and manifest.get("seed") == "d9a-heldout-9c18e6a4-20260831",
        "counts_rotation_and_state_exact": manifest.get("base_case_count") == 16
        and manifest.get("fixture_count") == 64
        and manifest.get("code_rotations_per_base_case") == 4
        and manifest.get("content_shared_within_base_case") is True
        and manifest.get("rotation_code_unique_per_fixture") is True
        and manifest.get("public_route_allowed") is False,
        "fixtures_exact": fixtures == expanded["heldout_manifest"]["fixtures"],
        "commitment_exact": manifest.get("expanded_commitment_sha256")
        == HELDOUT_COMMITMENT
        == expanded["heldout_manifest"]["commitment_sha256"],
        "unrun_noncore_exact": manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("model_executed") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-B heldout manifest failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def validate_schedule_manifest(
    manifest: Mapping[str, Any], expanded: Mapping[str, Any]
) -> dict[str, bool]:
    pair_blocks = manifest.get("heldout_pair_blocks", [])
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d9b-within-wrapper-schedule"
        and manifest.get("status")
        == "deterministic_schedule_manifest_frozen_unrun",
        "namespace_seed_and_contrasts_exact": manifest.get("namespace")
        == "psa-self-model-v0.1-d9a-within-wrapper-paired-schedule-20260831"
        and manifest.get("seed") == "d9a-schedule-2a6d805e-20260831"
        and manifest.get("contrasts") == list(CONTRASTS),
        "counts_exact": manifest.get("calibration_call_count") == 32
        and manifest.get("heldout_pair_count") == 448
        and manifest.get("heldout_forward_calls") == 896
        and manifest.get("total_future_forward_calls") == 928,
        "same_wrapper_only": manifest.get("public_route_allowed") is False
        and all(block.get("route") == "persistent_wrapper" for block in pair_blocks),
        "expanded_calls_exact": manifest.get("calibration_calls")
        == expanded["calibration_calls"]
        and pair_blocks == expanded["heldout_pair_blocks"],
        "commitment_exact": manifest.get("expanded_commitment_sha256")
        == SCHEDULE_COMMITMENT
        == expanded["commitment_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-B schedule manifest failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def validate_endpoint_manifest(
    manifest: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {
        "identity_exact": manifest.get("manifest_version")
        == "0.1-self-model-d9b-causal-endpoint"
        and manifest.get("status") == "endpoint_manifest_frozen_unrun",
        "design_exact": manifest.get("design_path") == DESIGN_RELATIVE_PATH
        and manifest.get("design_sha256") == DESIGN_SHA256,
        "analysis_and_metrics_exact": manifest.get("analysis_unit")
        == "sixteen_identity_goal_base_cases_after_four_rotation_label_marginalization"
        and manifest.get("metrics_contract")
        == {
            "pair_value": (
                "condition_target_alignment_margin_minus_paired_wrapper_zero_"
                "target_alignment_margin"
            ),
            "rotation_aggregation": (
                "arithmetic_mean_across_four_code_rotations_per_base_case_and_contrast"
            ),
            "bootstrap_unit": "identity_goal_base_case",
            "synthetic_active_unit": (
                "heldout_fixture_exact_target_layer_application_and_output_change"
            ),
        },
        "thresholds_unchanged": manifest.get("thresholds")
        == design["endpoint_contract"]
        and manifest.get("thresholds_changed_from_d9a") is False,
        "public_projection_and_claims_closed": manifest.get(
            "public_route_allowed"
        )
        is False
        and manifest.get("projection_contract_implemented") is False
        and manifest.get("self_effect_conclusion_allowed") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-B endpoint manifest failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def _fake_pair_record(block: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    contrast = str(block["contrast"])
    base = int(block["identity_index"]) * 4 + int(block["goal_index"])
    variation = (base % 4) * 0.002
    active = 0.22 + variation
    if scenario == "field_specific_candidate":
        deltas = {
            "active_true": active,
            "mask_identity": 0.06,
            "mask_goal": 0.06,
            "swap_identity": 0.16,
            "swap_goal": 0.16,
            "matched_random": 0.03,
            "synthetic_active": 0.25,
        }
        identity_delta = -0.12 if contrast == "mask_identity" else 0.10
        goal_delta = -0.12 if contrast == "mask_goal" else 0.10
        swap_identity_follows = contrast == "swap_identity"
        swap_goal_follows = contrast == "swap_goal"
        output_changed = contrast == "synthetic_active"
    elif scenario == "wrapper_route_only":
        deltas = {name: 0.0 for name in CONTRASTS}
        identity_delta = goal_delta = 0.0
        swap_identity_follows = swap_goal_follows = False
        output_changed = contrast == "synthetic_active"
    elif scenario == "nonspecific_active_or_random":
        deltas = {name: 0.18 for name in CONTRASTS}
        identity_delta = goal_delta = 0.18
        swap_identity_follows = swap_goal_follows = False
        output_changed = contrast == "synthetic_active"
    else:
        raise ValueError("unknown D9-B fake scenario")
    condition_margin = deltas[contrast]
    values = {
        "wrapper_zero": 0.0,
        contrast: condition_margin,
    }
    return {
        "pair_block_id": block["pair_block_id"],
        "fixture_id": block["fixture_id"],
        "base_case_id": block["base_case_id"],
        "identity_index": block["identity_index"],
        "goal_index": block["goal_index"],
        "code_rotation": block["code_rotation"],
        "contrast": contrast,
        "latin_position": block["latin_position"],
        "pair_order": block["pair_order"],
        "condition_order": list(block["condition_order"]),
        "route": "persistent_wrapper",
        "source_state_contract": block["source_state_contract"],
        "phase": "heldout",
        "observations": [
            {"condition": name, "target_alignment_margin": values[name]}
            for name in block["condition_order"]
        ],
        "identity_margin_delta": identity_delta,
        "goal_margin_delta": goal_delta,
        "swap_identity_follows": swap_identity_follows,
        "swap_goal_follows": swap_goal_follows,
        "output_changed": output_changed,
        "target_layer_applications": 1 if contrast == "synthetic_active" else 0,
    }


def make_fake_ledger(
    schedule: Mapping[str, Any], scenario: str
) -> list[dict[str, Any]]:
    calibration = [
        {
            **copy.deepcopy(call),
            "record_type": "calibration_capture",
            "route": "persistent_wrapper_capture",
            "finite_capture": True,
        }
        for call in schedule["calibration_calls"]
    ]
    heldout = [
        {"record_type": "heldout_pair", **_fake_pair_record(block, scenario)}
        for block in schedule["heldout_pair_blocks"]
    ]
    return calibration + heldout


def validate_fake_ledger(
    ledger: Sequence[Mapping[str, Any]], schedule: Mapping[str, Any]
) -> dict[str, bool]:
    expected_calibration = schedule["calibration_calls"]
    expected_pairs = schedule["heldout_pair_blocks"]
    if len(ledger) != 480:
        raise ValueError("D9-B ledger must contain 32 capture and 448 pair records")
    calibration = ledger[:32]
    pairs = ledger[32:]
    checks = {
        "record_count_exact": len(calibration) == 32 and len(pairs) == 448,
        "calibration_order_and_phase_exact": all(
            record.get("record_type") == "calibration_capture"
            and record.get("call_id") == expected.get("call_id")
            and record.get("fixture_id") == expected.get("fixture_id")
            and record.get("phase") == "calibration"
            and record.get("heldout_scored") is False
            and record.get("route") == "persistent_wrapper_capture"
            and record.get("finite_capture") is True
            for record, expected in zip(calibration, expected_calibration)
        ),
        "pair_order_identity_and_route_exact": all(
            record.get("record_type") == "heldout_pair"
            and record.get("pair_block_id") == expected.get("pair_block_id")
            and record.get("fixture_id") == expected.get("fixture_id")
            and record.get("contrast") == expected.get("contrast")
            and record.get("condition_order") == expected.get("condition_order")
            and record.get("pair_order") == expected.get("pair_order")
            and record.get("latin_position") == expected.get("latin_position")
            and record.get("route") == "persistent_wrapper"
            and record.get("phase") == "heldout"
            for record, expected in zip(pairs, expected_pairs)
        ),
        "all_ids_unique": len(
            {record.get("call_id") for record in calibration}
        )
        == 32
        and len({record.get("pair_block_id") for record in pairs}) == 448,
        "observations_exact_and_finite": all(
            [item.get("condition") for item in record.get("observations", [])]
            == record.get("condition_order")
            and len(record.get("observations", [])) == 2
            and all(
                math.isfinite(
                    _finite_number(item.get("target_alignment_margin"), "margin")
                )
                for item in record.get("observations", [])
            )
            for record in pairs
        ),
        "no_public_route_or_phase_leakage": all(
            "public" not in str(record.get("route", ""))
            and "public" not in json.dumps(record, sort_keys=True)
            for record in pairs
        )
        and not ({record.get("fixture_id") for record in calibration}
                 & {record.get("fixture_id") for record in pairs}),
        "future_forward_count_exact": len(calibration) + 2 * len(pairs) == 928,
    }
    if not all(checks.values()):
        raise ValueError(
            "D9-B fake ledger failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    return checks


def _bootstrap_lower_bound(values: Sequence[float], seed: int) -> float:
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError("D9-B bootstrap requires sixteen finite base-case values")
    rng = random.Random(seed)
    length = len(values)
    means = [
        sum(values[rng.randrange(length)] for _ in range(length)) / length
        for _ in range(100000)
    ]
    means.sort()
    return means[999]


def evaluate_fake_ledger(
    ledger: Sequence[Mapping[str, Any]], schedule: Mapping[str, Any]
) -> dict[str, Any]:
    validate_fake_ledger(ledger, schedule)
    pair_records = ledger[32:]
    by_base: dict[str, dict[str, list[float]]] = {}
    identity_by_base: dict[str, int] = {}
    goal_by_base: dict[str, int] = {}
    synthetic_changed = 0
    mask_identity_specific: set[str] = set()
    mask_goal_specific: set[str] = set()
    swap_identity_follow: set[str] = set()
    swap_goal_follow: set[str] = set()
    for record in pair_records:
        contrast = str(record["contrast"])
        observations = {
            item["condition"]: _finite_number(
                item["target_alignment_margin"], "target alignment margin"
            )
            for item in record["observations"]
        }
        delta = observations[contrast] - observations["wrapper_zero"]
        base_case = str(record["base_case_id"])
        by_base.setdefault(base_case, {}).setdefault(contrast, []).append(delta)
        identity_by_base[base_case] = int(record["identity_index"])
        goal_by_base[base_case] = int(record["goal_index"])
        if contrast == "mask_identity":
            if (
                _finite_number(record["identity_margin_delta"], "identity delta")
                < 0.0
                and _finite_number(record["goal_margin_delta"], "goal delta") > 0.0
            ):
                mask_identity_specific.add(base_case)
        elif contrast == "mask_goal":
            if (
                _finite_number(record["goal_margin_delta"], "goal delta") < 0.0
                and _finite_number(record["identity_margin_delta"], "identity delta")
                > 0.0
            ):
                mask_goal_specific.add(base_case)
        elif contrast == "swap_identity" and record["swap_identity_follows"] is True:
            swap_identity_follow.add(base_case)
        elif contrast == "swap_goal" and record["swap_goal_follows"] is True:
            swap_goal_follow.add(base_case)
        elif contrast == "synthetic_active":
            if record["output_changed"] is True and record["target_layer_applications"] == 1:
                synthetic_changed += 1
    if len(by_base) != 16 or any(
        set(contrasts) != set(CONTRASTS)
        or any(len(values) != 4 for values in contrasts.values())
        for contrasts in by_base.values()
    ):
        raise ValueError("D9-B base-case rotation aggregation is incomplete")
    base_ids = sorted(by_base)
    active = [statistics.fmean(by_base[key]["active_true"]) for key in base_ids]
    true_minus_random = [
        statistics.fmean(by_base[key]["active_true"])
        - statistics.fmean(by_base[key]["matched_random"])
        for key in base_ids
    ]
    positive_by_identity = [
        sum(value > 0.0 for key, value in zip(base_ids, active) if identity_by_base[key] == i)
        for i in range(4)
    ]
    positive_by_goal = [
        sum(value > 0.0 for key, value in zip(base_ids, active) if goal_by_base[key] == i)
        for i in range(4)
    ]
    metrics = {
        "active_minus_zero_mean": statistics.fmean(active),
        "active_minus_zero_lb99": _bootstrap_lower_bound(active, 29083102),
        "positive_base_cases": sum(value > 0.0 for value in active),
        "identity_level_min_positive": min(positive_by_identity),
        "goal_level_min_positive": min(positive_by_goal),
        "true_minus_random_lb99": _bootstrap_lower_bound(
            true_minus_random, 29083103
        ),
        "mask_identity_specific_count": len(mask_identity_specific),
        "mask_goal_specific_count": len(mask_goal_specific),
        "swap_identity_follow_count": len(swap_identity_follow),
        "swap_goal_follow_count": len(swap_goal_follow),
        "synthetic_active_changed_fixture_count": synthetic_changed,
    }
    decision = evaluate_candidate(metrics)
    return {
        "valid": True,
        "metrics": metrics,
        "checks": decision["checks"],
        "all_gates_pass": decision["all_gates_pass"],
        "decision": decision["decision"],
        "self_effect_conclusion": False,
    }


def _expect_failure(action: Any) -> bool:
    try:
        action()
    except (TypeError, ValueError, PermissionError):
        return True
    return False


def run_fake_acceptance(schedule: Mapping[str, Any]) -> dict[str, Any]:
    supported = make_fake_ledger(schedule, "field_specific_candidate")
    route_only = make_fake_ledger(schedule, "wrapper_route_only")
    nonspecific = make_fake_ledger(schedule, "nonspecific_active_or_random")
    supported_result = evaluate_fake_ledger(supported, schedule)
    route_result = evaluate_fake_ledger(route_only, schedule)
    nonspecific_result = evaluate_fake_ledger(nonspecific, schedule)
    missing = copy.deepcopy(supported[:-1])
    duplicate = copy.deepcopy(supported)
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    reordered = copy.deepcopy(supported)
    reordered[32], reordered[33] = reordered[33], reordered[32]
    public_route = copy.deepcopy(supported)
    public_route[32]["route"] = "public"
    nonfinite = copy.deepcopy(supported)
    nonfinite[32]["observations"][0]["target_alignment_margin"] = math.nan
    phase_leak = copy.deepcopy(supported)
    phase_leak[0]["fixture_id"] = phase_leak[32]["fixture_id"]
    mutated_order = copy.deepcopy(supported)
    mutated_order[32]["condition_order"].reverse()
    checks = {
        "field_specific_candidate_passes_all_frozen_gates": supported_result[
            "all_gates_pass"
        ],
        "wrapper_route_only_fails_primary": not route_result["all_gates_pass"],
        "nonspecific_active_or_random_fails_specificity": not nonspecific_result[
            "all_gates_pass"
        ],
        "all_scenarios_forbid_self_effect_conclusion": not any(
            result["self_effect_conclusion"]
            for result in (supported_result, route_result, nonspecific_result)
        ),
        "missing_record_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(missing, schedule)
        ),
        "duplicate_record_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(duplicate, schedule)
        ),
        "reordered_record_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(reordered, schedule)
        ),
        "public_route_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(public_route, schedule)
        ),
        "nonfinite_value_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(nonfinite, schedule)
        ),
        "calibration_heldout_phase_leakage_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(phase_leak, schedule)
        ),
        "condition_order_mutation_rejected": _expect_failure(
            lambda: evaluate_fake_ledger(mutated_order, schedule)
        ),
        "inputs_unchanged": supported
        == make_fake_ledger(schedule, "field_specific_candidate"),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "cases": {
            "field_specific_candidate": supported_result,
            "wrapper_route_only": route_result,
            "nonspecific_active_or_random": nonspecific_result,
        },
        "counts": {
            "calibration_records": 32,
            "heldout_fixtures": 64,
            "base_cases": 16,
            "pair_records": 448,
            "ledger_records": 480,
            "future_forward_calls": 928,
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
        raise PermissionError("D9-B config path is not frozen")
    config = _object(supplied, "contract config")
    config_checks = validate_contract_config(config)
    design = _object(root / DESIGN_RELATIVE_PATH, "D9-A design")
    design_checks = validate_design(design)
    expanded_fixtures = expand_fixtures(design)
    expanded_schedule = expand_schedule(design, expanded_fixtures)
    calibration = _object(root / CALIBRATION_RELATIVE_PATH, "calibration manifest")
    heldout = _object(root / HELDOUT_RELATIVE_PATH, "heldout manifest")
    schedule = _object(root / SCHEDULE_RELATIVE_PATH, "schedule manifest")
    endpoint = _object(root / ENDPOINT_RELATIVE_PATH, "endpoint manifest")
    manifest_hash_checks = {
        CALIBRATION_RELATIVE_PATH: sha256_file(root / CALIBRATION_RELATIVE_PATH)
        == CALIBRATION_SHA256,
        HELDOUT_RELATIVE_PATH: sha256_file(root / HELDOUT_RELATIVE_PATH)
        == HELDOUT_SHA256,
        SCHEDULE_RELATIVE_PATH: sha256_file(root / SCHEDULE_RELATIVE_PATH)
        == SCHEDULE_SHA256,
        ENDPOINT_RELATIVE_PATH: sha256_file(root / ENDPOINT_RELATIVE_PATH)
        == ENDPOINT_SHA256,
    }
    manifest_checks = {
        "calibration": validate_calibration_manifest(
            calibration, expanded_fixtures
        ),
        "heldout": validate_heldout_manifest(heldout, expanded_fixtures),
        "schedule": validate_schedule_manifest(schedule, expanded_schedule),
        "endpoint": validate_endpoint_manifest(endpoint, design),
    }
    fake_acceptance = run_fake_acceptance(schedule)
    namespaces = list(design["namespaces"].values())
    materialized_manifest_paths = {
        CALIBRATION_RELATIVE_PATH,
        HELDOUT_RELATIVE_PATH,
        SCHEDULE_RELATIVE_PATH,
        ENDPOINT_RELATIVE_PATH,
    }
    projection_contract_path = design["namespaces"][
        "projection_contract_future_path"
    ]
    projection_contract_materialized = (root / projection_contract_path).is_file()
    if projection_contract_materialized:
        materialized_manifest_paths.add(projection_contract_path)
        materialized_manifest_paths.add(
            design["namespaces"]["authorization_schema_future_path"]
        )
    future_artifacts_absent = all(
        not (root / value).exists()
        for value in namespaces
        if value not in materialized_manifest_paths
    )
    checks = {
        "config_valid": all(config_checks.values()),
        "d9a_design_valid": all(design_checks.values()),
        "four_manifest_hashes_frozen": all(manifest_hash_checks.values()),
        "four_manifests_valid": all(
            all(group.values()) for group in manifest_checks.values()
        ),
        "fake_acceptance_valid": fake_acceptance["valid"],
        "calibration_heldout_schedule_commitments_preserved": calibration[
            "expanded_commitment_sha256"
        ]
        == CALIBRATION_COMMITMENT
        and heldout["expanded_commitment_sha256"] == HELDOUT_COMMITMENT
        and schedule["expanded_commitment_sha256"] == SCHEDULE_COMMITMENT,
        "same_wrapper_only_and_no_public_scoring": all(
            block["route"] == "persistent_wrapper"
            for block in schedule["heldout_pair_blocks"]
        )
        and endpoint["public_route_allowed"] is False,
        "counts_exact": len(calibration["fixtures"]) == 32
        and len(heldout["fixtures"]) == 64
        and len(schedule["heldout_pair_blocks"]) == 448,
        "fourteen_namespaces_unique": len(namespaces) == 14
        and len(namespaces) == len(set(namespaces)),
        "authorization_claim_and_output_absent": future_artifacts_absent,
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "D9-B report failed closed: "
            + ", ".join(name for name, valid in checks.items() if not valid)
        )
    report: dict[str, Any] = {
        "report_version": CONTRACT_VERSION,
        "status": "d9b_deterministic_manifests_and_fake_endpoint_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "design_checks": design_checks,
        "manifest_hash_checks": manifest_hash_checks,
        "manifest_checks": manifest_checks,
        "fake_acceptance": fake_acceptance,
        "next_gate": NEXT_GATE,
        "safety": {
            "projection_contract_implemented": projection_contract_materialized,
            "real_projection_constructed": False,
            "installed_source_probed": False,
            "real_runner_modified": False,
            "execution_entry_implemented": (
                root
                / "configs/development/self_model_v0_1_d9c_projection_entry.json"
            ).is_file(),
            "authorization_created": False,
            "execution_claim_created": False,
            "output_created": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d9c_authorized": projection_contract_materialized,
            "d9d_real_execution_authorized": False,
            "d8c_or_historical_rerun": False,
            "d7d_authorized": False,
            "d7e_authorized": False,
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
