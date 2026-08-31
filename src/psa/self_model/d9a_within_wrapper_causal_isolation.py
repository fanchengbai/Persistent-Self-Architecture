from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d8_numerical_identifiability_design import (
    expand_fixtures as expand_d8_fixtures,
)


DESIGN_VERSION = "0.1-self-model-d9a-within-wrapper-causal-isolation-draft"
CONFIG_RELATIVE_PATH = (
    "configs/preregistration/"
    "self_model_v0_1_d9a_within_wrapper_causal_isolation.draft.json"
)
D8_CONFIG_RELATIVE_PATH = (
    "configs/preregistration/self_model_v0_1_d8_numerical_identifiability.draft.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D9-A within-wrapper causal isolation独立研究路线与无模型预注册设计；"
    "只允许定义在同一persistent wrapper路径内比较zero与未来active、以隔离D8-C已确认路径漂移的新研究问题，"
    "并冻结全新token/fixture/seed、训练与held-out评估分离、配对反平衡、synthetic active正控制、"
    "zero/mask/swap/random因果对照、通过标准及全新authorization/claim/output命名空间；"
    "D8-C结果只作路线依据，不复用其fixture、claim或结果作为新实验数据。"
    "本轮不实现或构造真实projection、不探测installed source、不修改真实runner、不导入RWKV/Torch、"
    "不访问权重、不加载或执行模型；不授权D9真实执行、D8-C或历史重跑、D7-D/D7-E、正式测试集、"
    "Self效果结论、Self Updater、raw-original路线或自动重跑。"
)
CLASSIFICATION = (
    "d9a_independent_within_wrapper_causal_isolation_preregistration_"
    "frozen_unimplemented_unrun"
)
NEXT_GATE = (
    "owner_reviews_d9a_design_then_separate_d9b_no_model_manifest_confirmation"
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
TOKEN_LENGTHS = (3, 5, 7, 9)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d9a_within_wrapper_causal_isolation.md",
    "scripts/verify_self_model_v0_1_d9a_within_wrapper_causal_isolation.py",
    "src/psa/self_model/d9a_within_wrapper_causal_isolation.py",
    "tests/test_self_model_d9a_within_wrapper_causal_isolation.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D9-A {label} must be an object")
    return value


def _project_path(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise PermissionError(f"D9-A {label} path is not frozen")
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise PermissionError(f"D9-A {label} escapes project root")
    return resolved


def _derive_unique_tokens(
    *, phase: str, seed: str, identity: str, count: int, used: set[int]
) -> list[int]:
    tokens: list[int] = []
    for position in range(count):
        nonce = 0
        while True:
            material = (
                f"d9a|{phase}|{seed}|{identity}|{position}|{nonce}"
            ).encode("utf-8")
            candidate = 1024 + int.from_bytes(
                hashlib.sha256(material).digest()[:8], "big"
            ) % 58977
            if candidate not in used:
                used.add(candidate)
                tokens.append(candidate)
                break
            nonce += 1
    return tokens


def expand_fixtures(config: Mapping[str, Any]) -> dict[str, Any]:
    calibration = config["calibration_design"]
    heldout = config["heldout_design"]
    used: set[int] = set()
    calibration_fixtures: list[dict[str, Any]] = []
    heldout_fixtures: list[dict[str, Any]] = []

    calibration_index = 0
    for identity_index in range(4):
        for goal_index in range(4):
            for replicate in range(2):
                calibration_index += 1
                fixture_id = f"d9cal-{calibration_index:03d}"
                token_count = TOKEN_LENGTHS[
                    (identity_index + goal_index + replicate) % len(TOKEN_LENGTHS)
                ]
                calibration_fixtures.append(
                    {
                        "fixture_id": fixture_id,
                        "phase": "calibration",
                        "identity_index": identity_index,
                        "goal_index": goal_index,
                        "replicate": replicate + 1,
                        "execution_path": "forward_seq",
                        "token_ids": _derive_unique_tokens(
                            phase="calibration",
                            seed=calibration["seed"],
                            identity=fixture_id,
                            count=token_count,
                            used=used,
                        ),
                        "full_output": bool(replicate),
                        "heldout_scored": False,
                    }
                )

    heldout_index = 0
    code_labels = ("A", "B", "C", "D")
    for identity_index in range(4):
        for goal_index in range(4):
            base_case_index = identity_index * 4 + goal_index
            base_case_id = f"d9case-{base_case_index + 1:02d}"
            content_token_count = TOKEN_LENGTHS[
                (identity_index + goal_index + 1) % len(TOKEN_LENGTHS)
            ]
            content_token_ids = _derive_unique_tokens(
                phase="heldout-content",
                seed=heldout["seed"],
                identity=base_case_id,
                count=content_token_count,
                used=used,
            )
            for rotation in range(4):
                heldout_index += 1
                fixture_id = f"d9hold-{heldout_index:03d}"
                rotation_token_ids = _derive_unique_tokens(
                    phase="heldout-rotation",
                    seed=heldout["seed"],
                    identity=fixture_id,
                    count=1,
                    used=used,
                )
                semantic_target_index = (identity_index * 3 + goal_index) % 4
                heldout_fixtures.append(
                    {
                        "fixture_id": fixture_id,
                        "phase": "heldout",
                        "base_case_id": base_case_id,
                        "identity_index": identity_index,
                        "goal_index": goal_index,
                        "code_rotation": rotation,
                        "target_code": code_labels[
                            (semantic_target_index + rotation) % 4
                        ],
                        "execution_path": "forward_seq",
                        "content_token_ids": list(content_token_ids),
                        "rotation_code_token_ids": rotation_token_ids,
                        "token_ids": list(content_token_ids) + rotation_token_ids,
                        "full_output": bool(rotation % 2),
                        "state_input": heldout["state_input"],
                        "heldout_scored": True,
                    }
                )

    calibration_manifest: dict[str, Any] = {
        "namespace": calibration["namespace"],
        "seed": calibration["seed"],
        "fixtures": calibration_fixtures,
    }
    calibration_manifest["commitment_sha256"] = sha256_json(calibration_manifest)
    heldout_manifest: dict[str, Any] = {
        "namespace": heldout["namespace"],
        "seed": heldout["seed"],
        "fixtures": heldout_fixtures,
    }
    heldout_manifest["commitment_sha256"] = sha256_json(heldout_manifest)
    return {
        "calibration_manifest": calibration_manifest,
        "heldout_manifest": heldout_manifest,
    }


def expand_schedule(
    config: Mapping[str, Any], fixtures: Mapping[str, Any]
) -> dict[str, Any]:
    schedule = config["schedule_design"]
    calibration_calls = [
        {
            "call_id": f"{fixture['fixture_id']}-capture",
            "fixture_id": fixture["fixture_id"],
            "phase": "calibration",
            "route": "persistent_wrapper_capture",
            "heldout_scored": False,
        }
        for fixture in fixtures["calibration_manifest"]["fixtures"]
    ]
    seeded_contrast_order = tuple(
        sorted(
            CONTRASTS,
            key=lambda condition: hashlib.sha256(
                f"{schedule['seed']}|{condition}".encode("utf-8")
            ).hexdigest(),
        )
    )
    pair_blocks: list[dict[str, Any]] = []
    for fixture_index, fixture in enumerate(
        fixtures["heldout_manifest"]["fixtures"]
    ):
        offset = fixture_index % len(seeded_contrast_order)
        rotated = seeded_contrast_order[offset:] + seeded_contrast_order[:offset]
        for position, contrast in enumerate(rotated, start=1):
            contrast_index = CONTRASTS.index(contrast)
            zero_first = (fixture_index + contrast_index) % 2 == 0
            condition_order = (
                ["wrapper_zero", contrast]
                if zero_first
                else [contrast, "wrapper_zero"]
            )
            pair_blocks.append(
                {
                    "pair_block_id": f"{fixture['fixture_id']}-{contrast}",
                    "fixture_id": fixture["fixture_id"],
                    "base_case_id": fixture["base_case_id"],
                    "identity_index": fixture["identity_index"],
                    "goal_index": fixture["goal_index"],
                    "code_rotation": fixture["code_rotation"],
                    "contrast": contrast,
                    "latin_position": position,
                    "pair_order": "zero_first" if zero_first else "condition_first",
                    "condition_order": condition_order,
                    "route": "persistent_wrapper",
                    "source_state_contract": (
                        "same_fresh_clone_of_fixture_prebuilt_zero_state_for_both_pair_calls"
                    ),
                    "heldout_scored": contrast != "synthetic_active",
                    "synthetic_positive_control": contrast == "synthetic_active",
                }
            )
    payload: dict[str, Any] = {
        "namespace": schedule["namespace"],
        "seed": schedule["seed"],
        "seeded_contrast_order": list(seeded_contrast_order),
        "calibration_calls": calibration_calls,
        "heldout_pair_blocks": pair_blocks,
    }
    payload["commitment_sha256"] = sha256_json(payload)
    return payload


def _namespace_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    values = list(config["namespaces"].values())
    checks = {
        "fourteen_namespaces_frozen": len(values) == 14,
        "all_namespaces_unique": len(values) == len(set(values)),
        "all_namespaces_are_d9": all("d9" in value.lower() for value in values),
        "no_d8_or_d7_namespace_reused": not any(
            fragment in value.lower()
            for value in values
            for fragment in ("d8_real", "d8c", "d7c", "d6d")
        ),
        "authorization_claim_and_output_are_separate": len(
            {
                config["namespaces"]["authorization_future_path"],
                config["namespaces"]["claim_future_path"],
                config["namespaces"]["output_future_dir"],
            }
        )
        == 3,
    }
    return {"valid": all(checks.values()), "checks": checks, "values": values}


def validate_design(config: Mapping[str, Any]) -> dict[str, bool]:
    historical = config.get("historical_boundary", {})
    calibration = config.get("calibration_design", {})
    heldout = config.get("heldout_design", {})
    conditions = config.get("causal_conditions", {})
    schedule = config.get("schedule_design", {})
    determinism = config.get("determinism_policy", {})
    endpoint = config.get("endpoint_contract", {})
    gates = config.get("gate_sequence", [])
    authority = config.get("authority", {})
    fixtures = expand_fixtures(config)
    expanded_schedule = expand_schedule(config, fixtures)
    checks = {
        "identity_exact": config.get("design_version") == DESIGN_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D9-A_within_wrapper_causal_isolation_preregistration"
        and config.get("status")
        == "preregistration_design_frozen_unimplemented_unrun"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "research_question_is_within_wrapper": config.get("research_question")
        == (
            "Within one persistent wrapper path, does a frozen future active Self "
            "projection produce held-out, field-specific causal changes beyond "
            "wrapper-zero and matched-random controls without using public-versus-wrapper "
            "differences?"
        ),
        "d8c_is_rationale_and_nonreuse_only": historical.get("d8c_use")
        == "route_confound_rationale_and_nonreuse_audit_only"
        and historical.get("d8c_execution_commit")
        == "e0ab61a58394e6eaef2567aa3a988afa6e47738c"
        and historical.get("d8c_claim_sha256")
        == "854036308665a3d2770a134bbf8d9ae218d7e0478756f4df4963e977cb1705db"
        and historical.get("d8c_report_digest_sha256")
        == "a0dad92b740e4776ba8c229b1c690751e50c63faed217c0d131defc3ff922ac5"
        and all(
            historical.get(field) is False
            for field in (
                "d8c_fixture_reused",
                "d8c_token_reused",
                "d8c_seed_reused",
                "d8c_claim_reused",
                "d8c_result_used_as_new_experiment_data",
                "d8c_rerun",
            )
        ),
        "calibration_contract_exact": calibration.get("fixture_count") == 32
        and calibration.get("identity_levels") == 4
        and calibration.get("goal_levels") == 4
        and calibration.get("replicates_per_identity_goal_cell") == 2
        and calibration.get("future_forward_calls") == 32
        and calibration.get("outputs_scored_for_heldout_endpoint") is False
        and calibration.get("heldout_access_before_projection_freeze") is False
        and calibration.get("expected_commitment_sha256")
        == fixtures["calibration_manifest"]["commitment_sha256"],
        "heldout_contract_exact": heldout.get("base_case_count") == 16
        and heldout.get("code_rotations_per_base_case") == 4
        and heldout.get("fixture_count") == 64
        and heldout.get("state_none_used") is False
        and heldout.get("rotation_content_contract")
        == (
            "four_rotations_share_identical_base_content_tokens_and_use_"
            "distinct_rotation_code_tokens"
        )
        and heldout.get("expected_commitment_sha256")
        == fixtures["heldout_manifest"]["commitment_sha256"],
        "within_wrapper_conditions_exact": conditions.get("wrapper_path_only")
        is True
        and conditions.get("public_route_scored_or_used") is False
        and conditions.get("zero_condition") == "wrapper_zero"
        and conditions.get("contrasts") == list(CONTRASTS)
        and conditions.get("real_projection_required_in_current_stage") is False,
        "paired_schedule_exact": schedule.get("pair_order")
        == ["zero_first", "condition_first"]
        and schedule.get("pair_orders_exactly_balanced_per_contrast") is True
        and schedule.get("pairs_per_heldout_fixture") == 7
        and schedule.get("calls_per_pair") == 2
        and schedule.get("heldout_pair_count") == 448
        and schedule.get("heldout_forward_calls") == 896
        and schedule.get("calibration_forward_calls") == 32
        and schedule.get("total_future_forward_calls") == 928
        and schedule.get("adaptive_retry_allowed") is False
        and schedule.get("expected_commitment_sha256")
        == expanded_schedule["commitment_sha256"],
        "determinism_exact_and_new": determinism.get("policy_seed") == 29083101
        and determinism.get("bootstrap_seed") == 29083102
        and determinism.get("environment")
        == {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "29083101",
            "RWKV_DE_VERSION": "unset",
        }
        and determinism.get("torch_use_deterministic_algorithms") is True
        and determinism.get("torch_deterministic_warn_only") is False
        and determinism.get("cudnn_deterministic") is True
        and determinism.get("cudnn_benchmark") is False
        and determinism.get("cuda_matmul_allow_tf32") is False
        and determinism.get("cudnn_allow_tf32") is False
        and determinism.get("float32_matmul_precision") == "highest",
        "endpoint_is_field_specific_and_nonself": endpoint.get("analysis_unit")
        == "sixteen_identity_goal_base_cases_after_four_rotation_label_marginalization"
        and endpoint.get("primary_estimand")
        == "mean_active_true_minus_wrapper_zero_target_alignment_margin"
        and endpoint.get("bootstrap_resamples") == 100000
        and endpoint.get("positive_base_case_requirement")
        == "at_least_13_of_16_strictly_positive"
        and endpoint.get("matched_random_specificity")
        == "active_true_minus_matched_random_99_percent_lower_bound_greater_than_zero"
        and endpoint.get("all_gates_required") is True
        and endpoint.get("formal_test_set") is False
        and endpoint.get("self_effect_conclusion_allowed") is False,
        "four_gates_separate_and_closed_after_d9a": [
            gate.get("gate_id") for gate in gates
        ]
        == ["D9-A", "D9-B", "D9-C", "D9-D"]
        and gates[0].get("current_authority") is True
        and gates[0].get("model_execution") is False
        and all(gate.get("current_authority") is False for gate in gates[1:])
        and gates[-1].get("model_execution") is True
        and gates[-1].get("future_forward_calls") == 928,
        "design_only_authority_exact": authority.get(
            "d9a_research_question_design_authorized"
        )
        is True
        and authority.get("d9a_preregistration_design_authorized") is True
        and authority.get("offline_design_verification_authorized") is True,
        "implementation_model_and_later_authority_closed": all(
            authority.get(field) is False
            for field in (
                "d9b_authorized",
                "d9c_authorized",
                "d9d_real_execution_authorized",
                "projection_implementation_authorized",
                "real_projection_construction_authorized",
                "installed_source_probe_authorized",
                "real_runner_modification_authorized",
                "execution_entry_implementation_authorized",
                "rwkv_import_authorized",
                "torch_import_authorized",
                "weights_access_authorized",
                "model_load_authorized",
                "model_execution_authorized",
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
        "next_gate_exact": config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D9-A design changed: " + ", ".join(failed))
    return checks


def _d8_tokens(root: Path) -> set[int]:
    d8_config = _object(root / D8_CONFIG_RELATIVE_PATH, "D8 design")
    expanded = expand_d8_fixtures(d8_config)
    return {
        token
        for fixture in (
            expanded["conditioning_fixtures"] + expanded["scored_fixtures"]
        )
        for token in fixture["token_ids"]
    }


def analyze_expansion_and_independence(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    fixtures = expand_fixtures(config)
    schedule = expand_schedule(config, fixtures)
    calibration = fixtures["calibration_manifest"]["fixtures"]
    heldout = fixtures["heldout_manifest"]["fixtures"]
    calibration_tokens = {
        token for fixture in calibration for token in fixture["token_ids"]
    }
    heldout_tokens = {token for fixture in heldout for token in fixture["token_ids"]}
    heldout_base_contents = {
        fixture["base_case_id"]: tuple(fixture["content_token_ids"])
        for fixture in heldout
    }
    heldout_content_definitions = {
        token for tokens in heldout_base_contents.values() for token in tokens
    }
    heldout_rotation_definitions = {
        token
        for fixture in heldout
        for token in fixture["rotation_code_token_ids"]
    }
    all_defined_tokens = (
        list(calibration_tokens)
        + list(heldout_content_definitions)
        + list(heldout_rotation_definitions)
    )
    d8_tokens = _d8_tokens(root)
    blocks = schedule["heldout_pair_blocks"]
    contrast_counts = {
        contrast: sum(block["contrast"] == contrast for block in blocks)
        for contrast in CONTRASTS
    }
    order_counts = {
        contrast: {
            order: sum(
                block["contrast"] == contrast and block["pair_order"] == order
                for block in blocks
            )
            for order in ("zero_first", "condition_first")
        }
        for contrast in CONTRASTS
    }
    position_counts = {
        contrast: {
            str(position): sum(
                block["contrast"] == contrast
                and block["latin_position"] == position
                for block in blocks
            )
            for position in range(1, 8)
        }
        for contrast in CONTRASTS
    }
    namespaces = _namespace_audit(config)
    checks = {
        "thirty_two_calibration_fixtures": len(calibration) == 32,
        "sixty_four_heldout_fixtures": len(heldout) == 64,
        "calibration_and_heldout_tokens_disjoint": calibration_tokens.isdisjoint(
            heldout_tokens
        ),
        "all_token_definitions_unique_except_intentional_rotation_content_reuse": len(
            all_defined_tokens
        )
        == len(set(all_defined_tokens)),
        "all_d9_tokens_in_frozen_range": all(
            1024 <= token <= 60000 for token in all_defined_tokens
        ),
        "all_d8_tokens_excluded": set(all_defined_tokens).isdisjoint(d8_tokens),
        "d8_fixture_schedule_and_policy_seeds_excluded": config[
            "calibration_design"
        ]["seed"]
        != "d8a-fixtures-b0876e51-20260831"
        and config["heldout_design"]["seed"]
        != "d8a-fixtures-b0876e51-20260831"
        and config["schedule_design"]["seed"]
        != "d8a-schedule-0f409cf2-20260831"
        and config["determinism_policy"]["policy_seed"] != 28083101,
        "sixteen_base_cases_four_rotations_each": len(
            {fixture["base_case_id"] for fixture in heldout}
        )
        == 16
        and all(
            sum(item["base_case_id"] == base for item in heldout) == 4
            for base in {fixture["base_case_id"] for fixture in heldout}
        ),
        "four_rotations_share_content_and_change_only_code_token": all(
            len(
                {
                    tuple(item["content_token_ids"])
                    for item in heldout
                    if item["base_case_id"] == base
                }
            )
            == 1
            and len(
                {
                    tuple(item["rotation_code_token_ids"])
                    for item in heldout
                    if item["base_case_id"] == base
                }
            )
            == 4
            for base in {fixture["base_case_id"] for fixture in heldout}
        ),
        "seven_contrasts_each_have_sixty_four_pairs": set(
            contrast_counts.values()
        )
        == {64},
        "pair_orders_exactly_thirty_two_each": all(
            set(counts.values()) == {32} for counts in order_counts.values()
        ),
        "latin_positions_differ_by_at_most_one": all(
            max(counts.values()) - min(counts.values()) <= 1
            and set(counts.values()).issubset({9, 10})
            for counts in position_counts.values()
        ),
        "all_scored_calls_remain_in_persistent_wrapper": all(
            block["route"] == "persistent_wrapper" for block in blocks
        ),
        "no_public_condition_present": all(
            "public" not in block["condition_order"] for block in blocks
        ),
        "four_hundred_forty_eight_pair_blocks": len(blocks) == 448,
        "nine_hundred_twenty_eight_future_calls": len(
            schedule["calibration_calls"]
        )
        + len(blocks) * 2
        == 928,
        "namespaces_are_new_and_valid": namespaces["valid"],
        "commitments_match": fixtures["calibration_manifest"][
            "commitment_sha256"
        ]
        == config["calibration_design"]["expected_commitment_sha256"]
        and fixtures["heldout_manifest"]["commitment_sha256"]
        == config["heldout_design"]["expected_commitment_sha256"]
        and schedule["commitment_sha256"]
        == config["schedule_design"]["expected_commitment_sha256"],
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "calibration_commitment_sha256": fixtures["calibration_manifest"][
            "commitment_sha256"
        ],
        "heldout_commitment_sha256": fixtures["heldout_manifest"][
            "commitment_sha256"
        ],
        "schedule_commitment_sha256": schedule["commitment_sha256"],
        "calibration_fixture_count": len(calibration),
        "heldout_fixture_count": len(heldout),
        "heldout_pair_count": len(blocks),
        "future_forward_call_count": len(schedule["calibration_calls"])
        + len(blocks) * 2,
        "unique_d9_token_count": len(set(all_defined_tokens)),
        "d8_token_count_excluded": len(d8_tokens),
        "contrast_counts": contrast_counts,
        "order_counts": order_counts,
        "position_counts": position_counts,
        "namespaces": namespaces,
    }


def evaluate_candidate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "active_minus_zero_mean",
        "active_minus_zero_lb99",
        "positive_base_cases",
        "identity_level_min_positive",
        "goal_level_min_positive",
        "true_minus_random_lb99",
        "mask_identity_specific_count",
        "mask_goal_specific_count",
        "swap_identity_follow_count",
        "swap_goal_follow_count",
        "synthetic_active_changed_fixture_count",
    }
    if set(metrics) != expected:
        raise ValueError("D9-A synthetic metrics are incomplete or changed")
    numeric_values = [float(value) for value in metrics.values()]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("D9-A synthetic metrics must be finite")
    checks = {
        "primary_mean_positive": float(metrics["active_minus_zero_mean"]) > 0.0,
        "primary_lb99_positive": float(metrics["active_minus_zero_lb99"]) > 0.0,
        "thirteen_of_sixteen_base_cases_positive": int(
            metrics["positive_base_cases"]
        )
        >= 13,
        "identity_level_consistency": int(metrics["identity_level_min_positive"])
        >= 3,
        "goal_level_consistency": int(metrics["goal_level_min_positive"]) >= 3,
        "true_beats_matched_random": float(metrics["true_minus_random_lb99"])
        > 0.0,
        "identity_mask_specific": int(metrics["mask_identity_specific_count"])
        >= 13,
        "goal_mask_specific": int(metrics["mask_goal_specific_count"]) >= 13,
        "identity_swap_follows": int(metrics["swap_identity_follow_count"])
        >= 12,
        "goal_swap_follows": int(metrics["swap_goal_follow_count"]) >= 12,
        "synthetic_positive_control_passes": int(
            metrics["synthetic_active_changed_fixture_count"]
        )
        >= 60,
    }
    return {
        "valid": True,
        "checks": checks,
        "all_gates_pass": all(checks.values()),
        "decision": (
            "within_wrapper_causal_specificity_candidate_supported_"
            "nonformal_nonself_engineering_only"
            if all(checks.values())
            else "revise_or_stop_without_self_effect_claim_or_rerun"
        ),
        "self_effect_conclusion": False,
    }


def run_synthetic_endpoint_review() -> dict[str, Any]:
    supported = {
        "active_minus_zero_mean": 0.20,
        "active_minus_zero_lb99": 0.08,
        "positive_base_cases": 15,
        "identity_level_min_positive": 3,
        "goal_level_min_positive": 3,
        "true_minus_random_lb99": 0.05,
        "mask_identity_specific_count": 14,
        "mask_goal_specific_count": 14,
        "swap_identity_follow_count": 13,
        "swap_goal_follow_count": 13,
        "synthetic_active_changed_fixture_count": 64,
    }
    route_only = dict(supported)
    route_only.update(
        {
            "active_minus_zero_mean": 0.0,
            "active_minus_zero_lb99": -0.01,
            "positive_base_cases": 8,
        }
    )
    nonspecific = dict(supported)
    nonspecific.update(
        {
            "true_minus_random_lb99": -0.02,
            "mask_identity_specific_count": 4,
            "mask_goal_specific_count": 4,
            "swap_identity_follow_count": 4,
            "swap_goal_follow_count": 4,
        }
    )
    cases = {
        "field_specific_candidate": evaluate_candidate(supported),
        "wrapper_route_only": evaluate_candidate(route_only),
        "nonspecific_active_or_random": evaluate_candidate(nonspecific),
    }
    checks = {
        "field_specific_case_passes_all_gates": cases[
            "field_specific_candidate"
        ]["all_gates_pass"],
        "route_only_case_fails": not cases["wrapper_route_only"][
            "all_gates_pass"
        ],
        "nonspecific_case_fails": not cases["nonspecific_active_or_random"][
            "all_gates_pass"
        ],
        "all_cases_forbid_self_effect_conclusion": all(
            not case["self_effect_conclusion"] for case in cases.values()
        ),
    }
    return {"valid": all(checks.values()), "checks": checks, "cases": cases}


def build_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    expected_config = (root / CONFIG_RELATIVE_PATH).resolve()
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected_config:
        raise PermissionError("D9-A design config path is not frozen")
    config = _object(supplied, "design")
    config_checks = validate_design(config)
    historical_lock_checks = {
        relative: sha256_file(_project_path(root, relative, "historical source lock"))
        == digest
        for relative, digest in config["frozen_historical_source_locks"].items()
    }
    if not all(historical_lock_checks.values()):
        failed = [path for path, valid in historical_lock_checks.items() if not valid]
        raise RuntimeError("D9-A historical source lock changed: " + ", ".join(failed))
    expansion = analyze_expansion_and_independence(config, root)
    synthetic = run_synthetic_endpoint_review()
    checks = {
        "config_valid": all(config_checks.values()),
        "historical_locks_valid": all(historical_lock_checks.values()),
        "expansion_and_independence_valid": expansion["valid"],
        "synthetic_endpoint_review_valid": synthetic["valid"],
        "same_wrapper_only": all(
            count == 64 for count in expansion["contrast_counts"].values()
        ),
        "calibration_heldout_separated": expansion["checks"][
            "calibration_and_heldout_tokens_disjoint"
        ],
        "d8_tokens_fixtures_seeds_claim_and_results_not_reused": expansion[
            "checks"
        ]["all_d8_tokens_excluded"]
        and expansion["checks"]["d8_fixture_schedule_and_policy_seeds_excluded"]
        and config["historical_boundary"]["d8c_claim_reused"] is False
        and config["historical_boundary"][
            "d8c_result_used_as_new_experiment_data"
        ]
        is False,
        "future_call_count_frozen": expansion["future_forward_call_count"] == 928,
        "all_later_gates_closed": not config["authority"]["d9b_authorized"]
        and not config["authority"]["d9c_authorized"]
        and not config["authority"]["d9d_real_execution_authorized"],
        "projection_not_implemented_or_constructed": not config["authority"][
            "projection_implementation_authorized"
        ]
        and not config["authority"]["real_projection_construction_authorized"],
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D9-A design verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": DESIGN_VERSION,
        "status": "d9a_within_wrapper_causal_isolation_preregistration_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "historical_source_lock_checks": historical_lock_checks,
        "expansion_and_independence": expansion,
        "synthetic_endpoint_review": synthetic,
        "frozen_decision_contract": {
            "positive": config["endpoint_contract"]["positive_decision"],
            "otherwise": config["endpoint_contract"]["otherwise_decision"],
            "public_route_allowed": False,
            "formal_test_set": False,
            "self_effect_conclusion_allowed": False,
        },
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
            "projection_implemented": False,
            "projection_constructed": False,
            "d9b_authorized": False,
            "d9c_authorized": False,
            "d9d_real_execution_authorized": False,
            "d8c_rerun": False,
            "historical_rerun": False,
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
