from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-self-model-d8a-numerical-identifiability-draft"
CONFIG_RELATIVE_PATH = (
    "configs/preregistration/self_model_v0_1_d8_numerical_identifiability.draft.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D8-A numerical identifiability与excess-drift独立研究路线的"
    "纯离线预注册设计；只允许定义新的研究问题、全新token/fixture/seed、public-public与"
    "wrapper-wrapper路径内重复性包络、public→wrapper与wrapper→public顺序平衡、预先冻结的"
    "确定性策略、差异中差异主要端点，以及全新的authorization/claim/output命名空间；不得复用"
    "D7-C的8个cell、claim或结果作为新实验数据。本轮不探测installed source、不修改真实runner、"
    "不实现执行入口、不导入RWKV/Torch、不访问权重、不加载或执行模型；不授权D8真实执行、"
    "D7-C修复或重跑、D7-D/D7-E、projection、正式测试集、Self效果结论、Self Updater、"
    "raw-original或自动重跑。"
)
CLASSIFICATION = (
    "d8a_independent_numerical_identifiability_and_excess_drift_"
    "preregistration_design_frozen_unimplemented_unrun"
)
NEXT_GATE = "owner_reviews_d8a_design_then_separate_d8b_no_model_implementation_confirmation"
PAIR_TYPES = (
    "public_public",
    "wrapper_wrapper",
    "public_wrapper",
    "wrapper_public",
)
PAIR_ROUTES = {
    "public_public": ("public", "public"),
    "wrapper_wrapper": ("wrapper_zero", "wrapper_zero"),
    "public_wrapper": ("public", "wrapper_zero"),
    "wrapper_public": ("wrapper_zero", "public"),
}
STRATA = (
    ("one_full_false", "forward_one", 1, False),
    ("one_full_true", "forward_one", 1, True),
    ("seq3_full_false", "forward_seq", 3, False),
    ("seq5_full_true", "forward_seq", 5, True),
)
FORBIDDEN_D7C_TOKENS = frozenset({187, 931, 2764})
D7C_SEED = 20260729
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d8_numerical_identifiability_design.md",
    "scripts/verify_self_model_v0_1_d8_numerical_identifiability_design.py",
    "src/psa/self_model/d8_numerical_identifiability_design.py",
    "tests/test_self_model_d8_numerical_identifiability_design.py",
)


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D8-A {label} must be an object")
    return value


def _project_path(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise PermissionError(f"D8-A {label} path is not frozen")
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise PermissionError(f"D8-A {label} escapes project root")
    return resolved


def _derive_unique_tokens(
    *, seed: str, identity: str, count: int, used: set[int]
) -> list[int]:
    tokens: list[int] = []
    for position in range(count):
        nonce = 0
        while True:
            material = f"{seed}|{identity}|{position}|{nonce}".encode("utf-8")
            candidate = 1024 + int.from_bytes(
                hashlib.sha256(material).digest()[:8], "big"
            ) % 58977
            if candidate not in used and candidate not in FORBIDDEN_D7C_TOKENS:
                used.add(candidate)
                tokens.append(candidate)
                break
            nonce += 1
    return tokens


def expand_fixtures(config: Mapping[str, Any]) -> dict[str, Any]:
    design = config["fixture_design"]
    used: set[int] = set()
    fixtures: list[dict[str, Any]] = []
    conditioning: list[dict[str, Any]] = []
    fixture_index = 0
    for stratum_index, (stratum, path, token_count, full_output) in enumerate(STRATA):
        conditioning_id = f"d8cond-{stratum_index + 1:02d}-{stratum}"
        conditioning.append(
            {
                "conditioning_id": conditioning_id,
                "stratum": stratum,
                "execution_path": path,
                "token_ids": _derive_unique_tokens(
                    seed=design["fixture_seed"],
                    identity=conditioning_id,
                    count=token_count,
                    used=used,
                ),
                "full_output": full_output,
                "route_order": ["public", "wrapper_zero"],
                "scored": False,
            }
        )
        for stratum_fixture_index in range(6):
            fixture_index += 1
            fixture_id = f"d8fx-{fixture_index:03d}-{stratum}"
            fixtures.append(
                {
                    "fixture_id": fixture_id,
                    "stratum": stratum,
                    "stratum_fixture_index": stratum_fixture_index,
                    "execution_path": path,
                    "token_ids": _derive_unique_tokens(
                        seed=design["fixture_seed"],
                        identity=fixture_id,
                        count=token_count,
                        used=used,
                    ),
                    "full_output": full_output,
                    "state_input": "fresh_clone_of_fixture_prebuilt_zero_state_for_every_call",
                }
            )
    payload = {
        "fixture_namespace": design["fixture_namespace"],
        "fixture_seed": design["fixture_seed"],
        "token_derivation": design["token_derivation"],
        "conditioning_fixtures": conditioning,
        "scored_fixtures": fixtures,
    }
    payload["fixture_commitment_sha256"] = sha256_json(payload)
    return payload


def expand_schedule(config: Mapping[str, Any], fixtures: Mapping[str, Any]) -> dict[str, Any]:
    schedule = config["schedule_design"]
    seeded_pair_order = tuple(
        sorted(
            PAIR_TYPES,
            key=lambda pair_type: hashlib.sha256(
                f"{schedule['schedule_seed']}|{pair_type}".encode("utf-8")
            ).hexdigest(),
        )
    )
    conditioning_calls: list[dict[str, Any]] = []
    for fixture in fixtures["conditioning_fixtures"]:
        for call_index, route in enumerate(fixture["route_order"], start=1):
            conditioning_calls.append(
                {
                    "call_id": f"{fixture['conditioning_id']}-call-{call_index}",
                    "conditioning_id": fixture["conditioning_id"],
                    "stratum": fixture["stratum"],
                    "route": route,
                    "scored": False,
                }
            )
    pair_blocks: list[dict[str, Any]] = []
    for fixture_index, fixture in enumerate(fixtures["scored_fixtures"]):
        for replicate in range(3):
            offset = (fixture_index + replicate) % len(seeded_pair_order)
            latin_order = seeded_pair_order[offset:] + seeded_pair_order[:offset]
            for position, pair_type in enumerate(latin_order):
                routes = PAIR_ROUTES[pair_type]
                pair_blocks.append(
                    {
                        "pair_block_id": (
                            f"{fixture['fixture_id']}-rep-{replicate + 1}-pos-{position + 1}"
                        ),
                        "fixture_id": fixture["fixture_id"],
                        "stratum": fixture["stratum"],
                        "replicate": replicate + 1,
                        "latin_position": position + 1,
                        "pair_type": pair_type,
                        "route_order": list(routes),
                        "source_state_contract": (
                            "same_fresh_clone_of_fixture_prebuilt_zero_state_for_both_calls"
                        ),
                        "scored": True,
                    }
                )
    payload = {
        "schedule_namespace": schedule["schedule_namespace"],
        "schedule_seed": schedule["schedule_seed"],
        "seeded_pair_order": list(seeded_pair_order),
        "conditioning_calls": conditioning_calls,
        "pair_blocks": pair_blocks,
    }
    payload["schedule_commitment_sha256"] = sha256_json(payload)
    return payload


def _namespace_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    values = list(config["namespaces"].values())
    forbidden_fragments = (
        "self_model_v0_1_d7_compatibility_v01",
        "self_model_v0_1_d7c",
        "d7c_real",
    )
    checks = {
        "eleven_namespaces_frozen": len(values) == 11,
        "all_namespaces_unique": len(values) == len(set(values)),
        "all_namespaces_are_d8": all("d8" in value.lower() for value in values),
        "no_d7c_namespace_reused": not any(
            fragment in value for value in values for fragment in forbidden_fragments
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
    fixture = config.get("fixture_design", {})
    schedule = config.get("schedule_design", {})
    determinism = config.get("determinism_policy", {})
    endpoint = config.get("distance_and_endpoint", {})
    gates = config.get("gate_sequence", [])
    authority = config.get("authority", {})
    expanded_fixtures = expand_fixtures(config)
    expanded_schedule = expand_schedule(config, expanded_fixtures)
    checks = {
        "identity_exact": config.get("design_version") == DESIGN_VERSION
        and config.get("stage")
        == "Self-Model-v0.1-D8-A_numerical_identifiability_and_excess_drift_preregistration"
        and config.get("status")
        == "preregistration_design_frozen_unimplemented_unrun"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "research_question_is_excess_cross_route_drift": config.get("research_question")
        == (
            "Does public-versus-instrumented output divergence exceed the within-public "
            "and within-instrumented numerical repeatability envelope under counterbalanced "
            "call order and a preregistered deterministic runtime policy?"
        ),
        "historical_d7c_is_boundary_only": historical.get("d7c_use")
        == "failure_boundary_and_nonreuse_constraint_only"
        and historical.get("d7c_execution_commit")
        == "665ac40026249fd8f1523aa2cae40486bb427d44"
        and historical.get("d7c_failure_status")
        == "d7c_real_public_semantics_compatibility_failed"
        and all(
            historical.get(field) is False
            for field in (
                "d7c_claim_reused",
                "d7c_eight_cells_reused",
                "d7c_token_fixture_reused",
                "d7c_seed_reused",
                "d7c_quantitative_results_used_as_new_experiment_data",
                "d7c_fix_or_rerun",
            )
        ),
        "new_fixture_contract_exact": fixture.get("fixture_seed")
        == "d8a-fixtures-b0876e51-20260831"
        and fixture.get("forbidden_d7c_token_ids") == [187, 931, 2764]
        and fixture.get("scored_fixture_count") == 24
        and fixture.get("state_input")
        == "fresh_clone_of_fixture_prebuilt_zero_state_for_every_call"
        and fixture.get("state_none_used") is False
        and fixture.get("conditioning_fixture_count") == 4
        and fixture.get("conditioning_calls_total") == 8
        and fixture.get("conditioning_outputs_scored") is False,
        "fixture_strata_exact": fixture.get("fixture_strata")
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
        "fixture_commitment_exact": fixture.get("expected_fixture_commitment_sha256")
        == expanded_fixtures["fixture_commitment_sha256"],
        "counterbalanced_schedule_exact": schedule.get("pair_types")
        == list(PAIR_TYPES)
        and schedule.get("replicates_per_pair_type_per_fixture") == 3
        and schedule.get("calls_per_pair") == 2
        and schedule.get("scored_pairs_per_fixture") == 12
        and schedule.get("scored_calls_per_fixture") == 24
        and schedule.get("scored_pair_count") == 288
        and schedule.get("scored_forward_calls") == 576
        and schedule.get("conditioning_forward_calls") == 8
        and schedule.get("total_future_forward_calls") == 584
        and schedule.get("cross_route_order_counterbalanced") is True
        and schedule.get("same_source_state_clone_for_both_calls_in_each_pair")
        is True,
        "schedule_commitment_exact": schedule.get("expected_schedule_commitment_sha256")
        == expanded_schedule["schedule_commitment_sha256"],
        "determinism_policy_exact": determinism.get("policy_seed") == 28083101
        and determinism.get("environment_must_be_applied_by_launcher_before_python_start")
        is True
        and determinism.get("runtime_flags_must_be_applied_before_model_load") is True
        and determinism.get("environment")
        == {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "28083101",
            "RWKV_DE_VERSION": "unset",
        }
        and determinism.get("torch_use_deterministic_algorithms") is True
        and determinism.get("torch_deterministic_warn_only") is False
        and determinism.get("cudnn_deterministic") is True
        and determinism.get("cudnn_benchmark") is False
        and determinism.get("cuda_matmul_allow_tf32") is False
        and determinism.get("cudnn_allow_tf32") is False
        and determinism.get("float32_matmul_precision") == "highest"
        and determinism.get("failure_action_if_policy_unavailable")
        == "stop_without_relaxing_policy_or_scoring_results",
        "primary_endpoint_is_conservative_difference_in_differences": endpoint.get(
            "within_route_envelope_per_fixture_replicate"
        )
        == "max(distance_public_public,distance_wrapper_wrapper)"
        and endpoint.get("symmetric_cross_route_floor_per_fixture_replicate")
        == "min(distance_public_wrapper,distance_wrapper_public)"
        and endpoint.get("replicate_excess_drift")
        == "symmetric_cross_route_floor_minus_within_route_envelope"
        and endpoint.get("fixture_excess_drift")
        == "median_of_three_replicate_excess_drift_values"
        and endpoint.get("primary_estimand")
        == "mean_fixture_excess_drift_across_24_fixtures",
        "primary_decision_and_inconclusive_rule_frozen": endpoint.get("primary_test")
        == "one_sided_cluster_bootstrap_99_percent_lower_bound_greater_than_zero"
        and endpoint.get("bootstrap_seed") == 28083102
        and endpoint.get("bootstrap_resamples") == 100000
        and endpoint.get("supporting_sign_requirement")
        == "at_least_21_of_24_fixture_excess_values_strictly_positive"
        and endpoint.get("stratum_consistency_requirement")
        == "at_least_5_of_6_fixture_excess_values_positive_in_each_of_four_strata"
        and endpoint.get("otherwise_decision")
        == "inconclusive_no_route_equivalence_claim"
        and endpoint.get("self_effect_conclusion_allowed") is False,
        "three_gates_separate_and_closed_after_d8a": [gate.get("gate_id") for gate in gates]
        == ["D8-A", "D8-B", "D8-C"]
        and gates[0].get("current_authority") is True
        and gates[0].get("model_execution") is False
        and all(gate.get("current_authority") is False for gate in gates[1:])
        and gates[1].get("model_execution") is False
        and gates[2].get("model_execution") is True
        and gates[2].get("future_forward_calls") == 584,
        "design_only_authority_exact": authority.get(
            "d8a_research_question_design_authorized"
        )
        is True
        and authority.get("d8a_preregistration_design_authorized") is True
        and authority.get("offline_design_verification_authorized") is True,
        "implementation_model_and_later_authority_closed": all(
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
                "d8b_authorized",
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
        "next_gate_exact": config.get("next_gate") == NEXT_GATE,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D8-A design changed: " + ", ".join(failed))
    return checks


def analyze_expansion_and_independence(config: Mapping[str, Any]) -> dict[str, Any]:
    fixtures = expand_fixtures(config)
    schedule = expand_schedule(config, fixtures)
    scored = fixtures["scored_fixtures"]
    conditioning = fixtures["conditioning_fixtures"]
    all_tokens = [
        token
        for fixture in conditioning + scored
        for token in fixture["token_ids"]
    ]
    blocks = schedule["pair_blocks"]
    position_counts = {
        pair_type: {
            str(position): sum(
                block["pair_type"] == pair_type
                and block["latin_position"] == position
                for block in blocks
            )
            for position in range(1, 5)
        }
        for pair_type in PAIR_TYPES
    }
    pair_counts = {
        pair_type: sum(block["pair_type"] == pair_type for block in blocks)
        for pair_type in PAIR_TYPES
    }
    stratum_counts = {
        stratum: sum(fixture["stratum"] == stratum for fixture in scored)
        for stratum, _, _, _ in STRATA
    }
    namespaces = _namespace_audit(config)
    checks = {
        "twenty_four_new_scored_fixtures": len(scored) == 24,
        "four_new_conditioning_fixtures": len(conditioning) == 4,
        "all_token_ids_globally_unique": len(all_tokens) == len(set(all_tokens)),
        "all_token_ids_in_frozen_range": all(1024 <= token <= 60000 for token in all_tokens),
        "d7c_token_ids_excluded": set(all_tokens).isdisjoint(FORBIDDEN_D7C_TOKENS),
        "d7c_seed_excluded": config["determinism_policy"]["policy_seed"] != D7C_SEED
        and config["distance_and_endpoint"]["bootstrap_seed"] != D7C_SEED,
        "six_fixtures_per_stratum": set(stratum_counts.values()) == {6},
        "four_pair_types_each_have_seventy_two_blocks": set(pair_counts.values()) == {72},
        "latin_positions_are_exactly_balanced": all(
            set(counts.values()) == {18} for counts in position_counts.values()
        ),
        "cross_route_orders_both_present": pair_counts["public_wrapper"] == 72
        and pair_counts["wrapper_public"] == 72,
        "two_hundred_eighty_eight_scored_pairs": len(blocks) == 288,
        "five_hundred_seventy_six_scored_calls": len(blocks) * 2 == 576,
        "eight_conditioning_calls": len(schedule["conditioning_calls"]) == 8,
        "namespaces_are_new_and_valid": namespaces["valid"],
        "fixture_and_schedule_commitments_match": fixtures[
            "fixture_commitment_sha256"
        ]
        == config["fixture_design"]["expected_fixture_commitment_sha256"]
        and schedule["schedule_commitment_sha256"]
        == config["schedule_design"]["expected_schedule_commitment_sha256"],
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "fixture_commitment_sha256": fixtures["fixture_commitment_sha256"],
        "schedule_commitment_sha256": schedule["schedule_commitment_sha256"],
        "stratum_counts": stratum_counts,
        "pair_counts": pair_counts,
        "latin_position_counts": position_counts,
        "unique_token_count": len(set(all_tokens)),
        "conditioning_call_count": len(schedule["conditioning_calls"]),
        "scored_pair_count": len(blocks),
        "scored_forward_call_count": len(blocks) * 2,
        "total_future_forward_call_count": len(schedule["conditioning_calls"])
        + len(blocks) * 2,
        "namespaces": namespaces,
    }


def excess_drift_from_distances(distances: Mapping[str, float]) -> dict[str, float]:
    if set(distances) != set(PAIR_TYPES):
        raise ValueError("D8-A abstract fixture requires exactly four pair distances")
    values = {name: float(value) for name, value in distances.items()}
    if any(value < 0.0 for value in values.values()):
        raise ValueError("D8-A distances must be nonnegative")
    within = max(values["public_public"], values["wrapper_wrapper"])
    cross = min(values["public_wrapper"], values["wrapper_public"])
    return {
        "within_route_envelope": within,
        "symmetric_cross_route_floor": cross,
        "excess_drift": cross - within,
        "order_interaction": abs(
            values["public_wrapper"] - values["wrapper_public"]
        ),
        "within_route_asymmetry": abs(
            values["public_public"] - values["wrapper_wrapper"]
        ),
    }


def run_synthetic_endpoint_review() -> dict[str, Any]:
    cases = {
        "route_specific_excess": {
            "public_public": 0.001,
            "wrapper_wrapper": 0.0015,
            "public_wrapper": 0.02,
            "wrapper_public": 0.018,
        },
        "public_first_order_only": {
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
    evaluated = {
        name: excess_drift_from_distances(distances)
        for name, distances in cases.items()
    }
    replicate_values = [0.1, 0.3, 0.2]
    checks = {
        "route_specific_case_positive": evaluated["route_specific_excess"][
            "excess_drift"
        ]
        > 0.0,
        "one_order_only_case_not_positive": evaluated["public_first_order_only"][
            "excess_drift"
        ]
        == 0.0,
        "shared_repeatability_case_not_positive": evaluated[
            "shared_background_repeatability"
        ]["excess_drift"]
        == 0.0,
        "median_replicate_rule_is_order_invariant": statistics.median(replicate_values)
        == statistics.median(reversed(replicate_values)),
        "secondary_metrics_do_not_change_primary_excess": evaluated[
            "public_first_order_only"
        ]["order_interaction"]
        > 0.0
        and evaluated["public_first_order_only"]["excess_drift"] == 0.0,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "cases": evaluated,
        "interpretation": (
            "the conservative cross-order floor detects only divergence present in both "
            "cross-route orders beyond the worst within-route repeatability distance"
        ),
    }


def build_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    expected_config = (root / CONFIG_RELATIVE_PATH).resolve()
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = (root / supplied).resolve()
    if supplied != expected_config:
        raise PermissionError("D8-A design config path is not frozen")
    config = _object(supplied, "design")
    config_checks = validate_design(config)
    lock_checks = {
        relative: sha256_file(_project_path(root, relative, "historical source lock"))
        == digest
        for relative, digest in config["frozen_historical_source_locks"].items()
    }
    if not all(lock_checks.values()):
        failed = [path for path, valid in lock_checks.items() if not valid]
        raise RuntimeError("D8-A historical source lock changed: " + ", ".join(failed))
    expansion = analyze_expansion_and_independence(config)
    endpoint_review = run_synthetic_endpoint_review()
    checks = {
        "config_valid": all(config_checks.values()),
        "historical_source_locks_valid": all(lock_checks.values()),
        "fixture_schedule_and_independence_valid": expansion["valid"],
        "synthetic_endpoint_review_valid": endpoint_review["valid"],
        "d7c_used_only_as_nonreuse_boundary": config["historical_boundary"][
            "d7c_use"
        ]
        == "failure_boundary_and_nonreuse_constraint_only"
        and not config["historical_boundary"][
            "d7c_quantitative_results_used_as_new_experiment_data"
        ],
        "within_route_envelopes_both_present": expansion["pair_counts"][
            "public_public"
        ]
        == 72
        and expansion["pair_counts"]["wrapper_wrapper"] == 72,
        "cross_route_order_counterbalanced": expansion["pair_counts"][
            "public_wrapper"
        ]
        == expansion["pair_counts"]["wrapper_public"]
        == 72,
        "determinism_policy_frozen_before_future_execution": config[
            "determinism_policy"
        ]["environment_must_be_applied_by_launcher_before_python_start"]
        and config["determinism_policy"]["runtime_flags_must_be_applied_before_model_load"]
        and config["determinism_policy"]["torch_use_deterministic_algorithms"],
        "future_forward_count_frozen": expansion["total_future_forward_call_count"]
        == 584,
        "d8b_and_d8c_not_authorized": not config["authority"]["d8b_authorized"]
        and not config["authority"]["d8c_real_execution_authorized"],
        "source_inventory_complete": all((root / path).is_file() for path in SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D8-A design verification failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": DESIGN_VERSION,
        "status": "d8a_numerical_identifiability_preregistration_design_verified",
        "valid": True,
        "development_only": True,
        "classification": CLASSIFICATION,
        "checks": checks,
        "config_checks": config_checks,
        "historical_source_lock_checks": lock_checks,
        "expansion_and_independence": expansion,
        "synthetic_endpoint_review": endpoint_review,
        "frozen_decision_contract": {
            "primary_positive": config["distance_and_endpoint"][
                "primary_decision_positive"
            ],
            "otherwise": config["distance_and_endpoint"]["otherwise_decision"],
            "equivalence_claim_allowed_on_nonpositive_result": False,
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
            "d8b_authorized": False,
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
