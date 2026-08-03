from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import NormalDist, mean, median, stdev
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.confirmatory.preflight import (
    EXPECTED_CORE_SET_DIGEST,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_FINAL_DIGEST,
    EXPECTED_MODEL_ID,
)
from psa.confirmatory.runner import (
    CONDITIONS,
    condition_evaluation_combo,
    condition_source_combo,
)
from psa.evaluation.contrasts import (
    argmax_combo,
    group_contrasts,
    joint_margin,
)
from psa.evaluation.resampling import (
    bca_mean_interval,
    equivalence_from_interval,
    holm_adjust,
    sign_flip_test,
)
from psa.preregistration import (
    verify_core_set_package,
    verify_final_preregistration_package,
)


EXPECTED_GROUP_COUNT = 320
EXPECTED_RAW_RECORD_COUNT = 40_960
EXPECTED_ANALYSIS_CONFIG_SHA256 = (
    "d97e01329ced3bb9d292d8223f9f105dfa5b88456e14d59d9db79a24b975b8ea"
)
COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))
PRIMARY_ENDPOINTS = (
    "E1_identity_transfer",
    "E2_goal_transfer",
    "E3_joint_binding",
)
ANALYSIS_SOURCE_FILES = (
    "configs/analysis/exp001_confirmatory_v1.json",
    "docs/exp001_confirmatory_analysis_plan.md",
    "src/psa/confirmatory/analysis.py",
    "src/psa/confirmatory/__init__.py",
    "src/psa/cli.py",
    "src/psa/evaluation/contrasts.py",
    "src/psa/evaluation/resampling.py",
    "scripts/analyze_exp001_confirmatory.sh",
    "tests/test_confirmatory_analysis.py",
)


Combo = tuple[int, int]
Scores = dict[Combo, float]


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _combo(value: Any, label: str) -> Combo:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two binary values")
    result = int(value[0]), int(value[1])
    if result not in COMBOS:
        raise ValueError(f"{label} is not an I x G combination")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires data")
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bca_statistic_interval(
    rows: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if len(rows) < 2:
        raise ValueError("BCa interval requires at least two groups")
    observed = float(statistic(rows))
    if not math.isfinite(observed):
        raise ValueError("observed statistic must be finite")
    rng = random.Random(seed)
    count = len(rows)
    boot = sorted(
        float(statistic([rows[rng.randrange(count)] for _ in range(count)]))
        for _ in range(replicates)
    )
    if not all(math.isfinite(value) for value in boot):
        raise ValueError("bootstrap statistic is not finite")
    normal = NormalDist()
    below = sum(value < observed for value in boot)
    proportion = min(1.0 - 1e-9, max(1e-9, below / replicates))
    z0 = normal.inv_cdf(proportion)
    jackknife = [
        float(statistic(list(rows[:index]) + list(rows[index + 1 :])))
        for index in range(count)
    ]
    jack_mean = mean(jackknife)
    deviations = [jack_mean - value for value in jackknife]
    numerator = sum(value**3 for value in deviations)
    denominator = sum(value**2 for value in deviations)
    acceleration = (
        numerator / (6.0 * denominator**1.5) if denominator > 0.0 else 0.0
    )
    alpha = (1.0 - confidence) / 2.0

    def adjusted(probability: float) -> float:
        z_alpha = normal.inv_cdf(probability)
        divisor = 1.0 - acceleration * (z0 + z_alpha)
        if abs(divisor) < 1e-12:
            return probability
        return normal.cdf(z0 + (z0 + z_alpha) / divisor)

    return _quantile(boot, adjusted(alpha)), _quantile(
        boot,
        adjusted(1.0 - alpha),
    )


def _describe(
    values: Sequence[float],
    *,
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = [float(value) for value in values]
    if len(data) < 2 or not all(math.isfinite(value) for value in data):
        raise ValueError("group statistic requires at least two finite values")
    interval = bca_mean_interval(
        data,
        confidence=float(bootstrap["confidence"]),
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    report = {
        "factorial_group_count": len(data),
        "mean": mean(data),
        "median": median(data),
        "standard_deviation": stdev(data),
        "q1": _quantile(data, 0.25),
        "q3": _quantile(data, 0.75),
        "iqr": _quantile(data, 0.75) - _quantile(data, 0.25),
        "confidence_interval": list(interval),
        "confidence": float(bootstrap["confidence"]),
        "interval_method": str(bootstrap["method"]),
    }
    if permutation is not None:
        report["raw_p_value"] = sign_flip_test(
            data,
            alternative=str(permutation["alternative"]),
            replicates=int(permutation["replicates"]),
            seed=int(permutation["seed"]),
        )
        report["test"] = str(permutation["method"])
    return report


def _semantic_condition_scores(
    group: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, dict[Combo, Scores]]:
    trials = group.get("trials")
    records = payload.get("records")
    if not isinstance(trials, list) or not isinstance(records, list):
        raise ValueError("group trials and raw records are required")
    trials_by_id = {
        trial["trial_id"]: trial
        for trial in trials
        if isinstance(trial, Mapping) and isinstance(trial.get("trial_id"), str)
    }
    if len(trials_by_id) != 16 or len(records) != 128:
        raise ValueError("each group requires 16 trials and 128 raw records")
    buckets: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("raw record must be an object")
        trial = trials_by_id.get(record.get("trial_id"))
        if trial is None:
            raise ValueError("raw record references an unknown trial")
        condition = str(record.get("condition"))
        if condition not in CONDITIONS:
            raise ValueError("raw record has an unknown condition")
        case_id = str(trial.get("semantic_case_id"))
        target = trial.get("target_fields")
        if not isinstance(target, Mapping):
            raise ValueError("trial target_fields are missing")
        query_combo = _combo(
            (target.get("identity"), target.get("goal")),
            "query target",
        )
        if _combo(record.get("query_target_combo"), "record query target") != query_combo:
            raise ValueError("raw record query target differs from Core Set")
        key = condition, case_id
        bucket = buckets.setdefault(
            key,
            {
                "query_combo": query_combo,
                "rotations": set(),
                "scores": defaultdict(list),
            },
        )
        if bucket["query_combo"] != query_combo:
            raise ValueError("semantic case has inconsistent targets")
        bucket["rotations"].add(int(trial["rotation_index"]))
        option_scores = record.get("option_scores")
        option_mapping = trial.get("option_mapping")
        if not isinstance(option_scores, Mapping) or not isinstance(option_mapping, list):
            raise ValueError("option scores or mapping are missing")
        for option in option_mapping:
            semantic_combo = _combo(
                (option.get("identity"), option.get("goal")),
                "option combo",
            )
            score = float(option_scores[option["code"]])
            if not math.isfinite(score):
                raise ValueError("option score is not finite")
            bucket["scores"][semantic_combo].append(score)

    result: dict[str, dict[Combo, Scores]] = {condition: {} for condition in CONDITIONS}
    for (condition, _), bucket in buckets.items():
        if bucket["rotations"] != {0, 1, 2, 3}:
            raise ValueError("semantic case does not contain a complete rotation")
        if set(bucket["scores"]) != set(COMBOS) or any(
            len(values) != 4 for values in bucket["scores"].values()
        ):
            raise ValueError("semantic score rotation is incomplete")
        query_combo = bucket["query_combo"]
        if query_combo in result[condition]:
            raise ValueError("duplicate semantic query combo within a condition")
        result[condition][query_combo] = {
            combo: mean(values) for combo, values in bucket["scores"].items()
        }
    if any(set(scores) != set(COMBOS) for scores in result.values()):
        raise ValueError("every condition must cover all four semantic targets")
    return result


def _condition_metrics(
    condition: str,
    scores_by_query: Mapping[Combo, Scores],
) -> dict[str, float]:
    margins = []
    joint = []
    identity = []
    goal = []
    for query_combo in COMBOS:
        scores = scores_by_query[query_combo]
        evaluation = condition_evaluation_combo(query_combo, condition)
        prediction = argmax_combo(scores)
        margins.append(joint_margin(scores, evaluation))
        joint.append(float(prediction == evaluation))
        identity.append(float(prediction[0] == evaluation[0]))
        goal.append(float(prediction[1] == evaluation[1]))
    return {
        "mean_joint_margin": mean(margins),
        "joint_accuracy": mean(joint),
        "identity_accuracy": mean(identity),
        "goal_accuracy": mean(goal),
    }


def analyze_confirmatory_group(
    group: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    scores = _semantic_condition_scores(group, payload)
    continuous = scores["continuous"]
    primary = group_contrasts(continuous)
    condition_metrics = {
        condition: _condition_metrics(condition, scores[condition])
        for condition in CONDITIONS
    }
    baseline_advantages = {}
    for condition in ("reset", "random_matched"):
        baseline_advantages[condition] = mean(
            joint_margin(continuous[combo], combo)
            - joint_margin(scores[condition][combo], combo)
            for combo in COMBOS
        )
    restored_errors = [
        abs(continuous[query][option] - scores["restored"][query][option])
        for query in COMBOS
        for option in COMBOS
    ]
    restored_matches = [
        argmax_combo(continuous[query]) == argmax_combo(scores["restored"][query])
        for query in COMBOS
    ]
    swap_reports = {}
    for condition in ("swapped_I", "swapped_G", "swapped_both"):
        source_scores: dict[Combo, Scores] = {}
        donor_advantages = []
        donor_correct = []
        for query in COMBOS:
            source = condition_source_combo(query, condition)
            if source is None or source in source_scores:
                raise ValueError("swap condition does not uniquely cover donor states")
            source_scores[source] = scores[condition][query]
            donor_advantages.append(
                joint_margin(scores[condition][query], source)
                - joint_margin(scores[condition][query], query)
            )
            donor_correct.append(float(argmax_combo(scores[condition][query]) == source))
        swap_reports[condition] = {
            "source_contrasts": group_contrasts(source_scores),
            "donor_over_query_joint_margin": mean(donor_advantages),
            "donor_joint_accuracy": mean(donor_correct),
        }
    return {
        "factorial_group_id": group["factorial_group_id"],
        "primary": primary,
        "condition_metrics": condition_metrics,
        "baseline_advantages": baseline_advantages,
        "restore_fidelity": {
            "option_score_max_abs_error": max(restored_errors),
            "semantic_argmax_match_rate": mean(float(value) for value in restored_matches),
        },
        "swap": swap_reports,
    }


def _endpoint_reports(
    groups: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fields = config["primary_endpoints"]
    bootstrap = config["bootstrap"]
    permutation = config["permutation"]
    reports = {}
    raw_p_values = []
    for endpoint in PRIMARY_ENDPOINTS:
        values = [float(group["primary"][fields[endpoint]]) for group in groups]
        report = _describe(values, bootstrap=bootstrap, permutation=permutation)
        report["estimand"] = fields[endpoint]
        report["sesoi"] = float(config["sesoi"][endpoint])
        report["equivalence_bounds"] = [-report["sesoi"], report["sesoi"]]
        report["equivalent_to_substantive_zero"] = equivalence_from_interval(
            tuple(report["confidence_interval"]),
            tuple(report["equivalence_bounds"]),
        )
        reports[endpoint] = report
        raw_p_values.append(report["raw_p_value"])
    adjusted = holm_adjust(raw_p_values)
    alpha = float(config["multiple_comparison"]["familywise_alpha"])
    for endpoint, adjusted_p in zip(PRIMARY_ENDPOINTS, adjusted, strict=True):
        report = reports[endpoint]
        report["holm_adjusted_p_value"] = adjusted_p
        report["endpoint_supported"] = bool(
            adjusted_p < alpha
            and report["mean"] >= report["sesoi"]
            and report["confidence_interval"][0] > 0.0
        )
    return reports


def _paired_ratio_report(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(zip(numerators, denominators, strict=True))
    denominator_mean = mean(denominators)
    if abs(denominator_mean) < 0.5:
        return {
            "reported": False,
            "reason": "prompt_visible_denominator_not_far_from_zero",
            "denominator_mean": denominator_mean,
        }

    def ratio(sample: Sequence[tuple[float, float]]) -> float:
        denominator = mean(row[1] for row in sample)
        if abs(denominator) < 1e-12:
            return float("nan")
        return mean(row[0] for row in sample) / denominator

    interval = _bca_statistic_interval(
        rows,
        ratio,
        confidence=float(config["bootstrap"]["confidence"]),
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    point = ratio(rows)
    threshold = float(config["sesoi"]["prompt_normalized_retention"])
    return {
        "reported": True,
        "point": point,
        "confidence_interval": list(interval),
        "denominator_mean": denominator_mean,
        "sesoi": threshold,
        "retention_supported": point >= threshold and interval[0] > 0.0,
    }


def summarize_confirmatory_groups(
    groups: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(groups) < 2:
        raise ValueError("confirmatory summary requires at least two groups")
    bootstrap = config["bootstrap"]
    permutation = config["permutation"]
    endpoints = _endpoint_reports(groups, config)
    specificity = {}
    for name, field in (
        ("identity", "identity_specificity"),
        ("goal", "goal_specificity"),
    ):
        report = _describe(
            [float(group["primary"][field]) for group in groups],
            bootstrap=bootstrap,
        )
        threshold = float(config["sesoi"][f"{name}_specificity"])
        report["sesoi"] = threshold
        report["supported"] = (
            report["mean"] >= threshold
            and report["confidence_interval"][0] > 0.0
        )
        specificity[name] = report
    baseline_reports = {}
    for condition in ("reset", "random_matched"):
        report = _describe(
            [float(group["baseline_advantages"][condition]) for group in groups],
            bootstrap=bootstrap,
        )
        threshold = float(config["sesoi"]["baseline_joint_margin_advantage"])
        report["sesoi"] = threshold
        report["supported"] = (
            report["mean"] >= threshold
            and report["confidence_interval"][0] > 0.0
        )
        baseline_reports[condition] = report
    condition_reports = {}
    for condition in CONDITIONS:
        condition_reports[condition] = {
            metric: _describe(
                [float(group["condition_metrics"][condition][metric]) for group in groups],
                bootstrap=bootstrap,
            )
            for metric in (
                "mean_joint_margin",
                "joint_accuracy",
                "identity_accuracy",
                "goal_accuracy",
            )
        }
    swap_reports = {}
    for condition, directional_field in (
        ("swapped_I", "identity_transfer"),
        ("swapped_G", "goal_transfer"),
        ("swapped_both", None),
    ):
        donor_advantage = _describe(
            [float(group["swap"][condition]["donor_over_query_joint_margin"]) for group in groups],
            bootstrap=bootstrap,
            permutation=permutation,
        )
        donor_accuracy = _describe(
            [float(group["swap"][condition]["donor_joint_accuracy"]) for group in groups],
            bootstrap=bootstrap,
        )
        report: dict[str, Any] = {
            "donor_over_query_joint_margin": donor_advantage,
            "donor_joint_accuracy": donor_accuracy,
        }
        if directional_field is not None:
            directional = _describe(
                [
                    float(group["swap"][condition]["source_contrasts"][directional_field])
                    for group in groups
                ],
                bootstrap=bootstrap,
            )
            report["directional_transfer"] = directional
            report["migration_supported"] = directional["confidence_interval"][0] > 0.0
        else:
            report["migration_supported"] = bool(
                donor_advantage["confidence_interval"][0] > 0.0
                and donor_accuracy["confidence_interval"][0] > 0.5
            )
        swap_reports[condition] = report
    restore_limit = float(
        config["engineering_gates"]["restored_option_score_max_abs_error"]
    )
    restore = {
        "worst_option_score_max_abs_error": max(
            float(group["restore_fidelity"]["option_score_max_abs_error"])
            for group in groups
        ),
        "minimum_semantic_argmax_match_rate": min(
            float(group["restore_fidelity"]["semantic_argmax_match_rate"])
            for group in groups
        ),
        "option_score_max_abs_error_limit": restore_limit,
    }
    restore["passed"] = bool(
        restore["worst_option_score_max_abs_error"] <= restore_limit
        and restore["minimum_semantic_argmax_match_rate"] == 1.0
    )
    prompt_capability = {
        key: condition_reports["prompt_visible"][key]
        for key in ("joint_accuracy", "identity_accuracy", "goal_accuracy")
    }
    prompt_capability["score_based_accuracy_thresholds_passed"] = bool(
        prompt_capability["joint_accuracy"]["confidence_interval"][0] >= 0.8
        and prompt_capability["identity_accuracy"]["confidence_interval"][0] >= 0.9
        and prompt_capability["goal_accuracy"]["confidence_interval"][0] >= 0.9
    )
    prompt_capability["generated_format_gate"] = "not_recorded_in_formal_raw_package"
    prompt_contrast_values = [
        group["prompt_visible_contrasts"]
        for group in groups
        if "prompt_visible_contrasts" in group
    ]
    if len(prompt_contrast_values) != len(groups):
        raise ValueError("prompt-visible group contrasts are missing")
    retention = {
        "identity": _paired_ratio_report(
            [float(group["primary"]["identity_transfer"]) for group in groups],
            [float(item["identity_transfer"]) for item in prompt_contrast_values],
            config=config,
        ),
        "goal": _paired_ratio_report(
            [float(group["primary"]["goal_transfer"]) for group in groups],
            [float(item["goal_transfer"]) for item in prompt_contrast_values],
            config=config,
        ),
    }
    primary_supported = all(
        endpoints[endpoint]["endpoint_supported"] for endpoint in PRIMARY_ENDPOINTS
    )
    specificity_supported = all(item["supported"] for item in specificity.values())
    baselines_supported = all(item["supported"] for item in baseline_reports.values())
    joint_accuracy = condition_reports["continuous"]["joint_accuracy"]
    single_variable_ceiling = _describe(
        [
            float(group["condition_metrics"]["continuous"]["joint_accuracy"])
            - 0.5
            for group in groups
        ],
        bootstrap=bootstrap,
        permutation=permutation,
    )
    single_variable_ceiling["reference_accuracy"] = 0.5
    single_variable_ceiling["minimum_advantage"] = 0.1
    single_variable_ceiling["covers_strategies"] = [
        "identity_only",
        "goal_only",
        "recent_variable",
        "fixed_answer_position_conservatively",
    ]
    single_variable_ceiling["supported"] = bool(
        single_variable_ceiling["raw_p_value"] < 0.05
        and single_variable_ceiling["mean"] >= 0.1
        and single_variable_ceiling["confidence_interval"][0] > 0.0
    )
    joint_binding_measured = bool(
        primary_supported
        and joint_accuracy["confidence_interval"][0]
        >= float(config["sesoi"]["joint_accuracy_lower_bound"])
        and single_variable_ceiling["supported"]
        and swap_reports["swapped_both"]["migration_supported"]
        and baselines_supported
    )
    gaps = dict(config["frozen_design_gaps"])
    if not restore["passed"]:
        allowed = "infrastructure_failure_no_semantic_interpretation"
        route = "hold_and_investigate_restore_fidelity_without_rerun"
    elif not primary_supported:
        allowed = "frozen_primary_claims_not_supported"
        route = "report_negative_or_inconclusive_confirmatory_result_no_rerun"
    elif not joint_binding_measured:
        allowed = "primary_transfer_detected_but_strong_joint_binding_not_supported"
        route = "report_weak_or_partial_persistence_no_rerun"
    else:
        allowed = (
            "measured_effects_pass_but_full_native_state_qualification_"
            "not_assessable_due_to_unmeasured_preregistered_controls"
        )
        route = "report_confirmatory_effects_and_protocol_gap_no_automatic_rerun"
    return {
        "primary_endpoints": endpoints,
        "specificity": specificity,
        "baseline_advantages": baseline_reports,
        "condition_metrics": condition_reports,
        "swap_migration": swap_reports,
        "restore_fidelity": restore,
        "prompt_visible_capability": prompt_capability,
        "prompt_normalized_retention": retention,
        "single_variable_strategy_ceiling": single_variable_ceiling,
        "measured_decisions": {
            "primary_endpoints_all_supported": primary_supported,
            "specificity_supported": specificity_supported,
            "reset_and_random_baselines_supported": baselines_supported,
            "joint_binding_measured_requirements_supported": joint_binding_measured,
        },
        "preregistered_but_unmeasured": gaps,
        "gate_2_single_variable_causal_transfer": {
            "status": "not_assessable_no_full_go",
            "reason": "matched-context and synchronized controls were not collected",
        },
        "gate_3_joint_binding": {
            "status": "go" if joint_binding_measured else "revise_or_stop",
        },
        "gate_4_native_state_carrier_qualification": {
            "status": "not_assessable_no_full_go",
            "reason": "Gate 2 cannot be fully assessed from the frozen Core Set",
        },
        "allowed_conclusion": allowed,
        "route_decision": route,
    }


def run_exp001_confirmatory_analysis(
    *,
    raw_output_dir: str | Path,
    raw_verification_path: str | Path,
    core_set_package_dir: str | Path,
    final_package_dir: str | Path,
    analysis_config_path: str | Path,
    analysis_output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Read a verified full raw package once and emit frozen aggregate results."""
    root = Path(project_root).resolve()
    raw_root = Path(raw_output_dir).resolve()
    core_root = Path(core_set_package_dir).resolve()
    final_root = Path(final_package_dir).resolve()
    config_path = Path(analysis_config_path).resolve()
    destination = Path(analysis_output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("analysis output directory must be empty; results are immutable")
    if sha256_file(config_path) != EXPECTED_ANALYSIS_CONFIG_SHA256:
        raise ValueError("analysis config differs from the pinned pre-observation plan")
    config = _load_object(config_path, "analysis config")
    verification = _load_object(raw_verification_path, "raw verification")
    completion = _load_object(raw_root / "completion.json", "raw completion")
    raw_manifest = _load_object(raw_root / "manifest.json", "raw manifest")
    core_set = _load_object(core_root / "core_set.json", "Core Set")
    final_manifest = _load_object(final_root / "manifest.json", "final preregistration")
    final_candidate = _load_object(final_root / "candidate.json", "final candidate")
    core_report = verify_core_set_package(core_root)
    final_report = verify_final_preregistration_package(final_root)
    frozen_statistics = final_candidate.get("statistics", {})
    frozen_bootstrap = frozen_statistics.get("bootstrap", {})
    frozen_permutation = frozen_statistics.get("permutation", {})
    frozen_sesoi = frozen_statistics.get("sesoi", {})
    analysis_sesoi = config.get("sesoi", {})
    prerequisites = {
        "raw_verification_valid": (
            verification.get("valid") is True
            and verification.get("status") == "raw_package_verified_unanalyzed"
            and verification.get("failed_checks") == []
            and verification.get("failed_group_count") == 0
            and verification.get("verified_record_count") == EXPECTED_RAW_RECORD_COUNT
        ),
        "raw_completion_valid": (
            completion.get("valid") is True
            and completion.get("status") == "confirmatory_raw_complete"
            and completion.get("confirmatory_results_observed") is False
        ),
        "raw_payload_identity_valid": (
            verification.get("group_payload_digest_sha256")
            == completion.get("group_payload_digest_sha256")
            == raw_manifest.get("group_payload_digest_sha256")
        ),
        "core_package_valid": core_report.get("valid") is True,
        "final_preregistration_valid": final_report.get("valid") is True,
        "frozen_identity_valid": (
            config.get("experiment_id") == EXPECTED_EXPERIMENT_ID
            and core_set.get("core_set_digest_sha256") == EXPECTED_CORE_SET_DIGEST
            and final_manifest.get("final_preregistration_digest_sha256")
            == EXPECTED_FINAL_DIGEST
            and final_manifest.get("model_id") == EXPECTED_MODEL_ID
        ),
        "analysis_statistics_match_frozen_candidate": (
            frozen_statistics.get("primary_endpoints")
            == list(PRIMARY_ENDPOINTS)
            and frozen_statistics.get("multiple_comparison_correction") == "Holm"
            and float(frozen_statistics.get("familywise_alpha"))
            == float(config["multiple_comparison"]["familywise_alpha"])
            and frozen_bootstrap.get("unit") == config["bootstrap"]["unit"]
            and int(frozen_bootstrap.get("replicates"))
            == int(config["bootstrap"]["replicates"])
            and float(frozen_bootstrap.get("confidence"))
            == float(config["bootstrap"]["confidence"])
            and frozen_bootstrap.get("method") == config["bootstrap"]["method"]
            and int(frozen_permutation.get("minimum_replicates"))
            == int(config["permutation"]["replicates"])
            and final_candidate.get("seeds", {}).get("bootstrap")
            == config["bootstrap"]["seed"]
            and final_candidate.get("seeds", {}).get("permutation")
            == config["permutation"]["seed"]
            and float(frozen_sesoi.get("identity_log_odds"))
            == float(analysis_sesoi["E1_identity_transfer"])
            and float(frozen_sesoi.get("goal_log_odds"))
            == float(analysis_sesoi["E2_goal_transfer"])
            and float(frozen_sesoi.get("joint_log_margin"))
            == float(analysis_sesoi["E3_joint_binding"])
            and float(frozen_sesoi.get("minimum_joint_accuracy"))
            == float(analysis_sesoi["joint_accuracy_lower_bound"])
            and float(frozen_sesoi.get("identity_specificity"))
            == float(analysis_sesoi["identity_specificity"])
            and float(frozen_sesoi.get("goal_specificity"))
            == float(analysis_sesoi["goal_specificity"])
            and float(frozen_sesoi.get("prompt_normalized_retention"))
            == float(analysis_sesoi["prompt_normalized_retention"])
        ),
        "design_counts_valid": (
            len(core_set.get("groups", [])) == EXPECTED_GROUP_COUNT
            and completion.get("completed_group_count") == EXPECTED_GROUP_COUNT
            and completion.get("raw_record_count") == EXPECTED_RAW_RECORD_COUNT
        ),
    }
    if not all(prerequisites.values()):
        failed = [key for key, value in prerequisites.items() if not value]
        raise ValueError(f"analysis prerequisites failed: {failed}")
    groups_by_id = {
        group["factorial_group_id"]: group for group in core_set["groups"]
    }
    reports = []
    for group_id in raw_manifest["expected_group_ids"]:
        payload = _load_object(raw_root / "groups" / f"{group_id}.json", group_id)
        report = analyze_confirmatory_group(groups_by_id[group_id], payload)
        prompt_scores = _semantic_condition_scores(groups_by_id[group_id], payload)[
            "prompt_visible"
        ]
        report["prompt_visible_contrasts"] = group_contrasts(prompt_scores)
        reports.append(report)
    if len(reports) != EXPECTED_GROUP_COUNT:
        raise ValueError("analysis did not produce all group reports")
    aggregate = summarize_confirmatory_groups(reports, config)
    source_digests = {
        path: sha256_file(root / path) for path in ANALYSIS_SOURCE_FILES
    }
    destination.mkdir(parents=True, exist_ok=True)
    group_payload = {
        "group_report_version": "1.0",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "factorial_group_count": len(reports),
        "confirmatory_results_observed": True,
        "groups": reports,
    }
    report_payload = {
        "report_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_id": EXPECTED_MODEL_ID,
        "analysis_id": config["analysis_id"],
        "analysis_config_sha256": EXPECTED_ANALYSIS_CONFIG_SHA256,
        "raw_group_payload_digest_sha256": verification[
            "group_payload_digest_sha256"
        ],
        "factorial_group_count": len(reports),
        "raw_record_count": EXPECTED_RAW_RECORD_COUNT,
        "confirmatory_experiment_run": True,
        "confirmatory_results_observed": True,
        "analysis_read_only": True,
        "prerequisite_checks": prerequisites,
        **aggregate,
    }
    group_path = destination / "group_level_contrasts.json"
    report_path = destination / "confirmatory_report.json"
    group_path.write_bytes(canonical_json_bytes(group_payload))
    report_path.write_bytes(canonical_json_bytes(report_payload))
    summary = {
        "summary_version": "1.0",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": "confirmatory_analysis_complete",
        "valid": True,
        "confirmatory_experiment_run": True,
        "confirmatory_results_observed": True,
        "analysis_read_only": True,
        "factorial_group_count": len(reports),
        "raw_record_count": EXPECTED_RAW_RECORD_COUNT,
        "analysis_config_sha256": EXPECTED_ANALYSIS_CONFIG_SHA256,
        "raw_group_payload_digest_sha256": verification[
            "group_payload_digest_sha256"
        ],
        "group_level_contrasts_sha256": sha256_file(group_path),
        "confirmatory_report_sha256": sha256_file(report_path),
        "analysis_source_digests": source_digests,
        "measured_decisions": aggregate["measured_decisions"],
        "gate_2_status": aggregate["gate_2_single_variable_causal_transfer"][
            "status"
        ],
        "gate_3_status": aggregate["gate_3_joint_binding"]["status"],
        "gate_4_status": aggregate["gate_4_native_state_carrier_qualification"][
            "status"
        ],
        "allowed_conclusion": aggregate["allowed_conclusion"],
        "route_decision": aggregate["route_decision"],
        "reports": ["group_level_contrasts.json", "confirmatory_report.json"],
    }
    summary["analysis_package_digest_sha256"] = sha256_json(
        {
            "group_level_contrasts.json": summary[
                "group_level_contrasts_sha256"
            ],
            "confirmatory_report.json": summary["confirmatory_report_sha256"],
        }
    )
    (destination / "summary.json").write_bytes(canonical_json_bytes(summary))
    return summary
