from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-d4b-steady-state-off-design"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d4b_steady_state_off_design.json"
)
D4_REPORT_DIGEST = "39d4611a6d50791f1677f9eb27e6fb2ea702151a26236fa4094699b821ca721a"
D4A_REPORT_DIGEST = "d6b0602a85553fddae184e2accb3ef06ed280a925ebc4d90a9e13032726b2e88"
ROUTES = [
    "original_baseline",
    "off_g1_passthrough",
    "g0_recompiled_unmodified",
    "off_g2_instrumented",
]
PRECONDITION_ORDER = list(ROUTES)
SCORED_ROUNDS = [
    ["original_baseline", "off_g1_passthrough", "g0_recompiled_unmodified", "off_g2_instrumented"],
    ["off_g1_passthrough", "g0_recompiled_unmodified", "off_g2_instrumented", "original_baseline"],
    ["g0_recompiled_unmodified", "off_g2_instrumented", "original_baseline", "off_g1_passthrough"],
    ["off_g2_instrumented", "original_baseline", "off_g1_passthrough", "g0_recompiled_unmodified"],
]
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d4a_real_diagnostic_observation.md",
    "docs/self_model_v0_1_d4b_steady_state_off_design.md",
    "scripts/verify_self_model_v0_1_d4b_steady_state_off_design.py",
    "src/psa/self_model/d4_real_off_equivalence.py",
    "src/psa/self_model/d4a_failure_diagnostic_runtime.py",
    "src/psa/self_model/d4b_steady_state_off_design.py",
    "tests/test_self_model_d4b_steady_state_off_design.py",
)


def _object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D4B design must be an object")
    return value


def _exact(value: Any, expected: Any, field: str) -> bool:
    return value == expected


def build_d4_call_trace() -> list[dict[str, Any]]:
    trace = [
        {
            "call_index": 1,
            "phase": "prefix_snapshot",
            "cell_id": None,
            "route": "original_baseline",
            "method_family": "upstream_original",
            "execution_path": "forward_seq",
            "state_input": "none",
            "scored": False,
            "output_recorded": False,
        }
    ]
    cells = [
        ("forward_one__none__full_output_false", "forward_one", "none", False),
        ("forward_one__restored__full_output_false", "forward_one", "restored", False),
        ("forward_seq__none__full_output_false", "forward_seq", "none", False),
        ("forward_seq__none__full_output_true", "forward_seq", "none", True),
        ("forward_seq__restored__full_output_false", "forward_seq", "restored", False),
        ("forward_seq__restored__full_output_true", "forward_seq", "restored", True),
    ]
    families = {
        "original_baseline": "upstream_original",
        "off_g1_passthrough": "upstream_original",
        "off_g2_instrumented": "recompiled_instrumented",
    }
    index = 1
    for cell_id, path, state_input, full_output in cells:
        for route in (
            "original_baseline",
            "off_g1_passthrough",
            "off_g2_instrumented",
        ):
            for phase, scored in (("route_warmup", False), ("route_score", True)):
                index += 1
                trace.append(
                    {
                        "call_index": index,
                        "phase": phase,
                        "cell_id": cell_id,
                        "route": route,
                        "method_family": families[route],
                        "execution_path": path,
                        "state_input": state_input,
                        "full_output": full_output,
                        "scored": scored,
                        "output_recorded": scored,
                    }
                )
    return trace


def build_d4a_call_trace() -> list[dict[str, Any]]:
    rounds = [
        ["original_baseline", "g0_recompiled_unmodified", "off_g2_instrumented"],
        ["g0_recompiled_unmodified", "off_g2_instrumented", "original_baseline"],
        ["off_g2_instrumented", "original_baseline", "g0_recompiled_unmodified"],
    ]
    families = {
        "original_baseline": "upstream_original",
        "g0_recompiled_unmodified": "recompiled_unmodified",
        "off_g2_instrumented": "recompiled_instrumented",
    }
    trace = []
    index = 0
    for round_index, routes in enumerate(rounds, start=1):
        for position, route in enumerate(routes, start=1):
            index += 1
            trace.append(
                {
                    "call_index": index,
                    "round_index": round_index,
                    "order_position": position,
                    "route": route,
                    "method_family": families[route],
                    "execution_path": "forward_one",
                    "state_input": "none",
                    "full_output": False,
                    "scored": True,
                    "output_recorded": True,
                    "observed_cluster": (
                        "first_original_transient"
                        if index == 1
                        else "first_g0_transient" if index == 2 else "shared_steady"
                    ),
                }
            )
    return trace


def validate_design(design: Mapping[str, Any]) -> dict[str, bool]:
    closure = design.get("diagnostic_closure")
    reused = design.get("reused_d4_evidence")
    target = design.get("target_cell")
    families = design.get("method_families")
    if not all(isinstance(item, Mapping) for item in (closure, reused, target, families)):
        raise ValueError("D4B design structure is incomplete")
    route_pairs = list(combinations(ROUTES, 2))
    checks = {
        "identity_valid": _exact(design.get("design_version"), DESIGN_VERSION, "version")
        and _exact(
            design.get("stage"),
            "D4B_prospective_steady_state_off_equivalence_design",
            "stage",
        )
        and design.get("status") == "design_only_runtime_absent_execution_not_authorized",
        "development_design_only": design.get("development_only") is True,
        "d4_failure_preserved": closure.get("d4_status") == "failed_preserved"
        and closure.get("d4_report_digest_sha256") == D4_REPORT_DIGEST,
        "d4a_evidence_frozen": closure.get("d4a_report_digest_sha256")
        == D4A_REPORT_DIGEST
        and closure.get("d4a_classification") == "within_route_instability_observed",
        "location_signature_exact": closure.get(
            "first_original_vs_first_g0_mismatched_state_components"
        )
        == 92
        and closure.get("first_original_vs_first_g0_mismatch_range") == [4, 95]
        and closure.get("d4_failed_state_component_range") == [4, 95],
        "association_not_overclaimed": closure.get(
            "shared_location_signature_is_association_not_identity_proof"
        )
        is True
        and closure.get("exact_low_level_cache_mechanism_identified") is False,
        "warmup_nonidentifiability_recorded": closure.get(
            "d4_off_g2_warmup_was_present"
        )
        is True
        and closure.get("d4_warmup_outputs_unavailable") is True
        and closure.get("d4a_reproduced_d4_prefix_and_schedule") is False
        and closure.get(
            "recorded_global_method_family_preconditioning_previously_tested"
        )
        is False
        and closure.get("d4b_preconditioning_is_prospective_control_not_causal_fix")
        is True,
        "old_passing_evidence_not_rerun": reused
        == {
            "off_g1_all_six_cells_exact": True,
            "off_g2_other_five_cells_exact": True,
            "cells_rerun": False,
        },
        "only_failed_cell_targeted": target
        == {
            "cell_id": "forward_one__none__full_output_false",
            "token_ids": [2764],
            "state_input": "none",
            "full_output": False,
        },
        "d4_prefix_context_reproduced": design.get(
            "prefix_snapshot_reproduced_before_preconditioning"
        )
        is True
        and design.get("prefix_snapshot_token_ids") == [187, 931]
        and design.get("prefix_snapshot_output_recorded_not_scored") is True,
        "four_routes_frozen": design.get("routes") == ROUTES
        and families
        == {
            "original_baseline": "upstream_original",
            "off_g1_passthrough": "upstream_original",
            "g0_recompiled_unmodified": "recompiled_unmodified",
            "off_g2_instrumented": "recompiled_instrumented",
        },
        "fixed_recorded_preconditioning": design.get(
            "fixed_recorded_preconditioning_order"
        )
        == PRECONDITION_ORDER
        and design.get("preconditioning_calls_per_route") == 1
        and design.get("preconditioning_outputs_recorded") is True
        and design.get("preconditioning_outputs_scored") is False,
        "no_adaptive_warmup": design.get("adaptive_convergence_or_extra_warmup_allowed")
        is False,
        "latin_scored_rounds": design.get("scored_rounds") == SCORED_ROUNDS
        and all(
            sorted(round_routes) == sorted(ROUTES) for round_routes in SCORED_ROUNDS
        )
        and all(
            sorted(round_routes[position] for round_routes in SCORED_ROUNDS)
            == sorted(ROUTES)
            for position in range(4)
        ),
        "call_counts_exact": design.get("scored_calls_per_route") == 4
        and design.get("scored_model_forward_call_count") == 16
        and design.get("total_model_forward_call_count_including_prefix_and_preconditioning")
        == 21,
        "comparison_counts_exact": design.get("within_route_comparison_count") == 24
        and design.get("cross_route_comparison_count") == len(route_pairs) * 16,
        "strict_exact_pass_rule": design.get("comparison") == "torch.equal"
        and design.get("pass_requires_all_scored_within_and_cross_route_pairs_exact")
        is True
        and design.get("cross_run_digest_comparison_prohibited") is True,
        "d4_status_immutable": design.get("d4_status_can_change") is False,
        "pass_only_allows_review": design.get("d4b_pass_effect")
        == "steady_state_off_qualification_candidate_allows_d5_review_only",
        "design_authority_only": design.get("design_authorized") is True
        and design.get("runtime_implementation_authorized") is False
        and design.get("model_execution_authorized") is False
        and design.get("result_observation_authorized") is False,
        "prohibited_authorities_false": all(
            design.get(field) is False
            for field in (
                "d4_rerun_authorized",
                "active_injection_authorized",
                "self_effect_experiment_authorized",
                "d5_authorized",
                "confirmatory_decision_authorized",
                "automatic_rerun_authorized",
            )
        ),
        "failure_stops": design.get("failure_action")
        == "stop_without_rerun_or_design_revision",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D4B design failed closed: " + ", ".join(failed))
    return checks


def build_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D4B design config path is not frozen")
    design = _object(config_file)
    checks = validate_design(design)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    report = {
        "report_version": DESIGN_VERSION,
        "status": "d4b_steady_state_off_design_static_verified",
        "valid": True,
        "checks": checks,
        "d4_trace": build_d4_call_trace(),
        "d4a_trace": build_d4a_call_trace(),
        "source_digests": source_digests,
        "safety": {
            "runtime_implemented": False,
            "machine_authorization_created": False,
            "execution_claim_created": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d4_status_changed": False,
            "d5_authorized": False,
            "active_injection_implemented": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
