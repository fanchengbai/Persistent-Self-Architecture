from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d4b_steady_state_off_design import (
    PRECONDITION_ORDER,
    ROUTES,
    SCORED_ROUNDS,
    validate_design,
)
from psa.self_model.d4b_steady_state_off_runtime import D4B_RUNTIME_VERSION


RUNTIME_CONFIG = (
    "configs/development/self_model_v0_1_d4b_steady_state_off_runtime.json"
)
RUNTIME_SOURCE = "src/psa/self_model/d4b_steady_state_off_runtime.py"
SOURCE_FILES = (
    RUNTIME_CONFIG,
    "configs/development/self_model_v0_1_d4b_steady_state_off_design.json",
    "docs/self_model_v0_1_d4b_steady_state_off_design.md",
    "docs/self_model_v0_1_d4b_steady_state_off_runtime.md",
    "scripts/verify_self_model_v0_1_d4b_steady_state_off_runtime.py",
    RUNTIME_SOURCE,
    "src/psa/self_model/d4b_steady_state_off_runtime_manifest.py",
    "tests/test_self_model_d4b_steady_state_off_runtime.py",
    "tests/test_self_model_d4b_steady_state_off_runtime_manifest.py",
)


def _object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _import_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def validate_d4b_runtime_config(
    *, config: Mapping[str, Any], design: Mapping[str, Any], runtime_source: str
) -> dict[str, bool]:
    prerequisite = config.get("prerequisite_design")
    implementation = config.get("implementation")
    verification = config.get("current_verification")
    authority = config.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (prerequisite, implementation, verification, authority)
    ):
        raise ValueError("D4B runtime config is incomplete")
    design_checks = validate_design(design)
    imported = _import_roots(runtime_source)
    expected_verification = {
        "fake_model_only": True,
        "installed_source_probe_included": False,
        "rwkv_model_import_included": False,
        "torch_import_included": False,
        "weights_access_included": False,
        "real_model_execution_included": False,
        "real_execution_claim_implemented": False,
        "real_result_observation_included": False,
    }
    expected_authority = {
        "runtime_core_implementation_authorized": True,
        "fake_test_authorized": True,
        "server_no_model_static_verification_authorized": True,
        "real_execution_entry_implementation_authorized": False,
        "rwkv_model_import_authorized": False,
        "torch_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "result_observation_authorized": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "automatic_rerun_authorized": False,
    }
    checks = {
        "implementation_identity_valid": (
            config.get("implementation_version") == D4B_RUNTIME_VERSION
            and config.get("stage") == "D4B_steady_state_off_runtime_fake_first"
            and config.get("status") == "runtime_core_implemented_real_entry_absent"
            and config.get("development_only") is True
        ),
        "prerequisite_design_locked": dict(prerequisite)
        == {
            "path": (
                "configs/development/"
                "self_model_v0_1_d4b_steady_state_off_design.json"
            ),
            "sha256": (
                "e3f5cd05f9e5bc7bb282bd82f543c0893d175f51805e56a3e8d53bbbabbe353f"
            ),
            "static_report_sha256": (
                "7f3cfb7fecf6892532f9ecdb27f528716b363077e37e288f8940d8e883ef658d"
            ),
        },
        "prerequisite_design_valid": all(design_checks.values()),
        "old_design_authority_preserved": (
            design.get("runtime_implementation_authorized") is False
            and design.get("model_execution_authorized") is False
            and design.get("d5_authorized") is False
        ),
        "runtime_path_and_real_entry_absence_fixed": (
            implementation.get("path") == RUNTIME_SOURCE
            and implementation.get("project_local_only") is True
            and implementation.get("real_model_entry_present") is False
        ),
        "four_routes_fixed": implementation.get("routes") == ROUTES,
        "fixtures_and_preconditioning_fixed": (
            implementation.get("prefix_token_ids") == [187, 931]
            and implementation.get("target_token_ids") == [2764]
            and implementation.get("fixed_preconditioning_order")
            == PRECONDITION_ORDER
        ),
        "latin_schedule_fixed": implementation.get("scored_rounds")
        == SCORED_ROUNDS,
        "call_counts_fixed": (
            implementation.get("prefix_call_count") == 1
            and implementation.get("preconditioning_call_count") == 4
            and implementation.get("scored_call_count") == 16
            and implementation.get("total_forward_call_count") == 21
        ),
        "pair_counts_and_exact_rule_fixed": (
            implementation.get("within_route_comparison_count") == 24
            and implementation.get("cross_route_comparison_count") == 96
            and implementation.get("comparison") == "torch.equal"
        ),
        "all_outputs_recorded_without_adaptation": (
            implementation.get("all_outputs_recorded") is True
            and implementation.get("adaptive_or_extra_calls_allowed") is False
        ),
        "runtime_cannot_upgrade_d4_or_d5": (
            implementation.get("d4_status_can_change") is False
            and implementation.get("d5_can_be_authorized") is False
        ),
        "verification_is_fake_only": dict(verification) == expected_verification,
        "runtime_has_no_rwkv_or_torch_import": (
            "rwkv" not in imported and "torch" not in imported
        ),
        "fixed_schedule_markers_present": all(
            marker in runtime_source
            for marker in (
                'phase="prefix_snapshot"',
                'phase="fixed_preconditioning"',
                'phase="scored_latin"',
                '"all_within_route_pairs_recorded"',
                '"all_cross_route_pairs_recorded"',
                '"all_scored_pairs_exact"',
            )
        ),
        "failure_and_safety_markers_present": all(
            marker in runtime_source
            for marker in (
                '"d4_status_changed": False',
                '"d5_authorized": False',
                '"stop_without_rerun"',
                '"real_model_entry_implemented": False',
                '"automatic_rerun_authorized": False',
            )
        ),
        "authority_is_fake_runtime_only": dict(authority) == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D4B runtime config failed closed: " + ", ".join(failed))
    return checks


def build_d4b_runtime_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / RUNTIME_CONFIG).resolve():
        raise PermissionError("D4B runtime config path is not frozen")
    config = _object(config_file, "D4B runtime config")
    prerequisite = config["prerequisite_design"]
    design_path = (root / prerequisite["path"]).resolve()
    if sha256_file(design_path) != prerequisite["sha256"]:
        raise RuntimeError("D4B prerequisite design digest changed")
    design = _object(design_path, "D4B design")
    runtime_source = (root / RUNTIME_SOURCE).read_text(encoding="utf-8")
    checks = validate_d4b_runtime_config(
        config=config, design=design, runtime_source=runtime_source
    )
    report = {
        "report_version": D4B_RUNTIME_VERSION,
        "status": "d4b_fake_first_runtime_static_verified",
        "valid": all(checks.values()),
        "development_only": True,
        "checks": checks,
        "source_digests": {
            relative: sha256_file(root / relative) for relative in SOURCE_FILES
        },
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "installed_rwkv_source_probed": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "runtime_core_implemented": True,
            "real_execution_entry_implemented": False,
            "machine_authorization_created": False,
            "execution_claim_created": False,
            "d4_status_changed": False,
            "d5_authorized": False,
            "active_injection_implemented": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
