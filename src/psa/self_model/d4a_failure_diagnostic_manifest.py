from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d4a_failure_diagnostic_runtime import (
    D4A_RECORDED_ROUNDS,
    D4A_RUNTIME_VERSION,
)


IMPLEMENTATION_CONFIG = (
    "configs/development/self_model_v0_1_d4a_failure_diagnostic_runtime.json"
)
SOURCE_FILES = (
    IMPLEMENTATION_CONFIG,
    "configs/development/self_model_v0_1_d4a_failure_diagnostic_design.json",
    "docs/self_model_v0_1_d4a_failure_diagnostic_design.md",
    "docs/self_model_v0_1_d4a_failure_diagnostic_runtime.md",
    "scripts/verify_self_model_v0_1_d4a_failure_diagnostic_runtime.py",
    "src/psa/self_model/d4a_failure_diagnostic_runtime.py",
    "src/psa/self_model/d4a_failure_diagnostic_manifest.py",
    "tests/test_self_model_d4a_failure_diagnostic_runtime.py",
    "tests/test_self_model_d4a_failure_diagnostic_manifest.py",
)


def _object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _import_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def validate_d4a_runtime_config(
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
        raise ValueError("D4A runtime config is incomplete")
    expected_rounds = D4A_RECORDED_ROUNDS
    expected_authority = {
        "diagnostic_runtime_implementation_authorized": True,
        "fake_test_authorized": True,
        "installed_source_static_verification_authorized": False,
        "real_execution_entry_implementation_authorized": False,
        "rwkv_model_import_authorized": False,
        "torch_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "diagnostic_result_observation_authorized": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    imported = _import_roots(runtime_source)
    checks = {
        "implementation_identity_valid": (
            config.get("implementation_version") == D4A_RUNTIME_VERSION
            and config.get("stage") == "D4A_failure_diagnostic_runtime_fake_only"
            and config.get("status") == "runtime_implemented_real_entry_absent"
            and config.get("development_only") is True
        ),
        "prerequisite_design_locked": dict(prerequisite)
        == {
            "path": (
                "configs/development/"
                "self_model_v0_1_d4a_failure_diagnostic_design.json"
            ),
            "sha256": (
                "3e4286ece9c78b5c2d8a9f7a347f6906cc08590c4e2196aae0f8391596545277"
            ),
        },
        "design_remains_unexecuted": (
            design.get("design_version") == "0.1-d4a-failure-diagnostic-design"
            and design.get("d4_failure_evidence", {}).get("valid") is False
            and design.get("authority", {}).get("model_execution_authorized") is False
        ),
        "runtime_path_and_boundary_fixed": (
            implementation.get("path")
            == "src/psa/self_model/d4a_failure_diagnostic_runtime.py"
            and implementation.get("project_local_only") is True
            and implementation.get("real_model_entry_present") is False
            and implementation.get("g0_recompiled_unmodified_implemented") is True
            and implementation.get("g0_callback_attribute_constructed") is False
            and implementation.get("g0_instrumentation_branch_present") is False
            and implementation.get("off_g2_reused_without_modification") is True
        ),
        "three_routes_and_latin_schedule_fixed": (
            implementation.get("routes")
            == [
                "original_baseline",
                "g0_recompiled_unmodified",
                "off_g2_instrumented",
            ]
            and implementation.get("recorded_rounds") == expected_rounds
            and implementation.get("model_forward_call_count") == 9
            and implementation.get("discarded_warmup_call_count") == 0
        ),
        "all_pair_counts_fixed": (
            implementation.get("within_route_comparison_count") == 9
            and implementation.get("cross_route_comparison_count") == 27
        ),
        "runtime_cannot_upgrade_d4_or_d5": (
            implementation.get("d4_status_can_change") is False
            and implementation.get("d5_can_be_authorized") is False
        ),
        "verification_is_fake_only": dict(verification)
        == {
            "fake_model_only": True,
            "installed_source_probe_included": False,
            "rwkv_model_import_included": False,
            "torch_import_included": False,
            "weights_access_included": False,
            "real_model_execution_included": False,
            "real_execution_claim_implemented": False,
            "real_result_observation_included": False,
        },
        "runtime_has_no_rwkv_or_torch_import": (
            "rwkv" not in imported and "torch" not in imported
        ),
        "g0_compile_boundary_present": all(
            marker in runtime_source
            for marker in (
                "cloned.decorator_list = []",
                "namespace = dict(upstream_globals)",
                "<psa-rwkv7-recompiled-unmodified>",
                "types.MethodType(function, self._base_model)",
            )
        ),
        "g0_has_no_callback_set_or_injection_template": (
            "setattr(self._base_model, CALLBACK_ATTRIBUTE, None)" not in runtime_source
            and "phase=\"post_ffn_residual\"" not in runtime_source
        ),
        "diagnostic_records_digests_and_error_magnitude": all(
            marker in runtime_source
            for marker in (
                '"sha256": _tensor_digest',
                '"unequal_element_count"',
                '"max_abs_error"',
                '"mean_abs_error"',
                '"first_mismatch_component"',
            )
        ),
        "diagnostic_safety_flags_present": (
            '"d4_status_changed": False' in runtime_source
            and '"d5_authorized": False' in runtime_source
            and '"automatic_rerun_authorized": False' in runtime_source
        ),
        "authority_is_fake_implementation_only": dict(authority)
        == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D4A runtime config failed closed: " + ", ".join(failed))
    return checks


def build_d4a_runtime_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    if config_path != (root / IMPLEMENTATION_CONFIG).resolve():
        raise PermissionError("D4A runtime config path is not frozen")
    config = _object(config_path, "D4A runtime config")
    prerequisite = config["prerequisite_design"]
    design_path = (root / prerequisite["path"]).resolve()
    if sha256_file(design_path) != prerequisite["sha256"]:
        raise RuntimeError("D4A prerequisite design digest changed")
    design = _object(design_path, "D4A design")
    runtime_source = (
        root / "src/psa/self_model/d4a_failure_diagnostic_runtime.py"
    ).read_text(encoding="utf-8")
    checks = validate_d4a_runtime_config(
        config=config, design=design, runtime_source=runtime_source
    )
    report = {
        "report_version": D4A_RUNTIME_VERSION,
        "status": "d4a_fake_only_diagnostic_runtime_static_verified",
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
            "diagnostic_runtime_implemented": True,
            "real_execution_entry_implemented": False,
            "d4_status_changed": False,
            "d5_authorized": False,
            "active_injection_implemented": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
