from __future__ import annotations

import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d4a_failure_diagnostic_runtime import (
    _select_unmodified_method_asts,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
    inspect_instrumented_source,
)


VERIFICATION_VERSION = "0.1-d4a-cloud-static"
VERIFICATION_CONFIG = (
    "configs/development/self_model_v0_1_d4a_cloud_static_verification.json"
)
SOURCE_FILES = (
    VERIFICATION_CONFIG,
    "configs/development/self_model_v0_1_d4a_failure_diagnostic_runtime.json",
    "docs/self_model_v0_1_d4a_cloud_static_verification.md",
    "scripts/verify_self_model_v0_1_d4a_cloud_static.py",
    "src/psa/self_model/d4a_cloud_static_verification.py",
    "src/psa/self_model/d4a_failure_diagnostic_runtime.py",
    "src/psa/self_model/d4a_failure_diagnostic_manifest.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d4a_cloud_static_verification.py",
    "tests/test_self_model_d4a_failure_diagnostic_runtime.py",
    "tests/test_self_model_d4a_failure_diagnostic_manifest.py",
)


def _object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def probe_installed_rwkv_source() -> tuple[dict[str, Any], bytes]:
    """Read installed source bytes via package metadata without importing RWKV."""
    distribution = metadata.distribution("rwkv")
    source_path = Path(distribution.locate_file("rwkv/model.py")).resolve()
    source_bytes = source_path.read_bytes()
    return (
        {
            "package": "rwkv",
            "version": distribution.version,
            "model_source_path": "rwkv/model.py",
            "model_source_sha256": _sha256_bytes(source_bytes),
            "model_source_size_bytes": len(source_bytes),
            "access_method": "importlib.metadata_and_read_bytes",
        },
        source_bytes,
    )


def validate_d4a_cloud_static_config(config: Mapping[str, Any]) -> dict[str, bool]:
    upstream = config.get("upstream")
    prerequisite = config.get("runtime_prerequisite")
    verification = config.get("verification")
    authority = config.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (upstream, prerequisite, verification, authority)
    ):
        raise ValueError("D4A cloud static config is incomplete")
    expected_authority = {
        "installed_source_static_verification_authorized": True,
        "rwkv_model_import_authorized": False,
        "torch_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "real_execution_entry_implementation_authorized": False,
        "execution_claim_authorized": False,
        "d4_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    checks = {
        "verification_identity_valid": (
            config.get("verification_version") == VERIFICATION_VERSION
            and config.get("stage") == "D4A_cloud_static_verification_without_model"
            and config.get("status") == "installed_source_bytes_and_project_runtime_only"
            and config.get("development_only") is True
        ),
        "upstream_lock_frozen": dict(upstream)
        == {
            "package": "rwkv",
            "version": "0.8.32",
            "model_source_path": "rwkv/model.py",
            "model_source_sha256": (
                "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
            ),
            "model_source_size_bytes": 85425,
            "rwkv_de_version_environment": "unset",
        },
        "runtime_prerequisite_frozen": dict(prerequisite)
        == {
            "config_path": (
                "configs/development/"
                "self_model_v0_1_d4a_failure_diagnostic_runtime.json"
            ),
            "config_sha256": (
                "b7249496806f3eeaa2f07409753d7263ab4320e6bddc62630f57cc5bff3d6c4f"
            ),
            "runtime_path": "src/psa/self_model/d4a_failure_diagnostic_runtime.py",
            "runtime_sha256": (
                "91001bc29aa404cd371d01aa78458440e90b1867fa9bf94cb8262d89f3ff967e"
            ),
            "manifest_path": "src/psa/self_model/d4a_failure_diagnostic_manifest.py",
            "manifest_sha256": (
                "ae729d442b6eca2146faf4e8a0dcd62fb96ec82ca193585354ecba3d7ff06e24"
            ),
        },
        "verification_scope_is_no_model": dict(verification)
        == {
            "installed_package_metadata_read": True,
            "installed_model_source_bytes_read": True,
            "g0_unmodified_ast_selection_included": True,
            "g2_instrumented_ast_selection_included": True,
            "real_decorator_inventory_included": True,
            "real_variant_alignment_included": True,
            "rwkv_model_import_included": False,
            "torch_import_included": False,
            "weights_access_included": False,
            "model_load_included": False,
            "model_execution_included": False,
            "real_execution_entry_included": False,
            "execution_claim_included": False,
        },
        "authority_is_static_only": dict(authority) == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PermissionError("D4A cloud static config failed closed: " + ", ".join(failed))
    return checks


def inspect_d4a_installed_source(source: str) -> dict[str, Any]:
    g0_methods, g0_selection = _select_unmodified_method_asts(
        source, rwkv_de_version=None
    )
    g0_rendered = {}
    for name, method in g0_methods.items():
        module = ast.Module(body=[method], type_ignores=[])
        ast.fix_missing_locations(module)
        g0_rendered[name] = ast.unparse(module)
    g2 = inspect_instrumented_source(source)
    g2_selection = g2["variant_selection"]
    checks = {
        "g0_both_paths_selected": set(g0_methods) == set(TARGET_METHODS),
        "g0_two_variants_per_path": all(
            g0_selection[name]["candidate_count"] == 2 for name in TARGET_METHODS
        ),
        "g0_unset_de_selects_else": all(
            g0_selection[name]["condition"]
            == "os.environ.get('RWKV_DE_VERSION') == '1'"
            and g0_selection[name]["selected_branch"]
            == "else_rwkv_de_version_unset"
            for name in TARGET_METHODS
        ),
        "real_original_decorators_are_myfunction": all(
            g0_selection[name]["original_decorators"] == ["MyFunction"]
            for name in TARGET_METHODS
        ),
        "g0_compiled_decorators_empty": all(
            g0_selection[name]["compiled_decorators"] == []
            for name in TARGET_METHODS
        ),
        "g0_contains_no_callback_or_injected_phase": all(
            CALLBACK_ATTRIBUTE not in rendered and "post_ffn_residual" not in rendered
            for rendered in g0_rendered.values()
        ),
        "g2_static_transform_valid": g2["valid"] is True,
        "g2_two_variants_and_one_site_each": all(
            g2_selection[name]["candidate_count"] == 2
            and g2_selection[name]["injection_counts"] == [1, 1]
            for name in TARGET_METHODS
        ),
        "g0_g2_variant_selection_aligned": all(
            g0_selection[name]["candidate_count"]
            == g2_selection[name]["candidate_count"]
            and g0_selection[name]["condition"] == g2_selection[name]["condition"]
            and g0_selection[name]["selected_branch"]
            == g2_selection[name]["selected_branch"]
            and g0_selection[name]["selected_source_line"]
            == g2_selection[name]["selected_source_line"]
            for name in TARGET_METHODS
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "g0_variant_selection": g0_selection,
        "g0_method_source_sha256": {
            name: hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            for name, rendered in g0_rendered.items()
        },
        "g2_injection_counts": g2["injection_counts"],
        "g2_variant_selection": g2_selection,
        "g2_method_source_sha256": g2["method_source_sha256"],
    }


def build_d4a_cloud_static_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    installed_source: Mapping[str, Any],
    upstream_source_bytes: bytes,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    if config_path != (root / VERIFICATION_CONFIG).resolve():
        raise PermissionError("D4A cloud static config path is not frozen")
    config = _object(config_path, "D4A cloud static config")
    config_checks = validate_d4a_cloud_static_config(config)
    prerequisite = config["runtime_prerequisite"]
    prerequisite_checks = {
        "runtime_config_digest_matches": sha256_file(root / prerequisite["config_path"])
        == prerequisite["config_sha256"],
        "runtime_source_digest_matches": sha256_file(root / prerequisite["runtime_path"])
        == prerequisite["runtime_sha256"],
        "runtime_manifest_digest_matches": sha256_file(root / prerequisite["manifest_path"])
        == prerequisite["manifest_sha256"],
    }
    upstream = config["upstream"]
    installed_checks = {
        "installed_package_matches": installed_source.get("package") == upstream["package"],
        "installed_version_matches": installed_source.get("version") == upstream["version"],
        "installed_source_path_matches": installed_source.get("model_source_path")
        == upstream["model_source_path"],
        "installed_source_digest_matches": (
            installed_source.get("model_source_sha256")
            == upstream["model_source_sha256"]
            == _sha256_bytes(upstream_source_bytes)
        ),
        "installed_source_size_matches": (
            installed_source.get("model_source_size_bytes")
            == upstream["model_source_size_bytes"]
            == len(upstream_source_bytes)
        ),
        "installed_source_access_is_bytes_only": installed_source.get("access_method")
        == "importlib.metadata_and_read_bytes",
    }
    inspection = inspect_d4a_installed_source(upstream_source_bytes.decode("utf-8"))
    inspection_checks = {
        f"installed_source_{name}": value
        for name, value in inspection["checks"].items()
    }
    module_checks = {
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    checks = {
        **config_checks,
        **prerequisite_checks,
        **installed_checks,
        **inspection_checks,
        **module_checks,
    }
    report = {
        "report_version": VERIFICATION_VERSION,
        "status": "d4a_cloud_static_verified" if all(checks.values()) else "d4a_cloud_static_failed",
        "valid": all(checks.values()),
        "development_only": True,
        "installed_source": dict(installed_source),
        "inspection": inspection,
        "checks": checks,
        "source_digests": {
            relative: sha256_file(root / relative) for relative in SOURCE_FILES
        },
        "safety": {
            "installed_rwkv_source_probed": True,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_execution_entry_implemented": False,
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
