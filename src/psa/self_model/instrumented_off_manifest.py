from __future__ import annotations

import ast
from importlib import metadata
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_CLASS,
    TARGET_METHODS,
    inspect_instrumented_source,
)


IMPLEMENTATION_VERSION = "0.1-instrumented-off"
IMPLEMENTATION_CONFIG_FILE = (
    "configs/development/self_model_v0_1_instrumented_off_runtime.json"
)
IMPLEMENTATION_SOURCE_FILES = (
    IMPLEMENTATION_CONFIG_FILE,
    "docs/self_model_v0_1_instrumented_off_runtime.md",
    "schemas/self_model_v0_1_instrumented_off_report.schema.json",
    "scripts/verify_self_model_v0_1_instrumented_off.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "src/psa/self_model/instrumented_off_manifest.py",
    "tests/test_self_model_instrumented_off_runtime.py",
    "tests/test_self_model_instrumented_off_manifest.py",
    "configs/development/self_model_v0_1_d3_static_verification.json",
    "src/psa/self_model/d3_static_verification.py",
)


def _object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _project_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} must stay inside the project root")
    resolved = (root / value).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the project root")
    return resolved


def validate_instrumented_off_config(
    config: Mapping[str, Any]
) -> dict[str, bool]:
    d3 = config.get("d3_prerequisite")
    upstream = config.get("upstream")
    implementation = config.get("implementation")
    verification = config.get("current_verification")
    authority = config.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (d3, upstream, implementation, verification, authority)
    ):
        raise ValueError("instrumented-off config is incomplete")
    expected_authority = {
        "instrumented_runtime_implementation_authorized": True,
        "installed_source_verification_authorized": True,
        "rwkv_model_import_authorized": False,
        "torch_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "site_packages_modification_authorized": False,
        "active_injection_implementation_authorized": False,
        "real_layer_selection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    checks = {
        "implementation_identity_valid": (
            config.get("implementation_version") == IMPLEMENTATION_VERSION
            and config.get("stage")
            == "D3B_off_g2_instrumented_off_implementation_without_model"
            and config.get("status")
            == "off_g2_project_local_runtime_model_unexecuted"
            and config.get("development_only") is True
        ),
        "d3_prerequisite_frozen": dict(d3)
        == {
            "config_path": (
                "configs/development/self_model_v0_1_d3_static_verification.json"
            ),
            "config_sha256": (
                "7cde700044a28334866b3bc16840d61cd8a17a54807316d783139945be3ba16d"
            ),
            "report_path": (
                "results/development/"
                "self_model_v0_1_d3_static_verification/report.json"
            ),
            "report_digest_sha256": (
                "fcb8dfeb58863c9bc5e6c02b8151fb581df5dd50c4158f164005560680c42918"
            ),
        },
        "upstream_lock_frozen": dict(upstream)
        == {
            "package": "rwkv",
            "version": "0.8.32",
            "model_source_path": "rwkv/model.py",
            "model_source_sha256": (
                "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
            ),
            "model_source_size_bytes": 85425,
            "target_class": "RWKV_x070",
        },
        "implementation_path_and_digest_frozen": (
            implementation.get("path")
            == "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
            and implementation.get("sha256")
            == "0d348ee6f80d91d0c25802f78a9b799751cf3a3ecb53922db741af985a99c5c7"
        ),
        "project_local_ast_transform_frozen": (
            implementation.get("project_local_only") is True
            and implementation.get("technique")
            == "locked_source_ast_transform_and_temporary_instance_method_binding"
            and implementation.get("execution_paths")
            == ["forward_one", "forward_seq"]
            and implementation.get("phase") == "post_ffn_residual"
            and implementation.get("required_injection_count_per_path") == 1
            and implementation.get("callback_attribute") == CALLBACK_ATTRIBUTE
            and implementation.get("callback_value_in_off_runtime") is None
        ),
        "restoration_contract_frozen": (
            implementation.get("base_methods_restored_after_call") is True
            and implementation.get("base_methods_restored_after_exception") is True
        ),
        "off_gates_advance_without_active": (
            implementation.get("off_g1_implemented") is True
            and implementation.get("off_g2_implemented") is True
            and implementation.get("active_injection_available") is False
            and implementation.get("self_projection_constructed") is False
            and implementation.get("real_layer_mask") == []
            and implementation.get("real_sequence_policy") == "unfrozen"
        ),
        "verification_scope_is_no_model": dict(verification)
        == {
            "fake_runtime_equivalence_included": True,
            "installed_source_static_transform_included": True,
            "real_model_import_included": False,
            "weights_access_included": False,
            "real_model_execution_included": False,
            "d4_real_2_9b_off_equivalence_completed": False,
        },
        "authority_is_instrumented_off_only": dict(authority)
        == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError(
            "instrumented-off config failed closed: " + ", ".join(failed)
        )
    return checks


def probe_installed_rwkv_source() -> tuple[dict[str, Any], bytes]:
    distribution = metadata.distribution("rwkv")
    source_path = Path(distribution.locate_file("rwkv/model.py")).resolve()
    source_bytes = source_path.read_bytes()
    facts = {
        "package": "rwkv",
        "version": distribution.version,
        "model_source_path": "rwkv/model.py",
        "model_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_source_size_bytes": len(source_bytes),
        "access_method": "importlib.metadata_and_read_bytes",
    }
    return facts, source_bytes


def _report_digest_valid(report: Mapping[str, Any], digest_field: str) -> bool:
    claimed = report.get(digest_field)
    payload = dict(report)
    payload.pop(digest_field, None)
    return isinstance(claimed, str) and sha256_json(payload) == claimed


def _runtime_source_audit(source: str) -> dict[str, bool]:
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    return {
        "runtime_has_no_rwkv_import": "rwkv" not in imported_roots,
        "runtime_has_no_torch_import": "torch" not in imported_roots,
        "runtime_targets_rwkv_x070": f'TARGET_CLASS = "{TARGET_CLASS}"' in source,
        "runtime_targets_both_paths": (
            'TARGET_METHODS = ("forward_one", "forward_seq")' in source
        ),
        "runtime_sets_callback_to_none": (
            f"setattr(self._base_model, CALLBACK_ATTRIBUTE, None)" in source
        ),
        "runtime_restores_in_finally": (
            "finally:" in source and "instance_dict.pop(name, None)" in source
        ),
        "active_method_only_rejects": (
            "active injection is not implemented or authorized in OFF-G2" in source
        ),
    }


def build_instrumented_off_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    installed_source: Mapping[str, Any],
    upstream_source_bytes: bytes,
    d3_report: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / IMPLEMENTATION_CONFIG_FILE).resolve():
        raise PermissionError("instrumented-off config path is not frozen")
    config = _object(config_file, "instrumented-off config")
    config_checks = validate_instrumented_off_config(config)

    implementation_path = _project_file(
        root, config["implementation"]["path"], "implementation path"
    )
    d3_config_path = _project_file(
        root, config["d3_prerequisite"]["config_path"], "D3 config"
    )
    implementation_source = implementation_path.read_text(encoding="utf-8")
    source_checks = {
        "implementation_digest_matches_frozen": (
            sha256_file(implementation_path) == config["implementation"]["sha256"]
        ),
        "d3_config_digest_matches_frozen": (
            sha256_file(d3_config_path)
            == config["d3_prerequisite"]["config_sha256"]
        ),
        **_runtime_source_audit(implementation_source),
    }

    upstream = config["upstream"]
    installed_checks = {
        "installed_package_matches": installed_source.get("package")
        == upstream["package"],
        "installed_version_matches": installed_source.get("version")
        == upstream["version"],
        "installed_source_path_matches": installed_source.get("model_source_path")
        == upstream["model_source_path"],
        "installed_source_digest_matches": (
            installed_source.get("model_source_sha256")
            == upstream["model_source_sha256"]
            == hashlib.sha256(upstream_source_bytes).hexdigest()
        ),
        "installed_source_size_matches": (
            installed_source.get("model_source_size_bytes")
            == upstream["model_source_size_bytes"]
            == len(upstream_source_bytes)
        ),
        "installed_source_access_is_bytes_only": (
            installed_source.get("access_method")
            == "importlib.metadata_and_read_bytes"
        ),
    }
    transformation = inspect_instrumented_source(
        upstream_source_bytes.decode("utf-8")
    )
    transform_checks = {
        f"upstream_transform_{name}": valid
        for name, valid in transformation["checks"].items()
    }
    d3_checks = d3_report.get("checks")
    d3_sources = d3_report.get("source_digests")
    d3_safety = d3_report.get("safety")
    prerequisite_checks = {
        "d3_report_digest_self_valid": _report_digest_valid(
            d3_report, "report_digest_sha256"
        ),
        "d3_report_digest_matches_frozen": (
            d3_report.get("report_digest_sha256")
            == config["d3_prerequisite"]["report_digest_sha256"]
        ),
        "d3_report_valid": d3_report.get("valid") is True,
        "d3_checks_complete": (
            isinstance(d3_checks, Mapping)
            and len(d3_checks) == 33
            and all(value is True for value in d3_checks.values())
        ),
        "d3_source_inventory_complete": (
            isinstance(d3_sources, Mapping) and len(d3_sources) == 10
        ),
        "d3_safety_is_static_only": (
            isinstance(d3_safety, Mapping)
            and d3_safety.get("installed_rwkv_source_probed") is True
            and d3_safety.get("rwkv_model_imported") is False
            and d3_safety.get("torch_imported") is False
            and d3_safety.get("weights_accessed") is False
            and d3_safety.get("model_loaded") is False
            and d3_safety.get("model_executed") is False
            and d3_safety.get("off_g2_implemented") is False
            and d3_safety.get("active_injection_implemented") is False
        ),
    }
    module_checks = {
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    all_checks = {
        **config_checks,
        **source_checks,
        **installed_checks,
        **transform_checks,
        **prerequisite_checks,
        **module_checks,
    }
    valid = all(all_checks.values())
    report = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "stage": "D3B_off_g2_instrumented_off_implementation_without_model",
        "status": (
            "instrumented_off_runtime_static_verified"
            if valid
            else "instrumented_off_runtime_static_failed"
        ),
        "valid": valid,
        "development_only": True,
        "installed_source": dict(installed_source),
        "transformation": {
            "target_class": TARGET_CLASS,
            "execution_paths": list(TARGET_METHODS),
            "injection_counts": transformation["injection_counts"],
            "method_source_sha256": transformation["method_source_sha256"],
        },
        "checks": all_checks,
        "off_gates": {
            "off_g1_implemented": True,
            "off_g2_implemented": True,
            "real_model_equivalence_executed": False,
        },
        "source_digests": {
            relative: sha256_file(root / relative)
            for relative in IMPLEMENTATION_SOURCE_FILES
        },
        "safety": {
            "installed_rwkv_source_probed": True,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "site_packages_modified": False,
            "off_only_adapter_implemented": True,
            "instrumented_runtime_implemented": True,
            "off_g2_implemented": True,
            "real_model_equivalence_executed": False,
            "callback_constructed": False,
            "self_projection_constructed": False,
            "active_injection_implemented": False,
            "real_layers_selected": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
