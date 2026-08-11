from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.off_only_adapter_manifest import audit_off_only_adapter_source


VERIFICATION_VERSION = "0.1-d3-static"
VERIFICATION_CONFIG_FILE = (
    "configs/development/self_model_v0_1_d3_static_verification.json"
)
VERIFICATION_SOURCE_FILES = (
    VERIFICATION_CONFIG_FILE,
    "docs/self_model_v0_1_d3_static_verification.md",
    "schemas/self_model_v0_1_d3_static_report.schema.json",
    "scripts/verify_self_model_v0_1_d3_static.py",
    "src/psa/self_model/d3_static_verification.py",
    "tests/test_self_model_d3_static_verification.py",
    "src/psa/self_model/rwkv7_coupling_adapter.py",
    "src/psa/self_model/off_only_adapter_manifest.py",
    "configs/development/self_model_v0_1_off_only_adapter.draft.json",
    "schemas/self_model_v0_1_off_only_adapter_report.schema.json",
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


def validate_d3_config(config: Mapping[str, Any]) -> dict[str, bool]:
    upstream = config.get("upstream")
    wrapper = config.get("wrapper")
    d2 = config.get("d2_prerequisite")
    verification = config.get("verification")
    authority = config.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (upstream, wrapper, d2, verification, authority)
    ):
        raise ValueError("D3 static verification config is incomplete")
    expected_authority = {
        "installed_source_verification_authorized": True,
        "rwkv_model_import_authorized": False,
        "torch_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "site_packages_modification_authorized": False,
        "instrumented_runtime_implementation_authorized": False,
        "off_g2_implementation_authorized": False,
        "active_injection_implementation_authorized": False,
        "real_layer_selection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    checks = {
        "config_identity_valid": (
            config.get("verification_version") == VERIFICATION_VERSION
            and config.get("stage")
            == "D3_off_only_adapter_cloud_static_verification_without_model"
            and config.get("status")
            == "installed_source_bytes_and_project_wrapper_only"
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
        },
        "wrapper_lock_frozen": dict(wrapper)
        == {
            "path": "src/psa/self_model/rwkv7_coupling_adapter.py",
            "sha256": (
                "74683bcdc5395e9fcbd2f603d452bb047766b2ad78722bfcc42e757d1f8178d3"
            ),
            "off_g1_implemented": True,
            "off_g2_implemented": False,
        },
        "d2_prerequisite_frozen": dict(d2)
        == {
            "config_path": (
                "configs/development/"
                "self_model_v0_1_off_only_adapter.draft.json"
            ),
            "config_sha256": (
                "74ca109fb3c4b305a0602817c1ba6b9766344b2c5c910a90825f89c30e75cf27"
            ),
            "report_path": (
                "results/development/"
                "self_model_v0_1_off_only_adapter/report.json"
            ),
            "report_digest_sha256": (
                "527fc6ed2cf308bdc485cae6d2738d432cba01a7506f863f4615d36be4a22e6c"
            ),
        },
        "verification_scope_is_static_only": dict(verification)
        == {
            "installed_package_metadata_read": True,
            "installed_model_source_bytes_read": True,
            "wrapper_digest_and_ast_audited": True,
            "d2_report_and_source_digests_reverified": True,
            "rwkv_model_import_included": False,
            "torch_import_included": False,
            "weights_access_included": False,
            "model_load_included": False,
            "model_execution_included": False,
        },
        "authority_is_d3_static_only": dict(authority) == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D3 config failed closed: " + ", ".join(failed))
    return checks


def probe_installed_rwkv_source() -> dict[str, Any]:
    """Read package metadata and model.py bytes without importing rwkv or torch."""
    distribution = metadata.distribution("rwkv")
    source = Path(distribution.locate_file("rwkv/model.py")).resolve()
    source_bytes = source.read_bytes()
    import hashlib

    return {
        "package": "rwkv",
        "version": distribution.version,
        "model_source_path": "rwkv/model.py",
        "model_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_source_size_bytes": len(source_bytes),
        "access_method": "importlib.metadata_and_read_bytes",
    }


def _d2_checks(
    *, report: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> dict[str, bool]:
    d2_config = config["d2_prerequisite"]
    claimed_digest = report.get("report_digest_sha256")
    digest_payload = dict(report)
    digest_payload.pop("report_digest_sha256", None)
    source_digests = report.get("source_digests")
    source_digests_are_current = False
    if isinstance(source_digests, Mapping):
        try:
            source_digests_are_current = all(
                isinstance(relative, str)
                and isinstance(expected, str)
                and sha256_file(_project_file(root, relative, "D2 source"))
                == expected
                for relative, expected in source_digests.items()
            )
        except (OSError, ValueError):
            source_digests_are_current = False
    checks = report.get("checks")
    safety = report.get("safety")
    return {
        "d2_report_digest_self_valid": (
            isinstance(claimed_digest, str)
            and sha256_json(digest_payload) == claimed_digest
        ),
        "d2_report_digest_matches_frozen": (
            claimed_digest == d2_config["report_digest_sha256"]
        ),
        "d2_report_valid": report.get("valid") is True,
        "d2_report_status_valid": (
            report.get("status") == "d2_off_only_adapter_verified"
        ),
        "d2_checks_complete": (
            isinstance(checks, Mapping)
            and len(checks) == 29
            and all(value is True for value in checks.values())
        ),
        "d2_source_inventory_complete": (
            isinstance(source_digests, Mapping) and len(source_digests) == 8
        ),
        "d2_source_digests_current": source_digests_are_current,
        "d2_off_gate_state_preserved": report.get("off_gates")
        == {"off_g1_implemented": True, "off_g2_implemented": False},
        "d2_safety_state_preserved": (
            isinstance(safety, Mapping)
            and safety.get("installed_rwkv_source_probed") is False
            and safety.get("rwkv_model_imported") is False
            and safety.get("torch_imported") is False
            and safety.get("weights_accessed") is False
            and safety.get("model_loaded") is False
            and safety.get("model_executed") is False
            and safety.get("off_only_adapter_implemented") is True
            and safety.get("instrumented_runtime_implemented") is False
            and safety.get("active_injection_implemented") is False
        ),
    }


def build_d3_static_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    installed_source: Mapping[str, Any],
    d2_report: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / VERIFICATION_CONFIG_FILE).resolve():
        raise PermissionError("D3 config path is not frozen")
    config = _object(config_file, "D3 static verification config")
    config_checks = validate_d3_config(config)

    d2_config_path = _project_file(
        root, config["d2_prerequisite"]["config_path"], "D2 config"
    )
    wrapper_path = _project_file(root, config["wrapper"]["path"], "wrapper")
    source_checks = {
        "wrapper_digest_matches_frozen": (
            sha256_file(wrapper_path) == config["wrapper"]["sha256"]
        ),
        "d2_config_digest_matches_frozen": (
            sha256_file(d2_config_path)
            == config["d2_prerequisite"]["config_sha256"]
        ),
    }
    source_checks.update(
        {
            f"wrapper_ast_{name}": valid
            for name, valid in audit_off_only_adapter_source(
                wrapper_path.read_text(encoding="utf-8")
            ).items()
        }
    )
    expected_upstream = config["upstream"]
    installed_checks = {
        "installed_package_matches": installed_source.get("package")
        == expected_upstream["package"],
        "installed_version_matches": installed_source.get("version")
        == expected_upstream["version"],
        "installed_source_path_matches": installed_source.get("model_source_path")
        == expected_upstream["model_source_path"],
        "installed_source_digest_matches": (
            installed_source.get("model_source_sha256")
            == expected_upstream["model_source_sha256"]
        ),
        "installed_source_size_matches": (
            installed_source.get("model_source_size_bytes")
            == expected_upstream["model_source_size_bytes"]
        ),
        "installed_source_access_is_bytes_only": (
            installed_source.get("access_method")
            == "importlib.metadata_and_read_bytes"
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
        **_d2_checks(report=d2_report, config=config, root=root),
        **module_checks,
    }
    valid = all(all_checks.values())
    report = {
        "verification_version": VERIFICATION_VERSION,
        "stage": "D3_off_only_adapter_cloud_static_verification_without_model",
        "status": (
            "d3_cloud_static_verified"
            if valid
            else "d3_cloud_static_verification_failed"
        ),
        "valid": valid,
        "development_only": True,
        "installed_source": dict(installed_source),
        "d2_report": {
            "path": config["d2_prerequisite"]["report_path"],
            "report_digest_sha256": d2_report.get("report_digest_sha256"),
        },
        "checks": all_checks,
        "off_gates": {
            "off_g1_implemented": True,
            "off_g2_implemented": False,
        },
        "source_digests": {
            relative: sha256_file(root / relative)
            for relative in VERIFICATION_SOURCE_FILES
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
            "instrumented_runtime_implemented": False,
            "off_g2_implemented": False,
            "active_injection_implemented": False,
            "real_layers_selected": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
