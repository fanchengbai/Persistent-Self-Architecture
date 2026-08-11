from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.state import FIELD_UPDATE_CLASSES, SELF_FIELDS


OFFLINE_CONTRACT_VERSION = "0.1-offline-draft"
OFFLINE_MANIFEST_VERSION = "0.1-offline"
SELF_MODEL_V01_SOURCE_FILES = (
    "configs/development/self_model_v0_1_offline.draft.json",
    "docs/self_model_v0_1_design.md",
    "schemas/self_model_v0_1_offline_contract.schema.json",
    "schemas/self_model_v0_1_offline_manifest.schema.json",
    "schemas/self_state_v0_1.schema.json",
    "scripts/build_self_model_v0_1_offline_manifest.py",
    "src/psa/self_model/__init__.py",
    "src/psa/self_model/contract.py",
    "src/psa/self_model/coupling.py",
    "src/psa/self_model/encoding.py",
    "src/psa/self_model/state.py",
    "tests/test_self_model_v0_1.py",
)


def _object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_self_model_v0_1_offline_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    encoder = contract.get("offline_encoder")
    coupling = contract.get("offline_coupling")
    authority = contract.get("authority")
    if (
        contract.get("contract_version") != OFFLINE_CONTRACT_VERSION
        or contract.get("phase") != "phase3_explicit_self_model_minimum_prototype"
        or contract.get("status") != "offline_interface_only_real_model_unapproved"
        or contract.get("development_only") is not True
        or contract.get("static_self_only") is not True
        or contract.get("automatic_self_updater_included") is not False
        or contract.get("self_fields") != list(SELF_FIELDS)
        or contract.get("field_update_classes")
        != {key: sorted(value) for key, value in FIELD_UPDATE_CLASSES.items()}
        or not isinstance(encoder, Mapping)
        or encoder.get("kind") != "deterministic_hash_fake_encoder"
        or encoder.get("embedding_dimension") != 16
        or encoder.get("natural_language_prompt_serialization_forbidden") is not True
        or encoder.get("model_loaded") is not False
        or not isinstance(coupling, Mapping)
        or coupling.get("kind") != "fake_gated_residual_adapter"
        or coupling.get("candidate_layers") != ["fake-layer-00", "fake-layer-01"]
        or coupling.get("default_gate") != 0.5
        or coupling.get("minimum_scale") != 0.0
        or coupling.get("maximum_scale") != 2.0
        or coupling.get("coupling_off_required") is not True
        or coupling.get("field_mask_required") is not True
        or coupling.get("layer_mask_required") is not True
        or not isinstance(authority, Mapping)
        or authority.get("offline_design_authorized") is not True
        or authority.get("offline_fake_adapter_tests_authorized") is not True
        or authority.get("real_rwkv_coupling_implementation_authorized") is not False
        or authority.get("model_execution_authorized") is not False
        or authority.get("noncore_effect_experiment_authorized") is not False
        or authority.get("formal_test_set_access_authorized") is not False
        or authority.get("formal_self_experiment_authorized") is not False
        or authority.get("automatic_rerun_authorized") is not False
    ):
        raise PermissionError("Self Model v0.1 contract is offline-only")
    return dict(contract)


def build_self_model_v0_1_offline_manifest(
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    contract = validate_self_model_v0_1_offline_contract(
        _object(config_file, "Self Model v0.1 offline contract")
    )
    source_digests = {
        relative: sha256_file(root / relative)
        for relative in SELF_MODEL_V01_SOURCE_FILES
    }
    manifest = {
        "manifest_version": OFFLINE_MANIFEST_VERSION,
        "status": "self_model_v0_1_offline_interfaces_verified",
        "valid": True,
        "development_only": True,
        "static_self_only": True,
        "automatic_self_updater_included": False,
        "model_loaded": False,
        "model_executed": False,
        "natural_language_prompt_serialization_used": False,
        "real_rwkv_coupling_implemented": False,
        "noncore_effect_experiment_run": False,
        "formal_test_set_accessed": False,
        "formal_self_experiment_run": False,
        "automatic_rerun_authorized": False,
        "config_sha256": sha256_file(config_file),
        "self_fields": contract["self_fields"],
        "offline_embedding_dimension": contract["offline_encoder"][
            "embedding_dimension"
        ],
        "offline_candidate_layers": contract["offline_coupling"][
            "candidate_layers"
        ],
        "source_digests": dict(sorted(source_digests.items())),
    }
    manifest["manifest_digest_sha256"] = sha256_json(manifest)
    return manifest


def verify_self_model_v0_1_offline_manifest(
    *,
    manifest_path: str | Path,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    persisted = _object(Path(manifest_path).resolve(), "Self Model v0.1 manifest")
    rebuilt = build_self_model_v0_1_offline_manifest(
        config_path=config_path,
        project_root=project_root,
    )
    valid = persisted == rebuilt
    return {
        "verification_version": OFFLINE_MANIFEST_VERSION,
        "status": "self_model_v0_1_offline_manifest_verified" if valid else "invalid",
        "valid": valid,
        "manifest_digest_sha256": persisted.get("manifest_digest_sha256"),
        "model_loaded": False,
        "model_executed": False,
        "real_rwkv_coupling_implemented": False,
    }
