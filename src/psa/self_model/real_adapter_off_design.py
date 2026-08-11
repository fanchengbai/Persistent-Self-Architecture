from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-off-equivalence-draft"
DESIGN_CONFIG_FILE = (
    "configs/development/self_model_v0_1_real_adapter_off_design.draft.json"
)
DESIGN_SOURCE_FILES = (
    DESIGN_CONFIG_FILE,
    "docs/self_model_v0_1_real_adapter_off_design.md",
    "schemas/self_model_v0_1_real_adapter_off_design_report.schema.json",
    "scripts/verify_self_model_v0_1_real_adapter_off_design.py",
    "src/psa/self_model/real_adapter_off_design.py",
    "tests/test_self_model_real_adapter_off_design.py",
    "configs/development/self_model_v0_1_rwkv_interface_audit.json",
    "configs/development/self_model_v0_1_fake_callback.draft.json",
    "configs/models/rwkv7_g1h_2.9b.candidate.json",
)


def _load_object(path: Path, label: str) -> dict[str, Any]:
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


def validate_real_adapter_off_design(
    *,
    design: Mapping[str, Any],
    interface_audit: Mapping[str, Any],
    fake_callback: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, bool]:
    upstream = design.get("upstream")
    adapter = design.get("adapter")
    gates = design.get("off_equivalence_gates")
    protocol = design.get("future_model_test_protocol")
    authority = design.get("authority")
    prerequisites = design.get("prerequisites")
    future_gates = design.get("future_gates")
    if not all(
        isinstance(value, Mapping)
        for value in (
            upstream,
            adapter,
            protocol,
            authority,
            prerequisites,
        )
    ) or not isinstance(gates, list):
        raise ValueError("real adapter off design structure is incomplete")

    expected_authority = {
        "design_authorized": True,
        "installed_source_read_authorized": True,
        "real_adapter_implementation_authorized": False,
        "active_injection_implementation_authorized": False,
        "rwkv_model_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "site_packages_modification_authorized": False,
        "real_layer_selection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    audit_package = interface_audit.get("package")
    audit_runtime = interface_audit.get("expected_runtime_environment")
    audit_authority = interface_audit.get("authority")
    model_runtime = model_config.get("runtime")
    model_architecture = model_config.get("architecture_hint")
    fake_authority = fake_callback.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (
            audit_package,
            audit_runtime,
            audit_authority,
            model_runtime,
            model_architecture,
            fake_authority,
        )
    ):
        raise ValueError("prerequisite configs are incomplete")

    gate_ids = [gate.get("gate_id") for gate in gates if isinstance(gate, Mapping)]
    checks = {
        "design_identity_valid": (
            design.get("design_version") == DESIGN_VERSION
            and design.get("status") == "design_only_unimplemented"
            and design.get("development_only") is True
        ),
        "prerequisite_paths_fixed": dict(prerequisites)
        == {
            "rwkv_interface_audit_config": (
                "configs/development/self_model_v0_1_rwkv_interface_audit.json"
            ),
            "fake_callback_config": (
                "configs/development/self_model_v0_1_fake_callback.draft.json"
            ),
            "model_config": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        },
        "upstream_matches_static_audit": (
            upstream.get("package") == audit_package.get("name") == "rwkv"
            and upstream.get("version") == audit_package.get("version") == "0.8.32"
            and upstream.get("model_source_sha256")
            == audit_package.get("model_source_sha256")
            == "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
            and upstream.get("runtime_environment") == audit_runtime
            and upstream.get("active_class") == "RWKV_x070"
        ),
        "model_config_matches_upstream": (
            model_config.get("model_id") == "rwkv7-g1h-2.9b-20260710"
            and model_config.get("architecture") == "RWKV-7"
            and model_runtime.get("rwkv") == upstream.get("version")
            and model_runtime.get("environment") == upstream.get("runtime_environment")
            and model_architecture.get("n_layer") == 32
            and model_architecture.get("n_embd") == 2560
            and model_architecture.get("head_size") == 64
        ),
        "adapter_is_project_local_and_unimplemented": (
            adapter.get("future_project_path")
            == "src/psa/self_model/rwkv7_coupling_adapter.py"
            and adapter.get("project_local_only") is True
            and adapter.get("site_packages_modification_forbidden") is True
            and adapter.get("active_injection_available") is False
        ),
        "real_choices_remain_unfrozen": (
            adapter.get("real_layer_mask") == []
            and adapter.get("real_sequence_policy") == "unfrozen"
            and adapter.get("candidate_phase_family") == "post_ffn_residual"
        ),
        "both_execution_paths_required": adapter.get("required_execution_paths")
        == ["forward_one", "forward_seq"],
        "state_ownership_preserves_upstream_semantics": adapter.get(
            "state_ownership"
        )
        == "runner_clones_source_adapter_preserves_upstream_mutation",
        "two_off_gates_are_ordered": gates
        == [
            {
                "gate_id": "OFF-G1",
                "name": "passthrough_wrapper_off",
                "implementation": "delegate_directly_to_original_model_forward",
                "callback_constructed": False,
                "self_projection_constructed": False,
                "required_before": "instrumented_runtime_implementation",
            },
            {
                "gate_id": "OFF-G2",
                "name": "instrumented_runtime_off",
                "implementation": (
                    "project_local_dual_path_runtime_with_callback_bypassed"
                ),
                "callback_constructed": False,
                "self_projection_constructed": False,
                "required_before": "any_active_injection_design_or_execution",
            },
        ]
        and gate_ids == ["OFF-G1", "OFF-G2"],
        "off_gates_construct_no_self_path": all(
            isinstance(gate, Mapping)
            and gate.get("callback_constructed") is False
            and gate.get("self_projection_constructed") is False
            for gate in gates
        ),
        "future_model_test_is_not_authorized": protocol.get(
            "currently_authorized"
        )
        is False,
        "future_test_uses_strict_exactness": protocol.get("required_checks")
        == [
            "logits_torch_equal",
            "all_state_tensors_torch_equal",
            "shape_dtype_device_equal",
            "source_snapshot_unchanged",
            "callback_call_count_zero",
            "self_projection_not_constructed",
        ],
        "future_test_covers_paths_states_and_full_output": (
            protocol.get("non_core_token_ids_only") is True
            and protocol.get("same_process") is True
            and protocol.get("paths") == ["forward_one", "forward_seq"]
            and protocol.get("state_inputs")
            == ["none", "cloned_restored_snapshot"]
            and protocol.get("sequence_full_output_modes") == [False, True]
            and protocol.get("identical_shape_warmup_count") == 1
        ),
        "failure_action_is_fail_closed": protocol.get("failure_action")
        == "stop_without_active_injection_or_tolerance_revision",
        "future_gate_order_is_fixed": future_gates
        == [
            "D1_design_static_verification",
            "D2_off_only_adapter_local_implementation_without_model",
            "D3_off_only_adapter_cloud_static_verification_without_model",
            "D4_separately_authorized_2_9b_off_equivalence_execution",
            "D5_new_design_and_authorization_before_active_injection",
        ],
        "authority_is_design_only": dict(authority) == expected_authority,
        "fake_stage_did_not_authorize_real_work": (
            audit_authority.get("source_read_authorized") is True
            and audit_authority.get("real_hook_implementation_authorized")
            is False
            and audit_authority.get("model_execution_authorized") is False
            and audit_authority.get("layer_selection_authorized") is False
            and fake_authority.get("rwkv_model_import_authorized") is False
            and fake_authority.get("weights_access_authorized") is False
            and fake_authority.get("model_execution_authorized") is False
            and fake_authority.get("real_layer_selection_authorized") is False
            and fake_authority.get("self_effect_experiment_authorized") is False
        ),
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError(
            "real adapter off design failed closed: " + ", ".join(failed)
        )
    return checks


def probe_installed_rwkv_source() -> dict[str, Any]:
    distribution = metadata.distribution("rwkv")
    source_path = Path(distribution.locate_file("rwkv")) / "model.py"
    source_bytes = source_path.read_bytes()
    return {
        "package_version": distribution.version,
        "model_source_path": str(source_path),
        "model_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_size_bytes": len(source_bytes),
    }


def build_real_adapter_off_design_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    installed_source: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / DESIGN_CONFIG_FILE).resolve():
        raise PermissionError("real adapter design config path is not frozen")
    design = _load_object(config_file, "real adapter off design")
    prerequisites = design.get("prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise ValueError("design prerequisites are missing")
    audit = _load_object(
        _project_file(
            root,
            prerequisites.get("rwkv_interface_audit_config"),
            "rwkv_interface_audit_config",
        ),
        "RWKV interface audit config",
    )
    fake = _load_object(
        _project_file(
            root,
            prerequisites.get("fake_callback_config"),
            "fake_callback_config",
        ),
        "fake callback config",
    )
    model = _load_object(
        _project_file(
            root, prerequisites.get("model_config"), "model_config"
        ),
        "model config",
    )
    checks = validate_real_adapter_off_design(
        design=design,
        interface_audit=audit,
        fake_callback=fake,
        model_config=model,
    )
    upstream = design["upstream"]
    installed_checks = {
        "installed_package_version_matches": installed_source.get(
            "package_version"
        )
        == upstream["version"],
        "installed_model_source_sha256_matches": installed_source.get(
            "model_source_sha256"
        )
        == upstream["model_source_sha256"],
        "installed_source_was_read_without_model_import": "rwkv.model"
        not in sys.modules,
    }
    future_adapter_path = _project_file(
        root, design["adapter"]["future_project_path"], "future_project_path"
    )
    implementation_checks = {
        "future_adapter_file_absent": not future_adapter_path.exists(),
        "real_adapter_implementation_authorized_false": design["authority"][
            "real_adapter_implementation_authorized"
        ]
        is False,
        "active_injection_implementation_authorized_false": design[
            "authority"
        ]["active_injection_implementation_authorized"]
        is False,
    }
    all_checks = {**checks, **installed_checks, **implementation_checks}
    valid = all(all_checks.values())
    report = {
        "design_version": DESIGN_VERSION,
        "status": (
            "real_adapter_off_design_static_verification_complete"
            if valid
            else "real_adapter_off_design_static_verification_failed"
        ),
        "valid": valid,
        "development_only": True,
        "checks": all_checks,
        "installed_source": dict(installed_source),
        "off_equivalence_gates": design["off_equivalence_gates"],
        "future_gates": design["future_gates"],
        "source_digests": {
            relative: sha256_file(root / relative)
            for relative in DESIGN_SOURCE_FILES
        },
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "site_packages_modified": False,
            "real_adapter_implemented": False,
            "active_injection_implemented": False,
            "real_layers_selected": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
