from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.rwkv7_coupling_adapter import (
    CouplingOffRequest,
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
    RWKV7CouplingOffAdapter,
)


IMPLEMENTATION_VERSION = "0.1-off-only"
IMPLEMENTATION_CONFIG_FILE = (
    "configs/development/self_model_v0_1_off_only_adapter.draft.json"
)
IMPLEMENTATION_SOURCE_FILES = (
    IMPLEMENTATION_CONFIG_FILE,
    "docs/self_model_v0_1_off_only_adapter.md",
    "schemas/self_model_v0_1_off_only_adapter_report.schema.json",
    "scripts/verify_self_model_v0_1_off_only_adapter.py",
    "src/psa/self_model/rwkv7_coupling_adapter.py",
    "src/psa/self_model/off_only_adapter_manifest.py",
    "tests/test_self_model_off_only_adapter.py",
    "configs/development/self_model_v0_1_real_adapter_off_design.draft.json",
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


def validate_off_only_implementation_config(
    config: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, bool]:
    prerequisite = config.get("prerequisite_design")
    upstream = config.get("upstream")
    implementation = config.get("implementation")
    verification = config.get("current_verification")
    authority = config.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (
            prerequisite,
            upstream,
            implementation,
            verification,
            authority,
        )
    ):
        raise ValueError("D2 off-only implementation config is incomplete")
    expected_authority = {
        "off_only_adapter_implementation_authorized": True,
        "active_injection_implementation_authorized": False,
        "installed_source_verification_authorized": False,
        "rwkv_model_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "site_packages_modification_authorized": False,
        "instrumented_runtime_implementation_authorized": False,
        "real_layer_selection_authorized": False,
        "self_effect_experiment_authorized": False,
        "automatic_rerun_authorized": False,
    }
    checks = {
        "implementation_identity_valid": (
            config.get("implementation_version") == IMPLEMENTATION_VERSION
            and config.get("stage")
            == "D2_off_only_adapter_local_implementation_without_model"
            and config.get("status") == "off_g1_passthrough_only"
            and config.get("development_only") is True
        ),
        "prerequisite_design_locked": dict(prerequisite)
        == {
            "path": (
                "configs/development/"
                "self_model_v0_1_real_adapter_off_design.draft.json"
            ),
            "cloud_report_digest_sha256": (
                "dbd64e9a44e99ccc2e9787716dbdac45af38c3c0e90e5d218acbe5a8720ef585"
            ),
        },
        "design_is_d1_only": (
            design.get("design_version") == "0.1-off-equivalence-draft"
            and design.get("status") == "design_only_unimplemented"
            and design.get("authority", {}).get(
                "real_adapter_implementation_authorized"
            )
            is False
        ),
        "upstream_source_lock_matches_constants": (
            upstream.get("package") == "rwkv"
            and upstream.get("version") == EXPECTED_RWKV_PACKAGE_VERSION
            and upstream.get("model_source_sha256")
            == EXPECTED_RWKV_MODEL_SOURCE_SHA256
        ),
        "target_path_matches_design": (
            implementation.get("path")
            == design.get("adapter", {}).get("future_project_path")
            == "src/psa/self_model/rwkv7_coupling_adapter.py"
        ),
        "off_g1_only": (
            implementation.get("project_local_only") is True
            and implementation.get("off_g1_implemented") is True
            and implementation.get("off_g2_implemented") is False
            and implementation.get("instrumented_rwkv_loop_included") is False
        ),
        "delegation_contract_exact": (
            implementation.get("delegation")
            == "base_model.forward(tokens, state, full_output)"
            and implementation.get("tokens_identity_preserved") is True
            and implementation.get("state_identity_preserved") is True
            and implementation.get("full_output_preserved") is True
        ),
        "self_path_absent": (
            implementation.get("callback_call_count") == 0
            and implementation.get("self_projection_constructed") is False
            and implementation.get("active_injection_available") is False
            and implementation.get("real_layer_mask") == []
            and implementation.get("real_sequence_policy") == "unfrozen"
        ),
        "verification_is_fake_only": dict(verification)
        == {
            "fake_base_only": True,
            "installed_rwkv_source_probe_included": False,
            "model_import_included": False,
            "weights_access_included": False,
            "model_execution_included": False,
            "d3_cloud_static_gate_completed": False,
            "d4_real_2_9b_off_gate_completed": False,
        },
        "authority_is_d2_only": dict(authority) == expected_authority,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D2 config failed closed: " + ", ".join(failed))
    return checks


def audit_off_only_adapter_source(source: str) -> dict[str, bool]:
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    return {
        "no_rwkv_import": "rwkv" not in imported_roots,
        "no_torch_import": "torch" not in imported_roots,
        "no_importlib_import": "importlib" not in imported_roots,
        "off_adapter_class_present": "class RWKV7CouplingOffAdapter" in source,
        "direct_delegation_present": (
            "return self._base_model.forward(tokens, state, full_output)" in source
        ),
        "no_instrumented_phase_marker": "post_ffn_residual" not in source,
        "no_projection_method": "def project(" not in source,
        "active_method_only_rejects": (
            "active injection is not implemented or authorized in D2" in source
        ),
    }


class _FakeBaseModel:
    def __init__(self) -> None:
        self.logits = object()
        self.calls: list[dict[str, Any]] = []

    def forward(self, tokens: Any, state: Any, full_output: bool = False) -> Any:
        self.calls.append(
            {
                "tokens": tokens,
                "state": state,
                "full_output": full_output,
                "tokens_id": id(tokens),
                "state_id": id(state),
            }
        )
        state[0][0]["count"] += 1
        return self.logits, state


def build_off_only_adapter_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / IMPLEMENTATION_CONFIG_FILE).resolve():
        raise PermissionError("D2 config path is not frozen")
    config = _object(config_file, "D2 off-only implementation config")
    prerequisite = config.get("prerequisite_design")
    if not isinstance(prerequisite, Mapping):
        raise ValueError("D2 prerequisite design is missing")
    design = _object(
        _project_file(root, prerequisite.get("path"), "prerequisite design"),
        "D1 design",
    )
    config_checks = validate_off_only_implementation_config(config, design)
    target_path = _project_file(
        root, config["implementation"]["path"], "D2 adapter path"
    )
    source_checks = audit_off_only_adapter_source(
        target_path.read_text(encoding="utf-8")
    )

    base = _FakeBaseModel()
    adapter = RWKV7CouplingOffAdapter(
        base_model=base,
        upstream_package_version=EXPECTED_RWKV_PACKAGE_VERSION,
        upstream_model_source_sha256=EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    )
    tokens = [3, 5, 8]
    state = [[{"count": 0}]]
    logits, next_state = adapter.forward(
        tokens, state, True, coupling=CouplingOffRequest()
    )
    fake_checks = {
        "tokens_identity_preserved": base.calls[0]["tokens"] is tokens,
        "state_identity_preserved": base.calls[0]["state"] is state,
        "full_output_preserved": base.calls[0]["full_output"] is True,
        "logits_identity_preserved": logits is base.logits,
        "returned_state_identity_preserved": next_state is state,
        "upstream_state_mutation_preserved": state[0][0]["count"] == 1,
        "single_direct_delegation_recorded": adapter.delegation_count == 1,
        "callback_call_count_zero": adapter.callback_call_count == 0,
        "self_projection_not_constructed": (
            adapter.self_projection_constructed is False
        ),
    }
    calls_before_rejection = len(base.calls)
    try:
        adapter.forward(tokens, state, coupling={"enabled": True})
        active_rejected = False
    except PermissionError:
        active_rejected = True
    fake_checks["active_request_rejected_before_base_call"] = bool(
        active_rejected and len(base.calls) == calls_before_rejection
    )
    try:
        adapter.forward_active(tokens, state)
        active_method_rejected = False
    except PermissionError:
        active_method_rejected = True
    fake_checks["active_method_rejected_before_base_call"] = bool(
        active_method_rejected and len(base.calls) == calls_before_rejection
    )

    all_checks = {**config_checks, **source_checks, **fake_checks}
    valid = all(all_checks.values())
    report = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": (
            "d2_off_only_adapter_verified"
            if valid
            else "d2_off_only_adapter_failed"
        ),
        "valid": valid,
        "development_only": True,
        "checks": all_checks,
        "off_gates": {
            "off_g1_implemented": True,
            "off_g2_implemented": False,
        },
        "source_digests": {
            relative: sha256_file(root / relative)
            for relative in IMPLEMENTATION_SOURCE_FILES
        },
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "installed_rwkv_source_probed": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "site_packages_modified": False,
            "off_only_adapter_implemented": True,
            "instrumented_runtime_implemented": False,
            "active_injection_implemented": False,
            "real_layers_selected": False,
            "self_effect_experiment_run": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
