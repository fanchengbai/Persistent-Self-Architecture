from __future__ import annotations

from importlib import metadata
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_json


AUDIT_VERSION = "0.1-read-only"

_SOURCE_MARKERS = {
    "jit_disabled_uses_torch_module": "MyModule = torch.nn.Module",
    "rwkv7_implementation_class": "class RWKV_x070(MyModule):",
    "zero_state_has_three_components_per_layer": (
        "state = [None for _ in range(self.args.n_layer * 3)]"
    ),
    "zero_state_role_order": (
        "state: 0=att_x_prev 1=att_kv 2=ffn_x_prev"
    ),
    "public_forward": "def forward(self, idx, state, full_output=False):",
    "none_state_is_initialized": "if state == None:",
    "sequence_dispatch": "return self.forward_seq(idx, state, full_output)",
    "single_token_dispatch": "return self.forward_one(idx[0], state)",
    "single_embedding_residual": "x = z['emb.weight'][idx]",
    "sequence_embedding_residual": "x = z['emb.weight'][idx]",
    "layer_loop": "for i in range(self.n_layer):",
    "attention_state_update": (
        "xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix"
    ),
    "attention_residual_add": "x = x + xx",
    "ffn_state_update": "xx, state[i*3+2] = RWKV_x070_CMix",
    "output_layer_norm": "x = F.layer_norm(x, (self.n_embd,)",
    "output_head": "x = x @ z['head.weight']",
    "rwkv7_alias_guard": "if os.environ.get('RWKV_V7_ON') == '1':",
    "rwkv7_public_alias": "RWKV = RWKV_x070",
}


def _line_evidence(source: str, marker: str) -> list[dict[str, Any]]:
    return [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(source.splitlines(), start=1)
        if marker in line
    ]


def inspect_rwkv_source(
    *,
    source: str,
    package_version: str,
    source_sha256: str,
    expected_version: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    evidence = {
        name: _line_evidence(source, marker)
        for name, marker in _SOURCE_MARKERS.items()
    }
    checks = {
        "package_version_matches": package_version == expected_version,
        "source_sha256_matches": source_sha256 == expected_source_sha256,
        "all_required_source_markers_present": all(evidence.values()),
        "two_or_more_residual_add_sites_present": (
            len(evidence["attention_residual_add"]) >= 2
        ),
        "single_and_sequence_paths_present": bool(
            evidence["single_token_dispatch"]
            and evidence["sequence_dispatch"]
        ),
        "rwkv7_public_alias_present": bool(
            evidence["rwkv7_alias_guard"] and evidence["rwkv7_public_alias"]
        ),
    }
    return {
        "checks": checks,
        "evidence": evidence,
        "valid": all(checks.values()),
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _project_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} must stay inside the project root")
    resolved = (root / value).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the project root")
    return resolved


def build_rwkv_interface_audit(
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_object(config_file.resolve(), "RWKV interface audit config")

    if config.get("audit_version") != AUDIT_VERSION:
        raise ValueError("unsupported RWKV interface audit version")
    authority = config.get("authority")
    if not isinstance(authority, Mapping) or any(
        authority.get(field) is not expected
        for field, expected in {
            "source_read_authorized": True,
            "rwkv_model_import_authorized": False,
            "weights_access_authorized": False,
            "model_execution_authorized": False,
            "real_hook_implementation_authorized": False,
            "layer_selection_authorized": False,
        }.items()
    ):
        raise PermissionError("audit authority must remain read-only")

    package = config.get("package")
    if not isinstance(package, Mapping) or package.get("name") != "rwkv":
        raise ValueError("package must identify rwkv")
    expected_version = package.get("version")
    expected_source_sha256 = package.get("model_source_sha256")
    if not isinstance(expected_version, str) or not isinstance(
        expected_source_sha256, str
    ):
        raise ValueError("package version and model source digest are required")

    distribution = metadata.distribution("rwkv")
    source_path = Path(distribution.locate_file("rwkv")) / "model.py"
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    inspection = inspect_rwkv_source(
        source=source,
        package_version=distribution.version,
        source_sha256=source_sha256,
        expected_version=expected_version,
        expected_source_sha256=expected_source_sha256,
    )

    model_config_path = _project_path(
        root, config.get("model_config"), "model_config"
    )
    model_config = _load_object(model_config_path, "model config")
    runtime = model_config.get("runtime")
    architecture = model_config.get("architecture_hint")
    expected_runtime = config.get("expected_runtime_environment")
    expected_architecture = config.get("expected_architecture")
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get("environment"), Mapping
    ):
        raise ValueError("model runtime environment is missing")
    if not isinstance(architecture, Mapping):
        raise ValueError("model architecture hint is missing")

    runtime_matches = dict(runtime["environment"]) == expected_runtime
    architecture_matches = all(
        architecture.get(key) == value
        for key, value in dict(expected_architecture).items()
    ) if isinstance(expected_architecture, Mapping) else False
    valid = bool(inspection["valid"] and runtime_matches and architecture_matches)
    report = {
        "audit_version": AUDIT_VERSION,
        "status": (
            "rwkv7_coupling_interface_static_audit_complete"
            if valid
            else "rwkv7_coupling_interface_static_audit_failed"
        ),
        "valid": valid,
        "development_only": True,
        "package": {
            "name": "rwkv",
            "version": distribution.version,
            "model_source_path": str(source_path),
            "model_source_sha256": source_sha256,
        },
        "runtime": {
            "configured_environment": dict(runtime["environment"]),
            "matches_expected": runtime_matches,
            "active_public_class_when_configured": "RWKV_x070",
            "jit_mode": "torch.nn.Module",
            "per_block_module_hooks_exposed": False,
        },
        "model_interface": {
            "model_id": model_config.get("model_id"),
            "architecture_hint": dict(architecture),
            "matches_expected": architecture_matches,
            "state_components_per_layer": 3,
            "expected_total_state_components": int(
                architecture.get("n_layer", 0)
            ) * 3,
            "state_argument_mutated_in_place": True,
            "single_and_sequence_paths_require_same_coupling_semantics": True,
        },
        "source_inspection": inspection,
        "candidate_boundaries": [
            {
                "name": "pre_layer_residual",
                "tensor": "x",
                "shape_by_path": {
                    "single_token": ["n_embd"],
                    "sequence": ["T", "n_embd"],
                },
                "status": "candidate_not_selected",
            },
            {
                "name": "post_attention_residual",
                "tensor": "x",
                "shape_by_path": {
                    "single_token": ["n_embd"],
                    "sequence": ["T", "n_embd"],
                },
                "status": "candidate_not_selected",
            },
            {
                "name": "post_ffn_residual",
                "tensor": "x",
                "shape_by_path": {
                    "single_token": ["n_embd"],
                    "sequence": ["T", "n_embd"],
                },
                "status": "preferred_minimum_prototype_family_layer_unselected",
            },
        ],
        "implementation_boundary": {
            "public_root_forward_hook_is_sufficient": False,
            "reason": (
                "RWKV_x070 stores block weights in a mapping and performs block "
                "operations functionally inside forward_one/forward_seq"
            ),
            "minimum_future_change": (
                "an explicit residual callback in both forward_one and forward_seq, "
                "or a locally maintained adapter subclass/fork"
            ),
            "callback_contract": (
                "callback(phase, layer_index, residual_x, self_vector) -> residual_x"
            ),
        },
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_hook_implemented": False,
            "final_layers_selected": False,
        },
    }
    report["audit_digest_sha256"] = sha256_json(report)
    return report
