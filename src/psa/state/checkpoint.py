from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from uuid import uuid4

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)
from psa.model.rwkv7 import (
    RWKV7Adapter,
    RWKV7ModelConfig,
    _flatten_state,
    clone_state,
    compare_states,
    compare_tensors,
    inventory_state,
    load_model_config,
)


FORMAT_VERSION = "0.1"
COMPONENT_ROLES = ("att_x_prev", "att_kv", "ffn_x_prev")


class CheckpointError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes(path, canonical_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("E_SCHEMA", f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("E_SCHEMA", f"{path.name} must contain a JSON object")
    return payload


def component_name(index: int, layer_count: int) -> str:
    expected = layer_count * len(COMPONENT_ROLES)
    if index < 0 or index >= expected:
        raise ValueError(f"component index {index} is outside 0..{expected - 1}")
    layer_index, role_index = divmod(index, len(COMPONENT_ROLES))
    return f"layers.{layer_index}.{COMPONENT_ROLES[role_index]}"


def _state_tensor_map(state: Any, layer_count: int) -> dict[str, Any]:
    flattened = list(_flatten_state(state))
    expected = layer_count * len(COMPONENT_ROLES)
    if len(flattened) != expected:
        raise CheckpointError(
            "E_SHAPE_MISMATCH",
            f"expected {expected} RWKV-7 components, got {len(flattened)}",
        )
    tensors: dict[str, Any] = {}
    for index, (path, tensor) in enumerate(flattened):
        if path != f"state[{index}]":
            raise CheckpointError(
                "E_SCHEMA",
                f"expected flat list state[{index}], got {path}",
            )
        tensors[component_name(index, layer_count)] = (
            tensor.detach().contiguous().cpu()
        )
    return tensors


def _checkpoint_inventory(
    state: Any,
    config: RWKV7ModelConfig,
    torch: Any,
) -> dict[str, Any]:
    source = inventory_state(state, torch)
    components = []
    for index, item in enumerate(source["components"]):
        components.append(
            {
                "name": component_name(index, config.architecture_hint["n_layer"]),
                "source_path": item["path"],
                "layer": index // 3,
                "role": COMPONENT_ROLES[index % 3],
                "shape": item["shape"],
                "dtype": item["dtype"],
                "device_at_capture": item["device"],
                "numel": item["numel"],
                "byte_length": item["size_bytes"],
                "finite": item["finite"],
                "l2_norm": item["l2_norm"],
                "sha256": item["sha256"],
            }
        )
    return {
        "inventory_version": FORMAT_VERSION,
        "created_at_utc": _utc_now(),
        "model_id": config.model_id,
        "component_count": source["component_count"],
        "total_size_bytes": source["total_size_bytes"],
        "all_finite": source["all_finite"],
        "components": components,
        "state_digest_sha256": sha256_json(
            [
                {
                    "name": item["name"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                    "sha256": item["sha256"],
                }
                for item in components
            ]
        ),
    }


def _checksum_text(file_digests: dict[str, str]) -> bytes:
    lines = [
        f"{digest}  {path}"
        for path, digest in sorted(file_digests.items(), key=lambda item: item[0])
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckpointError("E_CHECKSUM", f"cannot read checksums: {exc}") from exc
    parsed: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        relative_path = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in parsed
        ):
            raise CheckpointError(
                "E_CHECKSUM",
                f"invalid checksum entry on line {line_number}",
            )
        parsed[relative] = digest
    if not parsed:
        raise CheckpointError("E_CHECKSUM", "checksum list is empty")
    return parsed


def _checkpoint_file(
    checkpoint: Path,
    relative: str,
    *,
    error_code: str,
) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CheckpointError(error_code, f"unsafe checkpoint path: {relative}")
    resolved = (checkpoint / relative_path).resolve()
    if checkpoint not in resolved.parents or not resolved.is_file():
        raise CheckpointError(error_code, f"missing checkpoint file: {relative}")
    return resolved


def _safetensors_torch() -> Any:
    try:
        return import_module("safetensors.torch")
    except ImportError as exc:
        raise CheckpointError(
            "E_FORMAT_VERSION",
            "safetensors is required; run scripts/install_impl1_gpu.sh",
        ) from exc


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _apply_determinism_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("determinism policy must be an object")
    expected = {
        "enabled": True,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "highest",
        "allow_tf32": False,
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("determinism policy does not match Impl-2")
    seed = policy.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("determinism seed must be a non-negative integer")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = policy[
        "cublas_workspace_config"
    ]
    os.environ["PSA_DETERMINISTIC"] = "1"
    os.environ["PSA_DETERMINISTIC_SEED"] = str(seed)
    return dict(policy)


def _validate_acceptance_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("acceptance policy must be an object")
    if (
        policy.get("require_shape_dtype_compatibility") is not True
        or policy.get("require_top1_match") is not True
    ):
        raise ValueError("Impl-2 requires compatibility and top-1 agreement")
    for field in ("logits_max_abs_error", "state_max_abs_error"):
        value = policy.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"acceptance.{field} must be positive")
    return dict(policy)


def _top1_id(logits: Any) -> int:
    return int(logits.detach().float().argmax().item())


def save_native_checkpoint(
    state: Any,
    config: RWKV7ModelConfig,
    checkpoint_parent: str | Path,
    *,
    prefix_tokens: list[int],
    next_tokens: list[int],
    run_id: str,
    trajectory_id: str,
    gate_config_digest: str,
    code_revision: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    parent = Path(checkpoint_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = f"ckpt-{uuid4().hex}"
    final_path = parent / checkpoint_id
    if final_path.exists():
        raise FileExistsError(f"checkpoint already exists: {final_path}")
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{checkpoint_id}.tmp-", dir=parent)
    )
    started = time.perf_counter()
    try:
        native_dir = temporary_path / "native_state"
        provenance_dir = temporary_path / "provenance"
        validation_dir = temporary_path / "validation"
        native_dir.mkdir(parents=True)
        provenance_dir.mkdir()
        validation_dir.mkdir()

        tensor_map = _state_tensor_map(
            state, config.architecture_hint["n_layer"]
        )
        inventory = _checkpoint_inventory(state, config, torch)
        if not inventory["all_finite"]:
            raise CheckpointError("E_NUMERICAL", "state contains NaN or Infinity")

        tensor_path = native_dir / "tensors.safetensors"
        safetensors_torch = _safetensors_torch()
        safetensors_torch.save_file(
            tensor_map,
            str(tensor_path),
            metadata={
                "format": "psa-native-state",
                "format_version": FORMAT_VERSION,
                "model_id": config.model_id,
            },
        )
        with tensor_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _write_json(native_dir / "inventory.json", inventory)

        capture_event = {
            "event_id": f"event-{uuid4().hex}",
            "event_type": "capture",
            "step": 0,
            "actor": "development_gate",
            "source_checkpoint_id": None,
            "target_checkpoint_id": checkpoint_id,
            "parameters": {"mode": "native_state_disk_roundtrip"},
            "result": "success",
            "timestamp": _utc_now(),
        }
        _write_bytes(
            provenance_dir / "events.jsonl",
            canonical_json_bytes(capture_event),
        )

        payload_paths = (
            "native_state/tensors.safetensors",
            "native_state/inventory.json",
            "provenance/events.jsonl",
        )
        file_digests = {
            relative: sha256_file(temporary_path / relative)
            for relative in payload_paths
        }
        _write_bytes(
            validation_dir / "checksums.sha256",
            _checksum_text(file_digests),
        )

        try:
            safetensors_version = version("safetensors")
        except PackageNotFoundError as exc:
            raise CheckpointError(
                "E_FORMAT_VERSION", "cannot determine safetensors version"
            ) from exc
        manifest = {
            "format_version": FORMAT_VERSION,
            "checkpoint_id": checkpoint_id,
            "checkpoint_kind": "native_state",
            "created_at": _utc_now(),
            "experiment_id": "DEV-IMPL2",
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "parent_checkpoint_id": None,
            "fork_root_id": None,
            "environment_step": 0,
            "model": {
                "family": "RWKV",
                "architecture": "RWKV-7",
                "model_id": config.model_id,
                "revision": config.weights_revision,
                "weight_digest_sha256": config.weights_sha256,
                "layer_count": config.architecture_hint["n_layer"],
                "hidden_size": config.architecture_hint["n_embd"],
                "implementation": "rwkv",
            },
            "tokenizer": {
                "tokenizer_id": "rwkv-world-tokenizer-20230424",
                "revision": config.tokenizer_revision,
                "vocab_digest_sha256": config.tokenizer_sha256,
                "implementation": "rwkv.rwkv_tokenizer.TRIE_TOKENIZER",
            },
            "runtime": {
                "framework": "torch",
                "framework_version": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "dtype": "mixed_fp16_fp32",
                "kernel_family": (
                    "pytorch"
                    if config.environment["RWKV_CUDA_ON"] == "0"
                    else "rwkv_cuda"
                ),
                "deterministic_mode": (
                    os.environ.get("PSA_DETERMINISTIC", "0") == "1"
                ),
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
                "strategy": config.strategy,
                "safetensors_version": safetensors_version,
                "code_commit": code_revision["commit"],
                "code_dirty": code_revision["dirty"],
                "config_digest_sha256": gate_config_digest,
            },
            "state_components": inventory["components"],
            "input_boundary": {
                "last_token_id": prefix_tokens[-1],
                "tokens_consumed": len(prefix_tokens),
                "prefix_digest_sha256": sha256_json(prefix_tokens),
                "conversation_boundary": "development_probe_prefix_end",
                "eot_seen": False,
                "next_expected_input_digest_sha256": sha256_json(next_tokens),
            },
            "provenance": {
                "capture_mode": "development_only",
                "events": "provenance/events.jsonl",
            },
            "integrity": {
                "algorithm": "sha256",
                "payload_root_digest_sha256": payload_digest(file_digests),
                "checksums_file": "validation/checksums.sha256",
            },
            "status": "complete",
        }
        _write_json(temporary_path / "manifest.json", manifest)

        verification = verify_native_checkpoint(
            temporary_path,
            config=config,
            load_tensors=True,
            torch=torch,
        )
        if not verification["valid"]:
            raise CheckpointError("E_CHECKSUM", "temporary checkpoint is invalid")
        os.replace(temporary_path, final_path)
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(final_path),
        "save_seconds": time.perf_counter() - started,
        "tensor_file_size_bytes": (
            final_path / "native_state" / "tensors.safetensors"
        ).stat().st_size,
        "payload_size_bytes": inventory["total_size_bytes"],
        "payload_root_digest_sha256": manifest["integrity"][
            "payload_root_digest_sha256"
        ],
        "manifest_sha256": sha256_file(final_path / "manifest.json"),
    }


def verify_native_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: RWKV7ModelConfig | None = None,
    load_tensors: bool = True,
    torch: Any | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise CheckpointError("E_INCOMPLETE", "manifest.json is missing")
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointError("E_FORMAT_VERSION", "unsupported checkpoint version")
    if manifest.get("status") != "complete":
        raise CheckpointError("E_INCOMPLETE", "checkpoint is not complete")

    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        raise CheckpointError("E_SCHEMA", "integrity must be an object")
    checksum_relative = integrity.get("checksums_file")
    if not isinstance(checksum_relative, str):
        raise CheckpointError("E_SCHEMA", "checksums_file is missing")
    checksum_path = _checkpoint_file(
        checkpoint,
        checksum_relative,
        error_code="E_CHECKSUM",
    )
    file_digests = _parse_checksums(checksum_path)
    for relative, expected in file_digests.items():
        payload_path = _checkpoint_file(
            checkpoint,
            relative,
            error_code="E_CHECKSUM",
        )
        actual = sha256_file(payload_path)
        if actual != expected:
            raise CheckpointError("E_CHECKSUM", f"digest mismatch: {relative}")
    expected_root = integrity.get("payload_root_digest_sha256")
    if payload_digest(file_digests) != expected_root:
        raise CheckpointError("E_CHECKSUM", "payload root digest mismatch")

    model = manifest.get("model")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(model, dict) or not isinstance(tokenizer, dict):
        raise CheckpointError("E_SCHEMA", "model and tokenizer metadata are required")
    layer_count = model.get("layer_count")
    components = manifest.get("state_components")
    if (
        not isinstance(layer_count, int)
        or layer_count <= 0
        or not isinstance(components, list)
    ):
        raise CheckpointError(
            "E_SCHEMA", "model.layer_count and state_components are required"
        )
    expected_names = [
        component_name(index, layer_count) for index in range(layer_count * 3)
    ]
    observed_names = [
        component.get("name") if isinstance(component, dict) else None
        for component in components
    ]
    if observed_names != expected_names:
        raise CheckpointError(
            "E_SHAPE_MISMATCH",
            "state component names or ordering differ from the RWKV-7 contract",
        )
    if config is not None:
        if (
            model.get("model_id") != config.model_id
            or model.get("weight_digest_sha256") != config.weights_sha256
            or model.get("revision") != config.weights_revision
        ):
            raise CheckpointError("E_MODEL_MISMATCH", "model identity differs")
        if (
            tokenizer.get("revision") != config.tokenizer_revision
            or tokenizer.get("vocab_digest_sha256") != config.tokenizer_sha256
        ):
            raise CheckpointError(
                "E_TOKENIZER_MISMATCH", "tokenizer identity differs"
            )
        boundary = manifest.get("input_boundary")
        if not isinstance(boundary, dict):
            raise CheckpointError("E_SCHEMA", "input_boundary is required")

    result = {
        "requested_level": "L2" if config is not None else "L1",
        "achieved_level": "L1",
        "checkpoint_id": manifest.get("checkpoint_id"),
        "payload_root_digest_sha256": expected_root,
        "checksums_valid": True,
        "model_compatible": config is not None,
        "tensor_inventory_valid": None,
        "valid": True,
    }
    if not load_tensors:
        return result

    if torch is None:
        torch = import_module("torch")
    state, _ = load_native_state(
        checkpoint,
        torch=torch,
        device="cpu",
        verify=False,
    )
    observed = inventory_state(state, torch)
    inventory = _read_json(checkpoint / "native_state" / "inventory.json")
    expected_components = inventory.get("components")
    if not isinstance(expected_components, list):
        raise CheckpointError("E_SCHEMA", "inventory components must be an array")
    observed_components = observed["components"]
    if len(observed_components) != len(expected_components):
        raise CheckpointError("E_SHAPE_MISMATCH", "component count differs")
    for expected_component, observed_component in zip(
        expected_components, observed_components
    ):
        if (
            expected_component.get("shape") != observed_component["shape"]
            or expected_component.get("dtype") != observed_component["dtype"]
            or expected_component.get("sha256") != observed_component["sha256"]
        ):
            raise CheckpointError(
                "E_SHAPE_MISMATCH",
                f"inventory mismatch: {expected_component.get('name')}",
            )
    result["tensor_inventory_valid"] = True
    if config is not None:
        result["achieved_level"] = "L2"
    return result


def load_native_state(
    checkpoint_path: str | Path,
    *,
    torch: Any,
    device: str,
    verify: bool = True,
    config: RWKV7ModelConfig | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    checkpoint = Path(checkpoint_path).resolve()
    if verify:
        verify_native_checkpoint(
            checkpoint,
            config=config,
            load_tensors=False,
        )
    manifest = _read_json(checkpoint / "manifest.json")
    components = manifest.get("state_components")
    if not isinstance(components, list) or not components:
        raise CheckpointError("E_SCHEMA", "state_components are missing")
    loaded = _safetensors_torch().load_file(
        str(checkpoint / "native_state" / "tensors.safetensors"),
        device="cpu",
    )
    expected_names = [item.get("name") for item in components]
    if (
        any(not isinstance(name, str) for name in expected_names)
        or len(set(expected_names)) != len(expected_names)
        or set(loaded) != set(expected_names)
    ):
        raise CheckpointError("E_SHAPE_MISMATCH", "tensor names differ")
    state = []
    for component, name in zip(components, expected_names):
        tensor = loaded[name]
        if (
            list(tensor.shape) != component.get("shape")
            or str(tensor.dtype) != component.get("dtype")
        ):
            raise CheckpointError("E_SHAPE_MISMATCH", f"tensor differs: {name}")
        state.append(tensor.to(device=device))
    return state, manifest


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def run_restore_probe(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    reference_path: str | Path,
    probe_config_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    config = load_model_config(config_path, project_root, verify_files=True)
    probe_config = _read_json(Path(probe_config_path))
    repeat_count = probe_config.get("repeat_count")
    if not isinstance(repeat_count, int) or repeat_count < 100:
        raise ValueError("repeat_count must be at least 100")
    determinism_policy = _apply_determinism_policy(
        probe_config.get("determinism")
    )
    acceptance = _validate_acceptance_policy(
        probe_config.get("acceptance")
    )

    checkpoint_verification = verify_native_checkpoint(
        checkpoint_path,
        config=config,
        load_tensors=False,
    )
    manifest_digest_before = sha256_file(
        Path(checkpoint_path) / "manifest.json"
    )
    reference_file = Path(reference_path)
    expected_digest = probe_config.get("reference_sha256")
    if sha256_file(reference_file) != expected_digest:
        raise CheckpointError("E_CHECKSUM", "reference digest mismatch")
    suffix_tokens = probe_config.get("suffix_tokens")
    if (
        not isinstance(suffix_tokens, list)
        or not suffix_tokens
        or not all(isinstance(token, int) for token in suffix_tokens)
    ):
        raise ValueError("probe suffix_tokens are invalid")
    suffix_digest = sha256_json(suffix_tokens)
    checkpoint_manifest = _read_json(Path(checkpoint_path) / "manifest.json")
    boundary = checkpoint_manifest["input_boundary"]
    if (
        suffix_digest != probe_config.get("suffix_digest_sha256")
        or suffix_digest != boundary.get("next_expected_input_digest_sha256")
    ):
        raise CheckpointError(
            "E_BOUNDARY_MISMATCH",
            "restore suffix does not match the checkpoint input boundary",
        )

    adapter = RWKV7Adapter.load(config)
    torch = adapter.torch
    verification = verify_native_checkpoint(
        checkpoint_path,
        config=config,
        load_tensors=True,
        torch=torch,
    )
    if checkpoint_verification["payload_root_digest_sha256"] != verification[
        "payload_root_digest_sha256"
    ]:
        raise CheckpointError("E_CHECKSUM", "checkpoint changed during model load")
    reference = _safetensors_torch().load_file(
        str(reference_file), device="cpu"
    )
    components = checkpoint_manifest["state_components"]
    names = [component["name"] for component in components]
    expected_logits = reference["probe.logits"].to(device="cuda")
    expected_state = [reference[name].to(device="cuda") for name in names]
    expected_top1 = _top1_id(expected_logits)
    torch.cuda.reset_peak_memory_stats()
    trials = []
    load_times: list[float] = []
    inference_times: list[float] = []
    all_compatible = True
    all_within_tolerance = True
    all_top1_match = True
    worst_logits_error = 0.0
    worst_state_error = 0.0
    child_baseline_logits = None
    child_baseline_state = None
    intra_process_exact_count = 0
    intra_process_tolerance_count = 0
    for trial_index in range(repeat_count):
        load_started = time.perf_counter()
        restored_state, _ = load_native_state(
            checkpoint_path,
            torch=torch,
            device="cuda",
            verify=False,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started

        inference_started = time.perf_counter()
        logits, final_state = adapter.forward(suffix_tokens, restored_state)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        logits_comparison = compare_tensors(logits, expected_logits, torch)
        state_comparison = compare_states(final_state, expected_state, torch)
        exact = bool(logits_comparison["exact"] and state_comparison["exact"])
        compatible = bool(
            logits_comparison["compatible"] and state_comparison["compatible"]
        )
        all_compatible = all_compatible and compatible
        logits_error = logits_comparison["max_abs_error"]
        state_error = state_comparison["max_abs_error"]
        within_tolerance = bool(
            compatible
            and logits_error is not None
            and state_error is not None
            and logits_error <= acceptance["logits_max_abs_error"]
            and state_error <= acceptance["state_max_abs_error"]
        )
        top1_match = _top1_id(logits) == expected_top1
        all_within_tolerance = all_within_tolerance and within_tolerance
        all_top1_match = all_top1_match and top1_match

        if child_baseline_logits is None:
            intra_logits = {
                "compatible": True,
                "exact": True,
                "max_abs_error": 0.0,
            }
            intra_state = {
                "compatible": True,
                "exact": True,
                "max_abs_error": 0.0,
            }
            child_baseline_logits = logits.detach().clone()
            child_baseline_state = clone_state(final_state)
        else:
            intra_logits = compare_tensors(
                logits, child_baseline_logits, torch
            )
            intra_state = compare_states(
                final_state, child_baseline_state, torch
            )
        intra_exact = bool(intra_logits["exact"] and intra_state["exact"])
        intra_within_tolerance = bool(
            intra_logits["compatible"]
            and intra_state["compatible"]
            and intra_logits["max_abs_error"]
            <= acceptance["logits_max_abs_error"]
            and intra_state["max_abs_error"]
            <= acceptance["state_max_abs_error"]
        )
        intra_process_exact_count += int(intra_exact)
        intra_process_tolerance_count += int(intra_within_tolerance)
        worst_logits_error = max(
            worst_logits_error,
            float(logits_comparison["max_abs_error"] or 0.0),
        )
        worst_state_error = max(
            worst_state_error,
            float(state_comparison["max_abs_error"] or 0.0),
        )
        load_times.append(load_seconds)
        inference_times.append(inference_seconds)
        trials.append(
            {
                "trial": trial_index + 1,
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "logits_compatible": logits_comparison["compatible"],
                "logits_exact": logits_comparison["exact"],
                "logits_max_abs_error": logits_comparison["max_abs_error"],
                "top1_match": top1_match,
                "state_compatible": state_comparison["compatible"],
                "state_exact": state_comparison["exact"],
                "state_max_abs_error": state_comparison["max_abs_error"],
                "within_tolerance": within_tolerance,
                "intra_process_exact": intra_exact,
                "intra_process_within_tolerance": intra_within_tolerance,
            }
        )
        del restored_state, logits, final_state

    final_verification = verify_native_checkpoint(
        checkpoint_path,
        config=config,
        load_tensors=False,
    )
    checkpoint_stable = bool(
        manifest_digest_before
        == sha256_file(Path(checkpoint_path) / "manifest.json")
        and checkpoint_verification["payload_root_digest_sha256"]
        == final_verification["payload_root_digest_sha256"]
    )
    report = {
        "report_version": FORMAT_VERSION,
        "created_at_utc": _utc_now(),
        "development_only": True,
        "mode": "disk_cross_process_restore",
        "determinism": {
            "policy": determinism_policy,
            "observed": adapter.determinism,
        },
        "acceptance": acceptance,
        "process_is_distinct": os.getpid() != probe_config.get("parent_process_id"),
        "repeat_count": repeat_count,
        "checkpoint_verification": verification,
        "checkpoint_stable_during_probe": checkpoint_stable,
        "exact_repeat_count": sum(
            trial["logits_exact"] and trial["state_exact"] for trial in trials
        ),
        "tolerance_pass_count": sum(
            trial["within_tolerance"] for trial in trials
        ),
        "top1_match_count": sum(trial["top1_match"] for trial in trials),
        "intra_process_exact_count": intra_process_exact_count,
        "intra_process_tolerance_count": intra_process_tolerance_count,
        "worst_error": {
            "logits_max_abs_error": (
                worst_logits_error if all_compatible else None
            ),
            "state_max_abs_error": worst_state_error if all_compatible else None,
        },
        "timing_seconds": {
            "load_first": load_times[0],
            "load_median": _quantile(load_times, 0.5),
            "load_p95": _quantile(load_times, 0.95),
            "inference_median": _quantile(inference_times, 0.5),
            "inference_p95": _quantile(inference_times, 0.95),
        },
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "trials": trials,
        "achieved_level": (
            "L3"
            if (
                all_compatible
                and all_within_tolerance
                and all_top1_match
                and checkpoint_stable
            )
            else "L2"
        ),
        "valid": bool(
            verification["valid"]
            and os.getpid() != probe_config.get("parent_process_id")
            and all_compatible
            and all_within_tolerance
            and all_top1_match
            and checkpoint_stable
        ),
    }
    _write_json(Path(output_path), report)
    return report


def run_checkpoint_roundtrip_gate(
    *,
    config_path: str | Path,
    gate_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    gate_config_path = Path(gate_config_path).resolve()
    gate_config = _read_json(gate_config_path)
    gate_name = gate_config.get("gate")
    if (
        gate_config.get("gate_version") != "0.1"
        or gate_name
        not in {
            "impl2_checkpoint_roundtrip",
            "impl3m_g1h_2_9b_checkpoint_roundtrip",
        }
        or gate_config.get("development_only") is not True
    ):
        raise ValueError("unsupported checkpoint roundtrip gate config")
    repeat_count = gate_config.get("repeat_count")
    if not isinstance(repeat_count, int) or repeat_count < 100:
        raise ValueError("gate repeat_count must be at least 100")
    if (
        gate_config.get("required_validation_level") != "L3"
        or gate_config.get("bitwise_exact_is_diagnostic") is not True
    ):
        raise ValueError("gate must retain bitwise exactness as a diagnostic")
    determinism_policy = _apply_determinism_policy(
        gate_config.get("determinism")
    )
    acceptance = _validate_acceptance_policy(
        gate_config.get("acceptance")
    )
    checkpoint_options = gate_config.get("checkpoint")
    if (
        not isinstance(checkpoint_options, dict)
        or checkpoint_options.get("format_version") != FORMAT_VERSION
        or checkpoint_options.get("tensor_container") != "safetensors"
        or checkpoint_options.get("tensor_container_version") != "0.8.0"
        or checkpoint_options.get("preserve_capture_dtype") is not True
        or checkpoint_options.get("atomic_commit") is not True
        or checkpoint_options.get("checksum") != "sha256"
    ):
        raise ValueError("gate checkpoint policy does not match Impl-2")
    prefix_text = gate_config.get("prefix_text")
    suffix_text = gate_config.get("suffix_text")
    if not isinstance(prefix_text, str) or not prefix_text:
        raise ValueError("gate prefix_text must be non-empty")
    if not isinstance(suffix_text, str) or not suffix_text:
        raise ValueError("gate suffix_text must be non-empty")

    config = load_model_config(config_path, root, verify_files=True)
    adapter = RWKV7Adapter.load(config)
    torch = adapter.torch
    prefix_tokens = adapter.encode(prefix_text)
    suffix_tokens = adapter.encode(suffix_text)
    _, prefix_state = adapter.forward(prefix_tokens, None)
    run_id = f"run-{uuid4().hex}"
    trajectory_id = f"trajectory-{uuid4().hex}"
    capture = save_native_checkpoint(
        prefix_state,
        config,
        destination / "checkpoints",
        prefix_tokens=prefix_tokens,
        next_tokens=suffix_tokens,
        run_id=run_id,
        trajectory_id=trajectory_id,
        gate_config_digest=sha256_file(gate_config_path),
        code_revision=_git_metadata(root),
        torch=torch,
    )

    reference_logits, reference_state = adapter.forward(
        suffix_tokens, prefix_state
    )
    run_dir = destination / "runs" / run_id
    run_dir.mkdir(parents=True)
    reference_path = run_dir / "reference.safetensors"
    reference_tensors = {
        "probe.logits": reference_logits.detach().contiguous().cpu(),
        **_state_tensor_map(
            reference_state, config.architecture_hint["n_layer"]
        ),
    }
    _safetensors_torch().save_file(
        reference_tensors,
        str(reference_path),
        metadata={"kind": "development-restore-reference"},
    )
    reference_digest = sha256_file(reference_path)
    probe_config = {
        "probe_version": FORMAT_VERSION,
        "development_only": True,
        "parent_process_id": os.getpid(),
        "repeat_count": repeat_count,
        "suffix_tokens": suffix_tokens,
        "suffix_digest_sha256": sha256_json(suffix_tokens),
        "reference_sha256": reference_digest,
        "determinism": determinism_policy,
        "acceptance": acceptance,
    }
    probe_config_path = run_dir / "probe_config.json"
    _write_json(probe_config_path, probe_config)
    probe_report_path = run_dir / "cross_process_restore_report.json"

    del adapter, prefix_state, reference_logits, reference_state, reference_tensors
    gc.collect()
    torch.cuda.empty_cache()
    command = [
        sys.executable,
        "-m",
        "psa",
        "checkpoint-restore-probe",
        "--config",
        str(Path(config_path).resolve()),
        "--checkpoint",
        capture["checkpoint_path"],
        "--reference",
        str(reference_path),
        "--probe-config",
        str(probe_config_path),
        "--output",
        str(probe_report_path),
        "--project-root",
        str(root),
    ]
    child_started = time.perf_counter()
    child = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    child_seconds = time.perf_counter() - child_started
    if not probe_report_path.is_file():
        failure = {
            "failure_version": FORMAT_VERSION,
            "created_at_utc": _utc_now(),
            "development_only": True,
            "return_code": child.returncode,
            "stderr_tail": child.stderr[-4000:],
            "stdout_tail": child.stdout[-4000:],
        }
        _write_json(run_dir / "child_failure_report.json", failure)
        raise RuntimeError(
            "cross-process restore probe failed; see child_failure_report.json"
        )
    probe_report = _read_json(probe_report_path)

    summary = {
        "gate": gate_name,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "development_only": True,
        "run_id": run_id,
        "checkpoint_id": capture["checkpoint_id"],
        "checkpoint_path": capture["checkpoint_path"],
        "checkpoint_payload_size_bytes": capture["payload_size_bytes"],
        "checkpoint_tensor_file_size_bytes": capture["tensor_file_size_bytes"],
        "checkpoint_save_seconds": capture["save_seconds"],
        "checkpoint_manifest_sha256": capture["manifest_sha256"],
        "cross_process_seconds": child_seconds,
        "repeat_count": repeat_count,
        "exact_repeat_count": probe_report["exact_repeat_count"],
        "tolerance_pass_count": probe_report["tolerance_pass_count"],
        "top1_match_count": probe_report["top1_match_count"],
        "intra_process_exact_count": probe_report[
            "intra_process_exact_count"
        ],
        "achieved_level": probe_report["achieved_level"],
        "valid": probe_report["valid"],
        "reports": [
            str(probe_report_path),
            str(Path(capture["checkpoint_path"]) / "manifest.json"),
            str(
                Path(capture["checkpoint_path"])
                / "native_state"
                / "inventory.json"
            ),
        ],
    }
    _write_json(destination / "summary.json", summary)
    return summary
