from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json


@dataclass(frozen=True)
class RWKV7ModelConfig:
    model_id: str
    weights_path: Path
    weights_revision: str
    weights_sha256: str
    weights_size_bytes: int
    tokenizer_path: Path
    tokenizer_revision: str
    tokenizer_sha256: str
    tokenizer_size_bytes: int
    strategy: str
    environment: dict[str, str]
    architecture_hint: dict[str, int]
    config_path: Path


def _require_mapping(payload: Any, field: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be an object")
    return payload


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _project_file(
    project_root: Path,
    payload: dict[str, Any],
    field: str,
) -> Path:
    relative = Path(_require_string(payload, "path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field}.path must stay inside the project root")
    path = (project_root / relative).resolve()
    if project_root not in path.parents:
        raise ValueError(f"{field}.path escapes the project root")
    return path


def _verified_file(
    project_root: Path,
    payload: dict[str, Any],
    field: str,
) -> tuple[Path, str, int]:
    path = _project_file(project_root, payload, field)
    if not path.is_file():
        raise FileNotFoundError(f"{field} file is missing: {path}")

    expected_size = payload.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f"{field}.size_bytes must be a positive integer")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{field} size mismatch: expected {expected_size}, got {actual_size}"
        )

    expected_digest = _require_string(payload, "sha256")
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"{field} SHA-256 mismatch: expected {expected_digest}, "
            f"got {actual_digest}"
        )
    return path, actual_digest, actual_size


def load_model_config(
    path: str | Path,
    project_root: str | Path = ".",
    verify_files: bool = True,
) -> RWKV7ModelConfig:
    root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("config_version") != "0.1":
        raise ValueError("unsupported model config")
    if payload.get("architecture") != "RWKV-7":
        raise ValueError("model config is not for RWKV-7")

    weights = _require_mapping(payload.get("weights"), "weights")
    tokenizer = _require_mapping(payload.get("tokenizer"), "tokenizer")
    runtime = _require_mapping(payload.get("runtime"), "runtime")
    environment = _require_mapping(runtime.get("environment"), "runtime.environment")
    architecture = _require_mapping(
        payload.get("architecture_hint"), "architecture_hint"
    )

    if verify_files:
        weights_path, weights_digest, weights_size = _verified_file(
            root, weights, "weights"
        )
        tokenizer_path, tokenizer_digest, tokenizer_size = _verified_file(
            root, tokenizer, "tokenizer"
        )
    else:
        weights_path = _project_file(root, weights, "weights")
        tokenizer_path = _project_file(root, tokenizer, "tokenizer")
        weights_digest = _require_string(weights, "sha256")
        tokenizer_digest = _require_string(tokenizer, "sha256")
        weights_size = int(weights["size_bytes"])
        tokenizer_size = int(tokenizer["size_bytes"])

    env = {}
    for key in ("RWKV_V7_ON", "RWKV_JIT_ON", "RWKV_CUDA_ON"):
        value = environment.get(key)
        if value not in {"0", "1"}:
            raise ValueError(f"runtime.environment.{key} must be 0 or 1")
        env[key] = value

    architecture_hint = {}
    for key in ("n_layer", "n_embd", "head_size"):
        value = architecture.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"architecture_hint.{key} must be positive")
        architecture_hint[key] = value

    return RWKV7ModelConfig(
        model_id=_require_string(payload, "model_id"),
        weights_path=weights_path,
        weights_revision=_require_string(weights, "revision"),
        weights_sha256=weights_digest,
        weights_size_bytes=weights_size,
        tokenizer_path=tokenizer_path,
        tokenizer_revision=_require_string(tokenizer, "revision"),
        tokenizer_sha256=tokenizer_digest,
        tokenizer_size_bytes=tokenizer_size,
        strategy=_require_string(runtime, "strategy"),
        environment=env,
        architecture_hint=architecture_hint,
        config_path=config_path,
    )


def clone_state(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported state component: {type(value).__name__}")


def _flatten_state(value: Any, path: str = "state") -> Iterable[tuple[str, Any]]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten_state(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_state(value[key], f"{path}.{key}")
    elif value is not None:
        raise TypeError(f"unsupported state component at {path}: {type(value).__name__}")


def _tensor_digest(tensor: Any, torch: Any) -> str:
    byte_view = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _component_role(path: str, component_count: int) -> str | None:
    if component_count % 3 != 0 or not path.startswith("state["):
        return None
    try:
        index = int(path.removeprefix("state[").split("]", maxsplit=1)[0])
    except ValueError:
        return None
    return ("att_x_prev", "att_kv", "ffn_x_prev")[index % 3]


def inventory_state(state: Any, torch: Any) -> dict[str, Any]:
    flattened = list(_flatten_state(state))
    records = []
    total_bytes = 0
    finite = True
    for path, tensor in flattened:
        detached = tensor.detach()
        size_bytes = int(detached.numel() * detached.element_size())
        total_bytes += size_bytes
        floating = detached.is_floating_point()
        is_finite = bool(torch.isfinite(detached).all().item()) if floating else True
        finite = finite and is_finite
        numeric = detached.float()
        records.append(
            {
                "path": path,
                "role_hint": _component_role(path, len(flattened)),
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
                "numel": int(detached.numel()),
                "size_bytes": size_bytes,
                "finite": is_finite,
                "l2_norm": float(torch.linalg.vector_norm(numeric).item()),
                "sha256": _tensor_digest(detached, torch),
            }
        )
    return {
        "inventory_version": "0.1",
        "component_count": len(records),
        "total_size_bytes": total_bytes,
        "all_finite": finite,
        "components": records,
        "state_digest_sha256": sha256_json(
            [
                {
                    "path": item["path"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                    "sha256": item["sha256"],
                }
                for item in records
            ]
        ),
    }


def compare_tensors(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return {
            "compatible": False,
            "exact": False,
            "max_abs_error": None,
            "mean_abs_error": None,
        }
    difference = (left.detach().float() - right.detach().float()).abs()
    return {
        "compatible": True,
        "exact": bool(torch.equal(left, right)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def compare_states(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    left_items = list(_flatten_state(left))
    right_items = list(_flatten_state(right))
    if [path for path, _ in left_items] != [path for path, _ in right_items]:
        return {
            "compatible": False,
            "exact": False,
            "component_count": 0,
            "max_abs_error": None,
            "components": [],
        }
    records = []
    for (path, left_tensor), (_, right_tensor) in zip(left_items, right_items):
        comparison = compare_tensors(left_tensor, right_tensor, torch)
        records.append({"path": path, **comparison})
    return {
        "compatible": all(item["compatible"] for item in records),
        "exact": all(item["exact"] for item in records),
        "component_count": len(records),
        "max_abs_error": max(
            (
                item["max_abs_error"]
                for item in records
                if item["max_abs_error"] is not None
            ),
            default=0.0,
        ),
        "components": records,
    }


class RWKV7Adapter:
    def __init__(
        self,
        config: RWKV7ModelConfig,
        model: Any,
        tokenizer: Any,
        torch: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch

    @classmethod
    def load(cls, config: RWKV7ModelConfig) -> "RWKV7Adapter":
        for key, value in config.environment.items():
            os.environ[key] = value
        if "rwkv.model" in sys.modules:
            raise RuntimeError(
                "rwkv.model was imported before runtime flags were fixed; "
                "start a fresh process"
            )

        torch = import_module("torch")
        rwkv_model = import_module("rwkv.model")
        rwkv_tokenizer = import_module("rwkv.rwkv_tokenizer")

        model_path = str(config.weights_path)
        if model_path.endswith(".pth"):
            model_path = model_path[:-4]
        model = rwkv_model.RWKV(model=model_path, strategy=config.strategy)
        tokenizer = rwkv_tokenizer.TRIE_TOKENIZER(str(config.tokenizer_path))
        return cls(config, model, tokenizer, torch)

    def encode(self, text: str) -> list[int]:
        tokens = list(self.tokenizer.encode(text))
        if not tokens or not all(isinstance(token, int) for token in tokens):
            raise RuntimeError("tokenizer returned an invalid or empty token sequence")
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        return str(self.tokenizer.decode(list(tokens)))

    def forward(
        self,
        tokens: Iterable[int],
        state: Any = None,
    ) -> tuple[Any, Any]:
        token_list = list(tokens)
        if not token_list:
            raise ValueError("forward requires at least one token")
        with self.torch.inference_mode():
            logits, next_state = self.model.forward(token_list, state)
        return logits, next_state

    def model_metadata(self) -> dict[str, Any]:
        args = getattr(self.model, "args", None)
        observed = {}
        for field in (
            "n_layer",
            "n_embd",
            "vocab_size",
            "head_size_a",
            "version",
        ):
            value = getattr(args, field, None) if args is not None else None
            if isinstance(value, (str, int, float, bool)) or value is None:
                observed[field] = value
        return {
            "model_id": self.config.model_id,
            "strategy": self.config.strategy,
            "architecture_hint": self.config.architecture_hint,
            "observed_args": observed,
            "weights": {
                "path": str(self.config.weights_path),
                "revision": self.config.weights_revision,
                "sha256": self.config.weights_sha256,
                "size_bytes": self.config.weights_size_bytes,
            },
            "tokenizer": {
                "path": str(self.config.tokenizer_path),
                "revision": self.config.tokenizer_revision,
                "sha256": self.config.tokenizer_sha256,
                "size_bytes": self.config.tokenizer_size_bytes,
            },
            "runtime_environment": dict(self.config.environment),
        }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def run_interface_gate(
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    started_at = datetime.now(timezone.utc)
    config = load_model_config(config_path, root, verify_files=True)

    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    prefix = "Development-only recurrent state interface check.\n"
    suffix = "Continue deterministically with alpha beta gamma.\n"
    prefix_tokens = adapter.encode(prefix)
    suffix_tokens = adapter.encode(suffix)
    tokenizer_roundtrip = {
        "prefix_exact": adapter.decode(prefix_tokens) == prefix,
        "suffix_exact": adapter.decode(suffix_tokens) == suffix,
        "prefix_token_count": len(prefix_tokens),
        "suffix_token_count": len(suffix_tokens),
    }

    prefix_logits, prefix_state = adapter.forward(prefix_tokens, None)
    snapshot = clone_state(prefix_state)
    first_logits, first_state = adapter.forward(suffix_tokens, clone_state(snapshot))
    second_logits, second_state = adapter.forward(suffix_tokens, clone_state(snapshot))
    torch.cuda.synchronize()

    logits_comparison = compare_tensors(first_logits, second_logits, torch)
    state_comparison = compare_states(first_state, second_state, torch)
    state_inventory = inventory_state(snapshot, torch)

    interface_report = {
        "report_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "model": adapter.model_metadata(),
        "load_seconds": load_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "tokenizer_roundtrip": tokenizer_roundtrip,
        "logits": {
            "shape": list(prefix_logits.shape),
            "dtype": str(prefix_logits.dtype),
            "device": str(prefix_logits.device),
        },
    }
    state_inventory.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "model_id": config.model_id,
        }
    )
    roundtrip = {
        "report_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "mode": "in_memory_clone_restore",
        "prefix_token_count": len(prefix_tokens),
        "suffix_token_count": len(suffix_tokens),
        "logits": logits_comparison,
        "state": state_comparison,
        "valid": bool(
            tokenizer_roundtrip["prefix_exact"]
            and tokenizer_roundtrip["suffix_exact"]
            and state_inventory["all_finite"]
            and logits_comparison["exact"]
            and state_comparison["exact"]
        ),
    }

    _write_report(destination / "model_interface_report.json", interface_report)
    _write_report(destination / "state_inventory.json", state_inventory)
    _write_report(destination / "roundtrip_validation.json", roundtrip)
    summary = {
        "gate": "impl1_model_interface",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(destination),
        "valid": roundtrip["valid"],
        "reports": [
            "model_interface_report.json",
            "state_inventory.json",
            "roundtrip_validation.json",
        ],
    }
    _write_report(destination / "summary.json", summary)
    return summary
