from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.model.rwkv7 import RWKV7Adapter, clone_state, load_model_config
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
    RWKV7CouplingOffAdapter,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    RWKV7InstrumentedOffRuntime,
)


D4_EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D4_OFF_EQUIVALENCE"
D4_EXECUTION_LOCK_VALUE = "AUTHORIZED_D4_REAL_2_9B_OFF_EQUIVALENCE_ONCE"
REPORT_VERSION = "0.1-d4-real-off-equivalence"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise PermissionError(f"{field} must equal the frozen D4 value")


def _read_spec(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("D4 config must be an object")
    _require_exact(payload.get("protocol_version"), REPORT_VERSION, "protocol_version")
    _require_exact(payload.get("development_only"), True, "development_only")
    _require_exact(payload.get("model_id"), "rwkv7-g1h-2.9b-20260710", "model_id")
    _require_exact(
        payload.get("model_config_path"),
        "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_path",
    )
    _require_exact(
        payload.get("model_config_sha256"),
        "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75",
        "model_config_sha256",
    )
    _require_exact(payload.get("same_process"), True, "same_process")
    _require_exact(payload.get("warmup_count_per_route_per_cell"), 1, "warmup_count")
    _require_exact(payload.get("comparison"), "torch.equal", "comparison")
    _require_exact(payload.get("automatic_rerun_authorized"), False, "automatic_rerun")
    _require_exact(payload.get("active_injection_authorized"), False, "active_injection")
    _require_exact(payload.get("self_effect_experiment_authorized"), False, "self_effect")
    _require_exact(
        payload.get("confirmatory_decision_authorized"), False, "confirmatory_decision"
    )
    _require_exact(payload.get("execution_authorized"), True, "execution_authorized")
    _require_exact(
        payload.get("authorization_scope"),
        "real_2_9b_off_equivalence_only",
        "authorization_scope",
    )
    _require_exact(payload.get("execution_lock_env"), D4_EXECUTION_LOCK_ENV, "lock env")
    _require_exact(payload.get("execution_lock_value"), D4_EXECUTION_LOCK_VALUE, "lock value")
    _require_exact(payload.get("paths"), ["forward_one", "forward_seq"], "paths")
    _require_exact(
        payload.get("state_inputs"),
        ["none", "cloned_restored_snapshot"],
        "state_inputs",
    )
    _require_exact(
        payload.get("sequence_full_output_modes"), [False, True], "full_output modes"
    )
    _require_exact(
        payload.get("failure_action"),
        "stop_without_rerun_or_tolerance_revision",
        "failure_action",
    )
    token_sets = payload.get("non_core_token_ids")
    if not isinstance(token_sets, dict):
        raise ValueError("non_core_token_ids must be an object")
    for name in ("snapshot_prefix", "single", "sequence"):
        values = token_sets.get(name)
        if not isinstance(values, list) or not values or not all(
            isinstance(item, int) and item >= 0 for item in values
        ):
            raise ValueError(f"non_core_token_ids.{name} is invalid")
    if len(token_sets["single"]) != 1 or len(token_sets["sequence"]) <= 1:
        raise ValueError("D4 token sets do not force both dispatch paths")
    return payload


def _flatten_tensors(value: Any, path: str = "state") -> Iterable[tuple[str, Any]]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten_tensors(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_tensors(value[key], f"{path}.{key}")
    elif value is not None:
        raise TypeError(f"unsupported state component at {path}")


def _tensor_pair(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    device_equal = str(left.device) == str(right.device)
    exact = bool(torch.equal(left, right)) if shape_equal and dtype_equal else False
    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "device_equal": device_equal,
        "torch_equal": exact,
        "valid": shape_equal and dtype_equal and device_equal and exact,
    }


def _state_pair(left: Any, right: Any, torch: Any) -> dict[str, Any]:
    left_items = list(_flatten_tensors(left))
    right_items = list(_flatten_tensors(right))
    paths_equal = [path for path, _ in left_items] == [path for path, _ in right_items]
    records = []
    if paths_equal:
        for (path, left_tensor), (_, right_tensor) in zip(left_items, right_items):
            comparison = _tensor_pair(left_tensor, right_tensor, torch)
            if not comparison["valid"]:
                records.append({"path": path, **comparison})
    valid = paths_equal and not records
    return {
        "paths_equal": paths_equal,
        "component_count": len(left_items) if paths_equal else 0,
        "mismatches": records,
        "all_tensors_torch_equal": valid,
        "valid": valid,
    }


def _invoke(
    route: Any,
    tokens: list[int],
    state: Any,
    full_output: bool,
    torch: Any,
) -> tuple[Any, Any]:
    inference_mode = getattr(torch, "inference_mode", None)
    context = inference_mode() if callable(inference_mode) else nullcontext()
    with context:
        return route.forward(tokens, state, full_output)


def _matrix_cells(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    token_sets = spec["non_core_token_ids"]
    return [
        {
            "cell_id": "forward_one__none__full_output_false",
            "execution_path": "forward_one",
            "tokens": list(token_sets["single"]),
            "state_input": "none",
            "full_output": False,
        },
        {
            "cell_id": "forward_one__restored__full_output_false",
            "execution_path": "forward_one",
            "tokens": list(token_sets["single"]),
            "state_input": "cloned_restored_snapshot",
            "full_output": False,
        },
        *[
            {
                "cell_id": (
                    f"forward_seq__{state_name}__full_output_{str(full_output).lower()}"
                ),
                "execution_path": "forward_seq",
                "tokens": list(token_sets["sequence"]),
                "state_input": (
                    "none" if state_name == "none" else "cloned_restored_snapshot"
                ),
                "full_output": full_output,
            }
            for state_name in ("none", "restored")
            for full_output in (False, True)
        ],
    ]


def execute_equivalence_matrix(
    *,
    base_model: Any,
    off_g1: Any,
    off_g2: Any,
    torch: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the frozen six-cell OFF matrix on one already-loaded model."""
    prefix = list(spec["non_core_token_ids"]["snapshot_prefix"])
    _, restored_snapshot = _invoke(base_model, prefix, None, False, torch)
    snapshot_reference = clone_state(restored_snapshot)
    routes = {
        "original_baseline": base_model,
        "off_g1_passthrough": off_g1,
        "off_g2_instrumented": off_g2,
    }
    cells = []
    for cell in _matrix_cells(spec):
        outputs: dict[str, tuple[Any, Any]] = {}
        for route_name, route in routes.items():
            source = None if cell["state_input"] == "none" else restored_snapshot
            warm_state = None if source is None else clone_state(source)
            _invoke(route, cell["tokens"], warm_state, cell["full_output"], torch)
            score_state = None if source is None else clone_state(source)
            outputs[route_name] = _invoke(
                route, cell["tokens"], score_state, cell["full_output"], torch
            )

        baseline_logits, baseline_state = outputs["original_baseline"]
        comparisons = {}
        for candidate_name in ("off_g1_passthrough", "off_g2_instrumented"):
            candidate_logits, candidate_state = outputs[candidate_name]
            logits = _tensor_pair(baseline_logits, candidate_logits, torch)
            state = _state_pair(baseline_state, candidate_state, torch)
            comparisons[candidate_name] = {
                "logits": logits,
                "state": state,
                "valid": logits["valid"] and state["valid"],
            }
        g1_logits, g1_state = outputs["off_g1_passthrough"]
        g2_logits, g2_state = outputs["off_g2_instrumented"]
        g1_g2_logits = _tensor_pair(g1_logits, g2_logits, torch)
        g1_g2_state = _state_pair(g1_state, g2_state, torch)
        comparisons["off_g1_vs_off_g2"] = {
            "logits": g1_g2_logits,
            "state": g1_g2_state,
            "valid": g1_g2_logits["valid"] and g1_g2_state["valid"],
        }
        cells.append(
            {
                **cell,
                "warmup_count_per_route": 1,
                "comparisons": comparisons,
                "valid": all(item["valid"] for item in comparisons.values()),
            }
        )

    snapshot_unchanged = _state_pair(snapshot_reference, restored_snapshot, torch)
    managed_names_absent = all(
        name not in getattr(base_model, "__dict__", {})
        for name in ("forward_one", "forward_seq", CALLBACK_ATTRIBUTE)
    )
    checks = {
        "six_scored_cells_completed": len(cells) == 6,
        "all_cells_exact": all(cell["valid"] for cell in cells),
        "restored_snapshot_unchanged": snapshot_unchanged["valid"],
        "off_g1_callback_count_zero": off_g1.callback_call_count == 0,
        "off_g2_callback_count_zero": off_g2.callback_call_count == 0,
        "off_g1_projection_not_constructed": not off_g1.self_projection_constructed,
        "off_g2_projection_not_constructed": not off_g2.self_projection_constructed,
        "off_g2_temporary_bindings_restored": managed_names_absent,
        "off_g1_expected_call_count": off_g1.delegation_count == 12,
        "off_g2_expected_call_count": off_g2.execution_count == 12,
    }
    return {
        "cells": cells,
        "source_snapshot": snapshot_unchanged,
        "runtime_counts": {
            "off_g1_delegation_count": off_g1.delegation_count,
            "off_g2_execution_count": off_g2.execution_count,
            "off_g1_callback_call_count": off_g1.callback_call_count,
            "off_g2_callback_call_count": off_g2.callback_call_count,
        },
        "checks": checks,
        "valid": all(checks.values()),
    }


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _create_claim(output_dir: Path, config_path: Path, git: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        raise FileExistsError("D4 output directory is not empty; automatic rerun refused")
    claim_path = output_dir / "execution_claim.json"
    claim = {
        "claim_version": REPORT_VERSION,
        "created_at_utc": _utc_now(),
        "single_use": True,
        "automatic_rerun_authorized": False,
        "config_sha256": sha256_file(config_path),
        "git_commit": git["commit"],
    }
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(claim))
    return claim_path


def run_d4_real_off_equivalence(
    *,
    config_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    spec = _read_spec(config_file)
    if os.environ.get(D4_EXECUTION_LOCK_ENV) != D4_EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D4 execution environment lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D4 OFF-G2")

    git = _git_metadata(root)
    if git["branch"] != "main" or git["status_porcelain"]:
        raise RuntimeError("D4 requires a clean main worktree")
    model_config_path = (root / spec["model_config_path"]).resolve()
    _require_exact(
        sha256_file(model_config_path), spec["model_config_sha256"], "model config digest"
    )
    model_config = load_model_config(model_config_path, root, verify_files=True)
    _require_exact(model_config.model_id, spec["model_id"], "loaded model id")

    installed_version = version("rwkv")
    source_path = Path(distribution("rwkv").locate_file("rwkv/model.py")).resolve()
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    _require_exact(installed_version, EXPECTED_RWKV_PACKAGE_VERSION, "rwkv version")
    _require_exact(source_digest, EXPECTED_RWKV_MODEL_SOURCE_SHA256, "model.py digest")

    claim_path = _create_claim(destination, config_file, git)
    started = time.perf_counter()
    report_path = destination / "report.json"
    try:
        for key, value in model_config.environment.items():
            os.environ[key] = value
        adapter = RWKV7Adapter.load(model_config)
        torch = adapter.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        rwkv_module = sys.modules["rwkv.model"]
        off_g1 = RWKV7CouplingOffAdapter(
            base_model=adapter.model,
            upstream_package_version=installed_version,
            upstream_model_source_sha256=source_digest,
        )
        off_g2 = RWKV7InstrumentedOffRuntime(
            base_model=adapter.model,
            upstream_source_bytes=source_bytes,
            upstream_globals=vars(rwkv_module),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
        )
        matrix = execute_equivalence_matrix(
            base_model=adapter.model,
            off_g1=off_g1,
            off_g2=off_g2,
            torch=torch,
            spec=spec,
        )
        torch.cuda.synchronize()
        safety = {
            "development_only": True,
            "real_2_9b_model_loaded": True,
            "real_2_9b_model_executed": True,
            "off_g1_executed": True,
            "off_g2_executed": True,
            "active_injection_implemented": False,
            "active_injection_executed": False,
            "self_projection_constructed": False,
            "self_effect_experiment_run": False,
            "site_packages_modified": False,
            "confirmatory_decision_made": False,
            "automatic_rerun_authorized": False,
        }
        report = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": (
                "d4_real_off_equivalence_passed"
                if matrix["valid"]
                else "d4_real_off_equivalence_failed"
            ),
            "valid": matrix["valid"],
            "development_only": True,
            "git": git,
            "config": {
                "path": str(config_file),
                "sha256": sha256_file(config_file),
            },
            "model": adapter.model_metadata(),
            "upstream": {
                "package": "rwkv",
                "version": installed_version,
                "model_source_path": str(source_path),
                "model_source_sha256": source_digest,
            },
            "protocol": {
                "same_process": True,
                "comparison": "torch.equal",
                "warmup_count_per_route_per_cell": 1,
                "non_core_token_ids_only": True,
                "failure_action": "stop_without_rerun_or_tolerance_revision",
            },
            "matrix": matrix,
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "execution_claim_sha256": sha256_file(claim_path),
            "safety": safety,
        }
        report["report_digest_sha256"] = sha256_json(report)
        report_path.write_bytes(canonical_json_bytes(report))
        return report
    except BaseException as error:
        failure = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d4_execution_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        (destination / "failure.json").write_bytes(canonical_json_bytes(failure))
        raise
