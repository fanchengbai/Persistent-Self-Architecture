from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.confirmatory.runner import (
    build_non_core_development_fixture,
    run_development_fixture,
)
from psa.confirmatory.rwkv_backend import RWKVConfirmatoryBackend
from psa.model import RWKV7Adapter, load_model_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def run_confirmatory_runner_development_gate(
    *,
    model_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    started_at = datetime.now(timezone.utc)
    config = load_model_config(model_config_path, root, verify_files=True)
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    fixture = build_non_core_development_fixture()
    runner_source_digests = {
        relative: sha256_file(root / relative)
        for relative in (
            "src/psa/confirmatory/runner.py",
            "src/psa/confirmatory/rwkv_backend.py",
            "src/psa/confirmatory/development.py",
        )
    }
    fixture["runner_source_digests"] = runner_source_digests
    fixture.pop("fixture_digest_sha256", None)
    fixture["fixture_digest_sha256"] = sha256_json(fixture)
    fixture_path = destination / "development_fixture.json"
    _write_json(fixture_path, fixture)
    backend = RWKVConfirmatoryBackend(adapter=adapter)
    run_started = time.perf_counter()
    manifest = run_development_fixture(
        dataset=fixture,
        backend=backend,
        output_dir=destination,
    )
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started
    summary = {
        "summary_version": "0.1",
        "gate": "impl5b_confirmatory_runner_development",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "fixture_kind": fixture["fixture_kind"],
        "fixture_digest_sha256": fixture["fixture_digest_sha256"],
        "runner_source_digests": runner_source_digests,
        "model_id": config.model_id,
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "group_count": manifest["completed_group_count"],
        "trial_count": 16,
        "condition_count": 8,
        "raw_record_count": 128,
        "runner_manifest_status": manifest["status"],
        "contains_derived_accuracy": False,
        "contains_interim_decision": False,
        "formal_authorization_used": False,
        "confirmatory_experiment_run": False,
        "confirmatory_results_observed": False,
        "route_decision": "rerun_non_inference_preflight_with_runner_sources",
        "reports": [
            "development_fixture.json",
            "manifest.json",
            "groups/devgrp-impl5b-noncore-v1.json",
        ],
        "valid": bool(
            manifest.get("valid") is True
            and manifest.get("status") == "development_fixture_complete"
        ),
    }
    _write_json(destination / "summary.json", summary)
    return summary
