from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from psa.artifacts import sha256_file, sha256_json
from psa.confirmatory.runner import build_non_core_development_fixture, condition_source_combo
from psa.supplemental.formal_run import (
    RUNNER_SOURCE_FILES,
    _atomic_write,
    execute_supplemental_group,
)


def _build_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = build_non_core_development_fixture()
    core_group = parent["groups"][0]
    records = []
    for trial in core_group["trials"]:
        common = {
            "source_factorial_group_id": core_group["factorial_group_id"],
            "source_trial_id": trial["trial_id"],
            "target_code": next(
                item["code"]
                for item in trial["option_mapping"]
                if item["identity"] == trial["target_fields"]["identity"]
                and item["goal"] == trial["target_fields"]["goal"]
            ),
            "target_fields": trial["target_fields"],
            "option_mapping": trial["option_mapping"],
            "query_prompt": trial["query_prompt"],
        }
        records.append(
            {
                "record_id": "devb-matched-" + sha256_json(common)[:20],
                "record_kind": "matched_context",
                **common,
                "history_prompt": trial["history_prompt"].replace(
                    "STORE TWO TEST FIELDS", "REVIEW TWO UNBOUND TEST FIELDS"
                ),
            }
        )
        records.append(
            {
                "record_id": "devb-generation-" + sha256_json(common)[:20],
                "record_kind": "formal_generation_readout",
                **common,
                "prompt_visible_prompt": trial["history_prompt"] + "\n\n" + trial["query_prompt"],
                "assistant_prefix": "<think></think",
                "forced_answer_prefix": ">\n",
                "maximum_generated_tokens_after_prefix": 4,
            }
        )
    control_trial = core_group["trials"][0]
    combo = (0, 0)
    for condition in (
        "continuous", "restored", "reset", "random_matched",
        "swapped_I", "swapped_G", "swapped_both", "prompt_visible_reset",
    ):
        stable = {"condition": condition, "source": control_trial["trial_id"]}
        records.append(
            {
                "record_id": "devb-control-" + sha256_json(stable)[:20],
                "record_kind": "general_capability_control_condition",
                "condition": condition,
                "assigned_factorial_group_id": core_group["factorial_group_id"],
                "assigned_source_combo": list(combo),
                "state_source_combo": None if condition == "prompt_visible_reset" else (
                    list(condition_source_combo(combo, condition))
                    if condition_source_combo(combo, condition) is not None
                    else None
                ),
                "prompt": control_trial["query_prompt"],
                "target_code": "A",
                "target_fields": {"control_value": "A"},
                "option_mapping": [{"code": code, "control_value": code} for code in "ABCD"],
            }
        )
    records.sort(key=lambda item: (item["record_kind"], item["record_id"]))
    identities = [
        {"record_id": item["record_id"], "record_kind": item["record_kind"]}
        for item in records
    ]
    plan = {
        "factorial_group_id": core_group["factorial_group_id"],
        "record_count": len(records),
        "record_kind_counts": dict(Counter(item["record_kind"] for item in records)),
        "records": records,
        "plan_digest_sha256": sha256_json(identities),
    }
    return core_group, plan


def run_exp001b_runner_development_gate(
    *, model_config_path: str | Path, bdev1_thresholds_path: str | Path,
    output_dir: str | Path, project_root: str | Path = ".",
) -> dict[str, Any]:
    """Exercise the exact formal record router using only explicit non-Core text."""
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    started = datetime.now(timezone.utc)
    core_group, plan = _build_fixture()
    import json
    thresholds = json.loads(Path(bdev1_thresholds_path).read_text(encoding="utf-8"))
    from psa.model import RWKV7Adapter, load_model_config
    from psa.supplemental.rwkv_run_backend import RWKVSupplementalRunBackend
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    config = load_model_config(model_config_path, root, verify_files=True)
    load_started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    run_started = time.perf_counter()
    result = execute_supplemental_group(
        core_group=core_group,
        plan=plan,
        backend=RWKVSupplementalRunBackend(
            adapter=adapter,
            state_norm_thresholds=thresholds,
        ),
    )
    torch.cuda.synchronize()
    _atomic_write(destination / "development_fixture.json", {"fixture_kind": "non_core_exp001b_formal_runner_fixture", "core_group": core_group, "plan": plan})
    _atomic_write(destination / "group_result.json", result)
    kinds = Counter(item["record_kind"] for item in result["records"])
    summary = {
        "summary_version": "0.1-development",
        "gate": "exp001b_formal_runner_development",
        "fixture_kind": "non_core_exp001b_formal_runner_fixture",
        "development_only": True,
        "core_set_accessed": False,
        "supplemental_set_accessed": False,
        "formal_authorization_used": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "model_id": config.model_id,
        "record_count": result["record_count"],
        "record_kind_counts": dict(kinds),
        "all_three_record_routes_exercised": set(kinds) == {"matched_context", "formal_generation_readout", "general_capability_control_condition"},
        "all_eight_control_routes_exercised": len({item["metadata"].get("condition") for item in result["records"] if item["record_kind"] == "general_capability_control_condition"}) == 8,
        "contains_derived_accuracy": False,
        "contains_interim_decision": False,
        "runner_source_digests": {
            relative: sha256_file(root / relative)
            for relative in RUNNER_SOURCE_FILES
            if (root / relative).is_file()
        },
        "load_seconds": load_seconds,
        "run_seconds": time.perf_counter() - run_started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": ["development_fixture.json", "group_result.json"],
    }
    summary["valid"] = bool(
        summary["record_count"] == 40
        and summary["all_three_record_routes_exercised"]
        and summary["all_eight_control_routes_exercised"]
        and set(summary["runner_source_digests"]) == set(RUNNER_SOURCE_FILES)
    )
    summary["route_decision"] = "run_read_only_exp001b_formal_preflight" if summary["valid"] else "repair_formal_runner_without_reading_frozen_set"
    _atomic_write(destination / "summary.json", summary)
    return summary
