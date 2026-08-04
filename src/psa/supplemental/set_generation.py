from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, payload_digest, sha256_file, sha256_json
from psa.preregistration.core_set import verify_core_set_package
from psa.preregistration.formal_freeze import (
    _load_formal_config,
    generate_control_manifest,
)
from psa.supplemental.development import fit_matched_context_history
from psa.supplemental.finalize import verify_exp001b_final_preregistration_package


EXPERIMENT_ID = "EXP-001B"
SET_STATUS = "supplemental_set_frozen_unrun"
FINAL_PREREGISTRATION_DIGEST = (
    "976cce8c9e3b53bca2d21ae43f273228c45dfc4607f5b652a3d5b5cdc5d823be"
)
PARENT_CORE_SET_DIGEST = (
    "6ea2b6be15a7728c96d84dcc8e48da64e740438980f818e78c8ee8570a47eb9d"
)
PARENT_CORE_SET_PACKAGE_DIGEST = (
    "9659e286de4128b43226f2d6df27075eba60bd953c2330ee70c0ec3e677f1642"
)
CONTROL_MANIFEST_DIGEST = (
    "30d984fc3eac987a27f27b8539b96cc7e2fd600ccd9a8c8904c7aa4f67a2e348"
)
SET_GENERATION_EXECUTION_LOCK = "AUTHORIZED_EXP001B_SET_GENERATION"
EXPECTED_COUNTS = {
    "factorial_group_count": 320,
    "source_trial_count": 5120,
    "matched_context_record_count": 5120,
    "formal_generation_record_count": 5120,
    "control_trial_count": 96,
    "control_condition_count": 8,
    "control_condition_record_count": 768,
    "total_record_count": 11008,
}
CONDITIONS = (
    "continuous",
    "restored",
    "reset",
    "random_matched",
    "swapped_I",
    "swapped_G",
    "swapped_both",
    "prompt_visible_reset",
)
SOURCE_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _parse_utc(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def expected_set_authorization_text(preflight_digest_sha256: str) -> str:
    return (
        "我确认 EXP-001B 补充集生成预检 checksum："
        f"{preflight_digest_sha256}，以及最终预注册 checksum："
        f"{FINAL_PREREGISTRATION_DIGEST}；授权生成并冻结 EXP-001B 补充测试集，"
        "共 11,008 条记录；不授权运行 EXP-001B 正式补充实验。"
    )


def _validate_authorization(value: Mapping[str, Any]) -> None:
    expected_scope = {
        "generate_and_freeze_supplemental_set": True,
        "run_supplemental_experiment": False,
    }
    preflight_digest = value.get("set_preflight_digest_sha256")
    valid = bool(
        value.get("authorization_version") == "0.1"
        and value.get("experiment_id") == EXPERIMENT_ID
        and value.get("final_preregistration_digest_sha256")
        == FINAL_PREREGISTRATION_DIGEST
        and value.get("parent_core_set_digest_sha256") == PARENT_CORE_SET_DIGEST
        and isinstance(preflight_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", preflight_digest)
        and value.get("authorized_by_role") == "project_owner"
        and value.get("authorization_text")
        == expected_set_authorization_text(str(preflight_digest))
        and value.get("authorization") == expected_scope
        and value.get("total_record_count") == EXPECTED_COUNTS["total_record_count"]
    )
    try:
        _parse_utc(value.get("authorized_at_utc"), "authorized_at_utc")
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("EXP-001B supplemental-set authorization is invalid")


def build_exp001b_set_preflight(
    *,
    final_package_dir: str | Path,
    core_set_package_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Read-only package readiness check. It neither loads a model nor generates data."""
    final_report = verify_exp001b_final_preregistration_package(
        final_package_dir, project_root=project_root
    )
    core_report = verify_core_set_package(core_set_package_dir)
    checks = {
        "final_preregistration_package_valid": bool(final_report["valid"]),
        "final_preregistration_digest_pinned": (
            final_report.get("final_preregistration_digest_sha256")
            == FINAL_PREREGISTRATION_DIGEST
        ),
        "parent_core_set_package_valid": bool(core_report["valid"]),
        "parent_core_set_digest_pinned": (
            core_report.get("core_set_digest_sha256") == PARENT_CORE_SET_DIGEST
        ),
        "parent_core_set_unrun_boundary_intact": bool(
            core_report.get("confirmatory_experiment_run") is False
            and core_report.get("confirmatory_results_observed") is False
        ),
        "record_budget_locked": sum(
            EXPECTED_COUNTS[key]
            for key in (
                "matched_context_record_count",
                "formal_generation_record_count",
                "control_condition_record_count",
            )
        )
        == EXPECTED_COUNTS["total_record_count"],
    }
    valid = all(checks.values())
    unsigned = {
        "preflight_version": "0.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "set_preflight_valid_authorization_required" if valid else "invalid",
        "checks": checks,
        "expected_counts": dict(EXPECTED_COUNTS),
        "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
        "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
        "model_loaded": False,
        "supplemental_trial_scored": False,
        "supplemental_set_generation_authorized": False,
        "supplemental_set_generated": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "route_decision": "request_project_owner_set_generation_authorization",
        "valid": valid,
    }
    return {**unsigned, "preflight_digest_sha256": sha256_json(unsigned)}


def _control_source_combo(combo: tuple[int, int], condition: str) -> list[int] | None:
    identity, goal = combo
    if condition in {"continuous", "restored"}:
        return [identity, goal]
    if condition in {"reset", "random_matched", "prompt_visible_reset"}:
        return None
    if condition == "swapped_I":
        return [1 - identity, goal]
    if condition == "swapped_G":
        return [identity, 1 - goal]
    if condition == "swapped_both":
        return [1 - identity, 1 - goal]
    raise ValueError(f"unknown EXP-001B condition: {condition}")


def _balanced_assignments(group_ids: Sequence[str], seed: int) -> list[dict[str, Any]]:
    if len(group_ids) != 320 or len(set(group_ids)) != 320:
        raise ValueError("control assignment requires 320 unique frozen group IDs")
    rng = random.Random(seed)
    selected = rng.sample(sorted(group_ids), 96)
    combos = [combo for combo in SOURCE_COMBOS for _ in range(24)]
    rng.shuffle(combos)
    return [
        {"factorial_group_id": group_id, "source_combo": list(combo)}
        for group_id, combo in zip(selected, combos, strict=True)
    ]


def _build_records(
    *,
    core_set: Mapping[str, Any],
    design: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    token_counter: Callable[[str], int],
) -> dict[str, list[dict[str, Any]]]:
    groups = core_set.get("groups")
    if not isinstance(groups, list) or len(groups) != 320:
        raise ValueError("EXP-001B requires the complete 320-group Core Set")
    matched_design = design["matched_context"]
    templates = matched_design["templates"]
    fillers = {
        item["variant_id"]: item["text"] for item in core_set["filler_variants"]
    }
    template_indices = [index for index in range(4) for _ in range(1280)]
    random.Random(int(design["seeds"]["matched_context_generator"])).shuffle(
        template_indices
    )
    matched: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    fit_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    flat_index = 0
    for group in sorted(groups, key=lambda item: int(item["group_index"])):
        filler = fillers[group["filler_variant_id"]]
        for trial in group["trials"]:
            template = templates[template_indices[flat_index]]
            target = trial["target_fields"]
            cache_key = (
                trial["history_prompt_digest_sha256"],
                template["id"],
                target["domain"],
                target["operation"],
            )
            fitted = fit_cache.get(cache_key)
            if fitted is None:
                fitted = fit_matched_context_history(
                    original_history=trial["history_prompt"],
                    template=template,
                    domain=target["domain"],
                    operation=target["operation"],
                    filler=filler,
                    padding_fragments=matched_design["padding_fragments"],
                    token_counter=token_counter,
                )
                fit_cache[cache_key] = fitted
            text = fitted["text"]
            forbidden = [
                phrase
                for phrase in matched_design["forbidden_binding_phrases"]
                if phrase in text
            ]
            if not (
                fitted["token_count_exact"]
                and fitted["filler_exact_substring"]
                and text.count(target["domain"]) == 1
                and text.count(target["operation"]) == 1
                and not forbidden
            ):
                raise ValueError("matched-context record violates frozen rules")
            common = {
                "source_factorial_group_id": group["factorial_group_id"],
                "source_trial_id": trial["trial_id"],
                "source_semantic_case_id": trial["semantic_case_id"],
                "rotation_index": trial["rotation_index"],
                "target_code": trial["target_code"],
                "target_fields": trial["target_fields"],
                "option_mapping": trial["option_mapping"],
                "query_prompt": trial["query_prompt"],
                "query_prompt_digest_sha256": trial["query_prompt_digest_sha256"],
            }
            matched.append(
                {
                    "record_id": "exp001b-matched-" + sha256_json(common)[:24],
                    "record_kind": "matched_context",
                    **common,
                    "matched_context_template_id": template["id"],
                    "history_prompt": text,
                    "history_prompt_digest_sha256": sha256_json(text),
                    "source_history_prompt_digest_sha256": trial[
                        "history_prompt_digest_sha256"
                    ],
                    "target_history_token_count": fitted["target_token_count"],
                    "matched_history_token_count": fitted["matched_token_count"],
                    "token_count_exact": True,
                    "filler_variant_id": group["filler_variant_id"],
                    "filler_copied_exactly": True,
                    "padding_fragment": fitted["padding_fragment"],
                    "padding_repetitions": fitted["padding_repetitions"],
                }
            )
            prompt = trial["history_prompt"] + "\n\n" + trial["query_prompt"]
            generated.append(
                {
                    "record_id": "exp001b-generation-" + sha256_json(common)[:24],
                    "record_kind": "formal_generation_readout",
                    **common,
                    "prompt_visible_prompt": prompt,
                    "prompt_visible_prompt_digest_sha256": sha256_json(prompt),
                    "assistant_prefix": design["formal_generation_readout"][
                        "assistant_prefix"
                    ],
                    "forced_answer_prefix": design["formal_generation_readout"][
                        "forced_answer_prefix"
                    ],
                    "maximum_generated_tokens_after_prefix": design[
                        "formal_generation_readout"
                    ]["maximum_generated_tokens_after_prefix"],
                }
            )
            flat_index += 1
    controls = generate_control_manifest(formal_config)
    if controls["manifest_digest_sha256"] != CONTROL_MANIFEST_DIGEST:
        raise ValueError("D5 control manifest digest differs from frozen EXP-001B lock")
    assignments = _balanced_assignments(
        [group["factorial_group_id"] for group in groups],
        int(design["seeds"]["control_assignment"]),
    )
    control_records: list[dict[str, Any]] = []
    for trial, assignment in zip(controls["trials"], assignments, strict=True):
        combo = tuple(assignment["source_combo"])
        for condition in CONDITIONS:
            stable = {
                "source_control_sample_id": trial["sample_id"],
                "condition": condition,
                "assigned_factorial_group_id": assignment["factorial_group_id"],
            }
            control_records.append(
                {
                    "record_id": "exp001b-control-" + sha256_json(stable)[:24],
                    "record_kind": "general_capability_control_condition",
                    **stable,
                    "semantic_case_id": trial["semantic_case_id"],
                    "task_type": trial["task_type"],
                    "rotation_index": trial["rotation_index"],
                    "prompt": trial["prompt"],
                    "prompt_digest_sha256": trial["prompt_digest_sha256"],
                    "target_code": trial["target_code"],
                    "target_fields": trial["target_fields"],
                    "option_mapping": trial["option_mapping"],
                    "assigned_source_combo": list(combo),
                    "state_source_combo": _control_source_combo(combo, condition),
                }
            )
    return {"matched_context": matched, "formal_generation": generated, "controls": control_records}


def _validate_payload(payload: Mapping[str, Any]) -> None:
    expected = payload.get("supplemental_set_digest_sha256")
    unsigned = dict(payload)
    unsigned.pop("supplemental_set_digest_sha256", None)
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("supplemental-set records are missing")
    counts = {
        "matched_context_record_count": len(records.get("matched_context", [])),
        "formal_generation_record_count": len(records.get("formal_generation", [])),
        "control_condition_record_count": len(records.get("controls", [])),
    }
    ids = [item["record_id"] for values in records.values() for item in values]
    matched = records.get("matched_context", [])
    generated = records.get("formal_generation", [])
    controls = records.get("controls", [])
    matched_sources = {item.get("source_trial_id") for item in matched}
    generated_sources = {item.get("source_trial_id") for item in generated}
    control_by_trial: dict[str, list[Mapping[str, Any]]] = {}
    for item in controls:
        control_by_trial.setdefault(str(item.get("source_control_sample_id")), []).append(item)
    assignment_rows = [items[0] for items in control_by_trial.values() if items]
    control_structure_valid = bool(
        len(control_by_trial) == EXPECTED_COUNTS["control_trial_count"]
        and all(
            len(items) == len(CONDITIONS)
            and {item.get("condition") for item in items} == set(CONDITIONS)
            and len({item.get("assigned_factorial_group_id") for item in items}) == 1
            and len({tuple(item.get("assigned_source_combo", [])) for item in items}) == 1
            for items in control_by_trial.values()
        )
        and len({item.get("assigned_factorial_group_id") for item in assignment_rows})
        == EXPECTED_COUNTS["control_trial_count"]
        and Counter(
            tuple(item.get("assigned_source_combo", [])) for item in assignment_rows
        )
        == Counter({(0, 0): 24, (0, 1): 24, (1, 0): 24, (1, 1): 24})
    )
    valid = bool(
        payload.get("status") == SET_STATUS
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("final_preregistration_digest_sha256")
        == FINAL_PREREGISTRATION_DIGEST
        and payload.get("parent_core_set_digest_sha256") == PARENT_CORE_SET_DIGEST
        and isinstance(payload.get("set_preflight_digest_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("set_preflight_digest_sha256"))
        )
        and payload.get("record_counts") == EXPECTED_COUNTS
        and counts["matched_context_record_count"]
        == EXPECTED_COUNTS["matched_context_record_count"]
        and counts["formal_generation_record_count"]
        == EXPECTED_COUNTS["formal_generation_record_count"]
        and counts["control_condition_record_count"]
        == EXPECTED_COUNTS["control_condition_record_count"]
        and len(ids) == EXPECTED_COUNTS["total_record_count"]
        and len(set(ids)) == len(ids)
        and all(item.get("record_kind") == "matched_context" for item in matched)
        and all(item.get("token_count_exact") is True for item in matched)
        and all(item.get("filler_copied_exactly") is True for item in matched)
        and len(matched_sources) == EXPECTED_COUNTS["source_trial_count"]
        and None not in matched_sources
        and all(
            item.get("record_kind") == "formal_generation_readout"
            for item in generated
        )
        and len(generated_sources) == EXPECTED_COUNTS["source_trial_count"]
        and generated_sources == matched_sources
        and all(
            item.get("record_kind") == "general_capability_control_condition"
            for item in controls
        )
        and control_structure_valid
        and isinstance(expected, str)
        and sha256_json(unsigned) == expected
        and payload.get("supplemental_experiment_authorized") is False
        and payload.get("supplemental_experiment_run") is False
        and payload.get("supplemental_results_observed") is False
    )
    if not valid:
        raise ValueError("EXP-001B supplemental-set payload is invalid")


def verify_exp001b_supplemental_set_package(
    package_dir: str | Path,
) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    manifest = _load_object(root / "manifest.json", "EXP-001B set manifest")
    locked = manifest.get("locked_files")
    file_checks = {
        str(name): bool((root / str(name)).is_file() and sha256_file(root / str(name)) == digest)
        for name, digest in locked.items()
    } if isinstance(locked, Mapping) else {}
    unsigned_manifest = dict(manifest)
    expected_manifest_digest = unsigned_manifest.pop(
        "supplemental_set_package_digest_sha256", None
    )
    content_valid = False
    try:
        payload = _load_object(root / "supplemental_set.json", "supplemental set")
        authorization = _load_object(root / "set_generation_authorization.json", "set authorization")
        final_manifest = _load_object(
            root / "exp001b_final_manifest.json", "EXP-001B final manifest copy"
        )
        parent_manifest = _load_object(
            root / "parent_core_manifest.json", "parent Core Set manifest copy"
        )
        _validate_authorization(authorization)
        _validate_payload(payload)
        content_valid = bool(
            manifest.get("supplemental_set_digest_sha256")
            == payload.get("supplemental_set_digest_sha256")
            and final_manifest.get("final_preregistration_digest_sha256")
            == FINAL_PREREGISTRATION_DIGEST
            and parent_manifest.get("core_set_digest_sha256")
            == PARENT_CORE_SET_DIGEST
            and parent_manifest.get("core_set_package_digest_sha256")
            == PARENT_CORE_SET_PACKAGE_DIGEST
            and payload.get("set_preflight_digest_sha256")
            == authorization.get("set_preflight_digest_sha256")
            == manifest.get("set_preflight_digest_sha256")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        content_valid = False
    safety = manifest.get("safety_boundary")
    expected_locked_files = {
        "supplemental_set.json",
        "set_generation_authorization.json",
        "exp001b_final_manifest.json",
        "parent_core_manifest.json",
    }
    valid = bool(
        manifest.get("status") == SET_STATUS
        and manifest.get("final_preregistration_digest_sha256")
        == FINAL_PREREGISTRATION_DIGEST
        and manifest.get("parent_core_set_digest_sha256") == PARENT_CORE_SET_DIGEST
        and manifest.get("record_counts") == EXPECTED_COUNTS
        and manifest.get("authorization")
        == {
            "generate_and_freeze_supplemental_set": True,
            "run_supplemental_experiment": False,
        }
        and set(file_checks) == expected_locked_files
        and isinstance(expected_manifest_digest, str)
        and sha256_json(unsigned_manifest) == expected_manifest_digest
        and file_checks
        and all(file_checks.values())
        and payload_digest(dict(locked))
        == manifest.get("package_payload_root_digest_sha256")
        and content_valid
        and isinstance(safety, Mapping)
        and safety.get("supplemental_set_generated") is True
        and safety.get("supplemental_set_frozen") is True
        and safety.get("supplemental_experiment_authorized") is False
        and safety.get("supplemental_experiment_run") is False
        and safety.get("supplemental_results_observed") is False
    )
    return {
        "verification_version": "0.1",
        "package_dir": str(root),
        "status": manifest.get("status"),
        "supplemental_set_digest_sha256": manifest.get("supplemental_set_digest_sha256"),
        "supplemental_set_package_digest_sha256": expected_manifest_digest,
        "record_counts": manifest.get("record_counts"),
        "failed_locked_files": [name for name, ok in file_checks.items() if not ok],
        "content_valid": content_valid,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "valid": valid,
    }


def generate_and_freeze_exp001b_supplemental_set(
    *,
    final_package_dir: str | Path,
    core_set_package_dir: str | Path,
    authorization_path: str | Path,
    formal_config_path: str | Path,
    output_dir: str | Path,
    token_counter: Callable[[str], int],
    tokenizer_provenance: Mapping[str, Any],
    execution_lock: str,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    if execution_lock != SET_GENERATION_EXECUTION_LOCK:
        raise PermissionError("EXP-001B supplemental-set execution lock is absent")
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    authorization = _load_object(authorization_path, "set authorization")
    _validate_authorization(authorization)
    preflight = build_exp001b_set_preflight(
        final_package_dir=final_package_dir,
        core_set_package_dir=core_set_package_dir,
        project_root=root,
    )
    if not preflight["valid"]:
        raise ValueError("EXP-001B supplemental-set preflight failed")
    if authorization.get("set_preflight_digest_sha256") != preflight.get(
        "preflight_digest_sha256"
    ):
        raise ValueError("EXP-001B authorization does not bind the live preflight")
    if destination.exists():
        existing = verify_exp001b_supplemental_set_package(destination)
        if existing["valid"]:
            return existing
        raise ValueError("EXP-001B supplemental-set output already exists and is invalid")
    final_candidate = _load_object(
        Path(final_package_dir) / "preregistration_candidate.json", "EXP-001B candidate"
    )
    core_set = _load_object(Path(core_set_package_dir) / "core_set.json", "Core Set")
    formal_config = _load_formal_config(Path(formal_config_path).resolve(), root)
    records = _build_records(
        core_set=core_set,
        design=final_candidate["locked_design"],
        formal_config=formal_config,
        token_counter=token_counter,
    )
    payload: dict[str, Any] = {
        "supplemental_set_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "status": SET_STATUS,
        "frozen_at_utc": authorization["authorized_at_utc"],
        "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
        "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
        "set_preflight_digest_sha256": preflight["preflight_digest_sha256"],
        "control_manifest_digest_sha256": CONTROL_MANIFEST_DIGEST,
        "record_counts": dict(EXPECTED_COUNTS),
        "tokenizer": dict(tokenizer_provenance),
        "records": records,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
    }
    payload["supplemental_set_digest_sha256"] = sha256_json(payload)
    _validate_payload(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_json(temporary / "supplemental_set.json", payload)
        _write_json(temporary / "set_generation_authorization.json", authorization)
        shutil.copyfile(Path(final_package_dir) / "manifest.json", temporary / "exp001b_final_manifest.json")
        shutil.copyfile(Path(core_set_package_dir) / "manifest.json", temporary / "parent_core_manifest.json")
        locked_files = {
            name: sha256_file(temporary / name)
            for name in (
                "supplemental_set.json",
                "set_generation_authorization.json",
                "exp001b_final_manifest.json",
                "parent_core_manifest.json",
            )
        }
        manifest: dict[str, Any] = {
            "package_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "status": SET_STATUS,
            "frozen_at_utc": authorization["authorized_at_utc"],
            "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
            "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
            "set_preflight_digest_sha256": preflight["preflight_digest_sha256"],
            "supplemental_set_digest_sha256": payload["supplemental_set_digest_sha256"],
            "record_counts": dict(EXPECTED_COUNTS),
            "authorization": authorization["authorization"],
            "locked_files": locked_files,
            "package_payload_root_digest_sha256": payload_digest(locked_files),
            "safety_boundary": {
                "supplemental_set_generated": True,
                "supplemental_set_frozen": True,
                "supplemental_experiment_authorized": False,
                "supplemental_experiment_run": False,
                "supplemental_results_observed": False,
                "automatic_rerun_authorized": False,
            },
        }
        manifest["supplemental_set_package_digest_sha256"] = sha256_json(manifest)
        _write_json(temporary / "manifest.json", manifest)
        if not verify_exp001b_supplemental_set_package(temporary)["valid"]:
            raise RuntimeError("staged EXP-001B supplemental set failed self-check")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_exp001b_supplemental_set_package(destination)
