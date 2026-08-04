from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from psa.artifacts import canonical_json_bytes, payload_digest, sha256_file, sha256_json
from psa.assets import load_manifest, verify_manifest
from psa.confirmatory.preflight import _asset_model_consistency, _environment_checks
from psa.environment import collect_environment
from psa.preregistration import verify_core_set_package
from psa.supplemental.finalize import verify_exp001b_final_preregistration_package
from psa.supplemental.set_generation import (
    EXPECTED_COUNTS,
    FINAL_PREREGISTRATION_DIGEST,
    PARENT_CORE_SET_DIGEST,
    PARENT_CORE_SET_PACKAGE_DIGEST,
    verify_exp001b_supplemental_set_package,
)


EXPERIMENT_ID = "EXP-001B"
MODEL_ID = "rwkv7-g1h-2.9b-20260710"
SUPPLEMENTAL_SET_DIGEST = (
    "7c3606be819d4e6cc5420f0bf36efd1906f8954d362d83e912785cc943565d33"
)
SUPPLEMENTAL_SET_PACKAGE_DIGEST = (
    "68e9a9a79fe4e493a0c64ba8c0278c300cc832d940ab902feaceb4ad7f9d5954"
)
RUN_EXECUTION_LOCK = "AUTHORIZED_EXP001B_SUPPLEMENTAL_RUN"
EXPECTED_GROUP_COUNT = 320
EXPECTED_RAW_RECORD_COUNT = EXPECTED_COUNTS["total_record_count"]
RECORD_KINDS = (
    "matched_context",
    "formal_generation_readout",
    "general_capability_control_condition",
)
RUNNER_SOURCE_FILES = (
    ".gitignore",
    "configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json",
    "configs/models/rwkv7_g1h_2.9b.candidate.json",
    "src/psa/cli.py",
    "src/psa/artifacts/integrity.py",
    "src/psa/confirmatory/runner.py",
    "src/psa/confirmatory/rwkv_backend.py",
    "src/psa/development/history_binding.py",
    "src/psa/development/impl3.py",
    "src/psa/model/rwkv7.py",
    "src/psa/state/operations.py",
    "src/psa/supplemental/development.py",
    "src/psa/supplemental/formal_run.py",
    "src/psa/supplemental/run_development.py",
    "src/psa/supplemental/rwkv_run_backend.py",
    "src/psa/supplemental/finalize.py",
    "src/psa/supplemental/set_generation.py",
    "scripts/run_exp001b_runner_development_gate.sh",
    "scripts/preflight_exp001b_supplemental_run.sh",
    "scripts/run_exp001b_supplemental.sh",
    "scripts/verify_exp001b_supplemental_raw.sh",
    "schemas/exp001b_supplemental_run_authorization.schema.json",
)


class SupplementalRecordBackend(Protocol):
    def score_record(
        self,
        *,
        core_group: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        data = canonical_json_bytes(dict(value))
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


def _finite_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set("ABCD"):
        raise ValueError("record backend must return exactly A-D option scores")
    scores = {code: float(value[code]) for code in "ABCD"}
    if not all(math.isfinite(item) for item in scores.values()):
        raise ValueError("record backend option scores must be finite")
    return scores


def build_supplemental_group_plan(
    supplemental_set: Mapping[str, Any],
    core_set: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Map the 11,008 frozen records onto all 320 immutable parent groups."""
    records = supplemental_set.get("records")
    groups = core_set.get("groups")
    if not isinstance(records, Mapping) or not isinstance(groups, list):
        raise ValueError("supplemental or parent dataset structure is missing")
    group_ids = [
        group.get("factorial_group_id")
        for group in groups
        if isinstance(group, Mapping)
    ]
    if len(group_ids) != EXPECTED_GROUP_COUNT or len(set(group_ids)) != len(group_ids):
        raise ValueError("parent Core Set must contain 320 unique groups")
    by_group: dict[str, list[dict[str, Any]]] = {str(item): [] for item in group_ids}
    for kind in ("matched_context", "formal_generation"):
        values = records.get(kind)
        if not isinstance(values, list):
            raise ValueError(f"supplemental records.{kind} must be a list")
        for record in values:
            if not isinstance(record, Mapping):
                raise ValueError("supplemental records must be objects")
            group_id = record.get("source_factorial_group_id")
            if group_id not in by_group:
                raise ValueError("supplemental source group is not in the parent Core Set")
            by_group[str(group_id)].append(dict(record))
    controls = records.get("controls")
    if not isinstance(controls, list):
        raise ValueError("supplemental records.controls must be a list")
    for record in controls:
        if not isinstance(record, Mapping):
            raise ValueError("control records must be objects")
        group_id = record.get("assigned_factorial_group_id")
        if group_id not in by_group:
            raise ValueError("control assignment is not in the parent Core Set")
        by_group[str(group_id)].append(dict(record))

    plans = []
    all_ids: list[str] = []
    for group_id in group_ids:
        group_records = sorted(
            by_group[str(group_id)],
            key=lambda item: (str(item.get("record_kind")), str(item.get("record_id"))),
        )
        kind_counts = Counter(item.get("record_kind") for item in group_records)
        if (
            kind_counts["matched_context"] != 16
            or kind_counts["formal_generation_readout"] != 16
            or kind_counts["general_capability_control_condition"] not in {0, 8}
            or len(group_records) not in {32, 40}
        ):
            raise ValueError(f"invalid frozen record allocation for group {group_id}")
        identities = [
            {"record_id": item["record_id"], "record_kind": item["record_kind"]}
            for item in group_records
        ]
        all_ids.extend(item["record_id"] for item in group_records)
        plans.append(
            {
                "factorial_group_id": group_id,
                "record_count": len(group_records),
                "record_kind_counts": dict(sorted(kind_counts.items())),
                "records": group_records,
                "plan_digest_sha256": sha256_json(identities),
            }
        )
    if len(all_ids) != EXPECTED_RAW_RECORD_COUNT or len(set(all_ids)) != len(all_ids):
        raise ValueError("supplemental run plan does not cover 11,008 unique records")
    return plans


def execute_supplemental_group(
    *,
    core_group: Mapping[str, Any],
    plan: Mapping[str, Any],
    backend: SupplementalRecordBackend,
) -> dict[str, Any]:
    start_group = getattr(backend, "start_group", None)
    end_group = getattr(backend, "end_group", None)
    group_metadata = getattr(backend, "group_metadata", None)
    outputs: list[dict[str, Any]] = []
    try:
        if callable(start_group):
            start_group(core_group, plan["records"])
        for frozen in plan["records"]:
            result = backend.score_record(core_group=core_group, record=frozen)
            if not isinstance(result, Mapping):
                raise ValueError("record backend result must be an object")
            output = {
                "record_id": frozen["record_id"],
                "record_kind": frozen["record_kind"],
                "option_scores": _finite_scores(result.get("option_scores")),
                "metadata": dict(result.get("metadata", {})),
            }
            if not isinstance(result.get("metadata", {}), Mapping):
                raise ValueError("record backend metadata must be an object")
            if frozen["record_kind"] == "formal_generation_readout":
                generated_text = result.get("generated_text")
                token_ids = result.get("generated_token_ids")
                format_valid = result.get("format_valid")
                generated_choice = result.get("generated_choice")
                if (
                    not isinstance(generated_text, str)
                    or not isinstance(token_ids, list)
                    or any(not isinstance(item, int) for item in token_ids)
                    or not isinstance(format_valid, bool)
                    or generated_choice not in {None, "A", "B", "C", "D"}
                ):
                    raise ValueError("generation backend fields are invalid")
                output.update(
                    {
                        "generated_text": generated_text,
                        "generated_token_ids": token_ids,
                        "generated_choice": generated_choice,
                        "format_valid": format_valid,
                    }
                )
            outputs.append(output)
        metadata = dict(group_metadata()) if callable(group_metadata) else {}
    finally:
        if callable(end_group):
            end_group(core_group)
    return {
        "group_result_version": "0.1",
        "factorial_group_id": plan["factorial_group_id"],
        "plan_digest_sha256": plan["plan_digest_sha256"],
        "record_count": len(outputs),
        "record_kind_counts": dict(Counter(item["record_kind"] for item in outputs)),
        "records": outputs,
        "contains_derived_accuracy": False,
        "contains_interim_decision": False,
        "backend_group_metadata": metadata,
        "group_result_digest_sha256": sha256_json(outputs),
    }


def _runner_source_digests(root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(root / relative)
        for relative in RUNNER_SOURCE_FILES
        if (root / relative).is_file()
    }


def build_exp001b_run_preflight(
    *,
    project_root: str | Path,
    final_package_dir: str | Path,
    core_set_package_dir: str | Path,
    supplemental_set_package_dir: str | Path,
    model_config_path: str | Path,
    asset_manifest_path: str | Path,
    asset_root: str | Path,
    runner_evidence_path: str | Path | None = None,
    environment_report: Mapping[str, Any] | None = None,
    asset_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only, non-inference launch preflight for EXP-001B."""
    root = Path(project_root).resolve()
    final_root = Path(final_package_dir).resolve()
    core_root = Path(core_set_package_dir).resolve()
    set_root = Path(supplemental_set_package_dir).resolve()
    model_path = Path(model_config_path).resolve()
    asset_path = Path(asset_manifest_path).resolve()
    final_report = verify_exp001b_final_preregistration_package(final_root, project_root=root)
    core_report = verify_core_set_package(core_root)
    set_report = verify_exp001b_supplemental_set_package(set_root)
    final_manifest = _load_object(final_root / "manifest.json", "EXP-001B final manifest")
    core_manifest = _load_object(core_root / "manifest.json", "parent Core manifest")
    set_manifest = _load_object(set_root / "manifest.json", "supplemental manifest")
    core_set = _load_object(core_root / "core_set.json", "parent Core Set")
    supplemental_set = _load_object(set_root / "supplemental_set.json", "supplemental set")
    model_config = _load_object(model_path, "model config")
    plans = build_supplemental_group_plan(supplemental_set, core_set)
    if environment_report is None:
        environment_report = collect_environment(root)
    if asset_report is None:
        asset_report = verify_manifest(load_manifest(asset_path), root=asset_root)
    source_digests = _runner_source_digests(root)
    runner_sources_complete = set(source_digests) == set(RUNNER_SOURCE_FILES)
    evidence = None
    evidence_path = None
    if runner_evidence_path is not None:
        evidence_path = Path(runner_evidence_path).resolve()
        evidence = _load_object(evidence_path, "EXP-001B runner evidence")
    evidence_valid = bool(
        isinstance(evidence, Mapping)
        and evidence.get("valid") is True
        and evidence.get("gate") == "exp001b_formal_runner_development"
        and evidence.get("development_only") is True
        and evidence.get("fixture_kind") == "non_core_exp001b_formal_runner_fixture"
        and evidence.get("core_set_accessed") is False
        and evidence.get("supplemental_set_accessed") is False
        and evidence.get("formal_authorization_used") is False
        and evidence.get("supplemental_experiment_run") is False
        and evidence.get("supplemental_results_observed") is False
        and evidence.get("runner_source_digests") == source_digests
    )
    checks = {
        "final_preregistration_package_valid": final_report.get("valid") is True,
        "parent_core_set_package_valid": core_report.get("valid") is True,
        "supplemental_set_package_valid": set_report.get("valid") is True,
        "final_preregistration_digest_pinned": final_manifest.get("final_preregistration_digest_sha256") == FINAL_PREREGISTRATION_DIGEST,
        "parent_core_set_digest_pinned": core_manifest.get("core_set_digest_sha256") == PARENT_CORE_SET_DIGEST,
        "parent_core_package_digest_pinned": core_manifest.get("core_set_package_digest_sha256") == PARENT_CORE_SET_PACKAGE_DIGEST,
        "supplemental_set_digest_pinned": set_manifest.get("supplemental_set_digest_sha256") == SUPPLEMENTAL_SET_DIGEST,
        "supplemental_package_digest_pinned": set_manifest.get("supplemental_set_package_digest_sha256") == SUPPLEMENTAL_SET_PACKAGE_DIGEST,
        "model_identity_pinned": model_config.get("model_id") == final_manifest.get("model_id") == MODEL_ID,
        "record_plan_complete": len(plans) == EXPECTED_GROUP_COUNT and sum(item["record_count"] for item in plans) == EXPECTED_RAW_RECORD_COUNT,
        "runner_sources_complete": runner_sources_complete,
        **_environment_checks(environment_report),
        **_asset_model_consistency(model_config, asset_report),
    }
    stable_plan = {
        "plan_version": "0.1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "git_commit": environment_report.get("git", {}).get("commit"),
        "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
        "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
        "parent_core_set_package_digest_sha256": PARENT_CORE_SET_PACKAGE_DIGEST,
        "supplemental_set_digest_sha256": SUPPLEMENTAL_SET_DIGEST,
        "supplemental_set_package_digest_sha256": SUPPLEMENTAL_SET_PACKAGE_DIGEST,
        "expected_group_ids": [item["factorial_group_id"] for item in plans],
        "expected_group_count": EXPECTED_GROUP_COUNT,
        "expected_raw_record_count": EXPECTED_RAW_RECORD_COUNT,
        "expected_record_counts": EXPECTED_COUNTS,
        "group_plan_digests": {item["factorial_group_id"]: item["plan_digest_sha256"] for item in plans},
        "runner_source_digests": source_digests,
        "runner_development_evidence_sha256": sha256_file(evidence_path) if evidence_valid and evidence_path else None,
        "safety_rules": {
            "partial_run_forbidden": True,
            "intermediate_accuracy_reporting_forbidden": True,
            "automatic_rerun_forbidden": True,
            "frozen_design_mutation_forbidden": True,
            "rerun_exp001_primary_forbidden": True,
            "separate_project_owner_authorization_required": True,
        },
    }
    digest = sha256_json(stable_plan)
    valid = all(checks.values())
    ready = valid and evidence_valid
    return {
        "preflight_version": "0.1",
        "created_at_utc": _utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "development_only": True,
        "model_loaded": False,
        "supplemental_trial_scored": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "status": "preflight_valid_authorization_still_required" if ready else ("preflight_valid_runner_evidence_required" if valid else "preflight_failed"),
        "route_decision": "review_project_owner_supplemental_run_authorization" if ready else ("run_non_core_exp001b_runner_development_gate" if valid else "hold_and_repair_without_formal_inference"),
        "checks": checks,
        "runner_development_evidence": {"provided": evidence is not None, "valid": evidence_valid},
        "run_plan_candidate": stable_plan,
        "preflight_digest_sha256": digest,
        "paths": {
            "final_package": _relative(final_root, root),
            "core_set_package": _relative(core_root, root),
            "supplemental_set_package": _relative(set_root, root),
            "model_config": _relative(model_path, root),
            "asset_manifest": _relative(asset_path, root),
        },
        "authorization_boundary": {
            "set_generation_authorization_allows_run": False,
            "new_project_owner_authorization_required": True,
            "authorization_must_bind_preflight_digest_sha256": digest,
        },
        "valid": valid,
    }


def verify_exp001b_run_authorization(
    authorization: Mapping[str, Any], *, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    expected_scope = {
        "run_supplemental_experiment": True,
        "observe_results_after_full_completion": True,
        "modify_frozen_design": False,
        "automatic_rerun_after_results": False,
        "rerun_exp001_primary_experiment": False,
    }
    expected_keys = {
        "authorization_version", "experiment_id", "authorized_by_role",
        "authorized_at_utc", "authorization_text", "preflight_digest_sha256",
        "final_preregistration_digest_sha256", "parent_core_set_digest_sha256",
        "parent_core_set_package_digest_sha256", "supplemental_set_digest_sha256",
        "supplemental_set_package_digest_sha256", "model_id", "authorization",
    }
    checks = {
        "preflight_authorization_ready": preflight.get("valid") is True and preflight.get("status") == "preflight_valid_authorization_still_required" and preflight.get("runner_development_evidence", {}).get("valid") is True,
        "authorization_shape_exact": set(authorization) == expected_keys,
        "authorization_version_valid": authorization.get("authorization_version") == "1.0",
        "experiment_identity_valid": authorization.get("experiment_id") == EXPERIMENT_ID,
        "authorized_by_project_owner": authorization.get("authorized_by_role") == "project_owner",
        "authorization_timestamp_present": isinstance(authorization.get("authorized_at_utc"), str) and len(authorization.get("authorized_at_utc", "")) >= 20,
        "authorization_text_present": isinstance(authorization.get("authorization_text"), str) and len(authorization.get("authorization_text", "").strip()) >= 20,
        "preflight_digest_bound": authorization.get("preflight_digest_sha256") == preflight.get("preflight_digest_sha256"),
        "final_digest_bound": authorization.get("final_preregistration_digest_sha256") == FINAL_PREREGISTRATION_DIGEST,
        "parent_core_digest_bound": authorization.get("parent_core_set_digest_sha256") == PARENT_CORE_SET_DIGEST,
        "parent_core_package_bound": authorization.get("parent_core_set_package_digest_sha256") == PARENT_CORE_SET_PACKAGE_DIGEST,
        "supplemental_set_digest_bound": authorization.get("supplemental_set_digest_sha256") == SUPPLEMENTAL_SET_DIGEST,
        "supplemental_package_digest_bound": authorization.get("supplemental_set_package_digest_sha256") == SUPPLEMENTAL_SET_PACKAGE_DIGEST,
        "model_id_bound": authorization.get("model_id") == MODEL_ID,
        "scope_exact": authorization.get("authorization") == expected_scope,
    }
    return {"verification_version": "0.1", "checks": checks, "valid": all(checks.values())}


def prepare_exp001b_launch(
    *, project_root: str | Path, final_package_dir: str | Path,
    core_set_package_dir: str | Path, supplemental_set_package_dir: str | Path,
    model_config_path: str | Path, asset_manifest_path: str | Path,
    asset_root: str | Path, runner_evidence_path: str | Path,
    preflight_path: str | Path, authorization_path: str | Path,
) -> dict[str, Any]:
    persisted = _load_object(preflight_path, "persisted EXP-001B preflight")
    live = build_exp001b_run_preflight(
        project_root=project_root, final_package_dir=final_package_dir,
        core_set_package_dir=core_set_package_dir,
        supplemental_set_package_dir=supplemental_set_package_dir,
        model_config_path=model_config_path, asset_manifest_path=asset_manifest_path,
        asset_root=asset_root, runner_evidence_path=runner_evidence_path,
    )
    if not (
        persisted.get("preflight_digest_sha256") == live.get("preflight_digest_sha256")
        and persisted.get("run_plan_candidate") == live.get("run_plan_candidate")
        and persisted.get("status") == "preflight_valid_authorization_still_required"
    ):
        raise ValueError("persisted EXP-001B preflight does not match live sources and host")
    authorization = _load_object(authorization_path, "EXP-001B run authorization")
    report = verify_exp001b_run_authorization(authorization, preflight=live)
    if not report["valid"]:
        raise ValueError("EXP-001B run authorization is invalid: " + ", ".join(key for key, value in report["checks"].items() if not value))
    return {
        "launch_lock_version": "0.1", "valid": True,
        "experiment_id": EXPERIMENT_ID, "model_id": MODEL_ID,
        "preflight_digest_sha256": live["preflight_digest_sha256"],
        "authorization_file_sha256": sha256_file(authorization_path),
        "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
        "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
        "parent_core_set_package_digest_sha256": PARENT_CORE_SET_PACKAGE_DIGEST,
        "supplemental_set_digest_sha256": SUPPLEMENTAL_SET_DIGEST,
        "supplemental_set_package_digest_sha256": SUPPLEMENTAL_SET_PACKAGE_DIGEST,
        "expected_group_ids": live["run_plan_candidate"]["expected_group_ids"],
        "expected_group_count": EXPECTED_GROUP_COUNT,
        "expected_raw_record_count": EXPECTED_RAW_RECORD_COUNT,
    }


def _new_run_manifest(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_manifest_version": "0.1", "run_kind": "exp001b_supplemental_confirmatory",
        "experiment_id": EXPERIMENT_ID, "model_id": MODEL_ID,
        "preflight_digest_sha256": lock["preflight_digest_sha256"],
        "authorization_file_sha256": lock["authorization_file_sha256"],
        "final_preregistration_digest_sha256": FINAL_PREREGISTRATION_DIGEST,
        "parent_core_set_digest_sha256": PARENT_CORE_SET_DIGEST,
        "parent_core_set_package_digest_sha256": PARENT_CORE_SET_PACKAGE_DIGEST,
        "supplemental_set_digest_sha256": SUPPLEMENTAL_SET_DIGEST,
        "supplemental_set_package_digest_sha256": SUPPLEMENTAL_SET_PACKAGE_DIGEST,
        "expected_group_ids": list(lock["expected_group_ids"]),
        "expected_group_count": EXPECTED_GROUP_COUNT,
        "expected_raw_record_count": EXPECTED_RAW_RECORD_COUNT,
        "status": "initialized", "completed_group_files": {}, "completed_group_count": 0,
        "contains_derived_accuracy": False, "contains_interim_decision": False,
        "supplemental_experiment_authorized": True,
        "supplemental_experiment_run": True,
        "supplemental_results_observed": False,
    }


def _check_run_boundary(destination: Path, lock: Mapping[str, Any], resume: bool) -> dict[str, Any] | None:
    path = destination / "manifest.json"
    if not path.exists():
        if resume:
            raise ValueError("cannot resume before an EXP-001B run manifest exists")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("new EXP-001B output directory must be empty")
        return None
    manifest = _load_object(path, "EXP-001B run manifest")
    expected = _new_run_manifest(lock)
    immutable = (
        "run_kind", "experiment_id", "model_id", "preflight_digest_sha256",
        "authorization_file_sha256", "final_preregistration_digest_sha256",
        "parent_core_set_digest_sha256", "parent_core_set_package_digest_sha256",
        "supplemental_set_digest_sha256", "supplemental_set_package_digest_sha256",
        "expected_group_ids", "expected_group_count", "expected_raw_record_count",
        "contains_derived_accuracy", "contains_interim_decision",
        "supplemental_results_observed",
    )
    if any(manifest.get(key) != expected.get(key) for key in immutable):
        raise ValueError("existing EXP-001B manifest does not match launch lock")
    if manifest.get("status") == "supplemental_raw_complete":
        raise ValueError("completed EXP-001B supplemental run cannot be rerun")
    if not resume:
        raise ValueError("interrupted EXP-001B run requires explicit resume")
    return manifest


def run_locked_exp001b_groups(
    *, core_set: Mapping[str, Any], supplemental_set: Mapping[str, Any],
    backend: SupplementalRecordBackend, output_dir: str | Path,
    launch_lock: Mapping[str, Any], resume: bool = False,
) -> dict[str, Any]:
    if launch_lock.get("valid") is not True:
        raise ValueError("a valid EXP-001B launch lock is required")
    plans = build_supplemental_group_plan(supplemental_set, core_set)
    groups_by_id = {group["factorial_group_id"]: group for group in core_set["groups"]}
    plans_by_id = {plan["factorial_group_id"]: plan for plan in plans}
    destination = Path(output_dir).resolve()
    manifest = _check_run_boundary(destination, launch_lock, resume) or _new_run_manifest(launch_lock)
    completed = manifest.get("completed_group_files")
    if not isinstance(completed, dict):
        raise ValueError("completed_group_files must be an object")
    for group_id, digest in completed.items():
        path = destination / "groups" / f"{group_id}.json"
        if group_id not in plans_by_id or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"completed EXP-001B group integrity failed: {group_id}")
    manifest["status"] = "running"
    manifest.setdefault("started_at_utc", _utc_now())
    _atomic_write(destination / "manifest.json", manifest)
    try:
        for group_id in launch_lock["expected_group_ids"]:
            if group_id in completed:
                continue
            result = execute_supplemental_group(core_group=groups_by_id[group_id], plan=plans_by_id[group_id], backend=backend)
            path = destination / "groups" / f"{group_id}.json"
            _atomic_write(path, result)
            completed[group_id] = sha256_file(path)
            manifest["completed_group_files"] = dict(completed)
            manifest["completed_group_count"] = len(completed)
            _atomic_write(destination / "manifest.json", manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["failure"] = {"exception_type": type(exc).__name__, "message": str(exc), "failed_at_utc": _utc_now()}
        _atomic_write(destination / "manifest.json", manifest)
        raise
    manifest.pop("failure", None)
    manifest["status"] = "supplemental_raw_complete"
    manifest["completed_at_utc"] = _utc_now()
    manifest["raw_record_count"] = sum(plans_by_id[group_id]["record_count"] for group_id in completed)
    manifest["group_payload_digest_sha256"] = payload_digest(completed)
    manifest["valid"] = len(completed) == EXPECTED_GROUP_COUNT and manifest["raw_record_count"] == EXPECTED_RAW_RECORD_COUNT
    completion = {
        "completion_version": "0.1", "experiment_id": EXPERIMENT_ID,
        "status": manifest["status"], "preflight_digest_sha256": manifest["preflight_digest_sha256"],
        "supplemental_set_digest_sha256": SUPPLEMENTAL_SET_DIGEST,
        "completed_group_count": len(completed), "raw_record_count": manifest["raw_record_count"],
        "group_payload_digest_sha256": manifest["group_payload_digest_sha256"],
        "contains_derived_accuracy": False, "contains_interim_decision": False,
        "supplemental_experiment_run": True, "supplemental_results_observed": False,
        "route_decision": "verify_complete_supplemental_raw_package_before_analysis",
        "valid": manifest["valid"],
    }
    _atomic_write(destination / "completion.json", completion)
    _atomic_write(destination / "manifest.json", manifest)
    return completion


def run_exp001b_supplemental(
    *, project_root: str | Path, final_package_dir: str | Path,
    core_set_package_dir: str | Path, supplemental_set_package_dir: str | Path,
    model_config_path: str | Path, asset_manifest_path: str | Path,
    asset_root: str | Path, runner_evidence_path: str | Path,
    preflight_path: str | Path, authorization_path: str | Path,
    output_dir: str | Path, resume: bool = False,
    execution_lock: str | None = None,
) -> dict[str, Any]:
    if execution_lock != RUN_EXECUTION_LOCK:
        raise PermissionError("EXP-001B supplemental run execution lock is absent")
    lock = prepare_exp001b_launch(
        project_root=project_root, final_package_dir=final_package_dir,
        core_set_package_dir=core_set_package_dir, supplemental_set_package_dir=supplemental_set_package_dir,
        model_config_path=model_config_path, asset_manifest_path=asset_manifest_path,
        asset_root=asset_root, runner_evidence_path=runner_evidence_path,
        preflight_path=preflight_path, authorization_path=authorization_path,
    )
    destination = Path(output_dir).resolve()
    _check_run_boundary(destination, lock, resume)
    core_set = _load_object(Path(core_set_package_dir) / "core_set.json", "parent Core Set")
    supplemental_set = _load_object(Path(supplemental_set_package_dir) / "supplemental_set.json", "supplemental set")
    thresholds = _load_object(Path(final_package_dir) / "evidence/bdev1/state_norm_thresholds.json", "state norm thresholds")
    from psa.model import RWKV7Adapter, load_model_config
    from psa.supplemental.rwkv_run_backend import RWKVSupplementalRunBackend
    import time
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    config = load_model_config(model_config_path, project_root, verify_files=True)
    started = time.perf_counter()
    adapter = RWKV7Adapter.load(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    backend = RWKVSupplementalRunBackend(adapter=adapter, state_norm_thresholds=thresholds)
    completion = run_locked_exp001b_groups(core_set=core_set, supplemental_set=supplemental_set, backend=backend, output_dir=output_dir, launch_lock=lock, resume=resume)
    completion["model_load_seconds"] = load_seconds
    completion["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    _atomic_write(destination / "completion.json", completion)
    return completion


def _verify_group_payload(payload: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, bool]:
    records = payload.get("records")
    if not isinstance(records, list):
        return {"structure_valid": False}
    expected_identities = [
        (item["record_id"], item["record_kind"])
        for item in plan["records"]
    ]
    observed_identities = [
        (item.get("record_id"), item.get("record_kind"))
        for item in records
        if isinstance(item, Mapping)
    ]
    outputs_valid = True
    for item in records:
        try:
            _finite_scores(item.get("option_scores") if isinstance(item, Mapping) else None)
        except (TypeError, ValueError):
            outputs_valid = False
            break
        if item.get("record_kind") == "formal_generation_readout" and not (
            isinstance(item.get("generated_text"), str)
            and isinstance(item.get("generated_token_ids"), list)
            and isinstance(item.get("format_valid"), bool)
            and item.get("generated_choice") in {None, "A", "B", "C", "D"}
        ):
            outputs_valid = False
            break
    return {
        "group_identity_valid": payload.get("factorial_group_id") == plan.get("factorial_group_id"),
        "record_count_valid": payload.get("record_count") == plan.get("record_count") == len(records),
        "record_coverage_exact": Counter(observed_identities) == Counter(expected_identities),
        "record_outputs_valid": outputs_valid,
        "plan_digest_valid": payload.get("plan_digest_sha256") == plan.get("plan_digest_sha256"),
        "record_digest_valid": payload.get("group_result_digest_sha256") == sha256_json(records),
        "no_derived_accuracy": payload.get("contains_derived_accuracy") is False,
        "no_interim_decision": payload.get("contains_interim_decision") is False,
    }


def verify_exp001b_supplemental_raw_package(
    *, output_dir: str | Path, core_set_package_dir: str | Path,
    supplemental_set_package_dir: str | Path, preflight_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    manifest = _load_object(destination / "manifest.json", "EXP-001B run manifest")
    completion = _load_object(destination / "completion.json", "EXP-001B completion")
    preflight = _load_object(preflight_path, "EXP-001B preflight")
    authorization = _load_object(authorization_path, "EXP-001B run authorization")
    core_set = _load_object(Path(core_set_package_dir) / "core_set.json", "parent Core Set")
    supplemental_set = _load_object(Path(supplemental_set_package_dir) / "supplemental_set.json", "supplemental set")
    core_report = verify_core_set_package(core_set_package_dir)
    set_report = verify_exp001b_supplemental_set_package(supplemental_set_package_dir)
    auth_report = verify_exp001b_run_authorization(authorization, preflight=preflight)
    plans = build_supplemental_group_plan(supplemental_set, core_set)
    expected_ids = [item["factorial_group_id"] for item in plans]
    plan_by_id = {item["factorial_group_id"]: item for item in plans}
    group_dir = destination / "groups"
    actual_files = {item.name for item in group_dir.glob("*.json")} if group_dir.is_dir() else set()
    completed = manifest.get("completed_group_files") if isinstance(manifest.get("completed_group_files"), dict) else {}
    failed: list[str] = []
    digests: dict[str, str] = {}
    count = 0
    for group_id in expected_ids:
        path = group_dir / f"{group_id}.json"
        if not path.is_file():
            failed.append(group_id); continue
        digest = sha256_file(path); digests[group_id] = digest
        if completed.get(group_id) != digest:
            failed.append(group_id); continue
        payload = _load_object(path, f"EXP-001B group {group_id}")
        group_checks = _verify_group_payload(payload, plan_by_id[group_id])
        if not group_checks or not all(group_checks.values()):
            failed.append(group_id); continue
        count += int(payload["record_count"])
    computed = payload_digest(digests)
    checks = {
        "parent_core_package_valid": core_report.get("valid") is True,
        "supplemental_set_package_valid": set_report.get("valid") is True,
        "authorization_valid": auth_report.get("valid") is True,
        "experiment_identity_valid": manifest.get("experiment_id") == completion.get("experiment_id") == EXPERIMENT_ID,
        "model_identity_valid": manifest.get("model_id") == MODEL_ID,
        "preflight_digest_valid": manifest.get("preflight_digest_sha256") == completion.get("preflight_digest_sha256") == preflight.get("preflight_digest_sha256"),
        "authorization_file_digest_valid": manifest.get("authorization_file_sha256") == sha256_file(authorization_path),
        "frozen_digest_chain_valid": manifest.get("final_preregistration_digest_sha256") == FINAL_PREREGISTRATION_DIGEST and manifest.get("parent_core_set_digest_sha256") == PARENT_CORE_SET_DIGEST and manifest.get("parent_core_set_package_digest_sha256") == PARENT_CORE_SET_PACKAGE_DIGEST and manifest.get("supplemental_set_digest_sha256") == completion.get("supplemental_set_digest_sha256") == SUPPLEMENTAL_SET_DIGEST and manifest.get("supplemental_set_package_digest_sha256") == SUPPLEMENTAL_SET_PACKAGE_DIGEST,
        "run_status_complete": manifest.get("status") == completion.get("status") == "supplemental_raw_complete" and manifest.get("valid") is True and completion.get("valid") is True,
        "group_file_set_exact": actual_files == {f"{group_id}.json" for group_id in expected_ids},
        "completed_group_ledger_exact": set(completed) == set(expected_ids),
        "all_group_files_valid": not failed and len(digests) == EXPECTED_GROUP_COUNT,
        "raw_record_count_valid": count == manifest.get("raw_record_count") == completion.get("raw_record_count") == EXPECTED_RAW_RECORD_COUNT,
        "group_payload_digest_valid": computed == manifest.get("group_payload_digest_sha256") == completion.get("group_payload_digest_sha256"),
        "no_derived_accuracy": manifest.get("contains_derived_accuracy") is False and completion.get("contains_derived_accuracy") is False,
        "no_interim_decision": manifest.get("contains_interim_decision") is False and completion.get("contains_interim_decision") is False,
        "results_still_unobserved": manifest.get("supplemental_results_observed") is False and completion.get("supplemental_results_observed") is False,
    }
    valid = all(checks.values())
    return {
        "verification_version": "0.1", "created_at_utc": _utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "status": "raw_package_verified_unanalyzed" if valid else "raw_package_verification_failed",
        "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
        "failed_group_count": len(failed), "failed_group_ids": failed,
        "completed_group_count": manifest.get("completed_group_count"),
        "verified_record_count": count, "group_payload_digest_sha256": computed,
        "contains_derived_accuracy": False, "supplemental_experiment_run": True,
        "supplemental_results_observed": False,
        "route_decision": "begin_frozen_read_only_supplemental_analysis" if valid else "hold_without_analysis_and_repair_integrity",
        "valid": valid,
    }
