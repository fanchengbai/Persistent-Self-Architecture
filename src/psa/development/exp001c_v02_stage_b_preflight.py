from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)
from psa.development.exp001c_v02_stage_b_design import (
    EXPERIMENT_ID,
    verify_exp001c_v02_stage_b_design_manifest,
)
from psa.environment import collect_environment
from psa.model import load_model_config


STAGE_B_PREFLIGHT_VERSION = "0.1-development"
STAGE_B_AUTHORIZATION_TEXT = (
    "授权执行 EXP-001C v02 Stage B recurrent-state 非 Core 224 条 pilot，"
    "并授权观察本轮结果；不授权重跑 Stage A、正式测试集、正式运行、"
    "确认性决定或自动重跑。"
)


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _load_object(
    path: str | Path,
    label: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    candidate = _resolve(path, root) if root is not None else Path(path).resolve()
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _path_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def inspect_exp001c_v02_stage_a_artifacts(
    *,
    stage_a_result_path: str | Path,
    stage_a_summary_path: str | Path,
    design: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_path = _resolve(stage_a_result_path, root)
    summary_path = _resolve(stage_a_summary_path, root)
    result = _load_object(result_path, "EXP-001C v02 Stage A result")
    summary = _load_object(summary_path, "EXP-001C v02 Stage A summary")
    evidence = design.get("stage_a_pass_evidence")
    result_sha = sha256_file(result_path)
    summary_sha = sha256_file(summary_path)
    checks = {
        "evidence_present": isinstance(evidence, Mapping),
        "result_sha_matches_design": (
            isinstance(evidence, Mapping)
            and result_sha == evidence.get("stage_a_result_sha256")
        ),
        "result_scope_valid": bool(
            result.get("result_version") == "0.2-stage-a-development"
            and result.get("experiment_id") == EXPERIMENT_ID
            and result.get("status") == "v02_stage_a_prompt_visible_complete"
            and result.get("development_only") is True
            and result.get("non_core") is True
            and result.get("model_executed") is True
            and result.get("prompt_visible_only") is True
            and result.get("stage_b_recurrent_state_accessed") is False
            and result.get("formal_test_set_accessed") is False
            and result.get("formal_run") is False
            and result.get("contains_confirmatory_decision") is False
            and result.get("record_count") == 32
            and isinstance(result.get("records"), list)
            and len(result["records"]) == 32
            and isinstance(evidence, Mapping)
            and result.get("manifest_digest_sha256")
            == evidence.get("manifest_digest_sha256")
        ),
        "summary_scope_valid": bool(
            summary.get("experiment_id") == EXPERIMENT_ID
            and summary.get("status") == "v02_stage_a_prompt_visible_complete"
            and summary.get("valid") is True
            and summary.get("model_executed") is True
            and summary.get("prompt_visible_only") is True
            and summary.get("record_count") == 32
            and summary.get("stage_b_recurrent_state_accessed") is False
            and summary.get("formal_test_set_accessed") is False
            and summary.get("formal_run") is False
            and summary.get("contains_confirmatory_decision") is False
            and summary.get("automatic_rerun_authorized") is False
            and summary.get("stage_a_result_sha256") == result_sha
        ),
        "design_decision_is_pass": (
            isinstance(evidence, Mapping)
            and evidence.get("decision") == "stage_a_positive_control_pass"
            and evidence.get("label_marginalized_accuracy") == 0.875
        ),
    }
    return {
        "report_version": STAGE_B_PREFLIGHT_VERSION,
        "stage_a_result_path": _path_label(result_path, root),
        "stage_a_summary_path": _path_label(summary_path, root),
        "stage_a_result_sha256": result_sha,
        "stage_a_summary_sha256": summary_sha,
        "checks": checks,
        "valid": all(checks.values()),
        "model_loaded": False,
        "model_executed": False,
    }


def build_exp001c_v02_stage_b_preflight(
    *,
    design_manifest_path: str | Path,
    stage_a_result_path: str | Path,
    stage_a_summary_path: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    environment_report: Mapping[str, Any] | None = None,
    stage_a_artifact_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a live read-only preflight; model assets are hashed, never loaded."""
    root = Path(project_root).resolve()
    design_file = _resolve(design_manifest_path, root)
    design_verification = verify_exp001c_v02_stage_b_design_manifest(
        design_file,
        project_root=root,
    )
    if design_verification.get("valid") is not True:
        raise ValueError("Stage B design manifest verification failed")
    design = _load_object(design_file, "EXP-001C v02 Stage B design manifest")
    design_entry = design.get("design_config")
    if not isinstance(design_entry, Mapping):
        raise ValueError("Stage B design config entry is missing")
    design_config = _load_object(
        str(design_entry.get("path", "")),
        "EXP-001C v02 Stage B design config",
        root=root,
    )
    protocol = build_exp001c_protocol_v02_manifest(
        config_path=root / str(design_config.get("protocol_config_path", "")),
        project_root=root,
    )
    if (
        protocol.get("manifest_digest_sha256")
        != design.get("protocol_manifest_digest_sha256")
    ):
        raise ValueError("Stage B protocol manifest digest drifted")
    model_entry = protocol.get("model_config")
    requested_model = _resolve(model_config_path, root)
    if not isinstance(model_entry, Mapping):
        raise ValueError("Stage B protocol lacks a model config")
    locked_model = _resolve(str(model_entry.get("path", "")), root)
    model_lock_valid = bool(
        requested_model == locked_model
        and requested_model.is_file()
        and sha256_file(requested_model) == model_entry.get("sha256")
    )
    if not model_lock_valid:
        raise ValueError("model config does not match the locked Stage B design")
    model = load_model_config(
        requested_model,
        project_root=root,
        verify_files=True,
    )
    environment = (
        dict(environment_report)
        if environment_report is not None
        else collect_environment(root)
    )
    git = environment.get("git")
    git_clean = bool(
        isinstance(git, Mapping)
        and isinstance(git.get("commit"), str)
        and len(str(git.get("commit"))) == 40
        and git.get("dirty") is False
    )
    stage_a_report = (
        dict(stage_a_artifact_report)
        if stage_a_artifact_report is not None
        else inspect_exp001c_v02_stage_a_artifacts(
            stage_a_result_path=stage_a_result_path,
            stage_a_summary_path=stage_a_summary_path,
            design=design,
            project_root=root,
        )
    )
    destination = _resolve(output_dir, root)
    output_empty = not destination.exists() or not any(destination.iterdir())
    stable_plan = {
        "plan_version": STAGE_B_PREFLIGHT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_b_recurrent_state_noncore_pilot_once",
        "git_commit": git.get("commit") if isinstance(git, Mapping) else None,
        "git_branch": git.get("branch") if isinstance(git, Mapping) else None,
        "design_manifest_path": _path_label(design_file, root),
        "design_manifest_digest_sha256": design[
            "design_manifest_digest_sha256"
        ],
        "protocol_manifest_digest_sha256": design[
            "protocol_manifest_digest_sha256"
        ],
        "stage_a_result_path": stage_a_report.get("stage_a_result_path"),
        "stage_a_summary_path": stage_a_report.get("stage_a_summary_path"),
        "stage_a_result_sha256": stage_a_report.get("stage_a_result_sha256"),
        "stage_a_summary_sha256": stage_a_report.get("stage_a_summary_sha256"),
        "model_config_path": str(model_entry["path"]),
        "model_config_sha256": str(model_entry["sha256"]),
        "model_id": model.model_id,
        "weights_sha256": model.weights_sha256,
        "weights_size_bytes": model.weights_size_bytes,
        "tokenizer_sha256": model.tokenizer_sha256,
        "tokenizer_size_bytes": model.tokenizer_size_bytes,
        "output_dir": _path_label(destination, root),
        "record_count": 224,
        "condition_count": 7,
        "execution_command": "psa exp001c-v02-stage-b-run",
        "execution_lock_required_value": (
            "AUTHORIZED_EXP001C_V02_STAGE_B_NONCORE_ONCE"
        ),
        "safety_rules": {
            "stage_a_rerun_forbidden": True,
            "formal_test_set_access_forbidden": True,
            "formal_run_forbidden": True,
            "confirmatory_decision_forbidden": True,
            "automatic_rerun_forbidden": True,
            "single_use_execution_claim_required": True,
        },
    }
    checks = {
        "design_manifest_valid": True,
        "design_execution_still_unauthorized": (
            design.get("execution_authorized") is False
        ),
        "protocol_manifest_bound": True,
        "stage_a_artifacts_valid": stage_a_report.get("valid") is True,
        "stage_a_result_digest_bound": (
            stage_a_report.get("stage_a_result_sha256")
            == design.get("stage_a_pass_evidence", {}).get(
                "stage_a_result_sha256"
            )
        ),
        "model_config_lock_valid": model_lock_valid,
        "model_assets_verified": True,
        "environment_valid": environment.get("valid") is True,
        "git_clean": git_clean,
        "git_branch_main": (
            isinstance(git, Mapping) and git.get("branch") == "main"
        ),
        "output_directory_empty": output_empty,
        "record_count_locked": design.get("record_count") == 224,
        "stage_a_rerun_forbidden": design.get("stage_a_rerun_included") is False,
        "formal_test_set_unaccessed": (
            design.get("formal_test_set_accessed") is False
        ),
    }
    valid = all(checks.values())
    preflight_digest = sha256_json(stable_plan)
    return {
        "preflight_version": STAGE_B_PREFLIGHT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "preflight_valid_authorization_still_required"
            if valid
            else "preflight_failed"
        ),
        "valid": valid,
        "development_only": True,
        "non_core": True,
        "model_assets_verified": True,
        "model_loaded": False,
        "model_executed": False,
        "stage_b_execution_authorized": False,
        "stage_b_result_observation_authorized": False,
        "stage_a_rerun_authorized": False,
        "formal_test_set_accessed": False,
        "formal_run_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "checks": checks,
        "stage_a_artifact_report": stage_a_report,
        "run_plan_candidate": stable_plan,
        "preflight_digest_sha256": preflight_digest,
        "authorization_boundary": {
            "new_project_owner_authorization_required": True,
            "authorization_must_bind_design_manifest_digest_sha256": design[
                "design_manifest_digest_sha256"
            ],
            "authorization_must_bind_preflight_digest_sha256": preflight_digest,
            "required_authorization_text": STAGE_B_AUTHORIZATION_TEXT,
        },
    }


def verify_exp001c_v02_stage_b_preflight(
    *,
    preflight_path: str | Path,
    design_manifest_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
    environment_report: Mapping[str, Any] | None = None,
    stage_a_artifact_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    persisted = _load_object(preflight_path, "Stage B preflight", root=root)
    plan = persisted.get("run_plan_candidate")
    if not isinstance(plan, Mapping):
        return {"valid": False, "status": "invalid"}
    live = build_exp001c_v02_stage_b_preflight(
        design_manifest_path=design_manifest_path,
        stage_a_result_path=str(plan.get("stage_a_result_path", "")),
        stage_a_summary_path=str(plan.get("stage_a_summary_path", "")),
        model_config_path=model_config_path,
        output_dir=str(plan.get("output_dir", "")),
        project_root=root,
        environment_report=environment_report,
        stage_a_artifact_report=stage_a_artifact_report,
    )
    valid = bool(
        persisted.get("preflight_version") == STAGE_B_PREFLIGHT_VERSION
        and persisted.get("experiment_id") == EXPERIMENT_ID
        and persisted.get("valid") is True
        and persisted.get("status")
        == "preflight_valid_authorization_still_required"
        and persisted.get("model_loaded") is False
        and persisted.get("model_executed") is False
        and persisted.get("stage_b_execution_authorized") is False
        and persisted.get("stage_b_result_observation_authorized") is False
        and persisted.get("stage_a_rerun_authorized") is False
        and persisted.get("formal_test_set_accessed") is False
        and persisted.get("formal_run_authorized") is False
        and persisted.get("confirmatory_decision_authorized") is False
        and persisted.get("automatic_rerun_authorized") is False
        and persisted.get("preflight_digest_sha256")
        == live.get("preflight_digest_sha256")
        and persisted.get("run_plan_candidate") == live.get("run_plan_candidate")
        and persisted.get("checks") == live.get("checks")
        and persisted.get("stage_a_artifact_report")
        == live.get("stage_a_artifact_report")
        and persisted.get("authorization_boundary")
        == live.get("authorization_boundary")
    )
    return {
        "verification_version": STAGE_B_PREFLIGHT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preflight_verified" if valid else "invalid",
        "valid": valid,
        "preflight_digest_sha256": persisted.get("preflight_digest_sha256"),
        "design_manifest_digest_sha256": plan.get(
            "design_manifest_digest_sha256"
        ),
        "stage_a_result_sha256": plan.get("stage_a_result_sha256"),
        "model_loaded": False,
        "model_executed": False,
        "formal_test_set_accessed": False,
    }


def build_exp001c_v02_stage_b_authorization(
    *,
    design_manifest_path: str | Path,
    preflight_path: str | Path,
    model_config_path: str | Path,
    authorization_text: str,
    project_root: str | Path,
    environment_report: Mapping[str, Any] | None = None,
    stage_a_artifact_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if authorization_text != STAGE_B_AUTHORIZATION_TEXT:
        raise PermissionError("EXP-001C v02 Stage B authorization text is not exact")
    root = Path(project_root).resolve()
    verification = verify_exp001c_v02_stage_b_preflight(
        preflight_path=preflight_path,
        design_manifest_path=design_manifest_path,
        model_config_path=model_config_path,
        project_root=root,
        environment_report=environment_report,
        stage_a_artifact_report=stage_a_artifact_report,
    )
    if verification.get("valid") is not True:
        raise PermissionError("EXP-001C v02 Stage B preflight is invalid")
    authorization = {
        "authorization_version": "0.1",
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_b_recurrent_state_noncore_pilot_once",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_manifest_digest_sha256": verification[
            "design_manifest_digest_sha256"
        ],
        "preflight_digest_sha256": verification["preflight_digest_sha256"],
        "stage_a_result_sha256": verification["stage_a_result_sha256"],
        "model_execution_authorized": True,
        "stage_b_result_observation_authorized": True,
        "stage_a_rerun_authorized": False,
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_exp001c_v02_stage_b_machine_authority(
    *,
    design_manifest_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        verification = verify_exp001c_v02_stage_b_preflight(
            preflight_path=preflight_path,
            design_manifest_path=design_manifest_path,
            model_config_path=model_config_path,
            project_root=root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PermissionError(
            "EXP-001C v02 Stage B preflight is missing or invalid"
        ) from exc
    if verification.get("valid") is not True:
        raise PermissionError("EXP-001C v02 Stage B preflight is invalid")
    authorization = _load_object(
        authorization_path,
        "EXP-001C v02 Stage B authorization",
        root=root,
    )
    valid = bool(
        authorization.get("authorization_version") == "0.1"
        and authorization.get("experiment_id") == EXPERIMENT_ID
        and authorization.get("scope")
        == "v02_stage_b_recurrent_state_noncore_pilot_once"
        and authorization.get("authorized") is True
        and authorization.get("authorization_basis")
        == "project_owner_explicit_chat_authorization"
        and authorization.get("authorization_text") == STAGE_B_AUTHORIZATION_TEXT
        and isinstance(authorization.get("authorized_at_utc"), str)
        and authorization.get("design_manifest_digest_sha256")
        == verification.get("design_manifest_digest_sha256")
        and authorization.get("preflight_digest_sha256")
        == verification.get("preflight_digest_sha256")
        and authorization.get("stage_a_result_sha256")
        == verification.get("stage_a_result_sha256")
        and authorization.get("model_execution_authorized") is True
        and authorization.get("stage_b_result_observation_authorized") is True
        and authorization.get("stage_a_rerun_authorized") is False
        and authorization.get("formal_test_set_access_authorized") is False
        and authorization.get("formal_run_authorized") is False
        and authorization.get("confirmatory_decision_authorized") is False
        and authorization.get("automatic_rerun_authorized") is False
        and authorization.get("authorization_digest_sha256")
        == sha256_json(
            {
                key: value
                for key, value in authorization.items()
                if key != "authorization_digest_sha256"
            }
        )
    )
    if not valid:
        raise PermissionError("EXP-001C v02 Stage B authorization is invalid")
    return {
        "valid": True,
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_b_recurrent_state_noncore_pilot_once",
        "design_manifest_digest_sha256": verification[
            "design_manifest_digest_sha256"
        ],
        "preflight_digest_sha256": verification["preflight_digest_sha256"],
        "stage_a_result_sha256": verification["stage_a_result_sha256"],
        "model_execution_authorized": True,
        "stage_b_result_observation_authorized": True,
        "stage_a_rerun_authorized": False,
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
    }
