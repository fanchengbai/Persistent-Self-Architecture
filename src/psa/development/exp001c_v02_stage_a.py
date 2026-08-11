from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.development.exp001c_protocol_v02 import (
    verify_exp001c_protocol_v02_manifest,
)
from psa.development.prefix_instrumentation import (
    answer_boundary_evidence,
    instrument_forced_prefix,
)
from psa.environment import collect_environment
from psa.model import RWKV7Adapter, clone_state, load_model_config


EXPERIMENT_ID = "EXP-001C"
STAGE_A_RESULT_VERSION = "0.2-stage-a-development"
STAGE_A_EXECUTION_ENV = "PSA_EXP001C_V02_STAGE_A"
STAGE_A_EXECUTION_LOCK = "AUTHORIZED_EXP001C_V02_STAGE_A_POSITIVE_CONTROL"
STAGE_A_PREFLIGHT_VERSION = "0.1-development"
STAGE_A_AUTHORIZATION_TEXT = (
    "授权执行 EXP-001C v02 Stage A prompt-visible 非 Core 32 条 pilot，"
    "并授权观察本轮结果；不授权 Stage B、正式测试集、正式运行或自动重跑。"
)


class Exp001CV02StageABackend(Protocol):
    def run_stage_a(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _load_object(path: str | Path, label: str, *, root: Path | None = None) -> dict[str, Any]:
    candidate = _resolve(path, root) if root is not None else Path(path).resolve()
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _default_option_scorer(
    adapter: Any,
    *,
    logits: Any,
    state: Any,
    rendered_answers: Mapping[str, str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for code in "ABCD":
        answer_tokens = adapter.encode(rendered_answers[code])
        if not answer_tokens:
            raise ValueError(f"EXP-001C v02 answer {code} has no tokens")
        answer_logits = logits
        answer_state = clone_state(state)
        score = 0.0
        for index, token in enumerate(answer_tokens):
            log_probabilities = adapter.torch.log_softmax(
                answer_logits.float(),
                dim=-1,
            )
            score += float(log_probabilities[token].item())
            if index + 1 < len(answer_tokens):
                answer_logits, answer_state = adapter.forward(
                    [token],
                    answer_state,
                )
        scores[code] = score
    return scores


def build_exp001c_v02_stage_a_preflight(
    *,
    manifest_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
    environment_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only preflight; model weights are hashed but never loaded."""
    root = Path(project_root).resolve()
    manifest_file = _resolve(manifest_path, root)
    verification = verify_exp001c_protocol_v02_manifest(
        manifest_file,
        project_root=root,
    )
    if verification.get("valid") is not True:
        raise ValueError("EXP-001C v02 manifest verification failed")
    manifest = _load_object(manifest_file, "EXP-001C v02 manifest")
    model_entry = manifest.get("model_config")
    requested_model = _resolve(model_config_path, root)
    if not isinstance(model_entry, Mapping):
        raise ValueError("EXP-001C v02 manifest lacks a model config")
    locked_model = _resolve(str(model_entry.get("path", "")), root)
    model_lock_valid = bool(
        requested_model == locked_model
        and requested_model.is_file()
        and sha256_file(requested_model) == model_entry.get("sha256")
    )
    if not model_lock_valid:
        raise ValueError("model config does not match the locked EXP-001C v02 manifest")
    model = load_model_config(
        requested_model,
        project_root=root,
        verify_files=True,
    )
    report = (
        dict(environment_report)
        if environment_report is not None
        else collect_environment(root)
    )
    git = report.get("git")
    environment_valid = report.get("valid") is True
    git_clean = bool(
        isinstance(git, Mapping)
        and isinstance(git.get("commit"), str)
        and len(str(git.get("commit"))) == 40
        and git.get("dirty") is False
    )
    stable_plan = {
        "plan_version": STAGE_A_PREFLIGHT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_a_prompt_visible_only",
        "git_commit": git.get("commit") if isinstance(git, Mapping) else None,
        "manifest_digest_sha256": manifest["manifest_digest_sha256"],
        "model_config_path": str(model_entry["path"]),
        "model_config_sha256": str(model_entry["sha256"]),
        "model_id": model.model_id,
        "weights_sha256": model.weights_sha256,
        "weights_size_bytes": model.weights_size_bytes,
        "tokenizer_sha256": model.tokenizer_sha256,
        "tokenizer_size_bytes": model.tokenizer_size_bytes,
        "record_count": manifest["record_count"],
        "execution_command": "psa exp001c-v02-stage-a-run",
        "execution_environment_variable": STAGE_A_EXECUTION_ENV,
        "execution_lock_required_value": STAGE_A_EXECUTION_LOCK,
        "safety_rules": {
            "prompt_visible_only": True,
            "stage_b_recurrent_state_forbidden": True,
            "formal_test_set_access_forbidden": True,
            "formal_run_forbidden": True,
            "automatic_rerun_forbidden": True,
            "separate_result_observation_authorization_required": True,
        },
    }
    checks = {
        "manifest_valid": True,
        "manifest_still_unrun": manifest.get("model_executed") is False,
        "model_config_lock_valid": model_lock_valid,
        "model_assets_verified": True,
        "environment_valid": environment_valid,
        "git_clean": git_clean,
        "record_count_locked": manifest.get("record_count") == 32,
        "stage_b_still_forbidden": (
            manifest.get("stage_b_recurrent_state", {}).get("authorized")
            is False
        ),
        "formal_test_set_unaccessed": (
            manifest.get("formal_test_set_accessed") is False
        ),
    }
    valid = all(checks.values())
    preflight_digest = sha256_json(stable_plan)
    return {
        "preflight_version": STAGE_A_PREFLIGHT_VERSION,
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
        "stage_a_execution_authorized": False,
        "stage_a_result_observation_authorized": False,
        "stage_b_recurrent_state_authorized": False,
        "formal_test_set_accessed": False,
        "formal_run_authorized": False,
        "automatic_rerun_authorized": False,
        "checks": checks,
        "run_plan_candidate": stable_plan,
        "preflight_digest_sha256": preflight_digest,
        "authorization_boundary": {
            "new_project_owner_authorization_required": True,
            "authorization_must_bind_manifest_digest_sha256": manifest[
                "manifest_digest_sha256"
            ],
            "authorization_must_bind_preflight_digest_sha256": preflight_digest,
        },
    }


def verify_exp001c_v02_stage_a_preflight(
    *,
    preflight_path: str | Path,
    manifest_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
    environment_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    persisted = _load_object(
        preflight_path,
        "EXP-001C v02 Stage A preflight",
        root=root,
    )
    live = build_exp001c_v02_stage_a_preflight(
        manifest_path=manifest_path,
        model_config_path=model_config_path,
        project_root=root,
        environment_report=environment_report,
    )
    valid = bool(
        persisted.get("preflight_version") == STAGE_A_PREFLIGHT_VERSION
        and persisted.get("experiment_id") == EXPERIMENT_ID
        and persisted.get("valid") is True
        and persisted.get("status")
        == "preflight_valid_authorization_still_required"
        and persisted.get("development_only") is True
        and persisted.get("non_core") is True
        and persisted.get("model_assets_verified") is True
        and persisted.get("model_loaded") is False
        and persisted.get("model_executed") is False
        and persisted.get("stage_a_execution_authorized") is False
        and persisted.get("stage_a_result_observation_authorized") is False
        and persisted.get("stage_b_recurrent_state_authorized") is False
        and persisted.get("formal_test_set_accessed") is False
        and persisted.get("formal_run_authorized") is False
        and persisted.get("automatic_rerun_authorized") is False
        and persisted.get("preflight_digest_sha256")
        == live.get("preflight_digest_sha256")
        and persisted.get("run_plan_candidate")
        == live.get("run_plan_candidate")
        and persisted.get("checks") == live.get("checks")
        and persisted.get("authorization_boundary")
        == live.get("authorization_boundary")
    )
    return {
        "verification_version": STAGE_A_PREFLIGHT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preflight_verified" if valid else "invalid",
        "valid": valid,
        "preflight_digest_sha256": persisted.get("preflight_digest_sha256"),
        "model_loaded": False,
        "model_executed": False,
        "formal_test_set_accessed": False,
    }


def build_exp001c_v02_stage_a_authorization(
    *,
    manifest_path: str | Path,
    preflight_path: str | Path,
    model_config_path: str | Path,
    authorization_text: str,
    project_root: str | Path,
    environment_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the exact owner authorization without loading or running a model."""
    if authorization_text != STAGE_A_AUTHORIZATION_TEXT:
        raise PermissionError("EXP-001C v02 Stage A authorization text is not exact")
    root = Path(project_root).resolve()
    verification = verify_exp001c_v02_stage_a_preflight(
        preflight_path=preflight_path,
        manifest_path=manifest_path,
        model_config_path=model_config_path,
        project_root=root,
        environment_report=environment_report,
    )
    if verification.get("valid") is not True:
        raise PermissionError("EXP-001C v02 Stage A preflight is invalid")
    manifest = _load_object(
        manifest_path,
        "EXP-001C v02 manifest",
        root=root,
    )
    authorization = {
        "authorization_version": "0.1",
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_a_prompt_visible_only",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_digest_sha256": manifest["manifest_digest_sha256"],
        "preflight_digest_sha256": verification["preflight_digest_sha256"],
        "model_execution_authorized": True,
        "stage_a_result_observation_authorized": True,
        "stage_b_recurrent_state_authorized": False,
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "formal_result_observation_authorized": False,
        "automatic_rerun_authorized": False,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


class RWKVExp001CV02StageABackend:
    """Prompt-visible positive control; recurrent state is never constructed."""

    def __init__(
        self,
        *,
        adapter: Any,
        option_scorer: Callable[..., dict[str, float]] = _default_option_scorer,
    ) -> None:
        self.adapter = adapter
        self.option_scorer = option_scorer

    def _prewarm_shapes(self, prompts: list[str]) -> list[int]:
        lengths = sorted({len(self.adapter.encode(prompt)) for prompt in prompts})
        neutral_tokens = self.adapter.encode(" neutral")
        if not neutral_tokens:
            raise ValueError("EXP-001C v02 neutral warmup token is unavailable")
        for length in lengths:
            self.adapter.forward([neutral_tokens[0]] * length, None)
        return lengths

    def run_stage_a(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        trials = manifest.get("trials")
        if (
            manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("development_only") is not True
            or manifest.get("non_core") is not True
            or manifest.get("formal_test_set_accessed") is not False
            or manifest.get("model_executed") is not False
            or not isinstance(trials, list)
            or len(trials) != 32
            or not isinstance(manifest.get("stage_b_recurrent_state"), Mapping)
            or manifest["stage_b_recurrent_state"].get("authorized") is not False
        ):
            raise ValueError("EXP-001C v02 Stage A received an invalid manifest")

        forced_prefix = str(manifest.get("forced_answer_prefix", ""))
        prefix_tokens = self.adapter.encode(forced_prefix)
        if forced_prefix != ">\n" or len(prefix_tokens) != 2:
            raise ValueError("EXP-001C v02 forced prefix must be >\\n in two tokens")
        prompts = [
            str(trial["history_text"]) + str(trial["query_text"])
            for trial in trials
        ]
        warmup_shapes = self._prewarm_shapes(prompts)
        rendered_answers = {code: code for code in "ABCD"}
        records = []
        for trial, prompt_text in zip(trials, prompts):
            prompt_tokens = self.adapter.encode(prompt_text)
            logits, prompt_state = self.adapter.forward(prompt_tokens, None)
            prefix_evidence, answer_logits, answer_state = instrument_forced_prefix(
                self.adapter,
                logits=logits,
                state=prompt_state,
                prefix_token_ids=prefix_tokens,
                forced_prefix_text=forced_prefix,
                top_k=10,
            )
            scores = self.option_scorer(
                self.adapter,
                logits=answer_logits,
                state=answer_state,
                rendered_answers=rendered_answers,
            )
            target_code = str(trial["target_code"])
            records.append(
                {
                    "sample_id": str(trial["sample_id"]),
                    "semantic_case_id": str(trial["semantic_case_id"]),
                    "rotation_index": int(trial["rotation_index"]),
                    "target_code": target_code,
                    "prompt_token_count": len(prompt_tokens),
                    "prefix_evidence": prefix_evidence,
                    "option_log_probabilities": {
                        code: float(scores[code]) for code in "ABCD"
                    },
                    "answer_boundary_evidence": answer_boundary_evidence(
                        scores,
                        target_code=target_code,
                    ),
                }
            )

        return {
            "result_version": STAGE_A_RESULT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "v02_stage_a_prompt_visible_complete",
            "development_only": True,
            "non_core": True,
            "model_executed": True,
            "prompt_visible_only": True,
            "stage_b_recurrent_state_accessed": False,
            "formal_test_set_accessed": False,
            "formal_run": False,
            "contains_confirmatory_decision": False,
            "manifest_digest_sha256": manifest.get("manifest_digest_sha256"),
            "record_count": len(records),
            "warmup_token_lengths": warmup_shapes,
            "records": records,
        }


def validate_exp001c_v02_stage_a_authority(
    *,
    manifest_path: str | Path,
    authorization_path: str | Path,
    execution_lock: str,
    project_root: str | Path,
    preflight_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    # This guard intentionally precedes all path access and model construction.
    if execution_lock != STAGE_A_EXECUTION_LOCK:
        raise PermissionError("EXP-001C v02 Stage A execution lock is absent")
    root = Path(project_root).resolve()
    verification = verify_exp001c_protocol_v02_manifest(
        _resolve(manifest_path, root),
        project_root=root,
    )
    if verification.get("valid") is not True:
        raise ValueError("EXP-001C v02 manifest verification failed")
    manifest = _load_object(manifest_path, "EXP-001C v02 manifest", root=root)
    stage_a = manifest.get("stage_a_positive_control")
    if (
        manifest.get("execution_authorized") is not False
        or manifest.get("result_observation_authorized") is not False
        or not isinstance(stage_a, Mapping)
        or stage_a.get("authorized") is not False
        or stage_a.get("result_observation_authorized") is not False
        or stage_a.get("execution_requires_new_authorization") is not True
    ):
        raise ValueError("EXP-001C v02 Stage A manifest boundary is invalid")
    if preflight_path is None or model_config_path is None:
        raise PermissionError("EXP-001C v02 Stage A preflight is required")
    preflight_verification = verify_exp001c_v02_stage_a_preflight(
        preflight_path=preflight_path,
        manifest_path=manifest_path,
        model_config_path=model_config_path,
        project_root=root,
    )
    if preflight_verification.get("valid") is not True:
        raise PermissionError("EXP-001C v02 Stage A preflight is invalid")

    authorization = _load_object(
        authorization_path,
        "EXP-001C v02 Stage A authorization",
        root=root,
    )
    authorization_valid = bool(
        authorization.get("authorization_version") == "0.1"
        and authorization.get("experiment_id") == EXPERIMENT_ID
        and authorization.get("scope") == "v02_stage_a_prompt_visible_only"
        and authorization.get("authorized") is True
        and authorization.get("authorization_basis")
        == "project_owner_explicit_chat_authorization"
        and authorization.get("authorization_text")
        == STAGE_A_AUTHORIZATION_TEXT
        and isinstance(authorization.get("authorized_at_utc"), str)
        and len(str(authorization.get("authorized_at_utc"))) >= 20
        and authorization.get("manifest_digest_sha256")
        == manifest.get("manifest_digest_sha256")
        and authorization.get("preflight_digest_sha256")
        == preflight_verification.get("preflight_digest_sha256")
        and authorization.get("model_execution_authorized") is True
        and authorization.get("stage_a_result_observation_authorized") is True
        and authorization.get("stage_b_recurrent_state_authorized") is False
        and authorization.get("formal_test_set_access_authorized") is False
        and authorization.get("formal_run_authorized") is False
        and authorization.get("formal_result_observation_authorized") is False
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
    if not authorization_valid:
        raise PermissionError("EXP-001C v02 Stage A authorization is invalid")
    return {
        "valid": True,
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_a_prompt_visible_only",
        "manifest_digest_sha256": manifest["manifest_digest_sha256"],
        "preflight_digest_sha256": preflight_verification[
            "preflight_digest_sha256"
        ],
        "stage_b_recurrent_state_authorized": False,
        "formal_test_set_access_authorized": False,
        "formal_run_authorized": False,
        "automatic_rerun_authorized": False,
    }


def build_exp001c_v02_stage_a_backend(
    *,
    manifest_path: str | Path,
    model_config_path: str | Path,
    project_root: str | Path,
) -> RWKVExp001CV02StageABackend:
    """Load the real model only after the runner has validated all authority."""
    root = Path(project_root).resolve()
    manifest = _load_object(manifest_path, "EXP-001C v02 manifest", root=root)
    model_entry = manifest.get("model_config")
    requested = _resolve(model_config_path, root)
    if not isinstance(model_entry, Mapping):
        raise ValueError("EXP-001C v02 manifest lacks a model config")
    locked = _resolve(str(model_entry.get("path", "")), root)
    if (
        requested != locked
        or not requested.is_file()
        or sha256_file(requested) != model_entry.get("sha256")
    ):
        raise ValueError("model config does not match the locked EXP-001C v02 manifest")
    config = load_model_config(requested, project_root=root, verify_files=True)
    return RWKVExp001CV02StageABackend(adapter=RWKV7Adapter.load(config))


def run_exp001c_v02_stage_a(
    *,
    manifest_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    backend_factory: Callable[[], Exp001CV02StageABackend],
    execution_lock: str,
    project_root: str | Path,
) -> dict[str, Any]:
    authority = validate_exp001c_v02_stage_a_authority(
        manifest_path=manifest_path,
        authorization_path=authorization_path,
        execution_lock=execution_lock,
        project_root=project_root,
        preflight_path=preflight_path,
        model_config_path=model_config_path,
    )
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("EXP-001C v02 Stage A output directory must be empty")
    root = Path(project_root).resolve()
    manifest = _load_object(manifest_path, "EXP-001C v02 manifest", root=root)
    result = backend_factory().run_stage_a(manifest)
    records = result.get("records") if isinstance(result, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or result.get("development_only") is not True
        or result.get("non_core") is not True
        or result.get("model_executed") is not True
        or result.get("prompt_visible_only") is not True
        or result.get("stage_b_recurrent_state_accessed") is not False
        or result.get("formal_test_set_accessed") is not False
        or result.get("formal_run") is not False
        or result.get("contains_confirmatory_decision") is not False
        or result.get("manifest_digest_sha256")
        != authority["manifest_digest_sha256"]
        or result.get("record_count") != 32
        or not isinstance(records, list)
        or len(records) != 32
    ):
        raise ValueError("EXP-001C v02 Stage A result violates safety boundary")
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "stage_a_result.json"
    _atomic_write(result_path, result)
    summary = {
        "summary_version": STAGE_A_RESULT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "v02_stage_a_prompt_visible_complete",
        "valid": True,
        "development_only": True,
        "non_core": True,
        "model_executed": True,
        "prompt_visible_only": True,
        "manifest_digest_sha256": authority["manifest_digest_sha256"],
        "preflight_digest_sha256": authority["preflight_digest_sha256"],
        "stage_a_result_sha256": sha256_file(result_path),
        "record_count": 32,
        "stage_b_recurrent_state_accessed": False,
        "formal_test_set_accessed": False,
        "formal_run": False,
        "contains_confirmatory_decision": False,
        "automatic_rerun_authorized": False,
    }
    _atomic_write(destination / "summary.json", summary)
    return summary
