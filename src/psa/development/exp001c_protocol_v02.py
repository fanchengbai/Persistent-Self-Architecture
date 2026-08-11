from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.development.history_binding import generate_history_binding_manifest


PROTOCOL_VERSION = "0.2-development-draft"
MANIFEST_VERSION = "0.2-development-unrun"
PROTOCOL_SOURCE_FILES = (
    "configs/development/exp001c_noncore_protocol_v02.draft.json",
    "configs/models/rwkv7_g1h_2.9b.candidate.json",
    "docs/exp001c_noncore_pilot_v01_observation.md",
    "schemas/exp001c_protocol_v02_manifest.schema.json",
    "schemas/exp001c_v02_stage_a_authorization.schema.json",
    "schemas/exp001c_v02_stage_a_preflight.schema.json",
    "schemas/exp001c_v02_stage_a_result.schema.json",
    "src/psa/artifacts/integrity.py",
    "src/psa/cli.py",
    "src/psa/development/__init__.py",
    "src/psa/development/exp001c_protocol_v02.py",
    "src/psa/development/exp001c_v02_stage_a.py",
    "src/psa/development/history_binding.py",
    "src/psa/tasks/identity_goal.py",
    "tests/test_exp001c_protocol_v02.py",
    "tests/test_exp001c_v02_stage_a.py",
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "manifest_digest_sha256"
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


def _validate_config(config: Mapping[str, Any]) -> None:
    authority = config.get("authority")
    prompt = config.get("prompt_protocol")
    stage_a = config.get("stage_a_positive_control")
    stage_b = config.get("stage_b_recurrent_state")
    if (
        config.get("protocol_version") != PROTOCOL_VERSION
        or config.get("experiment_id") != "EXP-001C"
        or config.get("development_only") is not True
        or config.get("non_core") is not True
        or config.get("formal_test_set_accessed") is not False
        or not isinstance(authority, Mapping)
        or authority.get("offline_manifest_build_authorized") is not True
        or authority.get("offline_stage_a_runner_implementation_authorized")
        is not True
        or authority.get("model_execution_authorized") is not False
        or authority.get("automatic_rerun_authorized") is not False
        or authority.get("formal_test_set_access_authorized") is not False
        or authority.get("formal_run_authorized") is not False
    ):
        raise PermissionError("EXP-001C v02 authorizes offline manifest design only")
    if (
        not isinstance(prompt, Mapping)
        or prompt.get("source") != "existing_g1_history_binding_generator"
        or prompt.get("prompt_format")
        != "rwkv7-g1-state-only-fake-think-v0.1"
        or prompt.get("history_mode") != "single_statement"
        or prompt.get("assistant_prefix") != "<think></think"
        or prompt.get("forced_answer_prefix") != ">\n"
        or prompt.get("answer_codes") != list("ABCD")
        or prompt.get("rotation_count") != 4
    ):
        raise ValueError("EXP-001C v02 prompt protocol is not locked")
    if (
        not isinstance(stage_a, Mapping)
        or stage_a.get("scope") != "prompt_visible_only"
        or stage_a.get("authorized") is not False
        or stage_a.get("result_observation_authorized") is not False
        or stage_a.get("record_count") != 32
        or stage_a.get("minimum_label_marginalized_accuracy") != 0.8
        or stage_a.get("require_complete_four_code_rotation") is not True
        or stage_a.get("require_all_answer_codes_equally_represented") is not True
        or stage_a.get("maximum_single_predicted_code_share") != 0.5
        or stage_a.get("execution_requires_new_authorization") is not True
        or not isinstance(stage_b, Mapping)
        or stage_b.get("authorized") is not False
        or stage_b.get("requires_stage_a_pass") is not True
        or stage_b.get("requires_separate_owner_authorization") is not True
    ):
        raise ValueError("EXP-001C v02 staged safety gates are not locked")


def build_exp001c_protocol_v02_manifest(
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    config = _load_object(path, "EXP-001C v02 protocol config")
    _validate_config(config)
    prompt = config["prompt_protocol"]
    generated = generate_history_binding_manifest(
        answer_codes=prompt["answer_codes"],
        identity_label_pairs=prompt["identity_label_pairs"],
        goal_label_pairs=prompt["goal_label_pairs"],
        history_modes=[prompt["history_mode"]],
        repetitions=int(prompt["repetitions"]),
        base_seed=int(prompt["base_seed"]),
        delay_units=int(prompt["delay_units"]),
        assistant_prefix=str(prompt["assistant_prefix"]),
    )
    trials = generated["trials"]
    target_counts = {
        code: sum(trial["target_code"] == code for trial in trials)
        for code in "ABCD"
    }
    rotation_counts = {
        str(rotation): sum(
            trial["rotation_index"] == rotation for trial in trials
        )
        for rotation in range(4)
    }
    if (
        len(trials) != 32
        or set(target_counts.values()) != {8}
        or set(rotation_counts.values()) != {8}
    ):
        raise ValueError("EXP-001C v02 fixture is not fully code-balanced")
    source_digests = {}
    for relative in PROTOCOL_SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise ValueError(f"EXP-001C v02 source file is missing: {relative}")
        source_digests[relative] = sha256_file(source)
    config_relative = _relative(path, root)
    if source_digests.get(config_relative) != sha256_file(path):
        raise ValueError("EXP-001C v02 config is absent from source inventory")
    model_path = (root / str(config.get("model_config_path", ""))).resolve()
    model_relative = _relative(model_path, root)
    if (
        not model_path.is_file()
        or source_digests.get(model_relative) != sha256_file(model_path)
    ):
        raise ValueError("EXP-001C v02 model config is not source-locked")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "protocol_id": config["protocol_id"],
        "experiment_id": "EXP-001C",
        "status": "offline_positive_control_manifest_unrun",
        "development_only": True,
        "non_core": True,
        "formal_test_set_accessed": False,
        "model_executed": False,
        "execution_authorized": False,
        "result_observation_authorized": False,
        "automatic_rerun_authorized": False,
        "formal_run_authorized": False,
        "config": {
            "path": config_relative,
            "sha256": sha256_file(path),
        },
        "model_config": {
            "path": model_relative,
            "sha256": sha256_file(model_path),
        },
        "source_pilot": config["source_pilot"],
        "stage_a_positive_control": config["stage_a_positive_control"],
        "stage_b_recurrent_state": config["stage_b_recurrent_state"],
        "prompt_format": generated["prompt_format"],
        "assistant_prefix": generated["assistant_prefix"],
        "forced_answer_prefix": prompt["forced_answer_prefix"],
        "history_mode": prompt["history_mode"],
        "rotation_count": generated["rotation_count"],
        "semantic_case_count": generated["semantic_case_count_per_mode"],
        "record_count": len(trials),
        "target_code_counts": target_counts,
        "rotation_counts": rotation_counts,
        "trials": trials,
        "locked_source_digests": dict(sorted(source_digests.items())),
    }
    manifest["manifest_digest_sha256"] = sha256_json(
        _manifest_payload(manifest)
    )
    return manifest


def verify_exp001c_protocol_v02_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = _load_object(manifest_path, "EXP-001C v02 manifest")
    stored_sources = manifest.get("locked_source_digests")
    source_checks = {}
    for relative in PROTOCOL_SOURCE_FILES:
        expected = (
            stored_sources.get(relative)
            if isinstance(stored_sources, Mapping)
            else None
        )
        source = root / relative
        source_checks[relative] = bool(
            source.is_file()
            and isinstance(expected, str)
            and sha256_file(source) == expected
        )
    inventory_complete = bool(
        isinstance(stored_sources, Mapping)
        and set(stored_sources) == set(PROTOCOL_SOURCE_FILES)
    )
    trials = manifest.get("trials")
    target_counts = manifest.get("target_code_counts")
    rotation_counts = manifest.get("rotation_counts")
    boundary_valid = bool(
        manifest.get("manifest_version") == MANIFEST_VERSION
        and manifest.get("experiment_id") == "EXP-001C"
        and manifest.get("status") == "offline_positive_control_manifest_unrun"
        and manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("model_executed") is False
        and manifest.get("execution_authorized") is False
        and manifest.get("result_observation_authorized") is False
        and manifest.get("automatic_rerun_authorized") is False
        and manifest.get("formal_run_authorized") is False
        and isinstance(manifest.get("stage_b_recurrent_state"), Mapping)
        and isinstance(manifest.get("stage_a_positive_control"), Mapping)
        and manifest["stage_a_positive_control"].get("authorized") is False
        and manifest["stage_a_positive_control"].get(
            "result_observation_authorized"
        )
        is False
        and manifest["stage_b_recurrent_state"].get("authorized") is False
    )
    balance_valid = bool(
        isinstance(trials, list)
        and len(trials) == 32
        and manifest.get("record_count") == 32
        and target_counts == {code: 8 for code in "ABCD"}
        and rotation_counts == {str(index): 8 for index in range(4)}
        and all(
            trial.get("history_mode") == "single_statement"
            and trial.get("task_level") == "state_only_history_binding"
            for trial in trials
        )
    )
    digest_valid = manifest.get("manifest_digest_sha256") == sha256_json(
        _manifest_payload(manifest)
    )
    config_entry = manifest.get("config")
    deterministic_payload_valid = False
    if isinstance(config_entry, Mapping):
        config_path = root / str(config_entry.get("path", ""))
        try:
            expected_manifest = build_exp001c_protocol_v02_manifest(
                config_path=config_path,
                project_root=root,
            )
        except (OSError, TypeError, ValueError, PermissionError):
            expected_manifest = None
        deterministic_payload_valid = manifest == expected_manifest
    valid = bool(
        boundary_valid
        and balance_valid
        and digest_valid
        and deterministic_payload_valid
        and inventory_complete
        and all(source_checks.values())
    )
    return {
        "verification_version": MANIFEST_VERSION,
        "experiment_id": "EXP-001C",
        "status": "offline_positive_control_manifest_verified" if valid else "invalid",
        "valid": valid,
        "safety_boundary_valid": boundary_valid,
        "code_rotation_balance_valid": balance_valid,
        "manifest_digest_valid": digest_valid,
        "deterministic_payload_valid": deterministic_payload_valid,
        "source_inventory_complete": inventory_complete,
        "source_checks": source_checks,
        "model_executed": False,
        "formal_test_set_accessed": False,
    }
