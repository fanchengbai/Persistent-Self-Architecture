from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.assets import load_manifest, verify_manifest
from psa.environment import collect_environment
from psa.preregistration import (
    verify_core_set_package,
    verify_final_preregistration_package,
)


EXPECTED_EXPERIMENT_ID = "EXP-001"
EXPECTED_CANDIDATE_DIGEST = (
    "a354b208be0640da7ea70fe070f75bdec69186e496ba1cc14c3157dcd984e6cd"
)
EXPECTED_FINAL_DIGEST = (
    "0daf056dc6b38aa20fa69dd9e8df9b8065876529947cbc01353ffe604933d0c9"
)
EXPECTED_CORE_SET_DIGEST = (
    "6ea2b6be15a7728c96d84dcc8e48da64e740438980f818e78c8ee8570a47eb9d"
)
EXPECTED_CORE_PACKAGE_DIGEST = (
    "9659e286de4128b43226f2d6df27075eba60bd953c2330ee70c0ec3e677f1642"
)
EXPECTED_MODEL_ID = "rwkv7-g1h-2.9b-20260710"
EXPECTED_WEIGHT_DIGEST = (
    "295595b3b8dbff3f8c2a0585975622ddaba4feea7a377022f0bd75347c90c9b3"
)
EXPECTED_TOKENIZER_DIGEST = (
    "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
)
EXPECTED_CONDITIONS = (
    "continuous",
    "restored",
    "reset",
    "random_matched",
    "swapped_I",
    "swapped_G",
    "swapped_both",
    "prompt_visible",
)
MINIMUM_GPU_MEMORY_MIB = 8 * 1024
MINIMUM_FREE_DISK_BYTES = 20 * 1024**3

FROZEN_SCORING_SOURCES = (
    "configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json",
    "configs/models/rwkv7_g1h_2.9b.candidate.json",
    "src/psa/model/rwkv7.py",
    "src/psa/development/impl3.py",
    "src/psa/development/history_binding.py",
    "src/psa/evaluation/resampling.py",
)
RUNNER_SOURCES = (
    ".gitignore",
    "src/psa/confirmatory/preflight.py",
    "src/psa/confirmatory/runner.py",
    "src/psa/confirmatory/rwkv_backend.py",
    "src/psa/confirmatory/development.py",
    "src/psa/confirmatory/formal.py",
    "src/psa/cli.py",
    "scripts/preflight_exp001_confirmatory_run.sh",
    "scripts/run_impl5b_confirmatory_runner_development_gate.sh",
    "scripts/run_exp001_confirmatory.sh",
    "schemas/exp001_confirmatory_run_authorization.schema.json",
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


def _environment_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    git = report.get("git")
    disk = report.get("disk")
    runtime = report.get("runtime_environment")
    nvidia = report.get("nvidia_smi")
    gpus = nvidia.get("gpus", []) if isinstance(nvidia, Mapping) else []
    return {
        "environment_report_valid": report.get("valid") is True,
        "git_commit_recorded": bool(
            isinstance(git, Mapping) and git.get("commit")
        ),
        "git_worktree_clean": bool(
            isinstance(git, Mapping) and git.get("dirty") is False
        ),
        "runtime_flags_frozen": bool(
            isinstance(runtime, Mapping)
            and runtime.get("RWKV_V7_ON") == "1"
            and runtime.get("RWKV_JIT_ON") == "0"
            and runtime.get("RWKV_CUDA_ON") == "0"
        ),
        "gpu_memory_sufficient": bool(
            gpus
            and all(
                isinstance(gpu, Mapping)
                and isinstance(gpu.get("memory_mib"), int)
                and gpu["memory_mib"] >= MINIMUM_GPU_MEMORY_MIB
                for gpu in gpus
            )
        ),
        "disk_space_sufficient": bool(
            isinstance(disk, Mapping)
            and isinstance(disk.get("free_bytes"), int)
            and disk["free_bytes"] >= MINIMUM_FREE_DISK_BYTES
        ),
    }


def _asset_model_consistency(
    model_config: Mapping[str, Any],
    asset_report: Mapping[str, Any],
) -> dict[str, bool]:
    assets = asset_report.get("assets")
    by_id = {
        item.get("id"): item
        for item in assets
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    } if isinstance(assets, list) else {}
    model_asset = by_id.get("rwkv7-g1h-2.9b-20260710", {})
    tokenizer_asset = by_id.get("rwkv-world-tokenizer-20230424", {})
    weights = model_config.get("weights")
    tokenizer = model_config.get("tokenizer")
    return {
        "asset_bundle_valid": asset_report.get("valid") is True,
        "model_asset_matches_frozen_config": bool(
            isinstance(weights, Mapping)
            and model_asset.get("status") == "valid"
            and model_asset.get("sha256") == weights.get("sha256")
            == EXPECTED_WEIGHT_DIGEST
            and model_asset.get("size_bytes") == weights.get("size_bytes")
        ),
        "tokenizer_asset_matches_frozen_config": bool(
            isinstance(tokenizer, Mapping)
            and tokenizer_asset.get("status") == "valid"
            and tokenizer_asset.get("sha256") == tokenizer.get("sha256")
            == EXPECTED_TOKENIZER_DIGEST
            and tokenizer_asset.get("size_bytes") == tokenizer.get("size_bytes")
        ),
    }


def build_confirmatory_preflight(
    *,
    project_root: str | Path,
    final_package_dir: str | Path,
    core_set_package_dir: str | Path,
    model_config_path: str | Path,
    asset_manifest_path: str | Path,
    asset_root: str | Path,
    runner_evidence_path: str | Path | None = None,
    environment_report: Mapping[str, Any] | None = None,
    asset_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-inference preflight report for the still-locked Core Set."""
    root = Path(project_root).resolve()
    final_root = Path(final_package_dir).resolve()
    core_root = Path(core_set_package_dir).resolve()
    model_path = Path(model_config_path).resolve()
    asset_path = Path(asset_manifest_path).resolve()

    final_report = verify_final_preregistration_package(final_root)
    core_report = verify_core_set_package(core_root)
    if not final_report["valid"]:
        raise ValueError("final preregistration package is invalid")
    if not core_report["valid"]:
        raise ValueError("frozen Core Set package is invalid")

    final_manifest = _load_object(final_root / "manifest.json", "final manifest")
    candidate = _load_object(final_root / "candidate.json", "candidate")
    core_manifest = _load_object(core_root / "manifest.json", "Core Set manifest")
    core_set = _load_object(core_root / "core_set.json", "Core Set")
    model_config = _load_object(model_path, "model config")
    runner_evidence = None
    runner_evidence_source = None
    if runner_evidence_path is not None:
        runner_evidence_source = Path(runner_evidence_path).resolve()
        runner_evidence = _load_object(
            runner_evidence_source,
            "runner development evidence",
        )

    if environment_report is None:
        environment_report = collect_environment(root)
    if asset_report is None:
        asset_manifest = load_manifest(asset_path)
        asset_report = verify_manifest(asset_manifest, root=asset_root)

    frozen_digests = candidate.get("source_file_digests")
    if not isinstance(frozen_digests, Mapping):
        raise ValueError("candidate source_file_digests are missing")
    source_checks: dict[str, dict[str, Any]] = {}
    for relative in FROZEN_SCORING_SOURCES:
        expected = frozen_digests.get(relative)
        source = root / relative
        actual = sha256_file(source) if source.is_file() else None
        source_checks[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": bool(isinstance(expected, str) and actual == expected),
        }
    runner_source_digests = {
        relative: sha256_file(root / relative)
        for relative in RUNNER_SOURCES
        if (root / relative).is_file()
    }
    runner_sources_complete = set(runner_source_digests) == set(RUNNER_SOURCES)
    runner_evidence_valid = bool(
        isinstance(runner_evidence, Mapping)
        and runner_evidence.get("valid") is True
        and runner_evidence.get("gate")
        == "impl5b_confirmatory_runner_development"
        and runner_evidence.get("development_only") is True
        and runner_evidence.get("fixture_kind")
        == "non_core_confirmatory_runner_fixture"
        and runner_evidence.get("group_count") == 1
        and runner_evidence.get("trial_count") == 16
        and runner_evidence.get("condition_count") == 8
        and runner_evidence.get("raw_record_count") == 128
        and runner_evidence.get("runner_source_digests")
        == {
            relative: runner_source_digests[relative]
            for relative in (
                "src/psa/confirmatory/runner.py",
                "src/psa/confirmatory/rwkv_backend.py",
                "src/psa/confirmatory/development.py",
            )
        }
        and runner_evidence.get("contains_derived_accuracy") is False
        and runner_evidence.get("formal_authorization_used") is False
        and runner_evidence.get("confirmatory_experiment_run") is False
        and runner_evidence.get("confirmatory_results_observed") is False
    )

    environment_checks = _environment_checks(environment_report)
    asset_checks = _asset_model_consistency(model_config, asset_report)
    frozen_package_checks = {
        "candidate_digest_valid": (
            final_manifest.get("candidate_digest_sha256")
            == candidate.get("candidate_digest_sha256")
            == EXPECTED_CANDIDATE_DIGEST
        ),
        "final_preregistration_digest_valid": (
            final_manifest.get("final_preregistration_digest_sha256")
            == core_manifest.get("final_preregistration_digest_sha256")
            == core_set.get("final_preregistration_digest_sha256")
            == EXPECTED_FINAL_DIGEST
        ),
        "core_set_digest_valid": (
            core_manifest.get("core_set_digest_sha256")
            == core_set.get("core_set_digest_sha256")
            == EXPECTED_CORE_SET_DIGEST
        ),
        "core_set_package_digest_valid": (
            core_manifest.get("core_set_package_digest_sha256")
            == EXPECTED_CORE_PACKAGE_DIGEST
        ),
        "core_set_still_unrun": (
            core_manifest.get("status") == "core_set_frozen_unrun"
            and core_set.get("status") == "core_set_frozen_unrun"
            and core_set.get("confirmatory_experiment_run") is False
            and core_set.get("confirmatory_results_observed") is False
        ),
        "design_counts_frozen": (
            core_set.get("factorial_group_count") == 320
            and core_set.get("semantic_case_count") == 1280
            and core_set.get("trial_count") == 5120
            and len(core_set.get("groups", [])) == 320
        ),
        "conditions_frozen": tuple(core_set.get("conditions", ()))
        == EXPECTED_CONDITIONS,
        "model_identity_frozen": (
            model_config.get("model_id") == candidate.get("model_id")
            == EXPECTED_MODEL_ID
        ),
    }
    all_checks = {
        **frozen_package_checks,
        **environment_checks,
        **asset_checks,
        "frozen_scoring_sources_valid": all(
            item["valid"] for item in source_checks.values()
        ),
        "runner_sources_complete": runner_sources_complete,
    }

    git = environment_report.get("git", {})
    torch = environment_report.get("torch", {})
    nvidia = environment_report.get("nvidia_smi", {})
    stable_plan = {
        "plan_version": "0.1",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "stage": "impl5b_preflight_a",
        "git_commit": git.get("commit") if isinstance(git, Mapping) else None,
        "candidate_digest_sha256": EXPECTED_CANDIDATE_DIGEST,
        "final_preregistration_digest_sha256": EXPECTED_FINAL_DIGEST,
        "core_set_digest_sha256": EXPECTED_CORE_SET_DIGEST,
        "core_set_package_digest_sha256": EXPECTED_CORE_PACKAGE_DIGEST,
        "model_id": EXPECTED_MODEL_ID,
        "weights_sha256": EXPECTED_WEIGHT_DIGEST,
        "tokenizer_sha256": EXPECTED_TOKENIZER_DIGEST,
        "conditions": list(EXPECTED_CONDITIONS),
        "factorial_group_count": 320,
        "semantic_case_count": 1280,
        "rotation_trial_count": 5120,
        "planned_trial_condition_count": 5120 * len(EXPECTED_CONDITIONS),
        "statistics": candidate.get("statistics"),
        "environment": {
            "torch": torch.get("version") if isinstance(torch, Mapping) else None,
            "cuda": (
                torch.get("cuda_runtime")
                if isinstance(torch, Mapping)
                else None
            ),
            "gpus": nvidia.get("gpus") if isinstance(nvidia, Mapping) else None,
        },
        "frozen_scoring_source_digests": {
            relative: item["actual_sha256"]
            for relative, item in source_checks.items()
        },
        "runner_source_digests": runner_source_digests,
        "runner_development_evidence_sha256": (
            sha256_file(runner_evidence_source)
            if runner_evidence_valid and runner_evidence_source is not None
            else None
        ),
        "safety_rules": {
            "partial_core_runs_forbidden": True,
            "intermediate_accuracy_reporting_forbidden": True,
            "automatic_rerun_forbidden": True,
            "frozen_design_mutation_forbidden": True,
            "confirmatory_authorization_required": True,
        },
    }
    preflight_digest = sha256_json(stable_plan)
    valid = all(all_checks.values())
    authorization_ready = bool(valid and runner_evidence_valid)
    return {
        "report_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "model_loaded": False,
        "confirmatory_trial_scored": False,
        "confirmatory_experiment_authorized": False,
        "confirmatory_experiment_run": False,
        "confirmatory_results_observed": False,
        "status": (
            "preflight_valid_authorization_still_required"
            if authorization_ready
            else (
                "preflight_valid_runner_evidence_required"
                if valid
                else "preflight_failed"
            )
        ),
        "route_decision": (
            "review_project_owner_confirmatory_authorization"
            if authorization_ready
            else (
                "run_non_core_runner_development_gate"
                if valid
                else "hold_and_repair_preflight_without_core_inference"
            )
        ),
        "paths": {
            "project_root": str(root),
            "final_package": _relative(final_root, root),
            "core_set_package": _relative(core_root, root),
            "model_config": _relative(model_path, root),
            "asset_manifest": _relative(asset_path, root),
            "runner_evidence": (
                _relative(runner_evidence_source, root)
                if runner_evidence_source is not None
                else None
            ),
        },
        "checks": all_checks,
        "frozen_scoring_source_checks": source_checks,
        "run_plan_candidate": stable_plan,
        "runner_development_evidence": {
            "provided": runner_evidence is not None,
            "valid": runner_evidence_valid,
        },
        "preflight_digest_sha256": preflight_digest,
        "authorization_boundary": {
            "existing_core_set_authorization_allows_run": False,
            "new_project_owner_authorization_required": True,
            "authorization_must_bind_preflight_digest_sha256": preflight_digest,
        },
        "valid": valid,
    }


def verify_confirmatory_run_authorization(
    authorization: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a future explicit authorization; never infer it from Core Set auth."""
    scope = authorization.get("authorization")
    expected_scope = {
        "run_confirmatory_experiment": True,
        "observe_results_after_full_completion": True,
        "modify_frozen_design": False,
        "automatic_rerun_after_results": False,
    }
    expected_keys = {
        "authorization_version",
        "experiment_id",
        "authorized_by_role",
        "authorized_at_utc",
        "authorization_text",
        "preflight_digest_sha256",
        "final_preregistration_digest_sha256",
        "core_set_digest_sha256",
        "core_set_package_digest_sha256",
        "model_id",
        "authorization",
    }
    checks = {
        "preflight_valid": preflight.get("valid") is True,
        "runner_evidence_valid": bool(
            isinstance(preflight.get("runner_development_evidence"), Mapping)
            and preflight["runner_development_evidence"].get("valid") is True
            and preflight.get("status")
            == "preflight_valid_authorization_still_required"
        ),
        "authorization_version_valid": authorization.get(
            "authorization_version"
        )
        == "1.0",
        "experiment_id_valid": authorization.get("experiment_id")
        == EXPECTED_EXPERIMENT_ID,
        "authorized_by_project_owner": authorization.get(
            "authorized_by_role"
        )
        == "project_owner",
        "authorization_shape_exact": set(authorization) == expected_keys,
        "authorization_timestamp_present": bool(
            isinstance(authorization.get("authorized_at_utc"), str)
            and len(authorization["authorized_at_utc"]) >= 20
        ),
        "authorization_text_present": bool(
            isinstance(authorization.get("authorization_text"), str)
            and len(authorization["authorization_text"].strip()) >= 20
        ),
        "preflight_digest_bound": authorization.get(
            "preflight_digest_sha256"
        )
        == preflight.get("preflight_digest_sha256"),
        "final_digest_bound": authorization.get(
            "final_preregistration_digest_sha256"
        )
        == EXPECTED_FINAL_DIGEST,
        "core_set_digest_bound": authorization.get("core_set_digest_sha256")
        == EXPECTED_CORE_SET_DIGEST,
        "core_set_package_digest_bound": authorization.get(
            "core_set_package_digest_sha256"
        )
        == EXPECTED_CORE_PACKAGE_DIGEST,
        "model_id_bound": authorization.get("model_id") == EXPECTED_MODEL_ID,
        "scope_exact": scope == expected_scope,
    }
    return {
        "report_version": "0.1",
        "experiment_id": authorization.get("experiment_id"),
        "checks": checks,
        "valid": all(checks.values()),
    }
