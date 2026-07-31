from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)


FINAL_STATUS = "final_preregistration_frozen"
EXPECTED_EXPERIMENT_ID = "EXP-001"
EXPECTED_GATE = "impl3t_exp001_formal_v3_holdout"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_package_file(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("locked package filename must be a non-empty string")
    path = (root / filename).resolve()
    if path == root or root not in path.parents:
        raise ValueError("locked package filename escapes package directory")
    return path


def _require_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def _validate_candidate(candidate: dict[str, Any]) -> None:
    expected_digest = candidate.get("candidate_digest_sha256")
    unsigned = dict(candidate)
    unsigned.pop("candidate_digest_sha256", None)
    if not isinstance(expected_digest, str) or sha256_json(unsigned) != expected_digest:
        raise ValueError("candidate self digest is invalid")
    locked_payload = {
        **{
            f"source:{path}": digest
            for path, digest in candidate.get("source_file_digests", {}).items()
        },
        **{
            f"evidence:{path}": digest
            for path, digest in candidate.get(
                "evidence_file_digests", {}
            ).items()
        },
    }
    if not locked_payload:
        raise ValueError("candidate has no locked payload")
    if payload_digest(locked_payload) != candidate.get(
        "payload_root_digest_sha256"
    ):
        raise ValueError("candidate payload root is invalid")
    if candidate.get("status") != (
        "frozen_candidate_awaiting_human_checksum_confirmation"
    ):
        raise ValueError("candidate is not awaiting human checksum confirmation")
    if candidate.get("gate") != EXPECTED_GATE:
        raise ValueError("candidate is not the accepted Impl-3t holdout")
    qualification = candidate.get("qualification")
    if not isinstance(qualification, dict) or not qualification:
        raise ValueError("candidate qualification is incomplete")
    if not all(value is True for value in qualification.values()):
        raise ValueError("candidate qualification did not fully pass")
    if candidate.get("eligible_for_human_freeze") is not True:
        raise ValueError("candidate is not eligible for human freeze")
    if not (
        candidate.get("core_set_generated") is False
        and candidate.get("core_set_unsealed") is False
        and candidate.get("formal_state_only_results_observed") is False
        and candidate.get("human_checksum_confirmation_required") is True
    ):
        raise ValueError("candidate safety boundary is invalid")


def _validate_verification(
    verification: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    if verification.get("candidate_digest_sha256") != candidate.get(
        "candidate_digest_sha256"
    ):
        raise ValueError("verification refers to a different candidate")
    required_true = (
        "self_digest_valid",
        "payload_root_valid",
        "safety_boundary_valid",
        "eligible_for_human_freeze",
        "valid",
    )
    if not all(verification.get(key) is True for key in required_true):
        raise ValueError("manual candidate verification did not fully pass")
    source_checks = verification.get("source_file_checks")
    evidence_checks = verification.get("evidence_file_checks")
    if not isinstance(source_checks, dict) or set(source_checks) != set(
        candidate["source_file_digests"]
    ):
        raise ValueError("verification source-file inventory is incomplete")
    if not isinstance(evidence_checks, dict) or set(evidence_checks) != set(
        candidate["evidence_file_digests"]
    ):
        raise ValueError("verification evidence-file inventory is incomplete")
    if not all(value is True for value in source_checks.values()):
        raise ValueError("verification contains a failed source-file check")
    if not all(value is True for value in evidence_checks.values()):
        raise ValueError("verification contains a failed evidence-file check")


def _validate_confirmation(
    confirmation: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    if confirmation.get("confirmation_version") != "1.0":
        raise ValueError("unsupported confirmation version")
    if confirmation.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("confirmation is for a different experiment")
    if confirmation.get("candidate_digest_sha256") != candidate.get(
        "candidate_digest_sha256"
    ):
        raise ValueError("human confirmation checksum does not match candidate")
    if confirmation.get("confirmed_by_role") != "project_owner":
        raise ValueError("confirmation must be made by the project owner")
    _require_utc_timestamp(
        confirmation.get("confirmed_at_utc"),
        "confirmed_at_utc",
    )
    text = confirmation.get("confirmation_text")
    if (
        not isinstance(text, str)
        or candidate["candidate_digest_sha256"] not in text
    ):
        raise ValueError("confirmation text must contain the full checksum")
    authorization = confirmation.get("authorization")
    expected_authorization = {
        "upgrade_to_final_preregistration_package": True,
        "generate_core_set": False,
        "run_confirmatory_experiment": False,
    }
    if authorization != expected_authorization:
        raise ValueError("confirmation authorization scope is invalid")


def _manifest_without_digest(
    *,
    candidate: dict[str, Any],
    confirmation: dict[str, Any],
    locked_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "package_version": "1.0",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": FINAL_STATUS,
        "frozen_at_utc": confirmation["confirmed_at_utc"],
        "source_gate": candidate["gate"],
        "model_id": candidate["model_id"],
        "candidate_digest_sha256": candidate[
            "candidate_digest_sha256"
        ],
        "candidate_payload_root_digest_sha256": candidate[
            "payload_root_digest_sha256"
        ],
        "human_confirmation": {
            "confirmed_by_role": confirmation["confirmed_by_role"],
            "confirmed_at_utc": confirmation["confirmed_at_utc"],
            "confirmation_file": "human_confirmation.json",
            "confirmation_file_sha256": locked_files[
                "human_confirmation.json"
            ],
        },
        "qualification": candidate["qualification"],
        "confirmed_decision_ids": candidate["confirmed_decision_ids"],
        "conditions": candidate["conditions"],
        "factorial_group_count": candidate["factorial_group_count"],
        "seeds": candidate["seeds"],
        "statistics": candidate["statistics"],
        "authorization": confirmation["authorization"],
        "safety_boundary": {
            "core_set_generated": False,
            "core_set_unsealed": False,
            "formal_state_only_results_observed": False,
            "core_set_generation_authorized": False,
            "confirmatory_experiment_authorized": False,
        },
        "source_file_digest_count": len(
            candidate["source_file_digests"]
        ),
        "evidence_file_digest_count": len(
            candidate["evidence_file_digests"]
        ),
        "locked_files": locked_files,
        "package_payload_root_digest_sha256": payload_digest(locked_files),
    }


def verify_final_preregistration_package(
    package_dir: str | Path,
) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_object(manifest_path, "final preregistration manifest")
    expected_digest = manifest.get("final_preregistration_digest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("final_preregistration_digest_sha256", None)
    manifest_digest_valid = (
        isinstance(expected_digest, str)
        and sha256_json(unsigned) == expected_digest
    )
    locked_files = manifest.get("locked_files")
    locked_file_checks: dict[str, bool] = {}
    if isinstance(locked_files, dict):
        for filename, expected in locked_files.items():
            try:
                path = _resolve_package_file(root, filename)
                locked_file_checks[filename] = bool(
                    path.is_file() and sha256_file(path) == expected
                )
            except (TypeError, ValueError):
                locked_file_checks[str(filename)] = False
    payload_root_valid = bool(
        isinstance(locked_files, dict)
        and payload_digest(locked_files)
        == manifest.get("package_payload_root_digest_sha256")
    )
    safety = manifest.get("safety_boundary")
    authorization = manifest.get("authorization")
    safety_boundary_valid = bool(
        isinstance(safety, dict)
        and safety.get("core_set_generated") is False
        and safety.get("core_set_unsealed") is False
        and safety.get("formal_state_only_results_observed") is False
        and safety.get("core_set_generation_authorized") is False
        and safety.get("confirmatory_experiment_authorized") is False
        and authorization
        == {
            "upgrade_to_final_preregistration_package": True,
            "generate_core_set": False,
            "run_confirmatory_experiment": False,
        }
    )
    package_content_valid = False
    try:
        candidate = _load_object(root / "candidate.json", "locked candidate")
        verification = _load_object(
            root / "verification.json",
            "locked verification",
        )
        confirmation = _load_object(
            root / "human_confirmation.json",
            "locked confirmation",
        )
        _validate_candidate(candidate)
        _validate_verification(verification, candidate)
        _validate_confirmation(confirmation, candidate)
        package_content_valid = bool(
            manifest.get("candidate_digest_sha256")
            == candidate.get("candidate_digest_sha256")
            and manifest.get("candidate_payload_root_digest_sha256")
            == candidate.get("payload_root_digest_sha256")
            and manifest.get("source_gate") == candidate.get("gate")
            and manifest.get("model_id") == candidate.get("model_id")
            and manifest.get("authorization")
            == confirmation.get("authorization")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        package_content_valid = False
    valid = bool(
        manifest.get("status") == FINAL_STATUS
        and manifest_digest_valid
        and locked_file_checks
        and all(locked_file_checks.values())
        and payload_root_valid
        and safety_boundary_valid
        and package_content_valid
    )
    return {
        "report_version": "1.0",
        "package_dir": str(root),
        "status": manifest.get("status"),
        "candidate_digest_sha256": manifest.get(
            "candidate_digest_sha256"
        ),
        "final_preregistration_digest_sha256": expected_digest,
        "manifest_digest_valid": manifest_digest_valid,
        "locked_file_checks": locked_file_checks,
        "payload_root_valid": payload_root_valid,
        "safety_boundary_valid": safety_boundary_valid,
        "package_content_valid": package_content_valid,
        "core_set_generated": False,
        "confirmatory_experiment_run": False,
        "valid": valid,
    }


def finalize_preregistration_package(
    *,
    candidate_path: str | Path,
    verification_path: str | Path,
    confirmation_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    candidate_source = Path(candidate_path).resolve()
    verification_source = Path(verification_path).resolve()
    confirmation_source = Path(confirmation_path).resolve()
    destination = Path(output_dir).resolve()
    candidate = _load_object(candidate_source, "preregistration candidate")
    verification = _load_object(
        verification_source,
        "manual candidate verification",
    )
    confirmation = _load_object(
        confirmation_source,
        "human checksum confirmation",
    )
    _validate_candidate(candidate)
    _validate_verification(verification, candidate)
    _validate_confirmation(confirmation, candidate)
    if destination.exists():
        existing = verify_final_preregistration_package(destination)
        existing_manifest = _load_object(
            destination / "manifest.json",
            "existing final preregistration manifest",
        )
        expected_locked_files = {
            "candidate.json": sha256_file(candidate_source),
            "human_confirmation.json": sha256_file(confirmation_source),
            "verification.json": sha256_file(verification_source),
        }
        if (
            existing["valid"]
            and existing["candidate_digest_sha256"]
            == candidate["candidate_digest_sha256"]
            and existing_manifest.get("locked_files")
            == expected_locked_files
        ):
            return existing
        raise ValueError("final preregistration output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        sources = {
            "candidate.json": candidate_source,
            "verification.json": verification_source,
            "human_confirmation.json": confirmation_source,
        }
        for filename, source in sources.items():
            shutil.copyfile(source, staging / filename)
        locked_files = {
            filename: sha256_file(staging / filename)
            for filename in sorted(sources)
        }
        manifest = _manifest_without_digest(
            candidate=candidate,
            confirmation=confirmation,
            locked_files=locked_files,
        )
        manifest["final_preregistration_digest_sha256"] = sha256_json(
            manifest
        )
        (staging / "manifest.json").write_bytes(
            canonical_json_bytes(manifest)
        )
        shutil.copytree(staging, destination)
    report = verify_final_preregistration_package(destination)
    if not report["valid"]:
        raise RuntimeError("final preregistration package failed self-check")
    return report
