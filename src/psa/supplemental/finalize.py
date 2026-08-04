from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)
from psa.supplemental.freeze import (
    CANDIDATE_GATE,
    CANDIDATE_STATUS,
    verify_exp001b_preregistration_candidate,
)


FINAL_STATUS = "final_preregistration_frozen"
EXPECTED_EXPERIMENT_ID = "EXP-001B"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)))


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_under(root: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} must be a non-empty string")
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{label} escapes its package directory")
    return path


def _expected_confirmation_text(candidate_digest: str) -> str:
    return (
        "我确认 EXP-001B 预注册候选 checksum："
        f"{candidate_digest}，"
        "授权将该候选升级为最终预注册包；"
        "暂不授权生成 EXP-001B 补充测试集，不授权运行正式实验。"
    )


def _authorization() -> dict[str, bool]:
    return {
        "upgrade_to_final_preregistration_package": True,
        "generate_supplemental_set": False,
        "run_supplemental_experiment": False,
    }


def _validate_confirmation(
    confirmation: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    digest = candidate.get("candidate_digest_sha256")
    valid = bool(
        confirmation.get("confirmation_version") == "0.1"
        and confirmation.get("experiment_id") == EXPECTED_EXPERIMENT_ID
        and confirmation.get("candidate_digest_sha256") == digest
        and confirmation.get("confirmed_by_role") == "project_owner"
        and confirmation.get("confirmation_text")
        == _expected_confirmation_text(str(digest))
        and confirmation.get("authorization") == _authorization()
    )
    timestamp = confirmation.get("confirmed_at_utc")
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            valid = valid and parsed.tzinfo is not None
        except ValueError:
            valid = False
    else:
        valid = False
    if not valid:
        raise ValueError("EXP-001B confirmation text or authorization scope is invalid")


def _validate_candidate_shape(candidate: Mapping[str, Any]) -> None:
    safety = candidate.get("safety_boundary")
    valid = bool(
        candidate.get("experiment_id") == EXPECTED_EXPERIMENT_ID
        and candidate.get("gate") == CANDIDATE_GATE
        and candidate.get("status") == CANDIDATE_STATUS
        and candidate.get("eligible_for_human_confirmation") is True
        and isinstance(safety, Mapping)
        and safety.get("candidate_confirmed") is False
        and safety.get("supplemental_set_generated") is False
        and safety.get("supplemental_set_generation_authorized") is False
        and safety.get("supplemental_experiment_authorized") is False
        and safety.get("supplemental_experiment_run") is False
        and safety.get("supplemental_results_observed") is False
    )
    if not valid:
        raise ValueError("EXP-001B candidate safety boundary is invalid")


def _locked_file_checks(
    root: Path,
    locked_files: Any,
) -> dict[str, bool]:
    if not isinstance(locked_files, Mapping) or not locked_files:
        return {}
    checks: dict[str, bool] = {}
    for name, expected in locked_files.items():
        try:
            path = _resolve_under(root, name, "locked file")
            checks[str(name)] = bool(
                isinstance(expected, str)
                and path.is_file()
                and sha256_file(path) == expected
            )
        except (OSError, TypeError, ValueError):
            checks[str(name)] = False
    return checks


def verify_exp001b_final_preregistration_package(
    package_dir: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    manifest = _load_object(root / "manifest.json", "EXP-001B final manifest")
    expected_digest = manifest.get("final_preregistration_digest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("final_preregistration_digest_sha256", None)
    manifest_digest_valid = bool(
        isinstance(expected_digest, str) and sha256_json(unsigned) == expected_digest
    )
    locked_files = manifest.get("locked_files")
    file_checks = _locked_file_checks(root, locked_files)
    package_payload_root_valid = bool(
        isinstance(locked_files, Mapping)
        and payload_digest(dict(locked_files))
        == manifest.get("package_payload_root_digest_sha256")
    )

    candidate_path = root / "preregistration_candidate.json"
    candidate_verification = verify_exp001b_preregistration_candidate(
        candidate_path,
        project_root=project_root,
    )
    candidate = _load_object(candidate_path, "locked EXP-001B candidate")
    stored_verification = _load_object(
        root / "preregistration_verification.json",
        "locked EXP-001B verification",
    )
    confirmation = _load_object(
        root / "human_confirmation.json",
        "locked EXP-001B confirmation",
    )
    try:
        _validate_candidate_shape(candidate)
        _validate_confirmation(confirmation, candidate)
        stored_verification_valid = bool(
            stored_verification.get("valid") is True
            and stored_verification.get("candidate_digest_sha256")
            == candidate.get("candidate_digest_sha256")
            and all(
                stored_verification.get(field) is True
                for field in (
                    "self_digest_valid",
                    "payload_root_valid",
                    "safety_boundary_valid",
                )
            )
            and all(
                stored_verification.get(field) is False
                for field in (
                    "candidate_confirmed",
                    "supplemental_set_generated",
                    "supplemental_set_generation_authorized",
                    "supplemental_experiment_authorized",
                    "supplemental_experiment_run",
                    "supplemental_results_observed",
                )
            )
            and all(
                stored_verification.get("source_file_checks", {}).values()
            )
            and all(
                stored_verification.get("evidence_file_checks", {}).values()
            )
            and set(stored_verification.get("source_file_checks", {}))
            == set(candidate["source_file_digests"])
            and set(stored_verification.get("evidence_file_checks", {}))
            == set(candidate["evidence_file_digests"])
        )
        expected_locked_files = {
            "preregistration_candidate.json",
            "preregistration_verification.json",
            "human_confirmation.json",
            *candidate["evidence_file_digests"].keys(),
        }
        locked_inventory_valid = bool(
            isinstance(locked_files, Mapping)
            and set(locked_files) == expected_locked_files
            and manifest.get("locked_file_count") == len(expected_locked_files)
            and manifest.get("source_file_digest_count")
            == len(candidate["source_file_digests"])
            and manifest.get("evidence_file_digest_count")
            == len(candidate["evidence_file_digests"])
        )
        package_content_valid = bool(
            candidate_verification["valid"]
            and stored_verification_valid
            and locked_inventory_valid
            and manifest.get("candidate_digest_sha256")
            == candidate.get("candidate_digest_sha256")
            and manifest.get("candidate_payload_root_digest_sha256")
            == candidate.get("payload_root_digest_sha256")
            and manifest.get("model_id") == candidate["model_lock"]["model_id"]
            and manifest.get("authorization") == _authorization()
        )
    except (KeyError, TypeError, ValueError):
        stored_verification_valid = False
        locked_inventory_valid = False
        package_content_valid = False

    safety = manifest.get("safety_boundary")
    safety_boundary_valid = bool(
        isinstance(safety, Mapping)
        and safety.get("final_preregistration_frozen") is True
        and safety.get("candidate_confirmed") is True
        and safety.get("supplemental_set_generated") is False
        and safety.get("supplemental_set_generation_authorized") is False
        and safety.get("supplemental_experiment_authorized") is False
        and safety.get("supplemental_experiment_run") is False
        and safety.get("supplemental_results_observed") is False
        and safety.get("automatic_rerun_authorized") is False
        and manifest.get("authorization") == _authorization()
    )
    valid = bool(
        manifest.get("experiment_id") == EXPECTED_EXPERIMENT_ID
        and manifest.get("status") == FINAL_STATUS
        and manifest_digest_valid
        and file_checks
        and all(file_checks.values())
        and package_payload_root_valid
        and package_content_valid
        and safety_boundary_valid
    )
    return {
        "verification_version": "0.1",
        "package_dir": str(root),
        "status": manifest.get("status"),
        "candidate_digest_sha256": manifest.get("candidate_digest_sha256"),
        "final_preregistration_digest_sha256": expected_digest,
        "manifest_digest_valid": manifest_digest_valid,
        "locked_file_count": len(file_checks),
        "failed_locked_files": [name for name, ok in file_checks.items() if not ok],
        "package_payload_root_valid": package_payload_root_valid,
        "candidate_verification_valid": candidate_verification["valid"],
        "stored_verification_valid": stored_verification_valid,
        "locked_inventory_valid": locked_inventory_valid,
        "package_content_valid": package_content_valid,
        "safety_boundary_valid": safety_boundary_valid,
        "supplemental_set_generated": False,
        "supplemental_set_generation_authorized": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "valid": valid,
    }


def finalize_exp001b_preregistration_package(
    *,
    candidate_dir: str | Path,
    confirmation_text: str,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    source = Path(candidate_dir).resolve()
    destination = Path(output_dir).resolve()
    candidate_path = source / "preregistration_candidate.json"
    stored_verification_path = source / "preregistration_verification.json"
    candidate = _load_object(candidate_path, "EXP-001B candidate")
    _validate_candidate_shape(candidate)
    expected_text = _expected_confirmation_text(
        str(candidate["candidate_digest_sha256"])
    )
    if confirmation_text != expected_text:
        raise ValueError("confirmation text does not exactly match the frozen checksum")
    live_verification = verify_exp001b_preregistration_candidate(
        candidate_path,
        project_root=project_root,
    )
    if not live_verification["valid"]:
        raise ValueError("EXP-001B candidate no longer passes live verification")
    stored_verification = _load_object(
        stored_verification_path,
        "EXP-001B stored verification",
    )
    if not (
        stored_verification.get("valid") is True
        and stored_verification.get("candidate_digest_sha256")
        == candidate.get("candidate_digest_sha256")
    ):
        raise ValueError("stored candidate verification is invalid")

    if destination.exists():
        existing = verify_exp001b_final_preregistration_package(
            destination,
            project_root=project_root,
        )
        existing_confirmation = _load_object(
            destination / "human_confirmation.json",
            "existing EXP-001B confirmation",
        )
        if (
            existing["valid"]
            and existing["candidate_digest_sha256"]
            == candidate["candidate_digest_sha256"]
            and existing_confirmation.get("confirmation_text") == confirmation_text
        ):
            return existing
        raise ValueError("EXP-001B final preregistration output already exists")

    confirmation = {
        "confirmation_version": "0.1",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "candidate_digest_sha256": candidate["candidate_digest_sha256"],
        "confirmed_by_role": "project_owner",
        "confirmed_at_utc": _utc_now(),
        "confirmation_text": confirmation_text,
        "authorization": _authorization(),
    }
    _validate_confirmation(confirmation, candidate)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copyfile(candidate_path, temporary / "preregistration_candidate.json")
        shutil.copyfile(
            stored_verification_path,
            temporary / "preregistration_verification.json",
        )
        _write_json(temporary / "human_confirmation.json", confirmation)
        for name in candidate["evidence_file_digests"]:
            source_evidence = _resolve_under(source, name, "candidate evidence")
            target = _resolve_under(temporary, name, "final evidence")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_evidence, target)

        package_names = [
            "preregistration_candidate.json",
            "preregistration_verification.json",
            "human_confirmation.json",
            *candidate["evidence_file_digests"].keys(),
        ]
        locked_files = {
            name: sha256_file(_resolve_under(temporary, name, "package file"))
            for name in sorted(package_names)
        }
        manifest: dict[str, Any] = {
            "package_version": "0.1",
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "status": FINAL_STATUS,
            "frozen_at_utc": confirmation["confirmed_at_utc"],
            "source_gate": candidate["gate"],
            "model_id": candidate["model_lock"]["model_id"],
            "design_sha256": candidate["source_config"]["sha256"],
            "candidate_digest_sha256": candidate["candidate_digest_sha256"],
            "candidate_payload_root_digest_sha256": candidate[
                "payload_root_digest_sha256"
            ],
            "human_confirmation": {
                "confirmed_by_role": "project_owner",
                "confirmed_at_utc": confirmation["confirmed_at_utc"],
                "confirmation_file": "human_confirmation.json",
                "confirmation_file_sha256": locked_files[
                    "human_confirmation.json"
                ],
            },
            "authorization": _authorization(),
            "source_file_digest_count": len(candidate["source_file_digests"]),
            "evidence_file_digest_count": len(candidate["evidence_file_digests"]),
            "locked_file_count": len(locked_files),
            "locked_files": locked_files,
            "package_payload_root_digest_sha256": payload_digest(locked_files),
            "safety_boundary": {
                "final_preregistration_frozen": True,
                "candidate_confirmed": True,
                "supplemental_set_generated": False,
                "supplemental_set_generation_authorized": False,
                "supplemental_experiment_authorized": False,
                "supplemental_experiment_run": False,
                "supplemental_results_observed": False,
                "automatic_rerun_authorized": False,
            },
        }
        manifest["final_preregistration_digest_sha256"] = sha256_json(manifest)
        _write_json(temporary / "manifest.json", manifest)
        staging_report = verify_exp001b_final_preregistration_package(
            temporary,
            project_root=project_root,
        )
        if not staging_report["valid"]:
            raise RuntimeError(
                "staged EXP-001B final preregistration package failed self-check"
            )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    report = verify_exp001b_final_preregistration_package(
        destination,
        project_root=project_root,
    )
    if not report["valid"]:
        raise RuntimeError("EXP-001B final preregistration package failed self-check")
    return report
