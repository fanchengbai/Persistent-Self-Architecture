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
from psa.supplemental.development import (
    BDEV1_GATE,
    BDEV2_GATE,
    _load_confirmed_design,
)


CANDIDATE_GATE = "exp001b_preregistration_candidate_v1"
CANDIDATE_STATUS = "candidate_awaiting_human_checksum_confirmation"
BDEV2_V01_GATE = "exp001b_bdev2_non_core_runner"


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


def _relative_under(root: Path, path: Path, label: str) -> str:
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{label} must be inside the project root")
    return resolved.relative_to(root).as_posix()


def _resolve_report(directory: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("evidence report path must be a non-empty string")
    path = (directory / name).resolve()
    if path == directory or directory not in path.parents:
        raise ValueError("evidence report path escapes its gate directory")
    if not path.is_file():
        raise ValueError(f"missing evidence report: {name}")
    return path


def _safety_is_unrun(payload: Mapping[str, Any]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "core_set_accessed",
            "supplemental_set_generated",
            "supplemental_experiment_authorized",
            "supplemental_experiment_run",
            "supplemental_results_observed",
        )
        if field in payload
    ) and all(
        payload.get(field) is False
        for field in (
            "supplemental_set_generated",
            "supplemental_experiment_authorized",
            "supplemental_experiment_run",
            "supplemental_results_observed",
        )
    )


def _validate_bdev1(
    summary: Mapping[str, Any],
    *,
    design_sha256: str,
    model_id: str,
) -> None:
    valid = bool(
        summary.get("gate") == BDEV1_GATE
        and summary.get("valid") is True
        and summary.get("development_only") is True
        and summary.get("design_sha256") == design_sha256
        and summary.get("model_id") == model_id
        and summary.get("matched_context_case_count") == 64
        and summary.get("matched_context_valid") is True
        and summary.get("state_norm_case_count") == 64
        and summary.get("state_norm_thresholds_valid") is True
        and summary.get("state_component_count") == 96
        and _safety_is_unrun(summary)
    )
    if not valid:
        raise ValueError("B-Dev1 evidence is not an eligible non-Core calibration")


def _validate_bdev2_v01(
    summary: Mapping[str, Any],
    *,
    design_sha256: str,
    model_id: str,
) -> None:
    valid = bool(
        summary.get("gate") == BDEV2_V01_GATE
        and summary.get("valid") is False
        and summary.get("development_only") is True
        and summary.get("design_sha256") == design_sha256
        and summary.get("model_id") == model_id
        and summary.get("condition_runner_valid") is True
        and summary.get("condition_record_count") == 128
        and summary.get("matched_context_probe_valid") is True
        and summary.get("matched_context_record_count") == 16
        and summary.get("generation_probe_valid") is False
        and summary.get("generation_record_count") == 16
        and summary.get("forced_prefix_greedy_exact_rate") == 1.0
        and summary.get("format_valid_rate") == 0.875
        and summary.get("state_norm_probe_valid") is False
        and _safety_is_unrun(summary)
    )
    if not valid:
        raise ValueError("B-Dev2 v0.1 failure evidence is missing or was altered")


def _validate_bdev2_v02(
    summary: Mapping[str, Any],
    *,
    design_sha256: str,
    bdev1_dir: Path,
    model_id: str,
) -> None:
    expected_bdev1 = {
        "bdev1_summary_sha256": sha256_file(bdev1_dir / "summary.json"),
        "bdev1_thresholds_sha256": sha256_file(
            bdev1_dir / "state_norm_thresholds.json"
        ),
        "bdev1_matched_report_sha256": sha256_file(
            bdev1_dir / "matched_context_token_report.json"
        ),
    }
    valid = bool(
        summary.get("gate") == BDEV2_GATE
        and summary.get("revision_id") == "formal-shaped-non-core-probes-v0.2"
        and summary.get("valid") is True
        and summary.get("development_only") is True
        and summary.get("design_sha256") == design_sha256
        and summary.get("model_id") == model_id
        and all(summary.get(key) == value for key, value in expected_bdev1.items())
        and summary.get("bdev1_valid") is True
        and summary.get("condition_alias_valid") is True
        and summary.get("condition_runner_valid") is True
        and summary.get("condition_record_count") == 128
        and summary.get("matched_context_probe_valid") is True
        and summary.get("matched_context_record_count") == 16
        and summary.get("formal_probe_manifest_valid") is True
        and summary.get("formal_probe_shape_warmup_excluded_from_scoring") is True
        and summary.get("generation_probe_valid") is True
        and summary.get("generation_record_count") == 64
        and summary.get("forced_prefix_greedy_exact_rate") == 1.0
        and summary.get("format_valid_rate") == 1.0
        and summary.get("state_norm_probe_valid") is True
        and summary.get("state_norm_record_count") == 64
        and _safety_is_unrun(summary)
    )
    if not valid:
        raise ValueError("B-Dev2 v0.2 evidence is not eligible for candidate review")


def _copy_evidence_inventory(
    *,
    source_dir: Path,
    summary: Mapping[str, Any],
    destination: Path,
    prefix: str,
) -> dict[str, str]:
    reports = summary.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"{prefix} evidence inventory is missing")
    names = ["summary.json", *reports]
    if len(names) != len(set(names)):
        raise ValueError(f"{prefix} evidence inventory contains duplicates")
    digests: dict[str, str] = {}
    for name in names:
        source = _resolve_report(source_dir, name)
        relative = f"evidence/{prefix}/{name}"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digests[relative] = sha256_file(target)
    return digests


def _source_inventory(root: Path, design_relative: str) -> dict[str, str]:
    names = (
        design_relative,
        "src/psa/supplemental/development.py",
        "src/psa/supplemental/freeze.py",
        "src/psa/artifacts/integrity.py",
        "src/psa/confirmatory/runner.py",
        "src/psa/confirmatory/rwkv_backend.py",
        "scripts/run_exp001b_bdev1_gate.sh",
        "scripts/run_exp001b_bdev2_gate.sh",
        "scripts/build_exp001b_preregistration_candidate.sh",
        "tests/test_exp001b_design.py",
        "tests/test_exp001b_development.py",
        "tests/test_exp001b_freeze.py",
    )
    result: dict[str, str] = {}
    for name in names:
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"candidate source file is missing: {name}")
        result[name] = sha256_file(path)
    return result


def verify_exp001b_preregistration_candidate(
    candidate_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(candidate_path).resolve()
    candidate = _load_object(path, "EXP-001B preregistration candidate")
    expected_digest = candidate.get("candidate_digest_sha256")
    unsigned = dict(candidate)
    unsigned.pop("candidate_digest_sha256", None)
    self_digest_valid = bool(
        isinstance(expected_digest, str) and sha256_json(unsigned) == expected_digest
    )

    source_checks: dict[str, bool] = {}
    for name, expected in candidate.get("source_file_digests", {}).items():
        try:
            source = (root / name).resolve()
            source_checks[name] = bool(
                root in source.parents
                and source.is_file()
                and sha256_file(source) == expected
            )
        except (OSError, TypeError, ValueError):
            source_checks[str(name)] = False

    evidence_checks: dict[str, bool] = {}
    for name, expected in candidate.get("evidence_file_digests", {}).items():
        try:
            evidence = (path.parent / name).resolve()
            evidence_checks[name] = bool(
                path.parent in evidence.parents
                and evidence.is_file()
                and sha256_file(evidence) == expected
            )
        except (OSError, TypeError, ValueError):
            evidence_checks[str(name)] = False

    locked_payload = {
        **{
            f"source:{name}": digest
            for name, digest in candidate.get("source_file_digests", {}).items()
        },
        **{
            f"evidence:{name}": digest
            for name, digest in candidate.get("evidence_file_digests", {}).items()
        },
    }
    payload_root_valid = bool(
        locked_payload
        and payload_digest(locked_payload)
        == candidate.get("payload_root_digest_sha256")
    )
    safety = candidate.get("safety_boundary")
    safety_boundary_valid = bool(
        isinstance(safety, Mapping)
        and safety.get("candidate_confirmed") is False
        and safety.get("supplemental_set_generated") is False
        and safety.get("supplemental_set_generation_authorized") is False
        and safety.get("supplemental_experiment_authorized") is False
        and safety.get("supplemental_experiment_run") is False
        and safety.get("supplemental_results_observed") is False
        and safety.get("automatic_rerun_authorized") is False
        and safety.get("human_checksum_confirmation_required_before_generation")
        is True
    )
    valid = bool(
        candidate.get("experiment_id") == "EXP-001B"
        and candidate.get("gate") == CANDIDATE_GATE
        and candidate.get("status") == CANDIDATE_STATUS
        and candidate.get("eligible_for_human_confirmation") is True
        and self_digest_valid
        and payload_root_valid
        and source_checks
        and all(source_checks.values())
        and evidence_checks
        and all(evidence_checks.values())
        and safety_boundary_valid
    )
    return {
        "verification_version": "0.1",
        "candidate": path.name,
        "candidate_digest_sha256": expected_digest,
        "self_digest_valid": self_digest_valid,
        "payload_root_valid": payload_root_valid,
        "source_file_checks": source_checks,
        "evidence_file_checks": evidence_checks,
        "safety_boundary_valid": safety_boundary_valid,
        "candidate_confirmed": False,
        "supplemental_set_generated": False,
        "supplemental_set_generation_authorized": False,
        "supplemental_experiment_authorized": False,
        "supplemental_experiment_run": False,
        "supplemental_results_observed": False,
        "valid": valid,
    }


def build_exp001b_preregistration_candidate(
    *,
    design_path: str | Path,
    bdev1_dir: str | Path,
    bdev2_v01_dir: str | Path,
    bdev2_v02_dir: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    design_file = Path(design_path).resolve()
    design_relative = _relative_under(root, design_file, "design")
    design = _load_confirmed_design(design_file)
    design_sha256 = sha256_file(design_file)

    evidence_dirs = {
        "bdev1": Path(bdev1_dir).resolve(),
        "bdev2_v01": Path(bdev2_v01_dir).resolve(),
        "bdev2_v02": Path(bdev2_v02_dir).resolve(),
    }
    for label, directory in evidence_dirs.items():
        _relative_under(root, directory / "summary.json", f"{label} summary")
    summaries = {
        label: _load_object(directory / "summary.json", f"{label} summary")
        for label, directory in evidence_dirs.items()
    }
    model_id = design["model_lock"]["model_id"]
    _validate_bdev1(
        summaries["bdev1"],
        design_sha256=design_sha256,
        model_id=model_id,
    )
    _validate_bdev2_v01(
        summaries["bdev2_v01"],
        design_sha256=design_sha256,
        model_id=model_id,
    )
    _validate_bdev2_v02(
        summaries["bdev2_v02"],
        design_sha256=design_sha256,
        bdev1_dir=evidence_dirs["bdev1"],
        model_id=model_id,
    )

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            "candidate output already exists; refuse to overwrite a checksum package"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        evidence_digests: dict[str, str] = {}
        for label, directory in evidence_dirs.items():
            evidence_digests.update(
                _copy_evidence_inventory(
                    source_dir=directory,
                    summary=summaries[label],
                    destination=temporary,
                    prefix=label,
                )
            )
        source_digests = _source_inventory(root, design_relative)
        locked_payload = {
            **{f"source:{key}": value for key, value in source_digests.items()},
            **{f"evidence:{key}": value for key, value in evidence_digests.items()},
        }
        candidate: dict[str, Any] = {
            "candidate_version": "0.1",
            "experiment_id": "EXP-001B",
            "gate": CANDIDATE_GATE,
            "status": CANDIDATE_STATUS,
            "created_at_utc": _utc_now(),
            "role": design["role"],
            "model_lock": design["model_lock"],
            "parent_evidence": design["parent_evidence"],
            "locked_design": design,
            "source_config": {
                "path": design_relative,
                "sha256": design_sha256,
            },
            "development_qualification": {
                "bdev1_non_core_calibration_passed": True,
                "bdev2_v01_failure_preserved": True,
                "bdev2_v02_non_core_runner_passed": True,
                "formal_probe_manifest_digest_sha256": summaries["bdev2_v02"][
                    "formal_probe_manifest_digest_sha256"
                ],
                "generation_format_valid_rate": summaries["bdev2_v02"][
                    "format_valid_rate"
                ],
                "state_norm_probe_valid": summaries["bdev2_v02"][
                    "state_norm_probe_valid"
                ],
            },
            "source_evidence": {
                label: {
                    "path": _relative_under(root, directory / "summary.json", label),
                    "sha256": sha256_file(directory / "summary.json"),
                }
                for label, directory in evidence_dirs.items()
            },
            "source_file_digests": source_digests,
            "evidence_file_digests": evidence_digests,
            "payload_root_digest_sha256": payload_digest(locked_payload),
            "eligible_for_human_confirmation": True,
            "human_confirmation_scope": (
                "upgrade this candidate to a final EXP-001B preregistration package "
                "only; supplemental set generation and formal run each require "
                "separate project-owner authorization"
            ),
            "safety_boundary": {
                "candidate_confirmed": False,
                "supplemental_set_generated": False,
                "supplemental_set_generation_authorized": False,
                "supplemental_experiment_authorized": False,
                "supplemental_experiment_run": False,
                "supplemental_results_observed": False,
                "automatic_rerun_authorized": False,
                "modify_exp001_artifacts_authorized": False,
                "human_checksum_confirmation_required_before_generation": True,
                "separate_human_run_authorization_required_after_preflight": True,
            },
        }
        candidate["candidate_digest_sha256"] = sha256_json(candidate)
        candidate_path = temporary / "preregistration_candidate.json"
        _write_json(candidate_path, candidate)
        verification = verify_exp001b_preregistration_candidate(
            candidate_path,
            project_root=root,
        )
        if not verification["valid"]:
            raise RuntimeError("generated EXP-001B candidate did not verify")
        _write_json(temporary / "preregistration_verification.json", verification)
        summary = {
            "summary_version": "0.1",
            "gate": CANDIDATE_GATE,
            "development_only": True,
            "finished_at_utc": _utc_now(),
            "model_id": model_id,
            "design_sha256": design_sha256,
            "candidate_digest_sha256": candidate["candidate_digest_sha256"],
            "payload_root_digest_sha256": candidate["payload_root_digest_sha256"],
            "source_file_count": len(source_digests),
            "evidence_file_count": len(evidence_digests),
            "candidate_ready_for_human_review": True,
            "candidate_confirmed": False,
            "human_checksum_confirmation_required": True,
            "core_set_accessed": False,
            "supplemental_set_generated": False,
            "supplemental_set_generation_authorized": False,
            "supplemental_experiment_authorized": False,
            "supplemental_experiment_run": False,
            "supplemental_results_observed": False,
            "route_decision": "review_exp001b_candidate_checksum",
            "reports": [
                "preregistration_candidate.json",
                "preregistration_verification.json",
            ],
            "valid": True,
        }
        _write_json(temporary / "summary.json", summary)
        temporary.rename(destination)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
