from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from psa.assets import fetch_manifest, load_manifest, plan_manifest, verify_manifest
from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.confirmatory import (
    run_exp001_confirmatory_analysis,
    build_confirmatory_preflight,
    run_exp001_confirmatory,
    run_confirmatory_runner_development_gate,
    verify_exp001_confirmatory_raw_package,
)
from psa.development import (
    build_exp001c_probe_manifest,
    run_g1_capability_audit,
    run_capability_ladder_gate,
    run_g1_capability_ladder_gate,
    run_g1_code_rotation_gate,
    run_g1_code_rotation_review,
    run_history_binding_gate,
    run_impl3_development_gate,
    validate_exp001c_probe_execution_authority,
    verify_exp001c_probe_manifest,
)
from psa.environment import collect_environment
from psa.evaluation import group_contrasts
from psa.model import run_interface_gate
from psa.preregistration import (
    finalize_preregistration_package,
    generate_and_freeze_core_set,
    run_formal_freeze_candidate_gate,
    run_formal_freeze_review,
    verify_final_preregistration_package,
    verify_core_set_package,
    verify_preregistration_candidate,
)
from psa.preregistration.core_set import _load_token_counter
from psa.state import (
    run_checkpoint_roundtrip_gate,
    run_random_state_gate,
    run_reset_stability_diagnostic,
    run_state_operations_gate,
)
from psa.state.checkpoint import run_restore_probe
from psa.supplemental import (
    build_exp001b_run_preflight,
    build_exp001b_set_preflight,
    build_exp001b_preregistration_candidate,
    finalize_exp001b_preregistration_package,
    generate_and_freeze_exp001b_supplemental_set,
    run_exp001b_bdev1_gate,
    run_exp001b_bdev2_gate,
    run_exp001b_runner_development_gate,
    run_exp001b_supplemental,
    run_exp001b_supplemental_analysis,
    run_exp001b_posthoc_diagnostics,
    verify_exp001b_final_preregistration_package,
    verify_exp001b_supplemental_set_package,
    verify_exp001b_supplemental_raw_package,
)
from psa.tasks import generate_dataset
from psa.validation import validate_dataset


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _parse_pair(value: str) -> tuple[str, str]:
    parts = tuple(item.strip() for item in value.split(","))
    if len(parts) != 2 or any(not item for item in parts):
        raise argparse.ArgumentTypeError("expected two comma-separated labels")
    return parts[0], parts[1]


def _configured_gate_name(path: str | Path, fallback: str) -> str:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    gate = value.get("gate") if isinstance(value, dict) else None
    return gate if isinstance(gate, str) and gate else fallback


def _task_generate(args: argparse.Namespace) -> int:
    config: dict[str, Any] = {}
    config_provenance: dict[str, str] | None = None
    if args.config:
        config_path = Path(args.config)
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("task config must be a JSON object")
        config = loaded
        config_provenance = {
            "filename": config_path.name,
            "sha256": sha256_file(config_path),
        }

    identity_label_pairs = tuple(
        tuple(pair) for pair in config.get("identity_label_pairs", (args.identity_labels,))
    )
    goal_label_pairs = tuple(
        tuple(pair) for pair in config.get("goal_label_pairs", (args.goal_labels,))
    )
    groups = generate_dataset(
        group_count=int(config.get("group_count", args.count)),
        base_seed=int(config.get("base_seed", args.base_seed)),
        track=str(config.get("track", args.track)),
        identity_label_pairs=identity_label_pairs,
        goal_label_pairs=goal_label_pairs,
        answer_codes=tuple(config.get("answer_codes", args.answer_codes)),
        delay_units=int(config.get("delay_units", args.delay_units)),
        generator_version=str(config.get("generator_version", "0.1")),
    )
    report = validate_dataset(groups)
    if not report.valid:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 2

    group_payloads = [group.to_dict() for group in groups]
    dataset = {
        "dataset_version": "0.1",
        "track": str(config.get("track", args.track)),
        "group_count": len(groups),
        "base_seed": int(config.get("base_seed", args.base_seed)),
        "source_config": config_provenance,
        "groups": group_payloads,
        "validation": report.to_dict(),
    }
    dataset["dataset_digest_sha256"] = sha256_json(group_payloads)
    _write_json(Path(args.output), dataset)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "group_count": len(groups),
                "dataset_digest_sha256": dataset["dataset_digest_sha256"],
                "warnings": list(report.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_combo(value: str) -> tuple[int, int]:
    left, right = value.split(",", maxsplit=1)
    combo = int(left), int(right)
    if combo not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError(f"invalid combo key: {value}")
    return combo


def _stats_contrast(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    states = {
        _parse_combo(state_key): {
            _parse_combo(option_key): float(score)
            for option_key, score in option_scores.items()
        }
        for state_key, option_scores in payload["states"].items()
    }
    result = group_contrasts(states)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _asset_selection(args: argparse.Namespace) -> tuple[str, ...] | None:
    return tuple(args.only) if args.only else None


def _assets_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    result = plan_manifest(
        manifest,
        root=args.root,
        selected_ids=_asset_selection(args),
        hf_endpoint=args.hf_endpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _assets_fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    result = fetch_manifest(
        manifest,
        root=args.root,
        selected_ids=_asset_selection(args),
        retries=args.retries,
        timeout=args.timeout,
        hf_endpoint=args.hf_endpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _assets_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    result = verify_manifest(
        manifest,
        root=args.root,
        selected_ids=_asset_selection(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _environment_report(args: argparse.Namespace) -> int:
    result = collect_environment(args.project_root)
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _model_interface_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_interface_gate(
            config_path=args.config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": "impl1_model_interface",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _checkpoint_roundtrip_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_checkpoint_roundtrip_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": _configured_gate_name(
                args.gate_config, "impl2_checkpoint_roundtrip"
            ),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _checkpoint_restore_probe(args: argparse.Namespace) -> int:
    result = run_restore_probe(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        reference_path=args.reference,
        probe_config_path=args.probe_config,
        output_path=args.output,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "repeat_count": result["repeat_count"],
                "exact_repeat_count": result["exact_repeat_count"],
                "tolerance_pass_count": result["tolerance_pass_count"],
                "top1_match_count": result["top1_match_count"],
                "achieved_level": result["achieved_level"],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["valid"] else 2


def _state_operations_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_state_operations_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": _configured_gate_name(
                args.gate_config, "impl2b_state_operations"
            ),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _random_state_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_random_state_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": _configured_gate_name(
                args.gate_config, "impl2c_random_matched"
            ),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _reset_stability_diagnostic(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_reset_stability_diagnostic(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": _configured_gate_name(
                args.gate_config, "impl3na_g1h_2_9b_reset_stability"
            ),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _impl3_development_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_impl3_development_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": "impl3_development",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _capability_ladder_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_capability_ladder_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": "impl3b_capability_ladder",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _g1_capability_ladder_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    gate_name = "impl3d_g1_capability_ladder"
    try:
        gate_payload = json.loads(
            Path(args.gate_config).read_text(encoding="utf-8")
        )
        if isinstance(gate_payload, dict) and isinstance(
            gate_payload.get("gate"),
            str,
        ):
            gate_name = gate_payload["gate"]
    except (OSError, ValueError):
        pass
    try:
        result = run_g1_capability_ladder_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": gate_name,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _g1_capability_audit(args: argparse.Namespace) -> int:
    result = run_g1_capability_audit(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _g1_code_rotation_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_g1_code_rotation_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": "impl3k_g1h_2_9b_code_rotation",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _g1_code_rotation_review(args: argparse.Namespace) -> int:
    result = run_g1_code_rotation_review(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _history_binding_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    gate_name = "impl3p_g1h_2_9b_history_binding"
    try:
        result = run_history_binding_gate(
            config_path=args.config,
            gate_config_path=args.gate_config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        failure = {
            "failure_version": "0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "gate": gate_name,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _formal_freeze_candidate_gate(args: argparse.Namespace) -> int:
    failure_path = Path(args.output_dir) / "failure_report.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = run_formal_freeze_candidate_gate(
            config_path=args.config,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except Exception as exc:
        gate_name = _configured_gate_name(
            args.config,
            "impl3q_exp001_formal_freeze_candidate",
        )
        failure = {
            "failure_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True,
            "confirmatory_results_observed": False,
            "core_set_generated": False,
            "gate": gate_name,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _preregistration_verify(args: argparse.Namespace) -> int:
    result = verify_preregistration_candidate(
        args.candidate,
        project_root=args.project_root,
    )
    if args.output:
        _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _formal_freeze_review(args: argparse.Namespace) -> int:
    result = run_formal_freeze_review(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _preregistration_finalize(args: argparse.Namespace) -> int:
    result = finalize_preregistration_package(
        candidate_path=args.candidate,
        verification_path=args.verification,
        confirmation_path=args.confirmation,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _preregistration_final_verify(args: argparse.Namespace) -> int:
    result = verify_final_preregistration_package(args.package_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _core_set_generate(args: argparse.Namespace) -> int:
    result = generate_and_freeze_core_set(
        final_package_dir=args.final_package,
        authorization_path=args.authorization,
        config_path=args.config,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _core_set_verify(args: argparse.Namespace) -> int:
    result = verify_core_set_package(args.package_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _confirmatory_preflight(args: argparse.Namespace) -> int:
    result = build_confirmatory_preflight(
        project_root=args.project_root,
        final_package_dir=args.final_package,
        core_set_package_dir=args.core_set_package,
        model_config_path=args.model_config,
        asset_manifest_path=args.asset_manifest,
        asset_root=args.asset_root,
        runner_evidence_path=args.runner_evidence,
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _confirmatory_runner_dev_gate(args: argparse.Namespace) -> int:
    result = run_confirmatory_runner_development_gate(
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_bdev1_gate(args: argparse.Namespace) -> int:
    result = run_exp001b_bdev1_gate(
        design_path=args.design,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_bdev2_gate(args: argparse.Namespace) -> int:
    result = run_exp001b_bdev2_gate(
        design_path=args.design,
        model_config_path=args.model_config,
        bdev1_summary_path=args.bdev1_summary,
        bdev1_thresholds_path=args.bdev1_thresholds,
        bdev1_matched_report_path=args.bdev1_matched_report,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_candidate_build(args: argparse.Namespace) -> int:
    result = build_exp001b_preregistration_candidate(
        design_path=args.design,
        bdev1_dir=args.bdev1_dir,
        bdev2_v01_dir=args.bdev2_v01_dir,
        bdev2_v02_dir=args.bdev2_v02_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_preregistration_finalize(args: argparse.Namespace) -> int:
    result = finalize_exp001b_preregistration_package(
        candidate_dir=args.candidate_dir,
        confirmation_text=args.confirmation_text,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_preregistration_verify(args: argparse.Namespace) -> int:
    result = verify_exp001b_final_preregistration_package(
        args.package_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_set_preflight(args: argparse.Namespace) -> int:
    result = build_exp001b_set_preflight(
        final_package_dir=args.final_package,
        core_set_package_dir=args.core_set_package,
        project_root=args.project_root,
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_set_generate(args: argparse.Namespace) -> int:
    execution_lock = os.environ.get("PSA_EXP001B_SET_GENERATE", "")
    if execution_lock != "AUTHORIZED_EXP001B_SET_GENERATION":
        raise PermissionError("EXP-001B supplemental-set execution lock is absent")
    root = Path(args.project_root).resolve()
    counter, provenance = _load_token_counter(
        {"model_config": str(Path(args.model_config).resolve())}, root
    )
    result = generate_and_freeze_exp001b_supplemental_set(
        final_package_dir=args.final_package,
        core_set_package_dir=args.core_set_package,
        authorization_path=args.authorization,
        formal_config_path=args.formal_config,
        output_dir=args.output_dir,
        token_counter=counter,
        tokenizer_provenance=provenance,
        execution_lock=execution_lock,
        project_root=root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_set_verify(args: argparse.Namespace) -> int:
    result = verify_exp001b_supplemental_set_package(args.package_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_runner_dev_gate(args: argparse.Namespace) -> int:
    result = run_exp001b_runner_development_gate(
        model_config_path=args.model_config,
        bdev1_thresholds_path=args.bdev1_thresholds,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_run_preflight(args: argparse.Namespace) -> int:
    result = build_exp001b_run_preflight(
        project_root=args.project_root,
        final_package_dir=args.final_package,
        core_set_package_dir=args.core_set_package,
        supplemental_set_package_dir=args.supplemental_set_package,
        model_config_path=args.model_config,
        asset_manifest_path=args.asset_manifest,
        asset_root=args.asset_root,
        runner_evidence_path=args.runner_evidence,
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_run(args: argparse.Namespace) -> int:
    if os.environ.get("PSA_EXP001B_RUN", "") != "AUTHORIZED_EXP001B_SUPPLEMENTAL_RUN":
        raise PermissionError("EXP-001B supplemental run execution lock is absent")
    result = run_exp001b_supplemental(
        project_root=args.project_root,
        final_package_dir=args.final_package,
        core_set_package_dir=args.core_set_package,
        supplemental_set_package_dir=args.supplemental_set_package,
        model_config_path=args.model_config,
        asset_manifest_path=args.asset_manifest,
        asset_root=args.asset_root,
        runner_evidence_path=args.runner_evidence,
        preflight_path=args.preflight,
        authorization_path=args.authorization,
        output_dir=args.output_dir,
        resume=args.resume,
        execution_lock=os.environ.get("PSA_EXP001B_RUN", ""),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_raw_verify(args: argparse.Namespace) -> int:
    result = verify_exp001b_supplemental_raw_package(
        output_dir=args.output_dir,
        core_set_package_dir=args.core_set_package,
        supplemental_set_package_dir=args.supplemental_set_package,
        preflight_path=args.preflight,
        authorization_path=args.authorization,
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_analyze(args: argparse.Namespace) -> int:
    result = run_exp001b_supplemental_analysis(
        parent_raw_output_dir=args.parent_raw_output_dir,
        parent_raw_verification_path=args.parent_raw_verification,
        supplemental_raw_output_dir=args.supplemental_raw_output_dir,
        supplemental_raw_verification_path=args.supplemental_raw_verification,
        core_set_package_dir=args.core_set_package,
        supplemental_set_package_dir=args.supplemental_set_package,
        analysis_config_path=args.analysis_config,
        analysis_output_dir=args.analysis_output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001b_diagnose(args: argparse.Namespace) -> int:
    result = run_exp001b_posthoc_diagnostics(
        supplemental_raw_output_dir=args.supplemental_raw_output_dir,
        supplemental_raw_verification_path=args.supplemental_raw_verification,
        supplemental_set_package_dir=args.supplemental_set_package,
        analysis_output_dir=args.analysis_output_dir,
        diagnostic_output_dir=args.diagnostic_output_dir,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001c_probe_manifest(args: argparse.Namespace) -> int:
    result = build_exp001c_probe_manifest(
        design_config_path=args.design,
        model_config_path=args.model_config,
        project_root=args.project_root,
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _exp001c_probe_verify(args: argparse.Namespace) -> int:
    result = verify_exp001c_probe_manifest(
        args.manifest,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _exp001c_probe_authority_check(args: argparse.Namespace) -> int:
    result = validate_exp001c_probe_execution_authority(
        manifest_path=args.manifest,
        authorization_path=args.authorization,
        execution_lock=os.environ.get("PSA_EXP001C_NONCORE_PILOT", ""),
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _confirmatory_run(args: argparse.Namespace) -> int:
    result = run_exp001_confirmatory(
        project_root=args.project_root,
        final_package_dir=args.final_package,
        core_set_package_dir=args.core_set_package,
        model_config_path=args.model_config,
        asset_manifest_path=args.asset_manifest,
        asset_root=args.asset_root,
        runner_evidence_path=args.runner_evidence,
        preflight_path=args.preflight,
        authorization_path=args.authorization,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _confirmatory_raw_verify(args: argparse.Namespace) -> int:
    result = verify_exp001_confirmatory_raw_package(
        output_dir=args.output_dir,
        core_set_package_dir=args.core_set_package,
        preflight_path=args.preflight,
        authorization_path=args.authorization,
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _confirmatory_analyze(args: argparse.Namespace) -> int:
    result = run_exp001_confirmatory_analysis(
        raw_output_dir=args.raw_output_dir,
        raw_verification_path=args.raw_verification,
        core_set_package_dir=args.core_set_package,
        final_package_dir=args.final_package,
        analysis_config_path=args.analysis_config,
        analysis_output_dir=args.analysis_output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _add_asset_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".psa-assets")
    parser.add_argument(
        "--only",
        action="append",
        help="process only one asset ID; repeat to select multiple assets",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    task_generate = subparsers.add_parser(
        "task-generate", help="generate a deterministic development task set"
    )
    task_generate.add_argument("--output", required=True)
    task_generate.add_argument(
        "--config", help="optional JSON config whose values override CLI defaults"
    )
    task_generate.add_argument("--count", type=int, default=8)
    task_generate.add_argument("--base-seed", type=int, default=20260729)
    task_generate.add_argument(
        "--track", choices=("synthetic", "natural"), default="synthetic"
    )
    task_generate.add_argument(
        "--identity-labels", type=_parse_pair, default=("dax", "kel")
    )
    task_generate.add_argument(
        "--goal-labels", type=_parse_pair, default=("mip", "rov")
    )
    task_generate.add_argument(
        "--answer-codes", nargs=4, default=("A", "B", "C", "D")
    )
    task_generate.add_argument("--delay-units", type=int, default=1)
    task_generate.set_defaults(handler=_task_generate)

    stats_contrast = subparsers.add_parser(
        "stats-contrast", help="compute E1/E2/E3 for one factorial group"
    )
    stats_contrast.add_argument("--input", required=True)
    stats_contrast.set_defaults(handler=_stats_contrast)

    assets_plan = subparsers.add_parser(
        "assets-plan", help="show pinned remote assets and local destinations"
    )
    _add_asset_common_arguments(assets_plan)
    assets_plan.add_argument("--hf-endpoint")
    assets_plan.set_defaults(handler=_assets_plan)

    assets_fetch = subparsers.add_parser(
        "assets-fetch", help="download pinned assets with resume and verification"
    )
    _add_asset_common_arguments(assets_fetch)
    assets_fetch.add_argument("--hf-endpoint")
    assets_fetch.add_argument("--retries", type=int, default=3)
    assets_fetch.add_argument("--timeout", type=float, default=60.0)
    assets_fetch.set_defaults(handler=_assets_fetch)

    assets_verify = subparsers.add_parser(
        "assets-verify", help="verify downloaded assets against the manifest"
    )
    _add_asset_common_arguments(assets_verify)
    assets_verify.set_defaults(handler=_assets_verify)

    environment_report = subparsers.add_parser(
        "environment-report", help="record and validate the Impl-1 GPU environment"
    )
    environment_report.add_argument("--output", required=True)
    environment_report.add_argument("--project-root", default=".")
    environment_report.set_defaults(handler=_environment_report)

    model_interface_gate = subparsers.add_parser(
        "model-interface-gate",
        help="load RWKV-7 and run tokenizer, state inventory, and memory roundtrip checks",
    )
    model_interface_gate.add_argument("--config", required=True)
    model_interface_gate.add_argument("--output-dir", required=True)
    model_interface_gate.add_argument("--project-root", default=".")
    model_interface_gate.set_defaults(handler=_model_interface_gate)

    checkpoint_roundtrip_gate = subparsers.add_parser(
        "checkpoint-roundtrip-gate",
        help="save native state and validate 100 disk restores in a child process",
    )
    checkpoint_roundtrip_gate.add_argument("--config", required=True)
    checkpoint_roundtrip_gate.add_argument("--gate-config", required=True)
    checkpoint_roundtrip_gate.add_argument("--output-dir", required=True)
    checkpoint_roundtrip_gate.add_argument("--project-root", default=".")
    checkpoint_roundtrip_gate.set_defaults(handler=_checkpoint_roundtrip_gate)

    checkpoint_restore_probe = subparsers.add_parser(
        "checkpoint-restore-probe",
        help="internal child-process restore probe used by the Impl-2 gate",
    )
    checkpoint_restore_probe.add_argument("--config", required=True)
    checkpoint_restore_probe.add_argument("--checkpoint", required=True)
    checkpoint_restore_probe.add_argument("--reference", required=True)
    checkpoint_restore_probe.add_argument("--probe-config", required=True)
    checkpoint_restore_probe.add_argument("--output", required=True)
    checkpoint_restore_probe.add_argument("--project-root", default=".")
    checkpoint_restore_probe.set_defaults(handler=_checkpoint_restore_probe)

    state_operations_gate = subparsers.add_parser(
        "state-operations-gate",
        help="validate official reset, state diff, and immutable full-state swap",
    )
    state_operations_gate.add_argument("--config", required=True)
    state_operations_gate.add_argument("--gate-config", required=True)
    state_operations_gate.add_argument("--output-dir", required=True)
    state_operations_gate.add_argument("--project-root", default=".")
    state_operations_gate.set_defaults(handler=_state_operations_gate)

    reset_stability_diagnostic = subparsers.add_parser(
        "reset-stability-diagnostic",
        help="compare the first official reset call with stabilized later calls",
    )
    reset_stability_diagnostic.add_argument("--config", required=True)
    reset_stability_diagnostic.add_argument("--gate-config", required=True)
    reset_stability_diagnostic.add_argument("--output-dir", required=True)
    reset_stability_diagnostic.add_argument("--project-root", default=".")
    reset_stability_diagnostic.set_defaults(
        handler=_reset_stability_diagnostic
    )

    random_state_gate = subparsers.add_parser(
        "random-state-gate",
        help="validate deterministic per-component scale-matched random state",
    )
    random_state_gate.add_argument("--config", required=True)
    random_state_gate.add_argument("--gate-config", required=True)
    random_state_gate.add_argument("--output-dir", required=True)
    random_state_gate.add_argument("--project-root", default=".")
    random_state_gate.set_defaults(handler=_random_state_gate)

    impl3_development_gate = subparsers.add_parser(
        "impl3-development-gate",
        help="validate Batch 0 evidence and run the Batch 1 Prompt-visible dry run",
    )
    impl3_development_gate.add_argument("--config", required=True)
    impl3_development_gate.add_argument("--gate-config", required=True)
    impl3_development_gate.add_argument("--output-dir", required=True)
    impl3_development_gate.add_argument("--project-root", default=".")
    impl3_development_gate.set_defaults(handler=_impl3_development_gate)

    capability_ladder_gate = subparsers.add_parser(
        "capability-ladder-gate",
        help="diagnose copy, single-field, and two-field task capability",
    )
    capability_ladder_gate.add_argument("--config", required=True)
    capability_ladder_gate.add_argument("--gate-config", required=True)
    capability_ladder_gate.add_argument("--output-dir", required=True)
    capability_ladder_gate.add_argument("--project-root", default=".")
    capability_ladder_gate.set_defaults(handler=_capability_ladder_gate)

    g1_capability_ladder_gate = subparsers.add_parser(
        "g1-capability-ladder-gate",
        help="run live copy, single-field, and two-field checks with G1 prompts",
    )
    g1_capability_ladder_gate.add_argument("--config", required=True)
    g1_capability_ladder_gate.add_argument("--gate-config", required=True)
    g1_capability_ladder_gate.add_argument("--output-dir", required=True)
    g1_capability_ladder_gate.add_argument("--project-root", default=".")
    g1_capability_ladder_gate.set_defaults(handler=_g1_capability_ladder_gate)

    g1_capability_audit = subparsers.add_parser(
        "g1-capability-audit",
        help="audit G1 output variants, confusion matrices, and scoring errors",
    )
    g1_capability_audit.add_argument("--output-dir", required=True)
    g1_capability_audit.set_defaults(handler=_g1_capability_audit)

    g1_code_rotation_gate = subparsers.add_parser(
        "g1-code-rotation-gate",
        help="rotate A-D over identical two-field semantic cases",
    )
    g1_code_rotation_gate.add_argument("--config", required=True)
    g1_code_rotation_gate.add_argument("--gate-config", required=True)
    g1_code_rotation_gate.add_argument("--output-dir", required=True)
    g1_code_rotation_gate.add_argument("--project-root", default=".")
    g1_code_rotation_gate.set_defaults(handler=_g1_code_rotation_gate)

    g1_code_rotation_review = subparsers.add_parser(
        "g1-code-rotation-review",
        help="marginalize A-D scores in an existing code-rotation run",
    )
    g1_code_rotation_review.add_argument("--output-dir", required=True)
    g1_code_rotation_review.set_defaults(
        handler=_g1_code_rotation_review
    )

    history_binding_gate = subparsers.add_parser(
        "history-binding-gate",
        help="compare predeclared G1h recurrent-state history binding modes",
    )
    history_binding_gate.add_argument("--config", required=True)
    history_binding_gate.add_argument("--gate-config", required=True)
    history_binding_gate.add_argument("--output-dir", required=True)
    history_binding_gate.add_argument("--project-root", default=".")
    history_binding_gate.set_defaults(handler=_history_binding_gate)

    formal_freeze_candidate_gate = subparsers.add_parser(
        "formal-freeze-candidate-gate",
        help=(
            "qualify formal prompt-visible templates and controls, simulate "
            "power, and build a preregistration checksum candidate"
        ),
    )
    formal_freeze_candidate_gate.add_argument("--config", required=True)
    formal_freeze_candidate_gate.add_argument("--output-dir", required=True)
    formal_freeze_candidate_gate.add_argument(
        "--project-root",
        default=".",
    )
    formal_freeze_candidate_gate.set_defaults(
        handler=_formal_freeze_candidate_gate
    )

    preregistration_verify = subparsers.add_parser(
        "preregistration-verify",
        help="verify a preregistration candidate and all locked file digests",
    )
    preregistration_verify.add_argument("--candidate", required=True)
    preregistration_verify.add_argument("--project-root", default=".")
    preregistration_verify.add_argument("--output")
    preregistration_verify.set_defaults(
        handler=_preregistration_verify
    )

    preregistration_finalize = subparsers.add_parser(
        "preregistration-finalize",
        help=(
            "freeze a verified candidate plus project-owner confirmation "
            "without generating a Core Set"
        ),
    )
    preregistration_finalize.add_argument("--candidate", required=True)
    preregistration_finalize.add_argument("--verification", required=True)
    preregistration_finalize.add_argument("--confirmation", required=True)
    preregistration_finalize.add_argument("--output-dir", required=True)
    preregistration_finalize.set_defaults(
        handler=_preregistration_finalize
    )

    preregistration_final_verify = subparsers.add_parser(
        "preregistration-final-verify",
        help="verify a frozen final preregistration package",
    )
    preregistration_final_verify.add_argument(
        "--package-dir",
        required=True,
    )
    preregistration_final_verify.set_defaults(
        handler=_preregistration_final_verify
    )

    core_set_generate = subparsers.add_parser(
        "core-set-generate",
        help=(
            "generate and freeze the authorized EXP-001 Core Set without "
            "running the confirmatory experiment"
        ),
    )
    core_set_generate.add_argument("--final-package", required=True)
    core_set_generate.add_argument("--authorization", required=True)
    core_set_generate.add_argument("--config", required=True)
    core_set_generate.add_argument("--output-dir", required=True)
    core_set_generate.add_argument("--project-root", default=".")
    core_set_generate.set_defaults(handler=_core_set_generate)

    core_set_verify = subparsers.add_parser(
        "core-set-verify",
        help="verify a frozen unrun EXP-001 Core Set package",
    )
    core_set_verify.add_argument("--package-dir", required=True)
    core_set_verify.set_defaults(handler=_core_set_verify)

    confirmatory_preflight = subparsers.add_parser(
        "confirmatory-preflight",
        help=(
            "verify the frozen EXP-001 packages, assets, environment, and "
            "source digests without loading the model or scoring Core Set trials"
        ),
    )
    confirmatory_preflight.add_argument("--final-package", required=True)
    confirmatory_preflight.add_argument("--core-set-package", required=True)
    confirmatory_preflight.add_argument("--model-config", required=True)
    confirmatory_preflight.add_argument("--asset-manifest", required=True)
    confirmatory_preflight.add_argument(
        "--asset-root",
        default=".psa-assets",
    )
    confirmatory_preflight.add_argument("--output", required=True)
    confirmatory_preflight.add_argument("--runner-evidence")
    confirmatory_preflight.add_argument("--project-root", default=".")
    confirmatory_preflight.set_defaults(handler=_confirmatory_preflight)

    confirmatory_runner_dev = subparsers.add_parser(
        "confirmatory-runner-dev-gate",
        help=(
            "exercise all runner conditions with one explicit non-Core "
            "development fixture"
        ),
    )
    confirmatory_runner_dev.add_argument("--model-config", required=True)
    confirmatory_runner_dev.add_argument("--output-dir", required=True)
    confirmatory_runner_dev.add_argument("--project-root", default=".")
    confirmatory_runner_dev.set_defaults(handler=_confirmatory_runner_dev_gate)

    exp001b_bdev1 = subparsers.add_parser(
        "exp001b-bdev1-gate",
        help=(
            "calibrate non-Core matched-context token pairing and state norms "
            "without generating or scoring the EXP-001B supplemental set"
        ),
    )
    exp001b_bdev1.add_argument("--design", required=True)
    exp001b_bdev1.add_argument("--model-config", required=True)
    exp001b_bdev1.add_argument("--output-dir", required=True)
    exp001b_bdev1.add_argument("--project-root", default=".")
    exp001b_bdev1.set_defaults(handler=_exp001b_bdev1_gate)

    exp001b_bdev2 = subparsers.add_parser(
        "exp001b-bdev2-gate",
        help=(
            "exercise the EXP-001B non-Core condition runner, matched-context "
            "probe, generated-format probe, and state norm alert path"
        ),
    )
    exp001b_bdev2.add_argument("--design", required=True)
    exp001b_bdev2.add_argument("--model-config", required=True)
    exp001b_bdev2.add_argument("--bdev1-summary", required=True)
    exp001b_bdev2.add_argument("--bdev1-thresholds", required=True)
    exp001b_bdev2.add_argument("--bdev1-matched-report", required=True)
    exp001b_bdev2.add_argument("--output-dir", required=True)
    exp001b_bdev2.add_argument("--project-root", default=".")
    exp001b_bdev2.set_defaults(handler=_exp001b_bdev2_gate)

    exp001b_candidate = subparsers.add_parser(
        "exp001b-candidate-build",
        help=(
            "build an unconfirmed EXP-001B preregistration checksum candidate "
            "without reading Core Set data or authorizing a formal run"
        ),
    )
    exp001b_candidate.add_argument("--design", required=True)
    exp001b_candidate.add_argument("--bdev1-dir", required=True)
    exp001b_candidate.add_argument("--bdev2-v01-dir", required=True)
    exp001b_candidate.add_argument("--bdev2-v02-dir", required=True)
    exp001b_candidate.add_argument("--output-dir", required=True)
    exp001b_candidate.add_argument("--project-root", default=".")
    exp001b_candidate.set_defaults(handler=_exp001b_candidate_build)

    exp001b_finalize = subparsers.add_parser(
        "exp001b-preregistration-finalize",
        help=(
            "upgrade the exactly confirmed EXP-001B checksum candidate to a "
            "final preregistration package without generating a supplemental set"
        ),
    )
    exp001b_finalize.add_argument("--candidate-dir", required=True)
    exp001b_finalize.add_argument("--confirmation-text", required=True)
    exp001b_finalize.add_argument("--output-dir", required=True)
    exp001b_finalize.add_argument("--project-root", default=".")
    exp001b_finalize.set_defaults(handler=_exp001b_preregistration_finalize)

    exp001b_final_verify = subparsers.add_parser(
        "exp001b-preregistration-final-verify",
        help="independently verify the frozen EXP-001B preregistration package",
    )
    exp001b_final_verify.add_argument("--package-dir", required=True)
    exp001b_final_verify.add_argument("--project-root", default=".")
    exp001b_final_verify.set_defaults(handler=_exp001b_preregistration_verify)

    exp001b_set_preflight = subparsers.add_parser(
        "exp001b-set-preflight",
        help=(
            "verify the frozen EXP-001B and parent Core Set packages without "
            "loading a model, generating a supplemental set, or scoring a trial"
        ),
    )
    exp001b_set_preflight.add_argument("--final-package", required=True)
    exp001b_set_preflight.add_argument("--core-set-package", required=True)
    exp001b_set_preflight.add_argument("--output", required=True)
    exp001b_set_preflight.add_argument("--project-root", default=".")
    exp001b_set_preflight.set_defaults(handler=_exp001b_set_preflight)

    exp001b_set_generate = subparsers.add_parser(
        "exp001b-set-generate",
        help=(
            "deterministically generate and freeze the 11,008-record EXP-001B "
            "supplemental set only with exact owner authorization and execution lock"
        ),
    )
    exp001b_set_generate.add_argument("--final-package", required=True)
    exp001b_set_generate.add_argument("--core-set-package", required=True)
    exp001b_set_generate.add_argument("--authorization", required=True)
    exp001b_set_generate.add_argument("--formal-config", required=True)
    exp001b_set_generate.add_argument("--model-config", required=True)
    exp001b_set_generate.add_argument("--output-dir", required=True)
    exp001b_set_generate.add_argument("--project-root", default=".")
    exp001b_set_generate.set_defaults(handler=_exp001b_set_generate)

    exp001b_set_verify = subparsers.add_parser(
        "exp001b-set-verify",
        help="independently verify a frozen, unrun EXP-001B supplemental-set package",
    )
    exp001b_set_verify.add_argument("--package-dir", required=True)
    exp001b_set_verify.set_defaults(handler=_exp001b_set_verify)

    exp001b_runner_dev = subparsers.add_parser(
        "exp001b-runner-dev-gate",
        help=(
            "exercise the exact EXP-001B formal record router with an explicit "
            "non-Core fixture and without reading the frozen supplemental set"
        ),
    )
    exp001b_runner_dev.add_argument("--model-config", required=True)
    exp001b_runner_dev.add_argument("--bdev1-thresholds", required=True)
    exp001b_runner_dev.add_argument("--output-dir", required=True)
    exp001b_runner_dev.add_argument("--project-root", default=".")
    exp001b_runner_dev.set_defaults(handler=_exp001b_runner_dev_gate)

    exp001b_run_preflight = subparsers.add_parser(
        "exp001b-run-preflight",
        help=(
            "verify the frozen EXP-001B packages, runner evidence, assets, "
            "environment, and source digests without loading the model"
        ),
    )
    exp001b_run_preflight.add_argument("--final-package", required=True)
    exp001b_run_preflight.add_argument("--core-set-package", required=True)
    exp001b_run_preflight.add_argument("--supplemental-set-package", required=True)
    exp001b_run_preflight.add_argument("--model-config", required=True)
    exp001b_run_preflight.add_argument("--asset-manifest", required=True)
    exp001b_run_preflight.add_argument("--asset-root", default=".psa-assets")
    exp001b_run_preflight.add_argument("--runner-evidence")
    exp001b_run_preflight.add_argument("--output", required=True)
    exp001b_run_preflight.add_argument("--project-root", default=".")
    exp001b_run_preflight.set_defaults(handler=_exp001b_run_preflight)

    exp001b_run = subparsers.add_parser(
        "exp001b-run",
        help=(
            "run the complete frozen EXP-001B supplemental experiment only "
            "after exact project-owner authorization"
        ),
    )
    exp001b_run.add_argument("--final-package", required=True)
    exp001b_run.add_argument("--core-set-package", required=True)
    exp001b_run.add_argument("--supplemental-set-package", required=True)
    exp001b_run.add_argument("--model-config", required=True)
    exp001b_run.add_argument("--asset-manifest", required=True)
    exp001b_run.add_argument("--asset-root", default=".psa-assets")
    exp001b_run.add_argument("--runner-evidence", required=True)
    exp001b_run.add_argument("--preflight", required=True)
    exp001b_run.add_argument("--authorization", required=True)
    exp001b_run.add_argument("--output-dir", required=True)
    exp001b_run.add_argument("--resume", action="store_true")
    exp001b_run.add_argument("--project-root", default=".")
    exp001b_run.set_defaults(handler=_exp001b_run)

    exp001b_raw_verify = subparsers.add_parser(
        "exp001b-raw-verify",
        help=(
            "verify the complete EXP-001B raw package without deriving or "
            "reporting research metrics"
        ),
    )
    exp001b_raw_verify.add_argument("--output-dir", required=True)
    exp001b_raw_verify.add_argument("--core-set-package", required=True)
    exp001b_raw_verify.add_argument("--supplemental-set-package", required=True)
    exp001b_raw_verify.add_argument("--preflight", required=True)
    exp001b_raw_verify.add_argument("--authorization", required=True)
    exp001b_raw_verify.add_argument("--output", required=True)
    exp001b_raw_verify.set_defaults(handler=_exp001b_raw_verify)

    exp001b_analyze = subparsers.add_parser(
        "exp001b-analyze",
        help=(
            "run the pinned read-only EXP-001B supplemental analysis after exact "
            "raw-package verification"
        ),
    )
    exp001b_analyze.add_argument("--parent-raw-output-dir")
    exp001b_analyze.add_argument("--parent-raw-verification")
    exp001b_analyze.add_argument("--supplemental-raw-output-dir", required=True)
    exp001b_analyze.add_argument("--supplemental-raw-verification", required=True)
    exp001b_analyze.add_argument("--core-set-package", required=True)
    exp001b_analyze.add_argument("--supplemental-set-package", required=True)
    exp001b_analyze.add_argument("--analysis-config", required=True)
    exp001b_analyze.add_argument("--analysis-output-dir", required=True)
    exp001b_analyze.add_argument("--project-root", default=".")
    exp001b_analyze.set_defaults(handler=_exp001b_analyze)

    exp001b_diagnose = subparsers.add_parser(
        "exp001b-diagnose",
        help="run read-only post-hoc diagnostics without changing confirmatory decisions",
    )
    exp001b_diagnose.add_argument("--supplemental-raw-output-dir", required=True)
    exp001b_diagnose.add_argument("--supplemental-raw-verification", required=True)
    exp001b_diagnose.add_argument("--supplemental-set-package", required=True)
    exp001b_diagnose.add_argument("--analysis-output-dir", required=True)
    exp001b_diagnose.add_argument("--diagnostic-output-dir", required=True)
    exp001b_diagnose.add_argument("--tokenizer", required=True)
    exp001b_diagnose.set_defaults(handler=_exp001b_diagnose)

    exp001c_probe_manifest = subparsers.add_parser(
        "exp001c-probe-manifest",
        help=(
            "build an unrun EXP-001C development-probe manifest without "
            "loading a model"
        ),
    )
    exp001c_probe_manifest.add_argument("--design", required=True)
    exp001c_probe_manifest.add_argument("--model-config", required=True)
    exp001c_probe_manifest.add_argument("--output", required=True)
    exp001c_probe_manifest.add_argument("--project-root", default=".")
    exp001c_probe_manifest.set_defaults(handler=_exp001c_probe_manifest)

    exp001c_probe_verify = subparsers.add_parser(
        "exp001c-probe-verify",
        help="verify an unrun EXP-001C development-probe manifest",
    )
    exp001c_probe_verify.add_argument("--manifest", required=True)
    exp001c_probe_verify.add_argument("--project-root", default=".")
    exp001c_probe_verify.set_defaults(handler=_exp001c_probe_verify)

    exp001c_probe_authority = subparsers.add_parser(
        "exp001c-probe-authority-check",
        help=(
            "validate the separate EXP-001C non-Core pilot authorization and "
            "execution lock without loading a model"
        ),
    )
    exp001c_probe_authority.add_argument("--manifest", required=True)
    exp001c_probe_authority.add_argument("--authorization", required=True)
    exp001c_probe_authority.add_argument("--project-root", default=".")
    exp001c_probe_authority.set_defaults(
        handler=_exp001c_probe_authority_check
    )

    confirmatory_run = subparsers.add_parser(
        "confirmatory-run",
        help=(
            "run the complete frozen EXP-001 Core Set only after exact "
            "project-owner authorization"
        ),
    )
    confirmatory_run.add_argument("--final-package", required=True)
    confirmatory_run.add_argument("--core-set-package", required=True)
    confirmatory_run.add_argument("--model-config", required=True)
    confirmatory_run.add_argument("--asset-manifest", required=True)
    confirmatory_run.add_argument("--asset-root", default=".psa-assets")
    confirmatory_run.add_argument("--runner-evidence", required=True)
    confirmatory_run.add_argument("--preflight", required=True)
    confirmatory_run.add_argument("--authorization", required=True)
    confirmatory_run.add_argument("--output-dir", required=True)
    confirmatory_run.add_argument("--resume", action="store_true")
    confirmatory_run.add_argument("--project-root", default=".")
    confirmatory_run.set_defaults(handler=_confirmatory_run)

    confirmatory_raw_verify = subparsers.add_parser(
        "confirmatory-raw-verify",
        help=(
            "verify the complete EXP-001 raw package without deriving or "
            "reporting research metrics"
        ),
    )
    confirmatory_raw_verify.add_argument("--output-dir", required=True)
    confirmatory_raw_verify.add_argument("--core-set-package", required=True)
    confirmatory_raw_verify.add_argument("--preflight", required=True)
    confirmatory_raw_verify.add_argument("--authorization", required=True)
    confirmatory_raw_verify.add_argument("--output", required=True)
    confirmatory_raw_verify.set_defaults(handler=_confirmatory_raw_verify)

    confirmatory_analyze = subparsers.add_parser(
        "confirmatory-analyze",
        help=(
            "apply the pinned read-only EXP-001 analysis only after the full "
            "raw package has passed independent verification"
        ),
    )
    confirmatory_analyze.add_argument("--raw-output-dir", required=True)
    confirmatory_analyze.add_argument("--raw-verification", required=True)
    confirmatory_analyze.add_argument("--core-set-package", required=True)
    confirmatory_analyze.add_argument("--final-package", required=True)
    confirmatory_analyze.add_argument("--analysis-config", required=True)
    confirmatory_analyze.add_argument("--analysis-output-dir", required=True)
    confirmatory_analyze.add_argument("--project-root", default=".")
    confirmatory_analyze.set_defaults(handler=_confirmatory_analyze)

    formal_freeze_review = subparsers.add_parser(
        "formal-freeze-review",
        help=(
            "audit a formal freeze hold without rerunning the model or reading "
            "confirmatory results"
        ),
    )
    formal_freeze_review.add_argument("--output-dir", required=True)
    formal_freeze_review.set_defaults(handler=_formal_freeze_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
