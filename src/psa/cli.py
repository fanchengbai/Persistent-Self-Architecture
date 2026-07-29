from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from psa.assets import fetch_manifest, load_manifest, plan_manifest, verify_manifest
from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.environment import collect_environment
from psa.evaluation import group_contrasts
from psa.model import run_interface_gate
from psa.state import (
    run_checkpoint_roundtrip_gate,
    run_random_state_gate,
    run_state_operations_gate,
)
from psa.state.checkpoint import run_restore_probe
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
            "gate": "impl2_checkpoint_roundtrip",
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
            "gate": "impl2b_state_operations",
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
            "gate": "impl2c_random_matched",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "config": str(Path(args.config).resolve()),
            "gate_config": str(Path(args.gate_config).resolve()),
        }
        _write_json(failure_path, failure)
        raise
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

    random_state_gate = subparsers.add_parser(
        "random-state-gate",
        help="validate deterministic per-component scale-matched random state",
    )
    random_state_gate.add_argument("--config", required=True)
    random_state_gate.add_argument("--gate-config", required=True)
    random_state_gate.add_argument("--output-dir", required=True)
    random_state_gate.add_argument("--project-root", default=".")
    random_state_gate.set_defaults(handler=_random_state_gate)
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
