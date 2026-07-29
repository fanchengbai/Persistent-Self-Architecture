from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from psa.artifacts import canonical_json_bytes, sha256_json
from psa.evaluation import group_contrasts
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
    groups = generate_dataset(
        group_count=args.count,
        base_seed=args.base_seed,
        track=args.track,
        identity_label_pairs=(args.identity_labels,),
        goal_label_pairs=(args.goal_labels,),
        answer_codes=tuple(args.answer_codes),
        delay_units=args.delay_units,
    )
    report = validate_dataset(groups)
    if not report.valid:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 2

    group_payloads = [group.to_dict() for group in groups]
    dataset = {
        "dataset_version": "0.1",
        "track": args.track,
        "group_count": len(groups),
        "base_seed": args.base_seed,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    task_generate = subparsers.add_parser(
        "task-generate", help="generate a deterministic development task set"
    )
    task_generate.add_argument("--output", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

