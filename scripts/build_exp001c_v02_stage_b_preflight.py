from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.development.exp001c_v02_stage_b_design import (
    build_exp001c_v02_stage_b_design_manifest,
)
from psa.development.exp001c_v02_stage_b_preflight import (
    build_exp001c_v02_stage_b_preflight,
)


def _exclusive_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the read-only EXP-001C v02 Stage B live preflight. "
            "This verifies model files but never loads or executes the model."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--design-config",
        default="configs/development/exp001c_v02_stage_b_design.draft.json",
    )
    parser.add_argument(
        "--stage-a-result",
        default=(
            "results/development/exp001c_v02_stage_a_pilot_v01/"
            "stage_a_result.json"
        ),
    )
    parser.add_argument(
        "--stage-a-summary",
        default=(
            "results/development/exp001c_v02_stage_a_pilot_v01/summary.json"
        ),
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/rwkv7_g1h_2.9b.candidate.json",
    )
    parser.add_argument(
        "--stage-b-output",
        default="results/development/exp001c_v02_stage_b_pilot_v01",
    )
    parser.add_argument(
        "--evidence-dir",
        default="results/development/exp001c_v02_stage_b_preflight_v01",
    )
    arguments = parser.parse_args()

    root = Path(arguments.project_root).resolve()
    evidence_dir = (root / arguments.evidence_dir).resolve()
    design_path = evidence_dir / "design_manifest.json"
    preflight_path = evidence_dir / "preflight.json"
    if design_path.exists() or preflight_path.exists():
        raise FileExistsError(
            "Stage B preflight evidence already exists; refusing to overwrite"
        )

    design = build_exp001c_v02_stage_b_design_manifest(
        design_config_path=root / arguments.design_config,
        project_root=root,
    )
    _exclusive_write(design_path, design)
    preflight = build_exp001c_v02_stage_b_preflight(
        design_manifest_path=design_path,
        stage_a_result_path=arguments.stage_a_result,
        stage_a_summary_path=arguments.stage_a_summary,
        model_config_path=arguments.model_config,
        output_dir=arguments.stage_b_output,
        project_root=root,
    )
    _exclusive_write(preflight_path, preflight)
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "valid": preflight["valid"],
                "model_loaded": preflight["model_loaded"],
                "model_executed": preflight["model_executed"],
                "stage_b_execution_authorized": preflight[
                    "stage_b_execution_authorized"
                ],
                "preflight_digest_sha256": preflight[
                    "preflight_digest_sha256"
                ],
                "failed_checks": sorted(
                    key
                    for key, value in preflight["checks"].items()
                    if value is not True
                ),
                "design_manifest_path": str(design_path),
                "preflight_path": str(preflight_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if preflight["valid"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
