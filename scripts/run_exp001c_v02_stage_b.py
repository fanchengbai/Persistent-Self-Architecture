from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.development.exp001c_v02_stage_b_preflight import (
    STAGE_B_AUTHORIZATION_TEXT,
    build_exp001c_v02_stage_b_authorization,
)
from psa.development.exp001c_v02_stage_b_run import (
    STAGE_B_EXECUTION_LOCK,
    run_exp001c_v02_stage_b,
)
from psa.development.exp001c_v02_stage_b_rwkv import (
    build_exp001c_v02_stage_b_rwkv_backend,
)


EXECUTION_ENVIRONMENT_VARIABLE = "PSA_EXP001C_V02_STAGE_B_EXECUTION"


def _exclusive_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Consume the exact EXP-001C v02 Stage B authorization and run "
            "the 224-record non-Core pilot once."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--design-manifest",
        default=(
            "results/development/exp001c_v02_stage_b_preflight_v03/"
            "design_manifest.json"
        ),
    )
    parser.add_argument(
        "--preflight",
        default=(
            "results/development/exp001c_v02_stage_b_preflight_v03/"
            "preflight.json"
        ),
    )
    parser.add_argument(
        "--authorization",
        default="results/authorizations/exp001c_v02_stage_b_pilot_v01.json",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/rwkv7_g1h_2.9b.candidate.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/development/exp001c_v02_stage_b_pilot_v01",
    )
    arguments = parser.parse_args()

    execution_lock = os.environ.get(EXECUTION_ENVIRONMENT_VARIABLE, "")
    if execution_lock != STAGE_B_EXECUTION_LOCK:
        raise PermissionError(
            f"{EXECUTION_ENVIRONMENT_VARIABLE} single-use lock is absent"
        )
    root = Path(arguments.project_root).resolve()
    authorization_path = (root / arguments.authorization).resolve()
    output_dir = (root / arguments.output_dir).resolve()
    if authorization_path.exists():
        raise FileExistsError("Stage B authorization already exists")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Stage B output directory is not empty")

    authorization = build_exp001c_v02_stage_b_authorization(
        design_manifest_path=arguments.design_manifest,
        preflight_path=arguments.preflight,
        model_config_path=arguments.model_config,
        authorization_text=STAGE_B_AUTHORIZATION_TEXT,
        project_root=root,
    )
    _exclusive_write(authorization_path, authorization)

    summary = run_exp001c_v02_stage_b(
        design_manifest_path=arguments.design_manifest,
        preflight_path=arguments.preflight,
        authorization_path=authorization_path,
        model_config_path=arguments.model_config,
        output_dir=output_dir,
        backend_factory=lambda authority_validated: (
            build_exp001c_v02_stage_b_rwkv_backend(
                design_manifest_path=arguments.design_manifest,
                model_config_path=arguments.model_config,
                project_root=root,
                execution_authority_validated=authority_validated,
            )
        ),
        execution_lock=execution_lock,
        project_root=root,
    )
    print(
        json.dumps(
            {
                **summary,
                "authorization_digest_sha256": authorization[
                    "authorization_digest_sha256"
                ],
                "authorization_path": str(authorization_path),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
