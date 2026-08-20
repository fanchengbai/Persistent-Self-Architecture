from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.self_model.d4a_real_diagnostic import (
    D4A_EXECUTION_LOCK_ENV,
    D4A_EXECUTION_LOCK_VALUE,
    D4A_OWNER_AUTHORIZATION_TEXT,
    build_d4a_real_authorization,
    run_d4a_real_diagnostic,
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consume exact authority and run the single-use real D4A diagnostic"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if os.environ.get(D4A_EXECUTION_LOCK_ENV) != D4A_EXECUTION_LOCK_VALUE:
        raise PermissionError(f"{D4A_EXECUTION_LOCK_ENV} single-use lock is absent")
    root = Path(args.project_root).resolve()
    authorization_path = (root / args.authorization).resolve()
    output_dir = (root / args.output_dir).resolve()
    authorization_root = (root / "results/authorizations").resolve()
    development_root = (root / "results/development").resolve()
    if authorization_root not in authorization_path.parents:
        raise PermissionError("D4A authorization path must stay in results/authorizations")
    if development_root not in output_dir.parents:
        raise PermissionError("D4A output path must stay in results/development")
    if authorization_path.exists():
        raise FileExistsError("D4A machine authorization already exists")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("D4A output directory is not empty")

    authorization = build_d4a_real_authorization(
        config_path=args.config,
        project_root=root,
        authorization_text=D4A_OWNER_AUTHORIZATION_TEXT,
    )
    write_json_exclusive(authorization_path, authorization)
    report = run_d4a_real_diagnostic(
        config_path=args.config,
        authorization_path=authorization_path,
        project_root=root,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "diagnostic_classification": report["diagnostic"][
                    "diagnostic_classification"
                ],
                "report_digest_sha256": report["report_digest_sha256"],
                "execution_claim_sha256": report["execution_claim_sha256"],
                "runtime_seconds": report["runtime_seconds"],
                "cuda_peak_memory_bytes": report["cuda_peak_memory_bytes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
