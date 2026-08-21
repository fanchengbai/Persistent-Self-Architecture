from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.self_model.d4b_real_off_equivalence import write_json_exclusive
from psa.self_model.d5c_p1_real_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    EXECUTION_LOCK_ENV,
    EXECUTION_LOCK_VALUE,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    build_p1_authorization,
    run_p1_real_engineering_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the single-use D5C-P1 engineering validation")
    parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--authorization", default=AUTHORIZATION_RELATIVE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_RELATIVE_DIR)
    args = parser.parse_args()
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError(f"{EXECUTION_LOCK_ENV} single-use lock is absent")
    root = Path(args.project_root).resolve()
    config_path = (root / args.config).resolve()
    authorization_path = (root / args.authorization).resolve()
    output_dir = (root / args.output_dir).resolve()
    if config_path != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C-P1 config path is not frozen")
    if authorization_path != (root / AUTHORIZATION_RELATIVE_PATH).resolve():
        raise PermissionError("D5C-P1 authorization path is not frozen")
    if output_dir != (root / OUTPUT_RELATIVE_DIR).resolve():
        raise PermissionError("D5C-P1 output path is not frozen")
    if authorization_path.exists():
        raise FileExistsError("D5C-P1 machine authorization already exists")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("D5C-P1 output directory is not empty")
    authorization = build_p1_authorization(
        config_path=config_path, project_root=root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    )
    write_json_exclusive(authorization_path, authorization)
    report = run_p1_real_engineering_validation(
        config_path=config_path, authorization_path=authorization_path,
        project_root=root, output_dir=output_dir,
    )
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "decision_effect": report["decision_effect"],
        "report_digest_sha256": report["report_digest_sha256"],
        "execution_claim_sha256": report["execution_claim_sha256"],
        "runtime_seconds": report["runtime_seconds"],
        "cuda_peak_memory_bytes": report["cuda_peak_memory_bytes"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
