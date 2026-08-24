from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.self_model.d4b_real_off_equivalence import write_json_exclusive
from psa.self_model.d6c_real_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    EXECUTION_LOCK_ENV,
    EXECUTION_LOCK_VALUE,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    build_d6c_authorization,
    run_d6c_real_persistent_mechanism,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one future-authorized D6C persistent mechanism validation"
    )
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
        raise PermissionError("D6C config path is not frozen")
    if authorization_path != (root / AUTHORIZATION_RELATIVE_PATH).resolve():
        raise PermissionError("D6C authorization path is not frozen")
    if output_dir != (root / OUTPUT_RELATIVE_DIR).resolve():
        raise PermissionError("D6C output path is not frozen")
    if authorization_path.exists():
        raise FileExistsError("D6C machine authorization already exists")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("D6C output directory is not empty")
    authorization = build_d6c_authorization(
        config_path=config_path, project_root=root,
        authorization_text=FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    )
    write_json_exclusive(authorization_path, authorization)
    report = run_d6c_real_persistent_mechanism(
        config_path=config_path, authorization_path=authorization_path,
        project_root=root, output_dir=output_dir,
    )
    print(json.dumps({
        "status": report["status"], "valid": report["valid"],
        "decision_effect": report["decision_effect"],
        "report_digest_sha256": report["report_digest_sha256"],
        "execution_claim_sha256": report["execution_claim_sha256"],
        "runtime_seconds": report["runtime_seconds"],
        "cuda_peak_memory_bytes": report["cuda_peak_memory_bytes"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
