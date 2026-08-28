from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from psa.self_model.d7c_real_compatibility_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    EXECUTION_LOCK_ENV,
    EXECUTION_LOCK_VALUE,
    OUTPUT_RELATIVE_DIR,
    _write_json_exclusive,
    build_d7c_authorization,
    run_d7c_real_compatibility,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one separately authorized D7-C 18-call compatibility gate"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--authorization-text", required=True)
    args = parser.parse_args()
    if os.environ.get(EXECUTION_LOCK_ENV) != EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D7-C execution lock is absent")
    root = Path(args.project_root).resolve()
    config = root / CONFIG_RELATIVE_PATH
    authorization_path = root / AUTHORIZATION_RELATIVE_PATH
    output_dir = root / OUTPUT_RELATIVE_DIR
    authorization = build_d7c_authorization(
        config_path=config,
        project_root=root,
        authorization_text=args.authorization_text,
    )
    _write_json_exclusive(authorization_path, authorization)
    print(json.dumps(authorization, ensure_ascii=False, sort_keys=True))
    report = run_d7c_real_compatibility(
        config_path=config,
        authorization_path=authorization_path,
        project_root=root,
        output_dir=output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
