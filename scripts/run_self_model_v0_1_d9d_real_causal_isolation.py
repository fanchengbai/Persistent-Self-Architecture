from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d9c_real_entry import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    FUTURE_EXECUTION_AUTHORIZATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    build_d9_authorization,
    run_d9d_real_causal_isolation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create future D9-D authorization or consume it for one real joint run"
    )
    parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--authorization", default=AUTHORIZATION_RELATIVE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_RELATIVE_DIR)
    parser.add_argument("--create-authorization", action="store_true")
    parser.add_argument("--authorization-text")
    parser.add_argument("--entry-static-report-sha256")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    if args.create_authorization:
        if args.authorization_text != FUTURE_EXECUTION_AUTHORIZATION_TEXT:
            raise PermissionError("the exact future D9-D owner authorization is absent")
        if not args.entry_static_report_sha256:
            raise PermissionError("D9-C static report digest is required")
        payload = build_d9_authorization(
            project_root=root,
            authorization_text=args.authorization_text,
            entry_static_report_sha256=args.entry_static_report_sha256,
        )
        destination = (root / args.authorization).resolve()
        expected = (root / AUTHORIZATION_RELATIVE_PATH).resolve()
        if destination != expected:
            raise PermissionError("D9-D authorization path is not frozen")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    result = run_d9d_real_causal_isolation(
        config_path=args.config,
        authorization_path=args.authorization,
        project_root=root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
