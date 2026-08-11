from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.fake_callback_runtime import build_fake_callback_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the no-weight Self Model v0.1 residual callback contract"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_fake_callback_report(
        config_path=args.config,
        project_root=args.project_root,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "report_digest_sha256": report["report_digest_sha256"],
        "rwkv_model_imported": report["safety"]["rwkv_model_imported"],
        "model_loaded": report["safety"]["model_loaded"],
        "model_executed": report["safety"]["model_executed"],
        "real_hook_implemented": report["safety"]["real_hook_implemented"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
