from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.rwkv_interface_audit import build_rwkv_interface_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only static audit of the installed RWKV-7 coupling interface"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_rwkv_interface_audit(
        config_path=args.config,
        project_root=args.project_root,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "audit_digest_sha256": report["audit_digest_sha256"],
        "model_loaded": report["safety"]["model_loaded"],
        "model_executed": report["safety"]["model_executed"],
        "rwkv_model_imported": report["safety"]["rwkv_model_imported"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
