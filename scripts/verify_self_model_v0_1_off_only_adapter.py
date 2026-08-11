from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.off_only_adapter_manifest import build_off_only_adapter_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the D2 no-model RWKV coupling-off wrapper"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_off_only_adapter_report(
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
        "model_loaded": report["safety"]["model_loaded"],
        "model_executed": report["safety"]["model_executed"],
        "off_only_adapter_implemented": report["safety"][
            "off_only_adapter_implemented"
        ],
        "active_injection_implemented": report["safety"][
            "active_injection_implemented"
        ],
        "output": str(output),
    }, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
