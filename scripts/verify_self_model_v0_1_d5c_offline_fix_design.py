from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d5c_offline_fix_design import build_fix_design_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the design-only D5C offline fix plan")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_fix_design_report(config_path=args.config, project_root=args.project_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "classification": report["classification"],
        "checks": len(report["checks"]),
        "report_digest_sha256": report["report_digest_sha256"],
        "fake_fix_implemented": report["safety"]["fake_fix_implemented"],
        "real_runtime_modified": report["safety"]["real_runtime_modified"],
        "model_executed": report["safety"]["model_executed"],
        "next_gate": report["next_gate"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
