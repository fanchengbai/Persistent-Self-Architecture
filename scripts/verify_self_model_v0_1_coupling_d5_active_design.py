from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d5_active_injection_design import build_design_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the offline Coupling-D5 active injection design"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_design_report(
        config_path=args.config,
        project_root=args.project_root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "checks": len(report["checks"]),
                "next_gate": report["next_gate"],
                "report_digest_sha256": report["report_digest_sha256"],
                "model_executed": report["safety"]["model_executed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
