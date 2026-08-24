from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d5c_p1_reporter_adapter_fix import (
    build_reporter_adapter_fix_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the D5C-P1 explicit offline-adapter reporter fix"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_reporter_adapter_fix_report(
        config_path=args.config, project_root=args.project_root
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "classification": report["classification"],
        "checks": len(report["checks"]),
        "acceptance_categories": len(report["acceptance"]["checks"]),
        "report_digest_sha256": report["report_digest_sha256"],
        "model_executed": report["safety"]["model_executed"],
        "p1_rerun": report["safety"]["p1_rerun"],
        "next_gate": report["next_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
