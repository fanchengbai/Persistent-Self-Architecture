from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d5c_decorator_object_protocol_fixture import build_fixture_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic D5C decorator/object-protocol fixture")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_fixture_report(config_path=args.config, project_root=args.project_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "classification": report["classification"],
        "checks": len(report["checks"]),
        "case_count": report["matrix_summary"]["case_count"],
        "report_digest_sha256": report["report_digest_sha256"],
        "model_executed": report["safety"]["model_executed"],
        "fix_implemented": report["safety"]["fix_implemented"],
        "next_gate": report["next_gate"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
