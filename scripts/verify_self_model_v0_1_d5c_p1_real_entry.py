from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d5c_p1_real_entry import build_p1_entry_static_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the D5C-P1 no-model real entry")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_p1_entry_static_report(
        config_path=args.config, project_root=args.project_root
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "checks": len(report["checks"]),
        "report_digest_sha256": report["report_digest_sha256"],
        "model_executed": report["safety"]["model_executed"],
        "next_gate": report["next_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
