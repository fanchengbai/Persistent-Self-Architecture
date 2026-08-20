from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d4a_failure_diagnostic_manifest import build_d4a_runtime_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify fake-only D4A runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_d4a_runtime_report(
        config_path=args.config, project_root=args.project_root
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "checks": len(report["checks"]),
                "report_digest_sha256": report["report_digest_sha256"],
                "model_executed": report["safety"]["model_executed"],
                "real_execution_entry_implemented": report["safety"][
                    "real_execution_entry_implemented"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
