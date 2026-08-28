from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d7b_manifest_runtime_contract import build_d7b_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the no-model D7-B manifests and symbolic fake runtime"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_d7b_report(
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
                "classification": report["classification"],
                "checks": len(report["checks"]),
                "acceptance_categories": len(report["acceptance"]["checks"]),
                "calibration_records": report["acceptance"]["counts"][
                    "calibration_records"
                ],
                "heldout_fixtures": report["acceptance"]["counts"][
                    "heldout_fixtures"
                ],
                "heldout_forward_calls": report["acceptance"]["counts"][
                    "heldout_forward_calls"
                ],
                "report_digest_sha256": report["report_digest_sha256"],
                "model_executed": report["safety"]["model_executed"],
                "projection_constructed": report["safety"][
                    "projection_constructed"
                ],
                "next_gate": report["next_gate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
