from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d9b_manifest_endpoint_contract import (
    CONFIG_RELATIVE_PATH,
    build_contract_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify D9-B deterministic manifests and fake causal endpoint"
    )
    parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_contract_report(
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
                "acceptance_categories": len(report["fake_acceptance"]["checks"]),
                "calibration_records": report["fake_acceptance"]["counts"][
                    "calibration_records"
                ],
                "heldout_fixtures": report["fake_acceptance"]["counts"][
                    "heldout_fixtures"
                ],
                "pair_records": report["fake_acceptance"]["counts"]["pair_records"],
                "future_forward_calls": report["fake_acceptance"]["counts"][
                    "future_forward_calls"
                ],
                "model_executed": report["safety"]["model_executed"],
                "report_digest_sha256": report["report_digest_sha256"],
                "next_gate": report["next_gate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
