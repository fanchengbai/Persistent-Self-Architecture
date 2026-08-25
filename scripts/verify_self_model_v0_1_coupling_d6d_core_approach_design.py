from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d6d_core_approach_design import build_d6d_no_model_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the D6D joint core-approach design without a model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_d6d_no_model_review(
        config_path=args.config, project_root=args.project_root
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
                "planned_model_forward_calls": report["counts"][
                    "planned_model_forward_calls"
                ],
                "model_executed": report["safety"]["model_executed"],
                "real_self_projection_constructed": report["safety"][
                    "real_self_projection_constructed"
                ],
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
