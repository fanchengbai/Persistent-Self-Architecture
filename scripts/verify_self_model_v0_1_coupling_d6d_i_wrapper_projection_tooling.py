from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d6d_i_tooling import build_d6d_i_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify D6D-I wrapper/projection tooling without a model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_d6d_i_report(
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
                "acceptance_categories": len(report["acceptance"]["checks"]),
                "joint_conditions": report["acceptance"]["counts"]["joint_conditions"],
                "base_instance_mutated": report["safety"]["real_model_instance_mutated"],
                "real_projection_constructed": report["safety"][
                    "real_projection_artifact_constructed"
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
