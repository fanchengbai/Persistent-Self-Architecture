from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d9d_offline_causal_diagnostic import (
    CONFIG_RELATIVE_PATH,
    build_offline_diagnostic_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze consumed D9-D artifacts without importing or running the model"
    )
    parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_offline_diagnostic_report(
        config_path=args.config,
        project_root=args.project_root,
    )
    output = Path(args.output)
    expected = Path(args.project_root).resolve() / json.loads(
        (Path(args.project_root).resolve() / args.config).read_text(encoding="utf-8")
    )["output_path"]
    if output.resolve() != expected.resolve():
        raise PermissionError("D9-D offline diagnostic output path is not frozen")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "classification": report["classification"],
                "checks": len(report["checks"]),
                "positive_base_cases": report["causal_distribution"]["endpoint"][
                    "metrics"
                ]["positive_base_cases"],
                "synthetic_positive_control_passes": report[
                    "causal_distribution"
                ]["endpoint"]["checks"]["synthetic_positive_control_passes"],
                "all_gates_pass": report["causal_distribution"]["endpoint"][
                    "all_gates_pass"
                ],
                "replicate_numeric_stability_identifiable": report[
                    "calibration_replicate_audit"
                ]["replicate_numeric_distance_identifiable"],
                "d9d_rerun_allowed": report["route_review"]["d9d_rerun_allowed"],
                "model_executed": report["model_executed"],
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
