from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.development.exp001c_v02_stage_b_observation import (
    analyze_exp001c_v02_stage_b,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe EXP-001C v02 Stage B once")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--config",
        default="configs/analysis/exp001c_v02_stage_b_observation_v01.json",
    )
    parser.add_argument(
        "--design-manifest",
        default="results/development/exp001c_v02_stage_b_preflight_v03/design_manifest.json",
    )
    parser.add_argument(
        "--result",
        default="results/development/exp001c_v02_stage_b_pilot_v01/stage_b_result.json",
    )
    parser.add_argument(
        "--summary",
        default="results/development/exp001c_v02_stage_b_pilot_v01/summary.json",
    )
    parser.add_argument(
        "--protocol-config",
        default="configs/development/exp001c_noncore_protocol_v02.draft.json",
    )
    parser.add_argument(
        "--output",
        default="results/development/exp001c_v02_stage_b_observation_v01/observation.json",
    )
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    output = (root / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        report = analyze_exp001c_v02_stage_b(
            analysis_config_path=arguments.config,
            design_manifest_path=arguments.design_manifest,
            stage_b_result_path=arguments.result,
            stage_b_summary_path=arguments.summary,
            protocol_config_path=arguments.protocol_config,
            project_root=root,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "analysis_plan_digest_sha256": report[
                    "analysis_plan_digest_sha256"
                ],
                "condition_metrics": {
                    condition: {
                        "joint": values["label_marginalized_joint_accuracy"],
                        "domain": values["label_marginalized_domain_accuracy"],
                        "operation": values[
                            "label_marginalized_operation_accuracy"
                        ],
                        "diagnostic_reference_match": values[
                            "diagnostic_reference_match_rate"
                        ],
                        "margin": values[
                            "mean_semantic_target_or_reference_margin"
                        ],
                    }
                    for condition, values in report["condition_reports"].items()
                },
                "descriptive_contrasts": report["descriptive_contrasts"],
                "contains_confirmatory_decision": False,
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
