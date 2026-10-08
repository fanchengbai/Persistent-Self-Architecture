from __future__ import annotations

import argparse
import json

from psa.self_model.semantic_goal_offline import write_offline_package


def main() -> int:
    parser = argparse.ArgumentParser(description="Build semantic offline preview data and verify goal-state interfaces")
    parser.add_argument("--config", default="configs/development/self_model_v0_1_semantic_goal_offline.json")
    parser.add_argument("--output-dir", required=True, help="New directory; existing output is never overwritten")
    args = parser.parse_args()
    report = write_offline_package(args.config, args.output_dir)
    keys = ("status", "valid", "scenario_count", "public_task_count", "evaluation_reference_count",
            "model_executed", "actual_model_forward_calls", "report_digest_sha256")
    print(json.dumps({k: report[k] for k in keys}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
