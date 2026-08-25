from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d6d_ii_real_entry import build_d6d_ii_static_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify D6D-II manifests, single-use entry, and optional installed source without a model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-installed-source", action="store_true")
    args = parser.parse_args()
    report = build_d6d_ii_static_report(
        config_path=args.config,
        project_root=args.project_root,
        probe_installed_source=args.probe_installed_source,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "valid": report["valid"],
        "classification": report["classification"],
        "checks": len(report["checks"]),
        "manifest_checks": (
            len(report["manifest_report"]["checks"])
            + len(report["manifest_report"]["training_checks"])
            + len(report["manifest_report"]["pilot_checks"])
        ),
        "installed_source_probed": report["safety"]["installed_source_probed"],
        "installed_source_static_compiled_without_exec": report["safety"][
            "installed_source_static_compiled_without_exec"
        ],
        "model_executed": report["safety"]["model_executed"],
        "report_digest_sha256": report["report_digest_sha256"],
        "next_gate": report["next_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
