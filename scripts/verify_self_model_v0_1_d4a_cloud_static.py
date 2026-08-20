from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d4a_cloud_static_verification import (
    build_d4a_cloud_static_report,
    probe_installed_rwkv_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify D4A against installed source without a model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    installed, source_bytes = probe_installed_rwkv_source()
    report = build_d4a_cloud_static_report(
        config_path=args.config,
        project_root=args.project_root,
        installed_source=installed,
        upstream_source_bytes=source_bytes,
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
                "g0_variant_selection": report["inspection"]["g0_variant_selection"],
                "rwkv_model_imported": report["safety"]["rwkv_model_imported"],
                "torch_imported": report["safety"]["torch_imported"],
                "model_executed": report["safety"]["model_executed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
