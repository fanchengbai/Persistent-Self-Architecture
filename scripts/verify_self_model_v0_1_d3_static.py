from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.d3_static_verification import (
    build_d3_static_report,
    probe_installed_rwkv_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the D3 installed-source and wrapper static gate"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--d2-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d2_report = json.loads(Path(args.d2_report).read_text(encoding="utf-8"))
    installed_source = probe_installed_rwkv_source()
    report = build_d3_static_report(
        config_path=args.config,
        project_root=args.project_root,
        installed_source=installed_source,
        d2_report=d2_report,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "report_digest_sha256": report["report_digest_sha256"],
                "installed_source_sha256": report["installed_source"][
                    "model_source_sha256"
                ],
                "rwkv_model_imported": report["safety"]["rwkv_model_imported"],
                "torch_imported": report["safety"]["torch_imported"],
                "model_loaded": report["safety"]["model_loaded"],
                "model_executed": report["safety"]["model_executed"],
                "off_g2_implemented": report["safety"]["off_g2_implemented"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
