from __future__ import annotations

import argparse
import json
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model.instrumented_off_manifest import (
    build_instrumented_off_report,
    probe_installed_rwkv_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the no-model OFF-G2 instrumented runtime implementation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--d3-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d3_report = json.loads(Path(args.d3_report).read_text(encoding="utf-8"))
    installed_source, source_bytes = probe_installed_rwkv_source()
    report = build_instrumented_off_report(
        config_path=args.config,
        project_root=args.project_root,
        installed_source=installed_source,
        upstream_source_bytes=source_bytes,
        d3_report=d3_report,
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
                "injection_counts": report["transformation"]["injection_counts"],
                "rwkv_model_imported": report["safety"]["rwkv_model_imported"],
                "torch_imported": report["safety"]["torch_imported"],
                "model_loaded": report["safety"]["model_loaded"],
                "model_executed": report["safety"]["model_executed"],
                "off_g2_implemented": report["safety"]["off_g2_implemented"],
                "real_model_equivalence_executed": report["safety"][
                    "real_model_equivalence_executed"
                ],
                "active_injection_implemented": report["safety"][
                    "active_injection_implemented"
                ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
