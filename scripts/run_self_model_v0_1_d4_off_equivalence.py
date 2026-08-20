from __future__ import annotations

import argparse
import json

from psa.self_model.d4_real_off_equivalence import run_d4_real_off_equivalence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the single-use D4 real 2.9B OFF equivalence gate"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_d4_real_off_equivalence(
        config_path=args.config,
        project_root=args.project_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "report_digest_sha256": report["report_digest_sha256"],
                "cells": len(report["matrix"]["cells"]),
                "all_cells_exact": report["matrix"]["checks"]["all_cells_exact"],
                "cuda_peak_memory_bytes": report["cuda_peak_memory_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
