from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psa.artifacts import canonical_json_bytes
from psa.self_model import (
    build_self_model_v0_1_offline_manifest,
    verify_self_model_v0_1_offline_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Self Model v0.1 offline-only evidence manifest"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--config",
        default="configs/development/self_model_v0_1_offline.draft.json",
    )
    parser.add_argument(
        "--output",
        default="results/development/self_model_v0_1_offline/manifest.json",
    )
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    output = (root / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_self_model_v0_1_offline_manifest(
        config_path=arguments.config,
        project_root=root,
    )
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    verification = verify_self_model_v0_1_offline_manifest(
        manifest_path=output,
        config_path=arguments.config,
        project_root=root,
    )
    if verification["valid"] is not True:
        raise RuntimeError("Self Model v0.1 offline manifest verification failed")
    print(
        json.dumps(
            {
                **verification,
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
