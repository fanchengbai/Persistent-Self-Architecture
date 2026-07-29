"""Download and verify versioned external assets."""

from psa.assets.manager import (
    AssetManifest,
    AssetSpec,
    fetch_manifest,
    load_manifest,
    plan_manifest,
    verify_manifest,
)

__all__ = [
    "AssetManifest",
    "AssetSpec",
    "fetch_manifest",
    "load_manifest",
    "plan_manifest",
    "verify_manifest",
]
