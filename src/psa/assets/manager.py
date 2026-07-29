from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from psa.artifacts import canonical_json_bytes, sha256_file


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_KINDS = {"model", "tokenizer", "dataset", "code"}


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    kind: str
    destination: str
    license_name: str
    source_page: str
    repo_type: str
    repo_id: str
    revision: str
    filename: str
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None


@dataclass(frozen=True)
class AssetManifest:
    version: str
    bundle_id: str
    description: str
    assets: tuple[AssetSpec, ...]
    generated_assets: tuple[dict[str, Any], ...]
    manifest_path: Path


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _parse_asset(payload: Any, index: int) -> AssetSpec:
    if not isinstance(payload, dict):
        raise ValueError(f"assets[{index}] must be an object")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("type") != "huggingface":
        raise ValueError(f"assets[{index}].source must be a Hugging Face source")

    asset_id = _require_string(payload.get("id"), f"assets[{index}].id")
    kind = _require_string(payload.get("kind"), f"assets[{index}].kind")
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"assets[{index}].kind is unsupported: {kind}")

    expected_sha256 = payload.get("sha256")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or not _SHA256_PATTERN.fullmatch(expected_sha256)
    ):
        raise ValueError(f"assets[{index}].sha256 must be 64 lowercase hex digits")

    expected_size = payload.get("size_bytes")
    if expected_size is not None and (
        not isinstance(expected_size, int) or expected_size <= 0
    ):
        raise ValueError(f"assets[{index}].size_bytes must be a positive integer")

    revision = _require_string(
        source.get("revision"), f"assets[{index}].source.revision"
    )
    if revision in {"main", "master", "latest"}:
        raise ValueError(f"assets[{index}] uses a mutable revision: {revision}")

    repo_type = _require_string(
        source.get("repo_type"), f"assets[{index}].source.repo_type"
    )
    if repo_type not in {"model", "dataset"}:
        raise ValueError(f"assets[{index}].source.repo_type is invalid")

    return AssetSpec(
        asset_id=asset_id,
        kind=kind,
        destination=_require_string(
            payload.get("destination"), f"assets[{index}].destination"
        ),
        license_name=_require_string(
            payload.get("license"), f"assets[{index}].license"
        ),
        source_page=_require_string(
            payload.get("source_page"), f"assets[{index}].source_page"
        ),
        repo_type=repo_type,
        repo_id=_require_string(
            source.get("repo_id"), f"assets[{index}].source.repo_id"
        ),
        revision=revision,
        filename=_require_string(
            source.get("filename"), f"assets[{index}].source.filename"
        ),
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size,
    )


def load_manifest(path: str | Path) -> AssetManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("asset manifest must be a JSON object")
    if payload.get("manifest_version") != "0.1":
        raise ValueError("unsupported asset manifest version")

    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("asset manifest must contain at least one downloadable asset")
    assets = tuple(_parse_asset(item, index) for index, item in enumerate(raw_assets))
    asset_ids = [asset.asset_id for asset in assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("asset IDs must be unique")

    generated = payload.get("generated_assets", [])
    if not isinstance(generated, list):
        raise ValueError("generated_assets must be an array")

    return AssetManifest(
        version="0.1",
        bundle_id=_require_string(payload.get("bundle_id"), "bundle_id"),
        description=_require_string(payload.get("description"), "description"),
        assets=assets,
        generated_assets=tuple(generated),
        manifest_path=manifest_path,
    )


def _safe_destination(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"unsafe asset destination: {relative}")
    destination = root.joinpath(*pure_path.parts).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"asset destination escapes root: {relative}")
    return destination


def _asset_url(asset: AssetSpec, endpoint: str) -> str:
    prefix = "datasets/" if asset.repo_type == "dataset" else ""
    repo_id = quote(asset.repo_id, safe="/")
    revision = quote(asset.revision, safe="")
    filename = quote(asset.filename, safe="/")
    return f"{endpoint.rstrip('/')}/{prefix}{repo_id}/resolve/{revision}/{filename}"


def _selected_assets(
    manifest: AssetManifest, selected_ids: Iterable[str] | None
) -> tuple[AssetSpec, ...]:
    if selected_ids is None:
        return manifest.assets
    selected = set(selected_ids)
    unknown = selected.difference(asset.asset_id for asset in manifest.assets)
    if unknown:
        raise ValueError(f"unknown asset IDs: {', '.join(sorted(unknown))}")
    return tuple(asset for asset in manifest.assets if asset.asset_id in selected)


def plan_manifest(
    manifest: AssetManifest,
    root: str | Path,
    selected_ids: Iterable[str] | None = None,
    hf_endpoint: str | None = None,
) -> dict[str, Any]:
    asset_root = Path(root).resolve()
    endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    assets = []
    for asset in _selected_assets(manifest, selected_ids):
        destination = _safe_destination(asset_root, asset.destination)
        assets.append(
            {
                "id": asset.asset_id,
                "kind": asset.kind,
                "destination": str(destination),
                "exists": destination.is_file(),
                "source": _asset_url(asset, endpoint),
                "revision": asset.revision,
                "expected_sha256": asset.expected_sha256,
                "expected_size_bytes": asset.expected_size_bytes,
                "license": asset.license_name,
            }
        )
    return {
        "bundle_id": manifest.bundle_id,
        "manifest": str(manifest.manifest_path),
        "asset_root": str(asset_root),
        "assets": assets,
        "generated_assets": list(manifest.generated_assets),
    }


def _download_once(
    asset: AssetSpec,
    destination: Path,
    endpoint: str,
    token: str | None,
    timeout: float,
) -> tuple[int, str]:
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "persistent-self-architecture/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(_asset_url(asset, endpoint), headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        written = offset if append else 0

        content_length = response.headers.get("Content-Length")
        if content_length:
            remaining = int(content_length)
            free_bytes = shutil.disk_usage(destination.parent).free
            reserve = 512 * 1024 * 1024
            if free_bytes < remaining + reserve:
                raise OSError(
                    f"insufficient disk space for {asset.asset_id}: "
                    f"need at least {remaining + reserve} free bytes"
                )

        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)

    digest = sha256_file(partial)
    if asset.expected_size_bytes is not None and written != asset.expected_size_bytes:
        raise OSError(
            f"size mismatch for {asset.asset_id}: "
            f"expected {asset.expected_size_bytes}, got {written}"
        )
    if asset.expected_sha256 is not None and digest != asset.expected_sha256:
        raise OSError(
            f"SHA-256 mismatch for {asset.asset_id}: "
            f"expected {asset.expected_sha256}, got {digest}; "
            f"partial file retained at {partial}"
        )
    os.replace(partial, destination)
    return written, digest


def _existing_record(asset: AssetSpec, destination: Path) -> dict[str, Any] | None:
    if not destination.is_file():
        return None
    digest = sha256_file(destination)
    size = destination.stat().st_size
    if asset.expected_sha256 is not None and digest != asset.expected_sha256:
        raise OSError(
            f"existing file has wrong SHA-256 for {asset.asset_id}: {destination}"
        )
    if asset.expected_size_bytes is not None and size != asset.expected_size_bytes:
        raise OSError(f"existing file has wrong size for {asset.asset_id}: {destination}")
    return {
        "id": asset.asset_id,
        "status": "reused",
        "path": str(destination),
        "size_bytes": size,
        "sha256": digest,
    }


def fetch_manifest(
    manifest: AssetManifest,
    root: str | Path,
    selected_ids: Iterable[str] | None = None,
    retries: int = 3,
    timeout: float = 60.0,
    hf_endpoint: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    if retries < 1:
        raise ValueError("retries must be at least 1")
    asset_root = Path(root).resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    auth_token = token or os.environ.get("HF_TOKEN")
    records: list[dict[str, Any]] = []

    for asset in _selected_assets(manifest, selected_ids):
        destination = _safe_destination(asset_root, asset.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing = _existing_record(asset, destination)
        if existing is not None:
            records.append(existing)
            continue

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                size, digest = _download_once(
                    asset, destination, endpoint, auth_token, timeout
                )
                records.append(
                    {
                        "id": asset.asset_id,
                        "status": "downloaded",
                        "path": str(destination),
                        "size_bytes": size,
                        "sha256": digest,
                        "revision": asset.revision,
                    }
                )
                last_error = None
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if last_error is not None:
            raise OSError(
                f"failed to download {asset.asset_id} after {retries} attempts: "
                f"{last_error}"
            ) from last_error

    receipt = {
        "receipt_version": "0.1",
        "bundle_id": manifest.bundle_id,
        "manifest": str(manifest.manifest_path),
        "asset_root": str(asset_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_endpoint": endpoint,
        "assets": records,
    }
    receipt_path = asset_root / "receipts" / f"{manifest.bundle_id}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def verify_manifest(
    manifest: AssetManifest,
    root: str | Path,
    selected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    asset_root = Path(root).resolve()
    records = []
    valid = True
    for asset in _selected_assets(manifest, selected_ids):
        destination = _safe_destination(asset_root, asset.destination)
        if not destination.is_file():
            records.append(
                {"id": asset.asset_id, "status": "missing", "path": str(destination)}
            )
            valid = False
            continue
        size = destination.stat().st_size
        digest = sha256_file(destination)
        checksum_ok = (
            asset.expected_sha256 is None or digest == asset.expected_sha256
        )
        size_ok = (
            asset.expected_size_bytes is None
            or size == asset.expected_size_bytes
        )
        status = "valid" if checksum_ok and size_ok else "invalid"
        valid = valid and status == "valid"
        records.append(
            {
                "id": asset.asset_id,
                "status": status,
                "path": str(destination),
                "size_bytes": size,
                "sha256": digest,
                "checksum_pinned": asset.expected_sha256 is not None,
            }
        )
    return {
        "bundle_id": manifest.bundle_id,
        "asset_root": str(asset_root),
        "valid": valid,
        "assets": records,
    }
