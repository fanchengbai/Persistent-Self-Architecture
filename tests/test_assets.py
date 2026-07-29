from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psa.assets import fetch_manifest, load_manifest, plan_manifest, verify_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP001_MANIFEST = (
    PROJECT_ROOT / "configs" / "assets" / "exp001_rwkv7_world_0.4b.json"
)


class AssetManifestTests(unittest.TestCase):
    def test_exp001_manifest_is_pinned_and_safe(self) -> None:
        manifest = load_manifest(EXP001_MANIFEST)
        self.assertEqual(manifest.bundle_id, "exp001-rwkv7-world-0.4b")
        self.assertEqual(len(manifest.assets), 2)
        self.assertTrue(all(asset.revision != "main" for asset in manifest.assets))

        with tempfile.TemporaryDirectory() as directory:
            plan = plan_manifest(manifest, directory)
            root = Path(directory).resolve()
            for asset in plan["assets"]:
                self.assertIn(root, Path(asset["destination"]).parents)

    def test_mutable_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest_version": "0.1",
                        "bundle_id": "bad",
                        "description": "bad",
                        "assets": [
                            {
                                "id": "bad",
                                "kind": "model",
                                "destination": "models/bad.bin",
                                "license": "test",
                                "source_page": "https://example.test",
                                "source": {
                                    "type": "huggingface",
                                    "repo_type": "model",
                                    "repo_id": "owner/repo",
                                    "revision": "main",
                                    "filename": "bad.bin",
                                },
                            }
                        ],
                        "generated_assets": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mutable revision"):
                load_manifest(path)

    def test_destination_traversal_is_rejected(self) -> None:
        payload = json.loads(EXP001_MANIFEST.read_text(encoding="utf-8"))
        payload["assets"][0]["destination"] = "../outside.bin"
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_manifest(manifest_path)
            with self.assertRaisesRegex(ValueError, "unsafe asset destination"):
                plan_manifest(manifest, Path(directory) / "assets")

    def test_verify_checks_pinned_digest(self) -> None:
        content = b"small-test-asset"
        digest = hashlib.sha256(content).hexdigest()
        payload = {
            "manifest_version": "0.1",
            "bundle_id": "test-bundle",
            "description": "test",
            "assets": [
                {
                    "id": "fixture",
                    "kind": "dataset",
                    "destination": "datasets/fixture.bin",
                    "license": "test-only",
                    "source_page": "https://example.test/fixture",
                    "source": {
                        "type": "huggingface",
                        "repo_type": "dataset",
                        "repo_id": "owner/repo",
                        "revision": "1234567",
                        "filename": "fixture.bin",
                    },
                    "sha256": digest,
                }
            ],
            "generated_assets": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            asset_path = base / "assets" / "datasets" / "fixture.bin"
            asset_path.parent.mkdir(parents=True)
            asset_path.write_bytes(content)

            report = verify_manifest(load_manifest(manifest_path), base / "assets")
            self.assertTrue(report["valid"])
            self.assertEqual(report["assets"][0]["sha256"], digest)

    def test_fetch_resumes_and_does_not_record_token(self) -> None:
        content = b"small-test-asset"
        digest = hashlib.sha256(content).hexdigest()
        payload = {
            "manifest_version": "0.1",
            "bundle_id": "download-test",
            "description": "test",
            "assets": [
                {
                    "id": "fixture",
                    "kind": "dataset",
                    "destination": "datasets/fixture.bin",
                    "license": "test-only",
                    "source_page": "https://example.test/fixture",
                    "source": {
                        "type": "huggingface",
                        "repo_type": "dataset",
                        "repo_id": "owner/repo",
                        "revision": "1234567",
                        "filename": "fixture.bin",
                    },
                    "sha256": digest,
                }
            ],
            "generated_assets": [],
        }

        class FakeResponse:
            status = 206

            def __init__(self, body: bytes) -> None:
                self.body = body
                self.offset = 0
                self.headers = {"Content-Length": str(len(body))}

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def getcode(self) -> int:
                return self.status

            def read(self, size: int) -> bytes:
                chunk = self.body[self.offset : self.offset + size]
                self.offset += len(chunk)
                return chunk

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            asset_root = base / "assets"
            partial = asset_root / "datasets" / "fixture.bin.part"
            partial.parent.mkdir(parents=True)
            prefix = b"small-"
            partial.write_bytes(prefix)

            requests = []

            def fake_urlopen(request: object, timeout: float) -> FakeResponse:
                requests.append(request)
                return FakeResponse(content[len(prefix) :])

            with patch("psa.assets.manager.urlopen", side_effect=fake_urlopen):
                receipt = fetch_manifest(
                    load_manifest(manifest_path),
                    asset_root,
                    token="never-record-this-token",
                )

            final_path = asset_root / "datasets" / "fixture.bin"
            self.assertEqual(final_path.read_bytes(), content)
            self.assertEqual(receipt["assets"][0]["status"], "downloaded")
            self.assertEqual(requests[0].get_header("Range"), f"bytes={len(prefix)}-")
            receipt_text = Path(receipt["receipt_path"]).read_text(encoding="utf-8")
            self.assertNotIn("never-record-this-token", receipt_text)
