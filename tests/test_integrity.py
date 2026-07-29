from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)


class IntegrityTests(unittest.TestCase):
    def test_canonical_json_is_order_independent(self) -> None:
        left = {"b": 2, "a": 1}
        right = {"a": 1, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(sha256_json(left), sha256_json(right))
        self.assertEqual(json.loads(canonical_json_bytes(left)), left)

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"psa")
            self.assertEqual(
                sha256_file(path), hashlib.sha256(b"psa").hexdigest()
            )

    def test_payload_digest_is_path_sorted(self) -> None:
        first = payload_digest({"b": "2", "a": "1"})
        second = payload_digest({"a": "1", "b": "2"})
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

