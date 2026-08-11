from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import unittest

from psa.artifacts import sha256_json
from psa.cli import main
from psa.development.exp001c_protocol_v02 import (
    PROTOCOL_SOURCE_FILES,
    build_exp001c_protocol_v02_manifest,
    verify_exp001c_protocol_v02_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_noncore_protocol_v02.draft.json"
)


class Exp001CProtocolV02Tests(unittest.TestCase):
    def test_builds_balanced_unrun_positive_control_manifest(self) -> None:
        manifest = build_exp001c_protocol_v02_manifest(
            config_path=CONFIG,
            project_root=ROOT,
        )
        self.assertEqual(manifest["record_count"], 32)
        self.assertEqual(manifest["target_code_counts"], {code: 8 for code in "ABCD"})
        self.assertEqual(
            manifest["rotation_counts"],
            {str(index): 8 for index in range(4)},
        )
        self.assertFalse(manifest["execution_authorized"])
        self.assertFalse(manifest["result_observation_authorized"])
        self.assertFalse(manifest["model_executed"])
        self.assertFalse(manifest["formal_test_set_accessed"])
        self.assertFalse(manifest["stage_a_positive_control"]["authorized"])
        self.assertFalse(manifest["stage_b_recurrent_state"]["authorized"])
        self.assertEqual(
            manifest["model_config"]["path"],
            "configs/models/rwkv7_g1h_2.9b.candidate.json",
        )

    def test_each_semantic_case_has_complete_code_rotation(self) -> None:
        manifest = build_exp001c_protocol_v02_manifest(
            config_path=CONFIG,
            project_root=ROOT,
        )
        by_case = defaultdict(list)
        for trial in manifest["trials"]:
            by_case[trial["semantic_case_id"]].append(trial)
        self.assertEqual(len(by_case), 8)
        for trials in by_case.values():
            self.assertEqual(len(trials), 4)
            self.assertEqual(
                Counter(trial["rotation_index"] for trial in trials),
                Counter(range(4)),
            )
            self.assertEqual(
                {trial["target_code"] for trial in trials},
                set("ABCD"),
            )
            self.assertTrue(
                all(
                    trial["query_text"].endswith(
                        "Assistant: <think></think"
                    )
                    for trial in trials
                )
            )

    def test_manifest_verifier_locks_sources_and_rejects_execution(self) -> None:
        manifest = build_exp001c_protocol_v02_manifest(
            config_path=CONFIG,
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_protocol_v02_manifest(
                path,
                project_root=ROOT,
            )
            self.assertTrue(verification["valid"])
            self.assertTrue(verification["deterministic_payload_valid"])
            self.assertEqual(
                set(verification["source_checks"]),
                set(PROTOCOL_SOURCE_FILES),
            )
            manifest["execution_authorized"] = True
            manifest["manifest_digest_sha256"] = sha256_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_digest_sha256"
                }
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_protocol_v02_manifest(
                path,
                project_root=ROOT,
            )
            self.assertFalse(verification["valid"])
            self.assertFalse(verification["safety_boundary_valid"])

            manifest = build_exp001c_protocol_v02_manifest(
                config_path=CONFIG,
                project_root=ROOT,
            )
            manifest["stage_a_positive_control"][
                "minimum_label_marginalized_accuracy"
            ] = 0.7
            manifest["manifest_digest_sha256"] = sha256_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_digest_sha256"
                }
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_exp001c_protocol_v02_manifest(
                path,
                project_root=ROOT,
            )
            self.assertFalse(verification["valid"])
            self.assertFalse(verification["deterministic_payload_valid"])

    def test_cli_builds_and_verifies_without_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            self.assertEqual(
                main(
                    [
                        "exp001c-protocol-v02-build",
                        "--config",
                        str(CONFIG),
                        "--output",
                        str(path),
                        "--project-root",
                        str(ROOT),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "exp001c-protocol-v02-verify",
                        "--manifest",
                        str(path),
                        "--project-root",
                        str(ROOT),
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
