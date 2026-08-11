from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from psa.development.exp001c_v02_stage_b_design import (
    STAGE_B_CONDITIONS,
    build_exp001c_v02_stage_b_design_manifest,
)
from psa.development.exp001c_v02_stage_b_offline import (
    OFFLINE_TEST_LOCK,
    OfflineFakeStageBContractBackend,
    run_exp001c_v02_stage_b_model,
    run_exp001c_v02_stage_b_offline_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "development"
    / "exp001c_v02_stage_b_design.draft.json"
)


class _FakeAdapter:
    offline_fake_adapter = True
    model_loaded = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def score_route(self, record):
        self.calls.append(dict(record))
        target = record["expected_state_semantic_target_code"] or "A"
        scores = {code: -4.0 for code in "ABCD"}
        scores[target] = -0.1
        return scores


def _write_design_manifest(directory: Path) -> Path:
    manifest = build_exp001c_v02_stage_b_design_manifest(
        design_config_path=CONFIG,
        project_root=ROOT,
    )
    path = directory / "stage_b_design_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class Exp001CV02StageBOfflineTests(unittest.TestCase):
    def test_fake_backend_routes_all_records_without_model(self) -> None:
        manifest = build_exp001c_v02_stage_b_design_manifest(
            design_config_path=CONFIG,
            project_root=ROOT,
        )
        adapter = _FakeAdapter()
        result = OfflineFakeStageBContractBackend(
            adapter=adapter
        ).run_offline_contract(manifest)
        self.assertEqual(result["record_count"], 224)
        self.assertEqual(len(adapter.calls), 224)
        self.assertEqual(
            Counter(record["condition"] for record in result["records"]),
            Counter({condition: 32 for condition in STAGE_B_CONDITIONS}),
        )
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["model_executed"])
        self.assertFalse(result["stage_a_rerun"])
        self.assertTrue(result["synthetic_output_not_research_evidence"])
        for record in result["records"]:
            target = record["expected_state_semantic_target_code"]
            if target is None:
                self.assertIsNone(record["answer_boundary_evidence"])
            else:
                self.assertEqual(record["predicted_code"], target)
                self.assertEqual(
                    record["answer_boundary_evidence"]["target_code"],
                    target,
                )

    def test_offline_runner_writes_atomic_contract_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_design_manifest(root)
            adapter = _FakeAdapter()
            summary = run_exp001c_v02_stage_b_offline_contract(
                design_manifest_path=manifest_path,
                output_dir=root / "output",
                backend_factory=lambda: OfflineFakeStageBContractBackend(
                    adapter=adapter
                ),
                offline_test_lock=OFFLINE_TEST_LOCK,
                project_root=ROOT,
            )
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["record_count"], 224)
            self.assertFalse(summary["model_loaded"])
            self.assertFalse(summary["model_executed"])
            self.assertTrue((root / "output" / "summary.json").is_file())
            result_path = (
                root / "output" / "stage_b_offline_contract_result.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(len(result["records"]), 224)
            self.assertEqual(
                summary["offline_result_sha256"],
                hashlib.sha256(result_path.read_bytes()).hexdigest(),
            )

    def test_lock_is_checked_before_paths_and_backend_factory(self) -> None:
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            raise AssertionError("backend factory must remain unreachable")

        with self.assertRaisesRegex(PermissionError, "lock is absent"):
            run_exp001c_v02_stage_b_offline_contract(
                design_manifest_path="missing-manifest.json",
                output_dir="missing-output",
                backend_factory=factory,
                offline_test_lock="",
                project_root=ROOT,
            )
        self.assertFalse(factory_called)

    def test_backend_rejects_any_non_fake_or_loaded_adapter(self) -> None:
        class UnsafeAdapter:
            offline_fake_adapter = False
            model_loaded = True

        with self.assertRaisesRegex(PermissionError, "unloaded fake adapter"):
            OfflineFakeStageBContractBackend(adapter=UnsafeAdapter())

    def test_model_entry_remains_unconditionally_closed(self) -> None:
        with self.assertRaisesRegex(PermissionError, "live preflight"):
            run_exp001c_v02_stage_b_model()

    def test_runner_rejects_nonempty_output_without_calling_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_design_manifest(root)
            output = root / "output"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            factory_called = False

            def factory():
                nonlocal factory_called
                factory_called = True
                return OfflineFakeStageBContractBackend(adapter=_FakeAdapter())

            with self.assertRaisesRegex(ValueError, "must be empty"):
                run_exp001c_v02_stage_b_offline_contract(
                    design_manifest_path=manifest_path,
                    output_dir=output,
                    backend_factory=factory,
                    offline_test_lock=OFFLINE_TEST_LOCK,
                    project_root=ROOT,
                )
            self.assertFalse(factory_called)

    def test_offline_result_schema_is_valid_json(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "exp001c_v02_stage_b_offline_result.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["record_count"]["const"], 224)


if __name__ == "__main__":
    unittest.main()
