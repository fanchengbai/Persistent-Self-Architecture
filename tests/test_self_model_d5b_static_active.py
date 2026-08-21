from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from psa.artifacts import sha256_json
from psa.self_model.d5b_static_active import (
    CALLBACK_ATTRIBUTE,
    CONFIG_RELATIVE_PATH,
    REQUIRED_NEXT_CONFIRMATION,
    RWKV7ProjectStaticActiveRuntime,
    SYNTHETIC_SOURCE,
    StaticActiveRequest,
    StaticResidualTensor,
    StaticSyntheticCallback,
    _state,
    _synthetic_namespace,
    _vector,
    build_d5b_report,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH
MODULE = ROOT / "src/psa/self_model/d5b_static_active.py"


def _runtime():
    namespace, fixture_class = _synthetic_namespace()
    fixture = fixture_class()
    digest = hashlib.sha256(SYNTHETIC_SOURCE.encode("utf-8")).hexdigest()
    with mock.patch(
        "psa.self_model.d5b_static_active.EXPECTED_RWKV_MODEL_SOURCE_SHA256",
        digest,
    ):
        runtime = RWKV7ProjectStaticActiveRuntime(
            base_fixture=fixture,
            upstream_source_bytes=SYNTHETIC_SOURCE.encode("utf-8"),
            upstream_globals=namespace,
            upstream_package_version="0.8.32",
            upstream_de_version=None,
        )
    return runtime, fixture


def _callback(scale=1.0):
    return StaticSyntheticCallback(
        vector=_vector(),
        layer_mask=["fake-layer-01", "fake-layer-03"],
        scale=scale,
        gate=0.5,
    )


class CouplingD5BStaticActiveTests(unittest.TestCase):
    def test_report_verifies_project_path_without_model_imports(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_d5b_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["contract_checks"].values()))
        self.assertTrue(all(report["runtime_checks"].values()))
        self.assertTrue(report["safety"]["d5b_project_static_active_implemented"])
        self.assertTrue(report["safety"]["offline_static_fixture_executed"])
        for field in (
            "installed_rwkv_source_probed",
            "rwkv_model_imported",
            "torch_imported",
            "weights_accessed",
            "model_loaded",
            "model_executed",
            "real_model_active_injection_executed",
            "real_layers_selected",
            "real_self_projection_constructed",
            "formal_test_set_accessed",
            "self_effect_experiment_run",
            "self_updater_implemented",
            "automatic_rerun_authorized",
        ):
            self.assertFalse(report["safety"][field])
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            config["required_next_owner_confirmation_text"],
            REQUIRED_NEXT_CONFIRMATION,
        )

    def test_off_zero_and_active_restore_fixture(self) -> None:
        runtime, fixture = _runtime()
        baseline = fixture.forward([3, 5, 8], _state(), True)
        off = runtime.forward([3, 5, 8], _state(), True)
        zero_callback = _callback(scale=1.0)
        zero = runtime.forward(
            [3, 5, 8],
            _state(),
            True,
            coupling=StaticActiveRequest(enabled=True, scale=0.0, callback=zero_callback),
        )
        active_callback = _callback()
        active = runtime.forward(
            [3, 5, 8],
            _state(),
            True,
            coupling=StaticActiveRequest(enabled=True, scale=1.0, callback=active_callback),
        )
        self.assertEqual(off, baseline)
        self.assertEqual(zero, baseline)
        self.assertNotEqual(active, baseline)
        self.assertEqual(zero_callback.calls, [])
        self.assertEqual(len(active_callback.calls), 4)
        self.assertEqual(sum(call["applied"] for call in active_callback.calls), 2)
        self.assertNotIn(CALLBACK_ATTRIBUTE, fixture.__dict__)
        self.assertNotIn("forward_one", fixture.__dict__)
        self.assertNotIn("forward_seq", fixture.__dict__)

    def test_runtime_rejects_real_or_unmarked_fixture(self) -> None:
        namespace, fixture_class = _synthetic_namespace()
        digest = hashlib.sha256(SYNTHETIC_SOURCE.encode("utf-8")).hexdigest()
        for field, value in (
            ("offline_static_fixture", False),
            ("model_loaded", True),
            ("model_executed", True),
        ):
            fixture = fixture_class()
            setattr(fixture, field, value)
            with mock.patch(
                "psa.self_model.d5b_static_active.EXPECTED_RWKV_MODEL_SOURCE_SHA256",
                digest,
            ):
                with self.assertRaises(PermissionError):
                    RWKV7ProjectStaticActiveRuntime(
                        base_fixture=fixture,
                        upstream_source_bytes=SYNTHETIC_SOURCE.encode("utf-8"),
                        upstream_globals=namespace,
                        upstream_package_version="0.8.32",
                        upstream_de_version=None,
                    )

    def test_real_hidden_shape_and_nonfinite_fail_closed(self) -> None:
        single = StaticResidualTensor.from_tokens([3], squeeze=True)
        sequence = StaticResidualTensor.from_tokens([3, 5, 8], squeeze=False)
        self.assertEqual(single.shape, (2560,))
        self.assertEqual(sequence.shape, (3, 2560))
        self.assertEqual(single.add_broadcast(_vector()).shape, (2560,))
        self.assertEqual(sequence.add_broadcast(_vector()).shape, (3, 2560))
        with self.assertRaises(ValueError):
            single.add_broadcast([0.0] * 2559)
        invalid = list(_vector())
        invalid[4] = float("nan")
        with self.assertRaises(ValueError):
            StaticSyntheticCallback(
                vector=invalid,
                layer_mask=["fake-layer-01", "fake-layer-03"],
                scale=1.0,
                gate=0.5,
            )

    def test_request_and_contract_scope_fail_closed(self) -> None:
        with self.assertRaises(PermissionError):
            StaticActiveRequest(enabled=True, scale=1.0, callback=None)
        with self.assertRaises(ValueError):
            StaticActiveRequest(enabled=True, scale=0.5, callback=_callback(1.0))
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        for section, field, value in (
            ("authority", "d5c_real_model_mechanism_smoke_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("authority", "self_updater_authorized", True),
            ("synthetic_callback", "layer_mask_is_real_selection", True),
            ("upstream_lock", "installed_source_probed_this_round", True),
        ):
            changed = copy.deepcopy(config)
            changed[section][field] = value
            with self.assertRaises(PermissionError):
                validate_contract(changed)

    def test_module_has_no_rwkv_or_torch_import(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertNotIn("rwkv", imported)
        self.assertNotIn("torch", imported)

    def test_report_digest_and_alternate_path(self) -> None:
        report = build_d5b_report(config_path=CONFIG, project_root=ROOT)
        digest = report.pop("report_digest_sha256")
        self.assertEqual(digest, sha256_json(report))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_d5b_report(config_path=path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
