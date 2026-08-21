from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from psa.self_model import d5c_mechanism_runtime as runtime_module
from psa.self_model import d5c_real_mechanism as module
from psa.self_model.d5c_mechanism_runtime import (
    D5CCouplingRequest,
    D5CSyntheticProbe,
    HIDDEN_DIMENSION,
    RWKV7D5CActiveRuntime,
    ROUTES,
    SCORED_ROUNDS,
    TARGET_LAYER_INDEX,
    deterministic_unit_rms_vector,
)
from psa.self_model.d5b_static_active import (
    CALLBACK_ATTRIBUTE,
    SYNTHETIC_SOURCE,
    _state,
    _synthetic_namespace,
)
from psa.self_model.d5c_real_mechanism import (
    AUTHORIZATION_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    D5C_EXECUTION_LOCK_ENV,
    D5C_OWNER_AUTHORIZATION_TEXT,
    OUTPUT_RELATIVE_DIR,
    _read_spec,
    build_d5c_entry_static_report,
    run_d5c_real_mechanism,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class FakeBoolean:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, values, *, dtype="float16", device="fake-cuda:0"):
        self.values = float(values) if isinstance(values, (int, float)) else tuple(float(value) for value in values)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return () if isinstance(self.values, float) else (len(self.values),)

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.values, dtype="float32", device=self.device)

    def square(self):
        if isinstance(self.values, float):
            return FakeTensor(self.values * self.values, dtype=self.dtype, device=self.device)
        return FakeTensor((value * value for value in self.values), dtype=self.dtype, device=self.device)

    def mean(self):
        values = (self.values,) if isinstance(self.values, float) else self.values
        return FakeTensor(sum(values) / len(values), dtype=self.dtype, device=self.device)

    def sqrt(self):
        return FakeTensor(math.sqrt(self.values), dtype=self.dtype, device=self.device)

    def item(self):
        return self.values

    def to(self, *, dtype):
        return FakeTensor(self.values, dtype=dtype, device=self.device)

    def __mul__(self, other):
        factor = other.values if isinstance(other, FakeTensor) else float(other)
        if isinstance(self.values, float):
            values = self.values * factor
        else:
            values = (value * factor for value in self.values)
        return FakeTensor(values, dtype=self.dtype, device=self.device)

    def __add__(self, other):
        return FakeTensor(
            (left + right for left, right in zip(self.values, other.values)),
            dtype=self.dtype,
            device=self.device,
        )


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def isfinite(value):
        values = (value.values,) if isinstance(value.values, float) else value.values
        return FakeBoolean(all(math.isfinite(item) for item in values))

    @staticmethod
    def tensor(values, *, device, dtype):
        return FakeTensor(values, dtype=dtype, device=device)


class D5CRealMechanismTests(unittest.TestCase):
    def test_config_freezes_mechanism_only_schedule(self):
        spec = _read_spec(CONFIG)
        self.assertEqual(spec["routes"], list(ROUTES))
        self.assertEqual(spec["scored_rounds"], [list(value) for value in SCORED_ROUNDS])
        self.assertEqual(spec["counts"]["model_forward_calls_total"], 42)
        self.assertEqual(spec["counts"]["active_callback_calls_total"], 320)
        self.assertEqual(spec["counts"]["active_probe_applications_total"], 10)
        self.assertEqual(TARGET_LAYER_INDEX, 15)
        self.assertFalse(spec["execution_authorized_at_implementation"])
        self.assertFalse(spec["self_effect_conclusion_authorized"])
        self.assertFalse(spec["d5d_authorized"])

    def test_scope_changes_fail_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("execution_authorized_at_implementation", True),
            ("automatic_rerun_authorized", True),
            ("d5d_authorized", True),
            ("formal_test_set_authorized", True),
            ("self_effect_conclusion_authorized", True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field, value in mutations:
                changed = copy.deepcopy(payload)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(PermissionError):
                    _read_spec(path)

    def test_synthetic_vector_is_deterministic_unit_rms(self):
        left = deterministic_unit_rms_vector()
        right = deterministic_unit_rms_vector()
        self.assertEqual(left, right)
        self.assertEqual(len(left), HIDDEN_DIMENSION)
        rms = math.sqrt(sum(value * value for value in left) / len(left))
        self.assertAlmostEqual(rms, 1.0, places=12)
        self.assertTrue(all(math.isfinite(value) for value in left))

    def test_off_and_zero_requests_cannot_bind_callback(self):
        D5CCouplingRequest(enabled=False, scale=0.0, callback=None)
        D5CCouplingRequest(enabled=True, scale=0.0, callback=None)
        with self.assertRaises(PermissionError):
            D5CCouplingRequest(enabled=True, scale=0.5, callback=None)
        with self.assertRaises(PermissionError):
            D5CCouplingRequest(enabled=False, scale=0.0, callback=object())

    def test_probe_visits_all_layers_and_applies_only_layer_15(self):
        probe = D5CSyntheticProbe(
            torch=FakeTorch(), execution_claim_sha256="a" * 64,
            machine_authorization_sha256="b" * 64,
        )
        residual = FakeTensor((1.0 for _ in range(HIDDEN_DIMENSION)))
        for layer_index in range(32):
            output = probe(
                phase="post_ffn_residual", layer_index=layer_index,
                execution_path="forward_one", residual_x=residual,
            )
            if layer_index == TARGET_LAYER_INDEX:
                self.assertNotEqual(output.values, residual.values)
            else:
                self.assertIs(output, residual)
        self.assertEqual(probe.invocation_count, 32)
        self.assertEqual(probe.application_count, 1)
        self.assertEqual(probe.applications[0]["layer_index"], 15)
        self.assertEqual(output.shape, residual.shape)

    def test_real_runtime_off_path_restores_temporary_bindings(self):
        namespace, fixture_class = _synthetic_namespace()
        fixture = fixture_class()
        source_bytes = SYNTHETIC_SOURCE.encode("utf-8")
        digest = __import__("hashlib").sha256(source_bytes).hexdigest()
        with patch.object(runtime_module, "EXPECTED_RWKV_MODEL_SOURCE_SHA256", digest):
            runtime = RWKV7D5CActiveRuntime(
                base_model=fixture, upstream_source_bytes=source_bytes,
                upstream_globals=namespace, upstream_package_version="0.8.32",
                upstream_de_version=None, execution_claim_sha256="a" * 64,
                machine_authorization_sha256="b" * 64,
            )
            logits, state = runtime.forward(
                [3], _state(), False,
                coupling=D5CCouplingRequest(enabled=False, scale=0.0, callback=None),
            )
        self.assertEqual(logits.shape, (HIDDEN_DIMENSION,))
        self.assertEqual(len(state), 12)
        self.assertNotIn(CALLBACK_ATTRIBUTE, fixture.__dict__)
        self.assertNotIn("forward_one", fixture.__dict__)
        self.assertNotIn("forward_seq", fixture.__dict__)

    def test_missing_lock_fails_before_git_model_or_claim(self):
        environment = dict(os.environ)
        environment.pop(D5C_EXECUTION_LOCK_ENV, None)
        environment.pop("RWKV_DE_VERSION", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            module, "_git_metadata"
        ) as git, patch.object(module.RWKV7Adapter, "load") as load, patch.object(
            module, "_create_claim"
        ) as claim:
            with self.assertRaises(PermissionError):
                run_d5c_real_mechanism(
                    config_path=CONFIG,
                    authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
                    project_root=ROOT,
                    output_dir=ROOT / OUTPUT_RELATIVE_DIR,
                )
        git.assert_not_called()
        load.assert_not_called()
        claim.assert_not_called()

    def test_cli_missing_lock_creates_no_authorization_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(os.environ)
            environment.pop(D5C_EXECUTION_LOCK_ENV, None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_self_model_v0_1_coupling_d5c_real_mechanism.py"),
                    "--project-root", str(root),
                ],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("single-use lock is absent", completed.stderr)
            self.assertFalse((root / AUTHORIZATION_RELATIVE_PATH).exists())
            self.assertFalse((root / OUTPUT_RELATIVE_DIR).exists())

    def test_owner_text_is_exact_and_mechanism_only(self):
        spec = _read_spec(CONFIG)
        self.assertEqual(spec["required_owner_authorization_text"], D5C_OWNER_AUTHORIZATION_TEXT)
        self.assertIn("42次调用", D5C_OWNER_AUTHORIZATION_TEXT)
        self.assertIn("不授权", D5C_OWNER_AUTHORIZATION_TEXT)
        self.assertFalse(spec["probe"]["real_self_projection"])

    def test_static_report_is_no_model_and_complete(self):
        report = build_d5c_entry_static_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["safety"]["rwkv_model_imported"])
        self.assertFalse(report["safety"]["torch_imported"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["execution_claim_created"])
        self.assertFalse(report["safety"]["machine_authorization_created"])


if __name__ == "__main__":
    unittest.main()
