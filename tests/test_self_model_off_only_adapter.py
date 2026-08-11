from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from psa.self_model.off_only_adapter_manifest import (
    IMPLEMENTATION_CONFIG_FILE,
    audit_off_only_adapter_source,
    build_off_only_adapter_report,
    validate_off_only_implementation_config,
)
from psa.self_model.rwkv7_coupling_adapter import (
    CouplingOffRequest,
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
    RWKV7CouplingOffAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / IMPLEMENTATION_CONFIG_FILE
DESIGN = (
    ROOT
    / "configs/development/self_model_v0_1_real_adapter_off_design.draft.json"
)


class FakeBase:
    def __init__(self) -> None:
        self.logits = object()
        self.calls = []

    def forward(self, tokens, state, full_output=False):
        self.calls.append((tokens, state, full_output))
        state["count"] += 1
        return self.logits, state


class FailingBase:
    def forward(self, tokens, state, full_output=False):
        raise LookupError("sentinel upstream failure")


def _adapter(base=None):
    return RWKV7CouplingOffAdapter(
        base_model=base or FakeBase(),
        upstream_package_version=EXPECTED_RWKV_PACKAGE_VERSION,
        upstream_model_source_sha256=EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    )


class OffOnlyAdapterTests(unittest.TestCase):
    def test_report_verifies_d2_without_model_import(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        report = build_off_only_adapter_report(
            config_path=CONFIG,
            project_root=ROOT,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["checks"]), 29)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(report["source_digests"]), 8)
        self.assertTrue(report["safety"]["off_only_adapter_implemented"])
        self.assertFalse(report["safety"]["active_injection_implemented"])
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)

    def test_off_delegation_preserves_input_and_output_identity(self) -> None:
        base = FakeBase()
        adapter = _adapter(base)
        tokens = [3, 5, 8]
        state = {"count": 0}
        logits, next_state = adapter.forward(
            tokens,
            state,
            True,
            coupling=CouplingOffRequest(),
        )
        self.assertIs(base.calls[0][0], tokens)
        self.assertIs(base.calls[0][1], state)
        self.assertIs(base.calls[0][2], True)
        self.assertIs(logits, base.logits)
        self.assertIs(next_state, state)
        self.assertEqual(state["count"], 1)
        self.assertEqual(adapter.delegation_count, 1)
        self.assertEqual(adapter.callback_call_count, 0)
        self.assertFalse(adapter.self_projection_constructed)

    def test_default_request_is_off_and_full_output_false_is_preserved(self) -> None:
        base = FakeBase()
        adapter = _adapter(base)
        state = {"count": 0}
        adapter.forward([1], state)
        self.assertIs(base.calls[0][2], False)
        self.assertEqual(state["count"], 1)

    def test_active_and_malformed_requests_are_rejected_before_base_call(self) -> None:
        base = FakeBase()
        adapter = _adapter(base)

        class CouplingOffSubclass(CouplingOffRequest):
            pass

        with self.assertRaises(PermissionError):
            adapter.forward([1], {"count": 0}, coupling={"enabled": True})
        with self.assertRaises(PermissionError):
            adapter.forward(
                [1], {"count": 0}, coupling=CouplingOffSubclass()
            )
        with self.assertRaises(PermissionError):
            adapter.forward_active([1], {"count": 0})
        self.assertEqual(base.calls, [])
        for values in (
            {"mode": "active"},
            {"enabled": True},
            {"scale": 0.5},
        ):
            with self.assertRaises(PermissionError):
                CouplingOffRequest(**values)

    def test_source_version_and_digest_mismatch_fail_before_delegation(self) -> None:
        for version, digest in (
            ("0.8.33", EXPECTED_RWKV_MODEL_SOURCE_SHA256),
            (EXPECTED_RWKV_PACKAGE_VERSION, "0" * 64),
        ):
            base = FakeBase()
            with self.assertRaises(RuntimeError):
                RWKV7CouplingOffAdapter(
                    base_model=base,
                    upstream_package_version=version,
                    upstream_model_source_sha256=digest,
                )
            self.assertEqual(base.calls, [])

    def test_upstream_exception_is_not_rewritten(self) -> None:
        adapter = _adapter(FailingBase())
        with self.assertRaisesRegex(LookupError, "sentinel upstream failure"):
            adapter.forward([1], {"count": 0})

    def test_static_source_audit_rejects_model_import_or_instrumentation(self) -> None:
        clean = (
            ROOT / "src/psa/self_model/rwkv7_coupling_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertTrue(all(audit_off_only_adapter_source(clean).values()))
        dirty = clean + "\nimport torch\nimport rwkv\n# post_ffn_residual\n"
        checks = audit_off_only_adapter_source(dirty)
        self.assertFalse(checks["no_rwkv_import"])
        self.assertFalse(checks["no_torch_import"])
        self.assertFalse(checks["no_instrumented_phase_marker"])

    def test_config_escalation_or_alternate_path_fails_closed(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        config["authority"]["active_injection_implementation_authorized"] = True
        with self.assertRaises(PermissionError):
            validate_off_only_implementation_config(config, design)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_off_only_adapter_report(
                    config_path=path,
                    project_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
