from __future__ import annotations

import hashlib
import sys
import unittest

from psa.self_model.rwkv_interface_audit import inspect_rwkv_source


SOURCE = """
MyModule = torch.nn.Module
class RWKV_x070(MyModule):
    def generate_zero_state(self):
        state = [None for _ in range(self.args.n_layer * 3)]
        for i in range(self.args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
            pass
    def forward(self, idx, state, full_output=False):
        if state == None:
            state = self.generate_zero_state()
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)
    def forward_one(self, idx, state):
        x = z['emb.weight'][idx]
        for i in range(self.n_layer):
            xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one()
            x = x + xx
            xx, state[i*3+2] = RWKV_x070_CMix_one()
            x = x + xx
        x = F.layer_norm(x, (self.n_embd,))
        x = x @ z['head.weight']
    def forward_seq(self, idx, state, full_output=False):
        x = z['emb.weight'][idx]
if os.environ.get('RWKV_V7_ON') == '1':
    RWKV = RWKV_x070
""".strip()


class RWKVInterfaceAuditTests(unittest.TestCase):
    def test_fixture_passes_without_importing_rwkv_model_or_torch(self) -> None:
        before_rwkv = "rwkv.model" in sys.modules
        before_torch = "torch" in sys.modules
        digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
        report = inspect_rwkv_source(
            source=SOURCE,
            package_version="0.8.32",
            source_sha256=digest,
            expected_version="0.8.32",
            expected_source_sha256=digest,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(before_rwkv, "rwkv.model" in sys.modules)
        self.assertEqual(before_torch, "torch" in sys.modules)

    def test_version_or_source_drift_fails_closed(self) -> None:
        report = inspect_rwkv_source(
            source=SOURCE,
            package_version="0.8.33",
            source_sha256="0" * 64,
            expected_version="0.8.32",
            expected_source_sha256="1" * 64,
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["package_version_matches"])
        self.assertFalse(report["checks"]["source_sha256_matches"])

    def test_missing_state_update_marker_fails_closed(self) -> None:
        source = SOURCE.replace("state[i*3+2]", "new_state[i*3+2]")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        report = inspect_rwkv_source(
            source=source,
            package_version="0.8.32",
            source_sha256=digest,
            expected_version="0.8.32",
            expected_source_sha256=digest,
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["all_required_source_markers_present"])


if __name__ == "__main__":
    unittest.main()
