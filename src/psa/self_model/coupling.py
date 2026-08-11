from __future__ import annotations

import math
from typing import Any, Sequence

from psa.artifacts import sha256_json
from psa.self_model.encoding import EncodedSelf, encoded_self_digest


class FakeGatedResidualAdapter:
    """Records offline residual injections without importing or loading a model."""

    offline_fake_adapter = True
    model_loaded = False

    def __init__(
        self,
        *,
        hidden_dimension: int,
        layers: Sequence[str] = ("fake-layer-00", "fake-layer-01"),
        gate: float = 0.5,
    ) -> None:
        if hidden_dimension < 1 or not 0.0 <= gate <= 1.0:
            raise ValueError("fake coupling adapter configuration is invalid")
        self.hidden_dimension = hidden_dimension
        self.layers = tuple(layers)
        self.gate = float(gate)
        self.calls: list[dict[str, Any]] = []

    def project(self, vector: Sequence[float], layer: str) -> tuple[float, ...]:
        if layer not in self.layers or len(vector) != self.hidden_dimension:
            raise ValueError("fake coupling projection input is invalid")
        return tuple(float(value) for value in vector)

    def inject(self, *, layer: str, residual: Sequence[float], gate: float) -> None:
        self.calls.append(
            {"layer": layer, "residual": tuple(residual), "gate": float(gate)}
        )


def apply_offline_gated_injection(
    adapter: Any,
    encoded: EncodedSelf,
    *,
    enabled: bool,
    scale: float,
    layer_mask: Sequence[str],
) -> dict[str, Any]:
    if (
        getattr(adapter, "offline_fake_adapter", False) is not True
        or getattr(adapter, "model_loaded", None) is not False
        or encoded.offline_fake_encoder is not True
        or encoded.model_loaded is not False
        or encoded.prompt_serialization_used is not False
    ):
        raise PermissionError("Self v0.1 offline coupling requires unloaded fake adapters")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not 0.0 <= float(scale) <= 2.0:
        raise ValueError("Self coupling scale must be in [0, 2]")
    layers = tuple(layer_mask)
    if not layers or len(layers) != len(set(layers)) or not set(layers) <= set(adapter.layers):
        raise ValueError("Self coupling layer mask is invalid")
    if encoded.dimension != adapter.hidden_dimension:
        raise ValueError("Self encoding and coupling dimensions differ")
    applied = bool(enabled and float(scale) > 0.0)
    injections = []
    if applied:
        for layer in layers:
            projected = adapter.project(encoded.aggregate_vector, layer)
            gate = float(adapter.gate)
            residual = tuple(float(scale) * gate * value for value in projected)
            adapter.inject(layer=layer, residual=residual, gate=gate)
            injections.append(
                {
                    "layer": layer,
                    "gate": gate,
                    "scale": float(scale),
                    "residual_l2_norm": math.sqrt(sum(value * value for value in residual)),
                    "residual_digest_sha256": sha256_json(list(residual)),
                }
            )
    return {
        "coupling_version": "0.1-offline-fake",
        "development_only": True,
        "synthetic_output_not_research_evidence": True,
        "enabled": bool(enabled),
        "applied": applied,
        "scale": float(scale),
        "active_fields": list(encoded.active_fields),
        "layer_mask": list(layers),
        "encoded_self_digest_sha256": encoded_self_digest(encoded),
        "injections": injections,
        "model_loaded": False,
        "model_executed": False,
        "real_rwkv_coupling_implemented": False,
        "formal_test_set_accessed": False,
    }
