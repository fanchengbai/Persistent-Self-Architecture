from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPECTED_RWKV_PACKAGE_VERSION = "0.8.32"
EXPECTED_RWKV_MODEL_SOURCE_SHA256 = (
    "75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0"
)


@dataclass(frozen=True)
class CouplingOffRequest:
    mode: str = "off"
    enabled: bool = False
    scale: float = 0.0

    def __post_init__(self) -> None:
        if self.mode != "off" or self.enabled is not False or self.scale != 0.0:
            raise PermissionError("D2 only accepts an exact coupling-off request")


class RWKV7CouplingOffAdapter:
    """OFF-G1 passthrough only; no callback or active injection exists."""

    adapter_version = "0.1-off-only"
    development_only = True
    off_g1_implemented = True
    off_g2_implemented = False
    active_injection_available = False
    real_layers_selected = False
    real_sequence_policy_frozen = False

    def __init__(
        self,
        *,
        base_model: Any,
        upstream_package_version: str,
        upstream_model_source_sha256: str,
    ) -> None:
        if upstream_package_version != EXPECTED_RWKV_PACKAGE_VERSION:
            raise RuntimeError("RWKV package version differs from the D2 source lock")
        if upstream_model_source_sha256 != EXPECTED_RWKV_MODEL_SOURCE_SHA256:
            raise RuntimeError("RWKV model source differs from the D2 source lock")
        forward = getattr(base_model, "forward", None)
        if not callable(forward):
            raise TypeError("base_model must expose a callable forward")
        self._base_model = base_model
        self._delegation_count = 0

    @property
    def delegation_count(self) -> int:
        return self._delegation_count

    @property
    def callback_call_count(self) -> int:
        return 0

    @property
    def self_projection_constructed(self) -> bool:
        return False

    def forward(
        self,
        tokens: Any,
        state: Any,
        full_output: bool = False,
        *,
        coupling: CouplingOffRequest | None = None,
    ) -> Any:
        request = CouplingOffRequest() if coupling is None else coupling
        if type(request) is not CouplingOffRequest:
            raise PermissionError("D2 rejects non-off coupling requests")
        if request.mode != "off" or request.enabled or request.scale != 0.0:
            raise PermissionError("D2 active coupling is unavailable")
        self._delegation_count += 1
        return self._base_model.forward(tokens, state, full_output)

    def forward_active(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("active injection is not implemented or authorized in D2")
