from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json


FAKE_CALLBACK_CONTRACT_VERSION = "0.1-no-weight"
FAKE_CALLBACK_CONFIG_FILE = (
    "configs/development/self_model_v0_1_fake_callback.draft.json"
)
FAKE_CALLBACK_SOURCE_FILES = (
    FAKE_CALLBACK_CONFIG_FILE,
    "docs/self_model_v0_1_fake_callback_contract.md",
    "schemas/self_model_v0_1_fake_callback_report.schema.json",
    "scripts/verify_self_model_v0_1_fake_callback.py",
    "src/psa/self_model/fake_callback_runtime.py",
    "tests/test_self_model_fake_callback.py",
)


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} must be a non-empty finite vector")
    return vector


@dataclass(frozen=True)
class FakeResidualTensor:
    values: tuple[float, ...] | tuple[tuple[float, ...], ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("fake residual dtype is unsupported")
        if not self.device.startswith("fake-"):
            raise ValueError("fake residual device must be visibly synthetic")
        if not self.values:
            raise ValueError("fake residual tensor cannot be empty")
        first = self.values[0]
        if isinstance(first, tuple):
            rows = self.values
            width = len(first)
            if width < 1 or any(
                not isinstance(row, tuple) or len(row) != width
                for row in rows
            ):
                raise ValueError("fake sequence residual must be rectangular")
            flattened = [value for row in rows for value in row]
        else:
            if any(isinstance(value, tuple) for value in self.values):
                raise ValueError("fake single-token residual must be one-dimensional")
            flattened = list(self.values)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in flattened
        ):
            raise ValueError("fake residual values must be finite numbers")

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.values[0], tuple):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    @property
    def hidden_dimension(self) -> int:
        return self.shape[-1]

    def add_broadcast(self, vector: Sequence[float]) -> "FakeResidualTensor":
        addition = _finite_vector(vector, "broadcast residual")
        if len(addition) != self.hidden_dimension:
            raise ValueError("broadcast residual dimension mismatch")
        if len(self.shape) == 1:
            values = tuple(
                float(value) + delta
                for value, delta in zip(self.values, addition)
            )
        else:
            values = tuple(
                tuple(
                    float(value) + delta
                    for value, delta in zip(row, addition)
                )
                for row in self.values
            )
        return FakeResidualTensor(values=values, dtype=self.dtype, device=self.device)


@dataclass(frozen=True)
class ResidualCallbackRequest:
    phase: str
    layer_index: int
    layer_name: str
    execution_path: str
    residual_x: FakeResidualTensor
    self_vector: tuple[float, ...]


class FakePostFFNResidualCallback:
    offline_fake_callback = True
    model_loaded = False
    model_executed = False
    phase = "post_ffn_residual"

    def __init__(
        self,
        *,
        hidden_dimension: int,
        layer_mask: Sequence[str],
        enabled: bool,
        scale: float,
        gate: float,
        sequence_policy: str = "broadcast_all_tokens_fake_only",
    ) -> None:
        if hidden_dimension < 1:
            raise ValueError("hidden_dimension must be positive")
        layers = tuple(layer_mask)
        if (
            not layers
            or len(layers) != len(set(layers))
            or any(
                not layer.startswith("fake-layer-")
                or not layer.removeprefix("fake-layer-").isdigit()
                for layer in layers
            )
        ):
            raise ValueError("fake callback layer mask is invalid")
        if (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not 0.0 <= float(scale) <= 2.0
        ):
            raise ValueError("fake callback scale must be in [0, 2]")
        if (
            not isinstance(gate, (int, float))
            or isinstance(gate, bool)
            or not 0.0 <= float(gate) <= 1.0
        ):
            raise ValueError("fake callback gate must be in [0, 1]")
        if sequence_policy != "broadcast_all_tokens_fake_only":
            raise ValueError("unsupported fake sequence policy")
        self.hidden_dimension = hidden_dimension
        self.layer_mask = layers
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.gate = float(gate)
        self.sequence_policy = sequence_policy
        self.calls: list[dict[str, Any]] = []

    def should_apply(self, *, phase: str, layer_name: str) -> bool:
        return bool(
            self.enabled
            and self.scale > 0.0
            and phase == self.phase
            and layer_name in self.layer_mask
        )

    def apply(self, request: ResidualCallbackRequest) -> FakeResidualTensor:
        if not self.should_apply(
            phase=request.phase, layer_name=request.layer_name
        ):
            raise RuntimeError("inactive fake callback was invoked")
        if request.phase != self.phase:
            raise ValueError("fake callback phase mismatch")
        if request.layer_name not in self.layer_mask:
            raise ValueError("fake callback layer is outside the mask")
        if request.layer_name != f"fake-layer-{request.layer_index:02d}":
            raise ValueError("fake callback layer index and name differ")
        if request.execution_path not in {"forward_one", "forward_seq"}:
            raise ValueError("fake callback execution path is invalid")
        vector = _finite_vector(request.self_vector, "Self vector")
        if (
            len(vector) != self.hidden_dimension
            or request.residual_x.hidden_dimension != self.hidden_dimension
        ):
            raise ValueError("fake callback hidden dimension mismatch")
        addition = tuple(self.scale * self.gate * value for value in vector)
        result = request.residual_x.add_broadcast(addition)
        if result.shape != request.residual_x.shape:
            raise RuntimeError("fake callback changed residual shape")
        if (
            result.dtype != request.residual_x.dtype
            or result.device != request.residual_x.device
        ):
            raise RuntimeError("fake callback changed residual metadata")
        self.calls.append(
            {
                "phase": request.phase,
                "layer_index": request.layer_index,
                "layer_name": request.layer_name,
                "execution_path": request.execution_path,
                "input_shape": list(request.residual_x.shape),
                "output_shape": list(result.shape),
                "dtype": result.dtype,
                "device": result.device,
                "sequence_policy": self.sequence_policy,
                "addition_digest_sha256": sha256_json(list(addition)),
            }
        )
        return result


class FakeRWKV7ResidualRuntime:
    """No-weight structural fixture for the two RWKV-7 forward paths."""

    offline_fake_runtime = True
    model_loaded = False
    model_executed = False

    def __init__(
        self,
        *,
        n_layer: int,
        hidden_dimension: int,
        dtype: str,
        device: str,
    ) -> None:
        if n_layer < 1 or hidden_dimension < 1:
            raise ValueError("fake runtime dimensions must be positive")
        self.n_layer = n_layer
        self.hidden_dimension = hidden_dimension
        self.dtype = dtype
        self.device = device
        FakeResidualTensor(
            values=tuple(0.0 for _ in range(hidden_dimension)),
            dtype=dtype,
            device=device,
        )

    def zero_state(self) -> list[list[float]]:
        return [
            [0.0 for _ in range(self.hidden_dimension)]
            for _ in range(self.n_layer * 3)
        ]

    def _clone_state(self, state: Sequence[Sequence[float]]) -> list[list[float]]:
        if len(state) != self.n_layer * 3:
            raise ValueError("fake state must contain three components per layer")
        cloned = []
        for component in state:
            vector = list(_finite_vector(component, "fake state component"))
            if len(vector) != self.hidden_dimension:
                raise ValueError("fake state component dimension mismatch")
            cloned.append(vector)
        return cloned

    def _embedding(self, token: int) -> tuple[float, ...]:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError("fake token must be a non-negative integer")
        return tuple(
            (((token + 1) * (index + 1)) % 17) / 17.0
            for index in range(self.hidden_dimension)
        )

    def _run_layers(
        self,
        *,
        residual: FakeResidualTensor,
        state: Sequence[Sequence[float]],
        execution_path: str,
        self_vector: Sequence[float],
        callback: FakePostFFNResidualCallback | None,
    ) -> tuple[FakeResidualTensor, list[list[float]]]:
        working_state = self._clone_state(state)
        vector = _finite_vector(self_vector, "Self vector")
        if len(vector) != self.hidden_dimension:
            raise ValueError("Self vector dimension mismatch")
        for layer_index in range(self.n_layer):
            layer_name = f"fake-layer-{layer_index:02d}"
            attention_delta = tuple(
                (layer_index + 1) * (index + 1) / 100.0
                for index in range(self.hidden_dimension)
            )
            residual = residual.add_broadcast(attention_delta)
            working_state[layer_index * 3] = list(
                residual.values[-1]
                if len(residual.shape) == 2
                else residual.values
            )
            working_state[layer_index * 3 + 1] = [
                value + 0.25 for value in working_state[layer_index * 3]
            ]

            ffn_delta = tuple(
                (layer_index + 1) * (index + 1) / 200.0
                for index in range(self.hidden_dimension)
            )
            residual = residual.add_broadcast(ffn_delta)
            working_state[layer_index * 3 + 2] = list(
                residual.values[-1]
                if len(residual.shape) == 2
                else residual.values
            )
            if callback is not None and callback.should_apply(
                phase="post_ffn_residual", layer_name=layer_name
            ):
                residual = callback.apply(
                    ResidualCallbackRequest(
                        phase="post_ffn_residual",
                        layer_index=layer_index,
                        layer_name=layer_name,
                        execution_path=execution_path,
                        residual_x=residual,
                        self_vector=vector,
                    )
                )
        return residual, working_state

    def forward_one(
        self,
        token: int,
        state: Sequence[Sequence[float]],
        *,
        self_vector: Sequence[float],
        callback: FakePostFFNResidualCallback | None = None,
    ) -> tuple[FakeResidualTensor, list[list[float]]]:
        residual = FakeResidualTensor(
            values=self._embedding(token), dtype=self.dtype, device=self.device
        )
        return self._run_layers(
            residual=residual,
            state=state,
            execution_path="forward_one",
            self_vector=self_vector,
            callback=callback,
        )

    def forward_seq(
        self,
        tokens: Sequence[int],
        state: Sequence[Sequence[float]],
        *,
        self_vector: Sequence[float],
        callback: FakePostFFNResidualCallback | None = None,
    ) -> tuple[FakeResidualTensor, list[list[float]]]:
        token_list = tuple(tokens)
        if len(token_list) < 2:
            raise ValueError("fake sequence path requires at least two tokens")
        residual = FakeResidualTensor(
            values=tuple(self._embedding(token) for token in token_list),
            dtype=self.dtype,
            device=self.device,
        )
        return self._run_layers(
            residual=residual,
            state=state,
            execution_path="forward_seq",
            self_vector=self_vector,
            callback=callback,
        )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def build_fake_callback_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / FAKE_CALLBACK_CONFIG_FILE).resolve():
        raise PermissionError("fake callback config path is not the frozen draft")
    config = _load_object(config_file, "fake callback config")
    if config.get("contract_version") != FAKE_CALLBACK_CONTRACT_VERSION:
        raise ValueError("unsupported fake callback contract version")
    authority = config.get("authority")
    required_authority = {
        "no_weight_fake_runtime_authorized": True,
        "rwkv_model_import_authorized": False,
        "weights_access_authorized": False,
        "model_execution_authorized": False,
        "site_packages_modification_authorized": False,
        "real_layer_selection_authorized": False,
        "self_effect_experiment_authorized": False,
    }
    if not isinstance(authority, Mapping) or any(
        authority.get(key) is not value
        for key, value in required_authority.items()
    ):
        raise PermissionError("fake callback authority must remain no-weight only")

    runtime_config = config.get("fake_runtime")
    callback_config = config.get("fake_callback")
    if not isinstance(runtime_config, Mapping) or not isinstance(
        callback_config, Mapping
    ):
        raise ValueError("fake runtime and callback configs are required")
    if (
        callback_config.get("phase") != "post_ffn_residual"
        or callback_config.get("real_sequence_policy_frozen") is not False
        or callback_config.get("real_layers_selected") is not False
    ):
        raise PermissionError("fake callback cannot freeze real coupling choices")
    runtime = FakeRWKV7ResidualRuntime(
        n_layer=int(runtime_config["n_layer"]),
        hidden_dimension=int(runtime_config["hidden_dimension"]),
        dtype=str(runtime_config["dtype"]),
        device=str(runtime_config["device"]),
    )
    self_vector = _finite_vector(config["self_vector"], "Self vector")
    source_state = runtime.zero_state()
    source_snapshot = [list(component) for component in source_state]

    def callback(*, enabled: bool, scale: float) -> FakePostFFNResidualCallback:
        return FakePostFFNResidualCallback(
            hidden_dimension=runtime.hidden_dimension,
            layer_mask=callback_config["layer_mask"],
            enabled=enabled,
            scale=scale,
            gate=float(callback_config["gate"]),
            sequence_policy=str(callback_config["sequence_policy"]),
        )

    baseline_one = runtime.forward_one(
        3, source_state, self_vector=self_vector, callback=None
    )
    off_one_callback = callback(enabled=False, scale=1.0)
    off_one = runtime.forward_one(
        3, source_state, self_vector=self_vector, callback=off_one_callback
    )
    zero_one_callback = callback(enabled=True, scale=0.0)
    zero_one = runtime.forward_one(
        3, source_state, self_vector=self_vector, callback=zero_one_callback
    )
    active_one_callback = callback(enabled=True, scale=1.0)
    active_one = runtime.forward_one(
        3, source_state, self_vector=self_vector, callback=active_one_callback
    )

    tokens = (3, 5, 8)
    baseline_seq = runtime.forward_seq(
        tokens, source_state, self_vector=self_vector, callback=None
    )
    off_seq_callback = callback(enabled=False, scale=1.0)
    off_seq = runtime.forward_seq(
        tokens, source_state, self_vector=self_vector, callback=off_seq_callback
    )
    zero_seq_callback = callback(enabled=True, scale=0.0)
    zero_seq = runtime.forward_seq(
        tokens, source_state, self_vector=self_vector, callback=zero_seq_callback
    )
    active_seq_callback = callback(enabled=True, scale=1.0)
    active_seq = runtime.forward_seq(
        tokens, source_state, self_vector=self_vector, callback=active_seq_callback
    )

    expected_calls = len(callback_config["layer_mask"])
    active_calls = active_one_callback.calls + active_seq_callback.calls
    checks = {
        "source_state_unchanged": source_state == source_snapshot,
        "self_vector_unchanged": self_vector
        == _finite_vector(config["self_vector"], "Self vector"),
        "forward_one_off_exact": off_one == baseline_one,
        "forward_seq_off_exact": off_seq == baseline_seq,
        "forward_one_zero_scale_exact": zero_one == baseline_one,
        "forward_seq_zero_scale_exact": zero_seq == baseline_seq,
        "off_and_zero_callbacks_not_called": not (
            off_one_callback.calls
            or off_seq_callback.calls
            or zero_one_callback.calls
            or zero_seq_callback.calls
        ),
        "forward_one_active_changes_residual": active_one[0] != baseline_one[0],
        "forward_seq_active_changes_residual": active_seq[0] != baseline_seq[0],
        "active_callback_call_counts_match": (
            len(active_one_callback.calls) == expected_calls
            and len(active_seq_callback.calls) == expected_calls
        ),
        "both_execution_paths_observed": {
            call["execution_path"] for call in active_calls
        }
        == {"forward_one", "forward_seq"},
        "only_post_ffn_phase_observed": {
            call["phase"] for call in active_calls
        }
        == {"post_ffn_residual"},
        "shape_preserved": all(
            call["input_shape"] == call["output_shape"] for call in active_calls
        ),
        "dtype_preserved": all(
            call["dtype"] == runtime.dtype for call in active_calls
        ),
        "device_preserved": all(
            call["device"] == runtime.device for call in active_calls
        ),
        "callback_request_excludes_recurrent_state": set(
            ResidualCallbackRequest.__dataclass_fields__
        )
        == {
            "phase",
            "layer_index",
            "layer_name",
            "execution_path",
            "residual_x",
            "self_vector",
        },
    }
    valid = all(checks.values())
    report = {
        "contract_version": FAKE_CALLBACK_CONTRACT_VERSION,
        "status": (
            "self_model_v0_1_fake_callback_verified"
            if valid
            else "self_model_v0_1_fake_callback_failed"
        ),
        "valid": valid,
        "development_only": True,
        "checks": checks,
        "runtime": {
            "kind": "no_weight_fake_rwkv7_residual_runtime",
            "n_layer": runtime.n_layer,
            "hidden_dimension": runtime.hidden_dimension,
            "dtype": runtime.dtype,
            "device": runtime.device,
            "state_components_per_layer": 3,
            "execution_paths": ["forward_one", "forward_seq"],
        },
        "callback": {
            "phase": "post_ffn_residual",
            "layer_mask": list(callback_config["layer_mask"]),
            "sequence_policy": callback_config["sequence_policy"],
            "real_sequence_policy_frozen": False,
            "real_layers_selected": False,
            "active_calls": active_calls,
        },
        "source_digests": {
            relative: sha256_file(root / relative)
            for relative in FAKE_CALLBACK_SOURCE_FILES
        },
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "site_packages_modified": False,
            "real_hook_implemented": False,
            "real_layers_selected": False,
            "self_effect_experiment_run": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
