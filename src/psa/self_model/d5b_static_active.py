from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.rwkv7_coupling_adapter import (
    EXPECTED_RWKV_MODEL_SOURCE_SHA256,
    EXPECTED_RWKV_PACKAGE_VERSION,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
    compile_instrumented_methods,
    inspect_instrumented_source,
)


CONTRACT_VERSION = "0.1-coupling-d5b-static-active"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d5b_static_active.json"
)
D5A_REPORT_DIGEST = "48c6f609f53a2d5223366abcc9b8b3af3936ede282fcd880113edb4f1cff89d3"
D5_DESIGN_REPORT_DIGEST = "d41dc30460060a85031df38b6cfb27ad8fa41b57cd77ba6f7e0940e75eb898fd"
D4B_REPORT_DIGEST = "8befb5f4b2ce90241b66aff1f43bce59645d367c14f6594169e9c454fcf36a20"
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D5B 项目内active路径静态集成与无模型验证；"
    "不授权Coupling-D5C/D5D/D5E、RWKV/Torch导入、权重访问、模型加载或执行、"
    "真实层选择、真实Self projection构造、Self效果实验、Self Updater或自动重跑。"
)
REQUIRED_NEXT_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D5C 真实2.9B非Core机制冒烟设计与无模型安全入口实现；"
    "不授权模型加载或执行、D5D/D5E、正式测试集、Self效果结论、Self Updater或自动重跑。"
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d5b_implementation_authorization.md",
    "docs/self_model_v0_1_coupling_d5b_static_active.md",
    "scripts/verify_self_model_v0_1_coupling_d5b_static_active.py",
    "src/psa/self_model/d5b_static_active.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d5b_static_active.py",
)


@dataclass(frozen=True)
class StaticResidualTensor:
    values: tuple[float, ...] | tuple[tuple[float, ...], ...]
    dtype: str = "float16"
    device: str = "fake-cuda:0"

    def __post_init__(self) -> None:
        if self.dtype != "float16" or self.device != "fake-cuda:0" or not self.values:
            raise ValueError("D5B residual metadata is not the frozen fake shape")
        first = self.values[0]
        if isinstance(first, tuple):
            width = len(first)
            if width != 2560 or any(
                not isinstance(row, tuple) or len(row) != width for row in self.values
            ):
                raise ValueError("D5B sequence residual must have shape [T, 2560]")
            flattened = [value for row in self.values for value in row]
        else:
            if len(self.values) != 2560 or any(
                isinstance(value, tuple) for value in self.values
            ):
                raise ValueError("D5B single residual must have shape [2560]")
            flattened = list(self.values)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in flattened
        ):
            raise ValueError("D5B residual values must be finite")

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.values[0], tuple):
            return (len(self.values), 2560)
        return (2560,)

    def __add__(self, other: object) -> "StaticResidualTensor":
        if not isinstance(other, StaticResidualTensor) or other.shape != self.shape:
            return NotImplemented
        if len(self.shape) == 1:
            values = tuple(a + b for a, b in zip(self.values, other.values))
        else:
            values = tuple(
                tuple(a + b for a, b in zip(left, right))
                for left, right in zip(self.values, other.values)
            )
        return StaticResidualTensor(values=values, dtype=self.dtype, device=self.device)

    def add_broadcast(self, vector: Sequence[float]) -> "StaticResidualTensor":
        addition = tuple(float(value) for value in vector)
        if len(addition) != 2560 or not all(math.isfinite(value) for value in addition):
            raise ValueError("D5B broadcast vector must be finite [2560]")
        if len(self.shape) == 1:
            values = tuple(value + delta for value, delta in zip(self.values, addition))
        else:
            values = tuple(
                tuple(value + delta for value, delta in zip(row, addition))
                for row in self.values
            )
        return StaticResidualTensor(values=values, dtype=self.dtype, device=self.device)

    @classmethod
    def from_tokens(cls, tokens: Sequence[int], *, squeeze: bool) -> "StaticResidualTensor":
        token_list = tuple(tokens)
        if not token_list or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in token_list
        ):
            raise ValueError("D5B fake tokens are invalid")
        rows = tuple(
            tuple((((token + 1) * (index + 1)) % 29) / 29.0 for index in range(2560))
            for token in token_list
        )
        return cls(values=rows[0] if squeeze else rows)

    @classmethod
    def constant_like(cls, source: "StaticResidualTensor", value: float) -> "StaticResidualTensor":
        if not math.isfinite(value):
            raise ValueError("D5B fake delta must be finite")
        if len(source.shape) == 1:
            values = tuple(value for _ in range(2560))
        else:
            values = tuple(
                tuple(value for _ in range(2560)) for _ in range(source.shape[0])
            )
        return cls(values=values, dtype=source.dtype, device=source.device)


class StaticSyntheticCallback:
    offline_static_callback = True
    model_loaded = False
    model_executed = False
    real_self_projection = False
    real_layers_selected = False
    phase = "post_ffn_residual"

    def __init__(
        self,
        *,
        vector: Sequence[float],
        layer_mask: Sequence[str],
        scale: float,
        gate: float,
    ) -> None:
        values = tuple(float(value) for value in vector)
        layers = tuple(layer_mask)
        if len(values) != 2560 or not all(math.isfinite(value) for value in values):
            raise ValueError("D5B synthetic callback vector must be finite [2560]")
        if layers != ("fake-layer-01", "fake-layer-03"):
            raise PermissionError("D5B layer mask is synthetic and frozen")
        if not 0.0 < float(scale) <= 2.0 or not 0.0 <= float(gate) <= 1.0:
            raise ValueError("D5B active scale or gate is invalid")
        self.vector = values
        self.layer_mask = layers
        self.scale = float(scale)
        self.gate = float(gate)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        phase: str,
        layer_index: int,
        execution_path: str,
        residual_x: Any,
    ) -> StaticResidualTensor:
        layer_name = f"fake-layer-{layer_index:02d}"
        if (
            phase != self.phase
            or execution_path not in {"forward_one", "forward_seq"}
            or not isinstance(residual_x, StaticResidualTensor)
        ):
            raise PermissionError("D5B synthetic callback request is invalid")
        applied = layer_name in self.layer_mask
        addition = tuple(self.scale * self.gate * value for value in self.vector)
        output = residual_x.add_broadcast(addition) if applied else residual_x
        self.calls.append(
            {
                "phase": phase,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "execution_path": execution_path,
                "input_shape": list(residual_x.shape),
                "output_shape": list(output.shape),
                "dtype": output.dtype,
                "device": output.device,
                "applied": applied,
                "addition_digest_sha256": (
                    sha256_json(list(addition)) if applied else None
                ),
            }
        )
        return output


@dataclass(frozen=True)
class StaticActiveRequest:
    enabled: bool
    scale: float
    callback: StaticSyntheticCallback | None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or not isinstance(self.scale, (int, float)):
            raise ValueError("D5B request flags are invalid")
        if isinstance(self.scale, bool) or not 0.0 <= float(self.scale) <= 2.0:
            raise ValueError("D5B request scale must be in [0, 2]")
        applied = self.enabled and float(self.scale) > 0.0
        if applied:
            if type(self.callback) is not StaticSyntheticCallback:
                raise PermissionError("D5B active request requires the exact synthetic callback")
            if self.callback.scale != float(self.scale):
                raise ValueError("D5B request and callback scales differ")
        elif self.callback is not None and type(self.callback) is not StaticSyntheticCallback:
            raise PermissionError("D5B inactive request rejects unknown callbacks")


class RWKV7ProjectStaticActiveRuntime:
    """Active-capable AST wrapper that can execute only an offline static fixture."""

    runtime_version = CONTRACT_VERSION
    development_only = True
    d5b_static_active_implemented = True
    real_model_execution_available = False
    real_layers_selected = False
    real_self_projection_constructed = False

    def __init__(
        self,
        *,
        base_fixture: Any,
        upstream_source_bytes: bytes,
        upstream_globals: Mapping[str, Any],
        upstream_package_version: str,
        upstream_de_version: str | None,
    ) -> None:
        if (
            getattr(base_fixture, "offline_static_fixture", None) is not True
            or getattr(base_fixture, "model_loaded", None) is not False
            or getattr(base_fixture, "model_executed", None) is not False
        ):
            raise PermissionError("D5B runtime accepts only an unloaded static fixture")
        if upstream_package_version != EXPECTED_RWKV_PACKAGE_VERSION:
            raise RuntimeError("D5B RWKV package version differs from the source lock")
        if hashlib.sha256(upstream_source_bytes).hexdigest() != EXPECTED_RWKV_MODEL_SOURCE_SHA256:
            raise RuntimeError("D5B RWKV source differs from the source lock")
        if upstream_de_version is not None:
            raise PermissionError("D5B requires RWKV_DE_VERSION to be unset")
        if not callable(getattr(base_fixture, "forward", None)):
            raise TypeError("D5B fixture must expose forward")
        if CALLBACK_ATTRIBUTE in getattr(base_fixture, "__dict__", {}):
            raise RuntimeError("D5B fixture already owns the callback attribute")
        methods, counts = compile_instrumented_methods(
            upstream_source=upstream_source_bytes.decode("utf-8"),
            upstream_globals=upstream_globals,
            rwkv_de_version=upstream_de_version,
        )
        self._base_fixture = base_fixture
        self._methods = methods
        self._counts = counts
        self._execution_count = 0

    @property
    def injection_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def execution_count(self) -> int:
        return self._execution_count

    def forward(
        self,
        tokens: Any,
        state: Any,
        full_output: bool = False,
        *,
        coupling: StaticActiveRequest | None = None,
    ) -> Any:
        request = (
            StaticActiveRequest(enabled=False, scale=0.0, callback=None)
            if coupling is None
            else coupling
        )
        if type(request) is not StaticActiveRequest:
            raise PermissionError("D5B runtime rejects non-exact requests")
        callback = request.callback if request.enabled and request.scale > 0.0 else None
        instance_dict = getattr(self._base_fixture, "__dict__", None)
        if not isinstance(instance_dict, dict):
            raise TypeError("D5B fixture must expose a mutable instance dictionary")
        managed_names = (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
        if any(name in instance_dict for name in managed_names):
            raise RuntimeError("D5B fixture has conflicting instance overrides")
        try:
            setattr(self._base_fixture, CALLBACK_ATTRIBUTE, callback)
            for name, function in self._methods.items():
                setattr(
                    self._base_fixture,
                    name,
                    types.MethodType(function, self._base_fixture),
                )
            self._execution_count += 1
            return self._base_fixture.forward(tokens, state, full_output)
        finally:
            for name in managed_names:
                instance_dict.pop(name, None)


SYNTHETIC_SOURCE = """
class RWKV_x070:
    offline_static_fixture = True
    model_loaded = False
    model_executed = False

    def forward(self, idx, state, full_output=False):
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)

    def forward_one(self, idx, state):
        x = StaticResidualTensor.from_tokens([idx], squeeze=True)
        for i in range(4):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_one(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state

    def forward_seq(self, idx, state, full_output=False):
        x = StaticResidualTensor.from_tokens(idx, squeeze=False)
        for i in range(4):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state
"""


def _synthetic_namespace() -> tuple[dict[str, Any], type]:
    def cmix(x: StaticResidualTensor, state_value: int, layer_index: int):
        return StaticResidualTensor.constant_like(x, (layer_index + 1) / 1000.0), state_value + 1

    namespace: dict[str, Any] = {
        "StaticResidualTensor": StaticResidualTensor,
        "RWKV_x070_CMix_one": cmix,
        "RWKV_x070_CMix_seq": cmix,
    }
    exec(SYNTHETIC_SOURCE, namespace)
    return namespace, namespace["RWKV_x070"]


def _vector() -> tuple[float, ...]:
    return tuple(((index % 31) - 15) / 3100.0 for index in range(2560))


def _state() -> list[int]:
    return [0 for _ in range(12)]


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5B config must be an object")
    return value


def validate_contract(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisite = config.get("prerequisite")
    upstream = config.get("upstream_lock")
    fixture = config.get("static_fixture")
    callback = config.get("synthetic_callback")
    authority = config.get("authority")
    if not all(isinstance(value, Mapping) for value in (prerequisite, upstream, fixture, callback, authority)):
        raise ValueError("D5B contract structure is incomplete")
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION
        and config.get("stage") == "Coupling-D5B_project_active_path_static_integration"
        and config.get("status") == "project_local_static_fixture_only_model_unapproved"
        and config.get("development_only") is True,
        "prerequisites_frozen": prerequisite
        == {
            "d5a_status": "coupling_d5a_offline_active_contract_verified",
            "d5a_report_digest_sha256": D5A_REPORT_DIGEST,
            "d5_design_report_digest_sha256": D5_DESIGN_REPORT_DIGEST,
            "d4b_real_report_digest_sha256": D4B_REPORT_DIGEST,
        },
        "confirmation_exact": config.get("required_owner_confirmation_text") == REQUIRED_CONFIRMATION,
        "upstream_lock_exact": upstream
        == {
            "package": "rwkv",
            "version": "0.8.32",
            "model_source_sha256": EXPECTED_RWKV_MODEL_SOURCE_SHA256,
            "target_class": "RWKV_x070",
            "execution_paths": ["forward_one", "forward_seq"],
            "phase": "post_ffn_residual",
            "rwkv_de_version": "unset",
            "reuse_d4b_validated_ast_transform": True,
            "installed_source_probed_this_round": False,
        },
        "static_fixture_exact": fixture
        == {
            "offline_static_fixture_required": True,
            "n_layer": 4,
            "hidden_dimension": 2560,
            "dtype": "float16",
            "device": "fake-cuda:0",
            "state_components_per_layer": 3,
            "single_token": 3,
            "sequence_tokens": [3, 5, 8],
        },
        "synthetic_callback_exact": callback
        == {
            "kind": "static_shape_synthetic_probe_not_self_projection",
            "phase": "post_ffn_residual",
            "layer_mask": ["fake-layer-01", "fake-layer-03"],
            "layer_mask_is_real_selection": False,
            "gate": 0.5,
            "full_scale": 1.0,
            "sequence_policy": "broadcast_same_vector_to_each_position",
            "projection_trained": False,
            "real_self_projection": False,
        },
        "required_checks_exact": config.get("required_checks")
        == [
            "locked_ast_has_both_post_ffn_paths",
            "only_offline_static_fixture_accepted",
            "off_and_zero_callback_absent",
            "off_and_zero_exact",
            "active_both_paths_change",
            "active_callback_counts_exact",
            "real_hidden_shape_preserved",
            "sequence_broadcast_preserved",
            "source_state_immutable",
            "temporary_bindings_restored",
            "exception_restores_bindings",
            "nonfinite_and_wrong_shape_fail_closed",
        ],
        "only_d5b_authorized": authority.get("d5b_static_active_implementation_authorized") is True
        and all(
            authority.get(field) is False
            for field in authority
            if field != "d5b_static_active_implementation_authorized"
        ),
        "next_confirmation_exact": config.get(
            "required_next_owner_confirmation_text"
        )
        == REQUIRED_NEXT_CONFIRMATION,
        "next_gate_closed": config.get("next_gate")
        == "Coupling-D5C_requires_design_and_separate_real_execution_authorization",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("Coupling-D5B contract failed closed: " + ", ".join(failed))
    return checks


def build_d5b_report(*, config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5B config path is not frozen")
    config = _object(config_file)
    contract_checks = validate_contract(config)
    inspection = inspect_instrumented_source(SYNTHETIC_SOURCE)
    namespace, fixture_class = _synthetic_namespace()
    fixture = fixture_class()
    source_bytes = SYNTHETIC_SOURCE.encode("utf-8")
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    original_lock = EXPECTED_RWKV_MODEL_SOURCE_SHA256
    try:
        globals()["EXPECTED_RWKV_MODEL_SOURCE_SHA256"] = source_digest
        runtime = RWKV7ProjectStaticActiveRuntime(
            base_fixture=fixture,
            upstream_source_bytes=source_bytes,
            upstream_globals=namespace,
            upstream_package_version=EXPECTED_RWKV_PACKAGE_VERSION,
            upstream_de_version=None,
        )
    finally:
        globals()["EXPECTED_RWKV_MODEL_SOURCE_SHA256"] = original_lock
    source_state = _state()
    source_snapshot = copy.deepcopy(source_state)
    callback_config = config["synthetic_callback"]

    def callback() -> StaticSyntheticCallback:
        return StaticSyntheticCallback(
            vector=_vector(),
            layer_mask=callback_config["layer_mask"],
            scale=callback_config["full_scale"],
            gate=callback_config["gate"],
        )

    baseline_one = fixture.forward([3], _state(), False)
    baseline_seq = fixture.forward([3, 5, 8], _state(), True)
    off_one = runtime.forward([3], copy.deepcopy(source_state), False)
    off_seq = runtime.forward([3, 5, 8], copy.deepcopy(source_state), True)
    inactive = callback()
    zero = callback()
    off_request = StaticActiveRequest(enabled=False, scale=1.0, callback=inactive)
    zero_request = StaticActiveRequest(enabled=True, scale=0.0, callback=zero)
    off_explicit_one = runtime.forward(
        [3], copy.deepcopy(source_state), False, coupling=off_request
    )
    zero_seq = runtime.forward(
        [3, 5, 8], copy.deepcopy(source_state), True, coupling=zero_request
    )
    active_one_callback = callback()
    active_seq_callback = callback()
    active_one = runtime.forward(
        [3], copy.deepcopy(source_state), False,
        coupling=StaticActiveRequest(enabled=True, scale=1.0, callback=active_one_callback),
    )
    active_seq = runtime.forward(
        [3, 5, 8], copy.deepcopy(source_state), True,
        coupling=StaticActiveRequest(enabled=True, scale=1.0, callback=active_seq_callback),
    )
    instance_clean_after_success = not any(
        name in fixture.__dict__ for name in (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
    )
    try:
        runtime.forward(None, copy.deepcopy(source_state))
    except TypeError:
        pass
    instance_clean_after_exception = not any(
        name in fixture.__dict__ for name in (*TARGET_METHODS, CALLBACK_ATTRIBUTE)
    )
    invalid_fixture = fixture_class()
    invalid_fixture.model_loaded = True
    invalid_fixture_rejected = False
    try:
        RWKV7ProjectStaticActiveRuntime(
            base_fixture=invalid_fixture,
            upstream_source_bytes=source_bytes,
            upstream_globals=namespace,
            upstream_package_version=EXPECTED_RWKV_PACKAGE_VERSION,
            upstream_de_version=None,
        )
    except PermissionError:
        invalid_fixture_rejected = True
    invalid_shape_rejected = False
    try:
        StaticResidualTensor(values=tuple(0.0 for _ in range(2559)))
    except ValueError:
        invalid_shape_rejected = True
    nonfinite_rejected = False
    invalid_vector = list(_vector())
    invalid_vector[0] = float("nan")
    try:
        StaticSyntheticCallback(
            vector=invalid_vector,
            layer_mask=callback_config["layer_mask"],
            scale=1.0,
            gate=callback_config["gate"],
        )
    except ValueError:
        nonfinite_rejected = True
    calls = active_one_callback.calls + active_seq_callback.calls
    runtime_checks = {
        "locked_ast_has_both_post_ffn_paths": inspection["valid"]
        and inspection["injection_counts"] == {"forward_one": 1, "forward_seq": 1},
        "only_offline_static_fixture_accepted": runtime.real_model_execution_available is False
        and invalid_fixture_rejected,
        "off_and_zero_callback_absent": not inactive.calls and not zero.calls,
        "off_and_zero_exact": off_one == baseline_one
        and off_seq == baseline_seq
        and off_explicit_one == baseline_one
        and zero_seq == baseline_seq,
        "active_both_paths_change": active_one != baseline_one and active_seq != baseline_seq,
        "active_callback_counts_exact": len(active_one_callback.calls) == 4
        and len(active_seq_callback.calls) == 4
        and sum(call["applied"] for call in active_one_callback.calls) == 2
        and sum(call["applied"] for call in active_seq_callback.calls) == 2,
        "real_hidden_shape_preserved": all(
            call["input_shape"] == call["output_shape"]
            and call["input_shape"][-1] == 2560
            and call["dtype"] == "float16"
            and call["device"] == "fake-cuda:0"
            for call in calls
        ),
        "sequence_broadcast_preserved": all(
            call["input_shape"] == [3, 2560]
            for call in active_seq_callback.calls
        ),
        "source_state_immutable": source_state == source_snapshot,
        "temporary_bindings_restored": instance_clean_after_success,
        "exception_restores_bindings": instance_clean_after_exception,
        "nonfinite_and_wrong_shape_fail_closed": invalid_shape_rejected
        and nonfinite_rejected,
    }
    if not all(runtime_checks.values()):
        failed = [name for name, valid in runtime_checks.items() if not valid]
        raise RuntimeError("Coupling-D5B runtime failed: " + ", ".join(failed))
    report = {
        "report_version": CONTRACT_VERSION,
        "status": "coupling_d5b_project_static_active_verified",
        "valid": True,
        "contract_checks": contract_checks,
        "runtime_checks": runtime_checks,
        "ast_inspection": inspection,
        "fixture": {
            "n_layer": 4,
            "hidden_dimension": 2560,
            "single_shape": [2560],
            "sequence_shape": [3, 2560],
            "active_calls": calls,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "next_gate": "Coupling-D5C_requires_design_and_separate_real_execution_authorization",
        "safety": {
            "d5b_project_static_active_implemented": True,
            "offline_static_fixture_executed": True,
            "installed_rwkv_source_probed": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_model_active_injection_executed": False,
            "real_layers_selected": False,
            "real_self_projection_constructed": False,
            "formal_test_set_accessed": False,
            "self_effect_experiment_run": False,
            "self_updater_implemented": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
