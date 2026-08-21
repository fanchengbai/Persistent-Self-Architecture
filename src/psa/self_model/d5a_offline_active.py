from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.self_model.encoding import (
    DeterministicHashFakeSelfEncoder,
    EncodedSelf,
    encoded_self_digest,
    randomize_encoded_fields,
)
from psa.self_model.fake_callback_runtime import (
    FakePostFFNResidualCallback,
    FakeRWKV7ResidualRuntime,
    FakeResidualTensor,
)
from psa.self_model.state import build_self_state, swap_self_fields


CONTRACT_VERSION = "0.1-coupling-d5a-offline-active"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d5a_offline_active.json"
)
DESIGN_REPORT_DIGEST = "d41dc30460060a85031df38b6cfb27ad8fa41b57cd77ba6f7e0940e75eb898fd"
D4B_REPORT_DIGEST = "8befb5f4b2ce90241b66aff1f43bce59645d367c14f6594169e9c454fcf36a20"
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D5A 离线active contract与fake projection实现；"
    "不授权Coupling-D5B/D5C/D5D/D5E、RWKV/Torch导入、权重访问、模型加载或执行、"
    "真实层选择、真实Self projection构造、Self效果实验、Self Updater或自动重跑。"
)
REQUIRED_NEXT_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D5B 项目内active路径静态集成与无模型验证；"
    "不授权Coupling-D5C/D5D/D5E、RWKV/Torch导入、权重访问、模型加载或执行、"
    "真实层选择、真实Self projection构造、Self效果实验、Self Updater或自动重跑。"
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "configs/development/self_model_v0_1_coupling_d5_active_design.json",
    "docs/self_model_v0_1_coupling_d5a_implementation_authorization.md",
    "docs/self_model_v0_1_coupling_d5a_offline_active.md",
    "scripts/verify_self_model_v0_1_coupling_d5a_offline_active.py",
    "src/psa/self_model/d5a_offline_active.py",
    "src/psa/self_model/encoding.py",
    "src/psa/self_model/fake_callback_runtime.py",
    "src/psa/self_model/state.py",
    "tests/test_self_model_d5a_offline_active.py",
)


@dataclass(frozen=True)
class FakeProjectedSelf:
    projection_version: str
    encoded_self_digest_sha256: str
    matrix_digest_sha256: str
    output_digest_sha256: str
    input_dimension: int
    output_dimension: int
    vector: tuple[float, ...]
    offline_fake_projection: bool
    trained: bool
    real_self_projection: bool
    model_loaded: bool


class DeterministicHashMatrixFakeProjection:
    """No-weight fixture. It is deliberately not a real Self projection."""

    offline_fake_projection = True
    trained = False
    real_self_projection = False
    model_loaded = False

    def __init__(
        self, *, input_dimension: int, output_dimension: int, seed_namespace: str
    ) -> None:
        if input_dimension < 1 or output_dimension < 1 or not seed_namespace:
            raise ValueError("fake projection configuration is invalid")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.seed_namespace = seed_namespace
        self._matrix = tuple(
            tuple(self._coefficient(row, column) for column in range(input_dimension))
            for row in range(output_dimension)
        )

    def _coefficient(self, row: int, column: int) -> float:
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "namespace": self.seed_namespace,
                    "row": row,
                    "column": column,
                }
            )
        ).digest()
        integer = int.from_bytes(digest[:8], "big")
        return ((integer / 2**64) * 2.0 - 1.0) / math.sqrt(self.input_dimension)

    @property
    def matrix_digest_sha256(self) -> str:
        return sha256_json([list(row) for row in self._matrix])

    def project(self, encoded: EncodedSelf) -> FakeProjectedSelf:
        if (
            encoded.offline_fake_encoder is not True
            or encoded.model_loaded is not False
            or encoded.prompt_serialization_used is not False
            or encoded.dimension != self.input_dimension
        ):
            raise PermissionError("D5A projection accepts only unloaded fake encodings")
        vector = tuple(
            sum(weight * value for weight, value in zip(row, encoded.aggregate_vector))
            for row in self._matrix
        )
        if len(vector) != self.output_dimension or not all(
            math.isfinite(value) for value in vector
        ):
            raise RuntimeError("D5A fake projection output is invalid")
        encoded_digest = encoded_self_digest(encoded)
        return FakeProjectedSelf(
            projection_version=CONTRACT_VERSION,
            encoded_self_digest_sha256=encoded_digest,
            matrix_digest_sha256=self.matrix_digest_sha256,
            output_digest_sha256=sha256_json(list(vector)),
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            vector=vector,
            offline_fake_projection=True,
            trained=False,
            real_self_projection=False,
            model_loaded=False,
        )


def _item(item_id: str, value: str, update_class: str) -> dict[str, Any]:
    return {
        "field_item_id": item_id,
        "value": value,
        "value_type": "string",
        "confidence": 1.0,
        "update_class": update_class,
        "created_step": 0,
        "updated_step": 0,
        "source_evidence_ids": ["fixture:coupling-d5a-owner-set"],
        "status": "active",
    }


def _state(state_id: str, identity: str, goal: str) -> dict[str, Any]:
    return build_self_state(
        state_id=state_id,
        agent_instance_id="agent-coupling-d5a-fake",
        trajectory_id="trajectory-coupling-d5a-fake",
        step=0,
        model_id="offline-no-model-loaded",
        tokenizer_id="offline-no-tokenizer-loaded",
        fields={
            "identity_anchors": [_item("identity", identity, "protected")],
            "active_goals": [_item("goal", goal, "fast")],
        },
        provenance_refs=["fixture:coupling-d5a"],
    )


def _delta_l2(left: FakeResidualTensor, right: FakeResidualTensor) -> float:
    def flattened(value: FakeResidualTensor) -> tuple[float, ...]:
        if isinstance(value.values[0], tuple):
            return tuple(item for row in value.values for item in row)
        return tuple(value.values)

    left_values = flattened(left)
    right_values = flattened(right)
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left_values, right_values))
    )


def validate_contract(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisite = config.get("prerequisite")
    encoder = config.get("fake_encoder")
    projection = config.get("fake_projection")
    runtime = config.get("fake_runtime")
    callback = config.get("fake_callback")
    fixture = config.get("offline_fixture")
    authority = config.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (prerequisite, encoder, projection, runtime, callback, fixture, authority)
    ):
        raise ValueError("D5A contract structure is incomplete")
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION
        and config.get("stage")
        == "Coupling-D5A_offline_active_contract_and_fake_projection"
        and config.get("status") == "offline_fake_implementation_only"
        and config.get("development_only") is True,
        "prerequisites_frozen": prerequisite.get("coupling_d5_design_status")
        == "coupling_d5_active_design_static_verified"
        and prerequisite.get("coupling_d5_design_report_digest_sha256")
        == DESIGN_REPORT_DIGEST
        and prerequisite.get("d4b_real_report_digest_sha256") == D4B_REPORT_DIGEST,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "fake_encoder_exact": encoder
        == {
            "kind": "deterministic_hash_fake_encoder",
            "dimension": 16,
            "field_mask": ["identity_anchors", "active_goals"],
            "prompt_serialization_used": False,
        },
        "fake_projection_exact": projection
        == {
            "kind": "deterministic_hash_matrix_fake_projection",
            "seed_namespace": "PSA|Self-v0.1|Coupling-D5A|fake-projection",
            "input_dimension": 16,
            "output_dimension": 8,
            "trained": False,
            "real_self_projection": False,
        },
        "fake_runtime_exact": runtime
        == {
            "kind": "no_weight_fake_rwkv7_residual_runtime",
            "n_layer": 4,
            "hidden_dimension": 8,
            "dtype": "float16",
            "device": "fake-cuda:0",
            "state_components_per_layer": 3,
        },
        "callback_contract_exact": callback.get("phase") == "post_ffn_residual"
        and callback.get("execution_paths") == ["forward_one", "forward_seq"]
        and callback.get("layer_mask") == ["fake-layer-01", "fake-layer-03"]
        and callback.get("gate") == 0.5
        and callback.get("sequence_policy") == "broadcast_all_tokens_fake_only"
        and callback.get("minimum_scale") == 0.0
        and callback.get("maximum_scale") == 2.0,
        "fixture_exact": fixture
        == {
            "single_token": 3,
            "sequence_tokens": [3, 5, 8],
            "half_scale": 0.5,
            "full_scale": 1.0,
            "random_seed": 20260821,
            "field_swap": "identity_anchors",
        },
        "required_checks_exact": config.get("required_checks")
        == [
            "state_and_encoding_immutable",
            "projection_deterministic_and_finite",
            "projection_changes_for_field_swap_and_random",
            "off_and_zero_do_not_call_callback",
            "off_and_zero_exact_for_both_paths",
            "active_changes_both_paths",
            "active_repeat_deterministic",
            "callback_counts_exact",
            "shape_dtype_device_preserved",
            "scale_ordering_observed",
        ],
        "only_d5a_authorized": authority.get("d5a_offline_implementation_authorized")
        is True
        and all(
            authority.get(field) is False
            for field in authority
            if field != "d5a_offline_implementation_authorized"
        ),
        "next_confirmation_exact": config.get(
            "required_next_owner_confirmation_text"
        )
        == REQUIRED_NEXT_CONFIRMATION,
        "next_gate_closed": config.get("next_gate")
        == "Coupling-D5B_requires_new_explicit_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("Coupling-D5A contract failed closed: " + ", ".join(failed))
    return checks


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5A config must be an object")
    return value


def build_d5a_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5A config path is not frozen")
    config = _object(config_file)
    contract_checks = validate_contract(config)
    encoder_config = config["fake_encoder"]
    projection_config = config["fake_projection"]
    runtime_config = config["fake_runtime"]
    callback_config = config["fake_callback"]
    fixture = config["offline_fixture"]

    left = _state("d5a-self-left", "saffron", "spiral")
    right = _state("d5a-self-right", "indigo", "harbor")
    left_snapshot = copy.deepcopy(left)
    right_snapshot = copy.deepcopy(right)
    swapped_left, _ = swap_self_fields(
        left,
        right,
        fields=[fixture["field_swap"]],
        left_state_id="d5a-self-left-swap-identity",
        right_state_id="d5a-self-right-swap-identity",
    )
    encoder = DeterministicHashFakeSelfEncoder(encoder_config["dimension"])
    encoded = encoder.encode(left, field_mask=encoder_config["field_mask"])
    encoded_snapshot = copy.deepcopy(encoded)
    swapped_encoded = encoder.encode(
        swapped_left, field_mask=encoder_config["field_mask"]
    )
    random_encoded = randomize_encoded_fields(
        encoded,
        fields=[fixture["field_swap"]],
        seed=fixture["random_seed"],
    )
    projector = DeterministicHashMatrixFakeProjection(
        input_dimension=projection_config["input_dimension"],
        output_dimension=projection_config["output_dimension"],
        seed_namespace=projection_config["seed_namespace"],
    )
    projected = projector.project(encoded)
    repeated_projection = projector.project(encoded)
    swapped_projection = projector.project(swapped_encoded)
    random_projection = projector.project(random_encoded)

    runtime = FakeRWKV7ResidualRuntime(
        n_layer=runtime_config["n_layer"],
        hidden_dimension=runtime_config["hidden_dimension"],
        dtype=runtime_config["dtype"],
        device=runtime_config["device"],
    )
    source_state = runtime.zero_state()
    source_state_snapshot = copy.deepcopy(source_state)

    def callback(*, enabled: bool, scale: float) -> FakePostFFNResidualCallback:
        return FakePostFFNResidualCallback(
            hidden_dimension=runtime.hidden_dimension,
            layer_mask=callback_config["layer_mask"],
            enabled=enabled,
            scale=scale,
            gate=callback_config["gate"],
            sequence_policy=callback_config["sequence_policy"],
        )

    vector = projected.vector
    token = fixture["single_token"]
    tokens = fixture["sequence_tokens"]
    baseline_one = runtime.forward_one(token, source_state, self_vector=vector)
    baseline_seq = runtime.forward_seq(tokens, source_state, self_vector=vector)
    off_one_callback = callback(enabled=False, scale=1.0)
    off_seq_callback = callback(enabled=False, scale=1.0)
    zero_one_callback = callback(enabled=True, scale=0.0)
    zero_seq_callback = callback(enabled=True, scale=0.0)
    off_one = runtime.forward_one(
        token, source_state, self_vector=vector, callback=off_one_callback
    )
    off_seq = runtime.forward_seq(
        tokens, source_state, self_vector=vector, callback=off_seq_callback
    )
    zero_one = runtime.forward_one(
        token, source_state, self_vector=vector, callback=zero_one_callback
    )
    zero_seq = runtime.forward_seq(
        tokens, source_state, self_vector=vector, callback=zero_seq_callback
    )
    full_one_callback = callback(enabled=True, scale=fixture["full_scale"])
    full_seq_callback = callback(enabled=True, scale=fixture["full_scale"])
    full_one = runtime.forward_one(
        token, source_state, self_vector=vector, callback=full_one_callback
    )
    full_seq = runtime.forward_seq(
        tokens, source_state, self_vector=vector, callback=full_seq_callback
    )
    repeat_one_callback = callback(enabled=True, scale=fixture["full_scale"])
    repeat_seq_callback = callback(enabled=True, scale=fixture["full_scale"])
    repeat_one = runtime.forward_one(
        token, source_state, self_vector=vector, callback=repeat_one_callback
    )
    repeat_seq = runtime.forward_seq(
        tokens, source_state, self_vector=vector, callback=repeat_seq_callback
    )
    half_one_callback = callback(enabled=True, scale=fixture["half_scale"])
    half_seq_callback = callback(enabled=True, scale=fixture["half_scale"])
    half_one = runtime.forward_one(
        token, source_state, self_vector=vector, callback=half_one_callback
    )
    half_seq = runtime.forward_seq(
        tokens, source_state, self_vector=vector, callback=half_seq_callback
    )

    active_calls = full_one_callback.calls + full_seq_callback.calls
    expected_per_path = len(callback_config["layer_mask"])
    half_one_delta = _delta_l2(half_one[0], baseline_one[0])
    full_one_delta = _delta_l2(full_one[0], baseline_one[0])
    half_seq_delta = _delta_l2(half_seq[0], baseline_seq[0])
    full_seq_delta = _delta_l2(full_seq[0], baseline_seq[0])
    runtime_checks = {
        "state_and_encoding_immutable": left == left_snapshot
        and right == right_snapshot
        and encoded == encoded_snapshot
        and source_state == source_state_snapshot,
        "projection_deterministic_and_finite": projected == repeated_projection
        and all(math.isfinite(value) for value in projected.vector)
        and projected.real_self_projection is False
        and projected.trained is False,
        "projection_changes_for_field_swap_and_random": projected.vector
        != swapped_projection.vector
        and projected.vector != random_projection.vector,
        "off_and_zero_do_not_call_callback": not (
            off_one_callback.calls
            or off_seq_callback.calls
            or zero_one_callback.calls
            or zero_seq_callback.calls
        ),
        "off_and_zero_exact_for_both_paths": off_one == baseline_one
        and off_seq == baseline_seq
        and zero_one == baseline_one
        and zero_seq == baseline_seq,
        "active_changes_both_paths": full_one != baseline_one
        and full_seq != baseline_seq,
        "active_repeat_deterministic": full_one == repeat_one
        and full_seq == repeat_seq
        and full_one_callback.calls == repeat_one_callback.calls
        and full_seq_callback.calls == repeat_seq_callback.calls,
        "callback_counts_exact": len(full_one_callback.calls) == expected_per_path
        and len(full_seq_callback.calls) == expected_per_path,
        "shape_dtype_device_preserved": all(
            call["input_shape"] == call["output_shape"]
            and call["dtype"] == runtime.dtype
            and call["device"] == runtime.device
            for call in active_calls
        ),
        "scale_ordering_observed": 0.0 < half_one_delta < full_one_delta
        and 0.0 < half_seq_delta < full_seq_delta,
    }
    if not all(runtime_checks.values()):
        failed = [name for name, valid in runtime_checks.items() if not valid]
        raise RuntimeError("Coupling-D5A runtime failed: " + ", ".join(failed))

    report = {
        "report_version": CONTRACT_VERSION,
        "status": "coupling_d5a_offline_active_contract_verified",
        "valid": True,
        "contract_checks": contract_checks,
        "runtime_checks": runtime_checks,
        "projection": {
            "projection_version": projected.projection_version,
            "encoded_self_digest_sha256": projected.encoded_self_digest_sha256,
            "matrix_digest_sha256": projected.matrix_digest_sha256,
            "output_digest_sha256": projected.output_digest_sha256,
            "input_dimension": projected.input_dimension,
            "output_dimension": projected.output_dimension,
            "trained": projected.trained,
            "real_self_projection": projected.real_self_projection,
        },
        "callback": {
            "phase": callback_config["phase"],
            "execution_paths": callback_config["execution_paths"],
            "layer_mask": callback_config["layer_mask"],
            "sequence_policy": callback_config["sequence_policy"],
            "full_scale_active_calls": active_calls,
        },
        "scale_observations": {
            "half_one_delta_l2": half_one_delta,
            "full_one_delta_l2": full_one_delta,
            "half_seq_delta_l2": half_seq_delta,
            "full_seq_delta_l2": full_seq_delta,
            "behavior_claim_created": False,
        },
        "source_digests": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "next_gate": "Coupling-D5B_requires_new_explicit_confirmation",
        "safety": {
            "d5a_offline_active_contract_implemented": True,
            "fake_projection_constructed": True,
            "d5b_real_path_implemented": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
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
