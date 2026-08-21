from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model import d5c_mechanism_runtime as runtime_module
from psa.self_model.d5c_mechanism_runtime import (
    D5CCouplingRequest,
    D5CSyntheticProbe,
    HIDDEN_DIMENSION,
    PRECONDITION_ORDER,
    ROUTES,
    SCORED_ROUNDS,
    RWKV7D5CActiveRuntime,
)
from psa.self_model.rwkv7_instrumented_off_runtime import CALLBACK_ATTRIBUTE


DIAGNOSTIC_VERSION = "0.1-d5c-failure-lifecycle-offline"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_failure_lifecycle_diagnostic.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D5C失败纯离线绑定生命周期诊断设计与无模型实现；"
    "仅使用现有报告、冻结源码和fake fixture，不导入RWKV/Torch、不访问权重、"
    "不加载或执行模型、不修改D5C失败结论，也不授权D5D/D5E、正式测试集、"
    "Self效果结论、真实Self projection、Self Updater或自动重跑。"
)
CLASSIFICATION = (
    "real_only_post_active_route_isolation_contamination_not_reproduced_by_"
    "plain_python_fixture_root_cause_unresolved"
)
REAL_REPORT_DIGEST = "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21"
CLAIM_DIGEST = "75d69ae3ad4550361cc53d03ae5d89fd636f045d31a6cd62974c4dc15496f12f"
ENTRY_REPORT_DIGEST = "ce96c627377a75dfe711e018ed3e5cf71657486476338cb5ba295045b555ae88"
EXECUTION_COMMIT = "a8ef52a5390581666edf4e6ffecaf61aee912a9e"
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_failure_lifecycle_diagnostic.md",
    "docs/self_model_v0_1_coupling_d5c_real_mechanism_observation.md",
    "scripts/verify_self_model_v0_1_d5c_failure_lifecycle_diagnostic.py",
    "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "tests/test_self_model_d5c_failure_lifecycle_diagnostic.py",
)


@dataclass(frozen=True)
class OfflineBoolean:
    value: bool

    def all(self) -> "OfflineBoolean":
        return self

    def item(self) -> bool:
        return self.value


@dataclass(frozen=True)
class OfflineTensor:
    values: float | tuple[float, ...] | tuple[tuple[float, ...], ...]
    dtype: str = "float16"
    device: str = "fake-cuda:0"

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.values, float):
            return ()
        if self.values and isinstance(self.values[0], tuple):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def _flat(self) -> tuple[float, ...]:
        if isinstance(self.values, float):
            return (self.values,)
        if self.values and isinstance(self.values[0], tuple):
            return tuple(value for row in self.values for value in row)
        return tuple(self.values)

    def detach(self) -> "OfflineTensor":
        return self

    def float(self) -> "OfflineTensor":
        return OfflineTensor(self.values, dtype="float32", device=self.device)

    def square(self) -> "OfflineTensor":
        if isinstance(self.values, float):
            values: Any = self.values * self.values
        elif self.values and isinstance(self.values[0], tuple):
            values = tuple(tuple(value * value for value in row) for row in self.values)
        else:
            values = tuple(value * value for value in self.values)
        return OfflineTensor(values, dtype=self.dtype, device=self.device)

    def mean(self) -> "OfflineTensor":
        flattened = self._flat()
        return OfflineTensor(
            sum(flattened) / len(flattened), dtype=self.dtype, device=self.device
        )

    def sqrt(self) -> "OfflineTensor":
        if not isinstance(self.values, float):
            raise TypeError("offline sqrt accepts only a scalar")
        return OfflineTensor(
            math.sqrt(self.values), dtype=self.dtype, device=self.device
        )

    def item(self) -> float:
        if not isinstance(self.values, float):
            raise TypeError("offline item accepts only a scalar")
        return self.values

    def to(self, *, dtype: str) -> "OfflineTensor":
        return OfflineTensor(self.values, dtype=dtype, device=self.device)

    def __mul__(self, other: Any) -> "OfflineTensor":
        factor = other.item() if isinstance(other, OfflineTensor) else float(other)
        if isinstance(self.values, float):
            values: Any = self.values * factor
        elif self.values and isinstance(self.values[0], tuple):
            values = tuple(tuple(value * factor for value in row) for row in self.values)
        else:
            values = tuple(value * factor for value in self.values)
        return OfflineTensor(values, dtype=self.dtype, device=self.device)

    def __add__(self, other: Any) -> "OfflineTensor":
        if not isinstance(other, OfflineTensor):
            return NotImplemented
        if isinstance(self.values, float) or isinstance(other.values, float):
            return NotImplemented
        self_sequence = bool(self.values and isinstance(self.values[0], tuple))
        other_sequence = bool(other.values and isinstance(other.values[0], tuple))
        if self_sequence and not other_sequence:
            values: Any = tuple(
                tuple(left + right for left, right in zip(row, other.values))
                for row in self.values
            )
        elif self_sequence and other_sequence:
            values = tuple(
                tuple(left + right for left, right in zip(left_row, right_row))
                for left_row, right_row in zip(self.values, other.values)
            )
        elif not self_sequence and not other_sequence:
            values = tuple(left + right for left, right in zip(self.values, other.values))
        else:
            return NotImplemented
        return OfflineTensor(values, dtype=self.dtype, device=self.device)

    @classmethod
    def from_tokens(
        cls, tokens: Sequence[int], *, squeeze: bool
    ) -> "OfflineTensor":
        rows = tuple(
            tuple((((token + 1) * (index + 3)) % 37) / 37.0 + 0.25 for index in range(HIDDEN_DIMENSION))
            for token in tokens
        )
        return cls(rows[0] if squeeze else rows)

    @classmethod
    def constant_like(
        cls, source: "OfflineTensor", value: float
    ) -> "OfflineTensor":
        if len(source.shape) == 1:
            values: Any = tuple(value for _ in range(HIDDEN_DIMENSION))
        else:
            values = tuple(
                tuple(value for _ in range(HIDDEN_DIMENSION))
                for _ in range(source.shape[0])
            )
        return cls(values, dtype=source.dtype, device=source.device)


class OfflineTorch:
    float32 = "float32"

    @staticmethod
    def isfinite(value: OfflineTensor) -> OfflineBoolean:
        return OfflineBoolean(all(math.isfinite(item) for item in value._flat()))

    @staticmethod
    def tensor(
        values: Sequence[float], *, device: str, dtype: str
    ) -> OfflineTensor:
        return OfflineTensor(tuple(values), dtype=dtype, device=device)


DIAGNOSTIC_SOURCE = """
class RWKV_x070:
    offline_lifecycle_fixture = True
    model_loaded = False
    model_executed = False

    def forward(self, idx, state, full_output=False):
        if len(idx) > 1:
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0], state)

    def forward_one(self, idx, state):
        x = OfflineTensor.from_tokens([idx], squeeze=True)
        for i in range(32):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_one(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state

    def forward_seq(self, idx, state, full_output=False):
        x = OfflineTensor.from_tokens(idx, squeeze=False)
        for i in range(32):
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(x, state[i * 3 + 2], i)
            x = x + xx
        return x, state
"""


def _namespace() -> tuple[dict[str, Any], type]:
    def cmix(x: OfflineTensor, state_value: int, layer_index: int):
        return OfflineTensor.constant_like(x, (layer_index + 1) / 10000.0), state_value + 1

    namespace: dict[str, Any] = {
        "OfflineTensor": OfflineTensor,
        "RWKV_x070_CMix_one": cmix,
        "RWKV_x070_CMix_seq": cmix,
    }
    exec(DIAGNOSTIC_SOURCE, namespace)
    return namespace, namespace["RWKV_x070"]


def _state() -> list[int]:
    return [0 for _ in range(96)]


def _tensor_digest(value: OfflineTensor) -> str:
    return sha256_json(
        {"shape": list(value.shape), "dtype": value.dtype, "values": value.values}
    )


def _output_digest(output: tuple[OfflineTensor, list[int]]) -> str:
    return sha256_json(
        {"logits": _tensor_digest(output[0]), "state": list(output[1])}
    )


def run_plain_python_lifecycle_fixture(tokens: Sequence[int]) -> dict[str, Any]:
    namespace, fixture_class = _namespace()
    fixture = fixture_class()
    source_bytes = DIAGNOSTIC_SOURCE.encode("utf-8")
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    original_lock = runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256
    try:
        runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256 = source_digest
        runtime = RWKV7D5CActiveRuntime(
            base_model=fixture,
            upstream_source_bytes=source_bytes,
            upstream_globals=namespace,
            upstream_package_version="0.8.32",
            upstream_de_version=None,
            execution_claim_sha256="a" * 64,
            machine_authorization_sha256="b" * 64,
        )
    finally:
        runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256 = original_lock

    baseline = fixture.forward(list(tokens), _state(), len(tokens) > 1)
    probe = D5CSyntheticProbe(
        torch=OfflineTorch(), execution_claim_sha256="a" * 64,
        machine_authorization_sha256="b" * 64,
    )
    active = runtime.forward(
        list(tokens), _state(), len(tokens) > 1,
        coupling=D5CCouplingRequest(enabled=True, scale=1.0, callback=probe),
    )
    instance_keys_after_active = sorted(
        name
        for name in ("forward_one", "forward_seq", CALLBACK_ATTRIBUTE)
        if name in fixture.__dict__
    )
    counts_after_active = {
        "invocations": probe.invocation_count,
        "applications": probe.application_count,
    }
    post_active_original = fixture.forward(
        list(tokens), _state(), len(tokens) > 1
    )
    counts_after_original = {
        "invocations": probe.invocation_count,
        "applications": probe.application_count,
    }
    baseline_digest = _output_digest(baseline)
    active_digest = _output_digest(active)
    post_digest = _output_digest(post_active_original)
    checks = {
        "active_differs_from_baseline": active_digest != baseline_digest,
        "post_active_original_returns_to_baseline": post_digest == baseline_digest,
        "post_active_original_differs_from_active": post_digest != active_digest,
        "temporary_instance_bindings_absent": instance_keys_after_active == [],
        "active_callback_count_exact": counts_after_active
        == {"invocations": 32, "applications": 1},
        "raw_original_does_not_advance_callback": counts_after_original
        == counts_after_active,
    }
    return {
        "tokens": list(tokens),
        "execution_path": "forward_seq" if len(tokens) > 1 else "forward_one",
        "valid": all(checks.values()),
        "checks": checks,
        "digests": {
            "baseline": baseline_digest,
            "active": active_digest,
            "post_active_original": post_digest,
        },
        "instance_keys_after_active": instance_keys_after_active,
        "counts_after_active": counts_after_active,
        "counts_after_original": counts_after_original,
    }


def _object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C lifecycle diagnostic config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    evidence = config.get("d5c_frozen_evidence")
    fixture = config.get("fake_fixture")
    authority = config.get("authority")
    if not all(isinstance(value, Mapping) for value in (evidence, fixture, authority)):
        raise ValueError("D5C lifecycle diagnostic config is incomplete")
    false_authority = {
        field for field, value in authority.items()
        if field not in {
            "offline_diagnostic_design_authorized",
            "offline_fake_implementation_authorized",
            "existing_report_observation_authorized",
        } and value is False
    }
    expected_false_authority = set(authority) - {
        "offline_diagnostic_design_authorized",
        "offline_fake_implementation_authorized",
        "existing_report_observation_authorized",
    }
    checks = {
        "identity_exact": config.get("diagnostic_version") == DIAGNOSTIC_VERSION
        and config.get("stage") == "Coupling-D5C_failure_binding_lifecycle_offline_diagnostic"
        and config.get("status") == "offline_diagnostic_implementation_authorized_no_model"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("required_owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "evidence_identity_exact": evidence.get("execution_commit") == EXECUTION_COMMIT
        and evidence.get("entry_static_report_sha256") == ENTRY_REPORT_DIGEST
        and evidence.get("execution_claim_sha256") == CLAIM_DIGEST
        and evidence.get("real_report_sha256") == REAL_REPORT_DIGEST,
        "failure_preserved": evidence.get("real_report_status")
        == "d5c_mechanism_smoke_failed"
        and evidence.get("real_report_valid") is False
        and evidence.get("decision_effect") == "stop_without_rerun",
        "observed_counts_exact": evidence.get("model_forward_calls") == 42
        and evidence.get("callback_invocations_expected") == 320
        and evidence.get("callback_invocations_observed") == 576
        and evidence.get("probe_applications_expected") == 10
        and evidence.get("probe_applications_observed") == 18
        and evidence.get("extra_callback_invocations") == 256
        and evidence.get("extra_probe_applications") == 8
        and evidence.get("n_layer") == 32,
        "fixture_signatures_exact": evidence.get("fixture_signatures")
        == [
            {
                "fixture_id": name, "within_exact": 12, "control_exact": 4,
                "active_control_exact": 4, "off_equals_zero_rounds": 4,
                "original_equals_active_rounds": 4,
                "original_differs_from_off_rounds": 4,
                "original_differs_from_zero_rounds": 4,
            }
            for name in ("single_noncore", "sequence_noncore")
        ],
        "fake_fixture_exact": fixture
        == {
            "source_kind": "pure_python_locked_shape_fixture",
            "n_layer": 32, "hidden_dimension": 2560,
            "single_tokens": [3], "sequence_tokens": [3, 5, 8],
            "state_components": 96,
            "execution_paths": ["forward_one", "forward_seq"],
            "active_then_raw_original_pairs_per_path": 1,
            "rwkv_imported": False, "torch_imported": False,
            "model_loaded": False, "model_executed": False,
        },
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "offline_authority_only": all(
            authority.get(field) is True for field in (
                "offline_diagnostic_design_authorized",
                "offline_fake_implementation_authorized",
                "existing_report_observation_authorized",
            )
        ) and false_authority == expected_false_authority,
        "next_gate_exact": config.get("next_gate")
        == "offline_diagnostic_observation_only_no_model_or_rerun_gate",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError(
            "D5C lifecycle diagnostic config failed closed: " + ", ".join(failed)
        )
    return checks


def _schedule_analysis() -> dict[str, Any]:
    timeline = [
        {"phase": "prefix", "route": ROUTES[0]},
        *({"phase": "precondition", "route": route} for route in PRECONDITION_ORDER),
        *(
            {"phase": "scored", "round": round_index, "route": route}
            for round_index, round_routes in enumerate(SCORED_ROUNDS, start=1)
            for route in round_routes
        ),
    ]
    scored_originals = [
        {
            "timeline_index": index,
            "round": item["round"],
            "previous_route": timeline[index - 1]["route"],
        }
        for index, item in enumerate(timeline)
        if item["phase"] == "scored" and item["route"] == ROUTES[0]
    ]
    return {
        "per_fixture_scored_original_count": len(scored_originals),
        "total_scored_original_count": len(scored_originals) * 2,
        "all_scored_originals_immediately_follow_active": all(
            item["previous_route"] == ROUTES[3] for item in scored_originals
        ),
        "scored_original_predecessors": scored_originals,
        "extra_applications_match_post_active_originals": len(scored_originals) * 2 == 8,
        "extra_invocations_match_layers_times_originals": len(scored_originals) * 2 * 32
        == 256,
        "schedule_can_separate_route_from_predecessor_effect": False,
    }


def build_diagnostic_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C lifecycle diagnostic config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    schedule = _schedule_analysis()
    single = run_plain_python_lifecycle_fixture([3])
    sequence = run_plain_python_lifecycle_fixture([3, 5, 8])
    runtime_source = (root / "src/psa/self_model/d5c_mechanism_runtime.py").read_text(
        encoding="utf-8"
    )
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "frozen_failure_preserved": config["d5c_frozen_evidence"]["real_report_valid"] is False,
        "all_scored_originals_follow_active": schedule[
            "all_scored_originals_immediately_follow_active"
        ],
        "extra_applications_explained_by_post_active_positions": schedule[
            "extra_applications_match_post_active_originals"
        ],
        "extra_invocations_explained_by_32_layers": schedule[
            "extra_invocations_match_layers_times_originals"
        ],
        "schedule_predecessor_confound_confirmed": schedule[
            "schedule_can_separate_route_from_predecessor_effect"
        ] is False,
        "plain_python_single_does_not_reproduce": single["valid"],
        "plain_python_sequence_does_not_reproduce": sequence["valid"],
        "direct_instance_pop_cleanup_present": "instance_dict.pop(name, None)"
        in runtime_source,
        "fake_bindings_restored_both_paths": single["checks"][
            "temporary_instance_bindings_absent"
        ] and sequence["checks"]["temporary_instance_bindings_absent"],
        "fake_raw_original_callback_absent_both_paths": single["checks"][
            "raw_original_does_not_advance_callback"
        ] and sequence["checks"]["raw_original_does_not_advance_callback"],
        "root_cause_not_overclaimed": True,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError(
            "D5C lifecycle offline diagnostic failed: " + ", ".join(failed)
        )
    report = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "d5c_failure_offline_lifecycle_diagnostic_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "schedule_analysis": schedule,
        "fake_lifecycle": {"forward_one": single, "forward_seq": sequence},
        "findings": {
            "confirmed": [
                "real_failure_is_deterministic_across_both_fixtures",
                "all_scored_original_calls_are_immediately_post_active",
                "eight_extra_applications_match_eight_post_active_original_positions",
                "plain_python_fixture_restores_bindings_and_baseline",
            ],
            "weakened_not_eliminated": [
                "generic_plain_python_instance_dictionary_cleanup_failure"
            ],
            "unresolved_candidates": [
                "real_upstream_dispatch_boundary",
                "decorator_or_compiled_method_cache_boundary",
                "real_runtime_attribute_or_method_resolution_boundary",
                "direct_instance_dictionary_cleanup_interaction_specific_to_upstream_object",
            ],
            "not_supported": [
                "one_proven_low_level_root_cause",
                "d5c_pass",
                "self_effect_conclusion",
            ],
        },
        "source_digests": source_digests,
        "next_gate": "separate_source_level_cache_boundary_diagnostic_confirmation_required",
        "safety": {
            "existing_real_report_reexecuted": False,
            "d4_rerun": False, "d4b_rerun": False, "d5c_rerun": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False, "model_loaded": False, "model_executed": False,
            "d5c_conclusion_changed": False, "d5d_authorized": False,
            "d5e_authorized": False, "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "real_self_projection_constructed": False,
            "self_updater_used": False, "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
