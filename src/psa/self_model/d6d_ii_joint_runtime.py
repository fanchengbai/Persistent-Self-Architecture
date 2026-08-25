from __future__ import annotations

import copy
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import math
import threading
import types
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import sha256_json
from psa.self_model.d6d_core_approach_design import CONDITIONS, SELF_CONDITIONS
from psa.self_model.d6d_ii_manifests import (
    CHOICE_TOKEN_IDS,
    FORCED_PREFIX_TOKEN_IDS,
    IDENTITY_KEYS,
    GOAL_KEYS,
    build_pilot_call_plan,
    expand_pilot_fixtures,
    expand_training_records,
    validate_pilot_manifest,
    validate_training_manifest,
)
from psa.self_model.d6d_projection_artifact import (
    FrozenSelfProjection,
    ProjectionTrainingRecord,
    build_frozen_projection_artifact,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
)
from psa.self_model.state import build_self_state


RUNTIME_VERSION = "0.1-coupling-d6d-ii-joint-runtime"
TARGET_LAYER_INDEX = 15
HIDDEN_DIMENSION = 2560
_NO_REQUEST = object()


@dataclass(frozen=True)
class D6DIIResidualCallback:
    kind: str
    function: Callable[..., Any]

    def __post_init__(self) -> None:
        if self.kind not in {"training_capture", "synthetic_positive", "frozen_self"}:
            raise PermissionError("D6D-II callback kind is outside the joint contract")
        if not callable(self.function):
            raise TypeError("D6D-II callback function must be callable")

    def __call__(self, **payload: Any) -> Any:
        return self.function(**payload)


@dataclass(frozen=True)
class D6DIIRequest:
    condition: str
    enabled: bool
    scale: float
    callback: D6DIIResidualCallback | None

    def validate(self) -> None:
        if type(self.enabled) is not bool or type(self.scale) is not float:
            raise PermissionError("D6D-II request flags must be exact bool and float")
        if self.condition == "training_capture_observer_zero_delta":
            expected = (True, 0.0, "training_capture")
        elif self.condition == "wrapper_off":
            expected = (False, 0.0, None)
        elif self.condition == "wrapper_zero":
            expected = (True, 0.0, None)
        elif self.condition == "synthetic_positive":
            expected = (True, 1.0, "synthetic_positive")
        elif self.condition in SELF_CONDITIONS:
            expected = (True, 1.0, "frozen_self")
        else:
            raise PermissionError("D6D-II request condition is not frozen")
        callback_kind = (
            self.callback.kind
            if type(self.callback) is D6DIIResidualCallback
            else None
        )
        if (self.enabled, self.scale, callback_kind) != expected:
            raise PermissionError("D6D-II request does not match its frozen condition")


class D6DIIFixedDispatcher:
    def __init__(self, request_context: ContextVar[Any]) -> None:
        self._request_context = request_context
        self.dispatch_count = 0
        self.callback_count = 0

    def __call__(self, **payload: Any) -> Any:
        request = self._request_context.get()
        if type(request) is not D6DIIRequest:
            raise PermissionError("D6D-II dispatcher requires one exact scoped request")
        residual = payload.get("residual_x")
        if residual is None:
            raise TypeError("D6D-II dispatcher requires residual_x")
        self.dispatch_count += 1
        callback = request.callback
        if request.condition == "training_capture_observer_zero_delta":
            if type(callback) is not D6DIIResidualCallback:
                raise PermissionError("D6D-II training capture callback is unavailable")
            output = callback(**payload)
            if output is not residual:
                raise RuntimeError("D6D-II training capture changed the residual object")
            self.callback_count += 1
            return residual
        if not request.enabled or request.scale == 0.0:
            return residual
        if type(callback) is not D6DIIResidualCallback:
            raise PermissionError("D6D-II active callback is unavailable")
        output = callback(**payload)
        if output is None:
            raise TypeError("D6D-II callback returned no residual")
        self.callback_count += 1
        return output


def _instance_snapshot(model: Any) -> dict[str, Any]:
    value = getattr(model, "__dict__", None)
    if not isinstance(value, dict):
        raise TypeError("D6D-II base model must expose an instance dictionary")
    return dict(value)


def _identical(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(left[name] is right[name] for name in left)


class D6DIIWrapperOwnedRuntime:
    runtime_version = RUNTIME_VERSION

    def __init__(
        self,
        *,
        base_model: Any,
        compiled_methods: Mapping[str, Callable[..., Any]],
        injection_counts: Mapping[str, int],
    ) -> None:
        if set(compiled_methods) != set(TARGET_METHODS) or not all(
            callable(compiled_methods[name]) for name in TARGET_METHODS
        ):
            raise TypeError("D6D-II requires two compiled instrumented methods")
        if dict(injection_counts) != {"forward_one": 1, "forward_seq": 1}:
            raise RuntimeError("D6D-II injection counts changed")
        base_snapshot = _instance_snapshot(base_model)
        context: ContextVar[Any] = ContextVar(
            f"psa_d6dii_request_{id(self)}", default=_NO_REQUEST
        )
        object.__setattr__(self, "_base_model", base_model)
        object.__setattr__(self, "_base_snapshot", base_snapshot)
        object.__setattr__(self, "_request_context", context)
        object.__setattr__(self, "_dispatcher", D6DIIFixedDispatcher(context))
        object.__setattr__(self, "_call_lock", threading.Lock())
        object.__setattr__(self, "execution_count", 0)
        object.__setattr__(self, "rejection_count", 0)
        object.__setattr__(self, CALLBACK_ATTRIBUTE, self._dispatcher)
        for name in TARGET_METHODS:
            object.__setattr__(self, name, types.MethodType(compiled_methods[name], self))
        object.__setattr__(self, "installation_count", 3)
        object.__setattr__(self, "_owned_snapshot", self._owned_snapshot_now())
        if not self.base_dictionary_is_stable():
            raise RuntimeError("D6D-II wrapper constructor changed the base dictionary")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base_model"), name)

    def _owned_snapshot_now(self) -> dict[str, Any]:
        return {
            CALLBACK_ATTRIBUTE: object.__getattribute__(self, CALLBACK_ATTRIBUTE),
            "forward_one": object.__getattribute__(self, "forward_one"),
            "forward_seq": object.__getattribute__(self, "forward_seq"),
            "forward_function": type(self).forward,
        }

    def base_dictionary_is_stable(self) -> bool:
        return _identical(self._base_snapshot, _instance_snapshot(self._base_model))

    def owned_bindings_are_stable(self) -> bool:
        return _identical(self._owned_snapshot, self._owned_snapshot_now())

    def context_is_empty(self) -> bool:
        return self._request_context.get() is _NO_REQUEST

    @property
    def dispatcher(self) -> D6DIIFixedDispatcher:
        return self._dispatcher

    def forward(
        self,
        tokens: Sequence[int],
        state: Any,
        *,
        full_output: bool,
        coupling: D6DIIRequest,
    ) -> tuple[Any, Any]:
        if type(coupling) is not D6DIIRequest:
            raise PermissionError("D6D-II runtime rejects non-exact requests")
        coupling.validate()
        if not self._call_lock.acquire(blocking=False):
            object.__setattr__(self, "rejection_count", self.rejection_count + 1)
            raise RuntimeError("D6D-II rejects nested or concurrent requests")
        context_token: Token[Any] | None = None
        try:
            if not self.base_dictionary_is_stable() or not self.owned_bindings_are_stable():
                raise RuntimeError("D6D-II wrapper identity changed before forward")
            context_token = self._request_context.set(coupling)
            token_copy = list(tokens)
            if not token_copy or not all(type(value) is int for value in token_copy):
                raise ValueError("D6D-II tokens must be non-empty exact integers")
            object.__setattr__(self, "execution_count", self.execution_count + 1)
            if len(token_copy) == 1:
                output = self.forward_one(token_copy[0], copy.deepcopy(state))
            else:
                output = self.forward_seq(token_copy, copy.deepcopy(state), bool(full_output))
            if not self.base_dictionary_is_stable() or not self.owned_bindings_are_stable():
                raise RuntimeError("D6D-II wrapper identity changed during forward")
            return output
        finally:
            if context_token is not None:
                self._request_context.reset(context_token)
            self._call_lock.release()


class TrainingCaptureCallback:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.vector: tuple[float, ...] | None = None
        self.invocation_count = 0
        self.application_count = 0

    def __call__(self, **payload: Any) -> Any:
        residual = payload.get("residual_x")
        self.invocation_count += 1
        if payload.get("layer_index") != TARGET_LAYER_INDEX:
            return residual
        if self.vector is not None:
            raise RuntimeError("D6D-II training capture observed target layer twice")
        selected = residual[-1] if len(tuple(residual.shape)) == 2 else residual
        if tuple(selected.shape) != (HIDDEN_DIMENSION,):
            raise RuntimeError("D6D-II captured residual shape changed")
        finite = bool(self.torch.isfinite(selected).all().item())
        if not finite:
            raise RuntimeError("D6D-II captured residual is non-finite")
        self.vector = tuple(float(value) for value in selected.detach().float().cpu().tolist())
        self.application_count += 1
        return residual


class SyntheticPositiveCallback:
    def __init__(self, torch: Any, seed_material: str) -> None:
        self.torch = torch
        self.seed_material = seed_material
        self.invocation_count = 0
        self.application_count = 0

    def __call__(self, **payload: Any) -> Any:
        residual = payload.get("residual_x")
        self.invocation_count += 1
        if payload.get("layer_index") != TARGET_LAYER_INDEX:
            return residual
        selected = residual[-1] if len(tuple(residual.shape)) == 2 else residual
        rms = self.torch.sqrt(self.torch.mean(selected.detach().float() ** 2))
        sign = 1.0 if hashlib.sha256(self.seed_material.encode()).digest()[0] < 128 else -1.0
        delta = self.torch.ones_like(residual) * (rms.to(residual.dtype) * 0.01 * sign)
        self.application_count += 1
        return residual + delta


class FrozenProjectionCallback:
    def __init__(self, torch: Any, vector: Sequence[float]) -> None:
        self.torch = torch
        self.vector = tuple(float(value) for value in vector)
        self.invocation_count = 0
        self.application_count = 0

    def __call__(self, **payload: Any) -> Any:
        residual = payload.get("residual_x")
        self.invocation_count += 1
        if payload.get("layer_index") != TARGET_LAYER_INDEX:
            return residual
        delta = self.torch.tensor(
            self.vector, dtype=residual.dtype, device=residual.device
        )
        self.application_count += 1
        return residual + delta


def request_for_training_capture(callback: TrainingCaptureCallback) -> D6DIIRequest:
    return D6DIIRequest(
        "training_capture_observer_zero_delta",
        True,
        0.0,
        D6DIIResidualCallback("training_capture", callback),
    )


def request_for_pilot_condition(
    condition: str, callback: Callable[..., Any] | None = None
) -> D6DIIRequest:
    if condition == "wrapper_off":
        return D6DIIRequest(condition, False, 0.0, None)
    if condition == "wrapper_zero":
        return D6DIIRequest(condition, True, 0.0, None)
    if condition == "synthetic_positive":
        return D6DIIRequest(
            condition, True, 1.0,
            D6DIIResidualCallback("synthetic_positive", callback) if callback else None,
        )
    if condition in SELF_CONDITIONS:
        return D6DIIRequest(
            condition, True, 1.0,
            D6DIIResidualCallback("frozen_self", callback) if callback else None,
        )
    raise ValueError("D6D-II pilot condition is unknown")


def _rms(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector) / len(vector))


def _scaled(vector: Sequence[float], target_rms: float) -> tuple[float, ...]:
    current = _rms(vector)
    if current == 0.0 or not math.isfinite(current):
        raise RuntimeError("D6D-II branch training vector has invalid RMS")
    return tuple(float(value) * target_rms / current for value in vector)


def build_projection_from_captures(
    *,
    captures: Mapping[tuple[str, str], Sequence[float]],
    training_manifest_sha256: str,
    pilot_manifest_sha256: str,
) -> dict[str, Any]:
    expected = {(identity, goal) for identity in IDENTITY_KEYS for goal in GOAL_KEYS}
    if set(captures) != expected:
        raise RuntimeError("D6D-II training capture grid is incomplete")
    vectors = {key: tuple(float(value) for value in captures[key]) for key in expected}
    if any(len(value) != HIDDEN_DIMENSION for value in vectors.values()):
        raise RuntimeError("D6D-II training capture dimension changed")
    grand = tuple(
        sum(vector[index] for vector in vectors.values()) / len(vectors)
        for index in range(HIDDEN_DIMENSION)
    )
    grand_rms = _rms(grand)
    if grand_rms == 0.0 or not math.isfinite(grand_rms):
        raise RuntimeError("D6D-II training grand mean RMS is invalid")
    identity_targets = {}
    for identity in IDENTITY_KEYS:
        mean = tuple(
            sum(vectors[(identity, goal)][index] for goal in GOAL_KEYS) / len(GOAL_KEYS)
            for index in range(HIDDEN_DIMENSION)
        )
        raw = tuple(value - grand[index] / 2.0 for index, value in enumerate(mean))
        identity_targets[identity] = _scaled(raw, grand_rms * 0.005)
    goal_targets = {}
    for goal in GOAL_KEYS:
        mean = tuple(
            sum(vectors[(identity, goal)][index] for identity in IDENTITY_KEYS)
            / len(IDENTITY_KEYS)
            for index in range(HIDDEN_DIMENSION)
        )
        raw = tuple(value - grand[index] / 2.0 for index, value in enumerate(mean))
        goal_targets[goal] = _scaled(raw, grand_rms * 0.005)
    records = tuple(
        ProjectionTrainingRecord(
            identity_key=identity,
            goal_key=goal,
            identity_target=identity_targets[identity],
            goal_target=goal_targets[goal],
        )
        for identity in IDENTITY_KEYS
        for goal in GOAL_KEYS
    )
    return build_frozen_projection_artifact(
        records=records,
        output_dimension=HIDDEN_DIMENSION,
        training_manifest_sha256=training_manifest_sha256,
        pilot_manifest_commitment_sha256=pilot_manifest_sha256,
        optimizer_seed=260825,
        fixture_only=False,
    )


def execute_projection_training(
    *, adapter: Any, runtime: D6DIIWrapperOwnedRuntime,
    torch: Any, training_manifest: Mapping[str, Any],
    training_manifest_sha256: str, pilot_manifest_sha256: str,
) -> dict[str, Any]:
    validate_training_manifest(training_manifest)
    records = expand_training_records(training_manifest)
    captures = {}
    call_records = []
    for record in records:
        callback = TrainingCaptureCallback(torch)
        tokens = adapter.encode(record["prompt_text"])
        with torch.inference_mode():
            output = runtime.forward(
                tokens, None, full_output=False,
                coupling=request_for_training_capture(callback),
            )
        if output is None or callback.vector is None:
            raise RuntimeError("D6D-II training capture produced no vector")
        if callback.application_count != 1 or callback.invocation_count != 32:
            raise RuntimeError("D6D-II training capture counts changed")
        key = (record["identity_key"], record["goal_key"])
        captures[key] = callback.vector
        call_records.append(
            {
                "record_id": record["record_id"],
                "prompt_sha256": record["prompt_sha256"],
                "token_count": len(tokens),
                "capture_digest_sha256": sha256_json(
                    [format(value, ".12e") for value in callback.vector]
                ),
            }
        )
    artifact = build_projection_from_captures(
        captures=captures,
        training_manifest_sha256=training_manifest_sha256,
        pilot_manifest_sha256=pilot_manifest_sha256,
    )
    return {
        "status": "d6d_real_projection_trained_and_frozen_before_pilot",
        "valid": True,
        "call_count": len(call_records),
        "calls": call_records,
        "artifact": artifact,
        "artifact_digest_sha256": artifact["artifact_digest_sha256"],
        "parameter_digest_sha256": artifact["parameter_digest_sha256"],
    }


def _field_item(item_id: str, value: str, update_class: str) -> dict[str, Any]:
    return {
        "field_item_id": item_id,
        "value": value,
        "value_type": "string",
        "confidence": 1.0,
        "update_class": update_class,
        "created_step": 0,
        "updated_step": 0,
        "source_evidence_ids": ["d6d-ii-frozen-noncore-pilot"],
        "status": "active",
    }


def _self_state(fixture_id: str, identity: str, goal: str, role: str) -> dict[str, Any]:
    return build_self_state(
        state_id=f"{fixture_id}-{role}",
        agent_instance_id="d6d-real-joint-pilot",
        trajectory_id=fixture_id,
        step=0,
        model_id="rwkv7-g1h-2.9b-20260710",
        tokenizer_id="rwkv_vocab_v20230424",
        fields={
            "identity_anchors": [
                _field_item(f"{fixture_id}-{role}-identity", identity, "protected")
            ],
            "active_goals": [
                _field_item(f"{fixture_id}-{role}-goal", goal, "fast")
            ],
        },
        provenance_refs=["d6d-ii-frozen-pilot-manifest"],
    )


def _logits_and_scores(
    output: tuple[Any, Any], torch: Any, sentinel_index: int
) -> tuple[str, dict[str, float], str]:
    logits = output[0]
    if len(tuple(logits.shape)) != 2:
        raise RuntimeError("D6D-II pilot full-output logits shape changed")
    selected = logits[-1]
    sentinel_logits = logits[sentinel_index]
    if not bool(torch.isfinite(selected).all().item()):
        raise RuntimeError("D6D-II pilot logits are non-finite")
    scores = {
        code: float(selected[token_id].detach().float().item())
        for code, token_id in CHOICE_TOKEN_IDS.items()
    }
    byte_view = selected.detach().contiguous().cpu().view(torch.uint8)
    digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    sentinel_code = max(
        CHOICE_TOKEN_IDS,
        key=lambda code: float(
            sentinel_logits[CHOICE_TOKEN_IDS[code]].detach().float().item()
        ),
    )
    return digest, scores, sentinel_code


def execute_joint_pilot(
    *, adapter: Any, runtime: D6DIIWrapperOwnedRuntime, torch: Any,
    artifact: Mapping[str, Any], pilot_manifest: Mapping[str, Any],
    seed_material: str,
) -> dict[str, Any]:
    validate_pilot_manifest(pilot_manifest)
    for code, token_id in CHOICE_TOKEN_IDS.items():
        encoded = adapter.encode(code)
        if encoded != [token_id] or adapter.decode(encoded) != code:
            raise RuntimeError("D6D-II answer token roundtrip changed")
    if adapter.encode(">\n") != list(FORCED_PREFIX_TOKEN_IDS):
        raise RuntimeError("D6D-II forced prefix tokenization changed")
    projection = FrozenSelfProjection(artifact)
    fixtures = expand_pilot_fixtures(pilot_manifest)
    calls = build_pilot_call_plan(pilot_manifest)
    sentinel_prefix_tokens = adapter.encode(
        pilot_manifest["general_capability_sentinel"]["prefix_text"]
    )
    if not sentinel_prefix_tokens:
        raise RuntimeError("D6D-II sentinel prefix tokenization is empty")
    sentinel_index = len(sentinel_prefix_tokens) - 1
    records = []
    callbacks = []
    for call in calls:
        fixture = fixtures[call["fixture_index"]]
        condition = call["condition"]
        callback_function = None
        projection_digest = None
        if condition == "synthetic_positive":
            callback_function = SyntheticPositiveCallback(
                torch, f"{seed_material}|{fixture['fixture_id']}"
            )
        elif condition in SELF_CONDITIONS:
            matched = _self_state(
                fixture["fixture_id"], fixture["matched_identity"],
                fixture["matched_goal"], "matched",
            )
            paired = _self_state(
                fixture["fixture_id"], fixture["paired_identity"],
                fixture["paired_goal"], "paired",
            )
            random_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed_material}|{fixture['fixture_id']}|{condition}".encode()
                ).digest()[:8], "big"
            )
            projected = projection.project_condition(
                matched_state=matched, paired_state=paired,
                condition=condition, random_seed=random_seed,
            )
            callback_function = FrozenProjectionCallback(
                torch, projected["aggregate_vector"]
            )
            projection_digest = projected["aggregate_digest_sha256"]
        request = request_for_pilot_condition(condition, callback_function)
        tokens = adapter.encode(fixture["query_text"]) + list(FORCED_PREFIX_TOKEN_IDS)
        with torch.inference_mode():
            output = runtime.forward(tokens, None, full_output=True, coupling=request)
        logits_digest, scores, sentinel_code = _logits_and_scores(
            output, torch, sentinel_index
        )
        if callback_function is not None:
            callbacks.append(callback_function)
            if callback_function.invocation_count != 32 or callback_function.application_count != 1:
                raise RuntimeError("D6D-II pilot callback counts changed")
        target = fixture["target_code"]
        margin = scores[target] - max(value for code, value in scores.items() if code != target)
        records.append(
            {
                "call_index": call["call_index"],
                "fixture_id": fixture["fixture_id"],
                "task_family": fixture["task_family"],
                "phase": call["phase"],
                "condition": condition,
                "target_code": target,
                "scores": scores,
                "target_margin": margin,
                "general_capability_sentinel_code": sentinel_code,
                "logits_sha256": logits_digest,
                "projection_sha256": projection_digest,
            }
        )
    by_fixture = {}
    for fixture in fixtures:
        scored = {
            record["condition"]: record
            for record in records
            if record["fixture_id"] == fixture["fixture_id"]
            and record["phase"] == "scored"
        }
        by_fixture[fixture["fixture_id"]] = {
            "off_zero_exact": scored["wrapper_off"]["logits_sha256"]
            == scored["wrapper_zero"]["logits_sha256"],
            "zero_double_mask_exact": scored["wrapper_zero"]["logits_sha256"]
            == scored["self_identity_goal_mask"]["logits_sha256"],
            "synthetic_differs": scored["synthetic_positive"]["logits_sha256"]
            != scored["wrapper_off"]["logits_sha256"],
            "matched_margin": scored["self_matched"]["target_margin"],
        }
    engineering = {
        "all_144_calls_recorded": len(records) == 144,
        "all_outputs_finite": all(
            all(math.isfinite(value) for value in record["scores"].values())
            for record in records
        ),
        "off_zero_exact_all": all(value["off_zero_exact"] for value in by_fixture.values()),
        "zero_double_mask_exact_all": all(
            value["zero_double_mask_exact"] for value in by_fixture.values()
        ),
        "synthetic_differs_all": all(
            value["synthetic_differs"] for value in by_fixture.values()
        ),
        "general_capability_sentinel_changes_within_bound": sum(
            record["general_capability_sentinel_code"]
            != next(
                item["general_capability_sentinel_code"]
                for item in records
                if item["fixture_id"] == record["fixture_id"]
                and item["phase"] == "scored"
                and item["condition"] == "wrapper_off"
            )
            for record in records
            if record["phase"] == "scored" and record["condition"] != "wrapper_off"
        ) <= 1,
        "wrapper_base_dictionary_stable": runtime.base_dictionary_is_stable(),
        "wrapper_bindings_stable": runtime.owned_bindings_are_stable(),
        "wrapper_context_empty": runtime.context_is_empty(),
    }
    if not all(engineering.values()):
        failed = [name for name, valid in engineering.items() if not valid]
        raise RuntimeError("D6D-II joint pilot engineering failure: " + ", ".join(failed))
    return {
        "status": "d6d_real_joint_noncore_pilot_engineering_complete",
        "valid": True,
        "engineering_checks": engineering,
        "records": records,
        "fixture_summaries": by_fixture,
        "call_count": len(records),
        "callback_count": len(callbacks),
        "classification": "noncore_engineering_pilot_not_self_effect_conclusion",
        "self_effect_conclusion": False,
    }
