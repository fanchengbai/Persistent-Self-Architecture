from __future__ import annotations

import ast
from contextvars import ContextVar, Token
from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
import sys
import threading
import types
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json


CONTRACT_VERSION = "0.1-coupling-d6a-persistent-dispatcher"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6a_persistent_dispatcher.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6A persistent-instrumented dispatcher纯离线"
    "contract与fake lifecycle实现；只使用合成Python fixture验证一次安装、固定dispatcher、"
    "context-local OFF/zero/active请求、无模型对象属性切换、嵌套/并发失败关闭及输入不变性；"
    "不授权D6B/D6C/D6D/D6E、RWKV/Torch导入、权重访问、模型加载或执行、D5C/P1/P2重跑、"
    "真实层选择、真实Self projection、Self效果实验、Self Updater或自动重跑。"
)
CLASSIFICATION = (
    "persistent_dispatcher_fake_lifecycle_passes_no_model_object_switching_"
    "d5_line_still_stopped"
)
D5C_RUNTIME_DIGEST = "e4ae5c5bee74a85a4dea8a9b8eb16e3b6e19ef6b375020ffc849a09cbd7bbc32"
DISPATCHER_ATTRIBUTE = "_psa_d6_persistent_dispatcher"
TARGET_METHODS = ("forward_one", "forward_seq")
INSTALLED_NAMES = (DISPATCHER_ATTRIBUTE, *TARGET_METHODS)
LAYER_COUNT = 4
HIDDEN_DIMENSION = 8
ACTIVE_LAYERS = (1, 3)
_NO_REQUEST = object()
ACCEPTANCE_CATEGORIES = (
    "three_names_installed_once_before_first_forward",
    "second_install_rejected_without_binding_change",
    "dispatcher_and_method_identities_fixed_across_lifecycle",
    "off_and_zero_outputs_exact_with_no_probe_call",
    "active_output_changes_deterministically",
    "context_request_restored_after_success",
    "nested_request_rejected_before_inner_forward",
    "concurrent_request_rejected_before_inner_forward",
    "callback_exception_restores_context_and_runtime_reusable",
    "tokens_state_and_probe_source_unchanged",
    "forward_source_has_no_model_setattr_or_delattr",
    "historical_d5_runtime_unchanged",
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_post_p1_research_review.md",
    "docs/self_model_v0_1_coupling_d6a_persistent_dispatcher.md",
    "scripts/verify_self_model_v0_1_coupling_d6a_persistent_dispatcher.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/d6a_persistent_dispatcher.py",
    "tests/test_self_model_d6a_persistent_dispatcher.py",
)


@dataclass(frozen=True)
class D6ACouplingRequest:
    enabled: bool
    scale: float
    callback: Callable[..., tuple[float, ...]] | None

    def validate(self) -> None:
        if type(self.enabled) is not bool or type(self.scale) is not float:
            raise PermissionError("D6A request flags must be exact bool and float")
        if self.scale not in {0.0, 1.0}:
            raise PermissionError("D6A scale must be exactly zero or one")
        if not self.enabled and (self.scale != 0.0 or self.callback is not None):
            raise PermissionError("D6A OFF request must be disabled zero without callback")
        if self.enabled and self.scale == 1.0 and not callable(self.callback):
            raise PermissionError("D6A active request requires a callback")


@dataclass
class SyntheticCallLedger:
    inner_forward_calls: int = 0
    before_forward: Callable[[], None] | None = None


class SyntheticPersistentFixture:
    fixture_kind = "d6a_synthetic_python"

    def __init__(self, ledger: SyntheticCallLedger | None = None) -> None:
        self._ledger = SyntheticCallLedger() if ledger is None else ledger

    def forward(
        self, tokens: Sequence[int], state: Any = None, full_output: bool = False
    ) -> tuple[Any, Any]:
        self._ledger.inner_forward_calls += 1
        if self._ledger.before_forward is not None:
            self._ledger.before_forward()
        if len(tokens) == 1:
            return self.forward_one(tokens, state, full_output)
        return self.forward_seq(tokens, state, full_output)

    def forward_one(
        self, tokens: Sequence[int], state: Any, full_output: bool
    ) -> tuple[Any, Any]:
        raise RuntimeError("D6A original single method must be replaced before first forward")

    def forward_seq(
        self, tokens: Sequence[int], state: Any, full_output: bool
    ) -> tuple[Any, Any]:
        raise RuntimeError("D6A original sequence method must be replaced before first forward")


def _base_residual(tokens: Sequence[int]) -> tuple[float, ...]:
    token_sum = sum(int(token) for token in tokens)
    return tuple((token_sum + index + 1) / 100.0 for index in range(HIDDEN_DIMENSION))


def _layer_step(residual: tuple[float, ...], layer_index: int) -> tuple[float, ...]:
    increment = (layer_index + 1) / 1000.0
    return tuple(value + increment for value in residual)


def _persistent_forward(
    model: SyntheticPersistentFixture,
    tokens: Sequence[int],
    state: Any,
    *,
    path: str,
) -> tuple[Any, Any]:
    residual = _base_residual(tokens)
    dispatcher = getattr(model, DISPATCHER_ATTRIBUTE)
    for layer_index in range(LAYER_COUNT):
        residual = _layer_step(residual, layer_index)
        residual = dispatcher(residual, layer_index=layer_index, path=path)
    output: Any = residual[0] if path == "single" else tuple(residual for _ in tokens)
    return output, copy.deepcopy(state)


def _instrumented_forward_one(
    self: SyntheticPersistentFixture,
    tokens: Sequence[int],
    state: Any,
    full_output: bool,
) -> tuple[Any, Any]:
    return _persistent_forward(self, tokens, state, path="single")


def _instrumented_forward_seq(
    self: SyntheticPersistentFixture,
    tokens: Sequence[int],
    state: Any,
    full_output: bool,
) -> tuple[Any, Any]:
    return _persistent_forward(self, tokens, state, path="sequence")


class FixedPersistentDispatcher:
    def __init__(self, request_context: ContextVar[Any]) -> None:
        self._request_context = request_context
        self.dispatch_count = 0
        self.callback_count = 0

    def __call__(
        self,
        residual: tuple[float, ...],
        *,
        layer_index: int,
        path: str,
    ) -> tuple[float, ...]:
        request = self._request_context.get()
        if request is _NO_REQUEST:
            raise PermissionError("D6A dispatcher requires a scoped request")
        if type(request) is not D6ACouplingRequest:
            raise PermissionError("D6A dispatcher rejects non-exact requests")
        self.dispatch_count += 1
        if not request.enabled or request.scale == 0.0:
            return residual
        callback = request.callback
        if not callable(callback):
            raise PermissionError("D6A active callback is unavailable")
        projected = callback(residual, layer_index=layer_index, path=path)
        if type(projected) is not tuple or len(projected) != len(residual):
            raise TypeError("D6A callback output shape is invalid")
        if not all(type(value) is float and math.isfinite(value) for value in projected):
            raise ValueError("D6A callback output must be finite floats")
        self.callback_count += 1
        return projected


class SyntheticProbe:
    def __init__(self) -> None:
        self.invocation_count = 0
        self.application_count = 0
        self.source = tuple((index + 1) / 200.0 for index in range(HIDDEN_DIMENSION))

    def __call__(
        self,
        residual: tuple[float, ...],
        *,
        layer_index: int,
        path: str,
    ) -> tuple[float, ...]:
        self.invocation_count += 1
        if layer_index not in ACTIVE_LAYERS:
            return residual
        self.application_count += 1
        return tuple(left + right for left, right in zip(residual, self.source))


class FailingProbe:
    def __call__(self, *_: Any, **__: Any) -> tuple[float, ...]:
        raise RuntimeError("synthetic callback failure")


def _binding_snapshot(model: SyntheticPersistentFixture) -> dict[str, Any]:
    return dict(model.__dict__)


def _bindings_identical(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return left.keys() == right.keys() and all(left[name] is right[name] for name in left)


class PersistentInstrumentedRuntime:
    def __init__(self, model: SyntheticPersistentFixture) -> None:
        if type(model) is not SyntheticPersistentFixture:
            raise TypeError("D6A accepts only the exact synthetic fixture")
        if any(name in model.__dict__ for name in INSTALLED_NAMES):
            raise RuntimeError("D6A persistent bindings are already installed")
        self._model = model
        self._request_context: ContextVar[Any] = ContextVar(
            f"psa_d6a_request_{id(self)}", default=_NO_REQUEST
        )
        self._dispatcher = FixedPersistentDispatcher(self._request_context)
        self._call_lock = threading.Lock()
        self.execution_count = 0
        self.rejection_count = 0
        setattr(model, DISPATCHER_ATTRIBUTE, self._dispatcher)
        setattr(model, "forward_one", types.MethodType(_instrumented_forward_one, model))
        setattr(model, "forward_seq", types.MethodType(_instrumented_forward_seq, model))
        self.installation_count = 3
        self._installed_snapshot = _binding_snapshot(model)

    @property
    def dispatcher(self) -> FixedPersistentDispatcher:
        return self._dispatcher

    def context_is_empty(self) -> bool:
        return self._request_context.get() is _NO_REQUEST

    def bindings_are_stable(self) -> bool:
        return _bindings_identical(
            self._installed_snapshot, _binding_snapshot(self._model)
        )

    def forward(
        self,
        tokens: Sequence[int],
        state: Any,
        *,
        full_output: bool = False,
        coupling: D6ACouplingRequest,
    ) -> tuple[Any, Any]:
        if type(coupling) is not D6ACouplingRequest:
            raise PermissionError("D6A runtime rejects non-exact requests")
        coupling.validate()
        if not self._call_lock.acquire(blocking=False):
            self.rejection_count += 1
            raise RuntimeError("D6A rejects nested or concurrent requests")
        token: Token[Any] | None = None
        try:
            if not self.bindings_are_stable():
                raise RuntimeError("D6A model bindings changed before forward")
            token = self._request_context.set(coupling)
            self.execution_count += 1
            output = self._model.forward(list(tokens), state, full_output)
            if not self.bindings_are_stable():
                raise RuntimeError("D6A model bindings changed during forward")
            return output
        finally:
            if token is not None:
                self._request_context.reset(token)
            self._call_lock.release()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D6A config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    authority = config.get("authority", {})
    prerequisites = config.get("prerequisites", {})
    contract = config.get("contract", {})
    closed = (
        "d6b_authorized", "d6c_authorized", "d6d_authorized", "d6e_authorized",
        "rwkv_import_authorized", "torch_import_authorized", "weights_access_authorized",
        "model_load_authorized", "model_execution_authorized", "d5c_rerun_authorized",
        "p1_rerun_authorized", "p2_authorized", "real_layer_selection_authorized",
        "real_self_projection_authorized", "self_effect_experiment_authorized",
        "self_updater_authorized", "automatic_rerun_authorized",
    )
    checks = {
        "identity_exact": config.get("contract_version") == CONTRACT_VERSION,
        "confirmation_exact": config.get("owner_confirmation_text") == REQUIRED_CONFIRMATION,
        "d5_line_stopped": prerequisites.get("d5c_decision")
        == "stop_without_rerun_preserved"
        and prerequisites.get("p1_decision")
        == "failed_claim_consumed_no_rerun_preserved"
        and prerequisites.get("p2_same_question_allowed") is False,
        "historical_runtime_digest_frozen": prerequisites.get("d5c_runtime_sha256")
        == D5C_RUNTIME_DIGEST,
        "install_contract_exact": contract.get("install_phase")
        == "constructor_before_first_forward"
        and contract.get("installed_names") == list(INSTALLED_NAMES)
        and contract.get("installation_count") == 3,
        "fixed_identity_contract": contract.get("dispatcher_identity_fixed_for_lifetime")
        is True and contract.get("method_identity_fixed_for_lifetime") is True
        and contract.get("forward_time_model_setattr_or_delattr_allowed") is False,
        "request_contract_exact": contract.get("request_transport")
        == "contextvars_ContextVar" and contract.get("requests")
        == ["off", "zero", "active"],
        "lifecycle_contract_exact": contract.get("nested_request_policy")
        == "reject_before_inner_forward"
        and contract.get("concurrent_request_policy") == "reject_before_inner_forward"
        and contract.get("exception_policy")
        == "restore_context_discard_output_and_remain_reusable"
        and contract.get("input_mutation_allowed") is False,
        "acceptance_exact": config.get("acceptance_categories")
        == list(ACCEPTANCE_CATEGORIES),
        "d6a_authority_exact": authority.get("d6a_implementation_authorized") is True
        and authority.get("synthetic_python_fixture_authorized") is True,
        "all_later_and_model_authority_closed": all(
            authority.get(name) is False for name in closed
        ),
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "next_gate_exact": config.get("next_gate")
        == "remote_no_model_d6a_verification_then_separate_d6b_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6A config failed closed: " + ", ".join(failed))
    return checks


def _forward_ast_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PersistentInstrumentedRuntime"
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    calls = [
        node.func.id for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"setattr", "delattr"}
    ]
    return {
        "forward_line": function.lineno,
        "model_attribute_mutation_calls": calls,
        "model_attribute_mutation_call_count": len(calls),
    }


def run_fake_lifecycle_acceptance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ledger = SyntheticCallLedger()
    model = SyntheticPersistentFixture(ledger)
    runtime = PersistentInstrumentedRuntime(model)
    installed_snapshot = _binding_snapshot(model)
    installation_before_forward = ledger.inner_forward_calls == 0
    try:
        PersistentInstrumentedRuntime(model)
    except RuntimeError:
        second_install_rejected = True
    else:
        second_install_rejected = False
    second_install_unchanged = _bindings_identical(
        installed_snapshot, _binding_snapshot(model)
    )

    tokens = [3, 5]
    state = {"identity": "amber", "goal": ["orbit", 7]}
    tokens_before = copy.deepcopy(tokens)
    state_before = copy.deepcopy(state)
    probe = SyntheticProbe()
    probe_source_before = probe.source
    off = D6ACouplingRequest(False, 0.0, None)
    zero = D6ACouplingRequest(True, 0.0, probe)
    active = D6ACouplingRequest(True, 1.0, probe)
    off_output = runtime.forward(tokens, state, coupling=off)
    zero_output = runtime.forward(tokens, state, coupling=zero)
    probe_calls_before_active = probe.invocation_count
    active_output_a = runtime.forward(tokens, state, coupling=active)
    active_output_b = runtime.forward(tokens, state, coupling=active)
    stable_after_routes = runtime.bindings_are_stable()
    success_context_empty = runtime.context_is_empty()

    nested_before = ledger.inner_forward_calls
    nested_rejected = False

    class NestedProbe:
        def __call__(self, residual: tuple[float, ...], **_: Any) -> tuple[float, ...]:
            nonlocal nested_rejected
            try:
                runtime.forward([9], None, coupling=off)
            except RuntimeError:
                nested_rejected = True
            return residual

    runtime.forward([9], None, coupling=D6ACouplingRequest(True, 1.0, NestedProbe()))
    nested_inner_delta = ledger.inner_forward_calls - nested_before

    entered = threading.Event()
    release = threading.Event()

    def block_first_forward() -> None:
        entered.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("D6A synthetic concurrency fixture timed out")

    ledger.before_forward = block_first_forward
    thread_errors: list[BaseException] = []

    def outer_call() -> None:
        try:
            runtime.forward([4], None, coupling=off)
        except BaseException as error:
            thread_errors.append(error)

    thread = threading.Thread(target=outer_call)
    concurrent_before = ledger.inner_forward_calls
    thread.start()
    if not entered.wait(timeout=5.0):
        release.set()
        thread.join(timeout=5.0)
        raise TimeoutError("D6A outer concurrency call did not enter")
    try:
        runtime.forward([4], None, coupling=off)
    except RuntimeError:
        concurrent_rejected = True
    else:
        concurrent_rejected = False
    concurrent_during = ledger.inner_forward_calls - concurrent_before
    release.set()
    thread.join(timeout=5.0)
    ledger.before_forward = None
    if thread.is_alive() or thread_errors:
        raise RuntimeError("D6A synthetic outer concurrency call failed")

    try:
        runtime.forward(
            [7], None, coupling=D6ACouplingRequest(True, 1.0, FailingProbe())
        )
    except RuntimeError as error:
        callback_failure_preserved = str(error) == "synthetic callback failure"
    else:
        callback_failure_preserved = False
    empty_after_failure = runtime.context_is_empty()
    recovery_output = runtime.forward([7], None, coupling=off)
    reusable_after_failure = recovery_output is not None and runtime.context_is_empty()

    final_snapshot = _binding_snapshot(model)
    ast_audit = _forward_ast_audit()
    checks = {
        ACCEPTANCE_CATEGORIES[0]: runtime.installation_count == 3
        and installation_before_forward
        and all(name in model.__dict__ for name in INSTALLED_NAMES),
        ACCEPTANCE_CATEGORIES[1]: second_install_rejected and second_install_unchanged,
        ACCEPTANCE_CATEGORIES[2]: stable_after_routes
        and _bindings_identical(installed_snapshot, final_snapshot),
        ACCEPTANCE_CATEGORIES[3]: off_output == zero_output
        and probe_calls_before_active == 0,
        ACCEPTANCE_CATEGORIES[4]: active_output_a == active_output_b
        and active_output_a != off_output and probe.application_count > 0,
        ACCEPTANCE_CATEGORIES[5]: success_context_empty,
        ACCEPTANCE_CATEGORIES[6]: nested_rejected and nested_inner_delta == 1,
        ACCEPTANCE_CATEGORIES[7]: concurrent_rejected and concurrent_during == 1,
        ACCEPTANCE_CATEGORIES[8]: callback_failure_preserved
        and empty_after_failure and reusable_after_failure,
        ACCEPTANCE_CATEGORIES[9]: tokens == tokens_before and state == state_before
        and probe.source == probe_source_before,
        ACCEPTANCE_CATEGORIES[10]: ast_audit["model_attribute_mutation_call_count"] == 0,
        ACCEPTANCE_CATEGORIES[11]: sha256_file(
            root / "src/psa/self_model/d5c_mechanism_runtime.py"
        ) == D5C_RUNTIME_DIGEST,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "installation_count": runtime.installation_count,
            "inner_forward_calls": ledger.inner_forward_calls,
            "runtime_executions": runtime.execution_count,
            "runtime_rejections": runtime.rejection_count,
            "dispatcher_calls": runtime.dispatcher.dispatch_count,
            "callback_calls": runtime.dispatcher.callback_count,
            "probe_invocations": probe.invocation_count,
            "probe_applications": probe.application_count,
        },
        "ast_audit": ast_audit,
    }


def build_d6a_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D6A config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    acceptance = run_fake_lifecycle_acceptance(root)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "fake_lifecycle_valid": acceptance["valid"],
        "all_twelve_acceptance_categories_present": len(acceptance["checks"]) == 12,
        "persistent_forward_has_no_model_attribute_mutation": acceptance[
            "ast_audit"
        ]["model_attribute_mutation_call_count"] == 0,
        "historical_d5_runtime_unchanged": source_digests[
            "src/psa/self_model/d5c_mechanism_runtime.py"
        ] == D5C_RUNTIME_DIGEST,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6A verification failed: " + ", ".join(failed))
    report = {
        "contract_version": CONTRACT_VERSION,
        "status": "d6a_persistent_dispatcher_fake_lifecycle_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "acceptance": acceptance,
        "decision": {
            "d6a_implemented": True,
            "d6b_or_later_authorized": False,
            "d5c_p1_or_p2_rerun": False,
            "historical_d5_or_p1_conclusion_changed": False,
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d5c_rerun": False,
            "p1_rerun": False,
            "p2_run": False,
            "d6b_authorized": False,
            "d6c_authorized": False,
            "d6d_authorized": False,
            "d6e_authorized": False,
            "real_layer_selected": False,
            "real_self_projection_constructed": False,
            "self_effect_experiment_run": False,
            "self_updater_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
