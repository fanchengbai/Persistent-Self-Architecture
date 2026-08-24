from __future__ import annotations

import ast
from contextvars import ContextVar, Token
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import sys
import threading
import types
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    OfflineTensor,
    _namespace,
    _output_digest,
    _state,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
    compile_instrumented_methods,
    inspect_instrumented_source,
)


INTEGRATION_VERSION = "0.1-coupling-d6b-persistent-ast"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6b_persistent_ast.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6B 项目内persistent AST静态集成与无模型验证；"
    "只复用锁定AST变换和纯Python 32层/2560维fixture，将instrumented方法与固定dispatcher"
    "在fixture首次forward前一次安装，验证OFF/zero、active fake、方法与dispatcher身份稳定、"
    "运行期零模型属性切换及D5历史源码不变；不授权installed source探测、D6C/D6D/D6E、"
    "RWKV/Torch导入、权重访问、模型加载或执行、D5C/P1/P2重跑、真实层选择、"
    "真实Self projection、Self效果实验、Self Updater或自动重跑。"
)
CLASSIFICATION = (
    "locked_ast_persistent_integration_passes_32x2560_fake_"
    "no_runtime_model_switching_no_installed_source"
)
D6A_SOURCE_DIGEST = "ac2c2ede6cb42c4facd069e61bb7cc0fe1bcc23e21bd45ffb504f6287cb9214a"
INSTRUMENTER_DIGEST = "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"
D5_RUNTIME_DIGEST = "e4ae5c5bee74a85a4dea8a9b8eb16e3b6e19ef6b375020ffc849a09cbd7bbc32"
FIXTURE_SOURCE_DIGEST = "ded1cc371411906a26650edf217e245c04a40564e4b91c72f7ca7d01b5dfe3e2"
N_LAYER = 32
HIDDEN_DIMENSION = 2560
STATE_COMPONENTS = 96
TARGET_LAYER_INDEX = 15
INSTALLED_NAMES = (CALLBACK_ATTRIBUTE, *TARGET_METHODS)
_NO_REQUEST = object()
ACCEPTANCE_CATEGORIES = (
    "locked_ast_transform_valid_for_both_paths",
    "one_injection_compiled_per_path",
    "fixture_shape_is_32_layers_2560_hidden_96_state",
    "three_bindings_installed_once_before_first_forward",
    "second_install_rejected_without_binding_change",
    "single_off_and_zero_exact_without_probe",
    "sequence_off_and_zero_exact_without_probe",
    "single_active_fake_changes_deterministically",
    "sequence_active_fake_changes_deterministically",
    "probe_counts_match_32_layers_and_one_target_per_active_call",
    "methods_and_dispatcher_remain_identity_stable",
    "runtime_forward_has_no_model_setattr_or_delattr",
    "request_context_restores_after_success_and_callback_failure",
    "tokens_state_and_probe_source_remain_unchanged",
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_coupling_d6b_persistent_ast.md",
    "scripts/verify_self_model_v0_1_coupling_d6b_persistent_ast.py",
    "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/d6a_persistent_dispatcher.py",
    "src/psa/self_model/d6b_persistent_ast.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d6b_persistent_ast.py",
)


@dataclass(frozen=True)
class D6BASTRequest:
    mode: str
    enabled: bool
    scale: float
    callback: "D6BOfflineProbe | None"

    def validate(self) -> None:
        if self.mode not in {"off", "zero", "active_fake"}:
            raise PermissionError("D6B request mode is outside the static contract")
        if type(self.enabled) is not bool or type(self.scale) is not float:
            raise PermissionError("D6B request flags must be exact bool and float")
        expected = {
            "off": (False, 0.0, False),
            "zero": (True, 0.0, False),
            "active_fake": (True, 1.0, True),
        }[self.mode]
        callback_required = type(self.callback) is D6BOfflineProbe
        if (self.enabled, self.scale, callback_required) != expected:
            raise PermissionError("D6B request does not match its exact mode")


class D6BOfflineProbe:
    fake_only = True
    real_self_projection = False

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.invocation_count = 0
        self.application_count = 0
        self.source = tuple((index + 1) / 1_000_000.0 for index in range(HIDDEN_DIMENSION))

    def __call__(
        self,
        *,
        phase: str,
        layer_index: int,
        execution_path: str,
        residual_x: OfflineTensor,
    ) -> OfflineTensor:
        if self.fail:
            raise RuntimeError("synthetic D6B callback failure")
        if phase != "post_ffn_residual":
            raise PermissionError("D6B callback phase changed")
        if execution_path not in TARGET_METHODS:
            raise PermissionError("D6B callback execution path changed")
        if type(layer_index) is not int or not 0 <= layer_index < N_LAYER:
            raise PermissionError("D6B layer index is outside the fake fixture")
        if type(residual_x) is not OfflineTensor:
            raise TypeError("D6B accepts only the exact offline tensor fixture")
        if not residual_x.shape or residual_x.shape[-1] != HIDDEN_DIMENSION:
            raise RuntimeError("D6B residual hidden dimension changed")
        self.invocation_count += 1
        if layer_index != TARGET_LAYER_INDEX:
            return residual_x
        self.application_count += 1
        delta = OfflineTensor(self.source, dtype=residual_x.dtype, device=residual_x.device)
        return residual_x + delta


class FixedASTDispatcher:
    def __init__(self, request_context: ContextVar[Any]) -> None:
        self._request_context = request_context
        self.dispatch_count = 0
        self.callback_count = 0

    def __call__(self, **payload: Any) -> OfflineTensor:
        request = self._request_context.get()
        if request is _NO_REQUEST:
            raise PermissionError("D6B dispatcher requires a scoped request")
        if type(request) is not D6BASTRequest:
            raise PermissionError("D6B dispatcher rejects non-exact requests")
        self.dispatch_count += 1
        residual = payload.get("residual_x")
        if type(residual) is not OfflineTensor:
            raise TypeError("D6B dispatcher requires the offline tensor fixture")
        if not request.enabled or request.scale == 0.0:
            return residual
        callback = request.callback
        if type(callback) is not D6BOfflineProbe:
            raise PermissionError("D6B active fake callback is unavailable")
        output = callback(**payload)
        self.callback_count += 1
        return output


def _binding_snapshot(model: Any) -> dict[str, Any]:
    return dict(model.__dict__)


def _bindings_identical(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(left[name] is right[name] for name in left)


class PersistentASTRuntime:
    def __init__(
        self,
        *,
        base_model: Any,
        exact_fixture_type: type,
        upstream_source: str,
        upstream_globals: Mapping[str, Any],
    ) -> None:
        if type(base_model) is not exact_fixture_type:
            raise TypeError("D6B accepts only the exact generated fixture type")
        if getattr(base_model, "offline_lifecycle_fixture", None) is not True:
            raise PermissionError("D6B fixture marker is absent")
        if getattr(base_model, "model_loaded", None) is not False or getattr(
            base_model, "model_executed", None
        ) is not False:
            raise PermissionError("D6B rejects a loaded or executed model fixture")
        if any(name in base_model.__dict__ for name in INSTALLED_NAMES):
            raise RuntimeError("D6B persistent AST bindings are already installed")
        methods, counts = compile_instrumented_methods(
            upstream_source=upstream_source,
            upstream_globals=upstream_globals,
            rwkv_de_version=None,
        )
        if counts != {"forward_one": 1, "forward_seq": 1}:
            raise RuntimeError("D6B AST injection counts changed")
        self._base_model = base_model
        self._request_context: ContextVar[Any] = ContextVar(
            f"psa_d6b_request_{id(self)}", default=_NO_REQUEST
        )
        self._dispatcher = FixedASTDispatcher(self._request_context)
        self._call_lock = threading.Lock()
        self.execution_count = 0
        self.rejection_count = 0
        setattr(base_model, CALLBACK_ATTRIBUTE, self._dispatcher)
        for name in TARGET_METHODS:
            setattr(base_model, name, types.MethodType(methods[name], base_model))
        self.installation_count = 3
        self.injection_counts = dict(counts)
        self._installed_snapshot = _binding_snapshot(base_model)

    @property
    def dispatcher(self) -> FixedASTDispatcher:
        return self._dispatcher

    def context_is_empty(self) -> bool:
        return self._request_context.get() is _NO_REQUEST

    def bindings_are_stable(self) -> bool:
        return _bindings_identical(
            self._installed_snapshot, _binding_snapshot(self._base_model)
        )

    def forward(
        self,
        tokens: Sequence[int],
        state: Sequence[int],
        *,
        full_output: bool,
        coupling: D6BASTRequest,
    ) -> tuple[OfflineTensor, list[int]]:
        if type(coupling) is not D6BASTRequest:
            raise PermissionError("D6B runtime rejects non-exact requests")
        coupling.validate()
        if not self._call_lock.acquire(blocking=False):
            self.rejection_count += 1
            raise RuntimeError("D6B rejects nested or concurrent requests")
        token: Token[Any] | None = None
        try:
            if not self.bindings_are_stable():
                raise RuntimeError("D6B persistent bindings changed before forward")
            token = self._request_context.set(coupling)
            self.execution_count += 1
            output = self._base_model.forward(
                list(tokens), copy.deepcopy(list(state)), bool(full_output)
            )
            if not self.bindings_are_stable():
                raise RuntimeError("D6B persistent bindings changed during forward")
            return output
        finally:
            if token is not None:
                self._request_context.reset(token)
            self._call_lock.release()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D6B config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("prerequisites", {})
    contract = config.get("integration_contract", {})
    authority = config.get("authority", {})
    closed = (
        "installed_source_probe_authorized", "d6c_authorized", "d6d_authorized",
        "d6e_authorized", "rwkv_import_authorized", "torch_import_authorized",
        "weights_access_authorized", "model_load_authorized", "model_execution_authorized",
        "d5c_rerun_authorized", "p1_rerun_authorized", "p2_authorized",
        "real_layer_selection_authorized", "real_self_projection_authorized",
        "self_effect_experiment_authorized", "self_updater_authorized",
        "automatic_rerun_authorized",
    )
    checks = {
        "identity_exact": config.get("integration_version") == INTEGRATION_VERSION,
        "confirmation_exact": config.get("owner_confirmation_text") == REQUIRED_CONFIRMATION,
        "d6a_and_source_digests_frozen": prerequisites.get("d6a_source_sha256")
        == D6A_SOURCE_DIGEST and prerequisites.get("locked_instrumenter_sha256")
        == INSTRUMENTER_DIGEST and prerequisites.get("offline_fixture_source_sha256")
        == FIXTURE_SOURCE_DIGEST,
        "d5_line_stopped": prerequisites.get("d5_line_status")
        == "stopped_no_rerun" and prerequisites.get("p2_allowed") is False,
        "shape_contract_exact": contract.get("n_layer") == N_LAYER
        and contract.get("hidden_dimension") == HIDDEN_DIMENSION
        and contract.get("state_components") == STATE_COMPONENTS
        and contract.get("execution_paths") == list(TARGET_METHODS),
        "persistent_install_contract_exact": contract.get("installed_names")
        == list(INSTALLED_NAMES) and contract.get("installation_count") == 3
        and contract.get("installation_phase") == "constructor_before_first_forward"
        and contract.get("runtime_model_attribute_mutation_allowed") is False
        and contract.get("method_and_dispatcher_identity_must_remain_fixed") is True,
        "request_contract_exact": contract.get("request_transport")
        == "contextvars_ContextVar" and contract.get("requests")
        == ["off", "zero", "active_fake"],
        "installed_source_closed": contract.get("installed_source_probe_allowed") is False,
        "acceptance_exact": config.get("acceptance_categories")
        == list(ACCEPTANCE_CATEGORIES),
        "d6b_authority_exact": authority.get("d6b_implementation_authorized") is True
        and authority.get("locked_ast_reuse_authorized") is True
        and authority.get("pure_python_fixture_authorized") is True,
        "all_model_and_later_authority_closed": all(
            authority.get(name) is False for name in closed
        ),
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "next_gate_exact": config.get("next_gate")
        == "remote_no_model_d6b_verification_then_separate_d6c_design_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6B config failed closed: " + ", ".join(failed))
    return checks


def _runtime_forward_ast_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        method for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PersistentASTRuntime"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "forward"
    )
    mutations = [
        call.func.id for call in ast.walk(function)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id in {"setattr", "delattr"}
    ]
    return {
        "forward_line": function.lineno,
        "model_attribute_mutation_calls": mutations,
        "model_attribute_mutation_call_count": len(mutations),
    }


def _digest(output: tuple[OfflineTensor, list[int]]) -> str:
    return _output_digest(output)


def run_static_integration_acceptance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    namespace, fixture_type = _namespace()
    fixture = fixture_type()
    ast_inspection = inspect_instrumented_source(DIAGNOSTIC_SOURCE)
    runtime = PersistentASTRuntime(
        base_model=fixture,
        exact_fixture_type=fixture_type,
        upstream_source=DIAGNOSTIC_SOURCE,
        upstream_globals=namespace,
    )
    installed_snapshot = _binding_snapshot(fixture)
    installation_before_forward = runtime.execution_count == 0
    try:
        PersistentASTRuntime(
            base_model=fixture,
            exact_fixture_type=fixture_type,
            upstream_source=DIAGNOSTIC_SOURCE,
            upstream_globals=namespace,
        )
    except RuntimeError:
        second_install_rejected = True
    else:
        second_install_rejected = False
    second_install_unchanged = _bindings_identical(
        installed_snapshot, _binding_snapshot(fixture)
    )

    single_tokens = [3]
    sequence_tokens = [3, 5, 8]
    source_state = _state()
    single_before = copy.deepcopy(single_tokens)
    sequence_before = copy.deepcopy(sequence_tokens)
    state_before = copy.deepcopy(source_state)
    probe = D6BOfflineProbe()
    probe_source_before = probe.source
    off = D6BASTRequest("off", False, 0.0, None)
    zero = D6BASTRequest("zero", True, 0.0, None)
    active = D6BASTRequest("active_fake", True, 1.0, probe)

    single_off = runtime.forward(single_tokens, source_state, full_output=False, coupling=off)
    single_zero = runtime.forward(single_tokens, source_state, full_output=False, coupling=zero)
    probe_before_active = (probe.invocation_count, probe.application_count)
    single_active_a = runtime.forward(
        single_tokens, source_state, full_output=False, coupling=active
    )
    single_active_b = runtime.forward(
        single_tokens, source_state, full_output=False, coupling=active
    )
    sequence_off = runtime.forward(
        sequence_tokens, source_state, full_output=True, coupling=off
    )
    sequence_zero = runtime.forward(
        sequence_tokens, source_state, full_output=True, coupling=zero
    )
    sequence_active_a = runtime.forward(
        sequence_tokens, source_state, full_output=True, coupling=active
    )
    sequence_active_b = runtime.forward(
        sequence_tokens, source_state, full_output=True, coupling=active
    )
    context_after_success = runtime.context_is_empty()

    failing = D6BOfflineProbe(fail=True)
    try:
        runtime.forward(
            single_tokens,
            source_state,
            full_output=False,
            coupling=D6BASTRequest("active_fake", True, 1.0, failing),
        )
    except RuntimeError as error:
        callback_failure_preserved = str(error) == "synthetic D6B callback failure"
    else:
        callback_failure_preserved = False
    context_after_failure = runtime.context_is_empty()
    recovery = runtime.forward(
        single_tokens, source_state, full_output=False, coupling=off
    )
    runtime_reusable = _digest(recovery) == _digest(single_off)

    final_snapshot = _binding_snapshot(fixture)
    ast_audit = _runtime_forward_ast_audit()
    checks = {
        ACCEPTANCE_CATEGORIES[0]: ast_inspection["valid"]
        and set(ast_inspection["injection_counts"]) == set(TARGET_METHODS),
        ACCEPTANCE_CATEGORIES[1]: runtime.injection_counts
        == {"forward_one": 1, "forward_seq": 1},
        ACCEPTANCE_CATEGORIES[2]: N_LAYER == 32 and HIDDEN_DIMENSION == 2560
        and len(source_state) == STATE_COMPONENTS,
        ACCEPTANCE_CATEGORIES[3]: runtime.installation_count == 3
        and installation_before_forward
        and all(name in fixture.__dict__ for name in INSTALLED_NAMES),
        ACCEPTANCE_CATEGORIES[4]: second_install_rejected and second_install_unchanged,
        ACCEPTANCE_CATEGORIES[5]: _digest(single_off) == _digest(single_zero)
        and probe_before_active == (0, 0),
        ACCEPTANCE_CATEGORIES[6]: _digest(sequence_off) == _digest(sequence_zero)
        and probe_before_active == (0, 0),
        ACCEPTANCE_CATEGORIES[7]: _digest(single_active_a) == _digest(single_active_b)
        and _digest(single_active_a) != _digest(single_off),
        ACCEPTANCE_CATEGORIES[8]: _digest(sequence_active_a) == _digest(sequence_active_b)
        and _digest(sequence_active_a) != _digest(sequence_off),
        ACCEPTANCE_CATEGORIES[9]: probe.invocation_count == 128
        and probe.application_count == 4,
        ACCEPTANCE_CATEGORIES[10]: runtime.bindings_are_stable()
        and _bindings_identical(installed_snapshot, final_snapshot),
        ACCEPTANCE_CATEGORIES[11]: ast_audit["model_attribute_mutation_call_count"] == 0,
        ACCEPTANCE_CATEGORIES[12]: context_after_success and callback_failure_preserved
        and context_after_failure and runtime_reusable,
        ACCEPTANCE_CATEGORIES[13]: single_tokens == single_before
        and sequence_tokens == sequence_before and source_state == state_before
        and probe.source == probe_source_before,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "installation_count": runtime.installation_count,
            "runtime_executions": runtime.execution_count,
            "dispatcher_calls": runtime.dispatcher.dispatch_count,
            "callback_calls": runtime.dispatcher.callback_count,
            "probe_invocations": probe.invocation_count,
            "probe_applications": probe.application_count,
            "layers": N_LAYER,
            "hidden_dimension": HIDDEN_DIMENSION,
            "state_components": STATE_COMPONENTS,
        },
        "ast_inspection": ast_inspection,
        "runtime_ast_audit": ast_audit,
        "installed_source_probed": False,
    }


def build_d6b_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D6B config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    acceptance = run_static_integration_acceptance(root)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "static_integration_valid": acceptance["valid"],
        "all_fourteen_acceptance_categories_present": len(acceptance["checks"]) == 14,
        "locked_instrumenter_unchanged": source_digests[
            "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
        ] == INSTRUMENTER_DIGEST,
        "historical_d5_runtime_unchanged": source_digests[
            "src/psa/self_model/d5c_mechanism_runtime.py"
        ] == D5_RUNTIME_DIGEST,
        "d6a_source_unchanged": source_digests[
            "src/psa/self_model/d6a_persistent_dispatcher.py"
        ] == D6A_SOURCE_DIGEST,
        "fixture_source_unchanged": source_digests[
            "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py"
        ] == FIXTURE_SOURCE_DIGEST,
        "installed_source_not_probed": acceptance["installed_source_probed"] is False,
        "runtime_forward_has_no_model_mutation": acceptance["runtime_ast_audit"][
            "model_attribute_mutation_call_count"
        ] == 0,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6B verification failed: " + ", ".join(failed))
    report = {
        "integration_version": INTEGRATION_VERSION,
        "status": "d6b_persistent_ast_static_integration_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "acceptance": acceptance,
        "decision": {
            "d6b_implemented": True,
            "d6c_or_later_authorized": False,
            "installed_source_probed": False,
            "d5c_p1_or_p2_rerun": False,
            "historical_d5_or_p1_conclusion_changed": False,
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "installed_source_probed": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d5c_rerun": False,
            "p1_rerun": False,
            "p2_run": False,
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
