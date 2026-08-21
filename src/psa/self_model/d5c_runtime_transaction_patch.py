from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model import d5c_mechanism_runtime as runtime_module
from psa.self_model.d5b_static_active import SYNTHETIC_SOURCE, _state, _synthetic_namespace
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    run_plain_python_lifecycle_fixture,
)
from psa.self_model.d5c_mechanism_runtime import (
    D5CCleanupTransactionError,
    D5CCouplingRequest,
    RWKV7D5CActiveRuntime,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    TARGET_METHODS,
)


PATCH_VERSION = "0.1-d5c-runtime-transaction-patch"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_runtime_transaction_patch.json"
)
HISTORICAL_WRAPPER_DIGEST = (
    "e1de359da6d2087721dfd433a3e6ad90c6439bb474325a768c2f1d07fb08b5b7"
)
FAKE_TRANSACTION_REPORT_DIGEST = (
    "52519a5f1968ae5096ea35a31af885057696928545f53d93f9c0762cf2f3f57b"
)
REAL_D5C_REPORT_DIGEST = (
    "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21"
)
CLASSIFICATION = (
    "real_runtime_transaction_patch_passes_no_model_protocol_and_lifecycle_"
    "acceptance_real_2_9b_validation_not_run"
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_runtime_transaction_patch.md",
    "scripts/verify_self_model_v0_1_d5c_runtime_transaction_patch.py",
    "src/psa/self_model/d5c_runtime_transaction_patch.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d5c_runtime_transaction_patch.py",
)
MANAGED_NAMES = (CALLBACK_ATTRIBUTE, *TARGET_METHODS)


def _object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C runtime transaction patch config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisite = config.get("frozen_prerequisites")
    authority = config.get("authority")
    if not isinstance(prerequisite, Mapping) or not isinstance(authority, Mapping):
        raise ValueError("D5C runtime transaction patch config is incomplete")
    checks = {
        "identity_exact": config.get("patch_version") == PATCH_VERSION
        and config.get("stage") == "Coupling-D5C_real_runtime_transaction_patch"
        and config.get("status") == "real_runtime_patch_no_model_validation_only"
        and config.get("development_only") is True,
        "confirmation_bound": config.get("owner_confirmation_text")
        == "我会去远程执行，你继续下一轮"
        and config.get("confirmation_context")
        == "Implement the single pending real D5C runtime transaction patch, validate locally without RWKV or Torch, push main, and leave remote execution to the owner.",
        "failure_history_frozen": prerequisite
        == {
            "d5c_real_report_sha256": REAL_D5C_REPORT_DIGEST,
            "fake_transaction_report_sha256": FAKE_TRANSACTION_REPORT_DIGEST,
            "historical_d5c_wrapper_source_sha256": HISTORICAL_WRAPPER_DIGEST,
            "d5c_status": "d5c_mechanism_smoke_failed",
            "decision_effect": "stop_without_rerun",
        },
        "transaction_exact": config.get("transaction")
        == {
            "managed_names": list(MANAGED_NAMES),
            "phases": ["capture", "install", "forward", "restore", "verify", "commit_or_fail_closed"],
            "same_model_nested_or_concurrent_policy": "reject_before_inner_mutation",
            "cleanup_failure_policy": "attempt_all_names_discard_output_and_raise",
            "verification_uses_extra_forward": False,
        },
        "authority_exact": authority
        == {
            "real_runtime_transaction_patch_authorized": True,
            "local_no_model_validation_authorized": True,
            "remote_execution_delegated_to_owner": True,
            "rwkv_import_authorized": False,
            "torch_import_authorized": False,
            "weights_access_authorized": False,
            "model_load_authorized": False,
            "model_execution_authorized": False,
            "d5c_rerun_authorized": False,
            "d5d_authorized": False,
            "d5e_authorized": False,
            "formal_test_set_authorized": False,
            "self_effect_conclusion_authorized": False,
            "real_self_projection_authorized": False,
            "self_updater_authorized": False,
            "automatic_rerun_authorized": False,
        },
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "next_gate_exact": config.get("next_gate")
        == "owner_runs_remote_no_model_verification_then_reports_output",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C runtime transaction patch failed closed: " + ", ".join(failed))
    return checks


def _runtime(base_model: Any, namespace: Mapping[str, Any]) -> RWKV7D5CActiveRuntime:
    source_bytes = SYNTHETIC_SOURCE.encode("utf-8")
    digest = hashlib.sha256(source_bytes).hexdigest()
    original = runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256
    try:
        runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256 = digest
        return RWKV7D5CActiveRuntime(
            base_model=base_model,
            upstream_source_bytes=source_bytes,
            upstream_globals=namespace,
            upstream_package_version="0.8.32",
            upstream_de_version=None,
            execution_claim_sha256="a" * 64,
            machine_authorization_sha256="b" * 64,
        )
    finally:
        runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256 = original


def _off_request() -> D5CCouplingRequest:
    return D5CCouplingRequest(enabled=False, scale=0.0, callback=None)


def _restored(model: Any) -> bool:
    return not any(name in model.__dict__ for name in MANAGED_NAMES) and not model.__dict__.get(
        "_side_dispatch", {}
    )


class _ProtocolMixin:
    cooperative = True

    def __getattribute__(self, name: str) -> Any:
        if name in MANAGED_NAMES:
            values = object.__getattribute__(self, "__dict__").get("_side_dispatch", {})
            if name in values:
                return values[name]
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in MANAGED_NAMES and "_side_dispatch" in self.__dict__:
            self.__dict__["_side_dispatch"][name] = value

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if self.cooperative and "_side_dispatch" in self.__dict__:
            self.__dict__["_side_dispatch"].pop(name, None)


def _protocol_model(base: type, *, cooperative: bool) -> Any:
    model_type = type(
        "ProtocolBoundaryModel",
        (_ProtocolMixin, base),
        {"cooperative": cooperative},
    )
    model = model_type()
    model._side_dispatch = {}
    return model


def run_runtime_transaction_acceptance() -> dict[str, Any]:
    namespace, fixture_class = _synthetic_namespace()
    checks: dict[str, bool] = {}

    for label, tokens in (("single", [3]), ("sequence", [3, 5, 8])):
        model = fixture_class()
        runtime = _runtime(model, namespace)
        runtime.forward(tokens, _state(), len(tokens) > 1, coupling=_off_request())
        checks[f"standard_{label}_restored"] = _restored(model) and runtime.execution_count == 1

    cooperative = _protocol_model(fixture_class, cooperative=True)
    cooperative_runtime = _runtime(cooperative, namespace)
    cooperative_runtime.forward([3], _state(), False, coupling=_off_request())
    checks["cooperative_side_dispatch_restored"] = _restored(cooperative)

    sticky = _protocol_model(fixture_class, cooperative=False)
    sticky_runtime = _runtime(sticky, namespace)
    try:
        sticky_runtime.forward([3], _state(), False, coupling=_off_request())
    except D5CCleanupTransactionError as error:
        checks["sticky_side_dispatch_fails_closed"] = (
            error.output_was_produced and bool(error.verification_failures)
        )
    else:
        checks["sticky_side_dispatch_fails_closed"] = False

    class InstallFailure(fixture_class):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "forward_one":
                raise RuntimeError("synthetic install failure")
            super().__setattr__(name, value)

    install_failure = InstallFailure()
    try:
        _runtime(install_failure, namespace).forward(
            [3], _state(), False, coupling=_off_request()
        )
    except RuntimeError as error:
        checks["partial_install_restored"] = (
            str(error) == "synthetic install failure" and _restored(install_failure)
        )
    else:
        checks["partial_install_restored"] = False

    class ForwardFailure(fixture_class):
        def forward(self, *_: Any, **__: Any) -> Any:
            raise ValueError("synthetic forward failure")

    forward_failure = ForwardFailure()
    try:
        _runtime(forward_failure, namespace).forward(
            [3], _state(), False, coupling=_off_request()
        )
    except ValueError as error:
        checks["forward_exception_preserved_and_restored"] = (
            str(error) == "synthetic forward failure" and _restored(forward_failure)
        )
    else:
        checks["forward_exception_preserved_and_restored"] = False

    class CleanupFailure(fixture_class):
        cleanup_attempts: list[str]

        def __init__(self) -> None:
            self.cleanup_attempts = []

        def __delattr__(self, name: str) -> None:
            if name in MANAGED_NAMES:
                self.cleanup_attempts.append(name)
            if name == "forward_one":
                raise RuntimeError("synthetic cleanup failure")
            super().__delattr__(name)

    cleanup_failure = CleanupFailure()
    try:
        _runtime(cleanup_failure, namespace).forward(
            [3], _state(), False, coupling=_off_request()
        )
    except D5CCleanupTransactionError as error:
        checks["cleanup_attempts_all_and_discards_output"] = (
            set(cleanup_failure.cleanup_attempts) == set(MANAGED_NAMES)
            and error.output_was_produced
            and bool(error.cleanup_failures)
        )
    else:
        checks["cleanup_attempts_all_and_discards_output"] = False

    class NestedModel(fixture_class):
        nested_action: Any = None

        def forward(self, idx: Sequence[int], state: Any, full_output: bool = False) -> Any:
            if self.nested_action is not None:
                self.nested_action()
            return super().forward(idx, state, full_output)

    nested_model = NestedModel()
    outer = _runtime(nested_model, namespace)
    inner = _runtime(nested_model, namespace)
    nested_model.nested_action = lambda: inner.forward(
        [3], _state(), False, coupling=_off_request()
    )
    try:
        outer.forward([3], _state(), False, coupling=_off_request())
    except RuntimeError as error:
        checks["nested_rejected_before_inner_forward"] = (
            "nested or concurrent" in str(error)
            and inner.execution_count == 0
            and _restored(nested_model)
        )
    else:
        checks["nested_rejected_before_inner_forward"] = False

    entered = threading.Event()
    release = threading.Event()

    class BlockingModel(fixture_class):
        def forward(self, idx: Sequence[int], state: Any, full_output: bool = False) -> Any:
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("synthetic concurrent fixture timed out")
            return super().forward(idx, state, full_output)

    blocking_model = BlockingModel()
    blocking_outer = _runtime(blocking_model, namespace)
    blocking_inner = _runtime(blocking_model, namespace)
    result: dict[str, Any] = {}

    def run_outer() -> None:
        try:
            result["output"] = blocking_outer.forward(
                [3], _state(), False, coupling=_off_request()
            )
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=run_outer)
    thread.start()
    entered_ok = entered.wait(timeout=5)
    try:
        try:
            blocking_inner.forward([3], _state(), False, coupling=_off_request())
        except RuntimeError as error:
            concurrent_rejected = "nested or concurrent" in str(error)
        else:
            concurrent_rejected = False
    finally:
        release.set()
        thread.join(timeout=5)
    checks["concurrent_rejected_before_inner_forward"] = (
        entered_ok
        and concurrent_rejected
        and not thread.is_alive()
        and "error" not in result
        and blocking_inner.execution_count == 0
        and _restored(blocking_model)
    )

    lifecycle_single = run_plain_python_lifecycle_fixture([3])
    lifecycle_sequence = run_plain_python_lifecycle_fixture([3, 5, 8])
    checks["active_lifecycle_both_paths_restore"] = (
        lifecycle_single["valid"] and lifecycle_sequence["valid"]
    )
    checks["verification_adds_no_forward"] = (
        blocking_outer.execution_count == 1
        and blocking_inner.execution_count == 0
    )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "model_forward_calls": {
            "standard": 2,
            "cooperative": 1,
            "sticky_before_discard": 1,
            "concurrent_outer": blocking_outer.execution_count,
        },
    }


def build_runtime_transaction_patch_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C runtime transaction patch config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    acceptance = run_runtime_transaction_acceptance()
    runtime_path = root / "src/psa/self_model/d5c_mechanism_runtime.py"
    runtime_source = runtime_path.read_text(encoding="utf-8")
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "historical_wrapper_digest_preserved": config["frozen_prerequisites"][
            "historical_d5c_wrapper_source_sha256"
        ] == HISTORICAL_WRAPPER_DIGEST,
        "runtime_digest_changed_from_failure": sha256_file(runtime_path)
        != HISTORICAL_WRAPPER_DIGEST,
        "transaction_helpers_present": all(
            marker in runtime_source
            for marker in (
                "_capture_binding_snapshot",
                "_restore_bindings",
                "_verify_restored_bindings",
                "D5CCleanupTransactionError",
            )
        ),
        "historical_direct_pop_absent": "instance_dict.pop" not in runtime_source,
        "actual_runtime_acceptance_valid": acceptance["valid"],
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C runtime transaction patch verification failed: " + ", ".join(failed))
    report = {
        "patch_version": PATCH_VERSION,
        "status": "d5c_runtime_transaction_patch_no_model_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "acceptance": acceptance,
        "decision": {
            "real_runtime_patch_implemented": True,
            "local_no_model_acceptance_passed": True,
            "real_2_9b_validation_run": False,
            "d5c_failure_conclusion_changed": False,
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
            "d5d_authorized": False,
            "d5e_authorized": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "real_self_projection_constructed": False,
            "self_updater_used": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
