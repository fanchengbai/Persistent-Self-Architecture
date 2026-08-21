from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sys
import threading
import types
from typing import Any, Callable, Mapping
import weakref

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d5c_decorator_object_protocol_fixture import (
    CALLBACK_ATTRIBUTE,
    DECORATOR_KINDS,
    EXECUTION_PATHS,
    MANAGED_NAMES,
    NonCachingDescriptor,
    _active_function,
    _build_side_cache_class,
    _build_standard_class,
)


IMPLEMENTATION_VERSION = "0.1-d5c-fake-cleanup-transaction"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_fake_cleanup_transaction.json"
)
CLASSIFICATION = (
    "fake_transaction_restores_cooperative_boundaries_and_fails_closed_on_"
    "sticky_or_exceptional_boundaries_no_real_fix_claim"
)
WRAPPER_DIGEST = "e1de359da6d2087721dfd433a3e6ad90c6439bb474325a768c2f1d07fb08b5b7"
INSTALL_ORDER = (CALLBACK_ATTRIBUTE, "forward_one", "forward_seq")
RESTORE_ORDER = tuple(reversed(INSTALL_ORDER))
ACCEPTANCE_CATEGORIES = (
    "standard_decorator_matrix_both_paths",
    "cooperative_side_dispatch_restores",
    "noncooperative_sticky_side_dispatch_fails_closed",
    "failure_after_callback_install_restores",
    "failure_after_first_method_install_restores",
    "forward_exception_restores_and_is_preserved",
    "cleanup_exception_attempts_remaining_names_and_fails_closed",
    "post_cleanup_identity_mismatch_discards_output",
    "nested_or_concurrent_use_rejected_before_inner_mutation",
    "verification_adds_no_forward_call",
)
TRUE_AUTHORITY_FIELDS = {
    "fake_cleanup_transaction_implementation_authorized",
    "synthetic_acceptance_execution_authorized",
}
FALSE_AUTHORITY_FIELDS = {
    "real_runtime_modification_authorized",
    "rwkv_import_authorized",
    "torch_import_authorized",
    "weights_access_authorized",
    "model_load_authorized",
    "model_execution_authorized",
    "d5c_rerun_authorized",
    "d5d_authorized",
    "d5e_authorized",
    "formal_test_set_authorized",
    "self_effect_conclusion_authorized",
    "real_self_projection_authorized",
    "self_updater_authorized",
    "automatic_rerun_authorized",
}
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_fake_cleanup_transaction.md",
    "scripts/verify_self_model_v0_1_d5c_fake_cleanup_transaction.py",
    "src/psa/self_model/d5c_decorator_object_protocol_fixture.py",
    "src/psa/self_model/d5c_fake_cleanup_transaction.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "src/psa/self_model/d5c_offline_fix_design.py",
    "tests/test_self_model_d5c_fake_cleanup_transaction.py",
)
_ABSENT = object()
_ACTIVE_MODELS: weakref.WeakSet[Any] = weakref.WeakSet()
_ACTIVE_MODELS_LOCK = threading.RLock()


class SyntheticInstallFailure(RuntimeError):
    pass


class SyntheticForwardFailure(RuntimeError):
    pass


class NestedTransactionError(RuntimeError):
    pass


class CleanupTransactionError(RuntimeError):
    def __init__(
        self,
        *,
        primary_exception: BaseException | None,
        cleanup_failures: list[str],
        verification_failures: list[str],
        output_was_produced: bool,
    ) -> None:
        super().__init__("synthetic cleanup transaction failed closed")
        self.primary_exception = primary_exception
        self.cleanup_failures = tuple(cleanup_failures)
        self.verification_failures = tuple(verification_failures)
        self.output_was_produced = output_was_produced


@dataclass(frozen=True)
class BindingSnapshot:
    instance_values: Mapping[str, Any]
    static_descriptors: Mapping[str, Any]
    resolved_functions: Mapping[str, Any]
    callback_value: Any


class CountingCallback:
    def __init__(self, nested_action: Callable[[], Any] | None = None) -> None:
        self.count = 0
        self._nested_action = nested_action

    def __call__(self, execution_path: str) -> None:
        if execution_path not in EXECUTION_PATHS:
            raise ValueError("invalid synthetic callback execution path")
        self.count += 1
        if self._nested_action is not None:
            self._nested_action()


class SyntheticCleanupTransaction:
    """Fake-only transaction; it is deliberately not connected to the real runtime."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.forward_call_count = 0
        self.restore_attempts: list[str] = []

    @staticmethod
    def _resolved_function(value: Any) -> Any:
        return getattr(value, "__func__", value)

    def _snapshot(self) -> BindingSnapshot:
        instance_dict = getattr(self.model, "__dict__", None)
        if not isinstance(instance_dict, dict):
            raise TypeError("synthetic transaction requires an instance dictionary")
        if any(name in instance_dict for name in MANAGED_NAMES):
            raise RuntimeError("synthetic transaction rejects pre-existing managed names")
        return BindingSnapshot(
            instance_values={name: instance_dict.get(name, _ABSENT) for name in MANAGED_NAMES},
            static_descriptors={
                name: inspect.getattr_static(self.model, name) for name in EXECUTION_PATHS
            },
            resolved_functions={
                name: self._resolved_function(getattr(self.model, name))
                for name in EXECUTION_PATHS
            },
            callback_value=getattr(self.model, CALLBACK_ATTRIBUTE, _ABSENT),
        )

    def _install(
        self,
        callback: Callable[[str], None],
        fail_install_after: str | None,
    ) -> None:
        if fail_install_after not in {None, CALLBACK_ATTRIBUTE, "forward_one"}:
            raise ValueError("unsupported synthetic install failure point")
        setattr(self.model, CALLBACK_ATTRIBUTE, callback)
        if fail_install_after == CALLBACK_ATTRIBUTE:
            raise SyntheticInstallFailure("failure after callback installation")
        setattr(
            self.model,
            "forward_one",
            types.MethodType(_active_function("forward_one"), self.model),
        )
        if fail_install_after == "forward_one":
            raise SyntheticInstallFailure("failure after first method installation")
        setattr(
            self.model,
            "forward_seq",
            types.MethodType(_active_function("forward_seq"), self.model),
        )

    def _restore(self, snapshot: BindingSnapshot) -> list[str]:
        failures: list[str] = []
        instance_dict = self.model.__dict__
        for name in RESTORE_ORDER:
            self.restore_attempts.append(name)
            try:
                previous = snapshot.instance_values[name]
                if previous is _ABSENT:
                    if name in instance_dict:
                        delattr(self.model, name)
                else:
                    setattr(self.model, name, previous)
            except BaseException as error:
                failures.append(f"{name}:{type(error).__name__}:{error}")
        return failures

    def _verify(self, snapshot: BindingSnapshot) -> list[str]:
        failures: list[str] = []
        instance_dict = self.model.__dict__
        for name in MANAGED_NAMES:
            before = snapshot.instance_values[name]
            after = instance_dict.get(name, _ABSENT)
            if before is _ABSENT:
                if after is not _ABSENT:
                    failures.append(f"instance_ownership:{name}")
            elif after is not before:
                failures.append(f"instance_value:{name}")
        for name in EXECUTION_PATHS:
            if inspect.getattr_static(self.model, name) is not snapshot.static_descriptors[name]:
                failures.append(f"static_descriptor:{name}")
            resolved = self._resolved_function(getattr(self.model, name))
            if resolved is not snapshot.resolved_functions[name]:
                failures.append(f"resolved_function:{name}")
        callback = getattr(self.model, CALLBACK_ATTRIBUTE, _ABSENT)
        if callback is not snapshot.callback_value:
            failures.append("callback_resolution")
        return failures

    def execute(
        self,
        *,
        execution_path: str,
        payload: Any,
        callback: Callable[[str], None],
        fail_install_after: str | None = None,
    ) -> Any:
        if execution_path not in EXECUTION_PATHS:
            raise ValueError("unsupported synthetic execution path")
        with _ACTIVE_MODELS_LOCK:
            if self.model in _ACTIVE_MODELS:
                raise NestedTransactionError(
                    "nested or concurrent synthetic cleanup transaction rejected"
                )
            snapshot = self._snapshot()
            _ACTIVE_MODELS.add(self.model)
        primary: BaseException | None = None
        output: Any = _ABSENT
        try:
            try:
                self._install(callback, fail_install_after)
                self.forward_call_count += 1
                output = self.model.forward(execution_path, payload)
            except BaseException as error:
                primary = error
            cleanup_failures = self._restore(snapshot)
            verification_failures = self._verify(snapshot)
            if cleanup_failures or verification_failures:
                failure = CleanupTransactionError(
                    primary_exception=primary,
                    cleanup_failures=cleanup_failures,
                    verification_failures=verification_failures,
                    output_was_produced=output is not _ABSENT,
                )
                if primary is not None:
                    raise failure from primary
                raise failure
            if primary is not None:
                raise primary
            if output is _ABSENT:
                raise AssertionError("synthetic transaction produced no output")
            return output
        finally:
            with _ACTIVE_MODELS_LOCK:
                _ACTIVE_MODELS.discard(self.model)


def _build_sticky_side_cache_class() -> type:
    base = _build_standard_class("non_caching_descriptor")

    def __init__(self: Any) -> None:
        object.__setattr__(self, "_side_dispatch", {})
        object.__setattr__(self, "_protocol_events", [])

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if name in MANAGED_NAMES:
            self._protocol_events.append({"operation": "setattr", "name": name})
            self._side_dispatch[name] = value
        object.__setattr__(self, name, value)

    def __getattribute__(self: Any, name: str) -> Any:
        if name in MANAGED_NAMES:
            side = object.__getattribute__(self, "_side_dispatch")
            if name in side:
                object.__getattribute__(self, "_protocol_events").append(
                    {"operation": "sticky_get", "name": name}
                )
                return side[name]
        return object.__getattribute__(self, name)

    def __delattr__(self: Any, name: str) -> None:
        if name in MANAGED_NAMES:
            self._protocol_events.append({"operation": "sticky_delattr", "name": name})
        object.__delattr__(self, name)

    return type(
        "SyntheticStickySideCache",
        (base,),
        {
            "__init__": __init__,
            "__setattr__": __setattr__,
            "__getattribute__": __getattribute__,
            "__delattr__": __delattr__,
        },
    )


def _build_cleanup_error_class() -> type:
    base = _build_standard_class("non_caching_descriptor")

    def __init__(self: Any) -> None:
        object.__setattr__(self, "cleanup_attempts", [])

    def __delattr__(self: Any, name: str) -> None:
        if name in MANAGED_NAMES:
            self.cleanup_attempts.append(name)
        if name == "forward_one":
            raise RuntimeError("synthetic forward_one cleanup failure")
        object.__delattr__(self, name)

    return type(
        "SyntheticCleanupError",
        (base,),
        {"__init__": __init__, "__delattr__": __delattr__},
    )


def _build_forward_error_class() -> type:
    base = _build_standard_class("non_caching_descriptor")

    def forward(self: Any, execution_path: str, payload: Any) -> Any:
        raise SyntheticForwardFailure("synthetic forward failure")

    return type("SyntheticForwardError", (base,), {"forward": forward})


def _is_restored(model: Any) -> bool:
    if any(name in model.__dict__ for name in MANAGED_NAMES):
        return False
    return all(
        getattr(getattr(model, name), "__func__", None)._fixture_origin == "original"
        for name in EXECUTION_PATHS
    ) and getattr(model, CALLBACK_ATTRIBUTE, _ABSENT) is _ABSENT


def run_acceptance_suite() -> dict[str, Any]:
    standard_records = []
    for decorator_kind in DECORATOR_KINDS:
        for execution_path in EXECUTION_PATHS:
            model = _build_standard_class(decorator_kind)()
            transaction = SyntheticCleanupTransaction(model)
            callback = CountingCallback()
            output = transaction.execute(
                execution_path=execution_path, payload="payload", callback=callback
            )
            standard_records.append({
                "decorator_kind": decorator_kind,
                "execution_path": execution_path,
                "output": list(output),
                "callback_count": callback.count,
                "forward_call_count": transaction.forward_call_count,
                "restored": _is_restored(model),
            })

    cooperative_records = []
    for execution_path in EXECUTION_PATHS:
        model = _build_side_cache_class()()
        transaction = SyntheticCleanupTransaction(model)
        callback = CountingCallback()
        output = transaction.execute(
            execution_path=execution_path, payload="payload", callback=callback
        )
        cooperative_records.append({
            "execution_path": execution_path,
            "output": list(output),
            "callback_count": callback.count,
            "forward_call_count": transaction.forward_call_count,
            "side_keys": sorted(model._side_dispatch),
            "restored": _is_restored(model),
        })

    sticky_records = []
    for execution_path in EXECUTION_PATHS:
        model = _build_sticky_side_cache_class()()
        transaction = SyntheticCleanupTransaction(model)
        callback = CountingCallback()
        try:
            transaction.execute(
                execution_path=execution_path, payload="payload", callback=callback
            )
        except CleanupTransactionError as error:
            sticky_records.append({
                "execution_path": execution_path,
                "failed_closed": True,
                "output_was_produced": error.output_was_produced,
                "verification_failures": list(error.verification_failures),
                "instance_keys": sorted(name for name in MANAGED_NAMES if name in model.__dict__),
                "side_keys": sorted(model._side_dispatch),
                "forward_call_count": transaction.forward_call_count,
            })
        else:
            raise AssertionError("sticky side cache must fail closed")

    partial_records = []
    for failure_point in (CALLBACK_ATTRIBUTE, "forward_one"):
        model = _build_standard_class("non_caching_descriptor")()
        transaction = SyntheticCleanupTransaction(model)
        try:
            transaction.execute(
                execution_path="forward_one", payload="payload",
                callback=CountingCallback(), fail_install_after=failure_point,
            )
        except SyntheticInstallFailure:
            partial_records.append({
                "failure_point": failure_point,
                "restored": _is_restored(model),
                "forward_call_count": transaction.forward_call_count,
                "restore_attempts": list(transaction.restore_attempts),
            })
        else:
            raise AssertionError("partial installation must raise")

    forward_model = _build_forward_error_class()()
    forward_transaction = SyntheticCleanupTransaction(forward_model)
    try:
        forward_transaction.execute(
            execution_path="forward_one", payload="payload", callback=CountingCallback()
        )
    except SyntheticForwardFailure:
        forward_exception = {
            "primary_preserved": True,
            "restored": _is_restored(forward_model),
            "forward_call_count": forward_transaction.forward_call_count,
        }
    else:
        raise AssertionError("synthetic forward failure must be preserved")

    cleanup_model = _build_cleanup_error_class()()
    cleanup_transaction = SyntheticCleanupTransaction(cleanup_model)
    try:
        cleanup_transaction.execute(
            execution_path="forward_seq", payload="payload", callback=CountingCallback()
        )
    except CleanupTransactionError as error:
        cleanup_exception = {
            "failed_closed": True,
            "output_was_produced": error.output_was_produced,
            "cleanup_failures": list(error.cleanup_failures),
            "verification_failures": list(error.verification_failures),
            "attempts": list(cleanup_model.cleanup_attempts),
        }
    else:
        raise AssertionError("cleanup failure must fail closed")

    nested_model = _build_standard_class("plain")()
    outer = SyntheticCleanupTransaction(nested_model)
    inner = SyntheticCleanupTransaction(nested_model)
    keys_seen_by_inner: list[str] = []

    def nested_action() -> None:
        keys_seen_by_inner.extend(sorted(name for name in MANAGED_NAMES if name in nested_model.__dict__))
        inner.execute(
            execution_path="forward_one", payload="nested", callback=CountingCallback()
        )

    try:
        outer.execute(
            execution_path="forward_one", payload="outer",
            callback=CountingCallback(nested_action=nested_action),
        )
    except NestedTransactionError:
        nested = {
            "rejected": True,
            "inner_forward_call_count": inner.forward_call_count,
            "outer_restored": _is_restored(nested_model),
            "managed_keys_at_inner_entry": keys_seen_by_inner,
        }
    else:
        raise AssertionError("nested transaction must be rejected")

    concurrent_model = _build_standard_class("plain")()
    concurrent_outer = SyntheticCleanupTransaction(concurrent_model)
    concurrent_inner = SyntheticCleanupTransaction(concurrent_model)
    callback_entered = threading.Event()
    callback_release = threading.Event()
    outer_result: dict[str, Any] = {}

    def blocking_callback(execution_path: str) -> None:
        callback_entered.set()
        if not callback_release.wait(timeout=5):
            raise RuntimeError("synthetic concurrent callback timed out")

    def run_outer() -> None:
        try:
            outer_result["output"] = concurrent_outer.execute(
                execution_path="forward_seq", payload="outer",
                callback=blocking_callback,
            )
        except BaseException as error:
            outer_result["error"] = error

    thread = threading.Thread(target=run_outer, name="d5c-fake-transaction-outer")
    thread.start()
    if not callback_entered.wait(timeout=5):
        callback_release.set()
        thread.join(timeout=5)
        raise RuntimeError("synthetic concurrent outer transaction did not enter callback")
    try:
        concurrent_inner.execute(
            execution_path="forward_one", payload="inner", callback=CountingCallback()
        )
    except NestedTransactionError:
        concurrent_rejected = True
    else:
        concurrent_rejected = False
    finally:
        callback_release.set()
        thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("synthetic concurrent outer transaction did not finish")
    concurrent = {
        "rejected": concurrent_rejected,
        "inner_forward_call_count": concurrent_inner.forward_call_count,
        "outer_output": list(outer_result["output"])
        if "output" in outer_result else None,
        "outer_error": type(outer_result["error"]).__name__
        if "error" in outer_result else None,
        "outer_restored": _is_restored(concurrent_model),
    }

    checks = {
        "standard_matrix_six_cases_restore": len(standard_records) == 6
        and all(record["restored"] for record in standard_records),
        "standard_outputs_active_and_single_forward": all(
            record["output"][0] == "active"
            and record["callback_count"] == 1
            and record["forward_call_count"] == 1
            for record in standard_records
        ),
        "cooperative_both_paths_restore": len(cooperative_records) == 2
        and all(record["restored"] and record["side_keys"] == [] for record in cooperative_records),
        "sticky_both_paths_fail_closed": len(sticky_records) == 2
        and all(record["failed_closed"] for record in sticky_records),
        "sticky_outputs_discarded_after_one_forward": all(
            record["output_was_produced"] and record["forward_call_count"] == 1
            for record in sticky_records
        ),
        "partial_install_failures_restore_without_forward": len(partial_records) == 2
        and all(record["restored"] and record["forward_call_count"] == 0 for record in partial_records),
        "forward_exception_primary_preserved_and_restored": forward_exception
        == {"primary_preserved": True, "restored": True, "forward_call_count": 1},
        "cleanup_exception_attempts_all_names": cleanup_exception["attempts"]
        == list(RESTORE_ORDER),
        "cleanup_exception_discards_output": cleanup_exception["failed_closed"]
        and cleanup_exception["output_was_produced"],
        "nested_rejected_without_inner_forward": nested["rejected"]
        and nested["inner_forward_call_count"] == 0
        and nested["outer_restored"],
        "concurrent_rejected_without_inner_forward": concurrent["rejected"]
        and concurrent["inner_forward_call_count"] == 0
        and concurrent["outer_error"] is None
        and concurrent["outer_output"] == ["active", "forward_seq"]
        and concurrent["outer_restored"],
        "verification_adds_no_forward": all(
            record["forward_call_count"] == 1
            for record in standard_records + cooperative_records + sticky_records
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "standard": standard_records,
        "cooperative_side_dispatch": cooperative_records,
        "sticky_side_dispatch": sticky_records,
        "partial_installation": partial_records,
        "forward_exception": forward_exception,
        "cleanup_exception": cleanup_exception,
        "nested": nested,
        "concurrent": concurrent,
    }


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C fake cleanup transaction config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("frozen_prerequisites")
    contract = config.get("transaction_contract")
    authority = config.get("authority")
    if not all(isinstance(item, Mapping) for item in (prerequisites, contract, authority)):
        raise ValueError("D5C fake cleanup transaction config is incomplete")
    checks = {
        "identity_exact": config.get("implementation_version") == IMPLEMENTATION_VERSION
        and config.get("stage") == "Coupling-D5C_failure_fake_first_cleanup_transaction"
        and config.get("status") == "fake_only_implementation_authorized_no_real_patch_no_model"
        and config.get("development_only") is True,
        "confirmation_context_exact": config.get("owner_confirmation_text") == "下一轮确认"
        and config.get("confirmation_context")
        == (
            "The immediately preceding assistant message offered exactly one next gate: "
            "implement the fake-first cleanup transaction and ten synthetic acceptance "
            "categories without modifying the real runtime or executing a model."
        ),
        "failed_prerequisites_preserved": prerequisites
        == {
            "d5c_real_report_sha256": "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21",
            "boundary_fixture_report_sha256": "c2c9b98bcd213af6cae15fe9f8b4ba51448b327956e54b9552943430474c60fc",
            "fix_design_report_sha256": "a37ac870e7efd7992b55c74cbcea43a195ad451d8d040ea7ba868f8cd54a67b6",
            "d5c_wrapper_source_sha256": WRAPPER_DIGEST,
            "d5c_status": "d5c_mechanism_smoke_failed",
            "decision_effect": "stop_without_rerun",
        },
        "managed_names_exact": contract.get("managed_names") == list(INSTALL_ORDER),
        "install_and_restore_order_exact": contract.get("install_order")
        == list(INSTALL_ORDER) and contract.get("restore_order") == list(RESTORE_ORDER),
        "phase_order_exact": contract.get("phases")
        == ["snapshot", "install", "forward", "restore", "verify", "commit_or_fail_closed"],
        "commit_and_failure_rules_exact": contract.get("commit_rule")
        == "output is returned only after restoration verification succeeds"
        and contract.get("failure_rule")
        == "produced output is discarded when cleanup or verification fails",
        "precondition_exact": contract.get("precondition_rule")
        == "nested use and pre-existing managed instance names are rejected before managed-name mutation",
        "no_extra_forward_exact": contract.get("verification_uses_extra_forward") is False,
        "acceptance_categories_exact": config.get("acceptance_categories")
        == list(ACCEPTANCE_CATEGORIES),
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "authority_exact": set(authority) == TRUE_AUTHORITY_FIELDS | FALSE_AUTHORITY_FIELDS
        and all(authority.get(name) is True for name in TRUE_AUTHORITY_FIELDS)
        and all(authority.get(name) is False for name in FALSE_AUTHORITY_FIELDS),
        "next_gate_exact": config.get("next_gate")
        == "real_runtime_patch_implementation_requires_separate_owner_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C fake cleanup config failed closed: " + ", ".join(failed))
    return checks


def build_fake_cleanup_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C fake cleanup config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    acceptance = run_acceptance_suite()
    wrapper_path = root / "src/psa/self_model/d5c_mechanism_runtime.py"
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "acceptance_suite_valid": acceptance["valid"],
        "all_ten_acceptance_categories_covered": len(ACCEPTANCE_CATEGORIES) == 10,
        "standard_and_cooperative_boundaries_restore": acceptance["checks"][
            "standard_matrix_six_cases_restore"
        ] and acceptance["checks"]["cooperative_both_paths_restore"],
        "sticky_boundary_fails_closed": acceptance["checks"][
            "sticky_both_paths_fail_closed"
        ],
        "successful_output_is_discarded_on_verification_failure": acceptance["checks"][
            "sticky_outputs_discarded_after_one_forward"
        ],
        "partial_installation_restores": acceptance["checks"][
            "partial_install_failures_restore_without_forward"
        ],
        "primary_forward_exception_preserved": acceptance["checks"][
            "forward_exception_primary_preserved_and_restored"
        ],
        "cleanup_attempts_all_names": acceptance["checks"][
            "cleanup_exception_attempts_all_names"
        ],
        "nested_and_concurrent_use_rejected": acceptance["checks"][
            "nested_rejected_without_inner_forward"
        ] and acceptance["checks"]["concurrent_rejected_without_inner_forward"],
        "verification_uses_no_extra_forward": acceptance["checks"][
            "verification_adds_no_forward"
        ],
        "real_runtime_digest_unchanged": sha256_file(wrapper_path) == WRAPPER_DIGEST,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C fake cleanup transaction failed: " + ", ".join(failed))
    report = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "d5c_fake_cleanup_transaction_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "acceptance": acceptance,
        "decision": {
            "fake_candidate_valid": True,
            "real_patch_implemented": False,
            "real_fix_proven": False,
            "model_validation_authorized": False,
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "fake_cleanup_transaction_implemented": True,
            "real_runtime_modified": False,
            "existing_real_report_reexecuted": False,
            "d5c_rerun": False,
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "d5c_conclusion_changed": False,
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
