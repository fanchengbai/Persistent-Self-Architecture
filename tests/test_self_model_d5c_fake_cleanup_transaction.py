from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from psa.self_model.d5c_decorator_object_protocol_fixture import (
    CALLBACK_ATTRIBUTE,
    MANAGED_NAMES,
    _build_standard_class,
)
from psa.self_model.d5c_fake_cleanup_transaction import (
    CLASSIFICATION,
    CONFIG_RELATIVE_PATH,
    CleanupTransactionError,
    CountingCallback,
    NestedTransactionError,
    SyntheticCleanupTransaction,
    SyntheticForwardFailure,
    SyntheticInstallFailure,
    _build_cleanup_error_class,
    _build_forward_error_class,
    _build_sticky_side_cache_class,
    _is_restored,
    build_fake_cleanup_report,
    run_acceptance_suite,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH


class D5CFakeCleanupTransactionTests(unittest.TestCase):
    def test_config_freezes_fake_only_authority(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(all(validate_config(payload).values()))
        self.assertEqual(payload["owner_confirmation_text"], "下一轮确认")
        self.assertTrue(payload["authority"]["fake_cleanup_transaction_implementation_authorized"])
        self.assertFalse(payload["authority"]["real_runtime_modification_authorized"])
        self.assertFalse(payload["authority"]["model_execution_authorized"])
        self.assertFalse(payload["authority"]["d5c_rerun_authorized"])

    def test_scope_or_contract_mutation_fails_closed(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        for section, field, value in (
            ("authority", "real_runtime_modification_authorized", True),
            ("authority", "model_execution_authorized", True),
            ("frozen_prerequisites", "d5c_status", "passed"),
            ("transaction_contract", "verification_uses_extra_forward", True),
        ):
            changed = copy.deepcopy(payload)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_config(changed)

    def test_standard_transaction_commits_only_after_restoration(self):
        model = _build_standard_class("non_caching_descriptor")()
        transaction = SyntheticCleanupTransaction(model)
        callback = CountingCallback()
        output = transaction.execute(
            execution_path="forward_one", payload="payload", callback=callback
        )
        self.assertEqual(output, ("active", "forward_one"))
        self.assertEqual(callback.count, 1)
        self.assertEqual(transaction.forward_call_count, 1)
        self.assertTrue(_is_restored(model))

    def test_sticky_cache_discards_produced_output_and_fails_closed(self):
        model = _build_sticky_side_cache_class()()
        transaction = SyntheticCleanupTransaction(model)
        with self.assertRaises(CleanupTransactionError) as caught:
            transaction.execute(
                execution_path="forward_seq", payload="payload", callback=CountingCallback()
            )
        self.assertTrue(caught.exception.output_was_produced)
        self.assertIn("resolved_function:forward_seq", caught.exception.verification_failures)
        self.assertIn("callback_resolution", caught.exception.verification_failures)
        self.assertEqual(transaction.forward_call_count, 1)

    def test_partial_install_failures_restore_without_forward(self):
        for failure_point in (CALLBACK_ATTRIBUTE, "forward_one"):
            model = _build_standard_class("plain")()
            transaction = SyntheticCleanupTransaction(model)
            with self.subTest(failure_point=failure_point), self.assertRaises(
                SyntheticInstallFailure
            ):
                transaction.execute(
                    execution_path="forward_one", payload="payload",
                    callback=CountingCallback(), fail_install_after=failure_point,
                )
            self.assertEqual(transaction.forward_call_count, 0)
            self.assertTrue(_is_restored(model))

    def test_forward_exception_is_preserved_after_restoration(self):
        model = _build_forward_error_class()()
        transaction = SyntheticCleanupTransaction(model)
        with self.assertRaises(SyntheticForwardFailure):
            transaction.execute(
                execution_path="forward_one", payload="payload", callback=CountingCallback()
            )
        self.assertTrue(_is_restored(model))
        self.assertEqual(transaction.forward_call_count, 1)

    def test_cleanup_exception_attempts_every_name_and_fails_closed(self):
        model = _build_cleanup_error_class()()
        transaction = SyntheticCleanupTransaction(model)
        with self.assertRaises(CleanupTransactionError) as caught:
            transaction.execute(
                execution_path="forward_seq", payload="payload", callback=CountingCallback()
            )
        self.assertTrue(caught.exception.output_was_produced)
        self.assertEqual(model.cleanup_attempts, ["forward_seq", "forward_one", CALLBACK_ATTRIBUTE])
        self.assertTrue(any(item.startswith("forward_one:RuntimeError") for item in caught.exception.cleanup_failures))

    def test_nested_transaction_is_rejected_without_inner_mutation(self):
        model = _build_standard_class("plain")()
        outer = SyntheticCleanupTransaction(model)
        inner = SyntheticCleanupTransaction(model)

        def nested() -> None:
            inner.execute(
                execution_path="forward_one", payload="nested", callback=CountingCallback()
            )

        with self.assertRaises(NestedTransactionError):
            outer.execute(
                execution_path="forward_one", payload="outer",
                callback=CountingCallback(nested_action=nested),
            )
        self.assertEqual(inner.forward_call_count, 0)
        self.assertTrue(_is_restored(model))
        self.assertFalse(any(name in model.__dict__ for name in MANAGED_NAMES))

    def test_concurrent_transaction_is_rejected_without_inner_mutation(self):
        model = _build_standard_class("plain")()
        outer = SyntheticCleanupTransaction(model)
        inner = SyntheticCleanupTransaction(model)
        entered = threading.Event()
        release = threading.Event()
        result = {}

        def blocking_callback(execution_path: str) -> None:
            entered.set()
            self.assertTrue(release.wait(timeout=5))

        def run_outer() -> None:
            try:
                result["output"] = outer.execute(
                    execution_path="forward_seq", payload="outer",
                    callback=blocking_callback,
                )
            except BaseException as error:
                result["error"] = error

        thread = threading.Thread(target=run_outer)
        thread.start()
        self.assertTrue(entered.wait(timeout=5))
        try:
            with self.assertRaises(NestedTransactionError):
                inner.execute(
                    execution_path="forward_one", payload="inner",
                    callback=CountingCallback(),
                )
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", result)
        self.assertEqual(result["output"], ("active", "forward_seq"))
        self.assertEqual(inner.forward_call_count, 0)
        self.assertTrue(_is_restored(model))

    def test_acceptance_report_covers_all_categories_without_real_patch(self):
        suite = run_acceptance_suite()
        self.assertTrue(suite["valid"])
        report = build_fake_cleanup_report(config_path=CONFIG, project_root=ROOT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["classification"], CLASSIFICATION)
        self.assertTrue(report["decision"]["fake_candidate_valid"])
        self.assertFalse(report["decision"]["real_patch_implemented"])
        self.assertFalse(report["decision"]["real_fix_proven"])
        self.assertFalse(report["safety"]["real_runtime_modified"])
        self.assertFalse(report["safety"]["model_executed"])
        self.assertFalse(report["safety"]["d5c_rerun"])

    def test_wrong_config_path_and_model_modules_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(PermissionError):
                build_fake_cleanup_report(config_path=path, project_root=ROOT)
        self.assertNotIn("rwkv.model", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
