from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json


DESIGN_VERSION = "0.1-d5c-offline-fix-design"
CONFIG_RELATIVE_PATH = "configs/development/self_model_v0_1_d5c_offline_fix_design.json"
CLASSIFICATION = (
    "transactional_snapshot_restore_verify_recommended_for_future_fake_first_"
    "implementation_no_real_fix_claim"
)
WRAPPER_DIGEST = "e1de359da6d2087721dfd433a3e6ad90c6439bb474325a768c2f1d07fb08b5b7"
TRUE_AUTHORITY_FIELDS = {
    "offline_fix_design_authorized",
    "existing_source_report_and_fixture_observation_authorized",
}
FALSE_AUTHORITY_FIELDS = {
    "fake_fix_implementation_authorized",
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
EXPECTED_STRATEGIES = [
    {
        "strategy": "direct_instance_dict_pop",
        "decision": "reject_as_unverified_cleanup",
        "reason": "bypasses object deletion protocol and verifies neither resolved identities nor side state",
    },
    {
        "strategy": "delattr_only",
        "decision": "insufficient_as_standalone_fix",
        "reason": "works in the cooperative synthetic protocol but cannot prove unknown caches are cleared",
    },
    {
        "strategy": "transactional_snapshot_restore_verify",
        "decision": "recommend_for_future_fake_first_implementation",
        "reason": "pairs protocol-aware restoration with explicit post-cleanup identity verification and fail-closed output handling",
    },
]
EXPECTED_CAPTURE = [
    "exact managed-name instance ownership and values",
    "static class descriptor identities for forward_one and forward_seq",
    "resolved bound-method identity tokens for forward_one and forward_seq",
    "callback resolved absence sentinel",
]
EXPECTED_INSTALL = [
    "install callback through setattr",
    "install forward_one through setattr",
    "install forward_seq through setattr",
]
EXPECTED_RESTORE = [
    "attempt every managed-name restoration in reverse installation order",
    "use delattr for names absent in the snapshot",
    "use setattr with the exact snapshotted value for names owned in the snapshot",
    "retain the primary forward exception while attaching cleanup failures",
]
EXPECTED_VERIFY = [
    "instance ownership and values equal the snapshot",
    "static class descriptor identities remain unchanged",
    "resolved forward method identity tokens equal the snapshot",
    "callback resolution equals the snapshot sentinel",
]
EXPECTED_ACCEPTANCE = [
    "both execution paths across plain identity-decorator and non-caching-descriptor standard objects",
    "cooperative side-dispatch object restores through protocol-aware cleanup",
    "noncooperative sticky side-dispatch object is detected and fails closed",
    "failure after callback installation restores the snapshot",
    "failure after first method installation restores the snapshot",
    "forward exception still restores and preserves the primary exception",
    "cleanup exception attempts remaining names and fails closed",
    "post-cleanup identity mismatch discards a successful forward output",
    "nested or concurrent use is rejected before mutation",
    "no extra real-model forward is used for cleanup verification",
]
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_offline_fix_design.md",
    "scripts/verify_self_model_v0_1_d5c_offline_fix_design.py",
    "src/psa/self_model/d5c_offline_fix_design.py",
    "src/psa/self_model/d5c_decorator_object_protocol_fixture.py",
    "src/psa/self_model/d5c_mechanism_runtime.py",
    "tests/test_self_model_d5c_offline_fix_design.py",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C offline fix design config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("frozen_prerequisites")
    transaction = config.get("recommended_transaction")
    authority = config.get("authority")
    if not all(isinstance(item, Mapping) for item in (prerequisites, transaction, authority)):
        raise ValueError("D5C offline fix design config is incomplete")
    checks = {
        "identity_exact": config.get("design_version") == DESIGN_VERSION
        and config.get("stage") == "Coupling-D5C_failure_offline_fix_design"
        and config.get("status") == "design_only_authorized_no_implementation_no_model"
        and config.get("development_only") is True,
        "confirmation_is_context_bound": config.get("owner_confirmation_text") == "确认"
        and config.get("confirmation_context")
        == (
            "The immediately preceding assistant message offered exactly one next gate: "
            "D5C failure pure-offline fix design, with no runtime modification, model "
            "execution, or rerun authorization."
        ),
        "failed_prerequisites_preserved": prerequisites
        == {
            "d5c_real_report_sha256": "187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21",
            "source_audit_report_sha256": "652b1a4cc0bcf3f8c5b03f304133cc151f3af07160d8f8ecdddad9afb32d1342",
            "boundary_fixture_report_sha256": "c2c9b98bcd213af6cae15fe9f8b4ba51448b327956e54b9552943430474c60fc",
            "d5c_wrapper_source_sha256": WRAPPER_DIGEST,
            "d5c_status": "d5c_mechanism_smoke_failed",
            "decision_effect": "stop_without_rerun",
        },
        "strategy_review_exact": config.get("strategy_review") == EXPECTED_STRATEGIES,
        "transaction_precondition_exact": transaction.get("precondition")
        == "managed instance names remain absent exactly as required by the current wrapper",
        "transaction_capture_exact": transaction.get("capture") == EXPECTED_CAPTURE,
        "transaction_install_exact": transaction.get("install") == EXPECTED_INSTALL,
        "transaction_restore_exact": transaction.get("restore") == EXPECTED_RESTORE,
        "transaction_verify_exact": transaction.get("verify") == EXPECTED_VERIFY,
        "transaction_commit_and_failure_exact": transaction.get("commit_rule")
        == "return a forward output only after restoration verification succeeds"
        and transaction.get("failure_rule")
        == "discard the forward output and raise a cleanup verification error on any mismatch",
        "acceptance_matrix_exact": config.get("fake_acceptance_matrix")
        == EXPECTED_ACCEPTANCE,
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "authority_exact": set(authority) == TRUE_AUTHORITY_FIELDS | FALSE_AUTHORITY_FIELDS
        and all(authority.get(name) is True for name in TRUE_AUTHORITY_FIELDS)
        and all(authority.get(name) is False for name in FALSE_AUTHORITY_FIELDS),
        "next_gate_exact": config.get("next_gate")
        == "fake_first_cleanup_transaction_implementation_requires_separate_owner_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C offline fix design failed closed: " + ", ".join(failed))
    return checks


def _function(tree: ast.AST, class_name: str, function_name: str) -> ast.FunctionDef:
    classes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise RuntimeError(f"expected one {class_name} class")
    functions = [
        node for node in ast.walk(classes[0])
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(functions) != 1:
        raise RuntimeError(f"expected one {function_name} method")
    return functions[0]


def inspect_current_wrapper(root: Path) -> dict[str, Any]:
    path = root / "src/psa/self_model/d5c_mechanism_runtime.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forward = _function(tree, "RWKV7D5CActiveRuntime", "forward")
    call_names = [
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    ]
    return {
        "source_sha256": sha256_file(path),
        "setattr_call_count": call_names.count("setattr"),
        "dict_pop_call_count": call_names.count("pop"),
        "delattr_call_count": call_names.count("delattr"),
        "getattr_call_count": call_names.count("getattr"),
        "has_try_finally": any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(forward)),
        "has_snapshot_helper": "snapshot" in source.lower(),
        "has_post_cleanup_identity_verification": "resolved_method_identity" in source,
        "runtime_unchanged_from_frozen_failure": sha256_file(path) == WRAPPER_DIGEST,
    }


def build_fix_design_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C offline fix design config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    wrapper = inspect_current_wrapper(root)
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "real_runtime_digest_unchanged": wrapper["runtime_unchanged_from_frozen_failure"],
        "current_wrapper_still_installs_with_setattr": wrapper["setattr_call_count"] == 2,
        "current_wrapper_still_cleans_with_direct_pop": wrapper["dict_pop_call_count"] == 1,
        "current_wrapper_has_no_delattr_fix": wrapper["delattr_call_count"] == 0,
        "current_wrapper_has_no_snapshot_fix": wrapper["has_snapshot_helper"] is False,
        "current_wrapper_has_no_identity_verification_fix": wrapper[
            "has_post_cleanup_identity_verification"
        ] is False,
        "direct_pop_rejected_as_unverified": config["strategy_review"][0]["decision"]
        == "reject_as_unverified_cleanup",
        "delattr_only_not_overclaimed": config["strategy_review"][1]["decision"]
        == "insufficient_as_standalone_fix",
        "transactional_design_is_fake_first_only": config["strategy_review"][2]["decision"]
        == "recommend_for_future_fake_first_implementation",
        "capture_install_restore_verify_all_frozen": all(
            config["recommended_transaction"].get(name)
            for name in ("capture", "install", "restore", "verify")
        ),
        "output_commit_is_after_verification": config["recommended_transaction"][
            "commit_rule"
        ].startswith("return a forward output only after"),
        "verification_failure_discards_output": config["recommended_transaction"][
            "failure_rule"
        ].startswith("discard the forward output"),
        "noncooperative_cache_case_required": any(
            item.startswith("noncooperative sticky side-dispatch")
            for item in config["fake_acceptance_matrix"]
        ),
        "exception_paths_required": sum(
            "failure" in item or "exception" in item
            for item in config["fake_acceptance_matrix"]
        ) >= 4,
        "no_extra_real_forward_required": config["fake_acceptance_matrix"][-1]
        == "no extra real-model forward is used for cleanup verification",
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C offline fix design verification failed: " + ", ".join(failed))
    report = {
        "design_version": DESIGN_VERSION,
        "status": "d5c_offline_fix_design_complete",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "current_wrapper_observation": wrapper,
        "strategy_review": config["strategy_review"],
        "recommended_transaction": config["recommended_transaction"],
        "fake_acceptance_matrix": config["fake_acceptance_matrix"],
        "decision": {
            "recommended_for_future_fake_implementation": "transactional_snapshot_restore_verify",
            "real_fix_selected": False,
            "real_runtime_change_authorized": False,
            "model_validation_authorized": False,
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "fake_fix_implemented": False,
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
