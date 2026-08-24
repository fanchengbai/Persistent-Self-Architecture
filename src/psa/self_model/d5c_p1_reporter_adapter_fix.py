from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.self_model import d5c_mechanism_runtime as runtime_module
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    OfflineTensor,
    OfflineTorch,
    _namespace,
    _state,
)
from psa.self_model.d5c_mechanism_runtime import (
    D5CSyntheticProbe,
    RWKV7D5CActiveRuntime,
)
from psa.self_model.d5c_p1_engineering_validation import (
    _tensor_payload,
    execute_d5c_p1_engineering_core,
)
from psa.self_model.d5c_p1_reporter_fix_design import (
    ACCEPTANCE_CATEGORIES,
    P1_CORE_DIGEST,
    run_synthetic_dispatch_diagnostic,
)


FIX_VERSION = "0.1-d5c-p1-reporter-adapter-fix"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d5c_p1_reporter_adapter_fix.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 D5C-P1 reporter显式adapter fake-first修复实现；"
    "只修改reporter分派接口与纯Python fixture调用，真实runner固定不传offline adapter，"
    "并执行冻结的9类合成验收；不导入RWKV/Torch、不访问权重、不加载或执行模型、"
    "不授权任何D5C/P1真实重跑，也不改变D5C/P1结论或开放D5D/D5E、正式测试集、"
    "Self效果、真实Self projection、Self Updater或自动重跑。"
)
DESIGN_REPORT_DIGEST = "5940b7eee3467c935442422b2a3de132a7a0981215de976bb01d67d29260bcb7"
FAILURE_REPORT_DIGEST = "930c31ef6f70c431066cda3637c97fcc35344b8caabaeb2f2f7147a0b5d54483"
CLAIM_DIGEST = "7c49107b33a223be7ac11f3412328abc07e24e5fc6bcf68accfbe34b7ca97628"
CLASSIFICATION = (
    "explicit_offline_adapter_reporter_fix_passes_nine_fake_acceptance_"
    "real_runner_no_adapter_no_rerun"
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "docs/self_model_v0_1_d5c_p1_reporter_adapter_fix.md",
    "scripts/verify_self_model_v0_1_d5c_p1_reporter_adapter_fix.py",
    "src/psa/self_model/d5c_p1_engineering_validation.py",
    "src/psa/self_model/d5c_p1_real_entry.py",
    "src/psa/self_model/d5c_p1_reporter_adapter_fix.py",
    "src/psa/self_model/d5c_p1_reporter_fix_design.py",
    "tests/test_self_model_d5c_p1_real_entry.py",
    "tests/test_self_model_d5c_p1_reporter_adapter_fix.py",
    "tests/test_self_model_d5c_p1_reporter_fix_design.py",
)


class ExactOfflineTensorAdapter:
    def accepts(self, value: Any) -> bool:
        return type(value) is OfflineTensor

    def payload(self, value: OfflineTensor) -> dict[str, Any]:
        return {
            "kind": "offline_tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": sha256_json(value.values),
        }


class _BytesArray:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _ByteView:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def numpy(self) -> _BytesArray:
        return _BytesArray(self._data)


class RealLikeTensor:
    shape = (3,)
    dtype = "float16"
    device = "synthetic-cuda:0"

    def __init__(self, data: bytes = b"d5c-p1-real-like") -> None:
        self._data = data
        self.values_member_reads = 0

    @property
    def values(self):
        self.values_member_reads += 1

        def callable_member() -> tuple[int, ...]:
            return (1, 2, 3)

        return callable_member

    def detach(self) -> "RealLikeTensor":
        return self

    def contiguous(self) -> "RealLikeTensor":
        return self

    def cpu(self) -> "RealLikeTensor":
        return self

    def view(self, dtype: Any) -> _ByteView:
        if dtype != _SyntheticTorch.uint8:
            raise TypeError("unexpected synthetic view dtype")
        return _ByteView(self._data)


class RealLikeTensorWithoutValues:
    shape = (3,)
    dtype = "float16"
    device = "synthetic-cuda:0"

    def __init__(self, data: bytes = b"d5c-p1-real-like") -> None:
        self._data = data

    def detach(self) -> "RealLikeTensorWithoutValues":
        return self

    def contiguous(self) -> "RealLikeTensorWithoutValues":
        return self

    def cpu(self) -> "RealLikeTensorWithoutValues":
        return self

    def view(self, dtype: Any) -> _ByteView:
        if dtype != _SyntheticTorch.uint8:
            raise TypeError("unexpected synthetic view dtype")
        return _ByteView(self._data)


class AccidentalDataValuesObject:
    shape = (1,)
    dtype = "float16"
    device = "synthetic-cuda:0"
    values = (9.0,)


class _SyntheticTorch:
    uint8 = "uint8"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D5C-P1 adapter fix config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    authority = config.get("authority", {})
    prerequisites = config.get("prerequisites", {})
    contract = config.get("dispatch_contract", {})
    closed = (
        "rwkv_import_authorized", "torch_import_authorized",
        "weights_access_authorized", "model_load_authorized",
        "model_execution_authorized", "historical_d5c_rerun_authorized",
        "p1_rerun_authorized", "historical_d5c_conclusion_change_authorized",
        "p1_conclusion_change_authorized", "d5d_authorized", "d5e_authorized",
        "formal_test_set_authorized", "self_effect_conclusion_authorized",
        "real_self_projection_authorized", "self_updater_authorized",
        "automatic_rerun_authorized",
    )
    checks = {
        "identity_exact": config.get("fix_version") == FIX_VERSION,
        "confirmation_exact": config.get("owner_confirmation_text") == REQUIRED_CONFIRMATION,
        "design_digest_frozen": prerequisites.get("design_report_sha256") == DESIGN_REPORT_DIGEST,
        "historical_source_digest_frozen": prerequisites.get(
            "historical_reporter_source_sha256"
        ) == P1_CORE_DIGEST,
        "failure_and_claim_frozen": prerequisites.get("failure_report_sha256")
        == FAILURE_REPORT_DIGEST and prerequisites.get("execution_claim_sha256") == CLAIM_DIGEST,
        "adapter_contract_exact": contract.get("production_default")
        == "real_tensor_protocol_without_values_name_dispatch"
        and contract.get("offline_path") == "explicit_adapter_with_exact_fixture_type"
        and contract.get("real_runner_offline_adapter") is None
        and contract.get("unknown_object_policy") == "fail_closed"
        and contract.get("adapter_methods") == ["accepts", "payload"],
        "nine_categories_exact": config.get("acceptance_categories")
        == list(ACCEPTANCE_CATEGORIES),
        "implementation_authorized": authority.get(
            "reporter_fix_implementation_authorized"
        ) is True and authority.get("synthetic_fixture_execution_authorized") is True,
        "real_runner_audit_authorized": authority.get(
            "real_runner_signature_audit_authorized"
        ) is True,
        "all_prohibited_authority_closed": all(authority.get(name) is False for name in closed),
        "classification_exact": config.get("required_classification") == CLASSIFICATION,
        "next_gate_exact": config.get("next_gate")
        == "remote_no_model_adapter_fix_verification_then_research_review_no_rerun",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D5C-P1 adapter fix config failed closed: " + ", ".join(failed))
    return checks


def _reporter_ast_audit(root: Path) -> dict[str, Any]:
    reporter_path = root / "src/psa/self_model/d5c_p1_engineering_validation.py"
    tree = ast.parse(reporter_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tensor_payload"
    )
    values_reads = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and node.attr == "values"
    ]
    adapter_arguments = [argument.arg for argument in function.args.kwonlyargs]

    runner_path = root / "src/psa/self_model/d5c_p1_real_entry.py"
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "execute_d5c_p1_engineering_core"
    ]
    runner_keywords = sorted(
        keyword.arg for call in calls for keyword in call.keywords if keyword.arg
    )
    return {
        "reporter_source_sha256": sha256_file(reporter_path),
        "historical_reporter_source_sha256": P1_CORE_DIGEST,
        "values_attribute_read_count": len(values_reads),
        "offline_adapter_keyword_only": "offline_adapter" in adapter_arguments,
        "real_runner_core_call_count": len(calls),
        "real_runner_keywords": runner_keywords,
        "real_runner_passes_offline_adapter": "offline_adapter" in runner_keywords,
    }


def _full_core_fixture_acceptance() -> dict[str, Any]:
    namespace, fixture_class = _namespace()
    fixture = fixture_class()
    source_bytes = DIAGNOSTIC_SOURCE.encode("utf-8")
    digest = hashlib.sha256(source_bytes).hexdigest()
    previous = runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256
    runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256 = digest
    try:
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
        runtime_module.EXPECTED_RWKV_MODEL_SOURCE_SHA256 = previous
    probe = D5CSyntheticProbe(
        torch=OfflineTorch(), execution_claim_sha256="a" * 64,
        machine_authorization_sha256="b" * 64,
    )
    return execute_d5c_p1_engineering_core(
        base_model=fixture, active_runtime=runtime, probe=probe,
        torch=OfflineTorch(), state_factory=_state,
        offline_adapter=ExactOfflineTensorAdapter(),
    )


def run_nine_category_acceptance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    reporter_path = root / "src/psa/self_model/d5c_p1_engineering_validation.py"
    before = sha256_file(reporter_path)
    historical = run_synthetic_dispatch_diagnostic()
    adapter = ExactOfflineTensorAdapter()
    real_like = RealLikeTensor()
    real_payload = _tensor_payload(real_like, _SyntheticTorch(), offline_adapter=adapter)
    offline = OfflineTensor((1.0, 2.0, 3.0))
    offline_payload = _tensor_payload(offline, _SyntheticTorch(), offline_adapter=adapter)
    try:
        _tensor_payload(
            AccidentalDataValuesObject(), _SyntheticTorch(), offline_adapter=adapter
        )
    except TypeError:
        unknown_failed_closed = True
    else:
        unknown_failed_closed = False
    without_values = _tensor_payload(RealLikeTensorWithoutValues(), _SyntheticTorch())
    ast_audit = _reporter_ast_audit(root)
    core = _full_core_fixture_acceptance()
    after = sha256_file(reporter_path)
    checks = {
        ACCEPTANCE_CATEGORIES[0]: historical["checks"]["callable_values_collision_reproduced"],
        ACCEPTANCE_CATEGORIES[1]: real_payload["kind"] == "tensor",
        ACCEPTANCE_CATEGORIES[2]: offline_payload["kind"] == "offline_tensor" and core["valid"],
        ACCEPTANCE_CATEGORIES[3]: unknown_failed_closed,
        ACCEPTANCE_CATEGORIES[4]: historical["checks"][
            "callability_guard_still_misclassifies_accidental_data"
        ],
        ACCEPTANCE_CATEGORIES[5]: historical["checks"]["marker_strategy_is_spoofable"],
        ACCEPTANCE_CATEGORIES[6]: real_like.values_member_reads == 0,
        ACCEPTANCE_CATEGORIES[7]: real_payload["sha256"] == without_values["sha256"]
        and ast_audit["values_attribute_read_count"] == 0,
        ACCEPTANCE_CATEGORIES[8]: before == after,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "reporter_ast": ast_audit,
        "full_core_fixture": {
            "valid": core["valid"],
            "status": core["status"],
            "counts": core["counts"],
        },
    }


def build_reporter_adapter_fix_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D5C-P1 adapter fix config path is not frozen")
    config = _object(config_file)
    config_checks = validate_config(config)
    acceptance = run_nine_category_acceptance(root)
    ast_audit = acceptance["reporter_ast"]
    source_digests = {path: sha256_file(root / path) for path in SOURCE_PATHS}
    checks = {
        "config_valid": all(config_checks.values()),
        "nine_category_acceptance_valid": acceptance["valid"],
        "all_nine_categories_present": len(acceptance["checks"]) == 9,
        "historical_source_digest_preserved": ast_audit[
            "historical_reporter_source_sha256"
        ] == P1_CORE_DIGEST,
        "reporter_source_changed_from_failure": ast_audit[
            "reporter_source_sha256"
        ] != P1_CORE_DIGEST,
        "values_name_dispatch_absent": ast_audit["values_attribute_read_count"] == 0,
        "explicit_adapter_is_keyword_only": ast_audit["offline_adapter_keyword_only"],
        "real_runner_has_single_core_call": ast_audit["real_runner_core_call_count"] == 1,
        "real_runner_passes_no_offline_adapter": not ast_audit[
            "real_runner_passes_offline_adapter"
        ],
        "full_32_layer_fixture_passed": acceptance["full_core_fixture"]["valid"],
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D5C-P1 adapter fix verification failed: " + ", ".join(failed))
    report = {
        "fix_version": FIX_VERSION,
        "status": "d5c_p1_reporter_adapter_fix_fake_first_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "acceptance": acceptance,
        "decision": {
            "reporter_fix_implemented": True,
            "fake_acceptance_passed": True,
            "real_runner_offline_adapter": None,
            "historical_d5c_or_p1_conclusion_changed": False,
            "d5c_or_p1_rerun": False,
        },
        "source_digests": source_digests,
        "next_gate": config["next_gate"],
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "historical_d5c_rerun": False,
            "p1_rerun": False,
            "historical_d5c_conclusion_changed": False,
            "p1_conclusion_changed": False,
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
