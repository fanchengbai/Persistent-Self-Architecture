from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from psa.artifacts import canonical_json_bytes, sha256_file, sha256_json
from psa.model.rwkv7 import RWKV7Adapter, load_model_config
from psa.self_model.d4a_failure_diagnostic_runtime import (
    RWKV7RecompiledUnmodifiedRuntime,
)
from psa.self_model.d4a_real_diagnostic import _installed_source
from psa.self_model.d4b_steady_state_off_design import (
    PRECONDITION_ORDER,
    ROUTES,
    SCORED_ROUNDS,
)
from psa.self_model.d4b_steady_state_off_runtime import (
    D4B_PREFIX_TOKEN_IDS,
    D4B_TARGET_TOKEN_IDS,
    execute_d4b_fake_or_future_authorized_core,
)
from psa.self_model.rwkv7_coupling_adapter import RWKV7CouplingOffAdapter
from psa.self_model.rwkv7_instrumented_off_runtime import (
    RWKV7InstrumentedOffRuntime,
)


REPORT_VERSION = "0.1-d4b-real-off-equivalence"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_d4b_real_off_equivalence.json"
)
AUTHORIZATION_SCHEMA_RELATIVE_PATH = (
    "schemas/self_model_v0_1_d4b_real_authorization.schema.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "results/authorizations/self_model_v0_1_d4b_real_off_equivalence_v01.json"
)
OUTPUT_RELATIVE_DIR = (
    "results/development/self_model_v0_1_d4b_real_off_equivalence_v01"
)
D4B_EXECUTION_LOCK_ENV = "PSA_SELF_MODEL_D4B_REAL_OFF_GATE"
D4B_EXECUTION_LOCK_VALUE = "AUTHORIZED_D4B_REAL_2_9B_STEADY_OFF_ONCE"
D4B_OWNER_AUTHORIZATION_TEXT = (
    "授权执行 Self Model v0.1 D4B 真实2.9B稳态OFF等价门一次，并授权观察本次结果；"
    "不授权重跑D4或D4B、自动重跑、D5、active injection或Self效果实验。"
)
D4B_DESIGN_STATIC_REPORT_DIGEST = (
    "7f3cfb7fecf6892532f9ecdb27f528716b363077e37e288f8940d8e883ef658d"
)
D4B_RUNTIME_STATIC_REPORT_DIGEST = (
    "261325c459c08e5fa2c8d3e9ff08574c36e49ab36eccfabcf89e0045b7abae46"
)
AUTHORIZATION_FIELDS = {
    "authorization_version",
    "stage",
    "scope",
    "authorized",
    "authorization_basis",
    "authorization_text",
    "authorized_at_utc",
    "git_commit",
    "config_sha256",
    "runtime_static_report_sha256",
    "model_execution_authorized",
    "result_observation_authorized",
    "d4_status_change_authorized",
    "d4_rerun_authorized",
    "d4b_rerun_authorized",
    "active_injection_authorized",
    "self_effect_experiment_authorized",
    "d5_authorized",
    "confirmatory_decision_authorized",
    "automatic_rerun_authorized",
    "single_use",
    "authorization_digest_sha256",
}
ENTRY_SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    AUTHORIZATION_SCHEMA_RELATIVE_PATH,
    "docs/self_model_v0_1_d4b_real_off_equivalence_entry.md",
    "scripts/run_self_model_v0_1_d4b_real_off_equivalence.py",
    "scripts/verify_self_model_v0_1_d4b_real_off_equivalence_entry.py",
    "src/psa/self_model/d4b_real_off_equivalence.py",
    "src/psa/self_model/d4b_steady_state_off_runtime.py",
    "src/psa/self_model/d4a_failure_diagnostic_runtime.py",
    "src/psa/self_model/d4a_real_diagnostic.py",
    "src/psa/self_model/rwkv7_coupling_adapter.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "tests/test_self_model_d4b_real_off_equivalence.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise PermissionError(f"{field} must equal the frozen D4B value")


def _object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_spec(path: str | Path) -> dict[str, Any]:
    payload = _object(path, "D4B real OFF-equivalence config")
    exact = {
        "protocol_version": REPORT_VERSION,
        "stage": "D4B_real_2_9b_steady_state_off_entry",
        "status": "entry_implemented_execution_not_authorized",
        "development_only": True,
        "model_config_path": "configs/models/rwkv7_g1h_2.9b.candidate.json",
        "model_config_sha256": (
            "959143ab13eb9f86ad40e87a9164194ddb1fe6a74dbfdd4cb04bda354b0dae75"
        ),
        "model_id": "rwkv7-g1h-2.9b-20260710",
        "prefix_token_ids": D4B_PREFIX_TOKEN_IDS,
        "target_fixture": {
            "token_ids": D4B_TARGET_TOKEN_IDS,
            "state_input": "none",
            "full_output": False,
        },
        "routes": ROUTES,
        "fixed_preconditioning_order": PRECONDITION_ORDER,
        "scored_rounds": SCORED_ROUNDS,
        "prefix_call_count": 1,
        "preconditioning_call_count": 4,
        "scored_call_count": 16,
        "model_forward_call_count": 21,
        "within_route_comparison_count": 24,
        "cross_route_comparison_count": 96,
        "comparison": "torch.equal",
        "execution_lock_env": D4B_EXECUTION_LOCK_ENV,
        "execution_lock_value": D4B_EXECUTION_LOCK_VALUE,
        "authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE_PATH,
        "authorization_path": AUTHORIZATION_RELATIVE_PATH,
        "output_dir": OUTPUT_RELATIVE_DIR,
        "required_owner_authorization_text": D4B_OWNER_AUTHORIZATION_TEXT,
        "entry_implementation_authorized": True,
        "no_model_verification_authorized": True,
        "execution_authorized_at_implementation": False,
        "future_exact_owner_authorization_required": True,
        "future_machine_authorization_required": True,
        "single_use_claim_required": True,
        "claim_consumed_before_model_config_or_weights_or_load": True,
        "result_observation_requires_same_authorization": True,
        "d4_status_can_change": False,
        "d4_rerun_authorized": False,
        "d4b_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "failure_action": "persist_failure_and_stop_claim_consumed",
    }
    for field, expected in exact.items():
        _require_exact(payload.get(field), expected, field)
    _require_exact(
        payload.get("prerequisite"),
        {
            "d4_status": "failed_preserved",
            "d4b_design_static_report_digest_sha256": D4B_DESIGN_STATIC_REPORT_DIGEST,
            "d4b_runtime_static_status": "d4b_fake_first_runtime_static_verified",
            "d4b_runtime_static_report_digest_sha256": D4B_RUNTIME_STATIC_REPORT_DIGEST,
        },
        "prerequisite",
    )
    return payload


def _git_metadata(project_root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _require_clean_main(git: Mapping[str, str]) -> None:
    if git.get("branch") != "main" or git.get("status_porcelain"):
        raise RuntimeError("D4B requires a clean main worktree")
    commit = git.get("commit", "")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("D4B requires a full lowercase Git commit")


def _require_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("D4B authorization timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PermissionError("D4B authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PermissionError("D4B authorization timestamp must be UTC")


def _require_exact_path(root: Path, value: str | Path, relative: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != (root / relative).resolve():
        raise PermissionError(f"D4B {label} path is not frozen")
    return resolved


def _authorization_schema(root: Path) -> dict[str, Any]:
    schema = _object(root / AUTHORIZATION_SCHEMA_RELATIVE_PATH, "D4B authorization schema")
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or set(properties) != AUTHORIZATION_FIELDS:
        raise RuntimeError("D4B authorization schema properties changed")
    if not isinstance(required, list) or set(required) != AUTHORIZATION_FIELDS:
        raise RuntimeError("D4B authorization schema required fields changed")
    if schema.get("additionalProperties") is not False:
        raise RuntimeError("D4B authorization schema must reject extra fields")
    return schema


def build_d4b_real_authorization(
    *,
    config_path: str | Path,
    project_root: str | Path,
    authorization_text: str,
    git_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if authorization_text != D4B_OWNER_AUTHORIZATION_TEXT:
        raise PermissionError("D4B owner authorization text is not exact")
    root = Path(project_root).resolve()
    config_file = _require_exact_path(
        root, config_path, CONFIG_RELATIVE_PATH, "authorization config"
    )
    spec = _read_spec(config_file)
    _authorization_schema(root)
    git = dict(git_metadata) if git_metadata is not None else _git_metadata(root)
    _require_clean_main(git)
    authorization = {
        "authorization_version": REPORT_VERSION,
        "stage": spec["stage"],
        "scope": "real_2_9b_d4b_steady_state_off_equivalence_once",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": authorization_text,
        "authorized_at_utc": _utc_now(),
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_file),
        "runtime_static_report_sha256": D4B_RUNTIME_STATIC_REPORT_DIGEST,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "d4_status_change_authorized": False,
        "d4_rerun_authorized": False,
        "d4b_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    authorization["authorization_digest_sha256"] = sha256_json(authorization)
    return authorization


def validate_d4b_real_authorization(
    *,
    authorization_path: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    git: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    _authorization_schema(root)
    authorization = _object(authorization_path, "D4B machine authorization")
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise PermissionError("D4B machine authorization fields are not exact")
    stored_digest = authorization.get("authorization_digest_sha256")
    digest_payload = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_digest_sha256"
    }
    expected = {
        "authorization_version": REPORT_VERSION,
        "stage": "D4B_real_2_9b_steady_state_off_entry",
        "scope": "real_2_9b_d4b_steady_state_off_equivalence_once",
        "authorized": True,
        "authorization_basis": "project_owner_explicit_chat_authorization",
        "authorization_text": D4B_OWNER_AUTHORIZATION_TEXT,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "runtime_static_report_sha256": D4B_RUNTIME_STATIC_REPORT_DIGEST,
        "model_execution_authorized": True,
        "result_observation_authorized": True,
        "d4_status_change_authorized": False,
        "d4_rerun_authorized": False,
        "d4b_rerun_authorized": False,
        "active_injection_authorized": False,
        "self_effect_experiment_authorized": False,
        "d5_authorized": False,
        "confirmatory_decision_authorized": False,
        "automatic_rerun_authorized": False,
        "single_use": True,
    }
    for field, value in expected.items():
        _require_exact(authorization.get(field), value, f"authorization.{field}")
    _require_utc_timestamp(authorization.get("authorized_at_utc"))
    if not isinstance(stored_digest, str) or sha256_json(digest_payload) != stored_digest:
        raise PermissionError("D4B machine authorization digest is invalid")
    return authorization


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def _create_claim(
    *,
    output_dir: Path,
    config_path: Path,
    authorization_path: Path,
    git: Mapping[str, str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("D4B output directory is not empty; rerun refused")
    claim = {
        "claim_version": REPORT_VERSION,
        "status": "d4b_single_use_execution_claim_consumed",
        "created_at_utc": _utc_now(),
        "single_use": True,
        "automatic_rerun_authorized": False,
        "git_commit": git["commit"],
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
        "runtime_static_report_sha256": D4B_RUNTIME_STATIC_REPORT_DIGEST,
        "model_forward_call_count": 21,
    }
    return write_json_exclusive(output_dir / "execution_claim.json", claim)


def run_d4b_real_off_equivalence(
    *,
    config_path: str | Path,
    authorization_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_exact_path(
        root, config_path, CONFIG_RELATIVE_PATH, "execution config"
    )
    spec = _read_spec(config_file)
    if os.environ.get(D4B_EXECUTION_LOCK_ENV) != D4B_EXECUTION_LOCK_VALUE:
        raise PermissionError("the exact single-use D4B execution lock is absent")
    if os.environ.get("RWKV_DE_VERSION") is not None:
        raise PermissionError("RWKV_DE_VERSION must be unset for D4B")
    authorization_file = _require_exact_path(
        root, authorization_path, AUTHORIZATION_RELATIVE_PATH, "authorization"
    )
    destination = _require_exact_path(
        root, output_dir, OUTPUT_RELATIVE_DIR, "output"
    )
    git = _git_metadata(root)
    _require_clean_main(git)
    authorization = validate_d4b_real_authorization(
        authorization_path=authorization_file,
        config_path=config_file,
        project_root=root,
        git=git,
    )
    installed_version, source_path, source_bytes, source_digest = _installed_source()
    claim_path = _create_claim(
        output_dir=destination,
        config_path=config_file,
        authorization_path=authorization_file,
        git=git,
    )
    started = time.perf_counter()
    try:
        model_config_path = (root / spec["model_config_path"]).resolve()
        _require_exact(
            sha256_file(model_config_path),
            spec["model_config_sha256"],
            "model config digest",
        )
        model_config = load_model_config(model_config_path, root, verify_files=True)
        _require_exact(model_config.model_id, spec["model_id"], "loaded model id")
        for key, value in model_config.environment.items():
            os.environ[key] = value
        adapter = RWKV7Adapter.load(model_config)
        torch = adapter.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        rwkv_module = sys.modules["rwkv.model"]
        off_g1 = RWKV7CouplingOffAdapter(
            base_model=adapter.model,
            upstream_package_version=installed_version,
            upstream_model_source_sha256=source_digest,
        )
        g0 = RWKV7RecompiledUnmodifiedRuntime(
            base_model=adapter.model,
            upstream_source_bytes=source_bytes,
            upstream_globals=vars(rwkv_module),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
        )
        off_g2 = RWKV7InstrumentedOffRuntime(
            base_model=adapter.model,
            upstream_source_bytes=source_bytes,
            upstream_globals=vars(rwkv_module),
            upstream_package_version=installed_version,
            upstream_de_version=os.environ.get("RWKV_DE_VERSION"),
        )
        core = execute_d4b_fake_or_future_authorized_core(
            base_model=adapter.model,
            off_g1=off_g1,
            g0=g0,
            off_g2=off_g2,
            torch=torch,
        )
        torch.cuda.synchronize()
        core_record = copy.deepcopy(core)
        core_template_safety = core_record.pop("safety", {})
        core_record["execution_context"] = "authorized_real_2_9b"
        report = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": (
                "d4b_real_off_equivalence_passed"
                if core["valid"]
                else "d4b_real_off_equivalence_failed"
            ),
            "valid": core["valid"],
            "development_only": True,
            "git": git,
            "config": {"path": str(config_file), "sha256": sha256_file(config_file)},
            "authorization_digest_sha256": authorization[
                "authorization_digest_sha256"
            ],
            "execution_claim_sha256": sha256_file(claim_path),
            "model": adapter.model_metadata(),
            "upstream": {
                "package": "rwkv",
                "version": installed_version,
                "model_source_path": str(source_path),
                "model_source_sha256": source_digest,
            },
            "runtime_core": core_record,
            "runtime_core_template_safety": core_template_safety,
            "decision_effect": (
                "d5_review_candidate_only" if core["valid"] else "stop_without_rerun"
            ),
            "runtime_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "safety": {
                "real_2_9b_model_loaded": True,
                "real_2_9b_model_executed": True,
                "d4_status_changed": False,
                "d4_rerun": False,
                "d4b_rerun": False,
                "active_injection_implemented": False,
                "active_injection_executed": False,
                "self_projection_constructed": False,
                "self_effect_experiment_run": False,
                "d5_authorized": False,
                "confirmatory_decision_made": False,
                "automatic_rerun_authorized": False,
                "result_observation_authorized": True,
            },
        }
        report["report_digest_sha256"] = sha256_json(report)
        write_json_exclusive(destination / "report.json", report)
        return report
    except BaseException as error:
        failure = {
            "report_version": REPORT_VERSION,
            "created_at_utc": _utc_now(),
            "status": "d4b_execution_attempt_failed_claim_consumed",
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_claim_sha256": sha256_file(claim_path),
            "d4_status_changed": False,
            "d4b_rerun_authorized": False,
            "d5_authorized": False,
            "automatic_rerun_authorized": False,
        }
        failure["report_digest_sha256"] = sha256_json(failure)
        write_json_exclusive(destination / "failure.json", failure)
        raise


def _call_lines(source: str, function_name: str) -> dict[str, int]:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    lines: dict[str, int] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        lines.setdefault(name, node.lineno)
    return lines


def build_d4b_real_entry_static_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _require_exact_path(
        root, config_path, CONFIG_RELATIVE_PATH, "static config"
    )
    spec = _read_spec(config_file)
    schema = _authorization_schema(root)
    module_path = root / "src/psa/self_model/d4b_real_off_equivalence.py"
    module_source = module_path.read_text(encoding="utf-8")
    calls = _call_lines(module_source, "run_d4b_real_off_equivalence")
    required_calls = (
        "validate_d4b_real_authorization",
        "_installed_source",
        "_create_claim",
        "sha256_file",
        "load_model_config",
        "load",
        "execute_d4b_fake_or_future_authorized_core",
    )
    source_digests = {path: sha256_file(root / path) for path in ENTRY_SOURCE_PATHS}
    schema_properties = schema["properties"]
    checks = {
        "config_path_frozen": True,
        "config_valid": bool(spec),
        "d4_failure_prerequisite_preserved": spec["prerequisite"]["d4_status"]
        == "failed_preserved",
        "d4b_design_digest_frozen": spec["prerequisite"][
            "d4b_design_static_report_digest_sha256"
        ]
        == D4B_DESIGN_STATIC_REPORT_DIGEST,
        "d4b_runtime_digest_frozen": spec["prerequisite"][
            "d4b_runtime_static_report_digest_sha256"
        ]
        == D4B_RUNTIME_STATIC_REPORT_DIGEST,
        "fixtures_and_routes_frozen": spec["prefix_token_ids"] == [187, 931]
        and spec["target_fixture"]
        == {"token_ids": [2764], "state_input": "none", "full_output": False}
        and spec["routes"] == ROUTES,
        "precondition_and_latin_schedule_frozen": spec[
            "fixed_preconditioning_order"
        ]
        == PRECONDITION_ORDER
        and spec["scored_rounds"] == SCORED_ROUNDS,
        "twenty_one_calls_frozen": spec["prefix_call_count"] == 1
        and spec["preconditioning_call_count"] == 4
        and spec["scored_call_count"] == 16
        and spec["model_forward_call_count"] == 21,
        "comparison_counts_frozen": spec["within_route_comparison_count"] == 24
        and spec["cross_route_comparison_count"] == 96
        and spec["comparison"] == "torch.equal",
        "authorization_and_output_paths_frozen": spec["authorization_path"]
        == AUTHORIZATION_RELATIVE_PATH
        and spec["output_dir"] == OUTPUT_RELATIVE_DIR,
        "authorization_schema_exact": set(schema_properties) == AUTHORIZATION_FIELDS
        and set(schema["required"]) == AUTHORIZATION_FIELDS
        and schema["additionalProperties"] is False
        and schema_properties["authorization_text"]["const"]
        == D4B_OWNER_AUTHORIZATION_TEXT,
        "implementation_authorized": spec["entry_implementation_authorized"] is True,
        "execution_not_authorized_at_implementation": spec[
            "execution_authorized_at_implementation"
        ]
        is False,
        "future_exact_owner_authorization_required": spec[
            "future_exact_owner_authorization_required"
        ]
        is True,
        "single_use_claim_required": spec["single_use_claim_required"] is True,
        "all_required_entry_calls_present": all(name in calls for name in required_calls),
        "claim_precedes_model_config_weight_verification_and_load": calls[
            "_create_claim"
        ]
        < calls["sha256_file"]
        < calls["load_model_config"]
        < calls["load"],
        "runtime_core_runs_after_claim": calls["_create_claim"]
        < calls["execute_d4b_fake_or_future_authorized_core"],
        "d4_cannot_change": spec["d4_status_can_change"] is False,
        "d4_and_d4b_rerun_not_authorized": spec["d4_rerun_authorized"] is False
        and spec["d4b_rerun_authorized"] is False,
        "active_not_authorized": spec["active_injection_authorized"] is False,
        "self_effect_not_authorized": spec["self_effect_experiment_authorized"]
        is False,
        "d5_not_authorized": spec["d5_authorized"] is False,
        "automatic_rerun_not_authorized": spec["automatic_rerun_authorized"]
        is False,
        "source_inventory_complete": len(source_digests) == len(ENTRY_SOURCE_PATHS),
        "rwkv_model_not_imported": "rwkv.model" not in sys.modules,
        "torch_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D4B real entry static verification failed: " + ", ".join(failed))
    report = {
        "report_version": REPORT_VERSION,
        "status": "d4b_real_off_equivalence_entry_static_verified",
        "valid": True,
        "checks": checks,
        "source_digests": source_digests,
        "entry_call_lines": {name: calls[name] for name in required_calls},
        "safety": {
            "rwkv_model_imported": "rwkv.model" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "execution_claim_created": False,
            "machine_authorization_created": False,
            "active_injection_implemented": False,
            "self_effect_experiment_run": False,
            "d5_authorized": False,
            "automatic_rerun_authorized": False,
        },
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
