from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from psa.artifacts import sha256_file, sha256_json
from psa.development.exp001c_protocol_v02 import (
    build_exp001c_protocol_v02_manifest,
)


EXPERIMENT_ID = "EXP-001C"
DESIGN_MANIFEST_VERSION = "0.1-development-unrun"
STAGE_B_CONDITIONS = (
    "continuous",
    "restored",
    "swapped_I",
    "swapped_G",
    "swapped_both",
    "reset",
    "random_matched",
)
STATE_SEMANTIC_CONDITIONS = (
    "continuous",
    "restored",
    "swapped_I",
    "swapped_G",
    "swapped_both",
)
STAGE_B_DESIGN_SOURCE_FILES = (
    "configs/development/exp001c_noncore_protocol_v02.draft.json",
    "configs/development/exp001c_v02_stage_b_design.draft.json",
    "docs/exp001c_prospective_design.md",
    "docs/exp001c_v02_stage_a_pilot_v01_observation.md",
    "docs/exp001c_v02_stage_b_risk_review.md",
    "schemas/exp001c_v02_stage_b_authorization.schema.json",
    "schemas/exp001c_v02_stage_b_design.schema.json",
    "schemas/exp001c_v02_stage_b_execution_claim.schema.json",
    "schemas/exp001c_v02_stage_b_offline_result.schema.json",
    "schemas/exp001c_v02_stage_b_result.schema.json",
    "src/psa/artifacts/integrity.py",
    "src/psa/development/exp001c_protocol_v02.py",
    "src/psa/development/exp001c_v02_stage_b_design.py",
    "src/psa/development/exp001c_v02_stage_b_offline.py",
    "src/psa/development/exp001c_v02_stage_b_rwkv.py",
    "src/psa/development/exp001c_v02_stage_b_run.py",
    "src/psa/development/history_binding.py",
    "tests/test_exp001c_v02_stage_b_design.py",
    "tests/test_exp001c_v02_stage_b_offline.py",
    "tests/test_exp001c_v02_stage_b_rwkv.py",
    "tests/test_exp001c_v02_stage_b_run.py",
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "design_manifest_digest_sha256"
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    authority = config.get("authority")
    evidence = config.get("stage_a_pass_evidence")
    baseline = config.get("external_prompt_visible_baseline")
    plan = config.get("record_plan")
    gate = config.get("execution_gate")
    forbidden = config.get("forbidden_scope")
    conditions = config.get("conditions")
    if (
        config.get("design_version") != "0.1-development-draft"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scope")
        != "v02_stage_b_recurrent_state_noncore_pilot"
        or config.get("status") != "offline_design_only_execution_unapproved"
        or config.get("development_only") is not True
        or config.get("non_core") is not True
        or not isinstance(authority, Mapping)
        or authority.get("offline_design_authorized") is not True
        or authority.get("offline_risk_review_authorized") is not True
        or authority.get("offline_test_authorized") is not True
        or authority.get("offline_fake_runner_implementation_authorized")
        is not True
        or authority.get("real_rwkv_backend_implementation_authorized")
        is not True
        or authority.get("offline_execution_runner_implementation_authorized")
        is not True
        or authority.get("model_execution_authorized") is not False
        or authority.get("result_observation_authorized") is not False
        or authority.get("formal_test_set_access_authorized") is not False
        or authority.get("formal_run_authorized") is not False
        or authority.get("automatic_rerun_authorized") is not False
    ):
        raise PermissionError("EXP-001C v02 Stage B is offline-design-only")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("decision") != "stage_a_positive_control_pass"
        or evidence.get("record_count") != 32
        or evidence.get("label_marginalized_accuracy") != 0.875
        or evidence.get("remote_result_must_be_reverified_before_execution")
        is not True
        or not isinstance(evidence.get("stage_a_result_sha256"), str)
        or len(str(evidence.get("stage_a_result_sha256"))) != 64
        or not isinstance(baseline, Mapping)
        or baseline.get("source") != "completed_stage_a_result"
        or baseline.get("rerun_in_stage_b") is not False
        or baseline.get("record_count") != 32
    ):
        raise ValueError("EXP-001C v02 Stage A pass boundary is not locked")
    condition_names = (
        [item.get("name") for item in conditions]
        if isinstance(conditions, list)
        and all(isinstance(item, Mapping) for item in conditions)
        else None
    )
    if (
        condition_names != list(STAGE_B_CONDITIONS)
        or not isinstance(plan, Mapping)
        or plan.get("stage_a_trial_count") != 32
        or plan.get("stage_b_condition_count") != 7
        or plan.get("stage_b_record_count") != 224
        or plan.get("complete_four_code_rotation_required") is not True
        or plan.get("stage_a_rerun_forbidden") is not True
    ):
        raise ValueError("EXP-001C v02 Stage B record plan is not locked")
    if (
        not isinstance(gate, Mapping)
        or not all(value is True for value in gate.values())
        or not isinstance(forbidden, Mapping)
        or not all(value is True for value in forbidden.values())
    ):
        raise PermissionError("EXP-001C v02 Stage B execution gates are incomplete")


def _source_fields(
    condition: str,
    target_fields: Mapping[str, str],
    domain_values: tuple[str, str],
    operation_values: tuple[str, str],
) -> dict[str, str] | None:
    domain = str(target_fields["domain"])
    operation = str(target_fields["operation"])
    if condition in {"continuous", "restored"}:
        return {"domain": domain, "operation": operation}
    if condition == "swapped_I":
        return {
            "domain": next(value for value in domain_values if value != domain),
            "operation": operation,
        }
    if condition == "swapped_G":
        return {
            "domain": domain,
            "operation": next(
                value for value in operation_values if value != operation
            ),
        }
    if condition == "swapped_both":
        return {
            "domain": next(value for value in domain_values if value != domain),
            "operation": next(
                value for value in operation_values if value != operation
            ),
        }
    if condition in {"reset", "random_matched"}:
        return None
    raise ValueError(f"unsupported Stage B condition: {condition}")


def _matching_trial(
    trials: list[Mapping[str, Any]],
    fields: Mapping[str, str],
) -> Mapping[str, Any]:
    matches = [trial for trial in trials if trial.get("target_fields") == fields]
    if len(matches) != 1:
        raise ValueError("Stage B state source mapping is not unique")
    return matches[0]


def _target_code_for_fields(
    trial: Mapping[str, Any],
    fields: Mapping[str, str],
) -> str:
    matches = [
        option.get("code")
        for option in trial.get("option_mapping", [])
        if option.get("domain") == fields["domain"]
        and option.get("operation") == fields["operation"]
    ]
    if len(matches) != 1 or matches[0] not in set("ABCD"):
        raise ValueError("Stage B expected state target is not uniquely coded")
    return str(matches[0])


def build_exp001c_v02_stage_b_design_manifest(
    *,
    design_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    design_path = Path(design_config_path)
    if not design_path.is_absolute():
        design_path = root / design_path
    design_path = design_path.resolve()
    config = _load_object(design_path, "EXP-001C v02 Stage B design config")
    _validate_config(config)

    protocol_path = (root / str(config["protocol_config_path"])).resolve()
    protocol_manifest = build_exp001c_protocol_v02_manifest(
        config_path=protocol_path,
        project_root=root,
    )
    evidence = dict(config["stage_a_pass_evidence"])
    if (
        protocol_manifest.get("manifest_digest_sha256")
        != evidence.get("manifest_digest_sha256")
    ):
        raise ValueError("current v02 protocol no longer matches Stage A evidence")

    source_digests = {}
    for relative in STAGE_B_DESIGN_SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise ValueError(f"Stage B design source file is missing: {relative}")
        source_digests[relative] = sha256_file(source)
    design_relative = _relative(design_path, root)
    if source_digests.get(design_relative) != sha256_file(design_path):
        raise ValueError("Stage B design config is absent from source inventory")
    observation_path = root / str(evidence["observation_path"])
    evidence["observation_sha256"] = sha256_file(observation_path)

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for trial in protocol_manifest["trials"]:
        key = (str(trial["block_id"]), int(trial["rotation_index"]))
        groups.setdefault(key, []).append(trial)
    if len(groups) != 8 or any(len(trials) != 4 for trials in groups.values()):
        raise ValueError("Stage B requires eight complete four-state groups")

    records = []
    for condition in STAGE_B_CONDITIONS:
        for key in sorted(groups):
            trials = groups[key]
            domain_values = tuple(
                sorted({str(trial["target_fields"]["domain"]) for trial in trials})
            )
            operation_values = tuple(
                sorted(
                    {
                        str(trial["target_fields"]["operation"])
                        for trial in trials
                    }
                )
            )
            if len(domain_values) != 2 or len(operation_values) != 2:
                raise ValueError("Stage B group is not a complete 2x2 factorial")
            for trial in sorted(trials, key=lambda item: str(item["sample_id"])):
                source_fields = _source_fields(
                    condition,
                    trial["target_fields"],
                    domain_values,
                    operation_values,
                )
                if source_fields is None:
                    source_trial = trial if condition == "random_matched" else None
                    expected_target = None
                else:
                    source_trial = _matching_trial(trials, source_fields)
                    expected_target = _target_code_for_fields(trial, source_fields)
                records.append(
                    {
                        "record_id": (
                            f"stage-b-{condition}-{trial['sample_id']}"
                        ),
                        "condition": condition,
                        "condition_role": next(
                            item["role"]
                            for item in config["conditions"]
                            if item["name"] == condition
                        ),
                        "query_sample_id": trial["sample_id"],
                        "semantic_case_id": trial["semantic_case_id"],
                        "block_id": trial["block_id"],
                        "rotation_index": trial["rotation_index"],
                        "query_history_key": trial["history_key"],
                        "state_source_sample_id": (
                            source_trial["sample_id"]
                            if source_trial is not None
                            else None
                        ),
                        "state_source_history_key": (
                            source_trial["history_key"]
                            if source_trial is not None
                            else None
                        ),
                        "state_source_fields": source_fields,
                        "reference_stage_a_target_code": trial["target_code"],
                        "expected_state_semantic_target_code": expected_target,
                        "semantic_endpoint_role": (
                            "state_faithful_primary"
                            if condition in STATE_SEMANTIC_CONDITIONS
                            else "diagnostic_control"
                        ),
                        "history_digest_sha256": trial[
                            "history_digest_sha256"
                        ],
                        "query_digest_sha256": trial["query_digest_sha256"],
                    }
                )

    condition_counts = Counter(record["condition"] for record in records)
    if condition_counts != Counter({condition: 32 for condition in STAGE_B_CONDITIONS}):
        raise ValueError("Stage B condition counts are not balanced")
    if len(records) != 224 or len({record["record_id"] for record in records}) != 224:
        raise ValueError("Stage B record inventory is incomplete or duplicated")

    manifest = {
        "design_manifest_version": DESIGN_MANIFEST_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scope": "v02_stage_b_recurrent_state_noncore_pilot",
        "status": "offline_stage_b_design_verified_execution_unapproved",
        "development_only": True,
        "non_core": True,
        "model_executed": False,
        "execution_authorized": False,
        "result_observation_authorized": False,
        "formal_test_set_accessed": False,
        "formal_run_authorized": False,
        "automatic_rerun_authorized": False,
        "design_config": {
            "path": design_relative,
            "sha256": sha256_file(design_path),
        },
        "protocol_manifest_digest_sha256": protocol_manifest[
            "manifest_digest_sha256"
        ],
        "stage_a_pass_evidence": evidence,
        "external_prompt_visible_baseline": config[
            "external_prompt_visible_baseline"
        ],
        "condition_count": 7,
        "record_count": 224,
        "conditions": list(STAGE_B_CONDITIONS),
        "condition_counts": dict(condition_counts),
        "stage_a_rerun_included": False,
        "execution_gate": config["execution_gate"],
        "forbidden_scope": config["forbidden_scope"],
        "records": records,
        "locked_source_digests": dict(sorted(source_digests.items())),
    }
    manifest["design_manifest_digest_sha256"] = sha256_json(
        _manifest_payload(manifest)
    )
    return manifest


def verify_exp001c_v02_stage_b_design_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = _load_object(manifest_path, "EXP-001C v02 Stage B design manifest")
    stored_sources = manifest.get("locked_source_digests")
    source_checks = {}
    for relative in STAGE_B_DESIGN_SOURCE_FILES:
        source = root / relative
        expected = (
            stored_sources.get(relative)
            if isinstance(stored_sources, Mapping)
            else None
        )
        source_checks[relative] = bool(
            source.is_file()
            and isinstance(expected, str)
            and sha256_file(source) == expected
        )
    inventory_complete = bool(
        isinstance(stored_sources, Mapping)
        and set(stored_sources) == set(STAGE_B_DESIGN_SOURCE_FILES)
    )
    records = manifest.get("records")
    counts = (
        Counter(record.get("condition") for record in records)
        if isinstance(records, list)
        and all(isinstance(record, Mapping) for record in records)
        else Counter()
    )
    boundary_valid = bool(
        manifest.get("design_manifest_version") == DESIGN_MANIFEST_VERSION
        and manifest.get("experiment_id") == EXPERIMENT_ID
        and manifest.get("status")
        == "offline_stage_b_design_verified_execution_unapproved"
        and manifest.get("development_only") is True
        and manifest.get("non_core") is True
        and manifest.get("model_executed") is False
        and manifest.get("execution_authorized") is False
        and manifest.get("result_observation_authorized") is False
        and manifest.get("formal_test_set_accessed") is False
        and manifest.get("formal_run_authorized") is False
        and manifest.get("automatic_rerun_authorized") is False
        and manifest.get("stage_a_rerun_included") is False
        and manifest.get("conditions") == list(STAGE_B_CONDITIONS)
    )
    record_plan_valid = bool(
        isinstance(records, list)
        and len(records) == 224
        and manifest.get("record_count") == 224
        and manifest.get("condition_count") == 7
        and counts == Counter({condition: 32 for condition in STAGE_B_CONDITIONS})
        and len({record.get("record_id") for record in records}) == 224
        and all(
            record.get("expected_state_semantic_target_code") in set("ABCD")
            if record.get("condition") in STATE_SEMANTIC_CONDITIONS
            else record.get("expected_state_semantic_target_code") is None
            for record in records
        )
    )
    digest_valid = manifest.get("design_manifest_digest_sha256") == sha256_json(
        _manifest_payload(manifest)
    )
    config_entry = manifest.get("design_config")
    deterministic_payload_valid = False
    if isinstance(config_entry, Mapping):
        config_path = root / str(config_entry.get("path", ""))
        try:
            expected_manifest = build_exp001c_v02_stage_b_design_manifest(
                design_config_path=config_path,
                project_root=root,
            )
        except (OSError, TypeError, ValueError, PermissionError):
            expected_manifest = None
        deterministic_payload_valid = manifest == expected_manifest
    valid = bool(
        boundary_valid
        and record_plan_valid
        and digest_valid
        and deterministic_payload_valid
        and inventory_complete
        and all(source_checks.values())
    )
    return {
        "verification_version": DESIGN_MANIFEST_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "offline_stage_b_design_verified" if valid else "invalid",
        "valid": valid,
        "safety_boundary_valid": boundary_valid,
        "record_plan_valid": record_plan_valid,
        "design_manifest_digest_valid": digest_valid,
        "deterministic_payload_valid": deterministic_payload_valid,
        "source_inventory_complete": inventory_complete,
        "source_checks": source_checks,
        "model_executed": False,
        "formal_test_set_accessed": False,
        "stage_a_rerun_included": False,
    }
