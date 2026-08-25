from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Sequence

from psa.artifacts import sha256_file, sha256_json
from psa.self_model.d5c_failure_lifecycle_diagnostic import (
    DIAGNOSTIC_SOURCE,
    OfflineTensor,
    _namespace,
    _output_digest,
    _state,
)
from psa.self_model.d6d_core_approach_design import CONDITIONS, SELF_CONDITIONS
from psa.self_model.d6d_projection_artifact import (
    FrozenSelfProjection,
    ProjectionTrainingRecord,
    audit_frozen_projection_artifact,
    build_frozen_projection_artifact,
)
from psa.self_model.d6d_wrapper_runtime import (
    D6DIResidualCallback,
    D6DIWrapperOwnedRuntime,
    request_for_condition,
)
from psa.self_model.rwkv7_instrumented_off_runtime import (
    CALLBACK_ATTRIBUTE,
    compile_instrumented_methods,
)
from psa.self_model.state import build_self_state


IMPLEMENTATION_VERSION = "0.1-coupling-d6d-i-wrapper-projection-tooling"
CONFIG_RELATIVE_PATH = (
    "configs/development/self_model_v0_1_coupling_d6d_i_wrapper_projection_tooling.json"
)
REQUIRED_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6D-I wrapper-owned真实路径与冻结Self projection"
    "构建工具的无模型实现；必须保持D6D单一联合实验、不得修改真实RWKV实例字典，只实现"
    "wrapper、projection训练冻结与artifact审计接口及纯Python验收；不探测installed source、"
    "不导入RWKV/Torch、不访问权重、不加载或执行模型，不授权D6D真实执行、D6E、正式测试集、"
    "Self效果结论、Self Updater、D5C/P1/P2/D6C重跑或自动重跑。"
)
NEXT_CONFIRMATION = (
    "确认进入 Self Model v0.1 Coupling-D6D-II installed source静态兼容、联合训练/试验"
    "manifest与单次真实入口的无模型实现；只允许探测并静态编译锁定installed source、冻结"
    "projection训练与pilot清单、实现新Schema/唯一目录/single-use claim入口，不访问权重、"
    "不加载或执行模型，不构造真实projection，也不授权D6D真实执行、D6E、正式测试集、"
    "Self效果结论、Self Updater、任何历史重跑或自动重跑。"
)
CLASSIFICATION = (
    "d6d_i_wrapper_owned_and_projection_artifact_tooling_passes_joint_pure_python_"
    "acceptance_real_projection_and_model_not_run"
)
D6D_DESIGN_REPORT_DIGEST = "3862a681d3658b645f141eb543b1076aa008a3be0ed805a31e8d3d022b081f75"
D6D_DESIGN_CONFIG_DIGEST = "fac1424aebc5e7e2e498221ec460f90842bbbdde6811fa2c3dfeee1ce82531cc"
D6D_DESIGN_DOCUMENT_DIGEST = "e0b74b8359d3280153a6455b887a498c3ad1e576128be8b0419971480ec7a094"
D6D_DESIGN_SOURCE_DIGEST = "c497153a3a48acc26eb08f71ebaf70a0aed2b314247ee9adb34b86ec11c56c97"
INSTRUMENTER_DIGEST = "ce9862b6739980305f854c9a63a08a5b872e73d53ae6098f626998ee0324aea5"
FIXTURE_SOURCE_DIGEST = "ded1cc371411906a26650edf217e245c04a40564e4b91c72f7ca7d01b5dfe3e2"
HIDDEN_DIMENSION = 2560
TARGET_LAYER_INDEX = 15
ACCEPTANCE_CATEGORIES = (
    "locked_ast_compiles_once_per_execution_path",
    "wrapper_owns_forward_methods_dispatcher_and_context",
    "base_instance_dictionary_unchanged_by_constructor_and_all_calls",
    "runtime_forward_has_no_base_setattr_delattr_or_base_forward_call",
    "all_eleven_conditions_run_on_one_wrapper",
    "off_zero_and_double_mask_outputs_exact",
    "synthetic_positive_changes_output_at_frozen_layer",
    "matched_swap_mask_and_random_projection_routes_are_distinguishable",
    "field_branch_swap_mask_and_random_semantics_exact",
    "norm_matched_random_preserves_each_branch_norm",
    "projection_artifact_is_learned_frozen_bias_free_and_digest_bound",
    "synthetic_teacher_artifact_is_not_research_evidence",
    "artifact_parameter_or_metadata_tampering_fails_closed",
    "method_dispatcher_and_context_identities_stable",
    "context_restores_after_success_and_callback_failure",
    "nested_request_rejected_before_inner_forward",
    "concurrent_request_rejected_before_inner_forward",
    "single_and_sequence_paths_use_same_wrapper",
    "tokens_recurrent_state_self_states_and_projection_source_unchanged",
    "no_rwkv_torch_weight_model_or_later_gate_side_effect",
)
SOURCE_PATHS = (
    CONFIG_RELATIVE_PATH,
    "configs/development/self_model_v0_1_coupling_d6d_core_approach_design.json",
    "docs/self_model_v0_1_coupling_d6d_core_approach_design.md",
    "docs/self_model_v0_1_coupling_d6d_i_wrapper_projection_tooling.md",
    "scripts/verify_self_model_v0_1_coupling_d6d_i_wrapper_projection_tooling.py",
    "src/psa/self_model/d6d_core_approach_design.py",
    "src/psa/self_model/d6d_i_tooling.py",
    "src/psa/self_model/d6d_projection_artifact.py",
    "src/psa/self_model/d6d_wrapper_runtime.py",
    "src/psa/self_model/rwkv7_instrumented_off_runtime.py",
    "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py",
    "tests/test_self_model_d6d_i_tooling.py",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("D6D-I config must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, bool]:
    prerequisites = config.get("prerequisites", {})
    wrapper = config.get("wrapper_tooling", {})
    projection = config.get("projection_tooling", {})
    joint = config.get("joint_acceptance", {})
    authority = config.get("authority", {})
    closed = (
        "installed_source_probe_authorized",
        "rwkv_import_authorized",
        "torch_import_authorized",
        "weights_access_authorized",
        "model_load_authorized",
        "model_execution_authorized",
        "real_projection_training_run_authorized",
        "real_projection_artifact_construction_authorized",
        "d6d_real_execution_authorized",
        "d6e_authorized",
        "formal_test_set_authorized",
        "self_effect_conclusion_authorized",
        "self_updater_authorized",
        "d5c_rerun_authorized",
        "p1_rerun_authorized",
        "p2_authorized",
        "d6c_rerun_authorized",
        "automatic_rerun_authorized",
    )
    checks = {
        "identity_and_status_exact": config.get("implementation_version")
        == IMPLEMENTATION_VERSION
        and config.get("stage")
        == "Coupling-D6D-I_wrapper_owned_path_and_projection_artifact_tooling_no_model"
        and config.get("status")
        == "implementation_and_pure_python_acceptance_only_real_execution_not_authorized"
        and config.get("development_only") is True,
        "confirmation_exact": config.get("owner_confirmation_text")
        == REQUIRED_CONFIRMATION,
        "d6d_design_evidence_frozen": prerequisites.get(
            "d6d_design_report_digest_sha256"
        ) == D6D_DESIGN_REPORT_DIGEST
        and prerequisites.get("d6d_design_config_sha256")
        == D6D_DESIGN_CONFIG_DIGEST
        and prerequisites.get("d6d_design_document_sha256")
        == D6D_DESIGN_DOCUMENT_DIGEST
        and prerequisites.get("d6d_design_source_sha256")
        == D6D_DESIGN_SOURCE_DIGEST,
        "instrumenter_and_fixture_frozen": prerequisites.get(
            "locked_instrumenter_sha256"
        ) == INSTRUMENTER_DIGEST
        and prerequisites.get("offline_fixture_source_sha256")
        == FIXTURE_SOURCE_DIGEST,
        "d6c_stop_preserved": prerequisites.get("d6c_status")
        == "failed_claim_consumed_no_rerun"
        and prerequisites.get("d6c_execution_claim_sha256")
        == "82b94c33513da0137127ce44a85513c48c381d92b87e3a7c27916931821fe6a3",
        "wrapper_ownership_exact": wrapper.get("class")
        == "D6DIWrapperOwnedRuntime"
        and wrapper.get("scope") == "unloaded_pure_python_fixture_only"
        and wrapper.get("compiled_methods") == ["forward_one", "forward_seq"]
        and wrapper.get("wrapper_owns_forward") is True
        and wrapper.get("wrapper_owns_fixed_dispatcher") is True
        and wrapper.get("wrapper_owns_request_context") is True
        and wrapper.get("instrumented_methods_bound_to_wrapper_only") is True,
        "base_dictionary_contract_exact": wrapper.get(
            "base_model_instance_setattr_or_delattr_allowed"
        ) is False
        and wrapper.get("base_model_instance_dictionary_snapshot")
        == "exact_keys_and_object_identities"
        and wrapper.get("snapshot_required")
        == ["before_constructor", "after_constructor", "before_each_forward", "after_each_forward"]
        and wrapper.get("attribute_delegation") == "read_only_getattr",
        "wrapper_lifecycle_exact": wrapper.get("off_zero_and_active_share_one_wrapper")
        is True
        and wrapper.get("nested_and_concurrent_policy")
        == "reject_before_inner_forward"
        and wrapper.get("callback_exception_policy")
        == "restore_context_discard_output_and_remain_reusable"
        and wrapper.get("installed_source_probe") is False,
        "projection_artifact_contract_exact": projection.get("artifact_version")
        == "0.1-d6d-field-separated-frozen-projection"
        and projection.get("trainer_kind")
        == "categorical_branch_mean_pure_python_v0.1"
        and projection.get("source_fields") == ["identity_anchors", "active_goals"]
        and projection.get("field_branches_separate") is True
        and projection.get("bias_present") is False
        and projection.get("double_mask_projection_exact_zero") is True
        and projection.get("real_output_dimension") == HIDDEN_DIMENSION
        and projection.get("target_layer_index_zero_based") == TARGET_LAYER_INDEX,
        "projection_integrity_and_blinding_exact": projection.get(
            "training_manifest_digest_required"
        ) is True
        and projection.get("blinded_pilot_commitment_digest_required") is True
        and projection.get("training_and_pilot_digests_must_differ") is True
        and projection.get("parameter_digest_required") is True
        and projection.get("artifact_digest_required") is True
        and projection.get("tamper_detection_required") is True,
        "projection_safety_exact": projection.get("prompt_serialization_allowed")
        is False
        and projection.get("base_model_parameters_in_artifact_allowed") is False
        and projection.get("online_update_allowed") is False
        and projection.get("pure_python_acceptance_uses_synthetic_teacher") is True
        and projection.get("pure_python_acceptance_artifact_research_evidence_eligible")
        is False
        and projection.get("real_projection_artifact_constructed_this_round") is False,
        "one_joint_acceptance_exact": joint.get(
            "same_wrapper_contains_synthetic_and_self_conditions"
        ) is True
        and joint.get("separate_mechanism_acceptance_run") is False
        and joint.get("conditions") == list(CONDITIONS)
        and joint.get("fixture_shape")
        == {"layers": 32, "hidden_dimension": 2560, "state_components": 96},
        "acceptance_categories_exact": joint.get("acceptance_categories")
        == list(ACCEPTANCE_CATEGORIES),
        "implementation_authority_exact": authority.get(
            "d6d_i_implementation_authorized"
        ) is True
        and authority.get("pure_python_joint_acceptance_authorized") is True
        and authority.get("wrapper_fixture_runtime_authorized") is True
        and authority.get("projection_tooling_interface_authorized") is True
        and authority.get("synthetic_teacher_fixture_artifact_authorized") is True,
        "model_real_projection_and_later_authority_closed": all(
            authority.get(name) is False for name in closed
        ),
        "next_gate_exact": config.get("required_next_owner_confirmation_text")
        == NEXT_CONFIRMATION
        and config.get("next_gate")
        == "remote_no_model_d6d_i_verification_then_separate_d6d_ii_confirmation",
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise PermissionError("D6D-I config failed closed: " + ", ".join(failed))
    return checks


def _field_item(item_id: str, value: str, update_class: str) -> dict[str, Any]:
    return {
        "field_item_id": item_id,
        "value": value,
        "value_type": "string",
        "confidence": 1.0,
        "update_class": update_class,
        "created_step": 0,
        "updated_step": 0,
        "source_evidence_ids": ["d6d-i-synthetic-teacher"],
        "status": "active",
    }


def _self_state(state_id: str, identity: str, goal: str) -> dict[str, Any]:
    return build_self_state(
        state_id=state_id,
        agent_instance_id="d6d-i-offline-fixture",
        trajectory_id="d6d-i-joint-acceptance",
        step=0,
        model_id="no-model-fixture",
        tokenizer_id="no-tokenizer-fixture",
        fields={
            "identity_anchors": [_field_item(f"{state_id}-identity", identity, "protected")],
            "active_goals": [_field_item(f"{state_id}-goal", goal, "fast")],
        },
        provenance_refs=["pure-python-no-model"],
    )


def _branch_vector(kind: str, key: str) -> tuple[float, ...]:
    sign = 1.0 if key in {"amber", "orbit"} else -1.0
    offset = 1 if kind == "identity" else 3
    return tuple(
        sign * (((index + offset) % 17) + 1) / 100_000.0
        for index in range(HIDDEN_DIMENSION)
    )


def _fixture_artifact() -> dict[str, Any]:
    records = tuple(
        ProjectionTrainingRecord(
            identity_key=identity,
            goal_key=goal,
            identity_target=_branch_vector("identity", identity),
            goal_target=_branch_vector("goal", goal),
        )
        for identity in ("amber", "cobalt")
        for goal in ("orbit", "harbor")
    )
    return build_frozen_projection_artifact(
        records=records,
        output_dimension=HIDDEN_DIMENSION,
        training_manifest_sha256=sha256_json(
            {"kind": "d6d-i-synthetic-teacher-training", "records": 4}
        ),
        pilot_manifest_commitment_sha256=sha256_json(
            {"kind": "d6d-i-blinded-pilot-commitment", "fixtures": 12}
        ),
        optimizer_seed=260825,
        fixture_only=True,
    )


class _OfflineVectorCallback:
    def __init__(self, vector: Sequence[float], *, fail: bool = False) -> None:
        self.vector = tuple(float(value) for value in vector)
        self.fail = fail
        self.invocations = 0
        self.applications = 0

    def __call__(self, **payload: Any) -> OfflineTensor:
        if self.fail:
            raise RuntimeError("D6D-I synthetic callback failure")
        residual = payload.get("residual_x")
        if type(residual) is not OfflineTensor:
            raise TypeError("D6D-I acceptance requires OfflineTensor")
        self.invocations += 1
        if payload.get("layer_index") != TARGET_LAYER_INDEX:
            return residual
        self.applications += 1
        delta = OfflineTensor(self.vector, dtype=residual.dtype, device=residual.device)
        return residual + delta


def _runtime_forward_audit() -> dict[str, Any]:
    path = Path(__file__).with_name("d6d_wrapper_runtime.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "D6DIWrapperOwnedRuntime"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "forward"
    )
    mutation_calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"setattr", "delattr"}
    ]
    base_forward_calls = [
        ast.unparse(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward"
        and "_base_model" in ast.unparse(node.func)
    ]
    return {
        "forward_line": function.lineno,
        "setattr_or_delattr_calls": mutation_calls,
        "base_model_forward_calls": base_forward_calls,
    }


def run_joint_pure_python_acceptance() -> dict[str, Any]:
    namespace, fixture_type = _namespace()
    base_model = fixture_type()
    base_before = dict(base_model.__dict__)
    compiled, counts = compile_instrumented_methods(
        upstream_source=DIAGNOSTIC_SOURCE,
        upstream_globals=namespace,
        rwkv_de_version=None,
    )
    runtime = D6DIWrapperOwnedRuntime(
        base_model=base_model,
        compiled_methods=compiled,
        injection_counts=counts,
    )
    artifact = _fixture_artifact()
    artifact_audit = audit_frozen_projection_artifact(artifact)
    projection = FrozenSelfProjection(artifact)
    matched_state = _self_state("d6di-matched", "amber", "orbit")
    paired_state = _self_state("d6di-paired", "cobalt", "harbor")
    matched_before = copy.deepcopy(matched_state)
    paired_before = copy.deepcopy(paired_state)
    tokens = [7]
    source_state = _state()
    tokens_before = list(tokens)
    source_state_before = list(source_state)
    synthetic_vector = tuple(0.0002 for _ in range(HIDDEN_DIMENSION))
    callbacks: dict[str, _OfflineVectorCallback] = {
        "synthetic_positive": _OfflineVectorCallback(synthetic_vector)
    }
    projections: dict[str, dict[str, Any]] = {}
    for condition in SELF_CONDITIONS:
        projected = projection.project_condition(
            matched_state=matched_state,
            paired_state=paired_state,
            condition=condition,
            random_seed=260825,
        )
        projections[condition] = projected
        callbacks[condition] = _OfflineVectorCallback(projected["aggregate_vector"])

    outputs: dict[str, tuple[Any, Any]] = {}
    for condition in CONDITIONS:
        if condition in {"wrapper_off", "wrapper_zero"}:
            callback = None
        else:
            callback = D6DIResidualCallback(
                "synthetic_positive" if condition == "synthetic_positive" else "frozen_self",
                callbacks[condition],
            )
        outputs[condition] = runtime.forward(
            tokens,
            source_state,
            full_output=False,
            coupling=request_for_condition(condition, callback),
        )
    output_digests = {name: _output_digest(value) for name, value in outputs.items()}
    context_after_routes = runtime.context_is_empty()
    identity_after_routes = runtime.owned_bindings_are_stable()

    sequence_off = runtime.forward(
        [7, 11], _state(), full_output=True,
        coupling=request_for_condition("wrapper_off"),
    )
    sequence_zero = runtime.forward(
        [7, 11], _state(), full_output=True,
        coupling=request_for_condition("wrapper_zero"),
    )

    failing_callback = D6DIResidualCallback(
        "synthetic_positive", _OfflineVectorCallback(synthetic_vector, fail=True)
    )
    try:
        runtime.forward(
            [5], _state(), full_output=False,
            coupling=request_for_condition("synthetic_positive", failing_callback),
        )
    except RuntimeError as error:
        failure_preserved = str(error) == "D6D-I synthetic callback failure"
    else:
        failure_preserved = False
    context_after_failure = runtime.context_is_empty()
    recovery = runtime.forward(
        [5], _state(), full_output=False,
        coupling=request_for_condition("wrapper_off"),
    )
    reusable_after_failure = recovery is not None and runtime.context_is_empty()

    nested_rejected = False

    def nested_callback(**payload: Any) -> OfflineTensor:
        nonlocal nested_rejected
        try:
            runtime.forward(
                [3], _state(), full_output=False,
                coupling=request_for_condition("wrapper_off"),
            )
        except RuntimeError:
            nested_rejected = True
        return payload["residual_x"]

    runtime.forward(
        [3], _state(), full_output=False,
        coupling=request_for_condition(
            "synthetic_positive",
            D6DIResidualCallback("synthetic_positive", nested_callback),
        ),
    )

    entered = threading.Event()
    release = threading.Event()
    first = True

    def blocking_callback(**payload: Any) -> OfflineTensor:
        nonlocal first
        if first:
            first = False
            entered.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("D6D-I concurrent acceptance timed out")
        return payload["residual_x"]

    thread_errors: list[BaseException] = []

    def outer() -> None:
        try:
            runtime.forward(
                [4], _state(), full_output=False,
                coupling=request_for_condition(
                    "synthetic_positive",
                    D6DIResidualCallback("synthetic_positive", blocking_callback),
                ),
            )
        except BaseException as error:
            thread_errors.append(error)

    thread = threading.Thread(target=outer)
    thread.start()
    if not entered.wait(timeout=5.0):
        release.set()
        thread.join(timeout=5.0)
        raise TimeoutError("D6D-I concurrent outer call did not enter")
    try:
        runtime.forward(
            [4], _state(), full_output=False,
            coupling=request_for_condition("wrapper_off"),
        )
    except RuntimeError:
        concurrent_rejected = True
    else:
        concurrent_rejected = False
    release.set()
    thread.join(timeout=5.0)
    if thread.is_alive() or thread_errors:
        raise RuntimeError("D6D-I concurrent acceptance outer call failed")

    tampered = copy.deepcopy(artifact)
    tampered["parameters"]["identity_weights"]["amber"][0] += 1.0
    try:
        audit_frozen_projection_artifact(tampered)
    except RuntimeError:
        tamper_rejected = True
    else:
        tamper_rejected = False

    random_projection = projections["self_identity_goal_norm_matched_random"]
    matched_projection = projections["self_matched"]
    route_projection_digests = {
        condition: projections[condition]["aggregate_digest_sha256"]
        for condition in SELF_CONDITIONS
    }
    base_after = dict(base_model.__dict__)
    audit = _runtime_forward_audit()
    checks = {
        ACCEPTANCE_CATEGORIES[0]: counts == {"forward_one": 1, "forward_seq": 1},
        ACCEPTANCE_CATEGORIES[1]: runtime.installation_count == 3
        and CALLBACK_ATTRIBUTE in runtime.__dict__
        and "forward_one" in runtime.__dict__ and "forward_seq" in runtime.__dict__,
        ACCEPTANCE_CATEGORIES[2]: base_before.keys() == base_after.keys()
        and all(base_before[name] is base_after[name] for name in base_before)
        and runtime.base_dictionary_is_stable(),
        ACCEPTANCE_CATEGORIES[3]: not audit["setattr_or_delattr_calls"]
        and not audit["base_model_forward_calls"],
        ACCEPTANCE_CATEGORIES[4]: set(outputs) == set(CONDITIONS),
        ACCEPTANCE_CATEGORIES[5]: output_digests["wrapper_off"]
        == output_digests["wrapper_zero"]
        == output_digests["self_identity_goal_mask"],
        ACCEPTANCE_CATEGORIES[6]: output_digests["synthetic_positive"]
        != output_digests["wrapper_off"]
        and callbacks["synthetic_positive"].applications == 1,
        ACCEPTANCE_CATEGORIES[7]: len(set(route_projection_digests.values())) == 8,
        ACCEPTANCE_CATEGORIES[8]: projections["self_identity_swap"]["identity_key"]
        == "cobalt"
        and projections["self_identity_swap"]["goal_key"] == "orbit"
        and projections["self_goal_swap"]["identity_key"] == "amber"
        and projections["self_goal_swap"]["goal_key"] == "harbor"
        and projections["self_identity_mask"]["identity_l2_norm"] == 0.0
        and projections["self_goal_mask"]["goal_l2_norm"] == 0.0,
        ACCEPTANCE_CATEGORIES[9]: math.isclose(
            random_projection["identity_l2_norm"],
            matched_projection["identity_l2_norm"],
            rel_tol=1e-12,
        )
        and math.isclose(
            random_projection["goal_l2_norm"],
            matched_projection["goal_l2_norm"],
            rel_tol=1e-12,
        ),
        ACCEPTANCE_CATEGORIES[10]: artifact_audit["valid"]
        and artifact["status"] == "frozen"
        and artifact["bias_present"] is False
        and artifact["parameter_digest_sha256"]
        == artifact_audit["parameter_digest_sha256"],
        ACCEPTANCE_CATEGORIES[11]: artifact["fixture_only"] is True
        and artifact["research_evidence_eligible"] is False,
        ACCEPTANCE_CATEGORIES[12]: tamper_rejected,
        ACCEPTANCE_CATEGORIES[13]: identity_after_routes
        and runtime.owned_bindings_are_stable()
        and runtime.context_is_empty(),
        ACCEPTANCE_CATEGORIES[14]: context_after_routes and failure_preserved
        and context_after_failure and reusable_after_failure,
        ACCEPTANCE_CATEGORIES[15]: nested_rejected,
        ACCEPTANCE_CATEGORIES[16]: concurrent_rejected,
        ACCEPTANCE_CATEGORIES[17]: _output_digest(sequence_off)
        == _output_digest(sequence_zero)
        and len(sequence_off[0].shape) == 2,
        ACCEPTANCE_CATEGORIES[18]: tokens == tokens_before
        and source_state == source_state_before
        and matched_state == matched_before and paired_state == paired_before
        and artifact == artifact_audit["artifact"],
        ACCEPTANCE_CATEGORIES[19]: runtime.model_loaded is False
        and runtime.model_executed is False
        and "rwkv.model" not in sys.modules and "torch" not in sys.modules,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D-I joint acceptance failed: " + ", ".join(failed))
    return {
        "valid": True,
        "checks": checks,
        "counts": {
            "joint_conditions": len(outputs),
            "wrapper_executions": runtime.execution_count,
            "wrapper_rejections": runtime.rejection_count,
            "dispatcher_calls": runtime.dispatcher.dispatch_count,
            "dispatcher_callbacks": runtime.dispatcher.callback_count,
            "synthetic_applications": callbacks["synthetic_positive"].applications,
            "self_projection_conditions": len(SELF_CONDITIONS),
            "layers": 32,
            "hidden_dimension": HIDDEN_DIMENSION,
            "state_components": 96,
        },
        "artifact": {
            "artifact_digest_sha256": artifact["artifact_digest_sha256"],
            "parameter_digest_sha256": artifact["parameter_digest_sha256"],
            "fixture_only": True,
            "research_evidence_eligible": False,
            "output_dimension": artifact["output_dimension"],
            "bias_present": False,
        },
        "route_output_digests": output_digests,
        "route_projection_digests": route_projection_digests,
        "runtime_forward_ast_audit": audit,
        "base_instance_dictionary_before_keys": sorted(base_before),
        "base_instance_dictionary_after_keys": sorted(base_after),
        "installed_source_probed": False,
        "model_loaded": False,
        "model_executed": False,
    }


def _restricted_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return sorted(imported & {"rwkv", "torch"})


def build_d6d_i_report(
    *, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if config_file != (root / CONFIG_RELATIVE_PATH).resolve():
        raise PermissionError("D6D-I requires the frozen project config path")
    config = _object(config_file)
    config_checks = validate_config(config)
    acceptance = run_joint_pure_python_acceptance()
    source_digests = {relative: sha256_file(root / relative) for relative in SOURCE_PATHS}
    reviewed = (
        root / "src/psa/self_model/d6d_i_tooling.py",
        root / "src/psa/self_model/d6d_projection_artifact.py",
        root / "src/psa/self_model/d6d_wrapper_runtime.py",
        root / "scripts/verify_self_model_v0_1_coupling_d6d_i_wrapper_projection_tooling.py",
        root / "tests/test_self_model_d6d_i_tooling.py",
    )
    restricted = {
        str(path.relative_to(root)).replace("\\", "/"): _restricted_imports(path)
        for path in reviewed
    }
    checks = {
        "config_valid": all(config_checks.values()),
        "joint_acceptance_valid": acceptance["valid"],
        "all_twenty_acceptance_categories_present": len(acceptance["checks"]) == 20,
        "d6d_design_config_frozen": source_digests[
            "configs/development/self_model_v0_1_coupling_d6d_core_approach_design.json"
        ] == D6D_DESIGN_CONFIG_DIGEST,
        "d6d_design_document_frozen": source_digests[
            "docs/self_model_v0_1_coupling_d6d_core_approach_design.md"
        ] == D6D_DESIGN_DOCUMENT_DIGEST,
        "d6d_design_source_frozen": source_digests[
            "src/psa/self_model/d6d_core_approach_design.py"
        ] == D6D_DESIGN_SOURCE_DIGEST,
        "locked_instrumenter_frozen": source_digests[
            "src/psa/self_model/rwkv7_instrumented_off_runtime.py"
        ] == INSTRUMENTER_DIGEST,
        "offline_fixture_frozen": source_digests[
            "src/psa/self_model/d5c_failure_lifecycle_diagnostic.py"
        ] == FIXTURE_SOURCE_DIGEST,
        "source_inventory_complete": len(source_digests) == len(SOURCE_PATHS),
        "no_d6d_i_rwkv_or_torch_import": all(not values for values in restricted.values()),
        "installed_source_not_probed": acceptance["installed_source_probed"] is False,
        "model_not_loaded_or_executed": acceptance["model_loaded"] is False
        and acceptance["model_executed"] is False,
        "real_projection_not_constructed": acceptance["artifact"]["fixture_only"] is True
        and acceptance["artifact"]["research_evidence_eligible"] is False,
        "no_later_gate_or_rerun": all(
            config["authority"][name] is False
            for name in (
                "d6d_real_execution_authorized", "d6e_authorized",
                "formal_test_set_authorized", "self_effect_conclusion_authorized",
                "self_updater_authorized", "d5c_rerun_authorized",
                "p1_rerun_authorized", "p2_authorized", "d6c_rerun_authorized",
                "automatic_rerun_authorized",
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise RuntimeError("D6D-I report failed: " + ", ".join(failed))
    report: dict[str, Any] = {
        "report_version": IMPLEMENTATION_VERSION,
        "status": "d6d_i_wrapper_projection_tooling_no_model_verified",
        "valid": True,
        "classification": CLASSIFICATION,
        "config_checks": config_checks,
        "checks": checks,
        "acceptance": acceptance,
        "restricted_import_audit": restricted,
        "decision": {
            "d6d_i_tooling_implemented": True,
            "single_joint_experiment_preserved": True,
            "separate_mechanism_execution_round_created": False,
            "real_wrapper_validated_on_installed_source": False,
            "real_projection_constructed": False,
            "d6d_execution_authorized": False,
            "self_effect_conclusion": False,
        },
        "safety": {
            "installed_source_probed": False,
            "rwkv_model_imported": False,
            "torch_imported": False,
            "weights_accessed": False,
            "model_loaded": False,
            "model_executed": False,
            "real_model_instance_mutated": False,
            "real_projection_training_run": False,
            "real_projection_artifact_constructed": False,
            "formal_test_set_used": False,
            "self_effect_conclusion_made": False,
            "self_updater_used": False,
            "historical_rerun": False,
            "automatic_rerun_authorized": False,
        },
        "source_digests": dict(sorted(source_digests.items())),
        "next_gate": config["next_gate"],
        "required_next_owner_confirmation_text": NEXT_CONFIRMATION,
    }
    report["report_digest_sha256"] = sha256_json(report)
    return report
