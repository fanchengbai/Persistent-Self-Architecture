from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from importlib import import_module
import itertools
import json
from pathlib import Path
import random
import shutil
from typing import Any, Callable, Mapping, Sequence

from psa.artifacts import (
    canonical_json_bytes,
    payload_digest,
    sha256_file,
    sha256_json,
)
from psa.model import load_model_config
from psa.preregistration.finalize import (
    verify_final_preregistration_package,
)
from psa.preregistration.formal_freeze import (
    _fit_filler,
    _load_formal_config,
    _render_history,
    _render_query,
    _validate_seed_lock,
)
from psa.tasks import generate_factorial_group


CORE_SET_STATUS = "core_set_frozen_unrun"
EXPECTED_EXPERIMENT_ID = "EXP-001"
EXPECTED_FINAL_DIGEST = (
    "0daf056dc6b38aa20fa69dd9e8df9b8065876529947cbc01353ffe604933d0c9"
)
EXPECTED_CANDIDATE_DIGEST = (
    "a354b208be0640da7ea70fe070f75bdec69186e496ba1cc14c3157dcd984e6cd"
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def _validate_authorization(
    authorization: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
) -> None:
    if authorization.get("authorization_version") != "1.0":
        raise ValueError("unsupported Core Set authorization version")
    if authorization.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("Core Set authorization is for another experiment")
    if authorization.get("final_preregistration_digest_sha256") != (
        final_manifest.get("final_preregistration_digest_sha256")
    ):
        raise ValueError("Core Set authorization final digest does not match")
    if authorization.get("candidate_digest_sha256") != (
        final_manifest.get("candidate_digest_sha256")
    ):
        raise ValueError("Core Set authorization candidate digest does not match")
    if authorization.get("authorized_by_role") != "project_owner":
        raise ValueError("Core Set authorization must come from project owner")
    _require_utc_timestamp(
        authorization.get("authorized_at_utc"),
        "authorized_at_utc",
    )
    text = authorization.get("authorization_text")
    if (
        not isinstance(text, str)
        or "Core Set" not in text
        or "暂不运行正式实验" not in text
    ):
        raise ValueError(
            "Core Set authorization text must preserve the granted scope"
        )
    expected_scope = {
        "generate_and_freeze_core_set": True,
        "run_confirmatory_experiment": False,
    }
    if authorization.get("authorization") != expected_scope:
        raise ValueError("Core Set authorization scope is invalid")
    if authorization.get("factorial_group_count") != 320:
        raise ValueError("Core Set authorization must freeze 320 groups")


def _load_token_counter(
    config: Mapping[str, Any],
    root: Path,
) -> tuple[Callable[[str], int], dict[str, Any]]:
    model_config_path = (root / config["model_config"]).resolve()
    model_config = load_model_config(
        model_config_path,
        root,
        verify_files=False,
    )
    tokenizer_path = model_config.tokenizer_path
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer is missing: {tokenizer_path}")
    if tokenizer_path.stat().st_size != model_config.tokenizer_size_bytes:
        raise ValueError("tokenizer size does not match frozen model config")
    if sha256_file(tokenizer_path) != model_config.tokenizer_sha256:
        raise ValueError("tokenizer digest does not match frozen model config")
    tokenizer_module = import_module("rwkv.rwkv_tokenizer")
    tokenizer = tokenizer_module.TRIE_TOKENIZER(str(tokenizer_path))
    return (
        lambda text: len(list(tokenizer.encode(text))),
        {
            "path": str(tokenizer_path.relative_to(root)).replace("\\", "/"),
            "revision": model_config.tokenizer_revision,
            "sha256": model_config.tokenizer_sha256,
            "size_bytes": model_config.tokenizer_size_bytes,
        },
    )


def _generate_core_set(
    config: Mapping[str, Any],
    *,
    token_counter: Callable[[str], int],
    tokenizer_provenance: Mapping[str, Any],
    generated_at_utc: str,
    source_config: Mapping[str, Any],
    final_preregistration_digest_sha256: str,
) -> dict[str, Any]:
    _validate_seed_lock(config)
    core_design = config["core_design"]
    answer = config["answer_interface"]
    labels = config["labels"]
    histories = config["history_protocol"]["templates"]
    queries = config["query_protocol"]["templates"]
    group_count = int(core_design["factorial_group_count"])
    if (
        group_count != 320
        or int(core_design["states_per_group"]) != 4
        or int(core_design["history_query_pair_count"]) != 16
        or int(core_design["groups_per_history_query_pair"]) != 20
    ):
        raise ValueError("Core Set design does not match frozen D4-D8 values")
    if config["history_protocol"]["mode"] != "single_statement":
        raise ValueError("Core Set history mode is not frozen")
    codes = tuple(answer["answer_codes"])
    if len(codes) != 4 or len(set(codes)) != 4:
        raise ValueError("Core Set answer codes must contain A-D")
    if int(answer["rotation_count"]) != 4:
        raise ValueError("Core Set requires four answer-code rotations")
    identity_pairs = tuple(
        tuple(pair) for pair in labels["identity_label_pairs"]
    )
    goal_pairs = tuple(tuple(pair) for pair in labels["goal_label_pairs"])
    label_pair_pool = tuple(itertools.product(identity_pairs, goal_pairs))
    if len(label_pair_pool) != 4:
        raise ValueError("Core Set requires four label-pair combinations")
    fillers = [
        _fit_filler(
            variant_index=index,
            filler_config=config["filler_protocol"],
            token_counter=token_counter,
        )
        for index in range(4)
    ]
    if {item["token_count"] for item in fillers} != {131}:
        raise ValueError("Core Set filler token count is not frozen at 131")
    rng = random.Random(int(config["seeds"]["core_generator"]))
    groups: list[dict[str, Any]] = []
    global_group_index = 0
    for history_index, history_template in enumerate(histories):
        for query_index, query_template in enumerate(queries):
            for pair_group_index in range(20):
                label_pool_index = pair_group_index % len(label_pair_pool)
                identity_pair, goal_pair = label_pair_pool[label_pool_index]
                filler = fillers[pair_group_index % len(fillers)]
                group_seed = rng.getrandbits(63)
                group_id = "coregrp-" + sha256_json(
                    {
                        "core_generator_seed": config["seeds"][
                            "core_generator"
                        ],
                        "group_index": global_group_index,
                        "history_template_id": history_template["id"],
                        "query_template_id": query_template["id"],
                        "group_seed": group_seed,
                        "identity_pair": identity_pair,
                        "goal_pair": goal_pair,
                        "filler_variant_id": filler["variant_id"],
                    }
                )[:24]
                trials: list[dict[str, Any]] = []
                for rotation_index in range(4):
                    rotated_codes = codes[rotation_index:] + codes[:rotation_index]
                    generated_group = generate_factorial_group(
                        group_seed=group_seed,
                        track="synthetic",
                        identity_labels=identity_pair,
                        goal_labels=goal_pair,
                        answer_codes=rotated_codes,
                        delay_units=0,
                        generator_version="exp001-core-set-v1",
                        history_order="I_G",
                    )
                    option_mapping = [
                        {
                            "code": option.code,
                            "domain": generated_group.identity_labels[
                                option.identity
                            ],
                            "operation": generated_group.goal_labels[
                                option.goal
                            ],
                            "identity": option.identity,
                            "goal": option.goal,
                        }
                        for option in generated_group.options
                    ]
                    query_prompt = _render_query(
                        query_template,
                        option_mapping=option_mapping,
                        assistant_prefix=answer["assistant_prefix"],
                    )
                    query_before_options = query_prompt.split(
                        "OPTIONS:",
                        maxsplit=1,
                    )[0]
                    for sample in generated_group.trajectories:
                        domain = generated_group.identity_labels[
                            sample.identity
                        ]
                        operation = generated_group.goal_labels[sample.goal]
                        if (
                            domain in query_before_options
                            or operation in query_before_options
                        ):
                            raise ValueError(
                                "Core Set state-only query leaks current values"
                            )
                        history_prompt = _render_history(
                            history_template,
                            domain=domain,
                            operation=operation,
                            filler=filler["text"],
                        )
                        semantic_case_id = "corecase-" + sha256_json(
                            {
                                "factorial_group_id": group_id,
                                "identity": sample.identity,
                                "goal": sample.goal,
                            }
                        )[:24]
                        trial_id = "coretrial-" + sha256_json(
                            {
                                "semantic_case_id": semantic_case_id,
                                "rotation_index": rotation_index,
                                "correct_code": sample.correct_code,
                                "history_prompt": history_prompt,
                                "query_prompt": query_prompt,
                            }
                        )[:24]
                        trials.append(
                            {
                                "trial_id": trial_id,
                                "semantic_case_id": semantic_case_id,
                                "rotation_index": rotation_index,
                                "history_prompt": history_prompt,
                                "history_prompt_digest_sha256": sha256_json(
                                    history_prompt
                                ),
                                "query_prompt": query_prompt,
                                "query_prompt_digest_sha256": sha256_json(
                                    query_prompt
                                ),
                                "target_code": sample.correct_code,
                                "target_fields": {
                                    "domain": domain,
                                    "operation": operation,
                                    "identity": sample.identity,
                                    "goal": sample.goal,
                                },
                                "option_mapping": option_mapping,
                            }
                        )
                groups.append(
                    {
                        "factorial_group_id": group_id,
                        "group_index": global_group_index,
                        "group_seed": group_seed,
                        "history_template_id": history_template["id"],
                        "query_template_id": query_template["id"],
                        "filler_variant_id": filler["variant_id"],
                        "identity_labels": list(identity_pair),
                        "goal_labels": list(goal_pair),
                        "rotation_count": 4,
                        "semantic_case_count": 4,
                        "trial_count": len(trials),
                        "trials": trials,
                    }
                )
                global_group_index += 1
    core_set = {
        "core_set_version": "1.0",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": CORE_SET_STATUS,
        "generated_at_utc": generated_at_utc,
        "final_preregistration_digest_sha256": (
            final_preregistration_digest_sha256
        ),
        "source_config": dict(source_config),
        "tokenizer": dict(tokenizer_provenance),
        "generator_version": "exp001-core-set-v1",
        "core_generator_seed": config["seeds"]["core_generator"],
        "factorial_group_count": len(groups),
        "states_per_group": 4,
        "rotation_count": 4,
        "semantic_case_count": len(groups) * 4,
        "trial_count": len(groups) * 16,
        "conditions": list(core_design["conditions"]),
        "history_template_ids": [item["id"] for item in histories],
        "query_template_ids": [item["id"] for item in queries],
        "filler_variants": fillers,
        "answer_codes": list(codes),
        "confirmatory_results_observed": False,
        "confirmatory_experiment_run": False,
        "groups": groups,
    }
    core_set["core_set_digest_sha256"] = sha256_json(core_set)
    _validate_core_set(core_set)
    return core_set


def _validate_core_set(core_set: Mapping[str, Any]) -> None:
    expected_digest = core_set.get("core_set_digest_sha256")
    unsigned = dict(core_set)
    unsigned.pop("core_set_digest_sha256", None)
    if not isinstance(expected_digest, str) or sha256_json(unsigned) != (
        expected_digest
    ):
        raise ValueError("Core Set self digest is invalid")
    groups = core_set.get("groups")
    if not isinstance(groups, list) or len(groups) != 320:
        raise ValueError("Core Set must contain 320 factorial groups")
    if (
        core_set.get("status") != CORE_SET_STATUS
        or core_set.get("factorial_group_count") != 320
        or core_set.get("semantic_case_count") != 1280
        or core_set.get("trial_count") != 5120
        or core_set.get("confirmatory_results_observed") is not False
        or core_set.get("confirmatory_experiment_run") is not False
    ):
        raise ValueError("Core Set top-level safety or count fields are invalid")
    group_ids: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    history_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    filler_counts: Counter[str] = Counter()
    label_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    for group in groups:
        group_id = group.get("factorial_group_id")
        if not isinstance(group_id, str) or group_id in group_ids:
            raise ValueError("Core Set factorial group IDs are not unique")
        group_ids.add(group_id)
        history_id = group["history_template_id"]
        query_id = group["query_template_id"]
        pair_counts[(history_id, query_id)] += 1
        history_counts[history_id] += 1
        query_counts[query_id] += 1
        filler_counts[group["filler_variant_id"]] += 1
        label_counts[
            (
                tuple(group["identity_labels"]),
                tuple(group["goal_labels"]),
            )
        ] += 1
        trials = group.get("trials")
        if (
            not isinstance(trials, list)
            or len(trials) != 16
            or group.get("rotation_count") != 4
            or group.get("semantic_case_count") != 4
            or group.get("trial_count") != 16
        ):
            raise ValueError("Core Set factorial group is incomplete")
        by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for trial in trials:
            by_case[trial["semantic_case_id"]].append(trial)
            target = trial["target_fields"]
            prefix = trial["query_prompt"].split("OPTIONS:", maxsplit=1)[0]
            if target["domain"] in prefix or target["operation"] in prefix:
                raise ValueError("Core Set query leaks a target field")
        if len(by_case) != 4:
            raise ValueError("Core Set group does not contain four states")
        for rotations in by_case.values():
            if {item["rotation_index"] for item in rotations} != set(range(4)):
                raise ValueError("Core Set semantic case lacks four rotations")
            if {item["target_code"] for item in rotations} != {
                "A",
                "B",
                "C",
                "D",
            }:
                raise ValueError("Core Set code rotation is not balanced")
    if set(pair_counts.values()) != {20} or len(pair_counts) != 16:
        raise ValueError("Core Set history-query pairs are not 20-way balanced")
    if set(history_counts.values()) != {80} or len(history_counts) != 4:
        raise ValueError("Core Set history templates are not balanced")
    if set(query_counts.values()) != {80} or len(query_counts) != 4:
        raise ValueError("Core Set query templates are not balanced")
    if set(filler_counts.values()) != {80} or len(filler_counts) != 4:
        raise ValueError("Core Set fillers are not balanced")
    if set(label_counts.values()) != {80} or len(label_counts) != 4:
        raise ValueError("Core Set label pairs are not balanced")


def _build_package_manifest(
    *,
    core_set: Mapping[str, Any],
    authorization: Mapping[str, Any],
    locked_files: Mapping[str, str],
) -> dict[str, Any]:
    manifest = {
        "package_version": "1.0",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": CORE_SET_STATUS,
        "frozen_at_utc": authorization["authorized_at_utc"],
        "final_preregistration_digest_sha256": (
            core_set["final_preregistration_digest_sha256"]
        ),
        "core_set_digest_sha256": core_set["core_set_digest_sha256"],
        "factorial_group_count": core_set["factorial_group_count"],
        "semantic_case_count": core_set["semantic_case_count"],
        "trial_count": core_set["trial_count"],
        "authorization": authorization["authorization"],
        "safety_boundary": {
            "core_set_generated": True,
            "core_set_frozen": True,
            "confirmatory_experiment_authorized": False,
            "confirmatory_experiment_run": False,
            "confirmatory_results_observed": False,
        },
        "locked_files": dict(locked_files),
        "package_payload_root_digest_sha256": payload_digest(locked_files),
    }
    manifest["core_set_package_digest_sha256"] = sha256_json(manifest)
    return manifest


def verify_core_set_package(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    manifest = _load_object(root / "manifest.json", "Core Set manifest")
    expected_manifest_digest = manifest.get(
        "core_set_package_digest_sha256"
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("core_set_package_digest_sha256", None)
    manifest_digest_valid = bool(
        isinstance(expected_manifest_digest, str)
        and sha256_json(unsigned_manifest) == expected_manifest_digest
    )
    locked_files = manifest.get("locked_files")
    locked_file_checks: dict[str, bool] = {}
    if isinstance(locked_files, dict):
        for filename, expected in locked_files.items():
            path = (root / filename).resolve()
            locked_file_checks[filename] = bool(
                root in path.parents
                and path.is_file()
                and sha256_file(path) == expected
            )
    payload_root_valid = bool(
        isinstance(locked_files, dict)
        and payload_digest(locked_files)
        == manifest.get("package_payload_root_digest_sha256")
    )
    content_valid = False
    try:
        core_set = _load_object(root / "core_set.json", "Core Set")
        authorization = _load_object(
            root / "core_set_authorization.json",
            "Core Set authorization",
        )
        final_manifest = _load_object(
            root / "final_preregistration_manifest.json",
            "final preregistration manifest",
        )
        _validate_core_set(core_set)
        _validate_authorization(authorization, final_manifest)
        content_valid = bool(
            final_manifest.get("final_preregistration_digest_sha256")
            == EXPECTED_FINAL_DIGEST
            and core_set.get("final_preregistration_digest_sha256")
            == EXPECTED_FINAL_DIGEST
            and manifest.get("core_set_digest_sha256")
            == core_set.get("core_set_digest_sha256")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        content_valid = False
    safety = manifest.get("safety_boundary")
    safety_boundary_valid = bool(
        isinstance(safety, dict)
        and safety.get("core_set_generated") is True
        and safety.get("core_set_frozen") is True
        and safety.get("confirmatory_experiment_authorized") is False
        and safety.get("confirmatory_experiment_run") is False
        and safety.get("confirmatory_results_observed") is False
        and manifest.get("authorization")
        == {
            "generate_and_freeze_core_set": True,
            "run_confirmatory_experiment": False,
        }
    )
    valid = bool(
        manifest.get("status") == CORE_SET_STATUS
        and manifest_digest_valid
        and locked_file_checks
        and all(locked_file_checks.values())
        and payload_root_valid
        and content_valid
        and safety_boundary_valid
    )
    return {
        "report_version": "1.0",
        "package_dir": str(root),
        "status": manifest.get("status"),
        "final_preregistration_digest_sha256": manifest.get(
            "final_preregistration_digest_sha256"
        ),
        "core_set_digest_sha256": manifest.get("core_set_digest_sha256"),
        "core_set_package_digest_sha256": expected_manifest_digest,
        "factorial_group_count": manifest.get("factorial_group_count"),
        "semantic_case_count": manifest.get("semantic_case_count"),
        "trial_count": manifest.get("trial_count"),
        "manifest_digest_valid": manifest_digest_valid,
        "locked_file_checks": locked_file_checks,
        "payload_root_valid": payload_root_valid,
        "content_valid": content_valid,
        "safety_boundary_valid": safety_boundary_valid,
        "confirmatory_experiment_run": False,
        "confirmatory_results_observed": False,
        "valid": valid,
    }


def generate_and_freeze_core_set(
    *,
    final_package_dir: str | Path,
    authorization_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
    token_counter: Callable[[str], int] | None = None,
    tokenizer_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    final_root = Path(final_package_dir).resolve()
    final_report = verify_final_preregistration_package(final_root)
    if not final_report["valid"]:
        raise ValueError("final preregistration package is invalid")
    final_manifest_path = final_root / "manifest.json"
    final_manifest = _load_object(
        final_manifest_path,
        "final preregistration manifest",
    )
    if (
        final_manifest.get("final_preregistration_digest_sha256")
        != EXPECTED_FINAL_DIGEST
        or final_manifest.get("candidate_digest_sha256")
        != EXPECTED_CANDIDATE_DIGEST
    ):
        raise ValueError("final preregistration package is not EXP-001 v1")
    authorization_source = Path(authorization_path).resolve()
    authorization = _load_object(
        authorization_source,
        "Core Set authorization",
    )
    _validate_authorization(authorization, final_manifest)
    candidate = _load_object(final_root / "candidate.json", "candidate")
    formal_config_path = Path(config_path).resolve()
    source_config = candidate["source_config"]
    if (
        sha256_file(formal_config_path) != source_config["sha256"]
        or str(formal_config_path.relative_to(root)).replace("\\", "/")
        != source_config["path"]
    ):
        raise ValueError("formal config no longer matches frozen candidate")
    config = _load_formal_config(formal_config_path, root)
    if (
        config["core_design"]["conditions"] != candidate["conditions"]
        or config["core_design"]["factorial_group_count"]
        != candidate["factorial_group_count"]
        or config["seeds"] != candidate["seeds"]
        or config["statistics"] != candidate["statistics"]
    ):
        raise ValueError("resolved Core Set design differs from candidate")
    if token_counter is None:
        token_counter, observed_tokenizer = _load_token_counter(config, root)
        tokenizer_provenance = observed_tokenizer
    elif tokenizer_provenance is None:
        tokenizer_provenance = {
            "path": "test-tokenizer",
            "revision": "test",
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    core_set = _generate_core_set(
        config,
        token_counter=token_counter,
        tokenizer_provenance=tokenizer_provenance,
        generated_at_utc=authorization["authorized_at_utc"],
        source_config=source_config,
        final_preregistration_digest_sha256=EXPECTED_FINAL_DIGEST,
    )
    destination = Path(output_dir).resolve()
    if destination.exists():
        existing = verify_core_set_package(destination)
        if (
            existing["valid"]
            and existing["core_set_digest_sha256"]
            == core_set["core_set_digest_sha256"]
        ):
            return existing
        raise ValueError("Core Set output already exists")
    destination.mkdir(parents=True)
    (destination / "core_set.json").write_bytes(
        canonical_json_bytes(core_set)
    )
    shutil.copyfile(
        authorization_source,
        destination / "core_set_authorization.json",
    )
    shutil.copyfile(
        final_manifest_path,
        destination / "final_preregistration_manifest.json",
    )
    locked_files = {
        filename: sha256_file(destination / filename)
        for filename in (
            "core_set.json",
            "core_set_authorization.json",
            "final_preregistration_manifest.json",
        )
    }
    package_manifest = _build_package_manifest(
        core_set=core_set,
        authorization=authorization,
        locked_files=locked_files,
    )
    (destination / "manifest.json").write_bytes(
        canonical_json_bytes(package_manifest)
    )
    report = verify_core_set_package(destination)
    if not report["valid"]:
        raise RuntimeError("Core Set package failed final self-check")
    return report
