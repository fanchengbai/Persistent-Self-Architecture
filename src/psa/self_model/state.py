from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_json


SELF_STATE_VERSION = "0.1"
SELF_FIELDS = (
    "identity_anchors",
    "preferences",
    "capability_estimate",
    "active_goals",
    "confidence",
    "uncertainty_conflicts",
)
FIELD_UPDATE_CLASSES = {
    "identity_anchors": {"protected"},
    "preferences": {"slow"},
    "capability_estimate": {"slow"},
    "active_goals": {"fast"},
    "confidence": {"fast"},
    "uncertainty_conflicts": {"fast"},
}
_STATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VALUE_TYPES = {"string", "number", "boolean", "string_list", "object"}
_STATUSES = {"active", "inactive", "resolved"}


def _payload(state: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(state))
    integrity = value.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("payload_sha256", None)
    return value


def self_state_digest(state: Mapping[str, Any]) -> str:
    return sha256_json(_payload(state))


def _value_matches_type(value: Any, value_type: str) -> bool:
    if value_type == "string":
        return isinstance(value, str)
    if value_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type == "string_list":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    return isinstance(value, dict) if value_type == "object" else False


def validate_self_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("Self State must be an object")
    required = {
        "schema_version",
        "state_id",
        "parent_state_id",
        "agent_instance_id",
        "trajectory_id",
        "step",
        "static_phase3",
        *SELF_FIELDS,
        "provenance_refs",
        "integrity",
    }
    if set(state) != required:
        raise ValueError("Self State top-level fields do not match v0.1")
    if (
        state.get("schema_version") != SELF_STATE_VERSION
        or not isinstance(state.get("state_id"), str)
        or _STATE_ID.fullmatch(str(state.get("state_id"))) is None
        or not (
            state.get("parent_state_id") is None
            or isinstance(state.get("parent_state_id"), str)
        )
        or not isinstance(state.get("agent_instance_id"), str)
        or not state.get("agent_instance_id")
        or not isinstance(state.get("trajectory_id"), str)
        or not state.get("trajectory_id")
        or not isinstance(state.get("step"), int)
        or state.get("step") < 0
        or state.get("static_phase3") is not True
        or not isinstance(state.get("provenance_refs"), list)
        or not all(isinstance(item, str) and item for item in state["provenance_refs"])
    ):
        raise ValueError("Self State metadata is invalid")
    item_ids: set[str] = set()
    for field in SELF_FIELDS:
        items = state.get(field)
        if not isinstance(items, list):
            raise ValueError(f"Self State {field} must be an array")
        for item in items:
            if not isinstance(item, Mapping) or set(item) != {
                "field_item_id",
                "value",
                "value_type",
                "confidence",
                "update_class",
                "created_step",
                "updated_step",
                "source_evidence_ids",
                "status",
            }:
                raise ValueError(f"Self State {field} item is invalid")
            item_id = item.get("field_item_id")
            value_type = item.get("value_type")
            confidence = item.get("confidence")
            created = item.get("created_step")
            updated = item.get("updated_step")
            evidence = item.get("source_evidence_ids")
            if (
                not isinstance(item_id, str)
                or not item_id
                or item_id in item_ids
                or value_type not in _VALUE_TYPES
                or not _value_matches_type(item.get("value"), str(value_type))
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0
                or item.get("update_class") not in FIELD_UPDATE_CLASSES[field]
                or not isinstance(created, int)
                or not isinstance(updated, int)
                or created < 0
                or updated < created
                or updated > state["step"]
                or not isinstance(evidence, list)
                or not all(isinstance(value, str) and value for value in evidence)
                or item.get("status") not in _STATUSES
            ):
                raise ValueError(f"Self State {field}/{item_id} violates its contract")
            item_ids.add(item_id)
    integrity = state.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"model_id", "tokenizer_id", "payload_sha256"}
        or not isinstance(integrity.get("model_id"), str)
        or not integrity.get("model_id")
        or not isinstance(integrity.get("tokenizer_id"), str)
        or not integrity.get("tokenizer_id")
        or integrity.get("payload_sha256") != self_state_digest(state)
    ):
        raise ValueError("Self State integrity check failed")
    return copy.deepcopy(dict(state))


def build_self_state(
    *,
    state_id: str,
    agent_instance_id: str,
    trajectory_id: str,
    step: int,
    model_id: str,
    tokenizer_id: str,
    fields: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_state_id: str | None = None,
    provenance_refs: Sequence[str] = (),
) -> dict[str, Any]:
    unknown = set(fields) - set(SELF_FIELDS)
    if unknown:
        raise ValueError(f"unknown Self fields: {sorted(unknown)}")
    state: dict[str, Any] = {
        "schema_version": SELF_STATE_VERSION,
        "state_id": state_id,
        "parent_state_id": parent_state_id,
        "agent_instance_id": agent_instance_id,
        "trajectory_id": trajectory_id,
        "step": step,
        "static_phase3": True,
        **{field: [copy.deepcopy(dict(item)) for item in fields.get(field, ())] for field in SELF_FIELDS},
        "provenance_refs": list(provenance_refs),
        "integrity": {
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "payload_sha256": "",
        },
    }
    state["integrity"]["payload_sha256"] = self_state_digest(state)
    return validate_self_state(state)


def swap_self_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    fields: Sequence[str],
    left_state_id: str,
    right_state_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_valid = validate_self_state(left)
    right_valid = validate_self_state(right)
    if (
        left_valid["integrity"]["model_id"]
        != right_valid["integrity"]["model_id"]
        or left_valid["integrity"]["tokenizer_id"]
        != right_valid["integrity"]["tokenizer_id"]
    ):
        raise ValueError("Self field swap requires model/tokenizer compatibility")
    selected = tuple(dict.fromkeys(fields))
    if not selected or not set(selected) <= set(SELF_FIELDS):
        raise ValueError("Self field swap mask is invalid")
    swapped_left = copy.deepcopy(left_valid)
    swapped_right = copy.deepcopy(right_valid)
    for field in selected:
        swapped_left[field] = copy.deepcopy(right_valid[field])
        swapped_right[field] = copy.deepcopy(left_valid[field])
    for value, source, state_id in (
        (swapped_left, left_valid, left_state_id),
        (swapped_right, right_valid, right_state_id),
    ):
        value["state_id"] = state_id
        value["parent_state_id"] = source["state_id"]
        value["step"] = max(left_valid["step"], right_valid["step"])
        value["provenance_refs"] = list(value["provenance_refs"]) + [
            f"offline-field-swap:{','.join(selected)}"
        ]
        value["integrity"]["payload_sha256"] = self_state_digest(value)
        validate_self_state(value)
    return swapped_left, swapped_right


class SelfStore:
    """Immutable JSON snapshot store; Phase 3 v0.1 has no update method."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def save(self, state: Mapping[str, Any]) -> Path:
        validated = validate_self_state(state)
        path = self.root / f"{validated['state_id']}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json_bytes(validated))
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def load(self, state_id: str) -> dict[str, Any]:
        if _STATE_ID.fullmatch(state_id) is None:
            raise ValueError("Self State ID is invalid")
        value = json.loads((self.root / f"{state_id}.json").read_text(encoding="utf-8"))
        return validate_self_state(value)
