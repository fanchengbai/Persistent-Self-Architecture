from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Mapping, Sequence

from psa.artifacts import canonical_json_bytes, sha256_json
from psa.self_model.state import SELF_FIELDS, validate_self_state


@dataclass(frozen=True)
class EncodedSelf:
    encoder_version: str
    state_digest_sha256: str
    dimension: int
    active_fields: tuple[str, ...]
    field_vectors: Mapping[str, tuple[float, ...]]
    aggregate_vector: tuple[float, ...]
    offline_fake_encoder: bool
    model_loaded: bool
    prompt_serialization_used: bool


def _hash_vector(payload: object, dimension: int) -> tuple[float, ...]:
    values = []
    counter = 0
    seed = canonical_json_bytes(payload)
    while len(values) < dimension:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(digest), 4):
            integer = int.from_bytes(digest[offset : offset + 4], "big")
            values.append((integer / 2**32) * 2.0 - 1.0)
            if len(values) == dimension:
                break
        counter += 1
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


class DeterministicHashFakeSelfEncoder:
    """Offline interface fixture. It is not a trained neural Self Encoder."""

    offline_fake_encoder = True
    model_loaded = False
    prompt_serialization_used = False

    def __init__(self, dimension: int = 16) -> None:
        if not isinstance(dimension, int) or dimension < 4:
            raise ValueError("Self Encoder dimension must be at least four")
        self.dimension = dimension

    def encode(
        self,
        state: Mapping[str, object],
        *,
        field_mask: Sequence[str] | None = None,
    ) -> EncodedSelf:
        validated = validate_self_state(state)
        requested = tuple(field_mask) if field_mask is not None else SELF_FIELDS
        if (
            not requested
            or len(requested) != len(set(requested))
            or not set(requested) <= set(SELF_FIELDS)
        ):
            raise ValueError("Self Encoder field mask is empty, duplicated, or invalid")
        active = tuple(field for field in SELF_FIELDS if field in requested)
        vectors = {
            field: _hash_vector(
                {
                    "encoder": "deterministic-hash-fake-v0.1",
                    "field": field,
                    "items": validated[field],
                },
                self.dimension,
            )
            for field in active
        }
        aggregate = tuple(
            sum(vectors[field][index] for field in active) / len(active)
            for index in range(self.dimension)
        )
        return EncodedSelf(
            encoder_version="0.1-offline-fake",
            state_digest_sha256=validated["integrity"]["payload_sha256"],
            dimension=self.dimension,
            active_fields=active,
            field_vectors=vectors,
            aggregate_vector=aggregate,
            offline_fake_encoder=True,
            model_loaded=False,
            prompt_serialization_used=False,
        )


def encoded_self_digest(encoded: EncodedSelf) -> str:
    return sha256_json(
        {
            "encoder_version": encoded.encoder_version,
            "state_digest_sha256": encoded.state_digest_sha256,
            "dimension": encoded.dimension,
            "active_fields": list(encoded.active_fields),
            "field_vectors": {key: list(value) for key, value in encoded.field_vectors.items()},
            "aggregate_vector": list(encoded.aggregate_vector),
        }
    )


def randomize_encoded_fields(
    encoded: EncodedSelf,
    *,
    fields: Sequence[str],
    seed: int,
) -> EncodedSelf:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("encoded Self randomization seed must be non-negative")
    selected = set(fields)
    if not selected or not selected <= set(encoded.active_fields):
        raise ValueError("encoded Self randomization mask is invalid")
    vectors = dict(encoded.field_vectors)
    for field in sorted(selected):
        source = vectors[field]
        source_norm = math.sqrt(sum(value * value for value in source))
        generator = random.Random(f"PSA|Self-v0.1|{seed}|{field}")
        values = [generator.gauss(0.0, 1.0) for _ in source]
        center = sum(values) / len(values)
        values = [value - center for value in values]
        norm = math.sqrt(sum(value * value for value in values))
        vectors[field] = tuple(value * source_norm / norm for value in values)
    aggregate = tuple(
        sum(vectors[field][index] for field in encoded.active_fields)
        / len(encoded.active_fields)
        for index in range(encoded.dimension)
    )
    return EncodedSelf(
        encoder_version=encoded.encoder_version + "+randomized",
        state_digest_sha256=encoded.state_digest_sha256,
        dimension=encoded.dimension,
        active_fields=encoded.active_fields,
        field_vectors=vectors,
        aggregate_vector=aggregate,
        offline_fake_encoder=True,
        model_loaded=False,
        prompt_serialization_used=False,
    )
