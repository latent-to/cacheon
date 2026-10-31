"""Strict content identities for evaluation-stack composition.

This module is intentionally small and standard-library only.  It provides the
single canonical JSON and domain-separation rule used by stack manifests,
catalog snapshots, and pure planning records.  It does not accept Python's
lossy or process-local JSON extensions (notably floats, non-string keys, sets,
or arbitrary objects).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_JSON_SCALARS = frozenset((str, int, bool, type(None)))
_JSON_ARRAYS = frozenset((list, tuple))
_JSON_KEYS = frozenset((str,))


class StackIdentityError(ValueError):
    """A value cannot participate in a canonical stack identity."""


def _canonical_value(value: object, *, location: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise StackIdentityError(f"{location} contains a float")
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StackIdentityError(
                    f"{location} contains a non-string object key"
                )
            canonical[key] = _canonical_value(
                item, location=f"{location}.{key}"
            )
        return canonical
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [
            _canonical_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise StackIdentityError(
        f"{location} contains unsupported {type(value).__name__}"
    )


def _json_native(value: object) -> bool:
    """Whether value is already canonical data that json encodes as it is.

    That is dicts with str keys, lists and tuples, holding str, int, bool and None,
    with types compared exactly. The checked walk above builds a location string per
    element so its error can name the first violation; a DeepSeek PD speed record has
    about 170 million elements, and each walk cost minutes of idle GPUs (2026-10-30).
    """
    pending, seen = [value], set()
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is dict:
            if not set(map(type, item)) <= _JSON_KEYS:
                return False
            children = item.values()
        elif kind in _JSON_ARRAYS:
            children = item
        elif kind in _JSON_SCALARS:
            continue
        else:
            return False
        if id(item) in seen:
            continue
        seen.add(id(item))
        kinds = set(map(type, children))
        if kinds <= _JSON_SCALARS or (
            kinds <= _JSON_ARRAYS and set(map(type, chain.from_iterable(children))) <= _JSON_SCALARS
        ):
            continue
        pending.extend(child for child in children if type(child) not in _JSON_SCALARS)
    return True


def _dumps(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical UTF-8 JSON encoding accepted for stack data."""

    if _json_native(value):
        try:
            return _dumps(value).encode("utf-8")
        except ValueError:
            pass  # a cycle or an unpaired surrogate: the checked walk raises as it always has
    canonical = _canonical_value(value)
    try:
        encoded = _dumps(canonical)
    except (TypeError, ValueError) as exc:  # defensive: validation owns failures
        raise StackIdentityError("value is not canonical JSON data") from exc
    return encoded.encode("utf-8")


def canonical_object_bytes(members: Mapping[str, bytes]) -> bytes:
    """Join members' canonical encodings into the canonical encoding of their object.

    Canonical JSON sorts keys at every level and has no whitespace, so an object's
    encoding is its sorted members' encodings joined: a large member is encoded once
    and its bytes reused. Every value must be ``canonical_json_bytes`` output.
    """

    if not all(isinstance(key, str) for key in members):
        raise StackIdentityError("canonical object keys must be strings")
    return b"{" + b",".join(
        _dumps(key).encode("utf-8") + b":" + members[key] for key in sorted(members)
    ) + b"}"


def sha256_hex(data: bytes) -> str:
    """Hash exact bytes using lowercase SHA-256."""

    if not isinstance(data, bytes):
        raise TypeError("sha256_hex data must be bytes")
    return hashlib.sha256(data).hexdigest()


def require_sha256_hex(value: object, *, field: str = "digest") -> str:
    """Validate and return one canonical lowercase SHA-256 digest."""

    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise StackIdentityError(
            f"{field} must be a lowercase 64-hex SHA-256 digest"
        )
    return value


def canonical_digest(
    domain: str,
    payload: object,
    *,
    schema_version: int = 1,
) -> str:
    """Hash canonical data in an explicit schema and semantic domain.

    The envelope prevents an equal JSON payload used for two different concepts
    from acquiring the same identity.  Domain strings and schema versions are
    themselves strict identity inputs rather than caller-controlled decoration.
    """

    return canonical_payload_digest(
        domain, canonical_json_bytes(payload), schema_version=schema_version
    )


def canonical_payload_digest(
    domain: str,
    payload: bytes,
    *,
    schema_version: int = 1,
) -> str:
    """``canonical_digest`` of a payload already encoded by ``canonical_json_bytes``."""

    if not isinstance(domain, str) or not _DOMAIN_RE.fullmatch(domain):
        raise StackIdentityError(
            "digest domain must match [a-z0-9][a-z0-9._-]*"
        )
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise StackIdentityError("schema_version must be a positive integer")
    return sha256_hex(
        canonical_object_bytes(
            {
                "domain": canonical_json_bytes(domain),
                "payload": payload,
                "schema_version": canonical_json_bytes(schema_version),
            }
        )
    )
