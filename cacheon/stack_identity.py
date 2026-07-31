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
from typing import Any


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical UTF-8 JSON encoding accepted for stack data."""

    canonical = _canonical_value(value)
    try:
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:  # defensive: validation owns failures
        raise StackIdentityError("value is not canonical JSON data") from exc
    return encoded.encode("utf-8")


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
        canonical_json_bytes(
            {
                "domain": domain,
                "payload": payload,
                "schema_version": schema_version,
            }
        )
    )
