"""Rotatable shared-secret credentials for eval → weights-service push.

The eval host must not hold the chain-signing wallet path for weight
publication. It authenticates to ``serve-weights`` with HMAC credentials that
operators can rotate by adding a new active secret and retiring the old id.

This module is original Cacheon code (Apache-2.0). It does not vendor third-party
auth libraries.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


CREDENTIALS_SCHEMA = "cacheon.weight-push-credentials.v1"
PUSH_AUTH_DOMAIN = "cacheon.weight-share.push.v1"
PUSH_ACK_SCHEMA = "cacheon.weight-share.push-ack.v1"
PUSH_ACK_AUTH_DOMAIN = "cacheon.weight-share.push-ack-auth.v1"
LEGACY_CREDENTIALS_SCHEMA = "optima.weight-push-credentials.v1"
LEGACY_PUSH_AUTH_DOMAIN = "optima.weight-share.push.v1"
LEGACY_PUSH_ACK_SCHEMA = "optima.weight-share.push-ack.v1"
LEGACY_PUSH_ACK_AUTH_DOMAIN = "optima.weight-share.push-ack-auth.v1"
DEFAULT_MAX_SKEW_SECONDS = 60
# File path to a PushCredentialSet JSON (serve + eval).
ENV_PUSH_CREDENTIALS = "CACHEON_WEIGHT_PUSH_CREDENTIALS"
# Inline single active secret for eval (and optionally serve with one key).
ENV_PUSH_KEY = "CACHEON_WEIGHT_PUSH_KEY"
ENV_PUSH_CREDENTIAL_ID = "CACHEON_WEIGHT_PUSH_CREDENTIAL_ID"
LEGACY_ENV_PUSH_CREDENTIALS = "OPTIMA_WEIGHT_PUSH_CREDENTIALS"
LEGACY_ENV_PUSH_KEY = "OPTIMA_WEIGHT_PUSH_KEY"
LEGACY_ENV_PUSH_CREDENTIAL_ID = "OPTIMA_WEIGHT_PUSH_CREDENTIAL_ID"
DEFAULT_ENV_CREDENTIAL_ID = "env"


class WeightPushAuthError(RuntimeError):
    """Push credential or signature verification failed closed."""

    validator_fault = True
    retryable = False


@dataclass(frozen=True)
class PushCredential:
    credential_id: str
    secret: str
    status: str = "active"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.credential_id, str)
            or not self.credential_id
            or self.credential_id.strip() != self.credential_id
            or len(self.credential_id) > 128
            or "/" in self.credential_id
            or any(char in self.credential_id for char in " \t\r\n")
        ):
            raise WeightPushAuthError("push credential id is malformed")
        if (
            not isinstance(self.secret, str)
            or len(self.secret) < 32
            or len(self.secret) > 512
            or any(char in self.secret for char in "\x00\r\n")
        ):
            raise WeightPushAuthError("push credential secret is malformed")
        if self.status not in {"active", "retired"}:
            raise WeightPushAuthError("push credential status is unsupported")


@dataclass(frozen=True)
class PushCredentialSet:
    credentials: tuple[PushCredential, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.credentials)
        if not rows:
            raise WeightPushAuthError("push credential set is empty")
        if any(type(row) is not PushCredential for row in rows):
            raise WeightPushAuthError("push credentials are untyped")
        ids = tuple(row.credential_id for row in rows)
        if len(ids) != len(set(ids)):
            raise WeightPushAuthError("push credential ids must be unique")
        if not any(row.status == "active" for row in rows):
            raise WeightPushAuthError("push credential set has no active secret")
        object.__setattr__(self, "credentials", rows)

    def active(self) -> tuple[PushCredential, ...]:
        return tuple(row for row in self.credentials if row.status == "active")

    def get(self, credential_id: str) -> PushCredential | None:
        for row in self.credentials:
            if row.credential_id == credential_id:
                return row
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "credentials": [
                {
                    "credential_id": row.credential_id,
                    "secret": row.secret,
                    "status": row.status,
                }
                for row in self.credentials
            ],
            "schema": CREDENTIALS_SCHEMA,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PushCredentialSet":
        if type(value) is not dict or set(value) != {"credentials", "schema"}:
            raise WeightPushAuthError("push credential file fields do not match")
        if value["schema"] not in {CREDENTIALS_SCHEMA, LEGACY_CREDENTIALS_SCHEMA}:
            raise WeightPushAuthError("push credential schema is unsupported")
        raw = value["credentials"]
        if type(raw) is not list or not raw:
            raise WeightPushAuthError("push credentials array is malformed")
        rows: list[PushCredential] = []
        for item in raw:
            if type(item) is not dict or set(item) != {
                "credential_id",
                "secret",
                "status",
            }:
                raise WeightPushAuthError("push credential row fields do not match")
            rows.append(
                PushCredential(
                    str(item["credential_id"]),
                    str(item["secret"]),
                    str(item["status"]),
                )
            )
        return cls(tuple(rows))


def load_push_credentials(path: str | Path) -> PushCredentialSet:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WeightPushAuthError("push credential file is missing") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightPushAuthError(f"push credential file is unreadable: {exc}") from None
    return PushCredentialSet.from_dict(raw)


def push_credentials_from_env_key(
    *,
    secret: str | None = None,
    credential_id: str | None = None,
) -> PushCredentialSet:
    """Build a one-credential set from ``CACHEON_WEIGHT_PUSH_KEY`` (and optional id)."""

    import os

    cacheon_secret = os.environ.get(ENV_PUSH_KEY, "")
    uses_cacheon_environment = secret is None and bool(cacheon_secret)
    raw_secret = (
        secret
        if secret is not None
        else cacheon_secret or os.environ.get(LEGACY_ENV_PUSH_KEY, "")
    )
    if not isinstance(raw_secret, str) or not raw_secret:
        raise WeightPushAuthError(
            f"{ENV_PUSH_KEY} is unset or empty; provide --push-credentials, "
            f"{ENV_PUSH_CREDENTIALS}, or {ENV_PUSH_KEY}"
        )
    if secret is not None:
        environment_id = os.environ.get(
            ENV_PUSH_CREDENTIAL_ID, ""
        ) or os.environ.get(LEGACY_ENV_PUSH_CREDENTIAL_ID, "")
    elif uses_cacheon_environment:
        environment_id = os.environ.get(ENV_PUSH_CREDENTIAL_ID, "")
    else:
        environment_id = os.environ.get(LEGACY_ENV_PUSH_CREDENTIAL_ID, "")
    raw_id = (
        credential_id
        if credential_id is not None
        else environment_id or DEFAULT_ENV_CREDENTIAL_ID
    )
    return PushCredentialSet((PushCredential(str(raw_id), raw_secret, "active"),))


def resolve_push_credentials(
    path: str | Path | None = None,
    *,
    required: bool = False,
) -> PushCredentialSet | None:
    """Resolve push credentials: CLI path → file env → inline key env.

    Precedence:
    1. Explicit ``path`` / ``--push-credentials``
    2. ``CACHEON_WEIGHT_PUSH_CREDENTIALS`` (JSON file path)
    3. ``CACHEON_WEIGHT_PUSH_KEY`` (+ optional ``CACHEON_WEIGHT_PUSH_CREDENTIAL_ID``)
    4. The corresponding transitional ``OPTIMA_*`` aliases
    """

    import os

    explicit = str(path).strip() if path is not None else ""
    if explicit:
        return load_push_credentials(explicit)
    env_path = os.environ.get(ENV_PUSH_CREDENTIALS, "").strip()
    if env_path:
        return load_push_credentials(env_path)
    if os.environ.get(ENV_PUSH_KEY, "").strip():
        return push_credentials_from_env_key()
    legacy_env_path = os.environ.get(LEGACY_ENV_PUSH_CREDENTIALS, "").strip()
    if legacy_env_path:
        return load_push_credentials(legacy_env_path)
    if os.environ.get(LEGACY_ENV_PUSH_KEY, "").strip():
        return push_credentials_from_env_key()
    if required:
        raise WeightPushAuthError(
            "push credentials required: pass --push-credentials, set "
            f"{ENV_PUSH_CREDENTIALS}, or set {ENV_PUSH_KEY}"
        )
    return None


def write_push_credentials(path: str | Path, credentials: PushCredentialSet) -> Path:
    if type(credentials) is not PushCredentialSet:
        raise WeightPushAuthError("push credentials are untyped")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(credentials.to_dict()) + b"\n"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.chmod(0o600)
    tmp.replace(target)
    target.chmod(0o600)
    return target


def mint_push_credential(*, credential_id: str) -> PushCredential:
    return PushCredential(credential_id, secrets.token_urlsafe(48), "active")


def push_auth_digest(
    *,
    credential_id: str,
    method: str,
    path: str,
    timestamp: int,
    body_digest: str,
    legacy: bool = False,
) -> str:
    if method != "PUT" or path != "/v1/current-weights":
        raise WeightPushAuthError("push auth route is unsupported")
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightPushAuthError("push timestamp is malformed")
    body_digest = require_sha256_hex(body_digest, field="body_digest")
    return canonical_digest(
        LEGACY_PUSH_AUTH_DOMAIN if legacy else PUSH_AUTH_DOMAIN,
        {
            "body_digest": body_digest,
            "credential_id": credential_id,
            "method": method,
            "path": path,
            "timestamp": timestamp,
        },
    )


def sign_push_request(
    credential: PushCredential,
    *,
    timestamp: int,
    body: bytes,
    method: str = "PUT",
    path: str = "/v1/current-weights",
    legacy: bool = False,
) -> dict[str, str]:
    if type(credential) is not PushCredential or credential.status != "active":
        raise WeightPushAuthError("push signing requires an active credential")
    if not isinstance(body, (bytes, bytearray)):
        raise WeightPushAuthError("push body must be bytes")
    body_digest = hashlib.sha256(bytes(body)).hexdigest()
    digest = push_auth_digest(
        credential_id=credential.credential_id,
        method=method,
        path=path,
        timestamp=timestamp,
        body_digest=body_digest,
        legacy=legacy,
    )
    signature = hmac.new(
        credential.secret.encode("utf-8"),
        digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    prefix = "Optima" if legacy else "Cacheon"
    return {
        "Content-Type": "application/json; charset=utf-8",
        f"X-{prefix}-Push-Credential-Id": credential.credential_id,
        f"X-{prefix}-Push-Timestamp": str(timestamp),
        f"X-{prefix}-Push-Body-Digest": body_digest,
        f"X-{prefix}-Push-Signature": signature,
    }


def push_request_uses_legacy_protocol(headers: dict[str, str]) -> bool:
    """Select one exact push-header family and reject mixed downgrade attempts."""

    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    cacheon = any(key.startswith("x-cacheon-push-") for key in normalized)
    optima = any(key.startswith("x-optima-push-") for key in normalized)
    if cacheon and optima:
        raise WeightPushAuthError("push request mixes Cacheon and Optima headers")
    return optima


def verify_push_request(
    credentials: PushCredentialSet,
    *,
    headers: dict[str, str],
    body: bytes,
    now: int,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    method: str = "PUT",
    path: str = "/v1/current-weights",
) -> str:
    """Verify a push request; return the accepted credential id."""

    if type(credentials) is not PushCredentialSet:
        raise WeightPushAuthError("push credential set is untyped")
    if not isinstance(body, (bytes, bytearray)):
        raise WeightPushAuthError("push body must be bytes")
    if type(now) is not int or now <= 0:
        raise WeightPushAuthError("clock reading is malformed")
    if type(max_skew_seconds) is not int or not 1 <= max_skew_seconds <= 600:
        raise WeightPushAuthError("push skew bound is malformed")

    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    legacy = push_request_uses_legacy_protocol(normalized)
    prefix = "x-optima" if legacy else "x-cacheon"
    display_prefix = "X-Optima" if legacy else "X-Cacheon"
    credential_id = normalized.get(f"{prefix}-push-credential-id", "")
    timestamp_raw = normalized.get(f"{prefix}-push-timestamp", "")
    body_digest_header = normalized.get(f"{prefix}-push-body-digest", "")
    signature = normalized.get(f"{prefix}-push-signature", "")
    try:
        timestamp = int(timestamp_raw)
    except ValueError as exc:
        raise WeightPushAuthError("push timestamp is malformed") from exc
    if abs(now - timestamp) > max_skew_seconds:
        raise WeightPushAuthError("push timestamp is outside the accepted skew window")

    body_digest = hashlib.sha256(bytes(body)).hexdigest()
    declared = require_sha256_hex(
        body_digest_header,
        field=f"{display_prefix}-Push-Body-Digest",
    )
    if not hmac.compare_digest(declared, body_digest):
        raise WeightPushAuthError("push body digest mismatch")

    credential = credentials.get(credential_id)
    if credential is None or credential.status != "active":
        raise WeightPushAuthError("push credential is unknown or retired")

    digest = push_auth_digest(
        credential_id=credential.credential_id,
        method=method,
        path=path,
        timestamp=timestamp,
        body_digest=body_digest,
        legacy=legacy,
    )
    expected = hmac.new(
        credential.secret.encode("utf-8"),
        digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(char not in "0123456789abcdef" for char in signature)
        or not hmac.compare_digest(signature, expected)
    ):
        raise WeightPushAuthError("push signature is invalid")
    return credential.credential_id


def sign_push_acknowledgement(
    credential: PushCredential,
    *,
    offer_digest: str,
    projection_digest: str,
    request_timestamp: int,
    legacy: bool = False,
) -> dict[str, object]:
    """Authenticate the exact server acknowledgement to the eval client."""

    if type(credential) is not PushCredential or credential.status != "active":
        raise WeightPushAuthError(
            "push acknowledgement signing requires an active credential"
        )
    if type(request_timestamp) is not int or request_timestamp <= 0:
        raise WeightPushAuthError(
            "push acknowledgement request timestamp is malformed"
        )
    unsigned: dict[str, object] = {
        "credential_id": credential.credential_id,
        "offer_digest": require_sha256_hex(
            offer_digest,
            field="offer_digest",
        ),
        "projection_digest": require_sha256_hex(
            projection_digest,
            field="projection_digest",
        ),
        "request_timestamp": request_timestamp,
        "schema": LEGACY_PUSH_ACK_SCHEMA if legacy else PUSH_ACK_SCHEMA,
        "status": "accepted",
    }
    digest = canonical_digest(
        LEGACY_PUSH_ACK_AUTH_DOMAIN if legacy else PUSH_ACK_AUTH_DOMAIN,
        unsigned,
    )
    mac = hmac.new(
        credential.secret.encode("utf-8"),
        digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return {**unsigned, "mac": mac}


def verify_push_acknowledgement(
    credential: PushCredential,
    acknowledgement: object,
    *,
    offer_digest: str,
    projection_digest: str,
    request_timestamp: int,
    legacy: bool | None = None,
) -> None:
    """Require a fresh, HMAC-authenticated acknowledgement of one push."""

    if type(credential) is not PushCredential or credential.status != "active":
        raise WeightPushAuthError(
            "push acknowledgement verification requires an active credential"
        )
    if type(acknowledgement) is not dict:
        raise WeightPushAuthError("push acknowledgement fields do not match")
    schema = acknowledgement.get("schema")
    if schema == PUSH_ACK_SCHEMA:
        acknowledgement_is_legacy = False
    elif schema == LEGACY_PUSH_ACK_SCHEMA:
        acknowledgement_is_legacy = True
    elif schema is None:
        raise WeightPushAuthError("push acknowledgement fields do not match")
    else:
        raise WeightPushAuthError("push acknowledgement schema is unsupported")
    if legacy is not None and acknowledgement_is_legacy is not legacy:
        raise WeightPushAuthError("push acknowledgement protocol does not match request")
    expected = sign_push_acknowledgement(
        credential,
        offer_digest=offer_digest,
        projection_digest=projection_digest,
        request_timestamp=request_timestamp,
        legacy=acknowledgement_is_legacy,
    )
    if type(acknowledgement) is not dict or set(acknowledgement) != set(expected):
        raise WeightPushAuthError("push acknowledgement fields do not match")
    mac = acknowledgement.get("mac")
    if (
        not isinstance(mac, str)
        or len(mac) != 64
        or any(char not in "0123456789abcdef" for char in mac)
    ):
        raise WeightPushAuthError(
            "push acknowledgement authentication is malformed"
        )
    unsigned_matches = all(
        acknowledgement.get(field) == value
        for field, value in expected.items()
        if field != "mac"
    )
    if not unsigned_matches or not hmac.compare_digest(mac, str(expected["mac"])):
        raise WeightPushAuthError(
            "push acknowledgement authentication failed"
        )


__all__ = [
    "CREDENTIALS_SCHEMA",
    "DEFAULT_ENV_CREDENTIAL_ID",
    "ENV_PUSH_CREDENTIAL_ID",
    "ENV_PUSH_CREDENTIALS",
    "ENV_PUSH_KEY",
    "LEGACY_CREDENTIALS_SCHEMA",
    "LEGACY_ENV_PUSH_CREDENTIAL_ID",
    "LEGACY_ENV_PUSH_CREDENTIALS",
    "LEGACY_ENV_PUSH_KEY",
    "LEGACY_PUSH_ACK_AUTH_DOMAIN",
    "LEGACY_PUSH_ACK_SCHEMA",
    "LEGACY_PUSH_AUTH_DOMAIN",
    "PUSH_ACK_AUTH_DOMAIN",
    "PUSH_ACK_SCHEMA",
    "PushCredential",
    "PushCredentialSet",
    "WeightPushAuthError",
    "load_push_credentials",
    "mint_push_credential",
    "push_auth_digest",
    "push_credentials_from_env_key",
    "push_request_uses_legacy_protocol",
    "resolve_push_credentials",
    "sign_push_acknowledgement",
    "sign_push_request",
    "verify_push_acknowledgement",
    "verify_push_request",
    "write_push_credentials",
]
