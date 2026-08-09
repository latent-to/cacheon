"""Permit-gated sharing of current publishable legacy V1 weights.

Eval builds a :class:`CurrentWeightOffer` around a legacy V1
:class:`WeightProjection` and pushes it to ``serve-weights`` with rotatable
HMAC credentials. Eval never opens a chain-signing weight path.

Cheap ``serve-weights`` hosts persist the offer (object store or local file),
accept authenticated PUT from eval, and serve permit-gated GET to validators.
Followers rebind the signer-facing projection and publish via
``reconcile_weight_publication`` / commit-reveal (``follow-weights``).

This module is original Cacheon code (Apache-2.0). Similar subnet patterns
(public weight APIs, hotkey-signed request headers) exist elsewhere; no third-
party sources were copied. Object-store I/O goes through :mod:`cacheon.object_store`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cacheon import chain
from cacheon.chain.weight_push_auth import (
    PushCredential,
    PushCredentialSet,
    WeightPushAuthError,
    sign_push_acknowledgement,
    sign_push_request,
    verify_push_acknowledgement,
    verify_push_request,
)
from cacheon.chain.weights import (
    WeightProjection,
    WeightPublicationError,
    WeightPublicationJournal,
    reconcile_weight_publication,
)
from cacheon.object_store import (
    ObjectStore,
    ObjectStoreError,
    ObjectStoreNotFoundError,
)
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)


logger = logging.getLogger("cacheon.chain.weight_share")

OFFER_SCHEMA_V1 = "cacheon.current-weight-offer.v1"
OFFER_SCHEMA = "cacheon.current-weight-offer.v2"
STORED_OFFER_SCHEMA = "cacheon.authenticated-weight-offer.v1"
OFFER_DOMAIN = "cacheon.current-weight-offer"
LANE_LEGACY_V1 = "legacy_v1"
OFFER_LANES = frozenset({LANE_LEGACY_V1})
REQUEST_DOMAIN = "cacheon.weight-share.request.v1"
RESPONSE_DOMAIN = "cacheon.weight-share.response.v1"
STORAGE_AUTH_DOMAIN = "cacheon.weight-share.storage.v1"
CURRENT_WEIGHTS_PATH = "/v1/current-weights"
DEFAULT_MAX_SKEW_SECONDS = 60
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_REMOTE_OFFER_KEY = "current_weights.json"
OFFER_CONTENT_TYPE = "application/json; charset=utf-8"

SignFn = Callable[[bytes], bytes]
VerifyFn = Callable[[str, bytes, bytes], bool]
OfferLoader = Callable[[], "CurrentWeightOffer | AuthenticatedWeightOffer"]
OfferSink = Callable[["CurrentWeightOffer | AuthenticatedWeightOffer"], None]

_REMOTE_OFFER_LOCKS_GUARD = threading.Lock()
_REMOTE_OFFER_LOCKS: dict[tuple[int, str], threading.Lock] = {}


class WeightShareError(RuntimeError):
    """Auth, offer, or transport for shared weights failed closed."""

    validator_fault = True
    retryable = False


class WeightShareRetryableError(WeightShareError):
    """Transient transport or chain read failure that a follower may retry."""

    retryable = True


class HotkeySigner(Protocol):
    """Minimal wallet hotkey surface used for weight-share signatures."""

    ss58_address: str

    def sign(self, data: bytes) -> bytes: ...


@dataclass(frozen=True)
class CurrentWeightOffer:
    """Exact publishable weights for peer validators.

    Offers carry only a :class:`WeightProjection` on the legacy V1 lane.  The
    ``lane`` and ``debt_binding`` wire fields are retained so historical
    stored offers keep reopening byte-identically; the extracted V2 economics
    lanes are rejected.
    """

    lane: str
    projection: WeightProjection
    debt_binding: None = None

    def __post_init__(self) -> None:
        if self.lane not in OFFER_LANES:
            raise WeightShareError("current weight offer lane is unsupported")
        if type(self.projection) is not WeightProjection:
            raise WeightShareError("current weight offer projection is untyped")
        if self.debt_binding is not None:
            raise WeightShareError("legacy V1 offer cannot carry a debt binding")

    @property
    def digest(self) -> str:
        return canonical_digest(OFFER_DOMAIN, self.to_dict())

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "lane": self.lane,
            "projection": self.projection.to_dict(),
            "projection_digest": self.projection.digest,
            "schema": OFFER_SCHEMA,
        }
        row["debt_binding"] = None
        row["debt_binding_digest"] = None
        return row

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_legacy_projection(cls, projection: WeightProjection) -> "CurrentWeightOffer":
        return cls(LANE_LEGACY_V1, projection, None)

    @classmethod
    def from_dict(cls, value: object) -> "CurrentWeightOffer":
        if type(value) is not dict:
            raise WeightShareError("current weight offer fields do not match")
        schema = value.get("schema")
        if schema == OFFER_SCHEMA_V1:
            # Historical local files: projection-only legacy V1.
            if set(value) != {"projection", "projection_digest", "schema"}:
                raise WeightShareError("legacy weight offer fields do not match")
            try:
                projection = WeightProjection.from_dict(value["projection"])
            except WeightPublicationError as exc:
                raise WeightShareError(
                    f"current weight offer projection is malformed: {exc}"
                ) from None
            digest = require_sha256_hex(
                value["projection_digest"], field="projection_digest"
            )
            if projection.digest != digest:
                raise WeightShareError(
                    "current weight offer projection digest does not match"
                )
            return cls(LANE_LEGACY_V1, projection, None)
        if schema != OFFER_SCHEMA:
            raise WeightShareError("current weight offer schema is unsupported")
        expected = {
            "debt_binding",
            "debt_binding_digest",
            "lane",
            "projection",
            "projection_digest",
            "schema",
        }
        if set(value) != expected:
            raise WeightShareError("current weight offer fields do not match")
        lane = value["lane"]
        try:
            projection = WeightProjection.from_dict(value["projection"])
        except WeightPublicationError as exc:
            raise WeightShareError(
                f"current weight offer projection is malformed: {exc}"
            ) from None
        digest = require_sha256_hex(
            value["projection_digest"], field="projection_digest"
        )
        if projection.digest != digest:
            raise WeightShareError(
                "current weight offer projection digest does not match"
            )
        if value["debt_binding"] is not None or value["debt_binding_digest"] is not None:
            raise WeightShareError(
                "debt-lane weight offers were retired with the V2 economics extraction"
            )
        return cls(str(lane), projection, None)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CurrentWeightOffer":
        if not isinstance(raw, (bytes, bytearray)):
            raise WeightShareError("current weight offer bytes are malformed")
        try:
            value = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeightShareError(
                f"current weight offer is not canonical JSON: {exc}"
            ) from None
        return cls.from_dict(value)


def storage_auth_digest(
    *,
    credential_id: str,
    offer_digest: str,
) -> str:
    if (
        not isinstance(credential_id, str)
        or not credential_id
        or credential_id.strip() != credential_id
    ):
        raise WeightShareError("stored offer credential id is malformed")
    return canonical_digest(
        STORAGE_AUTH_DOMAIN,
        {
            "credential_id": credential_id,
            "offer_digest": require_sha256_hex(
                offer_digest, field="offer_digest"
            ),
        },
    )


def _storage_mac(credential: PushCredential, digest: str) -> str:
    if type(credential) is not PushCredential:
        raise WeightShareError("stored offer requires an exact push credential")
    return hmac.new(
        credential.secret.encode("utf-8"),
        bytes.fromhex(require_sha256_hex(digest, field="storage_auth_digest")),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True)
class AuthenticatedWeightOffer:
    """Offer bytes authenticated before entering mutable object storage."""

    credential_id: str
    offer: CurrentWeightOffer
    mac: str

    def __post_init__(self) -> None:
        if type(self.offer) is not CurrentWeightOffer:
            raise WeightShareError("authenticated weight offer is untyped")
        storage_auth_digest(
            credential_id=self.credential_id,
            offer_digest=self.offer.digest,
        )
        require_sha256_hex(self.mac, field="storage_mac")

    def to_dict(self) -> dict[str, object]:
        return {
            "credential_id": self.credential_id,
            "mac": self.mac,
            "offer": self.offer.to_dict(),
            "offer_digest": self.offer.digest,
            "schema": STORED_OFFER_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> "AuthenticatedWeightOffer":
        if type(value) is not dict or set(value) != {
            "credential_id",
            "mac",
            "offer",
            "offer_digest",
            "schema",
        }:
            raise WeightShareError(
                "authenticated weight offer fields do not match"
            )
        schema = value["schema"]
        if schema != STORED_OFFER_SCHEMA:
            raise WeightShareError(
                "authenticated weight offer schema is unsupported"
            )
        offer = CurrentWeightOffer.from_dict(value["offer"])
        offer_digest = require_sha256_hex(
            value["offer_digest"], field="offer_digest"
        )
        if offer.digest != offer_digest:
            raise WeightShareError(
                "authenticated weight offer digest does not match"
            )
        return cls(
            str(value["credential_id"]),
            offer,
            require_sha256_hex(value["mac"], field="storage_mac"),
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "AuthenticatedWeightOffer":
        if not isinstance(raw, (bytes, bytearray)):
            raise WeightShareError(
                "authenticated weight offer bytes are malformed"
            )
        try:
            value = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeightShareError(
                f"authenticated weight offer is not canonical JSON: {exc}"
            ) from None
        return cls.from_dict(value)


def authenticate_weight_offer(
    offer: CurrentWeightOffer,
    credential: PushCredential,
) -> AuthenticatedWeightOffer:
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer authentication requires an exact offer")
    digest = storage_auth_digest(
        credential_id=credential.credential_id,
        offer_digest=offer.digest,
    )
    return AuthenticatedWeightOffer(
        credential.credential_id,
        offer,
        _storage_mac(credential, digest),
    )


def verify_authenticated_weight_offer(
    stored: AuthenticatedWeightOffer,
    credentials: PushCredentialSet,
) -> CurrentWeightOffer:
    if type(stored) is not AuthenticatedWeightOffer:
        raise WeightShareError("stored weight offer is not authenticated")
    if type(credentials) is not PushCredentialSet:
        raise WeightShareError(
            "authenticated weight offer requires push credentials"
        )
    credential = credentials.get(stored.credential_id)
    if credential is None:
        raise WeightShareError(
            "stored weight offer names an unknown push credential"
        )
    digest = storage_auth_digest(
        credential_id=stored.credential_id,
        offer_digest=stored.offer.digest,
    )
    if not hmac.compare_digest(stored.mac, _storage_mac(credential, digest)):
        raise WeightShareError("stored weight offer authentication failed")
    return stored.offer


def _parse_offer_storage(
    raw: bytes,
) -> CurrentWeightOffer | AuthenticatedWeightOffer:
    if not isinstance(raw, (bytes, bytearray)):
        raise WeightShareError("stored weight offer bytes are malformed")
    try:
        value = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightShareError(
            f"stored weight offer is not canonical JSON: {exc}"
        ) from None
    if type(value) is not dict:
        raise WeightShareError("stored weight offer fields do not match")
    if value.get("schema") == STORED_OFFER_SCHEMA:
        return AuthenticatedWeightOffer.from_dict(value)
    return CurrentWeightOffer.from_dict(value)


def _offer_from_storage(
    stored: CurrentWeightOffer | AuthenticatedWeightOffer,
) -> CurrentWeightOffer:
    if type(stored) is CurrentWeightOffer:
        return stored
    if type(stored) is AuthenticatedWeightOffer:
        return stored.offer
    raise WeightShareError("stored weight offer is not exactly typed")


def assert_monotonic_offer_update(
    current: CurrentWeightOffer,
    proposed: CurrentWeightOffer,
) -> None:
    """Reject rollback, same-block equivocation, and V2-to-V1 regression."""

    if (
        type(current) is not CurrentWeightOffer
        or type(proposed) is not CurrentWeightOffer
    ):
        raise WeightShareError("weight offer update is not exactly typed")
    if (
        proposed.projection.chain_scope_digest
        != current.projection.chain_scope_digest
        or proposed.projection.netuid != current.projection.netuid
    ):
        raise WeightShareError("weight offer update changed chain scope")
    if proposed.projection.effective_block < current.projection.effective_block:
        raise WeightShareError("weight offer effective block regressed")
    if (
        proposed.projection.effective_block == current.projection.effective_block
        # Republishing the identical typed offer is idempotent.
        and proposed != current
    ):
        raise WeightShareError("weight offer conflicts at the current effective block")
def _remote_offer_lock(store: ObjectStore, key: str) -> threading.Lock:
    identity = (id(store), key)
    with _REMOTE_OFFER_LOCKS_GUARD:
        lock = _REMOTE_OFFER_LOCKS.get(identity)
        if lock is None:
            lock = threading.Lock()
            _REMOTE_OFFER_LOCKS[identity] = lock
        return lock


def default_offer_path(intake_db: str | Path) -> Path:
    """Sibling file next to the intake DB (local durability on the eval host)."""

    path = Path(intake_db)
    return path.with_name(path.name + ".current_weights.json")


def write_current_weight_offer(
    path: str | Path, offer: CurrentWeightOffer | WeightProjection
) -> Path:
    """Atomically persist the current offer locally."""

    if type(offer) is WeightProjection:
        offer = CurrentWeightOffer.from_legacy_projection(offer)
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer requires an exact CurrentWeightOffer")
    return _write_offer_storage(path, offer)


def _write_offer_storage(
    path: str | Path,
    stored: CurrentWeightOffer | AuthenticatedWeightOffer,
) -> Path:
    if type(stored) not in {CurrentWeightOffer, AuthenticatedWeightOffer}:
        raise WeightShareError("stored weight offer is not exactly typed")
    target = Path(path)
    if target.exists() and not target.is_file():
        raise WeightShareError("weight offer path is not a regular file")
    if target.exists():
        current = _parse_offer_storage(target.read_bytes())
        if (
            type(current) is AuthenticatedWeightOffer
            and type(stored) is CurrentWeightOffer
        ):
            raise WeightShareError(
                "authenticated current offer cannot be overwritten unsigned"
            )
        assert_monotonic_offer_update(
            _offer_from_storage(current),
            _offer_from_storage(stored),
        )
    payload = stored.to_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
        os.chmod(target, 0o600)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def write_authenticated_weight_offer(
    path: str | Path,
    stored: AuthenticatedWeightOffer,
) -> Path:
    """Atomically persist a gateway-authenticated offer envelope."""

    if type(stored) is not AuthenticatedWeightOffer:
        raise WeightShareError(
            "authenticated weight offer requires an exact envelope"
        )
    return _write_offer_storage(path, stored)


def read_current_weight_offer(path: str | Path) -> CurrentWeightOffer:
    """Reopen one persisted local current-weight offer."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError as exc:
        raise WeightShareError("current weight offer is missing") from exc
    except OSError as exc:
        raise WeightShareError(f"current weight offer cannot be read: {exc}") from None
    stored = _parse_offer_storage(raw)
    if type(stored) is not CurrentWeightOffer:
        raise WeightShareError(
            "current weight offer is authenticated storage, not a raw offer"
        )
    return stored


def read_offer_storage(
    path: str | Path,
) -> CurrentWeightOffer | AuthenticatedWeightOffer:
    target = Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError as exc:
        raise WeightShareError("current weight offer is missing") from exc
    except OSError as exc:
        raise WeightShareError(
            f"current weight offer cannot be read: {exc}"
        ) from None
    return _parse_offer_storage(raw)


def put_current_weight_offer(
    store: ObjectStore,
    offer: CurrentWeightOffer | WeightProjection,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> str:
    """Upload the exact offer bytes to a swappable object store."""

    if type(offer) is WeightProjection:
        offer = CurrentWeightOffer.from_legacy_projection(offer)
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer requires an exact CurrentWeightOffer")
    return _put_offer_storage(store, offer, key=key)


def _put_offer_storage(
    store: ObjectStore,
    stored: CurrentWeightOffer | AuthenticatedWeightOffer,
    *,
    key: str,
) -> str:
    if type(stored) not in {CurrentWeightOffer, AuthenticatedWeightOffer}:
        raise WeightShareError("stored weight offer is not exactly typed")
    with _remote_offer_lock(store, key):
        try:
            raw = store.get_bytes(key)
        except ObjectStoreNotFoundError:
            current = None
        except ObjectStoreError as exc:
            if getattr(exc, "retryable", False):
                raise WeightShareRetryableError(str(exc)) from None
            raise WeightShareError(str(exc)) from None
        else:
            current = _parse_offer_storage(raw)
        if current is not None:
            if (
                type(current) is AuthenticatedWeightOffer
                and type(stored) is CurrentWeightOffer
            ):
                raise WeightShareError(
                    "authenticated current offer cannot be overwritten unsigned"
                )
            assert_monotonic_offer_update(
                _offer_from_storage(current),
                _offer_from_storage(stored),
            )
        try:
            store.put_bytes(
                key,
                stored.to_bytes(),
                content_type=OFFER_CONTENT_TYPE,
            )
        except ObjectStoreError as exc:
            if getattr(exc, "retryable", False):
                raise WeightShareRetryableError(str(exc)) from None
            raise WeightShareError(str(exc)) from None
    return key


def put_authenticated_weight_offer(
    store: ObjectStore,
    stored: AuthenticatedWeightOffer,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> str:
    if type(stored) is not AuthenticatedWeightOffer:
        raise WeightShareError(
            "authenticated weight offer requires an exact envelope"
        )
    return _put_offer_storage(store, stored, key=key)


def load_current_weight_offer_from_store(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> CurrentWeightOffer:
    """Load the current offer from object storage (serve-weights path)."""

    try:
        raw = store.get_bytes(key)
    except ObjectStoreError as exc:
        if getattr(exc, "retryable", False):
            raise WeightShareRetryableError(str(exc)) from None
        raise WeightShareError(str(exc)) from None
    stored = _parse_offer_storage(raw)
    if type(stored) is not CurrentWeightOffer:
        raise WeightShareError(
            "current weight offer is authenticated storage, not a raw offer"
        )
    return stored


def load_offer_storage_from_store(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> CurrentWeightOffer | AuthenticatedWeightOffer:
    try:
        raw = store.get_bytes(key)
    except ObjectStoreError as exc:
        if getattr(exc, "retryable", False):
            raise WeightShareRetryableError(str(exc)) from None
        raise WeightShareError(str(exc)) from None
    return _parse_offer_storage(raw)


def publish_current_weight_offer(
    offer: CurrentWeightOffer | WeightProjection,
    *,
    local_path: str | Path,
    remote_store: ObjectStore | None = None,
    remote_key: str = DEFAULT_REMOTE_OFFER_KEY,
    async_remote: bool = True,
) -> Path:
    """Write locally, then publish to the object store (optionally in the background)."""

    if type(offer) is WeightProjection:
        offer = CurrentWeightOffer.from_legacy_projection(offer)
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer requires an exact CurrentWeightOffer")
    path = write_current_weight_offer(local_path, offer)
    if remote_store is None:
        return path

    def _upload() -> None:
        try:
            put_current_weight_offer(remote_store, offer, key=remote_key)
            logger.info(
                "published weight offer %s to object store key %s",
                offer.digest,
                remote_key,
            )
        except Exception:
            logger.exception(
                "async weight-offer object-store publish failed for %s",
                offer.digest,
            )

    if async_remote:
        threading.Thread(
            target=_upload,
            name="cacheon-weight-offer-upload",
            daemon=True,
        ).start()
    else:
        put_current_weight_offer(remote_store, offer, key=remote_key)
    return path


def local_offer_loader(path: str | Path) -> OfferLoader:
    target = Path(path)

    def load() -> CurrentWeightOffer | AuthenticatedWeightOffer:
        return read_offer_storage(target)

    return load


def object_store_offer_loader(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> OfferLoader:
    def load() -> CurrentWeightOffer | AuthenticatedWeightOffer:
        return load_offer_storage_from_store(store, key=key)

    return load


def object_store_offer_sink(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> OfferSink:
    def save(
        offer: CurrentWeightOffer | AuthenticatedWeightOffer,
    ) -> None:
        if type(offer) is AuthenticatedWeightOffer:
            put_authenticated_weight_offer(store, offer, key=key)
        else:
            put_current_weight_offer(store, offer, key=key)

    return save


def rebind_projection_signer(
    projection: WeightProjection, signer_hotkey: str
) -> WeightProjection:
    """Keep the economic vector; bind publication to the follower's hotkey."""

    if type(projection) is not WeightProjection:
        raise WeightShareError("rebind requires an exact WeightProjection")
    if (
        not isinstance(signer_hotkey, str)
        or not signer_hotkey
        or signer_hotkey.strip() != signer_hotkey
        or len(signer_hotkey) > 256
    ):
        raise WeightShareError("follower signer hotkey is malformed")
    if signer_hotkey == projection.validator_hotkey:
        return projection
    return replace(projection, validator_hotkey=signer_hotkey)


def rebind_offer_signer(
    offer: CurrentWeightOffer, signer_hotkey: str
) -> CurrentWeightOffer:
    """Rebind the signer-facing projection of a fetched offer."""

    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("rebind requires an exact CurrentWeightOffer")
    projection = rebind_projection_signer(offer.projection, signer_hotkey)
    return CurrentWeightOffer(LANE_LEGACY_V1, projection, None)


def request_auth_digest(
    *,
    hotkey: str,
    method: str,
    netuid: int,
    path: str,
    timestamp: int,
) -> str:
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or hotkey.strip() != hotkey
        or len(hotkey) > 256
    ):
        raise WeightShareError("request hotkey is malformed")
    if method != "GET" or path != CURRENT_WEIGHTS_PATH:
        raise WeightShareError("weight-share request route is unsupported")
    if type(netuid) is not int or netuid < 0:
        raise WeightShareError("request netuid is malformed")
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightShareError("request timestamp is malformed")
    return canonical_digest(
        REQUEST_DOMAIN,
        {
            "hotkey": hotkey,
            "method": method,
            "netuid": netuid,
            "path": path,
            "timestamp": timestamp,
        },
    )


def response_auth_digest(
    *,
    authority_hotkey: str,
    body_digest: str,
    netuid: int,
    timestamp: int,
) -> str:
    if (
        not isinstance(authority_hotkey, str)
        or not authority_hotkey
        or authority_hotkey.strip() != authority_hotkey
        or len(authority_hotkey) > 256
    ):
        raise WeightShareError("response authority hotkey is malformed")
    body_digest = require_sha256_hex(body_digest, field="body_digest")
    if type(netuid) is not int or netuid < 0:
        raise WeightShareError("response netuid is malformed")
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightShareError("response timestamp is malformed")
    return canonical_digest(
        RESPONSE_DOMAIN,
        {
            "authority_hotkey": authority_hotkey,
            "body_digest": body_digest,
            "netuid": netuid,
            "timestamp": timestamp,
        },
    )


def _message_bytes(digest: str) -> bytes:
    return require_sha256_hex(digest, field="auth_digest").encode("ascii")


def sign_auth_digest(signer: HotkeySigner, digest: str) -> str:
    try:
        signature = signer.sign(_message_bytes(digest))
    except Exception as exc:
        raise WeightShareError(f"weight-share signing failed: {exc}") from None
    if not isinstance(signature, (bytes, bytearray)) or not signature:
        raise WeightShareError("weight-share signature bytes are malformed")
    return bytes(signature).hex()


def verify_auth_digest(
    hotkey: str,
    digest: str,
    signature_hex: str,
    *,
    verify: VerifyFn,
) -> None:
    if (
        not isinstance(signature_hex, str)
        or len(signature_hex) < 64
        or len(signature_hex) > 256
        or len(signature_hex) % 2 != 0
        or any(char not in "0123456789abcdef" for char in signature_hex)
    ):
        raise WeightShareError("weight-share signature encoding is malformed")
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise WeightShareError("weight-share signature encoding is malformed") from exc
    try:
        ok = bool(verify(hotkey, _message_bytes(digest), signature))
    except Exception as exc:
        raise WeightShareError(
            f"weight-share signature verification failed: {exc}"
        ) from None
    if not ok:
        raise WeightShareError("weight-share signature is invalid")


def assert_fresh_timestamp(
    timestamp: int, *, now: int, max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS
) -> None:
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightShareError("timestamp is malformed")
    if type(now) is not int or now <= 0:
        raise WeightShareError("clock reading is malformed")
    if type(max_skew_seconds) is not int or max_skew_seconds <= 0 or max_skew_seconds > 600:
        raise WeightShareError("timestamp skew bound is malformed")
    if abs(now - timestamp) > max_skew_seconds:
        raise WeightShareError("timestamp is outside the accepted skew window")


def assert_validator_permit(metagraph: chain.MetagraphView, hotkey: str) -> int:
    if type(metagraph) is not chain.MetagraphView:
        raise WeightShareError("permit check requires an exact MetagraphView")
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or hotkey.strip() != hotkey
        or len(hotkey) > 256
    ):
        raise WeightShareError("permit hotkey is malformed")
    uid = metagraph.uid_of(hotkey)
    if uid is None:
        raise WeightShareError("hotkey is not registered on the metagraph")
    if uid >= len(metagraph.validator_permit) or not bool(
        metagraph.validator_permit[uid]
    ):
        raise WeightShareError("hotkey does not currently hold validator_permit")
    return uid


def default_verify_fn(hotkey: str, message: bytes, signature: bytes) -> bool:
    """Verify an sr25519 hotkey signature via the installed wallet stack."""

    try:
        import bittensor as bt
    except ImportError as exc:
        raise WeightShareError(
            "bittensor is required to verify weight-share signatures"
        ) from exc
    keypair_cls = getattr(bt, "Keypair", None)
    if keypair_cls is None:
        wallet_mod = getattr(bt, "wallet", None)
        keypair_cls = getattr(wallet_mod, "Keypair", None) if wallet_mod else None
    if keypair_cls is None:
        try:
            from bittensor_wallet import Keypair as keypair_cls  # type: ignore
        except ImportError as exc:
            raise WeightShareError(
                "no Keypair implementation is available for weight-share verify"
            ) from exc
    keypair = keypair_cls(ss58_address=hotkey)
    return bool(keypair.verify(message, signature))


def build_signed_offer_response(
    offer: CurrentWeightOffer,
    *,
    authority: HotkeySigner,
    netuid: int,
    timestamp: int,
) -> tuple[bytes, dict[str, str]]:
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("signed response requires an exact CurrentWeightOffer")
    if offer.projection.netuid != netuid:
        raise WeightShareError("offer netuid differs from the served netuid")
    # The HTTP response signer is the weights-service hotkey. It need not equal
    # the offer's projection.validator_hotkey: eval builds the economic vector,
    # followers rebind before chain publish.
    body_obj = {
        "authority_hotkey": authority.ss58_address,
        "netuid": netuid,
        "offer": offer.to_dict(),
        "timestamp": timestamp,
    }
    body = canonical_json_bytes(body_obj) + b"\n"
    body_digest = sha256_hex(body)
    digest = response_auth_digest(
        authority_hotkey=authority.ss58_address,
        body_digest=body_digest,
        netuid=netuid,
        timestamp=timestamp,
    )
    signature = sign_auth_digest(authority, digest)
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Cacheon-Authority-Hotkey": authority.ss58_address,
        "X-Cacheon-Netuid": str(netuid),
        "X-Cacheon-Timestamp": str(timestamp),
        "X-Cacheon-Signature": signature,
        "X-Cacheon-Body-Digest": body_digest,
    }
    return body, headers


def _lower_headers(headers: object) -> dict[str, str]:
    if not hasattr(headers, "items"):
        raise WeightShareError("weight-share headers are malformed")
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()  # type: ignore[attr-defined]
    }


def parse_signed_offer_response(
    body: bytes,
    headers: dict[str, str],
    *,
    netuid: int,
    now: int,
    max_skew_seconds: int,
    verify: VerifyFn,
    expected_authority: str | None = None,
    metagraph: chain.MetagraphView | None = None,
) -> CurrentWeightOffer:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightShareError(f"weight-share response is not JSON: {exc}") from None
    if type(payload) is not dict or set(payload) != {
        "authority_hotkey",
        "netuid",
        "offer",
        "timestamp",
    }:
        raise WeightShareError("weight-share response fields do not match")
    authority_hotkey = payload["authority_hotkey"]
    timestamp = payload["timestamp"]
    response_netuid = payload["netuid"]
    if response_netuid != netuid:
        raise WeightShareError("weight-share response netuid mismatch")
    if type(timestamp) is not int:
        raise WeightShareError("weight-share response timestamp is malformed")
    assert_fresh_timestamp(
        timestamp, now=now, max_skew_seconds=max_skew_seconds
    )
    normalized_headers = _lower_headers(headers)
    header_authority = normalized_headers.get("x-cacheon-authority-hotkey", "")
    header_timestamp = normalized_headers.get("x-cacheon-timestamp", "")
    header_netuid = normalized_headers.get("x-cacheon-netuid", "")
    signature = normalized_headers.get("x-cacheon-signature", "")
    header_body_digest = normalized_headers.get("x-cacheon-body-digest", "")
    if (
        header_authority != authority_hotkey
        or header_timestamp != str(timestamp)
        or header_netuid != str(netuid)
    ):
        raise WeightShareError("weight-share response headers disagree with body")
    body_digest = sha256_hex(body)
    if header_body_digest:
        declared = require_sha256_hex(
            header_body_digest, field="X-Cacheon-Body-Digest"
        )
        if declared != body_digest:
            raise WeightShareError("weight-share response body digest mismatch")
    digest = response_auth_digest(
        authority_hotkey=authority_hotkey,
        body_digest=body_digest,
        netuid=netuid,
        timestamp=timestamp,
    )
    verify_auth_digest(authority_hotkey, digest, signature, verify=verify)
    if expected_authority is not None and authority_hotkey != expected_authority:
        raise WeightShareError("weight-share authority hotkey is not the pinned authority")
    # The response authority is an HTTP signing identity, not a chain publisher.
    # Callers pin it with expected_authority. Live validator_permit is required of
    # the GET requester (enforced by serve_current_weights), not of this hotkey.
    offer = CurrentWeightOffer.from_dict(payload["offer"])
    if offer.projection.netuid != netuid:
        raise WeightShareError("offer projection netuid mismatch")
    return offer


def _normalize_headers(headers: object) -> dict[str, str]:
    if hasattr(headers, "items"):
        return {str(key): str(value) for key, value in headers.items()}  # type: ignore[arg-type]
    raise WeightShareError("response headers are malformed")


class _WeightShareHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler,
        *,
        load_offer: OfferLoader,
        save_offer: OfferSink | None,
        push_credentials: PushCredentialSet | None,
        authority: HotkeySigner,
        subtensor,
        netuid: int,
        max_skew_seconds: int,
        verify: VerifyFn,
        clock: Callable[[], int],
    ) -> None:
        super().__init__(server_address, handler)
        self.load_offer = load_offer
        self.save_offer = save_offer
        self.push_credentials = push_credentials
        self.authority = authority
        self.subtensor = subtensor
        self.netuid = netuid
        self.max_skew_seconds = max_skew_seconds
        self.verify = verify
        self.clock = clock
        self._offer_lock = threading.Lock()


class _WeightShareHandler(BaseHTTPRequestHandler):
    server: _WeightShareHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        logger.info("weight-share: " + format, *args)

    def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {"Content-Type": "application/json"}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        payload = canonical_json_bytes({"error": message}) + b"\n"
        self._send(status, payload, {"Content-Type": "application/json; charset=utf-8"})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.split("?", 1)[0] != CURRENT_WEIGHTS_PATH:
            self._error(404, "not found")
            return
        server = self.server
        try:
            request_headers = _lower_headers(self.headers)
            hotkey = request_headers.get("x-cacheon-hotkey", "")
            timestamp_raw = request_headers.get("x-cacheon-timestamp", "")
            signature = request_headers.get("x-cacheon-signature", "")
            header_netuid = request_headers.get("x-cacheon-netuid", "")
            if header_netuid != str(server.netuid):
                raise WeightShareError("request netuid header mismatch")
            try:
                timestamp = int(timestamp_raw)
            except ValueError as exc:
                raise WeightShareError("request timestamp is malformed") from exc
            now = int(server.clock())
            assert_fresh_timestamp(
                timestamp, now=now, max_skew_seconds=server.max_skew_seconds
            )
            digest = request_auth_digest(
                hotkey=hotkey,
                method="GET",
                netuid=server.netuid,
                path=CURRENT_WEIGHTS_PATH,
                timestamp=timestamp,
            )
            verify_auth_digest(hotkey, digest, signature, verify=server.verify)
            metagraph = chain.fetch_metagraph(server.subtensor, server.netuid)
            assert_validator_permit(metagraph, hotkey)
            with server._offer_lock:
                stored = server.load_offer()
            if server.push_credentials is not None:
                if type(stored) is not AuthenticatedWeightOffer:
                    raise WeightShareError(
                        "push-enabled weight service requires authenticated storage"
                    )
                offer = verify_authenticated_weight_offer(
                    stored,
                    server.push_credentials,
                )
            else:
                if type(stored) is not CurrentWeightOffer:
                    raise WeightShareError(
                        "authenticated storage cannot be verified without "
                        "push credentials"
                    )
                offer = stored
            if offer.projection.netuid != server.netuid:
                raise WeightShareError("stored offer netuid differs from server netuid")
            body, headers = build_signed_offer_response(
                offer,
                authority=server.authority,
                netuid=server.netuid,
                timestamp=now,
            )
        except WeightShareRetryableError as exc:
            self._error(503, str(exc))
            return
        except WeightShareError as exc:
            self._error(403, str(exc))
            return
        except Exception as exc:
            logger.exception("weight-share handler failed")
            self._error(500, f"internal error: {type(exc).__name__}")
            return
        self._send(200, body, headers)

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.split("?", 1)[0] != CURRENT_WEIGHTS_PATH:
            self._error(404, "not found")
            return
        server = self.server
        if server.push_credentials is None or server.save_offer is None:
            self._error(405, "weight push is not enabled on this server")
            return
        try:
            length_raw = self.headers.get("Content-Length", "")
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise WeightShareError("push Content-Length is malformed") from exc
            if length < 2 or length > 8_000_000:
                raise WeightShareError("push body length is out of bounds")
            body = self.rfile.read(length)
            if len(body) != length:
                raise WeightShareError("push body length mismatch")
            now = int(server.clock())
            push_headers = {
                str(key).lower(): str(value)
                for key, value in self.headers.items()
            }
            credential_id = verify_push_request(
                server.push_credentials,
                headers=push_headers,
                body=body,
                now=now,
                max_skew_seconds=server.max_skew_seconds,
            )
            offer = CurrentWeightOffer.from_bytes(body)
            if offer.projection.netuid != server.netuid:
                raise WeightShareError("pushed offer netuid differs from server netuid")
            credential = server.push_credentials.get(credential_id)
            if credential is None:
                raise WeightPushAuthError(
                    "verified push credential is no longer retained"
                )
            request_timestamp = int(push_headers["x-cacheon-push-timestamp"])
            stored = authenticate_weight_offer(offer, credential)
            with server._offer_lock:
                server.save_offer(stored)
            logger.info(
                "accepted weight offer %s lane=%s via push credential %s",
                offer.digest,
                offer.lane,
                credential_id,
            )
            response = canonical_json_bytes(
                sign_push_acknowledgement(
                    credential,
                    offer_digest=offer.digest,
                    projection_digest=offer.projection.digest,
                    request_timestamp=request_timestamp,
                )
            ) + b"\n"
        except WeightShareRetryableError as exc:
            self._error(503, str(exc))
            return
        except (WeightShareError, WeightPushAuthError) as exc:
            self._error(403, str(exc))
            return
        except Exception as exc:
            logger.exception("weight-share push failed")
            self._error(500, f"internal error: {type(exc).__name__}")
            return
        self._send(
            200,
            response,
            {"Content-Type": "application/json; charset=utf-8"},
        )


def serve_current_weights(
    *,
    host: str,
    port: int,
    authority: HotkeySigner,
    subtensor,
    netuid: int,
    load_offer: OfferLoader | None = None,
    save_offer: OfferSink | None = None,
    push_credentials: PushCredentialSet | None = None,
    offer_path: str | Path | None = None,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    verify: VerifyFn | None = None,
    clock: Callable[[], int] | None = None,
) -> ThreadingHTTPServer:
    """Start the permit-gated current-weights HTTP server (caller serves forever).

    Prefer ``load_offer`` / ``save_offer`` backed by object storage on a host
    separate from eval. ``PUT /v1/current-weights`` accepts eval pushes only when
    rotatable ``push_credentials`` are configured. Eval must not chain-publish.
    """

    if not isinstance(host, str) or not host.strip():
        raise WeightShareError("weight-share host is malformed")
    if type(port) is not int or not 0 <= port <= 65535:
        raise WeightShareError("weight-share port is malformed")
    if load_offer is None:
        if offer_path is None:
            raise WeightShareError(
                "serve-weights requires load_offer or offer_path"
            )
        load_offer = local_offer_loader(offer_path)
        if save_offer is None:
            path = Path(offer_path)

            def _save(
                offer: CurrentWeightOffer | AuthenticatedWeightOffer,
            ) -> None:
                if type(offer) is AuthenticatedWeightOffer:
                    write_authenticated_weight_offer(path, offer)
                else:
                    write_current_weight_offer(path, offer)

            save_offer = _save
    if push_credentials is not None and save_offer is None:
        raise WeightShareError("push credentials require a configured save_offer")
    if push_credentials is not None and type(push_credentials) is not PushCredentialSet:
        raise WeightShareError("push credentials are untyped")
    server = _WeightShareHTTPServer(
        (host, port),
        _WeightShareHandler,
        load_offer=load_offer,
        save_offer=save_offer,
        push_credentials=push_credentials,
        authority=authority,
        subtensor=subtensor,
        netuid=netuid,
        max_skew_seconds=max_skew_seconds,
        verify=verify or default_verify_fn,
        clock=clock or (lambda: int(time.time())),
    )
    return server


def push_current_weights(
    url: str,
    offer: CurrentWeightOffer,
    *,
    credential: PushCredential,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    clock: Callable[[], int] | None = None,
    opener: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Eval-side: push one offer to the weights service. Never touches the chain."""

    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("push requires an exact CurrentWeightOffer")
    if type(credential) is not PushCredential:
        raise WeightShareError("push requires an exact PushCredential")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise WeightShareError("weight-share URL must be http(s)")
    timeout = float(timeout_seconds)
    if not 0.1 <= timeout <= 600:
        raise WeightShareError("timeout is out of bounds")
    body = offer.to_bytes()
    now = int((clock or (lambda: int(time.time())))())
    headers = sign_push_request(
        credential,
        timestamp=now,
        body=body,
    )
    endpoint = url.rstrip("/") + CURRENT_WEIGHTS_PATH
    request = Request(endpoint, data=body, method="PUT", headers=headers)
    open_url = opener or urlopen
    try:
        with open_url(request, timeout=timeout) as response:  # type: ignore[arg-type]
            raw = response.read()
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        if 500 <= int(exc.code) <= 599:
            raise WeightShareRetryableError(
                f"weight-share push server error {exc.code}: {detail}"
            ) from None
        raise WeightShareError(
            f"weight-share push rejected ({exc.code}): {detail}"
        ) from None
    except URLError as exc:
        raise WeightShareRetryableError(
            f"weight-share push transport failed: {exc}"
        ) from None
    if status != 200:
        raise WeightShareError(f"weight-share push unexpected status {status}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightShareError(f"weight-share push response is not JSON: {exc}") from None
    try:
        verify_push_acknowledgement(
            credential,
            payload,
            offer_digest=offer.digest,
            projection_digest=offer.projection.digest,
            request_timestamp=now,
        )
    except WeightPushAuthError as exc:
        raise WeightShareError(str(exc)) from None
    return payload


def publish_followed_weights(
    *,
    subtensor,
    signer_wallet,
    offer: CurrentWeightOffer,
    journal: WeightPublicationJournal,
    refresh_blocks: int,
    dry_run: bool = False,
):
    """Publish a fetched offer through the normal weight reconciler / commit-reveal."""

    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("follow publish requires an exact CurrentWeightOffer")
    try:
        follower_hotkey = signer_wallet.hotkey.ss58_address
    except AttributeError as exc:
        raise WeightShareError("follower publish requires a signer wallet") from exc
    rebound = rebind_offer_signer(offer, follower_hotkey)
    require_crown = rebound.projection.crown_count > 0
    return reconcile_weight_publication(
        subtensor,
        None if dry_run else signer_wallet,
        rebound.projection,
        journal,
        refresh_blocks=refresh_blocks,
        dry_run=dry_run,
        allow_stale_initial=True,
        max_stale_initial_blocks=refresh_blocks,
        require_current_crown=require_crown,
    )


def fetch_current_weights(
    url: str,
    *,
    signer: HotkeySigner,
    netuid: int,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    verify: VerifyFn | None = None,
    clock: Callable[[], int] | None = None,
    expected_authority: str | None = None,
    metagraph: chain.MetagraphView | None = None,
    opener: Callable[..., object] | None = None,
) -> CurrentWeightOffer:
    """Authenticated GET of the current publishable weight offer."""

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise WeightShareError("weight-share URL must be http(s)")
    if type(timeout_seconds) is not float and type(timeout_seconds) is not int:
        raise WeightShareError("timeout is malformed")
    timeout = float(timeout_seconds)
    if not 0.1 <= timeout <= 600:
        raise WeightShareError("timeout is out of bounds")
    now = int((clock or (lambda: int(time.time())))())
    digest = request_auth_digest(
        hotkey=signer.ss58_address,
        method="GET",
        netuid=netuid,
        path=CURRENT_WEIGHTS_PATH,
        timestamp=now,
    )
    signature = sign_auth_digest(signer, digest)
    endpoint = url.rstrip("/") + CURRENT_WEIGHTS_PATH
    request = Request(
        endpoint,
        method="GET",
        headers={
            "Accept": "application/json",
            "X-Cacheon-Hotkey": signer.ss58_address,
            "X-Cacheon-Netuid": str(netuid),
            "X-Cacheon-Timestamp": str(now),
            "X-Cacheon-Signature": signature,
        },
    )
    open_url = opener or urlopen
    try:
        with open_url(request, timeout=timeout) as response:  # type: ignore[arg-type]
            body = response.read()
            headers = _normalize_headers(getattr(response, "headers", {}))
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        if 500 <= int(exc.code) <= 599:
            raise WeightShareRetryableError(
                f"weight-share server error {exc.code}: {detail}"
            ) from None
        raise WeightShareError(
            f"weight-share request rejected ({exc.code}): {detail}"
        ) from None
    except URLError as exc:
        raise WeightShareRetryableError(
            f"weight-share transport failed: {exc}"
        ) from None
    if status != 200:
        raise WeightShareError(f"weight-share unexpected status {status}")
    return parse_signed_offer_response(
        body,
        headers,
        netuid=netuid,
        now=now,
        max_skew_seconds=max_skew_seconds,
        verify=verify or default_verify_fn,
        expected_authority=expected_authority,
        metagraph=metagraph,
    )


__all__ = [
    "CURRENT_WEIGHTS_PATH",
    "AuthenticatedWeightOffer",
    "CurrentWeightOffer",
    "DEFAULT_MAX_SKEW_SECONDS",
    "DEFAULT_REMOTE_OFFER_KEY",
    "LANE_LEGACY_V1",
    "WeightShareError",
    "WeightShareRetryableError",
    "assert_monotonic_offer_update",
    "assert_fresh_timestamp",
    "assert_validator_permit",
    "authenticate_weight_offer",
    "build_signed_offer_response",
    "default_offer_path",
    "default_verify_fn",
    "fetch_current_weights",
    "load_current_weight_offer_from_store",
    "load_offer_storage_from_store",
    "local_offer_loader",
    "object_store_offer_loader",
    "object_store_offer_sink",
    "parse_signed_offer_response",
    "publish_current_weight_offer",
    "publish_followed_weights",
    "push_current_weights",
    "put_authenticated_weight_offer",
    "put_current_weight_offer",
    "read_offer_storage",
    "read_current_weight_offer",
    "rebind_offer_signer",
    "rebind_projection_signer",
    "request_auth_digest",
    "response_auth_digest",
    "serve_current_weights",
    "sign_auth_digest",
    "verify_auth_digest",
    "verify_authenticated_weight_offer",
    "write_authenticated_weight_offer",
    "write_current_weight_offer",
]
