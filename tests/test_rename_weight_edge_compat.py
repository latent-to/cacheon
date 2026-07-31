"""Rolling-rename compatibility for the authenticated weight-share edge."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon import chain, cli
from cacheon.chain.weight_push_auth import (
    CREDENTIALS_SCHEMA,
    ENV_PUSH_CREDENTIAL_ID,
    ENV_PUSH_CREDENTIALS,
    ENV_PUSH_KEY,
    LEGACY_CREDENTIALS_SCHEMA,
    LEGACY_ENV_PUSH_CREDENTIAL_ID,
    LEGACY_ENV_PUSH_CREDENTIALS,
    LEGACY_ENV_PUSH_KEY,
    PUSH_ACK_SCHEMA,
    PushCredential,
    PushCredentialSet,
    WeightPushAuthError,
    load_push_credentials,
    resolve_push_credentials,
    sign_push_acknowledgement,
    sign_push_request,
    verify_push_acknowledgement,
    verify_push_request,
)
from cacheon.chain.weight_share import (
    LEGACY_OFFER_DOMAIN,
    LEGACY_OFFER_SCHEMA,
    LEGACY_STORAGE_AUTH_DOMAIN,
    LEGACY_STORED_OFFER_SCHEMA,
    STORED_OFFER_SCHEMA,
    AuthenticatedWeightOffer,
    CurrentWeightOffer,
    WeightShareError,
    authenticate_weight_offer,
    build_signed_offer_response,
    fetch_current_weights,
    parse_signed_offer_response,
    push_current_weights,
    serve_current_weights,
    verify_authenticated_weight_offer,
)
from cacheon.chain.weights import WeightProjection
from cacheon.object_store import LocalDirectoryObjectStore, ObjectStoreConfig
from cacheon.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex


def _digest(label: str) -> str:
    return sha256_hex(label.encode("utf-8"))


def _projection() -> WeightProjection:
    return WeightProjection(
        _digest("scope"),
        307,
        "authority",
        _digest("policy"),
        _digest("settlement"),
        _digest("evaluation"),
        _digest("metagraph"),
        (_digest("arena"),),
        1,
        100,
        1,
        (_digest("evidence"),),
        (("miner", 1_000_000),),
    )


def _legacy_authenticated_bytes(
    projection: WeightProjection,
    credential: PushCredential,
) -> bytes:
    """Reproduce the exact #72 envelope without using the new serializers."""

    offer = {
        "debt_binding": None,
        "debt_binding_digest": None,
        "lane": "legacy_v1",
        "projection": projection.to_dict(),
        "projection_digest": projection.digest,
        "schema": LEGACY_OFFER_SCHEMA,
    }
    offer_digest = canonical_digest(LEGACY_OFFER_DOMAIN, offer)
    auth_digest = canonical_digest(
        LEGACY_STORAGE_AUTH_DOMAIN,
        {
            "credential_id": credential.credential_id,
            "offer_digest": offer_digest,
        },
    )
    mac = hmac.new(
        credential.secret.encode("utf-8"),
        bytes.fromhex(auth_digest),
        hashlib.sha256,
    ).hexdigest()
    envelope = {
        "credential_id": credential.credential_id,
        "mac": mac,
        "offer": offer,
        "offer_digest": offer_digest,
        "schema": LEGACY_STORED_OFFER_SCHEMA,
    }
    return canonical_json_bytes(envelope) + b"\n"


class _Signer:
    def __init__(self, address: str = "authority") -> None:
        self.ss58_address = address

    def sign(self, data: bytes) -> bytes:
        return hashlib.sha256(
            (self.ss58_address + "-secret").encode("ascii") + data
        ).digest()


def _verify(hotkey: str, message: bytes, signature: bytes) -> bool:
    return signature == hashlib.sha256(
        (hotkey + "-secret").encode("ascii") + message
    ).digest()


def test_number_72_authenticated_offer_reopens_without_auth_downgrade() -> None:
    credential = PushCredential("push-live", "s" * 48)
    credentials = PushCredentialSet((credential,))
    stored = AuthenticatedWeightOffer.from_bytes(
        _legacy_authenticated_bytes(_projection(), credential)
    )

    offer = verify_authenticated_weight_offer(stored, credentials)
    assert offer.projection == _projection()
    assert stored.to_dict()["schema"] == LEGACY_STORED_OFFER_SCHEMA
    assert stored.to_dict()["offer"]["schema"] == LEGACY_OFFER_SCHEMA

    # The same MAC cannot be relabelled as a Cacheon-domain envelope.
    relabelled = json.loads(stored.to_bytes())
    relabelled["schema"] = STORED_OFFER_SCHEMA
    relabelled["offer"]["schema"] = "cacheon.current-weight-offer.v2"
    with pytest.raises(WeightShareError, match="digest does not match"):
        AuthenticatedWeightOffer.from_dict(relabelled)

    new_offer = CurrentWeightOffer.from_legacy_projection(_projection())
    forged = AuthenticatedWeightOffer(
        credential.credential_id,
        new_offer,
        stored.mac,
    )
    with pytest.raises(WeightShareError, match="authentication failed"):
        verify_authenticated_weight_offer(forged, credentials)

    assert authenticate_weight_offer(new_offer, credential).to_dict()["schema"] == (
        STORED_OFFER_SCHEMA
    )


def test_old_push_and_response_signatures_require_the_old_vocabulary() -> None:
    credential = PushCredential("push-live", "s" * 48)
    credentials = PushCredentialSet((credential,))
    body = CurrentWeightOffer.from_legacy_projection(_projection()).to_bytes(
        legacy=True
    )
    headers = sign_push_request(
        credential,
        timestamp=1_700_000_000,
        body=body,
        legacy=True,
    )
    assert verify_push_request(
        credentials,
        headers=headers,
        body=body,
        now=1_700_000_001,
    ) == credential.credential_id

    relabelled = {
        key.replace("X-Optima-", "X-Cacheon-"): value
        for key, value in headers.items()
    }
    with pytest.raises(WeightPushAuthError, match="signature is invalid"):
        verify_push_request(
            credentials,
            headers=relabelled,
            body=body,
            now=1_700_000_001,
        )
    mixed = dict(headers)
    mixed["X-Cacheon-Push-Timestamp"] = "1700000000"
    with pytest.raises(WeightPushAuthError, match="mixes"):
        verify_push_request(
            credentials,
            headers=mixed,
            body=body,
            now=1_700_000_001,
        )

    offer = CurrentWeightOffer.from_legacy_projection(_projection())
    response_body, response_headers = build_signed_offer_response(
        offer,
        authority=_Signer(),
        netuid=307,
        timestamp=1_700_000_000,
        legacy=True,
    )
    assert parse_signed_offer_response(
        response_body,
        response_headers,
        netuid=307,
        now=1_700_000_001,
        max_skew_seconds=60,
        verify=_verify,
    ).projection == offer.projection
    response_relabelled = {
        key.replace("X-Optima-", "X-Cacheon-"): value
        for key, value in response_headers.items()
    }
    with pytest.raises(WeightShareError, match="signature is invalid"):
        parse_signed_offer_response(
            response_body,
            response_relabelled,
            netuid=307,
            now=1_700_000_001,
            max_skew_seconds=60,
            verify=_verify,
        )


def test_old_ack_mac_requires_the_old_schema() -> None:
    credential = PushCredential("push-live", "s" * 48)
    offer_digest = _digest("offer")
    projection_digest = _digest("projection")
    acknowledgement = sign_push_acknowledgement(
        credential,
        offer_digest=offer_digest,
        projection_digest=projection_digest,
        request_timestamp=1_700_000_000,
        legacy=True,
    )
    verify_push_acknowledgement(
        credential,
        acknowledgement,
        offer_digest=offer_digest,
        projection_digest=projection_digest,
        request_timestamp=1_700_000_000,
        legacy=True,
    )

    relabelled = {**acknowledgement, "schema": PUSH_ACK_SCHEMA}
    with pytest.raises(WeightPushAuthError, match="protocol does not match"):
        verify_push_acknowledgement(
            credential,
            relabelled,
            offer_digest=offer_digest,
            projection_digest=projection_digest,
            request_timestamp=1_700_000_000,
            legacy=True,
        )


def test_new_server_serves_number_72_storage_to_both_client_vocabularies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = PushCredential("push-live", "s" * 48)
    credentials = PushCredentialSet((credential,))
    stored = AuthenticatedWeightOffer.from_bytes(
        _legacy_authenticated_bytes(_projection(), credential)
    )
    view = chain.MetagraphView(
        307,
        100,
        "0x" + f"{100:064x}",
        [0],
        ["follower"],
        [True],
        [0],
    )
    monkeypatch.setattr(chain, "fetch_metagraph", lambda *_args, **_kwargs: view)
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        load_offer=lambda: stored,
        save_offer=lambda _offer: None,
        push_credentials=credentials,
        authority=_Signer(),
        subtensor=object(),
        netuid=307,
        verify=_verify,
        clock=lambda: 1_700_000_000,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        for legacy in (True, False):
            offer = fetch_current_weights(
                url,
                signer=_Signer("follower"),
                netuid=307,
                expected_authority="authority",
                verify=_verify,
                clock=lambda: 1_700_000_000,
                legacy=legacy,
            )
            assert offer.projection == _projection()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_new_server_accepts_old_and_new_push_protocols() -> None:
    credential = PushCredential("push-live", "s" * 48)
    credentials = PushCredentialSet((credential,))
    saved: list[AuthenticatedWeightOffer] = []
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        load_offer=lambda: saved[-1],
        save_offer=saved.append,
        push_credentials=credentials,
        authority=_Signer(),
        subtensor=object(),
        netuid=307,
        verify=_verify,
        clock=lambda: 1_700_000_000,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        offer = CurrentWeightOffer.from_legacy_projection(_projection())
        for legacy, expected_schema in (
            (True, "optima.weight-share.push-ack.v1"),
            (False, PUSH_ACK_SCHEMA),
        ):
            acknowledgement = push_current_weights(
                url,
                offer,
                credential=credential,
                clock=lambda: 1_700_000_000,
                legacy=legacy,
            )
            assert acknowledgement["schema"] == expected_schema
            assert saved[-1].to_dict()["schema"] == (
                LEGACY_STORED_OFFER_SCHEMA if legacy else STORED_OFFER_SCHEMA
            )
            assert verify_authenticated_weight_offer(
                saved[-1], credentials
            ).projection == offer.projection
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_old_credentials_and_environment_aliases_remain_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        ENV_PUSH_CREDENTIALS,
        ENV_PUSH_KEY,
        ENV_PUSH_CREDENTIAL_ID,
        LEGACY_ENV_PUSH_CREDENTIALS,
        LEGACY_ENV_PUSH_KEY,
        LEGACY_ENV_PUSH_CREDENTIAL_ID,
    ):
        monkeypatch.delenv(name, raising=False)

    credential_file = tmp_path / "push.json"
    credential_file.write_text(
        json.dumps(
            {
                "credentials": [
                    {
                        "credential_id": "legacy-file",
                        "secret": "f" * 48,
                        "status": "active",
                    }
                ],
                "schema": LEGACY_CREDENTIALS_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    assert load_push_credentials(credential_file).active()[0].credential_id == (
        "legacy-file"
    )
    monkeypatch.setenv(LEGACY_ENV_PUSH_CREDENTIALS, str(credential_file))
    assert resolve_push_credentials(required=True).active()[0].credential_id == (
        "legacy-file"
    )

    monkeypatch.delenv(LEGACY_ENV_PUSH_CREDENTIALS)
    monkeypatch.setenv(LEGACY_ENV_PUSH_KEY, "l" * 48)
    monkeypatch.setenv(LEGACY_ENV_PUSH_CREDENTIAL_ID, "legacy-inline")
    inline = resolve_push_credentials(required=True)
    assert inline is not None
    assert inline.active()[0].credential_id == "legacy-inline"
    assert inline.active()[0].secret == "l" * 48

    # Cacheon variables have deterministic precedence during the overlap window.
    cacheon_file = tmp_path / "cacheon-push.json"
    cacheon_file.write_text(
        json.dumps(
            {
                "credentials": [
                    {
                        "credential_id": "cacheon-file",
                        "secret": "c" * 48,
                        "status": "active",
                    }
                ],
                "schema": CREDENTIALS_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_PUSH_CREDENTIALS, str(cacheon_file))
    assert resolve_push_credentials(required=True).active()[0].credential_id == (
        "cacheon-file"
    )


def test_old_object_store_environment_is_a_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_root = tmp_path / "legacy-store"
    cacheon_root = tmp_path / "cacheon-store"
    for name in (
        "CACHEON_OBJECT_STORE_PROVIDER",
        "CACHEON_OBJECT_STORE_ROOT_DIR",
        "OPTIMA_OBJECT_STORE_PROVIDER",
        "OPTIMA_OBJECT_STORE_ROOT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_PROVIDER", "local")
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_ROOT_DIR", str(legacy_root))

    config = ObjectStoreConfig.from_env()
    assert config.provider == "local"
    assert config.root_dir == str(legacy_root)
    store, _ = cli._object_store_from_args(
        SimpleNamespace(object_store_provider="")
    )
    assert isinstance(store, LocalDirectoryObjectStore)
    assert store.root_dir == legacy_root

    monkeypatch.setenv("CACHEON_OBJECT_STORE_PROVIDER", "local")
    monkeypatch.setenv("CACHEON_OBJECT_STORE_ROOT_DIR", str(cacheon_root))
    assert ObjectStoreConfig.from_env().root_dir == str(cacheon_root)
    store, _ = cli._object_store_from_args(
        SimpleNamespace(object_store_provider="")
    )
    assert isinstance(store, LocalDirectoryObjectStore)
    assert store.root_dir == cacheon_root


def test_old_s3_environment_reaches_cli_publish_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = {
        "OPTIMA_OBJECT_STORE_PROVIDER": "s3",
        "OPTIMA_OBJECT_STORE_BUCKET": "legacy-bucket",
        "OPTIMA_OBJECT_STORE_KEY_PREFIX": "legacy/prefix",
        "OPTIMA_OBJECT_STORE_ENDPOINT_URL": "https://objects.example",
        "OPTIMA_OBJECT_STORE_REGION": "legacy-region",
        "OPTIMA_OBJECT_STORE_ACCESS_KEY_ID": "legacy-access",
        "OPTIMA_OBJECT_STORE_SECRET_ACCESS_KEY": "legacy-secret",
        "OPTIMA_OBJECT_STORE_ADDRESSING_STYLE": "path",
    }
    for old_name, value in legacy.items():
        new_name = old_name.replace("OPTIMA_", "CACHEON_", 1)
        monkeypatch.delenv(new_name, raising=False)
        monkeypatch.setenv(old_name, value)

    config = cli._s3_store_config_from_args(
        SimpleNamespace(),
        default_prefix="unused/default",
        prefix_env="CACHEON_OBJECT_STORE_KEY_PREFIX",
    )
    assert config.provider == "s3"
    assert config.bucket == "legacy-bucket"
    assert config.key_prefix == "legacy/prefix"
    assert config.endpoint_url == "https://objects.example"
    assert config.region_name == "legacy-region"
    assert config.access_key_id == "legacy-access"
    assert config.secret_access_key == "legacy-secret"
    assert config.addressing_style == "path"
