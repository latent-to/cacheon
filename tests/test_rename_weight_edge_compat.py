"""Single-vocabulary weight-share edge: Cacheon headers and schemas only."""

from __future__ import annotations

import pytest

from cacheon.chain.weight_push_auth import (
    CREDENTIALS_SCHEMA,
    ENV_PUSH_CREDENTIAL_ID,
    ENV_PUSH_CREDENTIALS,
    ENV_PUSH_KEY,
    PUSH_ACK_SCHEMA,
    PushCredential,
    PushCredentialSet,
    WeightPushAuthError,
    sign_push_acknowledgement,
    sign_push_request,
    verify_push_acknowledgement,
    verify_push_request,
)
from cacheon.chain.weight_share import (
    OFFER_SCHEMA,
    STORED_OFFER_SCHEMA,
    CurrentWeightOffer,
    WeightShareError,
)


SECRET = "s" * 48
NOW = 1_700_000_000


def _credential() -> PushCredential:
    return PushCredential("edge", SECRET, "active")


def test_weight_edge_vocabulary_is_cacheon() -> None:
    assert ENV_PUSH_CREDENTIALS == "CACHEON_WEIGHT_PUSH_CREDENTIALS"
    assert ENV_PUSH_KEY == "CACHEON_WEIGHT_PUSH_KEY"
    assert ENV_PUSH_CREDENTIAL_ID == "CACHEON_WEIGHT_PUSH_CREDENTIAL_ID"
    assert CREDENTIALS_SCHEMA == "cacheon.weight-push-credentials.v1"
    assert PUSH_ACK_SCHEMA == "cacheon.weight-share.push-ack.v1"
    assert OFFER_SCHEMA == "cacheon.current-weight-offer.v2"
    assert STORED_OFFER_SCHEMA == "cacheon.authenticated-weight-offer.v1"


def test_retired_credentials_schema_is_refused() -> None:
    value = {
        "credentials": [
            {"credential_id": "edge", "secret": SECRET, "status": "active"}
        ],
        "schema": "optima.weight-push-credentials.v1",
    }
    with pytest.raises(WeightPushAuthError, match="schema is unsupported"):
        PushCredentialSet.from_dict(value)


def test_push_headers_are_cacheon_and_verify_round_trip() -> None:
    body = b"{}\n"
    headers = sign_push_request(_credential(), timestamp=NOW, body=body)
    named = sorted(key for key in headers if key != "Content-Type")
    assert named == [
        "X-Cacheon-Push-Body-Digest",
        "X-Cacheon-Push-Credential-Id",
        "X-Cacheon-Push-Signature",
        "X-Cacheon-Push-Timestamp",
    ]
    credentials = PushCredentialSet((_credential(),))
    accepted = verify_push_request(
        credentials, headers=headers, body=body, now=NOW
    )
    assert accepted == "edge"


def test_retired_push_header_family_fails_closed() -> None:
    body = b"{}\n"
    headers = sign_push_request(_credential(), timestamp=NOW, body=body)
    renamed = {
        (
            "X-Optima" + key[len("X-Cacheon"):]
            if key.startswith("X-Cacheon")
            else key
        ): value
        for key, value in headers.items()
    }
    credentials = PushCredentialSet((_credential(),))
    with pytest.raises(WeightPushAuthError):
        verify_push_request(credentials, headers=renamed, body=body, now=NOW)


def test_retired_offer_schema_is_refused() -> None:
    with pytest.raises(WeightShareError, match="schema is unsupported"):
        CurrentWeightOffer.from_dict(
            {
                "lane": "legacy_v1",
                "projection": {},
                "projection_digest": "0" * 64,
                "schema": "optima.current-weight-offer.v2",
                "debt_binding": None,
                "debt_binding_digest": None,
            }
        )


def test_push_acknowledgement_accepts_cacheon_and_refuses_retired_schema() -> None:
    ack = sign_push_acknowledgement(
        _credential(),
        offer_digest="1" * 64,
        projection_digest="2" * 64,
        request_timestamp=5,
    )
    assert ack["schema"] == PUSH_ACK_SCHEMA
    verify_push_acknowledgement(
        _credential(),
        ack,
        offer_digest="1" * 64,
        projection_digest="2" * 64,
        request_timestamp=5,
    )
    with pytest.raises(WeightPushAuthError, match="schema is unsupported"):
        verify_push_acknowledgement(
            _credential(),
            {**ack, "schema": "optima.weight-share.push-ack.v1"},
            offer_digest="1" * 64,
            projection_digest="2" * 64,
            request_timestamp=5,
        )
