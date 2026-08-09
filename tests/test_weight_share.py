"""Tests for permit-gated current-weight sharing and follower publish reuse."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon import chain
from cacheon.chain.weight_share import (
    AuthenticatedWeightOffer,
    CURRENT_WEIGHTS_PATH,
    LANE_LEGACY_V1,
    OFFER_SCHEMA,
    CurrentWeightOffer,
    WeightShareError,
    WeightShareRetryableError,
    authenticate_weight_offer,
    assert_fresh_timestamp,
    assert_validator_permit,
    build_signed_offer_response,
    default_offer_path,
    fetch_current_weights,
    load_offer_storage_from_store,
    object_store_offer_loader,
    object_store_offer_sink,
    parse_signed_offer_response,
    publish_followed_weights,
    push_current_weights,
    read_offer_storage,
    read_current_weight_offer,
    rebind_offer_signer,
    rebind_projection_signer,
    request_auth_digest,
    serve_current_weights,
    sign_auth_digest,
    verify_authenticated_weight_offer,
    write_current_weight_offer,
)
from cacheon.chain.weights import WeightProjection, WeightPublicationRecord
from cacheon.object_store import MemoryObjectStore
from cacheon.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex


def _d(label: str) -> str:
    return sha256_hex(label.encode())


def _projection(
    *,
    hotkey: str = "authority",
    block: int = 10,
    recipient: str = "miner",
) -> WeightProjection:
    scope = _d("scope")
    metagraph_digest = canonical_digest(
        "cacheon.economics.metagraph-membership",
        {
            "block": block,
            "block_hash": "0x" + f"{block:064x}",
            "chain_scope_digest": scope,
            "members": [
                {"hotkey": "authority", "uid": 0},
                {"hotkey": "follower", "uid": 1},
                {"hotkey": recipient, "uid": 2},
            ],
        },
    )
    return WeightProjection(
        scope,
        307,
        hotkey,
        _d("policy"),
        _d("settlement"),
        _d("evaluation"),
        metagraph_digest,
        (_d("arena"),),
        1,
        block,
        1,
        (_d("evidence"),),
        ((recipient, 1_000_000),),
    )


class _FakeHotkey:
    def __init__(self, address: str, secret: bytes) -> None:
        self.ss58_address = address
        self._secret = secret

    def sign(self, data: bytes) -> bytes:
        return hashlib.sha256(self._secret + data).digest()


def _verify_factory(secrets: dict[str, bytes]):
    def verify(hotkey: str, message: bytes, signature: bytes) -> bool:
        secret = secrets.get(hotkey)
        if secret is None:
            return False
        return signature == hashlib.sha256(secret + message).digest()

    return verify


def _view(block: int = 10) -> chain.MetagraphView:
    return chain.MetagraphView(
        307,
        block,
        "0x" + f"{block:064x}",
        [0, 1, 2],
        ["authority", "follower", "miner"],
        [True, True, False],
        [0, 0, 0],
    )


def test_offer_roundtrip_and_default_path(tmp_path: Path) -> None:
    intake = tmp_path / "intake.sqlite3"
    path = default_offer_path(intake)
    assert path.name == "intake.sqlite3.current_weights.json"
    projection = _projection()
    write_current_weight_offer(path, projection)
    offer = read_current_weight_offer(path)
    assert offer.lane == LANE_LEGACY_V1
    assert offer.projection == projection
    assert offer.digest == CurrentWeightOffer.from_legacy_projection(projection).digest
    raw = json.loads(path.read_text())
    assert raw["schema"] == OFFER_SCHEMA
    assert raw["projection_digest"] == projection.digest
    assert raw["lane"] == LANE_LEGACY_V1


def test_rebind_keeps_weights_changes_signer() -> None:
    projection = _projection()
    rebound = rebind_projection_signer(projection, "follower")
    assert rebound.validator_hotkey == "follower"
    assert rebound.weights_ppm == projection.weights_ppm
    assert rebound.digest != projection.digest
    assert rebind_projection_signer(projection, "authority") is projection


def test_debt_lane_offers_are_rejected_after_extraction() -> None:
    offer = CurrentWeightOffer.from_legacy_projection(_projection())
    row = offer.to_dict()
    assert row["debt_binding"] is None
    assert row["debt_binding_digest"] is None
    with pytest.raises(WeightShareError, match="lane is unsupported"):
        CurrentWeightOffer("finite_debt", _projection(), None)
    stale = dict(row)
    stale["debt_binding"] = {"publication_kind": "finite_debt_core"}
    stale["debt_binding_digest"] = "0" * 64
    with pytest.raises(WeightShareError, match="retired"):
        CurrentWeightOffer.from_dict(stale)


def test_timestamp_skew_and_permit_gates() -> None:
    assert_fresh_timestamp(100, now=130, max_skew_seconds=60)
    with pytest.raises(WeightShareError, match="skew"):
        assert_fresh_timestamp(100, now=200, max_skew_seconds=60)
    view = _view()
    assert assert_validator_permit(view, "authority") == 0
    with pytest.raises(WeightShareError, match="validator_permit"):
        assert_validator_permit(view, "miner")
    with pytest.raises(WeightShareError, match="not registered"):
        assert_validator_permit(view, "missing")


def test_signed_response_rejects_tampered_body() -> None:
    secrets = {"authority": b"auth-secret"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    offer = CurrentWeightOffer.from_legacy_projection(_projection())
    body, headers = build_signed_offer_response(
        offer, authority=authority, netuid=307, timestamp=1_700_000_000
    )
    parsed = parse_signed_offer_response(
        body,
        headers,
        netuid=307,
        now=1_700_000_000,
        max_skew_seconds=60,
        verify=verify,
        metagraph=_view(),
    )
    assert parsed.projection.digest == offer.projection.digest

    tampered = body.replace(b"miner", b"Miner")
    with pytest.raises(WeightShareError, match="digest mismatch|signature"):
        parse_signed_offer_response(
            tampered,
            headers,
            netuid=307,
            now=1_700_000_000,
            max_skew_seconds=60,
            verify=verify,
            metagraph=_view(),
        )


def test_http_endpoint_requires_permit_and_fresh_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = {"authority": b"auth", "follower": b"follow", "miner": b"miner"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    follower = _FakeHotkey("follower", secrets["follower"])
    miner = _FakeHotkey("miner", secrets["miner"])
    offer_path = tmp_path / "offer.json"
    write_current_weight_offer(offer_path, _projection())

    monkeypatch.setattr(chain, "fetch_metagraph", lambda *_a, **_k: _view())
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        offer_path=offer_path,
        authority=authority,
        subtensor=object(),
        netuid=307,
        max_skew_seconds=60,
        verify=verify,
        clock=lambda: 1_700_000_100,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        base = f"http://127.0.0.1:{port}"
        offer = fetch_current_weights(
            base,
            signer=follower,
            netuid=307,
            max_skew_seconds=60,
            verify=verify,
            clock=lambda: 1_700_000_100,
            expected_authority="authority",
            metagraph=_view(),
        )
        assert offer.projection.weights_ppm == (("miner", 1_000_000),)

        with pytest.raises(WeightShareError, match="rejected|permit|validator_permit"):
            fetch_current_weights(
                base,
                signer=miner,
                netuid=307,
                max_skew_seconds=60,
                verify=verify,
                clock=lambda: 1_700_000_100,
                metagraph=_view(),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_endpoint_preserves_retryable_store_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {"authority": b"auth", "follower": b"follow"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    follower = _FakeHotkey("follower", secrets["follower"])

    def load():
        raise WeightShareRetryableError("temporary object-store outage")

    monkeypatch.setattr(chain, "fetch_metagraph", lambda *_a, **_k: _view())
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        load_offer=load,
        authority=authority,
        subtensor=object(),
        netuid=307,
        verify=verify,
        clock=lambda: 1_700_000_100,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(
            WeightShareRetryableError, match="503"
        ) as raised:
            fetch_current_weights(
                f"http://127.0.0.1:{server.server_address[1]}",
                signer=follower,
                netuid=307,
                verify=verify,
                clock=lambda: 1_700_000_100,
                metagraph=_view(),
            )
        assert raised.value.retryable is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fetch_binds_request_signature_over_timestamp() -> None:
    secrets = {"follower": b"follow"}
    follower = _FakeHotkey("follower", secrets["follower"])
    captured: dict[str, object] = {}

    class _Resp:
        status = 200
        headers = {
            "X-Cacheon-Authority-Hotkey": "authority",
            "X-Cacheon-Netuid": "307",
            "X-Cacheon-Timestamp": "100",
            "X-Cacheon-Signature": "00",
            "X-Cacheon-Body-Digest": "00",
        }

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def opener(request, timeout=30):
        captured["headers"] = dict(request.headers.items())
        captured["timeout"] = timeout
        return _Resp()

    with pytest.raises(WeightShareError):
        fetch_current_weights(
            "http://example.test",
            signer=follower,
            netuid=307,
            clock=lambda: 100,
            opener=opener,
            verify=lambda *_a: False,
        )
    headers = {str(k).lower(): v for k, v in captured["headers"].items()}  # type: ignore[union-attr]
    assert headers["x-cacheon-hotkey"] == "follower"
    assert headers["x-cacheon-timestamp"] == "100"
    digest = request_auth_digest(
        hotkey="follower",
        method="GET",
        netuid=307,
        path=CURRENT_WEIGHTS_PATH,
        timestamp=100,
    )
    assert headers["x-cacheon-signature"] == sign_auth_digest(follower, digest)


class _Journal:
    def __init__(self) -> None:
        self.row = None
        self.history = []

    def load(self):
        return self.row

    def compare_and_swap(self, expected_record_digest, replacement):
        assert expected_record_digest == (self.row.digest if self.row else None)
        self.row = replacement
        self.history.append(replacement)

    def retained_projection(self, projection_digest):
        raise AssertionError("not used")


def test_publish_followed_weights_uses_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer = CurrentWeightOffer.from_legacy_projection(_projection())
    wallet = SimpleNamespace(hotkey=_FakeHotkey("follower", b"f"))
    journal = _Journal()
    seen = {}

    def fake_reconcile(
        subtensor,
        signer_wallet,
        projection,
        journal_arg,
        *,
        refresh_blocks,
        dry_run=False,
        reconcile_only=False,
        allow_stale_initial=False,
        max_stale_initial_blocks=None,
        require_current_crown=True,
    ):
        seen["projection"] = projection
        seen["wallet"] = signer_wallet
        seen["refresh_blocks"] = refresh_blocks
        seen["dry_run"] = dry_run
        seen["allow_stale_initial"] = allow_stale_initial
        seen["max_stale_initial_blocks"] = max_stale_initial_blocks
        seen["require_current_crown"] = require_current_crown
        assert journal_arg is journal
        return SimpleNamespace(
            projection_digest=projection.digest,
            status="dry_run",
            chain_matches=False,
            submitted=False,
            refresh_due=False,
        )

    monkeypatch.setattr(
        "cacheon.chain.weight_share.reconcile_weight_publication",
        fake_reconcile,
    )
    result = publish_followed_weights(
        subtensor=object(),
        signer_wallet=wallet,
        offer=offer,
        journal=journal,
        refresh_blocks=100,
        dry_run=True,
    )
    assert result.status == "dry_run"
    assert seen["projection"].validator_hotkey == "follower"
    assert seen["projection"].weights_ppm == offer.projection.weights_ppm
    assert seen["wallet"] is None
    assert seen["dry_run"] is True
    assert seen["allow_stale_initial"] is True
    assert seen["max_stale_initial_blocks"] == 100
    assert seen["require_current_crown"] is True


def test_push_endpoint_accepts_rotatable_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.chain.weight_push_auth import (
        PushCredentialSet,
        mint_push_credential,
        write_push_credentials,
    )

    secrets = {"authority": b"auth", "follower": b"follow"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    follower = _FakeHotkey("follower", secrets["follower"])
    offer_path = tmp_path / "offer.json"
    write_current_weight_offer(
        offer_path, CurrentWeightOffer.from_legacy_projection(_projection())
    )
    credential = mint_push_credential(credential_id="eval-1")
    cred_path = tmp_path / "push.json"
    write_push_credentials(cred_path, PushCredentialSet((credential,)))

    monkeypatch.setattr(chain, "fetch_metagraph", lambda *_a, **_k: _view())
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        offer_path=offer_path,
        authority=authority,
        subtensor=object(),
        netuid=307,
        max_skew_seconds=60,
        verify=verify,
        clock=lambda: 1_700_000_100,
        push_credentials=PushCredentialSet((credential,)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        base = f"http://127.0.0.1:{port}"
        updated_offer = CurrentWeightOffer.from_legacy_projection(
            _projection(block=11)
        )
        accepted = push_current_weights(
            base,
            updated_offer,
            credential=credential,
            clock=lambda: 1_700_000_100,
        )
        assert accepted["status"] == "accepted"
        assert accepted["offer_digest"] == updated_offer.digest
        assert accepted["projection_digest"] == updated_offer.projection.digest
        assert accepted["credential_id"] == credential.credential_id
        assert accepted["request_timestamp"] == 1_700_000_100
        assert accepted["schema"] == "cacheon.weight-share.push-ack.v1"
        assert isinstance(accepted["mac"], str)
        stored = read_offer_storage(offer_path)
        assert isinstance(stored, AuthenticatedWeightOffer)
        assert (
            verify_authenticated_weight_offer(
                stored, PushCredentialSet((credential,))
            )
            == updated_offer
        )
        with pytest.raises(WeightShareError, match="authenticated storage"):
            read_current_weight_offer(offer_path)
        fetched = fetch_current_weights(
            base,
            signer=follower,
            netuid=307,
            clock=lambda: 1_700_000_100,
            verify=verify,
            metagraph=_view(),
        )
        assert fetched.lane == LANE_LEGACY_V1
        assert fetched.digest == updated_offer.digest
        bad = mint_push_credential(credential_id="intruder")
        with pytest.raises(WeightShareError, match="rejected"):
            push_current_weights(
                base,
                updated_offer,
                credential=bad,
                clock=lambda: 1_700_000_100,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_push_enabled_gateway_refuses_forged_object_store_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cacheon.chain.weight_push_auth import (
        PushCredential,
        PushCredentialSet,
        mint_push_credential,
    )

    secrets = {"authority": b"auth", "follower": b"follow"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    follower = _FakeHotkey("follower", secrets["follower"])
    credential = mint_push_credential(credential_id="eval")
    credentials = PushCredentialSet((credential,))
    store = MemoryObjectStore()

    monkeypatch.setattr(chain, "fetch_metagraph", lambda *_a, **_k: _view())
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        load_offer=object_store_offer_loader(store),
        save_offer=object_store_offer_sink(store),
        push_credentials=credentials,
        authority=authority,
        subtensor=object(),
        netuid=307,
        verify=verify,
        clock=lambda: 1_700_000_100,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        legitimate = CurrentWeightOffer.from_legacy_projection(_projection())
        push_current_weights(
            base,
            legitimate,
            credential=credential,
            clock=lambda: 1_700_000_100,
        )
        stored = load_offer_storage_from_store(store)
        assert isinstance(stored, AuthenticatedWeightOffer)
        assert verify_authenticated_weight_offer(stored, credentials) == legitimate

        forged = CurrentWeightOffer.from_legacy_projection(
            _projection(recipient="attacker")
        )
        wrong_secret = PushCredential(
            credential.credential_id,
            "attacker-secret-" * 4,
        )
        store.put_bytes(
            "current_weights.json",
            authenticate_weight_offer(forged, wrong_secret).to_bytes(),
        )
        with pytest.raises(WeightShareError, match="rejected"):
            fetch_current_weights(
                base,
                signer=follower,
                netuid=307,
                verify=verify,
                clock=lambda: 1_700_000_100,
                metagraph=_view(),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_push_client_rejects_an_incomplete_acknowledgement() -> None:
    from cacheon.chain.weight_push_auth import mint_push_credential

    credential = mint_push_credential(credential_id="eval")
    offer = CurrentWeightOffer.from_legacy_projection(_projection())

    class _Response:
        status = 200

        def read(self):
            return b'{"credential_id":"eval","status":"accepted"}\n'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    with pytest.raises(WeightShareError, match="fields do not match"):
        push_current_weights(
            "http://weights.example",
            offer,
            credential=credential,
            clock=lambda: 1_700_000_100,
            opener=lambda *_a, **_k: _Response(),
        )


def test_push_client_rejects_an_exact_acknowledgement_with_wrong_mac() -> None:
    from cacheon.chain.weight_push_auth import (
        PushCredential,
        mint_push_credential,
        sign_push_acknowledgement,
    )

    credential = mint_push_credential(credential_id="eval")
    offer = CurrentWeightOffer.from_legacy_projection(_projection())
    impostor = PushCredential(
        credential.credential_id,
        "impostor-secret-that-is-long-enough-" * 2,
    )
    forged = sign_push_acknowledgement(
        impostor,
        offer_digest=offer.digest,
        projection_digest=offer.projection.digest,
        request_timestamp=1_700_000_100,
    )

    class _Response:
        status = 200

        def read(self):
            return canonical_json_bytes(forged) + b"\n"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    with pytest.raises(WeightShareError, match="authentication failed"):
        push_current_weights(
            "http://weights.example",
            offer,
            credential=credential,
            clock=lambda: 1_700_000_100,
            opener=lambda *_a, **_k: _Response(),
        )


def test_push_weight_offer_cli_never_calls_set_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cacheon.cli as cli
    from cacheon.chain.intake import FinalizedIntakeStore, IntakeScope
    from cacheon.chain.weight_push_auth import (
        PushCredentialSet,
        mint_push_credential,
        write_push_credentials,
    )

    scope = IntakeScope("0x" + "0" * 64, 307)
    intake = tmp_path / "intake.sqlite3"
    with FinalizedIntakeStore(intake, scope=scope) as store:
        store.reserve_finalized(
            (),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )

    credential = mint_push_credential(credential_id="eval")
    cred_path = tmp_path / "push.json"
    write_push_credentials(cred_path, PushCredentialSet((credential,)))
    projection = _projection(hotkey="authority")
    pushed = {}

    class _Subtensor:
        def get_block_hash(self, block):
            return "0x" + "0" * 64

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "read_finalized_head",
        lambda _st: (10, "0x" + f"{10:064x}"),
    )
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda *_a, **_k: _view(),
    )
    monkeypatch.setattr(
        FinalizedIntakeStore,
        "build_weight_projection",
        lambda *_a, **_k: projection,
    )
    monkeypatch.setattr(
        FinalizedIntakeStore,
        "evaluation_stacks",
        lambda *_a, **_k: (),
    )

    def fake_push(url, offer, *, credential, **_k):
        pushed["url"] = url
        pushed["offer"] = offer
        pushed["credential"] = credential.credential_id
        return {
            "status": "accepted",
            "credential_id": credential.credential_id,
            "offer_digest": offer.digest,
        }

    monkeypatch.setattr("cacheon.chain.weight_share.push_current_weights", fake_push)
    monkeypatch.setattr(
        "cacheon.chain.weights.reconcile_weight_publication",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("eval must not publish to chain")
        ),
    )

    args = SimpleNamespace(
        intake_db=str(intake),
        netuid=307,
        network="finney",
        url="http://weights.example",
        push_credentials=str(cred_path),
        credential_id="",
        attribution_hotkey="authority",
        half_life_blocks=100,
        discovery_lifetime_blocks=200,
        discovery_pool_ppm=50_000,
        burn_hotkey="",
        weight_offer_path="",
        dry_run=False,
    )
    assert cli.cmd_push_weight_offer(args) == 0
    assert pushed["url"] == "http://weights.example"
    assert pushed["offer"].lane == LANE_LEGACY_V1
    assert pushed["credential"] == "eval"


@pytest.mark.parametrize(
    ("dry_run", "status", "return_code", "published"),
    [
        (True, "dry_run", 0, False),
        (False, "held", 2, False),
        (False, "confirmed", 0, True),
    ],
)
def test_set_weights_publishes_only_after_accepted_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
    status: str,
    return_code: int,
    published: bool,
) -> None:
    import cacheon.cli as cli
    from cacheon.chain.intake import FinalizedIntakeStore, IntakeScope

    scope = IntakeScope("0x" + "0" * 64, 307)
    path = tmp_path / "private" / "intake.sqlite3"
    with FinalizedIntakeStore(path, scope=scope):
        pass

    projection = _projection(hotkey="validator")
    offer_path = tmp_path / "offer.json"

    class _Subtensor:
        def get_block_hash(self, block):
            return "0x" + "0" * 64

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda *_a, **_k: chain.MetagraphView(
            307,
            10,
            "0x" + f"{10:064x}",
            [0, 1],
            ["validator", "miner"],
            [True, True],
            [0, 0],
        ),
    )

    def fake_build(self, **_kwargs):
        return projection

    monkeypatch.setattr(
        FinalizedIntakeStore, "build_burn_weight_projection", fake_build
    )

    def fake_reconcile(*_a, **_k):
        return SimpleNamespace(
            status=status,
            chain_matches=False,
            submitted=False,
            refresh_due=False,
        )

    monkeypatch.setattr(
        "cacheon.chain.weights.reconcile_weight_publication", fake_reconcile
    )
    monkeypatch.setattr(
        cli,
        "_cmd_set_weights_once",
        cli._cmd_set_weights_once,
    )

    import sys
    import types

    class _Hotkey:
        ss58_address = "validator"

    class _Wallet:
        def __init__(self, name, hotkey):
            self.hotkey = _Hotkey()

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Wallet=_Wallet)
    )

    args = SimpleNamespace(
        reconcile_only=False,
        dry_run=dry_run,
        release_hold="",
        burn_hotkey="miner",
        burn_to_subnet_owner=False,
        validator_hotkey="",
        wallet="default",
        hotkey="default",
        intake_db=str(path),
        netuid=307,
        network="finney",
        half_life_blocks=100,
        discovery_lifetime_blocks=20,
        discovery_pool_ppm=100_000,
        refresh_blocks=10,
        weight_offer_path=str(offer_path),
        watch=False,
    )
    assert cli.cmd_set_weights(args) == return_code
    assert offer_path.exists() is published
    if published:
        stored = read_current_weight_offer(offer_path)
        assert stored.projection.digest == projection.digest
