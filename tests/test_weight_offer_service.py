"""Weight-offer producer: sealed config, busy-database skip, failure breaker."""

from __future__ import annotations

import threading
import types
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.standing_weights_stage import (
    WEIGHTS_CONFIG_SCHEMA,
    WeightsStageConfig,
    compose_weight_offer_push,
)
from cacheon.chain.weight_offer_service import (
    CONFIG_SCHEMA,
    WeightOfferBusyError,
    WeightOfferServiceError,
    load_offer_service_config,
    main,
    run_forever,
)
from cacheon.chain.weight_push_auth import (
    PushCredentialSet,
    mint_push_credential,
    write_push_credentials,
)


def _private_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o400)


def _rewrite(path: Path, row: dict[str, object]) -> None:
    path.chmod(0o600)
    path.write_bytes(spool.spool_canonical_json(row) + b"\n")
    path.chmod(0o400)


def _setup(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    private = tmp_path / "offer-private"
    private.mkdir(mode=0o700)

    cred_path = private / "push-credentials.json"
    write_push_credentials(
        cred_path,
        PushCredentialSet((mint_push_credential(credential_id="test"),)),
    )

    weights_path = private / "weights-stage.json"
    _private_file(
        weights_path,
        spool.spool_canonical_json(
            {
                "attribution_hotkey": "validator",
                "burn_hotkey": "",
                "discovery_lifetime_blocks": 2160,
                "discovery_pool_ppm": 100_000,
                "fallback_endpoint": "",
                "half_life_blocks": 7200,
                "network": "wss://archive.example.invalid",
                "push_credentials": str(cred_path),
                "push_url": "http://127.0.0.1:8080",
                "refresh_blocks": 600,
                "schema": WEIGHTS_CONFIG_SCHEMA,
            }
        )
        + b"\n",
    )

    # load_offer_service_config only proves the screen authority is a sealed
    # owner-controlled file; build_offer_publisher is what fully reopens it.
    screen_path = private / "screen.json"
    _private_file(screen_path, b"{}\n")

    row: dict[str, object] = {
        "max_consecutive_failures": 10,
        "poll_ms": 60_000,
        "restart_initial_backoff_ms": 1_000,
        "restart_max_backoff_ms": 60_000,
        "schema": CONFIG_SCHEMA,
        "screen_dispatcher_config": str(screen_path),
        "weights_stage_config": str(weights_path),
    }
    config_path = private / "offer-service.json"
    _private_file(config_path, spool.spool_canonical_json(row) + b"\n")
    return config_path, row


def test_offer_service_config_reopens_exactly(tmp_path: Path) -> None:
    config_path, raw = _setup(tmp_path)
    config = load_offer_service_config(config_path)
    assert config.raw == raw
    assert config.poll_s == 60.0
    assert config.max_consecutive_failures == 10
    assert config.weights_stage.refresh_blocks == 600
    assert config.weights_stage.half_life_blocks == 7200


def test_unsupported_schema_fails_closed(tmp_path: Path) -> None:
    config_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["schema"] = "not-a-schema"
    _rewrite(config_path, bad)
    with pytest.raises(WeightOfferServiceError, match="schema"):
        load_offer_service_config(config_path)


def test_open_field_set_fails_closed(tmp_path: Path) -> None:
    config_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["unexpected"] = 1
    _rewrite(config_path, bad)
    with pytest.raises(WeightOfferServiceError, match="fields are not closed"):
        load_offer_service_config(config_path)


def test_backoff_ordering_fails_closed(tmp_path: Path) -> None:
    config_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["restart_initial_backoff_ms"] = 90_000
    _rewrite(config_path, bad)
    with pytest.raises(WeightOfferServiceError, match="exceeds its maximum"):
        load_offer_service_config(config_path)


def test_group_writable_config_fails_closed(tmp_path: Path) -> None:
    config_path, _ = _setup(tmp_path)
    config_path.chmod(0o664)
    with pytest.raises(WeightOfferServiceError, match="owner-controlled regular file"):
        load_offer_service_config(config_path)


def test_busy_error_passes_through_the_weights_stage_unwrapped(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole coexistence design rests on this.

    ``compose_weight_offer_push`` wraps foreign exceptions into a generic stage
    error, which would make a routine intake-controller lock collision
    indistinguishable from a real projection fault. ``WeightOfferBusyError``
    subclasses the stage error precisely so the stage re-raises it untouched.
    """

    from cacheon import chain

    cred_path = tmp_path / "push-credentials.json"
    write_push_credentials(
        cred_path,
        PushCredentialSet((mint_push_credential(credential_id="test"),)),
    )
    monkeypatch.setattr(chain, "connect", lambda _network, **_kw: object())
    monkeypatch.setattr(chain, "read_finalized_head", lambda _st: (100, "0x" + "0" * 64))
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _st, _netuid: types.SimpleNamespace(
            uids=[0], hotkeys=["validator"], block=100, block_hash="0x" + "0" * 64
        ),
    )

    def _busy() -> object:
        raise WeightOfferBusyError("intake controller owns the database this tick")

    stage = WeightsStageConfig(
        network="wss://archive.example.invalid",
        fallback_endpoint="",
        push_url="http://127.0.0.1:8080",
        push_credentials=cred_path,
        attribution_hotkey="validator",
        half_life_blocks=7200,
        discovery_lifetime_blocks=2160,
        discovery_pool_ppm=100_000,
        refresh_blocks=600,
        burn_hotkey="",
    )
    publish = compose_weight_offer_push(
        stage,
        store_factory=_busy,
        scope=types.SimpleNamespace(digest="f" * 64, netuid=14),
    )
    with pytest.raises(WeightOfferBusyError):
        publish()


def test_busy_database_is_a_skipped_pass_not_a_failure() -> None:
    stop = threading.Event()
    calls: list[int] = []
    events: list[str] = []

    def publish() -> object:
        calls.append(1)
        if len(calls) >= 3:
            stop.set()
        raise WeightOfferBusyError("busy")

    # max_consecutive_failures=2 would trip on the second pass if a busy
    # database were counted as a failure.
    run_forever(
        publish,
        stop,
        poll_s=0.0,
        max_consecutive_failures=2,
        wait=lambda _s: stop.is_set(),
        on_event=lambda kind, _detail: events.append(kind),
    )
    assert calls == [1, 1, 1]
    assert events == ["busy", "busy", "busy"]


def test_consecutive_failures_trip_the_circuit_breaker() -> None:
    stop = threading.Event()
    calls: list[int] = []

    def publish() -> object:
        calls.append(1)
        raise RuntimeError("projection exploded")

    with pytest.raises(RuntimeError, match="projection exploded"):
        run_forever(
            publish,
            stop,
            poll_s=0.0,
            restart_initial_backoff_s=0.0,
            restart_max_backoff_s=0.0,
            max_consecutive_failures=3,
            wait=lambda _s: False,
        )
    assert len(calls) == 3


def test_a_successful_pass_resets_the_failure_count() -> None:
    stop = threading.Event()
    events: list[str] = []
    script: list[object] = [
        RuntimeError("a"),
        RuntimeError("b"),
        None,
        RuntimeError("c"),
        RuntimeError("d"),
    ]

    def publish() -> object:
        if not script:
            stop.set()
            return None
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return None

    run_forever(
        publish,
        stop,
        poll_s=0.0,
        restart_initial_backoff_s=0.0,
        restart_max_backoff_s=0.0,
        max_consecutive_failures=3,
        wait=lambda _s: stop.is_set(),
        on_event=lambda kind, _detail: events.append(kind),
    )
    assert events == ["failed", "failed", "idle", "failed", "failed", "idle"]


def test_idle_and_pushed_passes_are_reported_distinctly() -> None:
    stop = threading.Event()
    events: list[tuple[str, str]] = []
    script: list[object] = [
        None,
        types.SimpleNamespace(disposition="accepted", request_id="d" * 64),
    ]

    def publish() -> object:
        if not script:
            stop.set()
            return None
        return script.pop(0)

    run_forever(
        publish,
        stop,
        poll_s=0.0,
        wait=lambda _s: stop.is_set(),
        on_event=lambda kind, detail: events.append((kind, detail)),
    )
    assert events[0] == ("idle", "refresh window not reached")
    assert events[1][0] == "pushed"
    assert "d" * 64 in events[1][1]


def test_main_returns_2_on_missing_config(tmp_path: Path) -> None:
    assert main(["--config", str(tmp_path / "missing.json")]) == 2
