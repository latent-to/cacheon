"""Weight-offer producer loop accounting and publication-record chronology.

The sealed config, its refusals, the busy skip before a reward checkpoint, the failure breaker
and the pushed/idle reports run end to end in ``tests/test_validator_weight_service.py``.
``_setup`` and ``_rewrite`` build the sealed producer configs the arena-allocation tests reuse.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.standing_weights_stage import WEIGHTS_CONFIG_SCHEMA
from cacheon.chain.weight_offer_service import CONFIG_SCHEMA, WeightOfferBusyError, run_forever
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

    _private_file(private / "signer.sqlite3", b"")
    weights_path = private / "weights-stage.json"
    _private_file(
        weights_path,
        spool.spool_canonical_json(
            {
                "attribution_hotkey": "validator",
                "burn_hotkey": "",
                "confirmation_journal": str(private / "signer.sqlite3"),
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


@pytest.mark.parametrize("reason,last_update,confirmed,valid", [
    ("block_inclusion", 0, 104, True),
    ("block_inclusion", 0, 99, False),
    ("block_inclusion", 105, 104, False),
    ("post_submit_authoritative_readback", 0, 104, False),
    ("post_submit_authoritative_readback", 102, 104, True),
])
def test_reopen_included_weight_record_before_active_readback(reason, last_update, confirmed, valid):
    from cacheon.chain.weights import WeightPublicationError, WeightPublicationRecord
    from cacheon.stack_identity import canonical_digest

    wire = dict(projection_digest="a" * 64, status="confirmed", prior_record_digest=None,
                submit_block=100, retry_after_block=700, reveal_round=23,
                confirmed_block=confirmed, confirmed_last_update=last_update, reason=reason)
    if not valid:
        with pytest.raises(WeightPublicationError, match="chronology"):
            WeightPublicationRecord.from_dict(wire)
        return
    record = WeightPublicationRecord.from_dict(wire)
    assert record.to_dict() == wire
    assert record.digest == canonical_digest("cacheon.chain.weight-publication", wire)
