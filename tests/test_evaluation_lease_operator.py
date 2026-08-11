from __future__ import annotations

import json
from pathlib import Path

import pytest

import cacheon.cli as cli
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain import evaluation_lease_operator as operator
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.stack_identity import sha256_hex


SCOPE = IntakeScope("0x" + "0" * 64, 14)
POLICY = IntakePolicy(max_cohort=4, expiry_blocks=100)
BLOCK = 10


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _block_hash(block: int) -> str:
    return "0x" + f"{block:064x}"


def _arrival(
    index: int, *, block: int = BLOCK, invalid_reason: str = ""
) -> FinalizedArrival:
    return FinalizedArrival(
        hotkey=f"miner-{index}",
        content_hash="" if invalid_reason else _h(f"content:{index}:{block}"),
        url="" if invalid_reason else f"https://example.invalid/{index}.tar.gz",
        block=block,
        block_hash=_block_hash(block),
        event_index=index,
        invalid_reason=invalid_reason,
    )


def _new_database(
    tmp_path: Path,
    *,
    policy: IntakePolicy = POLICY,
    block: int = BLOCK,
) -> Path:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "intake.sqlite3"
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        store.reserve_finalized(
            (), finalized_block=block, finalized_block_hash=_block_hash(block)
        )
    return database


def _policy_dict(policy: IntakePolicy) -> dict[str, int]:
    return {
        name: getattr(policy, name) for name in policy.__dataclass_fields__
    }


def _seal(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o400)
    return path


def _config(
    tmp_path: Path,
    database: Path,
    *,
    policy: IntakePolicy = POLICY,
    stage: str = "screen",
    owner: str = "operator-a",
    lease_blocks: int = 20,
    qualification_max_members: int = 2,
    lock_attempts: int = 3,
    lock_retry_delay_ms: int = 0,
    name: str = "fifo-config.json",
) -> tuple[Path, dict[str, object]]:
    raw: dict[str, object] = {
        "intake_db": str(database),
        "intake_policy": _policy_dict(policy),
        "intake_scope": SCOPE.to_dict(),
        "lease_blocks": lease_blocks,
        "lock_attempts": lock_attempts,
        "lock_retry_delay_ms": lock_retry_delay_ms,
        "owner": owner,
        "qualification_max_members": qualification_max_members,
        "schema": operator.CONFIG_SCHEMA,
        "stage": stage,
    }
    return _seal(tmp_path / name, raw), raw


def _publish(store: FinalizedIntakeStore, row, target_id: str):
    marker = row.reservation_id[:12]
    store.mark_fetching(row.reservation_id)
    return store.mark_published(
        row.reservation_id,
        delta_fingerprint=SubmittedDeltaFingerprint(
            "component",
            target_id,
            _h(f"base:{marker}"),
            (f"slot.{marker}",),
            _h(f"archive:{marker}"),
            _h(f"selected:{marker}"),
            _h(f"exact:{marker}"),
            (_h(f"source:{marker}"),),
            (_h(f"binary:{marker}"),),
        ),
        publication_digest=_h(f"publication:{marker}"),
        publication_root=f"/published/{marker}",
    )


def _published_rows(
    database: Path,
    target_ids: tuple[str, ...],
    *,
    policy: IntakePolicy = POLICY,
) -> tuple[object, ...]:
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        rows = store.reserve_finalized(
            tuple(_arrival(index) for index in range(len(target_ids))),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )
        return tuple(
            _publish(store, row, target_id)
            for row, target_id in zip(rows, target_ids, strict=True)
        )


def _advance(
    database: Path,
    block: int,
    *,
    policy: IntakePolicy = POLICY,
) -> None:
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        store.reserve_finalized(
            (), finalized_block=block, finalized_block_hash=_block_hash(block)
        )


def _screen(
    store: FinalizedIntakeStore,
    reservation_id: str,
    *,
    service: str,
    decision: PromotionDecision,
) -> None:
    active = store.begin_screen(reservation_id, service_digest=service)
    candidate = _h(f"candidate:{reservation_id}:{active.screen_attempts}:{service[:8]}")
    if decision is PromotionDecision.PROMOTE:
        results = tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1)
            for stage in SCREEN_STAGES
        )
    else:
        # HOLD/RETRY terminate on the first stage with NO_DECISION.
        results = (
            ScreenStageResult(
                SCREEN_STAGES[0], ScreenGrade.NO_DECISION, _h("nd"), 1
            ),
        )
    receipt = ArenaScreenReceipt(
        service,
        candidate,
        active.screen_attempts,
        results,
        decision,
    )
    store.apply_screen_receipt(
        reservation_id, candidate_digest=candidate, receipt=receipt
    )


def _promote(store: FinalizedIntakeStore, reservation_id: str) -> None:
    _screen(
        store,
        reservation_id,
        service=_h("service"),
        decision=PromotionDecision.PROMOTE,
    )


def test_config_and_tracked_cli_are_closed(tmp_path: Path, capsys) -> None:
    database = _new_database(tmp_path)
    config_path, raw = _config(tmp_path, database)
    config = operator.load_config(config_path)
    assert config.policy == POLICY and config.scope == SCOPE
    assert config.intake_db == database and config.stage == "screen"

    parser = cli.build_parser()
    parsed = (
        ["preview"],
        ["claim"],
        ["heartbeat", "a" * 64],
        ["release", "a" * 64, "--reason", "worker_unavailable"],
        ["requeue-expired", "--authority", "/tmp/requeue-authority.json"],
    )
    for suffix in parsed:
        args = parser.parse_args(
            ["chain-evaluation-lease", "--config", str(config_path), *suffix]
        )
        assert args.func is cli.cmd_chain_evaluation_lease
    with pytest.raises(SystemExit):
        parser.parse_args(["chain-evaluation-lease", "preview"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["chain-evaluation-lease", "--config", str(config_path), "release", "a" * 64]
        )

    extra = dict(raw)
    extra["candidate_command"] = ["python", "candidate.py"]
    with pytest.raises(operator.FifoLeaseError, match="fields are not closed"):
        operator.load_config(_seal(tmp_path / "extra.json", extra))
    relative = dict(raw)
    relative["intake_db"] = "intake.sqlite3"
    with pytest.raises(operator.FifoLeaseError, match="absolute path"):
        operator.load_config(_seal(tmp_path / "relative.json", relative))
    excessive = dict(raw)
    excessive.update(lock_attempts=3, lock_retry_delay_ms=60_000)
    with pytest.raises(operator.FifoLeaseError, match="exceeds 60 seconds"):
        operator.load_config(_seal(tmp_path / "excessive.json", excessive))

    assert cli.main(
        ["chain-evaluation-lease", "--config", str(config_path), "preview"]
    ) == 0
    output = capsys.readouterr().out.strip()
    assert output == operator.canonical_json(json.loads(output))
    assert json.loads(output)["operation"] == "preview"


def test_malformed_stage_is_typed_and_cli_emits_no_success(
    tmp_path: Path, capsys
) -> None:
    database = _new_database(tmp_path)
    _, raw = _config(tmp_path, database)
    malformed = dict(raw)
    malformed["stage"] = []
    config_path = _seal(tmp_path / "malformed-stage.json", malformed)

    with pytest.raises(operator.FifoLeaseError, match="stage is unsupported"):
        operator.load_config(config_path)
    assert cli.main(
        ["chain-evaluation-lease", "--config", str(config_path), "preview"]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evaluation stage is unsupported" in captured.err


def test_tracked_cli_executes_all_four_operations(tmp_path: Path, capsys) -> None:
    database = _new_database(tmp_path)
    row = _published_rows(database, ("profile.screen.alpha",))[0]
    config_path, _ = _config(tmp_path, database, lease_blocks=5)
    prefix = ["chain-evaluation-lease", "--config", str(config_path)]

    assert cli.main([*prefix, "preview"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["reservation_ids"] == [row.reservation_id]
    assert cli.main([*prefix, "claim"]) == 0
    claimed = json.loads(capsys.readouterr().out)
    lease_id = claimed["lease"]["lease_id"]

    _advance(database, BLOCK + 1)
    assert cli.main([*prefix, "heartbeat", lease_id]) == 0
    heartbeat = json.loads(capsys.readouterr().out)
    assert heartbeat["lease"]["expires_block"] == BLOCK + 6
    digest = _h("worker-diagnostic")
    assert cli.main(
        [
            *prefix,
            "release",
            lease_id,
            "--reason",
            "worker_transport_unavailable",
            "--result-digest",
            digest,
        ]
    ) == 0
    released = json.loads(capsys.readouterr().out)
    assert (released["operation"], released["result_digest"]) == (
        "release",
        digest,
    )


@pytest.mark.parametrize(
    "target_ids",
    [
        ("profile.zeta.collective", "profile.alpha.block"),
        ("profile.alpha.block", "profile.zeta.collective"),
    ],
)
def test_target_identity_never_reorders_screen_fifo(
    tmp_path: Path, target_ids: tuple[str, str]
) -> None:
    database = _new_database(tmp_path)
    rows = _published_rows(database, target_ids)
    assert tuple(row.target_id for row in rows) == target_ids
    config_path, _ = _config(tmp_path, database)
    config = operator.load_config(config_path)
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        before = store.all()

    preview = operator.preview(config)
    assert preview["reservation_ids"] == [rows[0].reservation_id]
    assert preview["lease"] is None
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        assert store.all() == before
        assert store.active_evaluation_leases() == ()

    claimed = operator.claim(config)
    assert claimed["lease"]["members"] == [
        {"prior_status": "published", "reservation_id": rows[0].reservation_id}
    ]


def test_screen_reproduction_priority_remains_store_policy(tmp_path: Path) -> None:
    database = _new_database(tmp_path)
    rows = _published_rows(
        database, ("target.first", "target.second", "target.reproduction")
    )
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        # Fixture-only state isolation mirrors the canonical store regression:
        # the operator itself has no SQL or priority policy.
        store._db.execute(
            "UPDATE reservations SET status='reproduction_pending',"
            "screen_lane='reproduction' WHERE reservation_id=?",
            (rows[2].reservation_id,),
        )
    config_path, _ = _config(tmp_path, database)
    config = operator.load_config(config_path)

    assert operator.preview(config)["reservation_ids"] == [
        rows[2].reservation_id
    ]
    assert operator.claim(config)["lease"]["members"][0]["reservation_id"] == (
        rows[2].reservation_id
    )


def test_canonical_failed_and_expired_rows_are_not_claimed(tmp_path: Path) -> None:
    policy = IntakePolicy(max_cohort=4, expiry_blocks=5)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "intake.sqlite3"
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        expired, failed, current = store.reserve_finalized(
            (
                _arrival(0, block=10),
                _arrival(1, block=11, invalid_reason="malformed_submission"),
                _arrival(2, block=20),
            ),
            finalized_block=20,
            finalized_block_hash=_block_hash(20),
        )
        assert (expired.status, failed.status, current.status) == (
            "expired",
            "failed",
            "reserved",
        )
        current = _publish(store, current, "target.current")
    config_path, _ = _config(tmp_path, database, policy=policy, lease_blocks=3)
    config = operator.load_config(config_path)

    assert operator.preview(config)["reservation_ids"] == [
        current.reservation_id
    ]
    assert operator.claim(config)["lease"]["members"][0]["reservation_id"] == (
        current.reservation_id
    )


def test_sealed_downtime_requeue_restores_phase_and_one_bounded_sla(
    tmp_path: Path,
) -> None:
    policy = IntakePolicy(max_cohort=4, expiry_blocks=5)
    database = _new_database(tmp_path, policy=policy)
    published, promoted = _published_rows(
        database,
        ("profile.screen", "profile.qualification"),
        policy=policy,
    )
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        _promote(store, promoted.reservation_id)
    _advance(database, BLOCK + 5, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(published.reservation_id).status,
            store.get(promoted.reservation_id).status,
        } == {"expired"}

    authority = _seal(
        tmp_path / "downtime-requeue.json",
        {
            "reason": "validator_worker_unavailable",
            "reservation_ids": [published.reservation_id, promoted.reservation_id],
            "retained_result_reservation_ids": [_h("retained-result")],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    config_path, _ = _config(
        tmp_path,
        database,
        policy=policy,
        stage="qualification",
        lease_blocks=3,
    )
    config = operator.load_config(config_path)
    result = operator.requeue_expired(config, authority)
    assert [item["status"] for item in result["requeued"]] == [
        "published",
        "promoted",
    ]
    assert operator.preview(config)["reservation_ids"] == [promoted.reservation_id]

    _advance(database, BLOCK + 9, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.expire_stale(current_block=BLOCK + 9) == ()
    _advance(database, BLOCK + 10, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(published.reservation_id).status,
            store.get(promoted.reservation_id).status,
        } == {"expired"}

    # One refresh is admitted after the cohort re-expires under the automatic
    # SLA (operator fault / mismatched window); a third attempt fails closed.
    refresh_authority = _seal(
        tmp_path / "downtime-requeue-refresh.json",
        {
            "reason": "validator_worker_unavailable",
            "reservation_ids": [published.reservation_id, promoted.reservation_id],
            "retained_result_reservation_ids": [_h("retained-result-refresh")],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    refreshed = operator.requeue_expired(config, refresh_authority)
    assert [item["status"] for item in refreshed["requeued"]] == [
        "published",
        "promoted",
    ]
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.get(published.reservation_id).reason == (
            "validator_downtime_requeued_refresh"
        )
    _advance(database, BLOCK + 15, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(published.reservation_id).status,
            store.get(promoted.reservation_id).status,
        } == {"expired"}
    with pytest.raises(IntakeError, match="budget is already consumed"):
        operator.requeue_expired(config, refresh_authority)


def test_downtime_requeue_restores_midscreen_and_rotated_promote(
    tmp_path: Path,
) -> None:
    """Mid-screen hold/retry drop back to published; a promote that was
    rescreened under a new service identity (two append-only promote
    dispositions) still restores from the live receipt."""

    policy = IntakePolicy(max_cohort=4, expiry_blocks=5)
    database = _new_database(tmp_path, policy=policy)
    held, retried, rotated = _published_rows(
        database,
        ("profile.hold", "profile.retry", "profile.rotated"),
        policy=policy,
    )
    old_service = _h("retired-service")
    live_service = _h("live-service")
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        _screen(
            store,
            held.reservation_id,
            service=live_service,
            decision=PromotionDecision.HOLD,
        )
        _screen(
            store,
            retried.reservation_id,
            service=live_service,
            decision=PromotionDecision.RETRY,
        )
        _screen(
            store,
            rotated.reservation_id,
            service=old_service,
            decision=PromotionDecision.PROMOTE,
        )
        store.demote_promoted_for_rescreen(
            rotated.reservation_id, reason="service_rotated"
        )
        _screen(
            store,
            rotated.reservation_id,
            service=live_service,
            decision=PromotionDecision.PROMOTE,
        )
        assert store.get(held.reservation_id).screen_status == "hold"
        assert store.get(retried.reservation_id).screen_status == "retry"
        assert (
            store._db.execute(
                "SELECT COUNT(*) AS n FROM arena_screen_dispositions "
                "WHERE reservation_id=? AND decision='promote'",
                (rotated.reservation_id,),
            ).fetchone()["n"]
            == 2
        )

    _advance(database, BLOCK + 5, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(held.reservation_id).status,
            store.get(retried.reservation_id).status,
            store.get(rotated.reservation_id).status,
        } == {"expired"}

    authority = _seal(
        tmp_path / "midscreen-requeue.json",
        {
            "reason": "validator_worker_unavailable",
            "reservation_ids": [
                held.reservation_id,
                retried.reservation_id,
                rotated.reservation_id,
            ],
            "retained_result_reservation_ids": [],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    config_path, _ = _config(
        tmp_path,
        database,
        policy=policy,
        stage="qualification",
        lease_blocks=3,
    )
    result = operator.requeue_expired(operator.load_config(config_path), authority)
    assert [item["status"] for item in result["requeued"]] == [
        "published",
        "published",
        "promoted",
    ]
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        held_row = store.get(held.reservation_id)
        retried_row = store.get(retried.reservation_id)
        rotated_row = store.get(rotated.reservation_id)
        assert (held_row.status, held_row.screen_status) == ("published", "")
        assert (retried_row.status, retried_row.screen_status) == ("published", "")
        assert (rotated_row.status, rotated_row.screen_status) == (
            "promoted",
            "promote",
        )


def test_lock_retry_is_exact_bounded_and_preserves_one_lease(
    tmp_path: Path, monkeypatch
) -> None:
    database = _new_database(tmp_path)
    row = _published_rows(database, ("profile.lock",))[0]
    config_path, _ = _config(
        tmp_path, database, lock_attempts=3, lock_retry_delay_ms=2
    )
    config = operator.load_config(config_path)
    real_store = operator.FinalizedIntakeStore
    calls = 0
    sleeps: list[float] = []

    def contended(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise IntakeError("another intake controller owns this database")
        return real_store(*args, **kwargs)

    monkeypatch.setattr(operator, "FinalizedIntakeStore", contended)
    monkeypatch.setattr(operator.time, "sleep", sleeps.append)
    claimed = operator.claim(config)
    assert calls == 3 and sleeps == [0.002, 0.002]
    assert claimed["lease"]["members"][0]["reservation_id"] == row.reservation_id
    assert operator.claim(config)["lease"] is None

    calls = 0
    sleeps.clear()

    def wrong_scope(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise IntakeError("intake database belongs to another chain scope")

    monkeypatch.setattr(operator, "FinalizedIntakeStore", wrong_scope)
    with pytest.raises(IntakeError, match="another chain scope"):
        operator.preview(config)
    assert calls == 1 and sleeps == []


def test_heartbeat_reopens_current_exact_lease_and_rejects_stale(
    tmp_path: Path,
) -> None:
    database = _new_database(tmp_path)
    row = _published_rows(database, ("profile.heartbeat",))[0]
    config_path, _ = _config(tmp_path, database, lease_blocks=5)
    config = operator.load_config(config_path)
    lease_id = operator.claim(config)["lease"]["lease_id"]

    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as reopened:
        original = reopened.active_evaluation_leases()[0]
        assert original.lease_id == lease_id
        assert original.reservation_ids == (row.reservation_id,)

    _advance(database, BLOCK + 1)
    first = operator.heartbeat(config, lease_id)
    assert first["lease"]["expires_block"] == BLOCK + 6
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        current = store.active_evaluation_leases()[0]
        with pytest.raises(IntakeError, match="stale"):
            store.heartbeat_evaluation_lease(
                original, current_block=BLOCK + 1, lease_blocks=5
            )

    _advance(database, BLOCK + 2)
    assert operator.heartbeat(config, lease_id)["lease"]["expires_block"] == BLOCK + 7


def test_release_requeues_without_attempt_and_increments_generation(
    tmp_path: Path,
) -> None:
    database = _new_database(tmp_path)
    row = _published_rows(database, ("profile.release",))[0]
    config_path, _ = _config(tmp_path, database, lease_blocks=5)
    config = operator.load_config(config_path)
    lease_id = operator.claim(config)["lease"]["lease_id"]
    _advance(database, BLOCK + 1)

    operator.release(
        config,
        lease_id,
        reason="worker_transport_unavailable",
        result_digest=_h("infrastructure-diagnostic"),
    )
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        retained = store.get(row.reservation_id)
        assert (retained.status, retained.screen_attempts) == ("published", 0)
        assert store.active_evaluation_leases() == ()
        event = store.evaluation_lease_events(lease_id=lease_id)[-1]
        assert (event.event_type, event.reason) == (
            "released",
            "worker_transport_unavailable",
        )

    reclaimed = operator.claim(config)["lease"]
    assert reclaimed["generation"] == 2
    assert reclaimed["members"][0]["reservation_id"] == row.reservation_id


def test_qualification_cohort_uses_store_order_and_sealed_maximum(
    tmp_path: Path,
) -> None:
    database = _new_database(tmp_path)
    rows = _published_rows(
        database,
        ("profile.zeta.collective", "profile.alpha.block", "profile.middle"),
    )
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        for row in rows:
            _promote(store, row.reservation_id)
    config_path, _ = _config(
        tmp_path,
        database,
        stage="qualification",
        qualification_max_members=2,
    )
    config = operator.load_config(config_path)
    expected = [row.reservation_id for row in rows[:2]]

    assert operator.preview(config)["reservation_ids"] == expected
    lease = operator.claim(config)["lease"]
    assert lease["stage"] == "qualification"
    assert [member["reservation_id"] for member in lease["members"]] == expected


def test_unknown_stale_and_wrong_authority_lease_ids_fail_closed(
    tmp_path: Path,
) -> None:
    database = _new_database(tmp_path)
    _published_rows(database, ("profile.closed",))
    config_path, _ = _config(tmp_path, database)
    config = operator.load_config(config_path)

    with pytest.raises(operator.FifoLeaseError, match="SHA-256"):
        operator.heartbeat(config, "not-a-lease")
    with pytest.raises(operator.FifoLeaseError, match="was not found"):
        operator.heartbeat(config, _h("unknown-lease"))

    lease_id = operator.claim(config)["lease"]["lease_id"]
    other_path, _ = _config(
        tmp_path, database, owner="operator-b", name="other-owner.json"
    )
    other = operator.load_config(other_path)
    with pytest.raises(operator.FifoLeaseError, match="owner or stage"):
        operator.release(other, lease_id, reason="must_not_release_peer")
    with pytest.raises(operator.FifoLeaseError, match="bounded printable"):
        operator.release(config, lease_id, reason="bad\nreason")
    with pytest.raises(operator.FifoLeaseError, match="SHA-256"):
        operator.release(config, lease_id, reason="bad_digest", result_digest="abc")

    operator.release(config, lease_id, reason="operator_requeue")
    with pytest.raises(operator.FifoLeaseError, match="was not found"):
        operator.release(config, lease_id, reason="duplicate_release")
    with pytest.raises(operator.FifoLeaseError, match="was not found"):
        operator.heartbeat(config, lease_id)
