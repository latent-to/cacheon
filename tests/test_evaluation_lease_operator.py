from __future__ import annotations

import json
from pathlib import Path

import pytest

import cacheon.cli as cli
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
        "stage": "qualification",
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


def test_config_and_tracked_cli_are_closed(tmp_path: Path, capsys) -> None:
    database = _new_database(tmp_path)
    config_path, raw = _config(tmp_path, database)
    config = operator.load_config(config_path)
    assert config.policy == POLICY and config.scope == SCOPE
    assert config.intake_db == database and config.stage == "qualification"

    parser = cli.build_parser()
    parsed = (
        ["preview"],
        ["claim"],
        ["requeue-expired", "--authority", "/tmp/requeue-authority.json"],
    )
    for suffix in parsed:
        args = parser.parse_args(
            ["chain-evaluation-lease", "--config", str(config_path), *suffix]
        )
        assert args.func is cli.cmd_chain_evaluation_lease
    with pytest.raises(SystemExit):
        parser.parse_args(["chain-evaluation-lease", "preview"])
    # Every lease is a qualification lease and moves only through its
    # recovery-owned transitions, so the operator has no heartbeat or release.
    for retired in ("heartbeat", "release"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["chain-evaluation-lease", "--config", str(config_path), retired, "a" * 64]
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
    retired = _seal(tmp_path / "retired-stage.json", {**raw, "stage": "screen"})

    with pytest.raises(operator.FifoLeaseError, match="stage is unsupported"):
        operator.load_config(config_path)
    with pytest.raises(operator.FifoLeaseError, match="stage is unsupported"):
        operator.load_config(retired)
    assert cli.main(
        ["chain-evaluation-lease", "--config", str(config_path), "preview"]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evaluation stage is unsupported" in captured.err


def test_tracked_cli_previews_and_claims_one_lease(tmp_path: Path, capsys) -> None:
    database = _new_database(tmp_path)
    row = _published_rows(database, ("profile.alpha",))[0]
    config_path, _ = _config(tmp_path, database, lease_blocks=5)
    prefix = ["chain-evaluation-lease", "--config", str(config_path)]

    assert cli.main([*prefix, "preview"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["reservation_ids"] == [row.reservation_id]
    assert cli.main([*prefix, "claim"]) == 0
    claimed = json.loads(capsys.readouterr().out)
    lease_id = claimed["lease"]["lease_id"]
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        (active,) = store.active_evaluation_leases()
        assert (active.lease_id, active.expires_block) == (lease_id, BLOCK + 5)


@pytest.mark.parametrize(
    "target_ids",
    [
        ("profile.zeta.collective", "profile.alpha.block"),
        ("profile.alpha.block", "profile.zeta.collective"),
    ],
)
def test_target_identity_never_reorders_qualification_fifo(
    tmp_path: Path, target_ids: tuple[str, str]
) -> None:
    database = _new_database(tmp_path)
    rows = _published_rows(database, target_ids)
    assert tuple(row.target_id for row in rows) == target_ids
    config_path, _ = _config(tmp_path, database, qualification_max_members=1)
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


def test_old_reproduction_marker_without_a_retained_pass_cannot_launch_work(tmp_path: Path) -> None:
    database = _new_database(tmp_path)
    rows = _published_rows(database, ("target.first", "target.second", "target.reproduction"))
    with FinalizedIntakeStore(database, POLICY, scope=SCOPE) as store:
        store._db.execute(
            "UPDATE reservations SET status='reproduction_pending',screen_lane='reproduction' "
            "WHERE reservation_id=?", (rows[2].reservation_id,),
        )
    config_path, _ = _config(tmp_path, database)
    with pytest.raises(IntakeError, match="pending primary qualification is inconsistent"):
        operator.preview(operator.load_config(config_path))


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
    first, second = _published_rows(
        database,
        ("profile.first", "profile.second"),
        policy=policy,
    )
    _advance(database, BLOCK + 5, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(first.reservation_id).status,
            store.get(second.reservation_id).status,
        } == {"expired"}

    authority = _seal(
        tmp_path / "downtime-requeue.json",
        {
            "reason": "validator_worker_unavailable",
            "reservation_ids": [first.reservation_id, second.reservation_id],
            "retained_result_reservation_ids": [_h("retained-result")],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    config_path, _ = _config(
        tmp_path,
        database,
        policy=policy,
        lease_blocks=3,
    )
    config = operator.load_config(config_path)
    result = operator.requeue_expired(config, authority)
    assert [item["status"] for item in result["requeued"]] == [
        "published",
        "published",
    ]
    assert operator.preview(config)["reservation_ids"] == [
        first.reservation_id,
        second.reservation_id,
    ]

    _advance(database, BLOCK + 9, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.expire_stale(current_block=BLOCK + 9) == ()
    _advance(database, BLOCK + 10, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(first.reservation_id).status,
            store.get(second.reservation_id).status,
        } == {"expired"}

    # One refresh is admitted after the cohort re-expires under the automatic
    # SLA (operator fault / mismatched window); a third attempt fails closed.
    refresh_authority = _seal(
        tmp_path / "downtime-requeue-refresh.json",
        {
            "reason": "validator_worker_unavailable",
            "reservation_ids": [first.reservation_id, second.reservation_id],
            "retained_result_reservation_ids": [_h("retained-result-refresh")],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    refreshed = operator.requeue_expired(config, refresh_authority)
    assert [item["status"] for item in refreshed["requeued"]] == [
        "published",
        "published",
    ]
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.get(first.reservation_id).reason == (
            "validator_downtime_requeued_refresh"
        )
    _advance(database, BLOCK + 15, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(first.reservation_id).status,
            store.get(second.reservation_id).status,
        } == {"expired"}
    with pytest.raises(IntakeError, match="budget is already consumed"):
        operator.requeue_expired(config, refresh_authority)

    # Owner-escalated repeat: only an authority explicitly carrying
    # allow_repeat_refresh reopens a consumed budget; the flag must be a
    # real boolean.
    malformed_repeat = _seal(
        tmp_path / "downtime-requeue-repeat-malformed.json",
        {
            "allow_repeat_refresh": "yes",
            "reason": "validator_worker_unavailable",
            "reservation_ids": [first.reservation_id, second.reservation_id],
            "retained_result_reservation_ids": [_h("retained-result-repeat")],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    with pytest.raises(operator.FifoLeaseError, match="must be boolean"):
        operator.requeue_expired(config, malformed_repeat)
    repeat_authority = _seal(
        tmp_path / "downtime-requeue-repeat.json",
        {
            "allow_repeat_refresh": True,
            "reason": "validator_worker_unavailable",
            "reservation_ids": [first.reservation_id, second.reservation_id],
            "retained_result_reservation_ids": [_h("retained-result-repeat")],
            "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
        },
    )
    repeated = operator.requeue_expired(config, repeat_authority)
    assert [item["status"] for item in repeated["requeued"]] == [
        "published",
        "published",
    ]
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.get(first.reservation_id).reason == (
            "validator_downtime_requeued_refresh"
        )


def test_downtime_requeue_refuses_a_row_that_expired_before_publication(
    tmp_path: Path,
) -> None:
    """Every requeued row re-enters the one published queue, so a cohort that
    names a row without a publication is refused whole."""

    policy = IntakePolicy(max_cohort=4, expiry_blocks=5)
    database = _new_database(tmp_path, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        published, unpublished = store.reserve_finalized(
            (_arrival(0), _arrival(1)),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )
        published = _publish(store, published, "profile.published")
    _advance(database, BLOCK + 5, policy=policy)
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert {
            store.get(published.reservation_id).status,
            store.get(unpublished.reservation_id).status,
        } == {"expired"}

    def authority(name: str, *rows) -> Path:
        return _seal(
            tmp_path / name,
            {
                "reason": "validator_worker_unavailable",
                "reservation_ids": [row.reservation_id for row in rows],
                "retained_result_reservation_ids": [],
                "schema": operator.REQUEUE_AUTHORITY_SCHEMA,
            },
        )

    config_path, _ = _config(tmp_path, database, policy=policy, lease_blocks=3)
    config = operator.load_config(config_path)
    with pytest.raises(IntakeError, match="cannot restore this pipeline phase"):
        operator.requeue_expired(
            config, authority("mixed.json", published, unpublished)
        )
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.get(published.reservation_id).status == "expired"

    result = operator.requeue_expired(config, authority("alone.json", published))
    assert [item["status"] for item in result["requeued"]] == ["published"]
    assert operator.preview(config)["reservation_ids"] == [published.reservation_id]
    with FinalizedIntakeStore(database, policy, scope=SCOPE) as store:
        assert store.get(unpublished.reservation_id).status == "expired"


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


def test_qualification_cohort_uses_store_order_and_sealed_maximum(
    tmp_path: Path,
) -> None:
    database = _new_database(tmp_path)
    rows = _published_rows(
        database,
        ("profile.zeta.collective", "profile.alpha.block", "profile.middle"),
    )
    config_path, _ = _config(
        tmp_path,
        database,
        qualification_max_members=2,
    )
    config = operator.load_config(config_path)
    expected = [row.reservation_id for row in rows[:2]]

    assert operator.preview(config)["reservation_ids"] == expected
    lease = operator.claim(config)["lease"]
    assert lease["stage"] == "qualification"
    assert [member["reservation_id"] for member in lease["members"]] == expected


