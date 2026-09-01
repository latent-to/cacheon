from __future__ import annotations

import argparse
import sys
import types

import pytest

import cacheon.cli as cli
from cacheon import chain
from cacheon.chain.intake import (
    FinalizedIntakeStore,
    IntakeError,
    IntakeScope,
    SQLiteWeightPublicationJournal,
)
from cacheon.chain.weights import (
    WeightProjection,
    WeightPublicationError,
    WeightPublicationRecord,
)
from cacheon.economics import EmissionsPolicyManifest
from cacheon.stack_identity import canonical_digest, sha256_hex


SCOPE = IntakeScope("0x" + "0" * 64, 307)
POLICY = EmissionsPolicyManifest(100, 20, 100_000, 1_800)


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _view(block: int) -> chain.MetagraphView:
    return chain.MetagraphView(
        307,
        block,
        "0x" + f"{block:064x}",
        [0, 1],
        ["validator", "miner"],
        [True, True],
        [10, 0],
    )


def _metagraph_digest(view) -> str:
    return canonical_digest(
        "cacheon.economics.metagraph-membership",
        {
            "block": view.block,
            "block_hash": view.block_hash,
            "chain_scope_digest": SCOPE.digest,
            "members": [
                {"hotkey": hotkey, "uid": uid}
                for uid, hotkey in zip(view.uids, view.hotkeys, strict=True)
            ],
        },
    )


def _projection() -> WeightProjection:
    return WeightProjection(
        SCOPE.digest, 307, "validator", POLICY.digest,
        _h("settlement"), _h("evaluation"), _metagraph_digest(_view(10)),
        (_h("arena-state"),), 1, 10, 1,
        (_h("off-pod-evidence"),), (("miner", 1_000_000),),
    )


def _seed(
    path, status: str = "pending", reason: str = "sdk_result_unconfirmed"
) -> tuple[WeightProjection, WeightPublicationRecord]:
    projection = _projection()
    record = WeightPublicationRecord(
        projection.digest, status, submit_block=10, retry_after_block=30,
        reason=reason,
    )
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        SQLiteWeightPublicationJournal(store, projection).compare_and_swap(
            None, record
        )
    return projection, record


def _args(path, **updates) -> argparse.Namespace:
    values = {
        "intake_db": str(path),
        "netuid": 307,
        "network": "mock",
        "wallet": "must-not-load",
        "hotkey": "must-not-load",
        "wallet_path": "",
        "validator_hotkey": "validator",
        "half_life_blocks": 100,
        "discovery_lifetime_blocks": 20,
        "discovery_pool_ppm": 100_000,
        "time_multiplier_scale_blocks": 1_800,
        "refresh_blocks": 20,
        "release_hold": "",
        "burn_hotkey": "",
        "burn_to_subnet_owner": False,
        "dry_run": False,
        "reconcile_only": True,
        "watch": False,
        "interval": 60.0,
        "weight_offer_path": "",
        "object_store_provider": "",
        "object_store_bucket": "",
        "object_store_prefix": "",
        "object_store_key": "",
        "object_store_endpoint": "",
        "object_store_region": "",
        "object_store_access_key": "",
        "object_store_secret_key": "",
        "object_store_addressing": "",
        "object_store_root": "",
    }
    values.update(updates)
    return argparse.Namespace(**values)


class _Subtensor:
    def get_block_hash(self, block: int) -> str:
        assert block == 0
        return SCOPE.genesis_hash

    def get_hyperparameter(self, param_name, netuid=None, block=None):
        assert param_name == "WeightsVersionKey"
        return 29


def _install_chain_readback(monkeypatch, fresh_block: int = 11) -> None:
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _subtensor, _netuid, *, block=None: _view(
            fresh_block if block is None else block
        ),
    )
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_args, **_kwargs: chain.ValidatorWeightSnapshot(
            {"miner": 1.0}, 10
        ),
    )


def _install_wallet(monkeypatch, *, expect_name: str | None = None) -> None:
    class Hotkey:
        ss58_address = "validator"

    class Wallet:
        def __init__(self, name, hotkey):
            assert expect_name is None or name == expect_name
            self.hotkey = Hotkey()

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Wallet=Wallet)
    )


def _owner_view(*, block_hash: str = "0x" + f"{11:064x}", last_update=(5, 0)):
    return chain.MetagraphView(
        netuid=307,
        block=11,
        block_hash=block_hash,
        uids=[0, 7],
        hotkeys=["validator", "owner-hk"],
        validator_permit=[True, False],
        last_update=list(last_update),
    )


def _owner_target(view) -> chain.SubnetOwnerBurnTarget:
    return chain.SubnetOwnerBurnTarget(
        uid=7,
        hotkey="owner-hk",
        owner_coldkey="owner-ck",
        owner_hotkey="owner-hk",
        candidate_uids=(7,),
        block=11,
        block_hash=view.block_hash,
        metagraph=view,
    )


def _fetch_factory(view):
    def fetch(_subtensor, _netuid, *, block=None):
        if block is None or block == view.block:
            return view
        return chain.MetagraphView(
            view.netuid,
            block,
            "0x" + f"{block:064x}",
            list(view.uids),
            list(view.hotkeys),
            list(view.validator_permit),
            list(view.last_update),
        )

    return fetch


def _burn_args(path, **updates) -> argparse.Namespace:
    return _args(
        path,
        reconcile_only=False,
        validator_hotkey="",
        burn_to_subnet_owner=True,
        **updates,
    )


def _empty_store(path) -> None:
    # An empty all-uncrowned store is the entire precondition.
    with FinalizedIntakeStore(path, scope=SCOPE):
        pass


def test_reconcile_only_cli_uses_retained_head_without_reopening_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, pending = _seed(path)
    _install_chain_readback(monkeypatch)

    def forbidden_fresh_projection(*_args, **_kwargs):
        raise AssertionError("reconcile-only reopened current settlement evidence")

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "build_weight_projection",
        forbidden_fresh_projection,
    )

    assert cli.cmd_set_weights(_args(path)) == 0

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        confirmed = journal.load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"
        assert confirmed.prior_record_digest == pending.digest


def test_reconcile_only_cli_reports_historical_confirmation_that_needs_refresh(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "private" / "intake.sqlite3"
    _projection_row, pending = _seed(path)
    _install_chain_readback(monkeypatch, fresh_block=31)

    assert cli.cmd_set_weights(_args(path)) == 3
    output = capsys.readouterr().out
    assert "status=confirmed" in output
    assert "refresh_due=True" in output

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        confirmed = SQLiteWeightPublicationJournal.reopen_from_head(store).load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"
        assert confirmed.prior_record_digest == pending.digest


def test_release_hold_reopens_retained_head_without_off_pod_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, held = _seed(
        path, "held", "publication_readback_deadline_expired"
    )
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())

    def forbidden_fresh_projection(*_args, **_kwargs):
        raise AssertionError("release-hold reopened current settlement evidence")

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "build_weight_projection",
        forbidden_fresh_projection,
    )

    assert cli.cmd_set_weights(
        _args(
            path,
            reconcile_only=False,
            release_hold="late reveal reviewed",
        )
    ) == 0

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        released = journal.load()
        assert released is not None
        assert released.status == "released"
        assert released.prior_record_digest == held.digest


def test_release_hold_authority_mismatch_does_not_mutate_head(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, held = _seed(
        path, "held", "publication_readback_deadline_expired"
    )
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())

    with pytest.raises(WeightPublicationError, match="public validator hotkey"):
        cli.cmd_set_weights(
            _args(
                path,
                reconcile_only=False,
                release_hold="must not land",
                validator_hotkey="other",
            )
        )

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        assert journal.load() == held


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"validator_hotkey": "other"}, "public validator hotkey"),
        ({"half_life_blocks": 101}, "emissions policy"),
    ],
)
def test_reconcile_only_cli_rejects_public_authority_mismatch_before_readback(
    tmp_path, monkeypatch, updates, message
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, pending = _seed(path)
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())

    def forbidden_readback(*_args, **_kwargs):
        raise AssertionError("mismatched retained authority reached chain readback")

    monkeypatch.setattr(chain, "fetch_metagraph", forbidden_readback)

    with pytest.raises(WeightPublicationError, match=message):
        cli.cmd_set_weights(_args(path, **updates))

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        assert journal.load() == pending


def test_burn_hotkey_cli_dry_run_projects_the_full_pool_pre_crown(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "private" / "intake.sqlite3"
    _empty_store(path)
    _install_chain_readback(monkeypatch)
    fresh_blocks = iter((10, 11))
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _subtensor, _netuid, *, block=None: _view(
            next(fresh_blocks) if block is None else block
        ),
    )

    _install_wallet(monkeypatch, expect_name="must-not-load")

    assert cli.cmd_set_weights(
        _args(
            path,
            reconcile_only=False,
            dry_run=True,
            validator_hotkey="",
            burn_hotkey="miner",
        )
    ) == 0
    out = capsys.readouterr().out
    assert "burn projection: full pool -> uid 1 hotkey miner" in out
    assert "status=dry_run" in out
    assert "submitted=False" in out

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        with pytest.raises(IntakeError, match="no retained head"):
            SQLiteWeightPublicationJournal.reopen_from_head(store)


def test_burn_hotkey_cli_refuses_head_only_combinations(tmp_path):
    path = tmp_path / "private" / "intake.sqlite3"
    with pytest.raises(SystemExit, match="burn-hotkey"):
        cli.cmd_set_weights(_args(path, burn_hotkey="miner"))
    with pytest.raises(SystemExit, match="burn-hotkey"):
        cli.cmd_set_weights(
            _args(
                path,
                reconcile_only=False,
                release_hold="operator request",
                burn_hotkey="miner",
            )
        )


def test_burn_hotkey_cli_publishes_real_weights_without_a_crown(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "private" / "intake.sqlite3"
    _empty_store(path)
    _install_chain_readback(monkeypatch)
    snapshots = [
        chain.ValidatorWeightSnapshot({"validator": 1.0}, 5),
        chain.ValidatorWeightSnapshot({"miner": 1.0}, 11),
    ]
    monkeypatch.setattr(
        chain, "read_validator_weight_snapshot", lambda *_args, **_kwargs: snapshots.pop(0)
    )
    submissions = []

    def _set_weights(_subtensor, wallet, netuid, weights, **kwargs):
        submissions.append((wallet, netuid, dict(weights), kwargs))
        return {"submitted": True}

    monkeypatch.setattr(chain, "set_weights", _set_weights)

    _install_wallet(monkeypatch)

    assert cli.cmd_set_weights(
        _args(
            path,
            reconcile_only=False,
            validator_hotkey="",
            burn_hotkey="miner",
        )
    ) == 0
    out = capsys.readouterr().out
    assert "status=confirmed" in out
    assert "submitted=True" in out
    assert len(submissions) == 1
    assert submissions[0][2] == {"miner": 1.0}
    assert submissions[0][3]["dry_run"] is False

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection.weights == {"miner": 1.0}
        assert journal.projection.crown_count == 0
        confirmed = journal.load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"


def test_watch_retries_transport_then_reconciles_without_external_scheduler(
    tmp_path, monkeypatch
):
    calls = []
    statuses = iter((3, 0))

    def run_once(args):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionError("temporary chain disconnect")
        try:
            return next(statuses)
        except StopIteration:
            raise KeyboardInterrupt

    sleeps = []
    monkeypatch.setattr(cli, "_cmd_set_weights_once", run_once)
    monkeypatch.setattr("time.sleep", sleeps.append)
    args = _args(
        tmp_path / "unused.sqlite3",
        reconcile_only=False,
        watch=True,
        interval=7.0,
    )

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_set_weights(args)

    assert calls == [args, args, args, args]
    assert sleeps == [14.0, 7.0, 7.0]


@pytest.mark.parametrize("field", ("dry_run", "reconcile_only", "release_hold"))
def test_watch_refuses_non_signer_modes(tmp_path, field):
    updates = {
        "reconcile_only": False,
        "watch": True,
        field: True if field != "release_hold" else "reviewed",
    }
    with pytest.raises(SystemExit, match="watch requires the signer path"):
        cli.cmd_set_weights(_args(tmp_path / "unused.sqlite3", **updates))


def test_watch_does_not_retry_nonretryable_publication_fault(tmp_path, monkeypatch):
    def fail(_args):
        raise WeightPublicationError("operator action required")

    monkeypatch.setattr(cli, "_cmd_set_weights_once", fail)
    with pytest.raises(WeightPublicationError, match="operator action"):
        cli.cmd_set_weights(
            _args(
                tmp_path / "unused.sqlite3",
                reconcile_only=False,
                watch=True,
            )
        )


def test_burn_to_subnet_owner_cli_dry_run_journals_through_reconcile(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "private" / "intake.sqlite3"
    _empty_store(path)

    view = _owner_view(block_hash="0x" + "ab" * 32, last_update=(1, 1))
    target = _owner_target(view)

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain, "resolve_subnet_owner_burn_target", lambda *_args, **_kwargs: target
    )
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _subtensor, _netuid, *, block=None: view,
    )
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_args, **_kwargs: chain.ValidatorWeightSnapshot({}, 0),
    )
    submissions = []

    def _set_weights(_subtensor, wallet, netuid, weights, **kwargs):
        submissions.append((wallet, netuid, dict(weights), kwargs.get("dry_run")))
        return {
            "submitted": False,
            "dry_run": True,
            "uids": [7],
            "weights": [1.0],
            "authority_block": 11,
        }

    monkeypatch.setattr(chain, "set_weights", _set_weights)

    _install_wallet(monkeypatch)

    assert cli.cmd_set_weights(_burn_args(path, dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "subnet-owner burn:" in out
    assert "uid 7 hotkey owner-hk" in out
    assert "status=dry_run" in out
    assert "submitted=False" in out
    assert submissions == [(None, 307, {"owner-hk": 1.0}, True)]

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        with pytest.raises(IntakeError, match="no retained head"):
            SQLiteWeightPublicationJournal.reopen_from_head(store)


def test_burn_to_subnet_owner_cli_confirms_via_journal(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "private" / "intake.sqlite3"
    _empty_store(path)

    view = _owner_view()
    target = _owner_target(view)

    _fetch = _fetch_factory(view)
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain, "resolve_subnet_owner_burn_target", lambda *_args, **_kwargs: target
    )
    monkeypatch.setattr(chain, "fetch_metagraph", _fetch)

    snapshots = [
        chain.ValidatorWeightSnapshot({"validator": 1.0}, 5),
        chain.ValidatorWeightSnapshot({"owner-hk": 1.0}, 11),
    ]
    monkeypatch.setattr(
        chain, "read_validator_weight_snapshot", lambda *_args, **_kwargs: snapshots.pop(0)
    )
    submissions = []

    def _set_weights(_subtensor, wallet, netuid, weights, **kwargs):
        submissions.append((wallet, netuid, dict(weights), kwargs.get("dry_run")))
        return {"submitted": True}

    monkeypatch.setattr(chain, "set_weights", _set_weights)

    _install_wallet(monkeypatch)

    assert cli.cmd_set_weights(_burn_args(path)) == 0
    out = capsys.readouterr().out
    assert "status=confirmed" in out
    assert "submitted=True" in out
    assert len(submissions) == 1
    assert submissions[0][2] == {"owner-hk": 1.0}
    assert submissions[0][3] is False

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection.weights == {"owner-hk": 1.0}
        assert journal.projection.crown_count == 0
        confirmed = journal.load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"


def test_burn_to_subnet_owner_refuses_foreign_in_flight_projection(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    settlement, pending = _seed(path)
    assert pending.status == "pending"
    assert settlement.weights == {"miner": 1.0}

    view = _owner_view()
    target = _owner_target(view)

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain, "resolve_subnet_owner_burn_target", lambda *_args, **_kwargs: target
    )

    _install_wallet(monkeypatch)

    with pytest.raises(WeightPublicationError, match="in-flight weight publication"):
        cli.cmd_set_weights(_burn_args(path))


@pytest.mark.parametrize(
    ("last_update", "expected_status"),
    [(5, "pending"), (11, "confirmed")],
)
def test_burn_to_subnet_owner_resumes_own_in_flight_refresh(
    tmp_path, monkeypatch, capsys, last_update, expected_status
):
    """A submitted-but-unconfirmed burn head re-resolves under a fresh
    block-bound digest one tick later; the watch must adopt its own retained
    head (confirm by readback, or keep waiting inside its retry bounds)
    instead of refusing it as a foreign in-flight publication."""

    path = tmp_path / "private" / "intake.sqlite3"
    view = _owner_view()
    _fetch = _fetch_factory(view)

    bound = _fetch(None, 307, block=10)
    retained = WeightProjection(
        SCOPE.digest,
        307,
        "validator",
        POLICY.digest,
        _h("prior-settlement"),
        _h("prior-evaluation"),
        canonical_digest(
            "cacheon.economics.metagraph-membership",
            {
                "block": bound.block,
                "block_hash": bound.block_hash,
                "chain_scope_digest": SCOPE.digest,
                "members": [
                    {"hotkey": hotkey, "uid": uid}
                    for uid, hotkey in zip(bound.uids, bound.hotkeys, strict=True)
                ],
            },
        ),
        (_h("arena-state"),),
        1,
        10,
        0,
        (),
        (("owner-hk", 1_000_000),),
    )
    pending = WeightPublicationRecord(
        retained.digest,
        "pending",
        submit_block=10,
        retry_after_block=30,
        reason="sdk_result_unconfirmed",
    )
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        SQLiteWeightPublicationJournal(store, retained).compare_and_swap(
            None, pending
        )

    target = _owner_target(view)

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain, "resolve_subnet_owner_burn_target", lambda *_a, **_k: target
    )
    monkeypatch.setattr(chain, "fetch_metagraph", _fetch)
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_a, **_k: chain.ValidatorWeightSnapshot(
            {"owner-hk": 1.0}, last_update
        ),
    )
    submissions = []

    def _set_weights(*a, **k):
        submissions.append((a, k))
        return {"submitted": True}

    monkeypatch.setattr(chain, "set_weights", _set_weights)

    _install_wallet(monkeypatch)

    assert cli.cmd_set_weights(_burn_args(path)) == 0
    out = capsys.readouterr().out
    assert f"status={expected_status}" in out
    assert f"weight projection={retained.digest}" in out
    assert not submissions
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        head = SQLiteWeightPublicationJournal.reopen_from_head(store)
        record = head.load()
        assert record is not None
        assert record.status == expected_status
        assert record.projection_digest == retained.digest


def test_burn_to_subnet_owner_settlement_refusal_is_nonretryable(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    _empty_store(path)

    view = _owner_view(block_hash="0x" + "ab" * 32, last_update=(1, 1))
    target = _owner_target(view)

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain, "resolve_subnet_owner_burn_target", lambda *_args, **_kwargs: target
    )

    def _refuse(self, **_kwargs):
        raise IntakeError(
            "subnet-owner burn weights refused: active reward claims exist; "
            "project real weights instead"
        )

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "build_subnet_owner_burn_weight_projection",
        _refuse,
    )

    _install_wallet(monkeypatch)

    with pytest.raises(
        WeightPublicationError, match="burn weights refused"
    ) as excinfo:
        cli.cmd_set_weights(_burn_args(path))
    # A settlement-state refusal must stop --watch, not spin behind it.
    assert excinfo.value.retryable is False


def test_burn_to_subnet_owner_requires_policy_flags(tmp_path):
    with pytest.raises(SystemExit, match="--half-life-blocks"):
        cli.cmd_set_weights(
            _args(
                tmp_path / "unused.sqlite3",
                burn_to_subnet_owner=True,
                reconcile_only=False,
                dry_run=True,
                half_life_blocks=None,
            )
        )


def test_burn_to_subnet_owner_refuses_conflicting_flags(tmp_path):
    path = tmp_path / "unused.sqlite3"
    with pytest.raises(SystemExit, match="burn-hotkey"):
        cli.cmd_set_weights(
            _args(path, burn_to_subnet_owner=True, burn_hotkey="miner")
        )
    with pytest.raises(SystemExit, match="reconcile-only"):
        cli.cmd_set_weights(
            _args(path, burn_to_subnet_owner=True, reconcile_only=True)
        )
    with pytest.raises(SystemExit, match="release-hold"):
        cli.cmd_set_weights(
            _args(
                path,
                burn_to_subnet_owner=True,
                reconcile_only=False,
                release_hold="nope",
            )
        )
    with pytest.raises(SystemExit, match="weight-offer-path"):
        cli.cmd_set_weights(
            _args(
                path,
                burn_to_subnet_owner=True,
                reconcile_only=False,
                weight_offer_path="/tmp/offer.json",
            )
        )
    with pytest.raises(SystemExit, match="object-store"):
        cli.cmd_set_weights(
            _args(
                path,
                burn_to_subnet_owner=True,
                reconcile_only=False,
                object_store_provider="hippius",
            )
        )


def test_normal_set_weights_still_requires_policy_flags(tmp_path):
    with pytest.raises(SystemExit, match="--half-life-blocks"):
        cli.cmd_set_weights(
            _args(
                tmp_path / "unused.sqlite3",
                half_life_blocks=None,
                reconcile_only=False,
                dry_run=True,
            )
        )


def test_emissions_policy_from_args_canonicalizes_exclusions() -> None:
    policy = cli._emissions_policy_from_args(
        argparse.Namespace(
            half_life_blocks=100,
            discovery_lifetime_blocks=20,
            discovery_pool_ppm=100_000,
            time_multiplier_scale_blocks=1_800,
            exclude_hotkey=["bob", "alice"],
            exclude_claim_digest=["b" * 64, "a" * 64],
        )
    )
    assert policy.excluded_hotkeys == ("alice", "bob")
    assert policy.excluded_claim_digests == ("a" * 64, "b" * 64)
    assert policy.policy_version == "cacheon.emissions.v1.3"
