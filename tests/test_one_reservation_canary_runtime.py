from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain import one_reservation_canary_runtime as runtime_module
from cacheon.chain.evaluation_lease_store import EvaluationClaimConflict
from cacheon.chain.one_reservation_canary import (
    CanaryCheckpoint,
    CanaryReservationPhase,
    CanaryStage,
    CanaryStageDisposition,
    CanaryStoreObservation,
    CanaryTerminalOutcome,
    CanaryTransition,
    OneReservationCanaryBoundaries,
    OneReservationCanaryController,
)
from cacheon.chain.one_reservation_canary_runtime import (
    CANARY_CHECKPOINT_JOURNAL_SCHEMA,
    CanaryCheckpointJournal,
    OneReservationCanaryRuntime,
    OneReservationCanaryRuntimeError,
)
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcher,
)
from cacheon.chain.remote_evaluation_dispatcher import RemoteEvaluationDispatcher
from cacheon.stack_identity import canonical_json_bytes


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _load(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _qualification_stub(coordinator, identity, credential, incumbent, tree, callback):
    dispatcher = object.__new__(RecoverableQualificationDispatcher)
    dispatcher.coordinator = coordinator
    dispatcher.transport = SimpleNamespace(identity=identity)
    dispatcher.credential = credential
    dispatcher.transport_identity = identity
    dispatcher.qualification_evidence_root = Path("/unused-in-runtime-test")
    dispatcher.qualification_incumbent_stack = incumbent
    dispatcher.qualification_incumbent_tree_digest = tree
    dispatcher.dispatch_guarded_once = callback
    return dispatcher


def _screen_stub(coordinator, identity, credential):
    dispatcher = object.__new__(RemoteEvaluationDispatcher)
    dispatcher.coordinator = coordinator
    dispatcher.transport = SimpleNamespace(identity=identity)
    dispatcher.credential = credential
    dispatcher.transport_identity = identity
    return dispatcher


def _screen_setup(root: Path, label: str, *, count: int = 1, cohort: int = 1):
    remote = _load(
        f"cacheon_canary_runtime_remote_{label}_{count}_{cohort}",
        "test_remote_evaluation_dispatcher.py",
    )
    base = remote._manifest()
    runtime_identity = dataclasses.replace(
        base.runtime,
        arena_id=f"canary-{label}",
        topology_class=f"tp-{label}",
        topology_digest=remote._h(f"topology:{label}"),
    )
    manifest = dataclasses.replace(base, runtime=runtime_identity)
    remote._manifest = lambda: manifest
    rows = remote._published_rows(root, count)
    service = ArenaService(manifest, remote._Provider())
    cursor = remote._Cursor((remote.BLOCK, remote._block_hash(remote.BLOCK)))
    coordinator = remote._coordinator(
        root,
        service,
        cursor,
        qualification_max_members=cohort,
        store_factory=RecoverableFinalizedIntakeStore,
    )
    credential = remote.RemoteWorkerCredential("canary-runtime-v1", b"c" * 32)
    screen_transport = remote._Transport(coordinator, credential)
    screen_dispatcher = remote._dispatcher(
        coordinator, screen_transport, credential
    )
    incumbent = remote._incumbent(service, marker=label)
    calls: list[dict[str, object]] = []

    def qualification_callback(**guards):
        calls.append(guards)
        return None

    qualification_dispatcher = _qualification_stub(
        coordinator,
        screen_transport.identity,
        credential,
        incumbent,
        remote._h("incumbent-tree"),
        qualification_callback,
    )
    canary = OneReservationCanaryRuntime(
        screen_dispatcher=screen_dispatcher,
        qualification_dispatcher=qualification_dispatcher,
        expected_reservation_digest=rows[0].reservation_id,
    )
    return SimpleNamespace(
        runtime=canary,
        remote=remote,
        coordinator=coordinator,
        credential=credential,
        rows=rows,
        screen_transport=screen_transport,
        qualification_calls=calls,
    )


def _checkpoint(
    observation: CanaryStoreObservation,
    *,
    screen: bool = False,
    qualification: bool = False,
    request_id: str | None = None,
) -> CanaryCheckpoint:
    active = observation.active_lease
    return CanaryCheckpoint(
        expected_reservation_digest=observation.reservation_digest,
        target_profile_digest=observation.target_profile_digest,
        started_monotonic=10.0,
        deadline_monotonic=100.0,
        max_ticks=8,
        max_stage_receipts=32,
        ticks_used=1,
        screen_claim_started=screen,
        qualification_started=qualification,
        qualification_lease_id=(
            None if active is None else active.lease_id
        ),
        qualification_request_id=(
            request_id
            if qualification
            else None
        ),
    )


def test_two_target_profiles_use_the_same_typed_screen_path(tmp_path: Path) -> None:
    profiles: list[str] = []
    for label in ("collective-alpha", "attention-beta"):
        setup = _screen_setup(tmp_path / label, label)
        before = setup.runtime.observe_store()
        receipt = setup.runtime.screen_once(
            before, _checkpoint(before, screen=True)
        )
        after = setup.runtime.observe_store()

        assert receipt.request_id == setup.screen_transport.requests[0].digest
        assert receipt.lease_id is not None
        assert receipt.reservation_digests == (
            setup.rows[0].reservation_id,
        )
        assert before.phase is CanaryReservationPhase.PUBLISHED
        assert after.phase is CanaryReservationPhase.PROMOTED
        assert after.next_qualification_reservation_digests == (
            setup.rows[0].reservation_id,
        )
        profiles.append(before.target_profile_digest)
    assert len(set(profiles)) == 2


def test_screen_preview_race_cannot_claim_or_transport_the_second_row(
    tmp_path: Path,
) -> None:
    setup = _screen_setup(tmp_path, "preview-race", count=2)
    before = setup.runtime.observe_store()
    first, second = setup.rows
    assert before.fifo_head_reservation_digest == first.reservation_id
    with RecoverableFinalizedIntakeStore(
        setup.remote._db_path(tmp_path),
        setup.remote.POLICY,
        scope=setup.remote.SCOPE,
    ) as store:
        store._db.execute(
            "UPDATE reservations SET status='held' WHERE reservation_id=?",
            (first.reservation_id,),
        )

    with pytest.raises(EvaluationClaimConflict) as raised:
        setup.runtime.screen_once(before, _checkpoint(before, screen=True))
    assert raised.value.observed_members[0].reservation_id == second.reservation_id
    assert setup.screen_transport.requests == []
    with RecoverableFinalizedIntakeStore(
        setup.remote._db_path(tmp_path),
        setup.remote.POLICY,
        scope=setup.remote.SCOPE,
    ) as store:
        assert store.active_evaluation_leases() == ()
        assert store.get(second.reservation_id).status == "published"


def test_qualification_cohort_widening_fails_before_dispatch(tmp_path: Path) -> None:
    setup = _screen_setup(tmp_path, "wide-cohort", count=2, cohort=2)
    assert setup.coordinator.run_screen_once() is not None
    assert setup.coordinator.run_screen_once() is not None
    observation = setup.runtime.observe_store()
    assert observation.next_qualification_reservation_digests == tuple(
        row.reservation_id for row in setup.rows
    )
    with pytest.raises(
        OneReservationCanaryRuntimeError, match="preview or retained guard"
    ):
        setup.runtime.qualification_once(
            observation, _checkpoint(observation, qualification=True)
        )
    assert setup.qualification_calls == []


def _qualification_setup(root: Path, *, profile: str, fail_resume: bool = False):
    recoverable = _load(
        f"cacheon_canary_runtime_recoverable_{profile}",
        "test_recoverable_qualification_dispatcher.py",
    )
    fixtures = recoverable._fixtures()
    authority = fixtures._authority(
        root,
        profile=profile,
        recoverable=True,
    )
    transport = recoverable._Transport(
        authority, fixtures, fail_resume=fail_resume
    )
    qualification_dispatcher = recoverable._dispatcher(authority, transport)
    screen_dispatcher = _screen_stub(
        authority.coordinator, transport.identity, authority.credential
    )
    runtime = OneReservationCanaryRuntime(
        screen_dispatcher=screen_dispatcher,
        qualification_dispatcher=qualification_dispatcher,
        expected_reservation_digest=authority.claim.lease.reservation_ids[0],
    )
    return SimpleNamespace(
        runtime=runtime,
        recoverable=recoverable,
        fixtures=fixtures,
        authority=authority,
        transport=transport,
        root=root,
        remote=authority.fixtures,
    )


def test_active_recovery_wrong_guard_stops_before_transport_then_resumes_request(
    tmp_path: Path,
) -> None:
    setup = _qualification_setup(
        tmp_path, profile="resume-exact", fail_resume=True
    )
    first = setup.runtime.observe_store()
    assert first.active_lease is not None
    assert first.active_lease.request_id is None
    with pytest.raises(
        setup.recoverable.RecoverableQualificationDispatcherError,
        match="same-request qualification result is not ready",
    ):
        setup.runtime.qualification_once(
            first, _checkpoint(first, qualification=True)
        )
    assert setup.transport.plan is not None

    retained = setup.runtime.observe_store()
    assert retained.active_lease is not None
    request_id = setup.transport.plan.request_id
    assert retained.active_lease.request_id == request_id
    before_counts = (
        setup.transport.plans,
        setup.transport.materializations,
        setup.transport.publications,
        setup.transport.resumes,
    )
    wrong_request = _h("wrong-active-request")
    wrong_lease = replace(retained.active_lease, request_id=wrong_request)
    wrong_observation = replace(retained, active_lease=wrong_lease)
    with pytest.raises(EvaluationClaimConflict):
        setup.runtime.qualification_once(
            wrong_observation,
            _checkpoint(
                wrong_observation,
                qualification=True,
                request_id=wrong_request,
            ),
        )
    assert (
        setup.transport.plans,
        setup.transport.materializations,
        setup.transport.publications,
        setup.transport.resumes,
    ) == before_counts

    setup.fixtures._write_completed_result(setup.authority, setup.transport.plan)
    setup.transport.fail_resume = False
    receipt = setup.runtime.qualification_once(
        retained,
        _checkpoint(retained, qualification=True, request_id=request_id),
    )
    assert receipt.request_id == request_id
    assert receipt.lease_id == retained.active_lease.lease_id
    assert receipt.expensive_stage_receipt_digests == tuple(
        sorted(set(receipt.expensive_stage_receipt_digests))
    )
    assert len(receipt.expensive_stage_receipt_digests) >= 2
    assert (setup.transport.plans, setup.transport.materializations) == (1, 1)


def test_pre_request_hold_preserves_before_lease_and_none_request(
    tmp_path: Path,
) -> None:
    setup = _qualification_setup(tmp_path, profile="hold-before-request")
    observation = setup.runtime.observe_store()
    assert observation.active_lease is not None
    assert observation.active_lease.request_id is None
    lease = setup.authority.claim.lease
    setup.recoverable._advance_finalized(setup.authority, lease.expires_block)

    receipt = setup.runtime.qualification_once(
        observation, _checkpoint(observation, qualification=True)
    )
    assert receipt.request_id is None
    assert receipt.lease_id == observation.active_lease.lease_id
    assert receipt.reason == "lease_expired_before_request_plan"
    assert (
        setup.transport.plans,
        setup.transport.materializations,
        setup.transport.publications,
        setup.transport.resumes,
    ) == (0, 0, 0, 0)


def test_requeue_uses_before_lease_and_real_new_request(tmp_path: Path) -> None:
    execution = _load(
        "cacheon_canary_runtime_requeue",
        "test_execution_disposition.py",
    )
    fixtures = execution._fixtures()
    authority = fixtures._authority(
        tmp_path, profile="runtime-requeue", recoverable=True
    )
    transport = execution._Transport(
        authority,
        fixtures,
        on_publish=lambda plan: execution._write_pod_refusal_result(
            authority, plan, "adapter_start_failed"
        ),
    )
    qualification_dispatcher = execution._dispatcher(authority, transport)
    runtime = OneReservationCanaryRuntime(
        screen_dispatcher=_screen_stub(
            authority.coordinator, transport.identity, authority.credential
        ),
        qualification_dispatcher=qualification_dispatcher,
        expected_reservation_digest=authority.claim.lease.reservation_ids[0],
    )
    observation = runtime.observe_store()
    assert observation.active_lease is not None
    assert observation.active_lease.request_id is None

    receipt = runtime.qualification_once(
        observation, _checkpoint(observation, qualification=True)
    )
    assert transport.plan is not None
    assert receipt.disposition is CanaryStageDisposition.REQUEUE
    assert receipt.lease_id == observation.active_lease.lease_id
    assert receipt.request_id == transport.plan.request_id
    assert receipt.reason == "adapter_start_failed"
    assert runtime.observe_store().active_lease is None


def test_observation_rejects_missing_fingerprint_stale_service_and_live_drift(
    tmp_path: Path,
) -> None:
    missing = _screen_setup(tmp_path / "missing", "missing-fingerprint")
    with RecoverableFinalizedIntakeStore(
        missing.remote._db_path(tmp_path / "missing"),
        missing.remote.POLICY,
        scope=missing.remote.SCOPE,
    ) as store:
        store._db.execute(
            "UPDATE reservations SET delta_fingerprint_json='' WHERE reservation_id=?",
            (missing.rows[0].reservation_id,),
        )
    with pytest.raises(OneReservationCanaryRuntimeError, match="fingerprint"):
        missing.runtime.observe_store()

    stale = _screen_setup(tmp_path / "stale", "stale-service")
    with RecoverableFinalizedIntakeStore(
        stale.remote._db_path(tmp_path / "stale"),
        stale.remote.POLICY,
        scope=stale.remote.SCOPE,
    ) as store:
        store._db.execute(
            "UPDATE reservations SET arena_service_digest=? WHERE reservation_id=?",
            (_h("foreign-service"), stale.rows[0].reservation_id),
        )
    with pytest.raises(OneReservationCanaryRuntimeError, match="service identity"):
        stale.runtime.observe_store()

    drift = _screen_setup(tmp_path / "drift", "live-drift")
    drift.runtime.qualification_dispatcher.transport.identity = replace(
        drift.runtime.transport_identity,
        endpoint_identity_digest=_h("drifted-endpoint"),
    )
    with pytest.raises(OneReservationCanaryRuntimeError, match="authority drifted"):
        drift.runtime.observe_store()


def test_terminal_row_is_observed_without_stage_mutation(tmp_path: Path) -> None:
    setup = _screen_setup(tmp_path, "terminal-row")
    expected = setup.rows[0].reservation_id
    with RecoverableFinalizedIntakeStore(
        setup.remote._db_path(tmp_path), setup.remote.POLICY, scope=setup.remote.SCOPE
    ) as store:
        store._db.execute(
            "UPDATE reservations SET status='expired' WHERE reservation_id=?",
            (expected,),
        )
    persisted: list[CanaryCheckpoint] = []
    controller = OneReservationCanaryController(
        boundaries=setup.runtime.boundaries(persisted.append),
        expected_fifo_head_reservation_digest=expected,
        deadline_monotonic=100.0,
        max_ticks=4,
        max_stage_receipts=8,
        monotonic=lambda: 10.0,
    )
    receipt = controller.run()
    assert receipt.outcome is CanaryTerminalOutcome.HOLD
    assert receipt.final_phase is CanaryReservationPhase.EXPIRED
    assert setup.screen_transport.requests == []
    assert setup.qualification_calls == []
    with RecoverableFinalizedIntakeStore(
        setup.remote._db_path(tmp_path), setup.remote.POLICY, scope=setup.remote.SCOPE
    ) as store:
        assert store.get(expected).status == "expired"


def _journal_checkpoint(*, terminal: bool = False) -> CanaryCheckpoint:
    before = _h("journal-before")
    after = _h("journal-after")
    transition = CanaryTransition(
        sequence=1,
        before_phase=CanaryReservationPhase.PUBLISHED,
        after_phase=CanaryReservationPhase.PROMOTED,
        before_observation_digest=before,
        after_observation_digest=after,
        stage=CanaryStage.SCREEN,
        stage_receipt_digest=_h("journal-stage"),
    )
    return CanaryCheckpoint(
        expected_reservation_digest=_h("journal-reservation"),
        target_profile_digest=_h("journal-profile"),
        started_monotonic=10.0,
        deadline_monotonic=100.0,
        max_ticks=8,
        max_stage_receipts=16,
        ticks_used=1,
        screen_claim_started=True,
        screen_lease_id=_h("journal-lease"),
        screen_request_id=_h("journal-request"),
        stage_receipt_digests=(_h("journal-stage"),),
        transitions=(transition,),
        terminal_outcome=(CanaryTerminalOutcome.COMPLETED if terminal else None),
        terminal_reason=("qualification_complete_in_store" if terminal else None),
    )


def test_journal_roundtrip_restart_and_terminal_idempotence(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    terminal = _journal_checkpoint(terminal=True)
    with CanaryCheckpointJournal(path) as journal:
        assert journal.load() is None
        journal.persist(terminal)
        assert journal.load() == terminal
    retained_bytes = path.read_bytes()

    with CanaryCheckpointJournal(path) as restarted:
        loaded = restarted.load()
        assert loaded == terminal
        observation = CanaryStoreObservation(
            reservation_digest=terminal.expected_reservation_digest,
            target_profile_digest=terminal.target_profile_digest,
            phase=CanaryReservationPhase.REPRODUCTION_PENDING,
            fifo_head_reservation_digest=None,
        )
        # Construct directly so a retained terminal checkpoint has no stage path.
        controller = OneReservationCanaryController(
            boundaries=OneReservationCanaryBoundaries(
                observe_store=lambda: observation,
                screen_once=lambda *_args: pytest.fail("terminal screen mutation"),
                qualification_once=lambda *_args: pytest.fail(
                    "terminal qualification mutation"
                ),
                persist_checkpoint=restarted.persist,
            ),
            expected_fifo_head_reservation_digest=terminal.expected_reservation_digest,
            deadline_monotonic=terminal.deadline_monotonic,
            max_ticks=terminal.max_ticks,
            max_stage_receipts=terminal.max_stage_receipts,
            monotonic=lambda: 20.0,
            retained_checkpoint=loaded,
        )
        assert controller.run().outcome is CanaryTerminalOutcome.COMPLETED
    assert path.read_bytes() == retained_bytes


def test_journal_rejects_corrupt_noncanonical_unknown_and_wrong_digest(
    tmp_path: Path,
) -> None:
    checkpoint = _journal_checkpoint()
    valid_path = tmp_path / "valid.json"
    with CanaryCheckpointJournal(valid_path) as journal:
        journal.persist(checkpoint)
    valid = json.loads(valid_path.read_text())
    cases = {
        "corrupt": b"{",
        "noncanonical": json.dumps(valid, indent=2).encode(),
        "unknown": canonical_json_bytes({**valid, "unknown": True}),
        "wrong-digest": canonical_json_bytes(
            {**valid, "checkpoint_digest": _h("wrong-checkpoint")}
        ),
    }
    for label, raw in cases.items():
        path = tmp_path / f"{label}.json"
        path.write_bytes(raw)
        path.chmod(0o600)
        with CanaryCheckpointJournal(path) as journal:
            with pytest.raises(OneReservationCanaryRuntimeError):
                journal.load()


def test_journal_rejects_symlink_noncanonical_path_and_second_owner(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    with pytest.raises(OneReservationCanaryRuntimeError, match="owner-controlled"):
        CanaryCheckpointJournal(symlink)

    directory_path = tmp_path / "directory.json"
    directory_path.mkdir()
    with pytest.raises(OneReservationCanaryRuntimeError, match="owner-controlled"):
        CanaryCheckpointJournal(directory_path)

    lock_symlink_path = tmp_path / "lock-symlink.json"
    lock_symlink_path.with_name(lock_symlink_path.name + ".lock").symlink_to(target)
    with pytest.raises(OneReservationCanaryRuntimeError, match="lock cannot open"):
        CanaryCheckpointJournal(lock_symlink_path)

    subdirectory = tmp_path / "sub"
    subdirectory.mkdir()
    with pytest.raises(OneReservationCanaryRuntimeError, match="canonical"):
        CanaryCheckpointJournal(subdirectory / ".." / "noncanonical.json")

    path = tmp_path / "owned.json"
    first = CanaryCheckpointJournal(path)
    try:
        with pytest.raises(OneReservationCanaryRuntimeError, match="active owner"):
            CanaryCheckpointJournal(path)
    finally:
        first.close()
    with CanaryCheckpointJournal(path) as reopened:
        assert reopened.load() is None


def test_journal_partial_failures_stale_state_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _journal_checkpoint()
    path = tmp_path / "mode-and-partial.json"
    lock_path = path.with_name(path.name + ".lock")
    lock_path.write_bytes(b"")
    lock_path.chmod(0o644)
    with CanaryCheckpointJournal(path) as journal:
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        with monkeypatch.context() as context:
            context.setattr(
                runtime_module.os,
                "replace",
                lambda *_args: (_ for _ in ()).throw(OSError("injected")),
            )
            with pytest.raises(
                OneReservationCanaryRuntimeError, match="persist failed"
            ):
                journal.persist(checkpoint)
        assert not tuple(tmp_path.glob(f".{path.name}.*"))
        retained_partial = tmp_path / f".{path.name}.retained"
        retained_partial.write_bytes(b"partial")
        with pytest.raises(OneReservationCanaryRuntimeError, match="stale partial"):
            journal.persist(checkpoint)
        assert retained_partial.read_bytes() == b"partial"
        retained_partial.unlink()
        journal.persist(checkpoint)
    installed = path.stat()
    assert installed.st_uid == os.geteuid()
    assert stat.S_IMODE(installed.st_mode) == 0o600
    path.chmod(0o644)
    with pytest.raises(OneReservationCanaryRuntimeError, match="owner-controlled"):
        CanaryCheckpointJournal(path)

    stale_path = tmp_path / "stale-checkpoint.json"
    stale = tmp_path / f".{stale_path.name}.retained"
    stale.write_bytes(b"partial")
    stale.chmod(0o600)
    with pytest.raises(OneReservationCanaryRuntimeError, match="stale partial"):
        CanaryCheckpointJournal(stale_path)


def test_authority_surface_and_file_bounds_are_closed(tmp_path: Path) -> None:
    setup = _screen_setup(tmp_path, "authority-surface")
    boundaries = setup.runtime.boundaries(lambda _checkpoint: None)
    assert {field.name for field in fields(type(boundaries))} == {
        "observe_store",
        "screen_once",
        "qualification_once",
        "persist_checkpoint",
    }
    source_path = (
        Path(__file__).parents[1]
        / "cacheon/chain/one_reservation_canary_runtime.py"
    )
    source = source_path.read_text().lower()
    for forbidden in (
        "sql",
        "sqlite3",
        "._db",
        "settlement",
        "incentive",
        "weight",
        "supervisor",
    ):
        assert forbidden not in source
    tree = ast.parse(source)
    observer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "observe_store"
    )
    store_calls = {
        node.func.attr
        for node in ast.walk(observer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    }
    assert store_calls == {
        "get",
        "preview_evaluation_claim",
        "active_evaluation_leases",
        "pending_qualification_recovery",
        "close",
    }
    assert CANARY_CHECKPOINT_JOURNAL_SCHEMA.endswith("journal-v1")
    assert len(source_path.read_text().splitlines()) < 1000
    assert len(Path(__file__).read_text().splitlines()) < 1000
