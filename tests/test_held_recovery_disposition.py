from __future__ import annotations

import dataclasses
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.chain.evaluation_recovery import (
    EvaluationRecovery,
    RecoveryEventType,
    RecoveryPhase,
    RecoveryResolution,
    evaluation_recovery_event_id,
    evaluation_recovery_id,
    reviewed_legacy_screen_only_reason_digests,
)
from cacheon.chain.held_recovery_disposition import (
    HeldRecoveryDispositionError,
    ReviewedLegacyScreenOnlyDisposition,
)
from cacheon.chain.intake import IntakeError
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.stack_identity import sha256_hex


def _fixtures():
    path = Path(__file__).with_name("test_remote_worker_request_plan.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_held_recovery_disposition_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _store(case: "_Case") -> RecoverableFinalizedIntakeStore:
    return RecoverableFinalizedIntakeStore(
        case.authority.fixtures._db_path(case.authority.root),
        case.authority.fixtures.POLICY,
        scope=case.authority.fixtures.SCOPE,
    )


def _request_ready(fixtures, authority, store, plan):
    recovery = store.pending_qualification_recovery()
    assert recovery is not None
    prepared = store.prepare_qualification_recovery(
        recovery, plan, current_block=authority.fixtures.BLOCK
    )
    assert fixtures._materialize(authority, plan).state == "carrier_materialized"
    committed = store.commit_recovery_publication(
        prepared, current_block=authority.fixtures.BLOCK
    )
    assert fixtures._publish(authority, plan).state == "request_ready"
    return store.observe_recovery_request_ready(
        committed, current_block=authority.fixtures.BLOCK
    )


@dataclass(frozen=True)
class _Case:
    authority: object
    plan: object
    held: EvaluationRecovery
    disposition: ReviewedLegacyScreenOnlyDisposition
    outbox_names: tuple[str, ...]


def _case(tmp_path: Path, profile: str) -> _Case:
    fixtures = _fixtures()
    authority = fixtures._authority(
        tmp_path / profile,
        endpoint=f"endpoint-{profile}",
        profile=profile,
        recoverable=True,
    )
    plan = fixtures._plan(authority)
    with RecoverableFinalizedIntakeStore(
        authority.fixtures._db_path(authority.root),
        authority.fixtures.POLICY,
        scope=authority.fixtures.SCOPE,
    ) as store:
        ready = _request_ready(fixtures, authority, store, plan)
        authority.results.mkdir(parents=True, exist_ok=True)
        spool.write_local_no_decision(
            authority.results,
            plan.request_dict(),
            "adapter_request_failed",
        )
        observation = fixtures._inspect(authority, plan)
        assert (
            observation.state,
            observation.failure_code,
            observation.response,
            observation.refusal,
        ) == ("result_ready", "adapter_request_failed", None, None)
        held = store.hold_recovery(
            ready,
            current_block=authority.fixtures.BLOCK,
            reason="transport_hold:adapter_request_failed_screen_only_adapter",
        )
    result_root = authority.results / plan.request_id
    result_bytes = (result_root / "result.json").read_bytes()
    result = spool.strict_json_object(result_bytes.decode("utf-8"))
    artifact = next(
        row for row in result["artifacts"] if row["role"] == "adapter_result"
    )
    adapter_bytes = (result_root / "blobs" / artifact["sha256"]).read_bytes()
    disposition = ReviewedLegacyScreenOnlyDisposition.review(
        recovery=held,
        plan=plan,
        observation=observation,
        registration=authority.registration,
        result_envelope_bytes=result_bytes,
        adapter_result_bytes=adapter_bytes,
        result_envelope_digest=sha256_hex(result_bytes),
        adapter_result_blob_digest=sha256_hex(adapter_bytes),
        operator_review_authority_digest=sha256_hex(
            f"{profile}:review-authority".encode()
        ),
        operator_review_evidence_digest=sha256_hex(
            f"{profile}:review-evidence".encode()
        ),
    )
    return _Case(
        authority,
        plan,
        held,
        disposition,
        tuple(sorted(path.name for path in authority.outbox.iterdir())),
    )


def _snapshot(store, recovery: EvaluationRecovery) -> tuple[object, ...]:
    recovery_row = store._db.execute(
        "SELECT * FROM evaluation_recoveries WHERE recovery_id=?",
        (recovery.recovery_id,),
    ).fetchone()
    lease_row = store._db.execute(
        "SELECT * FROM evaluation_leases WHERE lease_id=?",
        (recovery.lease.lease_id,),
    ).fetchone()
    members = tuple(
        tuple(row)
        for row in store._db.execute(
            "SELECT * FROM evaluation_lease_members WHERE lease_id=? "
            "ORDER BY reservation_id",
            (recovery.lease.lease_id,),
        )
    )
    recovery_events = tuple(
        tuple(row)
        for row in store._db.execute(
            "SELECT * FROM evaluation_recovery_events WHERE recovery_id=? "
            "ORDER BY sequence",
            (recovery.recovery_id,),
        )
    )
    lease_events = tuple(
        tuple(row)
        for row in store._db.execute(
            "SELECT * FROM evaluation_lease_events WHERE lease_id=? ORDER BY sequence",
            (recovery.lease.lease_id,),
        )
    )
    reservations = tuple(
        (member.reservation_id, store.get(member.reservation_id).status)
        for member in recovery.lease.members
    )
    return (
        tuple(recovery_row),
        tuple(lease_row),
        members,
        recovery_events,
        lease_events,
        reservations,
    )


@pytest.mark.parametrize(
    "profile", ("collective-alpha-norm", "attention-beta-projection")
)
def test_reviewed_legacy_release_is_target_neutral_restart_safe_and_never_reuses_request(
    tmp_path: Path, profile: str
) -> None:
    case = _case(tmp_path, profile)
    original_plan_bytes = case.held.request_plan
    original_request_id = case.held.request_id

    with _store(case) as reopened:
        held = reopened.pending_qualification_recovery()
        assert held == case.held
        assert reopened.reopen_recovery_request_plan(held) == case.plan
        held_events = reopened.evaluation_recovery_events(held)
        assert [event.event_type for event in held_events[-2:]] == [
            RecoveryEventType.REQUEST_READY,
            RecoveryEventType.HELD,
        ]
        origin = held_events[-2]
        forged_event_type = RecoveryEventType.RENEWED
        forged_previous = dataclasses.replace(
            origin,
            event_id=evaluation_recovery_event_id(
                recovery_id=origin.recovery_id,
                lease_id=origin.lease_id,
                revision=origin.revision,
                event_type=forged_event_type,
                phase=origin.phase,
                resolution=origin.resolution,
                finalized_block=origin.finalized_block,
                expires_block=origin.expires_block,
                plan_digest=origin.plan_digest,
                request_id=origin.request_id,
                reason=origin.reason,
            ),
            event_type=forged_event_type,
        )
        forged_origin = (
            *held_events[:-2],
            forged_previous,
            held_events[-1],
        )
        with pytest.raises(
            HeldRecoveryDispositionError, match="exact held store state"
        ):
            case.disposition.require_exact_store_state(
                held,
                case.plan,
                forged_origin,
            )
        released = reopened.release_reviewed_legacy_screen_only_recovery(
            held,
            disposition=case.disposition,
            current_block=case.authority.fixtures.BLOCK,
        )
        assert released == held.lease
        assert reopened.pending_qualification_recovery() is None
        row = reopened._db.execute(
            "SELECT phase,resolution,request_plan,plan_digest,request_id,reason "
            "FROM evaluation_recoveries WHERE recovery_id=?",
            (held.recovery_id,),
        ).fetchone()
        assert (
            row["phase"],
            row["resolution"],
            bytes(row["request_plan"]),
            row["plan_digest"],
            row["request_id"],
        ) == (
            RecoveryPhase.REQUEST_READY.value,
            RecoveryResolution.PRE_RESIDENT_RELEASED.value,
            original_plan_bytes,
            case.plan.plan_digest,
            original_request_id,
        )
        reason_digests = reviewed_legacy_screen_only_reason_digests(row["reason"])
        assert reason_digests == (
            case.disposition.held_reason_digest,
            case.disposition.digest,
        )
        events = reopened.evaluation_recovery_events(held)
        assert [event.event_type.value for event in events[-2:]] == [
            "held",
            "pre_resident_released",
        ]
        assert events[-1].phase is RecoveryPhase.REQUEST_READY
        assert events[-1].reason == row["reason"]
        assert all(
            reopened.get(member.reservation_id).status == member.prior_status
            for member in held.lease.members
        )
        assert reopened._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_recoveries"
        ).fetchone()["n"] == 1

    assert tuple(sorted(path.name for path in case.authority.outbox.iterdir())) == (
        case.outbox_names
    )


def test_every_store_identity_drift_rejects_without_mutation(tmp_path: Path) -> None:
    case = _case(tmp_path, "identity-drift")
    with _store(case) as store:
        before = _snapshot(store, case.held)

        stale = dataclasses.replace(case.held, revision=case.held.revision - 1)
        wrong_request = dataclasses.replace(case.held, request_id="f" * 64)
        changed_lease = EvaluationLease(
            case.held.lease.lease_id,
            case.held.lease.generation + 1,
            case.held.lease.stage,
            case.held.lease.owner,
            case.held.lease.members,
            case.held.lease.claimed_block,
            case.held.lease.initial_expires_block,
            case.held.lease.expires_block,
        )
        wrong_generation = dataclasses.replace(
            case.held,
            lease=changed_lease,
            recovery_id=evaluation_recovery_id(changed_lease),
        )
        for recovery in (stale, wrong_request, wrong_generation):
            with pytest.raises((IntakeError, RuntimeError), match="stale|exact|forbidden"):
                store.release_reviewed_legacy_screen_only_recovery(
                    recovery,
                    disposition=case.disposition,
                    current_block=case.authority.fixtures.BLOCK,
                )
            assert _snapshot(store, case.held) == before

        other = _case(tmp_path, "foreign-target")
        with pytest.raises(IntakeError, match="forbidden"):
            store.release_reviewed_legacy_screen_only_recovery(
                case.held,
                disposition=other.disposition,
                current_block=case.authority.fixtures.BLOCK,
            )
        assert _snapshot(store, case.held) == before


def _reseal_registration(payload: bytes, **changes: object) -> dict[str, object]:
    value = spool.strict_json_object(payload.decode("utf-8"))
    value.update(changes)
    unsigned = dict(value)
    unsigned.pop("registration_digest")
    value["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, unsigned
    )
    return value


@pytest.mark.parametrize(
    "registration_change",
    (
        {"worker_epoch": "e" * 32},
        {"adapter_sha256": "e" * 64},
        {"remote_service_sha256": "e" * 64},
        {"service_identity": "foreign-service@" + "e" * 64},
    ),
)
def test_worker_registration_adapter_and_service_drift_cannot_form_disposition(
    tmp_path: Path, registration_change: dict[str, object]
) -> None:
    case = _case(tmp_path, "registration-drift")
    changed = _reseal_registration(
        case.disposition.registration_bytes, **registration_change
    )
    with _store(case) as store:
        before = _snapshot(store, case.held)
        with pytest.raises(HeldRecoveryDispositionError):
            ReviewedLegacyScreenOnlyDisposition.review(
                recovery=case.held,
                plan=case.plan,
                observation=case.disposition.observation,
                registration=changed,
                result_envelope_bytes=case.disposition.result_envelope_bytes,
                adapter_result_bytes=case.disposition.adapter_result_bytes,
                result_envelope_digest=case.disposition.result_envelope_digest,
                adapter_result_blob_digest=(
                    case.disposition.adapter_result_blob_digest
                ),
                operator_review_authority_digest=(
                    case.disposition.operator_review_authority_digest
                ),
                operator_review_evidence_digest=(
                    case.disposition.operator_review_evidence_digest
                ),
            )
        assert _snapshot(store, case.held) == before


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("failure_code", "adapter_start_failed"),
        ("decision", "FAIL"),
        ("resident_marker_state", "present"),
        ("resident_marker_state", "ambiguous"),
        ("review_basis", "marker_absence_alone.v1"),
        ("operator_review_authority_digest", "not-a-digest"),
        ("operator_review_evidence_digest", "not-a-digest"),
    ),
)
def test_failure_decision_marker_and_basis_are_closed_before_store_mutation(
    tmp_path: Path, field_name: str, value: str
) -> None:
    case = _case(tmp_path, f"closed-{field_name}-{value}")
    with _store(case) as store:
        before = _snapshot(store, case.held)
        with pytest.raises(HeldRecoveryDispositionError):
            dataclasses.replace(case.disposition, **{field_name: value})
        assert _snapshot(store, case.held) == before


@pytest.mark.parametrize(
    "which", ("envelope", "blob", "envelope-digest", "blob-digest")
)
def test_result_digest_or_blob_drift_rejects_without_mutation(
    tmp_path: Path, which: str
) -> None:
    case = _case(tmp_path, f"result-{which}")
    changes = {
        "envelope": {
            "result_envelope_bytes": case.disposition.result_envelope_bytes + b" "
        },
        "blob": {
            "adapter_result_bytes": case.disposition.adapter_result_bytes + b" "
        },
        "envelope-digest": {"result_envelope_digest": "e" * 64},
        "blob-digest": {"adapter_result_blob_digest": "e" * 64},
    }[which]
    with _store(case) as store:
        before = _snapshot(store, case.held)
        with pytest.raises(HeldRecoveryDispositionError):
            dataclasses.replace(case.disposition, **changes)
        assert _snapshot(store, case.held) == before


def test_expiry_repeat_and_nonheld_calls_reject_idempotently(tmp_path: Path) -> None:
    expired = _case(tmp_path, "expired")
    with _store(expired) as store:
        expiry = expired.held.lease.expires_block
        store.reserve_finalized(
            (),
            finalized_block=expiry,
            finalized_block_hash="0x" + f"{expiry:064x}",
        )
        before = _snapshot(store, expired.held)
        with pytest.raises(IntakeError, match="release is forbidden"):
            store.release_reviewed_legacy_screen_only_recovery(
                expired.held,
                disposition=expired.disposition,
                current_block=expiry,
            )
        assert _snapshot(store, expired.held) == before

    released = _case(tmp_path, "repeat")
    with _store(released) as store:
        store.release_reviewed_legacy_screen_only_recovery(
            released.held,
            disposition=released.disposition,
            current_block=released.authority.fixtures.BLOCK,
        )
        after = _snapshot(store, released.held)
        with pytest.raises((IntakeError, RuntimeError), match="resolved|HOLD|forbidden"):
            store.release_reviewed_legacy_screen_only_recovery(
                released.held,
                disposition=released.disposition,
                current_block=released.authority.fixtures.BLOCK,
            )
        assert _snapshot(store, released.held) == after

    nonheld = _case(tmp_path, "nonheld")
    request_ready = dataclasses.replace(
        nonheld.held,
        revision=nonheld.held.revision - 1,
        phase=RecoveryPhase.REQUEST_READY,
        reason="",
    )
    with _store(nonheld) as store:
        before = _snapshot(store, nonheld.held)
        with pytest.raises((IntakeError, RuntimeError), match="stale|forbidden|HOLD"):
            store.release_reviewed_legacy_screen_only_recovery(
                request_ready,
                disposition=nonheld.disposition,
                current_block=nonheld.authority.fixtures.BLOCK,
            )
        assert _snapshot(store, nonheld.held) == before
