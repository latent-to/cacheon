"""Read-only, privacy-safe validator operator diagnostics."""

from __future__ import annotations

import json

import pytest

import cacheon.cli as cli
from cacheon.chain.audit_log import append_chain_audit, pass_audit_record
from cacheon.chain.intake import FinalizedArrival, FinalizedIntakeStore, IntakeScope
from cacheon.chain.miner_feedback import format_miner_submissions, miner_submissions
from cacheon.chain.operator_status import (
    OperatorStatusError,
    format_reservation_status,
    reservation_status,
)
from cacheon.chain.validator_loop import PassResult
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
)
from cacheon.eval.remote_run_forensics import (
    append_event as append_run_event,
    journal_path,
    publish_worker_log,
)
from tests.test_chain_intake import ATTEMPT, _claim, _fingerprint, _h, _publish


SCOPE = IntakeScope("0x" + "0" * 64, 307)
BLOCK = 40
BLOCK_HASH = "0x" + "4" * 64


def _arrival(
    index: int,
    *,
    hotkey: str | None = None,
    content_hash: str | None = None,
) -> FinalizedArrival:
    return FinalizedArrival(
        hotkey or f"miner-{index}",
        content_hash or f"{index + 1:064x}",
        f"https://private-host.example/bundle-{index}.tar.gz?token=secret",
        BLOCK,
        BLOCK_HASH,
        index,
    )


def _store(tmp_path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(tmp_path / "state" / "intake.sqlite3", scope=SCOPE)


def _reserve(store: FinalizedIntakeStore, *arrivals: FinalizedArrival) -> None:
    store.reserve_finalized(
        arrivals,
        finalized_block=BLOCK,
        finalized_block_hash=BLOCK_HASH,
    )


def test_live_writer_read_is_snapshot_safe_and_redacts_private_fields(tmp_path):
    arrival = _arrival(0)
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        private_root = tmp_path / "private" / "worker-publication"
        private_root.mkdir(parents=True)
        store._db.execute(
            "UPDATE reservations SET status='failed',decision='',reason=?,"
            "publication_digest=?,publication_root=? WHERE reservation_id=?",
            (
                "fetch:https://private-host.example/path?token=secret",
                "a" * 64,
                str(private_root),
                arrival.reservation_id,
            ),
        )

        # Hold an uncommitted writer transaction.  WAL readers must still open and see
        # the last committed snapshot without acquiring FinalizedIntakeStore's lock.
        store._db.execute("BEGIN IMMEDIATE")
        store._db.execute(
            "UPDATE reservations SET reason='uncommitted_secret' "
            "WHERE reservation_id=?",
            (arrival.reservation_id,),
        )
        try:
            value = reservation_status(
                store.path,
                reservation_id=arrival.reservation_id,
            )
        finally:
            store._db.execute("ROLLBACK")

    assert value["schema"] == "cacheon.operator.reservation-status.v3"
    assert value["reservation"]["attribution"] == {
        "class": "unattributed",
        "basis": "status_without_typed_decision",
    }
    assert value["reservation"]["reason"]["code"] == "fetch:detail_redacted"
    assert value["reservation"]["reason"]["detail_redacted"] is True
    assert value["reservation"]["publication"] == {
        "recorded": True,
        "available_on_this_host": True,
    }
    assert value["queue"] is None
    encoded = json.dumps(value, sort_keys=True)
    rendered = format_reservation_status(value)
    for secret in (
        "https://private-host.example",
        "token=secret",
        str(private_root),
        "uncommitted_secret",
    ):
        assert secret not in encoded
        assert secret not in rendered
    assert "publication_root" not in encoded
    assert "url" not in value["reservation"]


def test_exact_selectors_and_ambiguity_refusal(tmp_path):
    duplicate_hash = "d" * 64
    first = _arrival(0, hotkey="shared", content_hash=duplicate_hash)
    second = _arrival(1, hotkey="shared")
    third = _arrival(2, hotkey="unique", content_hash=duplicate_hash)
    with _store(tmp_path) as store:
        _reserve(store, first, second, third)
        by_id = reservation_status(store.path, reservation_id=first.reservation_id)
        by_hash = reservation_status(store.path, content_hash=second.content_hash)
        by_hotkey = reservation_status(store.path, hotkey="unique")

        assert by_id["reservation"]["reservation_id"] == first.reservation_id
        assert by_hash["reservation"]["reservation_id"] == second.reservation_id
        assert by_hotkey["reservation"]["reservation_id"] == third.reservation_id
        with pytest.raises(OperatorStatusError, match="matches 2 reservations"):
            reservation_status(store.path, content_hash=duplicate_hash)
        with pytest.raises(OperatorStatusError, match="matches 2 reservations"):
            reservation_status(store.path, hotkey="shared")
        with pytest.raises(OperatorStatusError, match="exactly one"):
            reservation_status(
                store.path,
                reservation_id=first.reservation_id,
                hotkey="shared",
            )
        with pytest.raises(OperatorStatusError, match="lowercase hexadecimal"):
            reservation_status(store.path, reservation_id="A" * 64)


def test_queue_position_matches_actual_selectable_lane_order(tmp_path):
    first, second, third = (_arrival(index) for index in range(3))
    with _store(tmp_path) as store:
        _reserve(store, first, second, third)
        store._db.execute(
            "UPDATE reservations SET status='published' WHERE reservation_id IN (?,?)",
            (first.reservation_id, third.reservation_id),
        )
        store._db.execute(
            "UPDATE reservations SET status='reproduction_pending',"
            "screen_lane='reproduction' WHERE reservation_id=?",
            (second.reservation_id,),
        )

        reproduction = reservation_status(
            store.path, reservation_id=second.reservation_id
        )
        primary_first = reservation_status(
            store.path, reservation_id=first.reservation_id
        )
        primary_second = reservation_status(
            store.path, reservation_id=third.reservation_id
        )
        assert reproduction["queue"]["position"] == 1
        assert reproduction["queue"]["lane"] == "reproduction"
        assert primary_first["queue"]["position"] == 2
        assert primary_second["queue"]["position"] == 3
        assert primary_second["queue"]["depth"] == 3
        assert primary_second["queue"]["ordering_authority"] == (
            "reproduction_priority_then_finalized_arrival"
        )

        lease = store.claim_evaluation_lease(
            stage="qualification",
            owner="private-worker-name",
            current_block=BLOCK,
            lease_blocks=10,
        )
        assert lease is not None
        assert lease.members[0].reservation_id == second.reservation_id
        leased = reservation_status(store.path, reservation_id=second.reservation_id)
        assert leased["queue"] == {
            "phase": "arena_qualification",
            "state": "leased",
            "lane": "reproduction",
            "position": None,
            "depth": 2,
            "ordering_authority": "reproduction_and_retry_group_qualification_scheduler",
        }
        assert leased["evaluation_lease"] == {
            "lease_id": lease.lease_id,
            "generation": 1,
            "stage": "qualification",
            "claimed_block": BLOCK,
            "initial_expires_block": BLOCK + 10,
            "expires_block": BLOCK + 10,
            "member_position": 0,
            "member_count": 1,
            "prior_status": "reproduction_pending",
        }
        assert "private-worker-name" not in json.dumps(leased, sort_keys=True)
        primary_after_lease = reservation_status(
            store.path, reservation_id=first.reservation_id
        )
        assert primary_after_lease["queue"]["position"] == 1
        assert primary_after_lease["queue"]["depth"] == 2

        store._db.execute(
            "UPDATE reservations SET status='qualifying',screen_lane='primary' "
            "WHERE reservation_id=?",
            (first.reservation_id,),
        )
        active = reservation_status(store.path, reservation_id=first.reservation_id)
        assert active["queue"] == {
            "phase": "arena_qualification",
            "state": "active",
            "lane": "primary",
            "position": None,
            "depth": 1,
            "ordering_authority": "reproduction_priority_then_finalized_arrival",
        }


def test_typed_fail_disposition_explains_rejection_without_status_inference(tmp_path):
    arrival = _arrival(0)
    report_digest = "4" * 64
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        _publish(
            store, arrival.reservation_id, _fingerprint("target.a", "slot.a"),
            digest="d" * 64, root="/published/a",
        )
        _claim(store, arrival.reservation_id)
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "7" * 64,
                (
                    QualificationIntakeOutcome(
                        arrival.reservation_id,
                        "3" * 64,
                        "7" * 64,
                        QualificationDecision.FAIL,
                        "speed_regression",
                        False,
                        attempt_artifact_sha256=ATTEMPT.sha256,
                        report_digest=report_digest,
                    ),
                ),
                ATTEMPT,
            ),
            current_finalized_block=BLOCK,
        )
        value = reservation_status(store.path, reservation_id=arrival.reservation_id)

    assert value["reservation"]["status"] == "failed"
    assert value["reservation"]["attribution"] == {
        "class": "fail_disposition",
        "basis": "persisted_FAIL",
    }
    assert value["qualification_dispositions"] == [
        {
            "attempt_index": 0,
            "authority_digest": "7" * 64,
            "evidence_digest": ATTEMPT.sha256,
            "attempt_ref": ATTEMPT.to_dict(),
            "report_digest": report_digest,
            "failure_digest": "",
            "decision": "FAIL",
            "reason": {
                "code": "speed_regression",
                "digest": _h("speed_regression"),
                "detail_redacted": False,
            },
            "diagnostic_reference": "attempt_artifact",
        }
    ]
    assert "screens" not in value
    assert value["evidence_limitations"] == []
    assert (
        "qualification[0]: decision=FAIL reason=speed_regression "
        "reference=attempt_artifact"
    ) in format_reservation_status(value)


def test_digest_only_infrastructure_failure_is_reported_as_partial_evidence(tmp_path):
    arrival = _arrival(0)
    failure_digest = "f" * 64
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        store._db.execute(
            "UPDATE reservations SET status='no_decision',decision='NO_DECISION',"
            "reason='oci_backend',qualification_evidence_digest=? "
            "WHERE reservation_id=?",
            (failure_digest, arrival.reservation_id),
        )
        store._db.execute(
            "INSERT INTO qualification_dispositions("
            "reservation_id,attempt_index,authority_digest,authority_manifest_json,"
            "evidence_digest,attempt_ref_json,report_digest,failure_digest,decision,reason"
            ") VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                arrival.reservation_id,
                0,
                "a" * 64,
                "{}",
                failure_digest,
                "",
                "",
                failure_digest,
                "NO_DECISION",
                "oci_backend",
            ),
        )
        value = reservation_status(store.path, reservation_id=arrival.reservation_id)

    disposition = value["qualification_dispositions"][0]
    assert disposition["diagnostic_reference"] == "failure_digest_only"
    assert disposition["failure_digest"] == failure_digest
    assert value["reservation"]["attribution"]["class"] == (
        "validator_or_policy_no_decision"
    )
    assert value["evidence_limitations"] == [
        "qualification_failure_retained_by_digest_only"
    ]


def test_miner_report_joins_lease_transport_and_exact_worker_failure(tmp_path):
    arrival = _arrival(0)
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        path = store.path
    request_id = "a" * 64
    spool = tmp_path / "remote-worker"
    carrier = spool / "outbox" / f"00000000000000000001-{request_id}"
    carrier.mkdir(parents=True)
    (carrier / "request.json").write_text(
        json.dumps(
            {
                "lease": {"members": [{"reservation_id": arrival.reservation_id}]},
                "request_id": request_id,
            }
        )
    )
    (spool / "events.jsonl").write_text(
        json.dumps({"event": "request_transferred", "request_id": request_id})
        + "\n"
    )
    result = spool / "results" / request_id
    result.mkdir(parents=True)
    append_run_event(
        journal_path(result),
        request_id,
        "adapter.qualification",
        "failed",
        failure={
            "component": "sglang",
            "exceptions": [
                {
                    "frames": [
                        {
                            "component": "sglang",
                            "file": "/usr/local/lib/python3.12/dist-packages/sglang/runtime.py",
                            "function": "capture",
                            "line": 77,
                        }
                    ],
                    "message": "graph capture failed",
                    "type": "RuntimeError",
                }
            ],
        },
    )
    worker_log = publish_worker_log(result, request_id)
    (result / "result.json").write_text(
        json.dumps(
            {
                "artifacts": [worker_log],
                "failure_code": "adapter_epoch_failed",
                "request_id": request_id,
                "state": "no_decision",
            }
        )
    )

    report = miner_submissions(path, hotkey="miner-0", spool_roots=(spool,))
    text = format_miner_submissions(report)
    assert "remote request aaaaaaaaaaaaaaaa: no_decision (adapter_epoch_failed)" in text
    assert "adapter.qualification in sglang" in text
    assert "sglang/runtime.py:77 in capture" in text


def test_redacted_audit_chronology_is_attached_with_utc_time(tmp_path):
    arrival = _arrival(0)
    audit = tmp_path / "audit" / "chain.jsonl"
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        result = PassResult(BLOCK, BLOCK_HASH)
        result.reserved.append(arrival.reservation_id)
        append_chain_audit(audit, pass_audit_record(result, timestamp_ns=0))
        value = reservation_status(
            store.path,
            reservation_id=arrival.reservation_id,
            audit_log=audit,
        )

    assert value["audit_events"] == [
        {
            "audit_line": 1,
            "finalized_block": BLOCK,
            "timestamp_ns": 0,
            "timestamp_utc": "1970-01-01T00:00:00+00:00",
            "field": "reserved",
        }
    ]


def test_corrupt_audit_is_not_silently_used_for_miner_support(tmp_path):
    arrival = _arrival(0)
    audit = tmp_path / "corrupt.jsonl"
    audit.write_text('{"event":"pass","reservation":"secret"}\n')
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        with pytest.raises(OperatorStatusError, match="audit line 1 is corrupt"):
            reservation_status(
                store.path,
                reservation_id=arrival.reservation_id,
                audit_log=audit,
            )


def test_cli_emits_the_same_safe_json_and_refuses_unknown_rows(tmp_path, capsys):
    arrival = _arrival(0)
    with _store(tmp_path) as store:
        _reserve(store, arrival)
        assert (
            cli.main(
                [
                    "chain-reservation-status",
                    "--intake-db",
                    str(store.path),
                    "--reservation-id",
                    arrival.reservation_id,
                    "--json",
                ]
            )
            == 0
        )
        value = json.loads(capsys.readouterr().out)
        assert value["reservation"]["reservation_id"] == arrival.reservation_id
        assert "url" not in value["reservation"]

        assert (
            cli.main(
                [
                    "chain-reservation-status",
                    "--intake-db",
                    str(store.path),
                    "--reservation-id",
                    "f" * 64,
                ]
            )
            == 2
        )
        assert "no retained reservation" in capsys.readouterr().out
