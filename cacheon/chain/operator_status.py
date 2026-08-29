"""Privacy-safe, read-only operator view of one durable miner reservation.

The long-running validator owns the writable :class:`FinalizedIntakeStore` lock.
Support and incident-response commands therefore inspect the live WAL database through
an explicit read transaction instead of pretending to be another controller.  The
result contains finalized arrival authority, actual selectable queue position, typed
screen and qualification references, and only path-free evidence availability.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


class OperatorStatusError(RuntimeError):
    """The requested operator status cannot be read safely and unambiguously."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_SAFE_REASON = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_INTAKE_QUEUED = frozenset({"reserved", "transport_retry"})
_INTAKE_ACTIVE = frozenset({"fetching"})
_ARENA_QUEUED = frozenset({"published", "reproduction_pending"})
_ARENA_ACTIVE = frozenset({"screening", "promoted", "qualifying"})


def _readonly_connection(path: str | Path) -> sqlite3.Connection:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.is_file():
        raise OperatorStatusError(f"intake database does not exist: {source}")
    uri = f"file:{quote(str(source), safe='/')}?mode=ro"
    try:
        db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA foreign_keys=ON")
    except sqlite3.Error as exc:
        raise OperatorStatusError(
            f"cannot open intake database read-only: {exc}"
        ) from None
    return db


def _cursor(db: sqlite3.Connection) -> dict[str, object] | None:
    row = db.execute(
        "SELECT value FROM metadata WHERE key='finalized_cursor'"
    ).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row["value"])
    except (TypeError, ValueError) as exc:
        raise OperatorStatusError(f"finalized cursor is corrupt: {exc}") from None
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or value[0] < 0
        or not isinstance(value[1], str)
        or _BLOCK_HASH.fullmatch(value[1]) is None
    ):
        raise OperatorStatusError("finalized cursor is malformed")
    return {"block": value[0], "block_hash": value[1]}


def _reason(value: object) -> dict[str, object]:
    """Return a useful disposition code without printing an exception or URL."""

    if not isinstance(value, str) or not value:
        return {"code": None, "digest": None, "detail_redacted": False}
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    if _SAFE_REASON.fullmatch(value) is not None:
        return {"code": value, "digest": digest, "detail_redacted": False}
    prefix = value.partition(":")[0].strip().lower()
    code = prefix if _SAFE_REASON.fullmatch(prefix) is not None else "untyped"
    return {
        "code": f"{code}:detail_redacted",
        "digest": digest,
        "detail_redacted": True,
    }


def _path_availability(value: object) -> dict[str, bool]:
    """Report retention without returning the private filesystem location."""

    if not isinstance(value, str) or not value:
        return {"recorded": False, "available_on_this_host": False}
    try:
        available = Path(value).is_dir()
    except (OSError, ValueError):
        available = False
    return {"recorded": True, "available_on_this_host": available}


def _attribution(row: sqlite3.Row) -> dict[str, str]:
    """Classify only from a persisted typed decision, never ``status`` alone."""

    decision = row["decision"]
    status = row["status"]
    if decision == "FAIL":
        return {"class": "fail_disposition", "basis": "persisted_FAIL"}
    if decision == "NO_DECISION":
        return {
            "class": "validator_or_policy_no_decision",
            "basis": "persisted_NO_DECISION",
        }
    if decision == "PASS" or status == "qualified":
        return {"class": "qualified", "basis": "persisted_PASS_pair"}
    if status == "expired":
        return {"class": "submission_window", "basis": "persisted_expiry"}
    if status in _INTAKE_QUEUED | _INTAKE_ACTIVE | _ARENA_QUEUED | _ARENA_ACTIVE:
        return {"class": "pending", "basis": "active_state"}
    # In particular, ``failed`` without FAIL is not converted into candidate blame.
    return {"class": "unattributed", "basis": "status_without_typed_decision"}


def _ordered_ids(db: sqlite3.Connection, sql: str) -> list[str]:
    return [row["reservation_id"] for row in db.execute(sql).fetchall()]


def _lease_schema_present(db: sqlite3.Connection) -> bool:
    names = {
        row["name"]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('evaluation_leases','evaluation_lease_members')"
        )
    }
    if not names:
        return False
    if names != {"evaluation_leases", "evaluation_lease_members"}:
        raise OperatorStatusError("evaluation lease schema is incomplete")
    return True


def _active_evaluation_lease(
    db: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    lease_schema: bool,
) -> dict[str, object] | None:
    if not lease_schema:
        return None
    rows = db.execute(
        "SELECT el.lease_id,el.generation,el.stage,el.claimed_block,"
        "el.initial_expires_block,el.expires_block,el.state,em.position,"
        "em.prior_status,(SELECT COUNT(*) FROM evaluation_lease_members AS cohort "
        "WHERE cohort.lease_id=el.lease_id) AS member_count "
        "FROM evaluation_lease_members AS em JOIN evaluation_leases AS el "
        "USING(lease_id) WHERE em.reservation_id=? AND em.active=1",
        (row["reservation_id"],),
    ).fetchall()
    if len(rows) > 1:
        raise OperatorStatusError("reservation has more than one active evaluation lease")
    if not rows:
        return None
    lease = rows[0]
    if lease["state"] != "active" or lease["prior_status"] != row["status"]:
        raise OperatorStatusError(
            "active evaluation lease differs from the reservation queue state"
        )
    return {
        "lease_id": lease["lease_id"],
        "generation": lease["generation"],
        "stage": lease["stage"],
        "claimed_block": lease["claimed_block"],
        "initial_expires_block": lease["initial_expires_block"],
        "expires_block": lease["expires_block"],
        "member_position": lease["position"],
        "member_count": lease["member_count"],
        "prior_status": lease["prior_status"],
    }


def _unleased_clause(alias: str, lease_schema: bool) -> str:
    if not lease_schema:
        return ""
    return (
        " AND NOT EXISTS (SELECT 1 FROM evaluation_lease_members AS em "
        f"WHERE em.reservation_id={alias}.reservation_id AND em.active=1)"
    )


def _queue_position(
    db: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    active_lease: dict[str, object] | None,
    lease_schema: bool,
) -> dict[str, object] | None:
    status = row["status"]
    if active_lease is not None:
        stage = active_lease["stage"]
        if stage == "screen":
            predicate = "r.status IN ('published','reproduction_pending')"
            ordering = "reproduction_priority_then_finalized_arrival"
            phase = "arena_screen"
        elif stage == "qualification":
            predicate = "r.status='promoted'"
            ordering = "reproduction_and_retry_group_qualification_scheduler"
            phase = "arena_qualification"
        else:
            raise OperatorStatusError("active evaluation lease stage is unsupported")
        depth = db.execute(
            "SELECT COUNT(*) AS n FROM reservations AS r WHERE "
            + predicate
            + _unleased_clause("r", lease_schema)
        ).fetchone()["n"]
        prior = active_lease["prior_status"]
        lane = (
            "reproduction"
            if prior == "reproduction_pending" or row["screen_lane"] == "reproduction"
            else "primary"
        )
        return {
            "phase": phase,
            "state": "leased",
            "lane": lane,
            "position": None,
            "depth": depth,
            "ordering_authority": ordering,
        }
    if status in _INTAKE_QUEUED:
        ordered = _ordered_ids(
            db,
            "SELECT reservation_id FROM reservations WHERE status IN "
            "('reserved','transport_retry') ORDER BY "
            "block,event_index,event_subindex,hotkey,content_hash",
        )
        return {
            "phase": "intake_transport",
            "state": "queued",
            "lane": "intake",
            "position": ordered.index(row["reservation_id"]) + 1,
            "depth": len(ordered),
            "ordering_authority": "finalized_arrival",
        }
    if status in _INTAKE_ACTIVE:
        depth = db.execute(
            "SELECT COUNT(*) AS n FROM reservations WHERE status IN "
            "('reserved','transport_retry')"
        ).fetchone()["n"]
        return {
            "phase": "intake_transport",
            "state": "active",
            "lane": "intake",
            "position": None,
            "depth": depth,
            "ordering_authority": "finalized_arrival",
        }
    if status in _ARENA_QUEUED:
        # Keep the exact selector text shared with the lease-aware scheduler.  A
        # separate branch avoids referring to a new table when inspecting a legacy
        # database that has not yet been migrated by its writer.
        sql = (
            "SELECT r.reservation_id FROM reservations AS r WHERE r.status IN "
            "('published','reproduction_pending')"
            + _unleased_clause("r", lease_schema)
            + " ORDER BY CASE r.status WHEN 'reproduction_pending' THEN 0 ELSE 1 END,"
            "r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash"
        )
        ordered = _ordered_ids(db, sql)
        lane = "reproduction" if status == "reproduction_pending" else "primary"
        return {
            "phase": "arena_screen",
            "state": "queued",
            "lane": lane,
            "position": ordered.index(row["reservation_id"]) + 1,
            "depth": len(ordered),
            "ordering_authority": "reproduction_priority_then_finalized_arrival",
        }
    if status in _ARENA_ACTIVE:
        if status == "promoted":
            depth = db.execute(
                "SELECT COUNT(*) AS n FROM reservations AS r WHERE r.status='promoted'"
                + _unleased_clause("r", lease_schema)
            ).fetchone()["n"]
            phase = "arena_qualification"
            state = "awaiting_qualification"
            ordering = "reproduction_and_retry_group_qualification_scheduler"
        else:
            depth = db.execute(
                "SELECT COUNT(*) AS n FROM reservations AS r WHERE r.status IN "
                "('published','reproduction_pending')"
                + _unleased_clause("r", lease_schema)
            ).fetchone()["n"]
            phase = "arena_screen" if status == "screening" else "arena_qualification"
            state = "active"
            ordering = "reproduction_priority_then_finalized_arrival"
        lane = row["screen_lane"] or "primary"
        return {
            "phase": phase,
            "state": state,
            "lane": lane,
            "position": None,
            "depth": depth,
            "ordering_authority": ordering,
        }
    return None


def _evidence_ref(encoded: object, *, field: str) -> dict[str, object] | None:
    if not encoded:
        return None
    if not isinstance(encoded, str):
        raise OperatorStatusError(f"{field} evidence reference is malformed")
    from cacheon.eval.evidence_store import EvidenceArtifactRef, EvidenceStoreError

    try:
        reference = EvidenceArtifactRef.from_dict(json.loads(encoded))
    except (EvidenceStoreError, TypeError, ValueError) as exc:
        raise OperatorStatusError(f"{field} evidence reference is corrupt: {exc}") from None
    return reference.to_dict()


def _screen_dispositions(
    db: sqlite3.Connection, reservation_id: str
) -> list[dict[str, object]]:
    from cacheon.arena_service import (
        ArenaScreenReceipt,
        ArenaServiceError,
        PromotionDecision,
        ScreenStageResult,
    )

    result: list[dict[str, object]] = []
    rows = db.execute(
        "SELECT attempt_index,service_digest,candidate_digest,receipt_digest,"
        "receipt_json,decision,stage_count,lane "
        "FROM arena_screen_dispositions "
        "WHERE reservation_id=? ORDER BY attempt_index",
        (reservation_id,),
    )
    for row in rows:
        try:
            raw = json.loads(row["receipt_json"])
            stages = tuple(
                ScreenStageResult.from_dict(item) for item in raw["results"]
            )
            receipt = ArenaScreenReceipt(
                raw["service_digest"],
                raw["candidate_digest"],
                raw["screen_attempt"],
                stages,
                PromotionDecision(raw["decision"]),
            )
        except (KeyError, TypeError, ValueError, ArenaServiceError) as exc:
            raise OperatorStatusError(f"screen receipt is corrupt: {exc}") from None
        if (
            receipt.digest != row["receipt_digest"]
            or receipt.service_digest != row["service_digest"]
            or receipt.candidate_digest != row["candidate_digest"]
            or receipt.decision.value != row["decision"]
            or len(receipt.results) != row["stage_count"]
            or receipt.screen_attempt != row["attempt_index"] + 1
        ):
            raise OperatorStatusError("screen receipt differs from its retained index")
        result.append(
            {
                "attempt_index": row["attempt_index"],
                "lane": row["lane"],
                "decision": receipt.decision.value,
                "service_digest": receipt.service_digest,
                "candidate_digest": receipt.candidate_digest,
                "receipt_digest": receipt.digest,
                "stages": [stage.to_dict() for stage in receipt.results],
            }
        )
    return result


def _qualification_dispositions(
    db: sqlite3.Connection, reservation_id: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    rows = db.execute(
        "SELECT attempt_index,authority_digest,evidence_digest,report_digest,"
        "failure_digest,decision,reason,attempt_ref_json "
        "FROM qualification_dispositions WHERE reservation_id=? "
        "ORDER BY attempt_index",
        (reservation_id,),
    )
    for row in rows:
        reference = _evidence_ref(
            row["attempt_ref_json"], field="qualification attempt"
        )
        if reference is not None and reference["sha256"] != row["evidence_digest"]:
            raise OperatorStatusError(
                "qualification attempt reference differs from its evidence digest"
            )
        if reference is None and row["failure_digest"]:
            if row["failure_digest"] != row["evidence_digest"]:
                raise OperatorStatusError(
                    "qualification failure digest differs from its evidence digest"
                )
            diagnostic_reference = "failure_digest_only"
        elif reference is not None:
            diagnostic_reference = "attempt_artifact"
        else:
            diagnostic_reference = "none"
        result.append(
            {
                "attempt_index": row["attempt_index"],
                "authority_digest": row["authority_digest"],
                "evidence_digest": row["evidence_digest"],
                "attempt_ref": reference,
                "report_digest": row["report_digest"],
                "failure_digest": row["failure_digest"],
                "decision": row["decision"],
                "reason": _reason(row["reason"]),
                "diagnostic_reference": diagnostic_reference,
            }
        )
    return result


def _settlement_qualifications(
    db: sqlite3.Connection, reservation_id: str
) -> list[dict[str, object]]:
    from cacheon.settlement import SettlementError, SettlementQualification

    result: list[dict[str, object]] = []
    rows = db.execute(
        "SELECT reproduction_index,qualification_digest,qualification_json,"
        "attempt_ref_json,evidence_root,retained_block "
        "FROM settlement_qualifications WHERE reservation_id=? "
        "ORDER BY reproduction_index",
        (reservation_id,),
    )
    for row in rows:
        try:
            qualification = SettlementQualification.from_dict(
                json.loads(row["qualification_json"])
            )
        except (SettlementError, TypeError, ValueError) as exc:
            raise OperatorStatusError(
                f"settlement qualification is corrupt: {exc}"
            ) from None
        reference = _evidence_ref(
            row["attempt_ref_json"], field="settlement qualification"
        )
        if (
            qualification.digest != row["qualification_digest"]
            or qualification.reservation_digest != reservation_id
            or reference is None
            or qualification.qualification_attempt_digest != reference["sha256"]
        ):
            raise OperatorStatusError(
                "settlement qualification differs from its retained index"
            )
        result.append(
            {
                "reproduction_index": row["reproduction_index"],
                "qualification_digest": qualification.digest,
                "retained_block": row["retained_block"],
                "lane": qualification.lane,
                "target_id": qualification.target_id,
                "selected_delta_digest": qualification.selected_delta_digest,
                "authority_digest": qualification.qualification_authority_digest,
                "attempt_ref": reference,
                "report_digest": qualification.qualification_report_digest,
                "speedup": qualification.speedup,
                "evidence": _path_availability(row["evidence_root"]),
            }
        )
    return result


def _utc_timestamp(value: object) -> str | None:
    if type(value) is not int or value < 0:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def _audit_events(
    path: str | Path | None, reservation_id: str
) -> list[dict[str, object]]:
    if path is None:
        return []
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.is_file():
        return []
    from cacheon.chain.audit_log import (
        ChainAuditLogError,
        validate_chain_audit_record,
    )

    events: list[dict[str, object]] = []
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    validate_chain_audit_record(record)
                except (TypeError, ValueError, ChainAuditLogError) as exc:
                    raise OperatorStatusError(
                        f"chain audit line {line_number} is corrupt: {exc}"
                    ) from None
                if record.get("event") != "pass":
                    continue
                matches: list[dict[str, object]] = []
                for field in ("reserved", "held"):
                    values = record[field]
                    if reservation_id in values:
                        matches.append({"field": field})
                for field in (
                    "published",
                    "copies",
                    "rejected",
                    "screens",
                    "decisions",
                    "settlements",
                ):
                    values = record[field]
                    if reservation_id in values:
                        matches.append({"field": field, "value": values[reservation_id]})
                for match in matches:
                    events.append(
                        {
                            "audit_line": line_number,
                            "finalized_block": record["finalized_block"],
                            "timestamp_ns": record["timestamp_ns"],
                            "timestamp_utc": _utc_timestamp(record["timestamp_ns"]),
                            **match,
                        }
                    )
    except OSError as exc:
        raise OperatorStatusError(f"cannot read chain audit log: {exc}") from None
    return events


def _evidence_limitations(
    row: sqlite3.Row,
    screens: list[dict[str, object]],
    qualifications: list[dict[str, object]],
    settlements: list[dict[str, object]],
) -> list[str]:
    limitations: list[str] = []
    if any(item["diagnostic_reference"] == "failure_digest_only" for item in qualifications):
        limitations.append("qualification_failure_retained_by_digest_only")
    if (
        row["status"] in {"failed", "held", "no_decision"}
        and not screens
        and not qualifications
    ):
        limitations.append("no_typed_failure_artifact_reference")
    if any(
        not item["evidence"]["available_on_this_host"]  # type: ignore[index]
        for item in settlements
    ):
        limitations.append("settlement_evidence_not_available_on_this_host")
    return limitations


def reservation_status(
    intake_db: str | Path,
    *,
    reservation_id: str | None = None,
    content_hash: str | None = None,
    hotkey: str | None = None,
    audit_log: str | Path | None = None,
) -> dict[str, object]:
    """Return one privacy-safe support record from a live validator database.

    Exactly one selector is required.  Content hash and hotkey selectors must resolve
    to one row so an operator can never accidentally explain the wrong submission.
    The SQLite reads share one explicit snapshot and do not acquire the controller's
    process lock or a write transaction.
    """

    selectors = [reservation_id is not None, content_hash is not None, hotkey is not None]
    if sum(selectors) != 1:
        raise OperatorStatusError(
            "exactly one of reservation_id, content_hash, or hotkey is required"
        )
    if reservation_id is not None and _SHA256.fullmatch(reservation_id) is None:
        raise OperatorStatusError(
            "reservation_id must be 64 lowercase hexadecimal characters"
        )
    if content_hash is not None and _SHA256.fullmatch(content_hash) is None:
        raise OperatorStatusError(
            "content_hash must be 64 lowercase hexadecimal characters"
        )
    if hotkey is not None and (
        not hotkey
        or len(hotkey) > 256
        or hotkey.strip() != hotkey
        or any(ord(character) < 32 for character in hotkey)
    ):
        raise OperatorStatusError("hotkey selector is malformed")

    db = _readonly_connection(intake_db)
    try:
        db.execute("BEGIN")
        column, selector = (
            ("reservation_id", reservation_id)
            if reservation_id is not None
            else ("content_hash", content_hash)
            if content_hash is not None
            else ("hotkey", hotkey)
        )
        rows = db.execute(
            f"SELECT * FROM reservations WHERE {column}=? ORDER BY "
            "block,event_index,event_subindex,hotkey,content_hash",
            (selector,),
        ).fetchall()
        if not rows:
            raise OperatorStatusError("no retained reservation matches the selector")
        if len(rows) != 1:
            raise OperatorStatusError(
                f"selector matches {len(rows)} reservations; use the exact reservation_id"
            )
        row = rows[0]
        lease_schema = _lease_schema_present(db)
        active_lease = _active_evaluation_lease(
            db, row, lease_schema=lease_schema
        )
        screens = _screen_dispositions(db, row["reservation_id"])
        qualifications = _qualification_dispositions(db, row["reservation_id"])
        settlements = _settlement_qualifications(db, row["reservation_id"])
        result: dict[str, Any] = {
            "schema": "cacheon.operator.reservation-status.v2",
            "cursor": _cursor(db),
            "arrival_authority": {
                "kind": "finalized_chain",
                "order_key": {
                    "block": row["block"],
                    "event_index": row["event_index"],
                    "event_subindex": row["event_subindex"],
                    "hotkey": row["hotkey"],
                    "content_hash": row["content_hash"],
                },
            },
            "reservation": {
                "reservation_id": row["reservation_id"],
                "hotkey": row["hotkey"],
                "content_hash": row["content_hash"],
                "block": row["block"],
                "block_hash": row["block_hash"],
                "event_index": row["event_index"],
                "event_subindex": row["event_subindex"],
                "status": row["status"],
                "decision": row["decision"] or None,
                "reason": _reason(row["reason"]),
                "attribution": _attribution(row),
                "target_id": row["target_id"] or None,
                "transport_attempts": row["transport_attempts"],
                "screen_attempts": row["screen_attempts"],
                "screen_lane": row["screen_lane"] or None,
                "publication_digest": row["publication_digest"] or None,
                "publication": _path_availability(row["publication_root"]),
                "qualification_evidence_digest": row[
                    "qualification_evidence_digest"
                ]
                or None,
            },
            "queue": _queue_position(
                db,
                row,
                active_lease=active_lease,
                lease_schema=lease_schema,
            ),
            "evaluation_lease": active_lease,
            "screens": screens,
            "qualification_dispositions": qualifications,
            "settlement_qualifications": settlements,
            "audit_events": _audit_events(audit_log, row["reservation_id"]),
            "evidence_limitations": _evidence_limitations(
                row, screens, qualifications, settlements
            ),
        }
        return result
    except OperatorStatusError:
        raise
    except sqlite3.Error as exc:
        raise OperatorStatusError(f"intake status query failed: {exc}") from None
    finally:
        if db.in_transaction:
            db.execute("ROLLBACK")
        db.close()


def _reason_text(value: object) -> str:
    if not isinstance(value, dict):
        return "-"
    code = value.get("code") or "-"
    digest = value.get("digest")
    suffix = f" digest={digest}" if value.get("detail_redacted") and digest else ""
    return f"{code}{suffix}"


def format_reservation_status(value: dict[str, object]) -> str:
    """Render the same privacy-safe support record as compact operator text."""

    row = value["reservation"]
    assert isinstance(row, dict)
    cursor = value.get("cursor")
    queue = value.get("queue")
    attribution = row["attribution"]
    assert isinstance(attribution, dict)
    lines = [
        f"reservation: {row['reservation_id']}",
        f"content_hash: {row['content_hash']}",
        f"hotkey: {row['hotkey']}",
        f"arrival_authority: finalized block={row['block']} "
        f"event={row['event_index']}.{row['event_subindex']} hash={row['block_hash']}",
        f"state: status={row['status']} decision={row['decision'] or '-'} "
        f"attribution={attribution['class']} reason={_reason_text(row['reason'])}",
    ]
    if isinstance(cursor, dict):
        lines.append(f"intake_cursor: {cursor['block']} {cursor['block_hash']}")
    if isinstance(queue, dict):
        position = (
            f"{queue['position']}/{queue['depth']}"
            if queue["position"] is not None
            else f"not-queued (waiting={queue['depth']})"
        )
        lines.append(
            f"queue: phase={queue['phase']} state={queue['state']} lane={queue['lane']} "
            f"position={position} ordering={queue['ordering_authority']}"
        )
    active_lease = value.get("evaluation_lease")
    if isinstance(active_lease, dict):
        lines.append(
            f"evaluation_lease: id={active_lease['lease_id']} "
            f"generation={active_lease['generation']} stage={active_lease['stage']} "
            f"member={active_lease['member_position'] + 1}/"
            f"{active_lease['member_count']} expires_block={active_lease['expires_block']}"
        )
    publication = row["publication"]
    assert isinstance(publication, dict)
    lines.extend(
        [
            f"publication: digest={row['publication_digest'] or '-'} "
            f"recorded={str(publication['recorded']).lower()} "
            f"available={str(publication['available_on_this_host']).lower()}",
            f"qualification_evidence: {row['qualification_evidence_digest'] or '-'}",
        ]
    )
    for screen in value["screens"]:  # type: ignore[assignment]
        lines.append(
            f"screen[{screen['attempt_index']}]: lane={screen['lane']} "
            f"decision={screen['decision']} receipt={screen['receipt_digest']}"
        )
        for stage in screen["stages"]:
            lines.append(
                f"  stage[{stage['stage']}]: grade={stage['grade']} "
                f"evidence={stage['evidence_digest']} elapsed_ms={stage['elapsed_ms']}"
            )
    for attempt in value["qualification_dispositions"]:  # type: ignore[assignment]
        lines.append(
            f"qualification[{attempt['attempt_index']}]: "
            f"decision={attempt['decision']} reason={_reason_text(attempt['reason'])} "
            f"reference={attempt['diagnostic_reference']} "
            f"evidence={attempt['evidence_digest'] or '-'} "
            f"report={attempt['report_digest'] or '-'} "
            f"failure={attempt['failure_digest'] or '-'}"
        )
    for qualification in value["settlement_qualifications"]:  # type: ignore[assignment]
        evidence = qualification["evidence"]
        lines.append(
            f"settlement_qualification[{qualification['reproduction_index']}]: "
            f"digest={qualification['qualification_digest']} "
            f"report={qualification['report_digest']} speedup={qualification['speedup']} "
            f"evidence_available={str(evidence['available_on_this_host']).lower()}"
        )
    for event in value["audit_events"]:  # type: ignore[assignment]
        suffix = f" value={event['value']}" if "value" in event else ""
        lines.append(
            f"audit[{event['audit_line']}]: block={event['finalized_block']} "
            f"utc={event['timestamp_utc']} field={event['field']}{suffix}"
        )
    for limitation in value["evidence_limitations"]:  # type: ignore[assignment]
        lines.append(f"evidence_limit: {limitation}")
    return "\n".join(lines)


__all__ = [
    "OperatorStatusError",
    "format_reservation_status",
    "reservation_status",
]
