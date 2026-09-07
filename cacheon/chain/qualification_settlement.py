"""Transactional acceptance of a complete qualification without a second GPU run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.settlement import SettlementCandidate, SettlementQualification

if TYPE_CHECKING:
    from cacheon.chain.intake import FinalizedIntakeStore, IntakeReservation


def _insert_candidate(store: FinalizedIntakeStore, qualification: SettlementQualification,
                      root: str) -> SettlementCandidate:
    candidate = SettlementCandidate.from_qualification(qualification)
    store._db.execute(
        "INSERT INTO settlement_candidates(reservation_id,authority_digest,"
        "candidate_digest,candidate_json,evidence_root,reproduction_evidence_root,status) "
        "VALUES(?,?,?,?,?,'','pending')",
        (candidate.reservation_digest, qualification.qualification_authority_digest,
         candidate.digest, json.dumps(candidate.to_dict(), separators=(",", ":"), sort_keys=True),
         root),
    )
    return candidate


def retain_complete_pass(store: FinalizedIntakeStore, row: IntakeReservation,
                         qualification: SettlementQualification,
                         attempt_ref: EvidenceArtifactRef, root: Path,
                         current_finalized_block: int) -> None:
    """Retain the imported PASS and make it available to the existing settlement stage."""

    from cacheon.chain.intake import IntakeError

    prior = store._db.execute(
        "SELECT 1 FROM settlement_qualifications WHERE reservation_id=?",
        (row.reservation_id,),
    ).fetchone()
    if prior is not None or row.screen_lane != "primary":
        raise IntakeError("qualification is already retained; a second run is not accepted")
    store._db.execute(
        "INSERT INTO settlement_qualifications(reservation_id,reproduction_index,"
        "qualification_digest,qualification_json,attempt_ref_json,evidence_root,retained_block) "
        "VALUES(?,0,?,?,?,?,?)",
        (row.reservation_id, qualification.digest,
         json.dumps(qualification.to_dict(), separators=(",", ":"), sort_keys=True),
         json.dumps(attempt_ref.to_dict(), separators=(",", ":"), sort_keys=True),
         str(root), current_finalized_block),
    )
    _insert_candidate(store, qualification, str(root))


def accept_retained_primary_passes(store: FinalizedIntakeStore) -> None:
    """Finish old-controller primary PASSes on restart without repeating paid work."""

    from cacheon.chain.intake import IntakeError

    with store._transaction():
        rows = store._db.execute(
            "SELECT r.reservation_id FROM reservations r WHERE r.status='reproduction_pending' "
            "AND NOT EXISTS (SELECT 1 FROM evaluation_lease_members m "
            "WHERE m.reservation_id=r.reservation_id AND m.active=1) "
            "ORDER BY r.block,r.event_index,r.event_subindex,r.reservation_id"
        ).fetchall()
        for row in rows:
            retained = store._db.execute(
                "SELECT * FROM settlement_qualifications WHERE reservation_id=? "
                "ORDER BY reproduction_index", (row["reservation_id"],),
            ).fetchall()
            if len(retained) != 1 or retained[0]["reproduction_index"] != 0:
                raise IntakeError("pending primary qualification is inconsistent")
            item = retained[0]
            qualification = SettlementQualification.from_dict(json.loads(item["qualification_json"]))
            if qualification.digest != item["qualification_digest"]:
                raise IntakeError("retained primary qualification digest differs")
            candidate = _insert_candidate(store, qualification, item["evidence_root"])
            store._db.execute(
                "UPDATE reservations SET status='qualified',decision='PASS',reason='qualified' "
                "WHERE reservation_id=?", (row["reservation_id"],),
            )
            store.reopen_settlement_evidence(candidate)


def settlement_evidence_metadata(
    store,
    candidate: SettlementCandidate,
):
    """Reopen the exact accepted attempts before the store can settle or reward them."""

    from cacheon.chain.intake import IntakeError
    from cacheon.settlement import SettlementEvidence, SettlementQualification

    row = store._db.execute(
        "SELECT sc.evidence_root,sc.reproduction_evidence_root,"
        "sc.candidate_digest,r.status,r.decision FROM settlement_candidates sc "
        "JOIN reservations r USING(reservation_id) WHERE sc.reservation_id=?",
        (candidate.reservation_digest,),
    ).fetchone()
    if (
        row is None
        or row["candidate_digest"] != candidate.digest
        or row["status"] != "qualified"
        or row["decision"] != "PASS"
        or not row["evidence_root"]
        or (candidate.reproduction is not None and not row["reproduction_evidence_root"])
    ):
        raise IntakeError("settlement evidence no longer has standing authority")
    retained = tuple(
        store._db.execute(
            "SELECT reproduction_index,qualification_digest,qualification_json,"
            "attempt_ref_json,evidence_root FROM settlement_qualifications "
            "WHERE reservation_id=? ORDER BY reproduction_index",
            (candidate.reservation_digest,),
        )
    )
    if len(retained) != len(candidate.qualifications) or tuple(
        item["reproduction_index"] for item in retained
    ) != tuple(range(len(retained))):
        raise IntakeError("settlement candidate lacks its retained qualifications")
    qualifications = []
    references = []
    try:
        for item in retained:
            qualification = SettlementQualification.from_dict(
                json.loads(item["qualification_json"])
            )
            reference = EvidenceArtifactRef.from_dict(
                json.loads(item["attempt_ref_json"])
            )
            if (
                qualification.digest != item["qualification_digest"]
                or reference.sha256
                != qualification.qualification_attempt_digest
            ):
                raise IntakeError("retained reproduction identity differs")
            disposition = store._db.execute(
                "SELECT authority_digest,report_digest,decision FROM "
                "qualification_dispositions WHERE reservation_id=? "
                "AND evidence_digest=?",
                (
                    candidate.reservation_digest,
                    qualification.qualification_attempt_digest,
                ),
            ).fetchone()
            if (
                disposition is None
                or disposition["decision"] != "PASS"
                or disposition["authority_digest"]
                != qualification.qualification_authority_digest
                or disposition["report_digest"]
                != qualification.qualification_report_digest
            ):
                raise IntakeError("retained reproduction lost PASS authority")
            qualifications.append(qualification)
            references.append(reference)
    except IntakeError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntakeError(f"settlement reproduction is corrupt: {exc}") from None
    if tuple(qualifications) != candidate.qualifications:
        raise IntakeError("retained reproductions differ from settlement candidate")
    roots = tuple(Path(item["evidence_root"]) for item in retained)
    expected_roots = (Path(row["evidence_root"]),)
    if candidate.reproduction is not None:
        expected_roots += (Path(row["reproduction_evidence_root"]),)
    if roots != expected_roots:
        raise IntakeError("settlement reproduction roots differ")
    receipt = SettlementEvidence.bind(
        candidate,
        primary_attempt_ref=references[0],
        reproduction_attempt_ref=references[1] if len(references) == 2 else None,
    )
    return roots, tuple(references), receipt
