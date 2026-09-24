"""The completed arrival prefix shared by settlement, rewards and the dashboard."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.chain.intake import FinalizedIntakeStore

# Callers join reservations as r. An infrastructure HOLD remains unresolved;
# later measurements may finish and release devices, but cannot earn yet.
# Physical worker state is deliberately absent: later active jobs do not block
# earlier completed submissions. Existing terminal disposition policy owns
# whether an expiry or FAIL resolves a reservation.
COMPLETED_ARRIVAL_PREFIX = """
NOT EXISTS (
    SELECT 1 FROM reservations AS predecessor
    WHERE predecessor.competition_arena=r.competition_arena
      AND (predecessor.block,predecessor.event_index,predecessor.event_subindex,
           predecessor.hotkey,predecessor.content_hash)
        < (r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash)
      AND predecessor.status NOT IN ('qualified','failed','expired')
)
"""


def ensure_reward_prefix(store: "FinalizedIntakeStore") -> None:
    """Extend settlement candidates with durable, monotone reward eligibility.

    Preserve previously earned PASSes on migration. A later reopening of an
    earlier submission must not retract other miners' already finalized credit.
    The remeasurement authority removes only the reopened candidate's record.
    """
    with store._transaction():
        columns = {row["name"] for row in store._db.execute("PRAGMA table_info(settlement_candidates)")}
        if "reward_eligible" not in columns:
            store._db.execute(
                "ALTER TABLE settlement_candidates ADD COLUMN reward_eligible "
                "INTEGER NOT NULL DEFAULT 0 CHECK(reward_eligible IN (0,1))"
            )
            store._db.execute(
                "UPDATE settlement_candidates SET reward_eligible=1 WHERE reservation_id IN ("
                "SELECT reservation_id FROM reservations WHERE status='qualified' AND decision='PASS')"
            )
        finalize_reward_prefix(store)


def finalize_reward_prefix(store: "FinalizedIntakeStore") -> None:
    """Make only the completed arrival prefix available to reward consumers."""
    with store._transaction():
        store._db.execute(
            "UPDATE settlement_candidates SET reward_eligible=1 WHERE reward_eligible=0 "
            "AND reservation_id IN (SELECT r.reservation_id FROM reservations r "
            "WHERE r.status='qualified' AND r.decision='PASS' AND "
            + COMPLETED_ARRIVAL_PREFIX + ")"
        )


def reward_visibility_sql(db) -> str:
    """Read upgraded and historical arena databases without migrating a dashboard source."""
    columns = {row[1] for row in db.execute("PRAGMA table_info(settlement_candidates)")}
    return "sc.reward_eligible=1" if "reward_eligible" in columns else "1"


def reward_winner_ids(db) -> set[str]:
    """Select threshold-clearing records in arrival order within each measured baseline.

    Prefix eligibility only says earlier work finished. It does not establish a
    performance record. Recompute this filter for historical and new PASSes so
    an old eligibility bit cannot preserve a non-winning reward claim.
    """
    import json
    from decimal import Decimal

    from cacheon.chain.intake import IntakeError
    from cacheon.settlement import SettlementCandidate

    best = {}
    seen = set()
    winners = set()
    grandfathered = set(reward_grandfathered_runtimes(db))
    rows = db.execute(
        "SELECT sc.* FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
        "WHERE r.status='qualified' AND r.decision='PASS' "
        "AND sc.status!='duplicate_proposal' AND " + reward_visibility_sql(db) +
        " ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash"
    ).fetchall()
    for row in rows:
        candidate = SettlementCandidate.from_dict(json.loads(row["candidate_json"]))
        if candidate.digest != row["candidate_digest"]:
            raise IntakeError("reward candidate digest differs from stored bytes")
        if candidate.candidate_manifest is None:
            continue
        contribution = candidate.candidate_manifest.entries[candidate.target_id]
        identity = (candidate.arena_digest, candidate.target_id, contribution.digest)
        if identity in seen:
            continue
        seen.add(identity)
        if candidate.incumbent_manifest.runtime_digest in grandfathered:
            winners.add(candidate.reservation_digest)
            continue
        # Scores describe the complete workload, including different target slots.
        # A newly commissioned baseline has a different denominator.
        group = (candidate.arena_digest, candidate.incumbent_stack_digest)
        score = Decimal(candidate.speedup)
        previous = best.get(group)
        best[group] = max(score, previous or score)
        if previous is None or (score > previous and
                score >= previous * (1 + _reward_min_margin(db, candidate))):
            winners.add(candidate.reservation_digest)
    return winners


def reward_grandfathered_runtimes(db) -> list[str]:
    """Read the operator's frozen pre-policy runtime generations."""
    import json
    from cacheon.stack_identity import require_sha256_hex

    row = db.execute(
        "SELECT value FROM metadata WHERE key='reward_grandfathered_runtimes'"
    ).fetchone()
    values = [] if row is None else json.loads(row[0])
    if not isinstance(values, list) or values != sorted(set(values)):
        raise ValueError("grandfathered reward runtimes must be a sorted unique list")
    for value in values:
        require_sha256_hex(value, field="grandfathered reward runtime")
    return values


def _reward_min_margin(db, candidate):
    """Read the configured margin from the same retained attempts as the score."""
    import json
    from decimal import Decimal
    from pathlib import Path

    from cacheon.chain.intake import IntakeError
    from cacheon.eval.evidence_store import EvidenceArtifactRef, reopen_evidence

    margins = []
    rows = db.execute(
        "SELECT attempt_ref_json,evidence_root FROM settlement_qualifications "
        "WHERE reservation_id=? ORDER BY reproduction_index",
        (candidate.reservation_digest,),
    ).fetchall()
    if len(rows) != len(candidate.qualifications):
        raise IntakeError("reward comparison lacks retained qualifications")
    for row, qualification in zip(rows, candidate.qualifications, strict=True):
        reference = EvidenceArtifactRef.from_dict(json.loads(row["attempt_ref_json"]))
        if reference.sha256 != qualification.qualification_attempt_digest:
            raise IntakeError("reward comparison attempt differs from qualification")
        try:
            payload = json.loads(reopen_evidence(Path(row["evidence_root"]), reference))
            reports = payload.get("reports", [payload])
            if "reports" in payload:
                reports = [report for report in reports
                           if report["selected_delta_digest"] == candidate.selected_delta_digest]
            if len(reports) != 1:
                raise ValueError("reward comparison report is ambiguous")
            margin = Decimal(str(reports[0]["speed_witness"]["resident_policy"]["min_margin"]))
            if not margin.is_finite() or not 0 < margin < 1:
                raise ValueError("reward comparison margin is invalid")
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            raise IntakeError(f"reward comparison cannot read retained margin: {exc}") from None
        margins.append(margin)
    return max(margins)
