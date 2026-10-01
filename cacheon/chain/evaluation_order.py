"""The completed arrival prefix shared by settlement, rewards and the dashboard."""

from decimal import Decimal
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


def reward_comparisons(db) -> dict[str, dict]:
    """Compare each PASS with the best preceding rewarded PASS on its measured baseline.

    Prefix eligibility only says earlier work finished. It does not establish a
    performance record. Recompute this filter for historical and new PASSes so
    an old eligibility bit cannot preserve a non-winning reward claim.
    """
    import json
    from decimal import Decimal

    best = {}
    seen = set()
    comparisons = {}
    grandfathered = set(reward_grandfathered_runtimes(db))
    rows = db.execute(
        "SELECT sc.* FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
        "WHERE r.status='qualified' AND r.decision='PASS' "
        "AND sc.status!='duplicate_proposal' AND " + reward_visibility_sql(db) +
        " ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash"
    ).fetchall()
    for row in rows:
        payload = json.loads(row["candidate_json"])
        # Read economic fields without reinterpreting historical audit schemas.
        # The producer separately reopens and validates every retained candidate.
        primary = payload["primary"]
        exempt = primary["incumbent_manifest"]["runtime_digest"] in grandfathered
        manifest = primary["candidate_manifest"]
        if manifest is None:
            continue
        contribution = manifest["entries"][primary["target_id"]]
        identity = (primary["arena_digest"], primary["target_id"],
                    json.dumps(contribution, sort_keys=True, separators=(",", ":")))
        if identity in seen and not exempt:
            continue
        seen.add(identity)
        # Scores describe the complete workload, including different target slots.
        # A newly commissioned baseline has a different denominator.
        group = (primary["arena_digest"], primary["incumbent_stack_digest"])
        qualifications = tuple(payload[key] for key in ("primary", "reproduction") if key in payload)
        score = min(Decimal(q["speedup"]) for q in qualifications)
        previous, previous_id = best.get(group, (Decimal(1), None))
        relative = score / previous
        margin = Decimal(0)
        if not exempt and previous_id is not None and score > previous:
            margin = _reward_min_margin(db, qualifications)
        eligible = exempt or previous_id is None or (score > previous and
            score >= previous * (1 + margin))
        comparisons[row["reservation_id"]] = {
            "previous_best_reservation_id": previous_id,
            "previous_best_speedup": previous,
            "relative_speedup": relative,
            "score_speedup": score if exempt else relative,
            "reward_eligible": eligible,
            "grandfathered": exempt,
        }
        if eligible and score > previous:
            best[group] = (score, row["reservation_id"])
    return comparisons


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


# Owner ruling 2026-09-30: a later V17 PASS pays when it beats the previous best by 1.5%. The
# statistical contrast it replaces (about 3% at the median) paid a truly 3%-better successor ~27%
# of the time; 1.5% pays it ~81%.
V17_REWARD_MARGIN = Decimal("0.015")


def _reward_min_margin(db, qualifications):
    """Use the ruled V17 margin for statistical policies; preserve historical sealed margins."""
    from cacheon.chain.intake import IntakeError

    policies = [report["speed_witness"]["resident_policy"] for report in _reward_reports(db, qualifications)]
    statistical = [policy.get("version") == 17 for policy in policies]
    if any(statistical) and not all(statistical):
        raise IntakeError("reward comparison mixes statistical and historical qualifications")
    if any(statistical):
        return V17_REWARD_MARGIN
    margins = [Decimal(str(policy["min_margin"])) for policy in policies]
    if any(not value.is_finite() or not 0 < value < 1 for value in margins):
        raise IntakeError("reward comparison margin is invalid")
    return max(margins)


def _reward_reports(db, qualifications):
    """Reopen the existing attempt authority without introducing another reward record."""
    import json
    from pathlib import Path

    from cacheon.chain.intake import IntakeError
    from cacheon.eval.evidence_store import EvidenceArtifactRef, reopen_evidence

    result = []
    rows = db.execute(
        "SELECT attempt_ref_json,evidence_root FROM settlement_qualifications "
        "WHERE reservation_id=? ORDER BY reproduction_index",
        (qualifications[0]["reservation_digest"],),
    ).fetchall()
    if len(rows) != len(qualifications):
        raise IntakeError("reward comparison lacks retained qualifications")
    for row, qualification in zip(rows, qualifications, strict=True):
        reference = EvidenceArtifactRef.from_dict(json.loads(row["attempt_ref_json"]))
        if reference.sha256 != qualification["qualification_attempt_digest"]:
            raise IntakeError("reward comparison attempt differs from qualification")
        try:
            payload = json.loads(reopen_evidence(Path(row["evidence_root"]), reference))
            reports = payload.get("reports", [payload])
            if "reports" in payload:
                reports = [report for report in reports
                           if report["selected_delta_digest"] == qualification["selected_delta_digest"]]
            if len(reports) != 1:
                raise ValueError("reward comparison report is ambiguous")
            policy = reports[0]["speed_witness"]["resident_policy"]
            if policy.get("version") != 17:
                policy["min_margin"]
            result.append(reports[0])
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            raise IntakeError(f"reward comparison cannot read retained margin: {exc}") from None
    return result
