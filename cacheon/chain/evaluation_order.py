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
