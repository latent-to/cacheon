"""Arena-scoped queue and lineage operations for the shared intake authority.

The default competition retains its historical empty namespace. Named arenas
use the runtime's stable arena_id; service digests continue to name immutable
commissions within that competition. No signed historical identity is rewritten.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.chain.intake import IntakeReservation


def _error(message: str) -> RuntimeError:
    from cacheon.chain.intake import IntakeError

    return IntakeError(message)


class ArenaStateMixin:
    """Share one intake database while selecting one competition per dispatcher."""

    _competition_arena = ""

    def select_arena(self, arena_id: str, *, accept_legacy_bundles: bool) -> None:
        """Bind the dispatcher's verified runtime and the sole legacy alias."""
        from cacheon.arena_service import _identifier

        _identifier(arena_id, "arena_id")
        if type(accept_legacy_bundles) is not bool:
            raise _error("legacy arena selection must be boolean")
        with self._transaction():
            row = self._db.execute(
                "SELECT value FROM metadata WHERE key='legacy_arena_id'"
            ).fetchone()
            if accept_legacy_bundles:
                if row is not None and row["value"] != arena_id:
                    raise _error("legacy bundles already belong to another arena")
                if self._db.execute(
                    "SELECT 1 FROM evaluation_leases WHERE competition_arena=? UNION ALL "
                    "SELECT 1 FROM evaluation_stacks WHERE competition_arena=? LIMIT 1",
                    (arena_id, arena_id),
                ).fetchone() is not None:
                    raise _error("a commissioned named arena cannot become the legacy alias")
                self._db.execute(
                    "INSERT OR IGNORE INTO metadata(key,value) VALUES('legacy_arena_id',?)",
                    (arena_id,),
                )
                self._db.execute(
                    "UPDATE reservations SET competition_arena='' WHERE competition_arena=?",
                    (arena_id,),
                )
            elif row is not None and row["value"] == arena_id:
                raise _error("the legacy arena must retain its existing namespace")
        self._competition_arena = "" if accept_legacy_bundles else arena_id

    def publication_arena(self, arena_id: str) -> str:
        """Resolve a hash-bound bundle selector, preserving the legacy alias."""
        from cacheon.arena_service import _identifier

        if arena_id != "":
            _identifier(arena_id, "competition arena")
        legacy = self._db.execute(
            "SELECT value FROM metadata WHERE key='legacy_arena_id'"
        ).fetchone()
        return "" if legacy is not None and arena_id == legacy["value"] else arena_id

    def _ensure_arena_schema(self) -> None:
        from cacheon.chain.evaluation_order import ensure_reward_prefix
        ensure_reward_prefix(self)
        # Preserve the pre-namespace journal and lineage, including old stores
        # whose tips predate parent/speedup columns. Copy transactionally.
        fields = ("target_id", "artifact_digest", "parent_artifact_digest",
                  "winner_speedup", "arena_id", "stack_digest", "transition_event_id",
                  "crowned_block")
        with self._transaction():
            for table, key in (("target_lineage_tips", "target_id"),
                               ("target_lineage_nodes", "target_id,artifact_digest")):
                columns = {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
                migrate = bool(columns) and "competition_arena" not in columns
                if migrate:
                    self._db.execute(f"ALTER TABLE {table} RENAME TO {table}_unscoped")
                unique = ",UNIQUE(transition_event_id)" if table.endswith("nodes") else ""
                self._db.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                    competition_arena TEXT NOT NULL DEFAULT '', target_id TEXT NOT NULL,
                    artifact_digest TEXT NOT NULL, parent_artifact_digest TEXT NOT NULL,
                    winner_speedup TEXT NOT NULL, arena_id TEXT NOT NULL,
                    stack_digest TEXT NOT NULL, transition_event_id TEXT NOT NULL,
                    crowned_block INTEGER NOT NULL CHECK(crowned_block>=0),
                    PRIMARY KEY(competition_arena,{key}){unique}) STRICT""")
                if migrate:
                    projection = ",".join(name if name in columns else "''" for name in fields)
                    self._db.execute(f"INSERT INTO {table}({','.join(fields)}) "
                                     f"SELECT {projection} FROM {table}_unscoped")
                    self._db.execute(f"DROP TABLE {table}_unscoped")

    def _write_lineage(self, candidate, artifact, parent, event_id, crowned_block) -> None:
        """Use the same lineage update during settlement and journal backfill."""
        scope = self.get(candidate.reservation_digest).competition_arena
        fields = ("competition_arena", "target_id", "artifact_digest", "parent_artifact_digest",
                  "winner_speedup", "arena_id", "stack_digest", "transition_event_id", "crowned_block")
        values = (scope, candidate.target_id, artifact, parent, candidate.speedup,
                  candidate.arena_digest, candidate.candidate_stack_digest, event_id, crowned_block)
        for table, key in (("target_lineage_nodes", "competition_arena,target_id,artifact_digest"),
                           ("target_lineage_tips", "competition_arena,target_id")):
            updates = ",".join(f"{name}=excluded.{name}" for name in fields if name not in key.split(","))
            self._db.execute(
                f"INSERT INTO {table}({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?) "
                f"ON CONFLICT({key}) DO UPDATE SET {updates}", values,
            )

    def screenable(self, *, limit: int | None = None) -> tuple[IntakeReservation, ...]:
        """Return validator-selected work awaiting a fresh non-crown screen."""

        bound = self.policy.max_cohort if limit is None else limit
        if type(bound) is not int or bound <= 0 or bound > self.policy.max_cohort:
            raise _error("screen cohort limit is invalid")
        rows = self._db.execute(
            "SELECT r.* FROM reservations AS r WHERE status IN "
            "('published','reproduction_pending') AND NOT EXISTS ("
            "SELECT 1 FROM evaluation_lease_members AS em WHERE "
            "em.reservation_id=r.reservation_id AND em.active=1) AND r.competition_arena=? ORDER BY "
            "CASE status WHEN 'reproduction_pending' THEN 0 ELSE 1 END,"
            "block,event_index,event_subindex,hotkey,content_hash LIMIT ?",
            (self._competition_arena, bound),
        )
        return tuple(self._row(row) for row in rows)


    def arena_queue_snapshot(self, *, current_block: int):
        """Read capacity pressure within the selected competition."""
        from cacheon.arena_service import ArenaQueueSnapshot

        if type(current_block) is not int or current_block < 0:
            raise _error("arena queue block is malformed")
        rows = tuple(self._db.execute(
            "SELECT r.status,r.block,el.stage FROM reservations AS r LEFT JOIN "
            "evaluation_lease_members AS em ON em.reservation_id=r.reservation_id AND em.active=1 "
            "LEFT JOIN evaluation_leases AS el USING(lease_id) WHERE r.competition_arena=? "
            "AND r.status IN ('published','reproduction_pending','promoted','screening','qualifying')",
            (self._competition_arena,),
        ))
        queued = [row for row in rows if row["stage"] is None and row["status"] in
                  {"published", "reproduction_pending", "promoted"}]
        return ArenaQueueSnapshot(
            len(queued), max((current_block - row["block"] for row in queued), default=0),
            sum(row["status"] == "screening" or row["stage"] == "screen" for row in rows),
            sum(row["status"] == "qualifying" or row["stage"] == "qualification" for row in rows),
        )


    def target_lineage_tips(self, competition_arena: str | None = None) -> Mapping[str, object]:
        """Reopen each target's contiguous active root-to-tip lineage."""

        from cacheon.chain.intake import IntakeError
        from cacheon.settlement import TargetLineage, TargetLineageNode

        scope = self._competition_arena if competition_arena is None else competition_arena
        try:
            lineages: dict[str, TargetLineage] = {}
            for tip in self._db.execute(
                "SELECT target_id,artifact_digest FROM target_lineage_tips "
                "WHERE competition_arena=? ORDER BY target_id", (scope,),
            ):
                nodes: list[TargetLineageNode] = []
                artifact = tip["artifact_digest"]
                seen: set[str] = set()
                while artifact:
                    if artifact in seen:
                        raise IntakeError("target lineage contains a cycle")
                    seen.add(artifact)
                    row = self._db.execute(
                        "SELECT * FROM target_lineage_nodes "
                        "WHERE competition_arena=? AND target_id=? AND artifact_digest=?",
                        (scope, tip["target_id"], artifact),
                    ).fetchone()
                    if row is None:
                        if nodes:
                            break
                        raise IntakeError(
                            "target lineage tips require a successful backfill"
                        )
                    nodes.append(
                        TargetLineageNode(
                            row["artifact_digest"],
                            row["parent_artifact_digest"],
                            row["winner_speedup"],
                            row["transition_event_id"],
                        )
                    )
                    artifact = row["parent_artifact_digest"]
                lineages[tip["target_id"]] = TargetLineage(
                    tuple(reversed(nodes))
                )
            return MappingProxyType(lineages)
        except (TypeError, ValueError, IntakeError) as exc:
            raise IntakeError(
                f"target lineage tips require a successful backfill: {exc}"
            ) from None


    def backfill_target_lineage_tips(self) -> Mapping[str, object]:
        """Seed the lineage ledger from the latest CROWN recorded per target.

        Stores that settled before the ledger existed carry crowned history but
        no tips, which would leave the fork guard inert.  Replaying the newest
        CROWN per target reconstructs the artifact each target's lineage rests
        on.  Idempotent: it recomputes the same rows from the same journal.
        """

        from cacheon.chain.intake import IntakeError
        from cacheon.settlement import TargetLineageNode

        with self._transaction():
            self._db.execute("DELETE FROM target_lineage_tips")
            self._db.execute("DELETE FROM target_lineage_nodes")
            self._db.execute(
                "DELETE FROM target_lineage_pretransition_reservations"
            )
            for row in self._db.execute(
                "SELECT se.target_id,se.reservation_id,se.event_id,se.arena_id,"
                "se.sequence FROM settlement_events se "
                "WHERE se.event_type='CROWN' "
                "ORDER BY se.sequence"
            ).fetchall():
                candidate_row = self._db.execute(
                    "SELECT candidate_json,candidate_digest FROM settlement_candidates "
                    "WHERE reservation_id=?",
                    (row["reservation_id"],),
                ).fetchone()
                if candidate_row is None:
                    raise IntakeError("crowned target has no settlement candidate")
                candidate = self._settlement_candidate(candidate_row)
                if candidate.candidate_manifest is None:
                    raise IntakeError("crowned candidate lacks its stack manifest")
                contribution = candidate.candidate_manifest.entries.get(row["target_id"])
                if contribution is None:
                    raise IntakeError("crowned candidate does not name its target")
                incumbent = candidate.incumbent_manifest.entries.get(row["target_id"])
                parent_artifact = (
                    "" if incumbent is None else incumbent.artifact_digest
                )
                node = TargetLineageNode(
                    contribution.artifact_digest,
                    parent_artifact,
                    candidate.speedup,
                    row["event_id"],
                )
                self._write_lineage(
                    candidate, node.artifact_digest, node.parent_artifact_digest,
                    node.transition_event_id, candidate.finalized_block,
                )
                # Historical journals did not retain the transition-time
                # reservation snapshot. A crown cannot precede its last
                # retained qualification product, so reservations from an
                # earlier block than that product are provably pre-transition.
                # Same-block arrivals remain excluded because retained_block
                # has no event index. Legacy zero timestamps fall back to the
                # winner's exact arrival order.
                retained = self._db.execute(
                    "SELECT MAX(retained_block) AS block "
                    "FROM settlement_qualifications WHERE reservation_id=?",
                    (candidate.reservation_digest,),
                ).fetchone()
                proof_block = 0 if retained is None else int(retained["block"])
                if proof_block > 0:
                    predicate = "block<? OR reservation_id=?"
                    parameters = (proof_block, candidate.reservation_digest)
                else:
                    predicate = "(block,event_index,event_subindex)<=(?,?,?)"
                    parameters = (candidate.finalized_block, candidate.event_index, candidate.event_subindex)
                self._db.execute(
                    "INSERT OR IGNORE INTO target_lineage_pretransition_reservations("
                    "transition_event_id,reservation_id) SELECT ?,reservation_id "
                    "FROM reservations WHERE competition_arena=? AND (" + predicate + ")",
                    (node.transition_event_id, self.get(candidate.reservation_digest).competition_arena,
                     *parameters),
                )
        return self.target_lineage_tips()


    def _active_qualification_rows(self, competition_arena: str | None = None, *, owner: str | None = None):
        scope = self._competition_arena if competition_arena is None else competition_arena
        return tuple(self._db.execute(
            "SELECT * FROM evaluation_leases WHERE stage='qualification' "
            "AND state='active' AND competition_arena=?"
            + (" AND owner=?" if owner is not None else ""),
            (scope, owner) if owner is not None else (scope,),
        ))

    def _select_evaluation_rows(
        self, stage: str, bound: int, *, owner: str | None = None,
        max_active: int | None = None,
    ) -> tuple[sqlite3.Row, ...]:
        """Shared, non-mutating ordered selector for preview and atomic claim."""

        if max_active is not None and (type(max_active) is not int or max_active <= 0):
            raise _error("evaluation capacity must be a positive integer")
        active = tuple(self._db.execute(
            "SELECT owner,stage FROM evaluation_leases WHERE state='active' "
            "AND competition_arena=?", (self._competition_arena,),
        ))
        if owner is not None and any(row["owner"] == owner for row in active):
            return ()
        capacity = 1 if stage == "qualification" and max_active is None else max_active
        if capacity is not None and sum(row["stage"] == stage for row in active) >= capacity:
            return ()
        if stage == "screen":
            predicate = "r.status IN ('published','reproduction_pending')"
            priority = "CASE r.status WHEN 'reproduction_pending' THEN 0 ELSE 1 END"
            segment_join = " "
            segment_predicate = " AND r.competition_arena=?"
            segment_parameters: tuple[object, ...] = (self._competition_arena,)
        else:
            baseline = self.qualification_queue_baseline()
            if baseline is None:
                # Genesis may be installed by the recoverable dispatcher only
                # after reopening a qualification lease claimed by legacy CPU
                # composition. Preserve that bootstrap path; once any durable
                # stack exists, an unbound queue head must wait for binding.
                if self._db.execute(
                    "SELECT 1 FROM evaluation_stacks WHERE competition_arena=? LIMIT 1",
                    (self._competition_arena,),
                ).fetchone() is not None:
                    return ()
                segment_join = " "
                segment_predicate = " AND r.competition_arena=?"
                segment_parameters = (self._competition_arena,)
            else:
                segment_join = (
                    " JOIN reservation_baseline_segments AS b USING(reservation_id) "
                )
                segment_predicate = (
                    " AND r.competition_arena=? AND b.arena_id=? AND b.generation=? AND b.stack_digest=? "
                    "AND b.tree_digest=? AND b.stack_json=? "
                    "AND b.transition_event_id=?"
                )
                segment_parameters = (
                    self._competition_arena, baseline.arena_digest,
                    baseline.generation,
                    baseline.manifest.digest,
                    baseline.tree_digest,
                    self._encoded_stack_manifest(baseline),
                    baseline.transition_event_id,
                )
            predicate = "r.status='promoted'"
            priority = "CASE r.screen_lane WHEN 'reproduction' THEN 0 ELSE 1 END"
        query = (
            "SELECT r.* FROM reservations AS r" + segment_join + "WHERE " + predicate +
            " AND NOT EXISTS (SELECT 1 FROM evaluation_lease_members AS em "
            "WHERE em.reservation_id=r.reservation_id AND em.active=1)" + segment_predicate
        )
        order = f" ORDER BY {priority},r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash"
        first = self._db.execute(query + order + " LIMIT 1", segment_parameters).fetchone()
        if first is None:
            return ()
        if stage == "screen" or first["screen_lane"] == "reproduction":
            return (first,)
        if first["retry_group_digest"]:
            selected = tuple(self._db.execute(
                query + " AND r.retry_group_digest=? ORDER BY r.retry_position",
                (*segment_parameters, first["retry_group_digest"]),
            ))
            total = self._db.execute(
                "SELECT COUNT(*) AS n FROM reservations WHERE status='promoted' "
                "AND retry_group_digest=? AND competition_arena=?",
                (first["retry_group_digest"], self._competition_arena),
            ).fetchone()["n"]
            if len(selected) != total:
                raise _error("qualification retry group is partially leased")
            if len(selected) > bound:
                raise _error("qualification retry group exceeds lease capacity")
            return selected
        return tuple(self._db.execute(
            query + " AND r.screen_lane='primary' AND r.retry_group_digest=''" + order + " LIMIT ?",
            (*segment_parameters, bound),
        ))

    def _expire_before_screen(self, reservation_id: str, reason: str) -> IntakeReservation:
        """Reject unmeasured work and release its payment or credit atomically."""
        with self._transaction():
            self._require_evaluation_mutation_authority(reservation_id)
            row = self.get(reservation_id)
            if row.status not in {"fetching", "published"} or row.screen_attempts:
                raise _error(
                    f"pre-screen disposal from {row.status!r} is forbidden"
                )
            self._db.execute(
                "UPDATE reservations SET status='expired',"
                "decision='NO_DECISION',reason=? WHERE reservation_id=?",
                (reason, reservation_id),
            )
            self._db.execute(
                "DELETE FROM eval_cost_payments WHERE reservation_id=?",
                (reservation_id,),
            )
            self._db.execute(
                "UPDATE eval_cost_credits SET reservation_id='',spent_block=0 "
                "WHERE reservation_id=?",
                (reservation_id,),
            )
        return self.get(reservation_id)

    def prepare_screen_queue(
        self, *, service_digest: str, closed_targets: tuple[str, ...] = (), limit: int | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Admit routed submissions before their first screen and replay prior losers.

        Duplicate FAIL replay follows duplicate_replay; a PASS and the
        reproduction lane are never replayed. Closed targets release payment
        through the existing no-decision transaction, before acquiring a lease.
        The first crown on this commissioned baseline closes new commitments.
        Earlier finalized commitments may drain even when fetched after the crown;
        work already screened is never subjected to a second admission cutoff.
        """
        from cacheon.chain.duplicate_replay import PriorVerdict, decide_replay
        from cacheon.stack_identity import require_sha256_hex

        require_sha256_hex(service_digest, field="arena service digest")
        cutoff = self._db.execute(
            "SELECT MIN(crowned_block) AS block FROM target_lineage_nodes "
            "WHERE competition_arena=? AND arena_id=?",
            (self._competition_arena, service_digest),
        ).fetchone()["block"]
        priors = tuple(
            PriorVerdict(
                reservation_id=row["reservation_id"],
                content_hash=row["content_hash"],
                arena_service_digest=row["arena_service_digest"],
                decision=row["decision"],
                reason=row["reason"],
            )
            for row in self._db.execute(
                "SELECT reservation_id,content_hash,arena_service_digest,decision,"
                "reason FROM reservations WHERE decision IN ('PASS','FAIL') "
                "AND arena_service_digest=?",
                (service_digest,),
            )
        )
        retired: list[tuple[str, str]] = []
        while True:
            before = len(retired)
            for row in self.screenable(limit=limit):
                if (row.status == "published" and not row.screen_attempts
                        and cutoff is not None and row.arrival.block > cutoff):
                    rejected = self._expire_before_screen(row.reservation_id, "baseline_closed_at_submission")
                    retired.append((row.reservation_id, rejected.reason))
                    continue
                if row.target_id in closed_targets and row.status == "published" and not row.screen_attempts:
                    parked = self.mark_target_unavailable(row.reservation_id, target_id=row.target_id)
                    retired.append((row.reservation_id, parked.reason))
                    continue
                decision = decide_replay(
                    content_hash=row.arrival.content_hash,
                    arena_service_digest=service_digest,
                    screen_lane=row.screen_lane,
                    priors=priors,
                )
                if decision.replay:
                    self.mark_failed(row.reservation_id, decision.reason)
                    retired.append((row.reservation_id, decision.reason))
            if len(retired) == before:
                return tuple(retired)
