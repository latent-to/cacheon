"""Transactional acceptance of a complete qualification without a second GPU run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.chain.evaluation_order import (
    finalize_reward_prefix, reward_grandfathered_runtimes, reward_winner_ids,
)
from cacheon.settlement import SettlementCandidate, SettlementQualification

if TYPE_CHECKING:
    from cacheon.chain.intake import FinalizedIntakeStore, IntakeReservation
    from cacheon.chain.weights import WeightProjection


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


def passed_reward_claims(store: FinalizedIntakeStore) -> tuple[object, ...]:
    """Return retained claims without changing their submission clocks or evidence."""
    return passed_reward_evidence(store)[0]


def passed_reward_evidence(store: FinalizedIntakeStore) -> tuple[tuple, tuple]:
    """Reopen each distinct earned contribution without rewriting its retained evidence."""
    from decimal import Decimal, ROUND_FLOOR
    from cacheon.chain.intake import IntakeError

    from cacheon.economics import StandingRewardClaim, WEIGHT_PPM

    claims = []
    contributions = []
    seen: set[tuple[str, str, str]] = set()
    finalize_reward_prefix(store)
    winners = reward_winner_ids(store._db)
    rows = store._db.execute(
        "SELECT sc.* FROM settlement_candidates sc "
        "JOIN reservations r USING(reservation_id) "
        "WHERE r.status='qualified' AND r.decision='PASS' "
        "AND sc.status!='duplicate_proposal' "
        "AND sc.reward_eligible=1 "
        "ORDER BY r.block,r.event_index,r.event_subindex,r.reservation_id"
    )
    for row in rows:
        candidate = store._settlement_candidate(row)
        if candidate.candidate_manifest is None:
            continue  # discovery PASSes use the bounded discovery pool
        contribution = candidate.candidate_manifest.entries[candidate.target_id]
        key = (candidate.arena_digest, candidate.target_id, contribution.digest)
        if key in seen:
            continue
        evidence = store.reopen_settlement_evidence(candidate)
        retained = row["settlement_evidence_digest"]
        if retained and retained != evidence.digest:
            raise IntakeError("PASS candidate differs from retained evidence")
        seen.add(key)
        if candidate.reservation_digest not in winners:
            continue
        claims.append(
            StandingRewardClaim(
                candidate.arena_digest,
                candidate.target_id,
                contribution.target_spec_digest,
                contribution.digest,
                candidate.hotkey,
                int(
                    (Decimal(candidate.speedup) * WEIGHT_PPM).to_integral_value(
                        rounding=ROUND_FLOOR
                    )
                ),
                candidate.finalized_block,
                evidence.digest,
            )
        )
        contributions.append(contribution)
    return tuple(claims), tuple(contributions)


def reward_decay_adjustments(store: FinalizedIntakeStore) -> list[dict]:
    """Reopen the append-only publication decay starts and recovery adjustments from the intake metadata."""
    from cacheon.chain.intake import IntakeError

    row = store._db.execute(
        "SELECT value FROM metadata WHERE key='reward_decay_adjustments'"
    ).fetchone()
    if row is None:
        return []
    try:
        records = json.loads(row["value"])
        if not isinstance(records, list) or any(
            type(item) is not dict
            or set(item) != {"claim_digest", "start_block", "reason"}
            or not isinstance(item["reason"], str) or not item["reason"].strip()
            for item in records
        ):
            raise ValueError("invalid adjustment records")
    except (ValueError, TypeError) as exc:
        raise IntakeError(f"reward decay adjustments are corrupt: {exc}") from None
    return records


def record_reward_decay_start(
    store: FinalizedIntakeStore, *, claim_digest: str,
    start_block: int | None, reason: str,
) -> None:
    """Record a publication start or operator recovery without rewriting PASS evidence.

    None holds decay until the first confirmed publication. A subsequent call
    fixes the start to the finalized block where its vector was confirmed. Once set, the
    start cannot be moved by retries or restarts.
    """
    from cacheon.chain.intake import IntakeError

    claims = {row.digest: row for row in passed_reward_claims(store)}
    claim = claims.get(claim_digest)
    if claim is None:
        raise IntakeError("reward decay adjustment has no earning PASS")
    if start_block is not None and (
        type(start_block) is not int or start_block < claim.crowned_block
    ):
        raise IntakeError("reward decay start predates the submission")
    if not isinstance(reason, str) or not reason.strip():
        raise IntakeError("reward decay adjustment requires a reason")
    with store._transaction():
        records = reward_decay_adjustments(store)
        prior = [row for row in records if row["claim_digest"] == claim_digest]
        if prior:
            if prior[-1]["start_block"] == start_block:
                return
            if prior[-1]["start_block"] is not None:
                raise IntakeError("reward decay start is already fixed")
        records.append({"claim_digest": claim_digest, "start_block": start_block, "reason": reason})
        store._db.execute(
            "INSERT INTO metadata(key,value) VALUES('reward_decay_adjustments',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(records, sort_keys=True, separators=(",", ":")),),
        )


def _hold_unpublished_claims(store, claims):
    # 'reward_decay_legacy_claims' was written once, when this producer took over the GLM store on
    # 2026-09-20: those PASSes keep their crown-block clock. A new store never has the key.
    row = store._db.execute(
        "SELECT value FROM metadata WHERE key='reward_decay_legacy_claims'"
    ).fetchone()
    legacy = set(json.loads(row["value"])) if row is not None else set()
    tracked = {row["claim_digest"] for row in reward_decay_adjustments(store)}
    for claim in claims:
        if claim.digest not in legacy | tracked:
            record_reward_decay_start(
                store, claim_digest=claim.digest, start_block=None,
                reason="Awaiting first confirmed on-chain weights containing this PASS",
            )


def confirm_reward_decay(store: FinalizedIntakeStore, projection, record) -> None:
    """Start only the pending claims included in a finalized publication receipt."""
    from cacheon.chain.intake import IntakeError

    if record.projection_digest != projection.digest:
        raise IntakeError("decay confirmation differs from its retained projection")
    # The signer confirms at inclusion with confirmed_last_update=0: until 2026-09-21 no clock had
    # started since block 9097653 and two unclocked PASSes held 52% of the vector.
    published = record.confirmed_block if record.reason == "block_inclusion" else record.confirmed_last_update
    if record.status != "confirmed" or published < projection.effective_block:
        return  # an older identical vector does not publish a newly accepted PASS
    starts = {row["claim_digest"]: row["start_block"] for row in reward_decay_adjustments(store)}
    pending = {digest for digest, start in starts.items() if start is None}
    if not pending:
        return
    recipients = dict(projection.weights_ppm)
    earning_evidence = (projection.evidence_digests if projection.rewarded_evidence_digests is None
                        else projection.rewarded_evidence_digests)
    for claim in passed_reward_claims(store):
        if (claim.digest in pending and claim.retained_evidence_digest in earning_evidence
                and recipients.get(claim.hotkey, 0) > 0):
            record_reward_decay_start(
                store, claim_digest=claim.digest, start_block=record.confirmed_block,
                reason="First confirmed publication: " + projection.digest,
            )


def reconcile_follower_reward_decay(store: FinalizedIntakeStore, journal_path, *, validator_hotkey: str) -> None:
    """Consume the existing signer's confirmed journal, including across producer restarts."""
    import sqlite3
    from cacheon.chain.intake import IntakeError
    from cacheon.chain.weight_share import CurrentWeightOffer
    from cacheon.chain.weights import WeightPublicationRecord

    row = store._db.execute(
        "SELECT value FROM metadata WHERE key='reward_decay_confirmation_cursor'"
    ).fetchone()
    cursor = json.loads(row["value"]) if row else {"path": str(journal_path), "sequence": 0}
    if cursor["path"] != str(journal_path):
        raise IntakeError("reward confirmation journal changed")
    with sqlite3.connect(Path(journal_path).as_uri() + '?mode=ro', uri=True) as journal:
        journal.row_factory = sqlite3.Row
        maximum = journal.execute(
            "SELECT COALESCE(MAX(sequence),0) FROM followed_weight_publications"
        ).fetchone()[0]
        if maximum < cursor["sequence"]:
            raise IntakeError("reward confirmation journal regressed")
        rows = journal.execute(
            "SELECT * FROM followed_weight_publications WHERE sequence>? "
            "AND status='confirmed' ORDER BY sequence", (cursor["sequence"],),
        ).fetchall()
    with store._transaction():
        starts = {row["claim_digest"]: row["start_block"] for row in reward_decay_adjustments(store)}
        pending = {digest for digest, start in starts.items() if start is None}
        evidence = {row.retained_evidence_digest for row in passed_reward_claims(store)
                    if row.digest in pending} if pending else set()
        for row in rows:
            offer = CurrentWeightOffer.from_dict(json.loads(row["offer_json"]))
            record = WeightPublicationRecord.from_dict(json.loads(row["record_json"]))
            projection = offer.projection
            if (offer.digest != row["offer_digest"] or record.digest != row["record_digest"]
                    or record.status != row["status"]
                    or projection.digest != row["projection_digest"]
                    or projection.validator_hotkey != validator_hotkey
                    or projection.chain_scope_digest != store.scope.digest
                    or projection.netuid != store.scope.netuid):
                raise IntakeError("reward confirmation differs from its signer or chain authority")
            if evidence.intersection(projection.evidence_digests):
                confirm_reward_decay(store, projection, record)
        store._db.execute(
            "INSERT INTO metadata(key,value) VALUES('reward_decay_confirmation_cursor',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps({"path": str(journal_path), "sequence": maximum}, sort_keys=True),),
        )


def _reward_projection_inputs(store, *, include_uncrowned: bool = False) -> dict:
    """Reopen one store's reward evidence and publication clocks for its producer.

    Retained claims, stack checks and publication clocks stay together so a
    combined producer can reuse the existing evidence authority.
    """
    from cacheon.chain.intake import IntakeError
    from cacheon.economics import ArenaRewardAuthority

    standing, discovery = store.active_reward_claims()
    earning, contributions = passed_reward_evidence(store)
    _hold_unpublished_claims(store, earning)
    adjustments = reward_decay_adjustments(store)
    live = {claim.digest for claim in earning}
    starts = {row["claim_digest"]: row["start_block"] for row in adjustments if row["claim_digest"] in live}
    by_arena: dict[str, list[object]] = {}
    for claim in standing:
        by_arena.setdefault(claim.arena_digest, []).append(claim)
    states = store.evaluation_stacks()
    state_ids = {row.arena_digest for row in states}
    active_states = tuple(row for row in states if row.generation > 0)
    active_ids = {row.arena_digest for row in active_states}
    if set(by_arena) - state_ids:
        raise IntakeError("active reward claim belongs to an absent evaluation arena")
    if set(by_arena) - active_ids:
        raise IntakeError("active reward claim belongs to an uncrowned evaluation arena")
    for claim in standing:
        store._reopen_claim_evidence(claim.retained_evidence_digest, "crowned")
    for claim in discovery:
        store._reopen_claim_evidence(
            claim.retained_evidence_digest, "discovery_bounty"
        )
    authorities = []
    for state in (states if include_uncrowned else active_states):
        authorities.append(
            ArenaRewardAuthority(
                state.manifest,
                state.generation,
                tuple(by_arena.get(state.arena_digest, ())),
            )
        )
    return {
        "arenas": tuple(authorities), "earning_claims": earning,
        "discovery_claims": discovery, "earned_contributions": contributions,
        "decay_start_blocks": starts, "adjustments": adjustments,
        "standing_claims": standing, "states": states,
        "grandfathered_runtimes": reward_grandfathered_runtimes(store._db),
    }


def build_weight_projection(
    store,
    *,
    policy,
    context,
    netuid: int,
) -> WeightProjection:
    """Pool all retained earning claims under each crown's sealed catalog."""

    from cacheon.chain.intake import IntakeError
    from cacheon.stack_identity import canonical_digest
    from cacheon.chain.weights import WeightProjection
    from cacheon.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        project_global_rewards,
    )

    if (
        type(policy) is not EmissionsPolicyManifest
        or type(context) is not GlobalRewardProjectionContext
        or type(netuid) is not int
        or netuid < 0
    ):
        raise IntakeError("weight projection authority is malformed")
    from cacheon.chain.arena_weight_projection import require_legacy_projection

    require_legacy_projection(store, context.current_block)
    inputs = _reward_projection_inputs(store)
    standing = inputs.pop("standing_claims")
    adjustments = inputs.pop("adjustments")
    grandfathered = inputs.pop("grandfathered_runtimes")
    active_states = tuple(row for row in inputs.pop("states") if row.generation > 0)
    earning, discovery = inputs["earning_claims"], inputs["discovery_claims"]
    projection = project_global_rewards(
        policy, context, **inputs,
    )
    store._bind_emissions_policy(policy)
    evidence = tuple(
        sorted(
            {
                claim.retained_evidence_digest
                for claim in (*standing, *earning, *discovery)
            }
        )
    )
    policy_digest = policy.digest
    if grandfathered:
        boundary_digest = canonical_digest("cacheon.operator.reward-grandfathering.v1", grandfathered)
        evidence = tuple(sorted({*evidence, boundary_digest}))
        policy_digest = canonical_digest("cacheon.operator.reward-record-policy.v1", {
            "base_policy": policy_digest, "grandfathered_runtimes": grandfathered,
        })
    if adjustments:
        adjustment_digest = canonical_digest("cacheon.operator.reward-decay.v1", adjustments)
        evidence = tuple(sorted({*evidence, adjustment_digest}))
        policy_digest = canonical_digest("cacheon.operator.reward-decay-policy.v1", {
            "base_policy": policy_digest, "adjustment_digest": adjustment_digest,
        })
    return WeightProjection(
        context.chain_scope_digest,
        netuid,
        context.validator_hotkey,
        policy_digest,
        store.settlement_state_digest(),
        projection.digest,
        context.metagraph_digest,
        projection.arena_authority_digests,
        max((row.generation for row in active_states), default=0),
        context.current_block,
        len(standing),
        evidence,
        tuple(
            (row.hotkey, row.weight_ppm) for row in projection.weights
        ),
    )
