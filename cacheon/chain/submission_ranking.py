"""Completion-time competition using retained qualification speeds, without GPU reruns."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal

from cacheon.eval.evidence_store import reopen_evidence
from cacheon.stack_identity import canonical_digest


def create_schema(store) -> None:
    """Persist the competitive decision separately from the A-versus-B PASS."""

    store._db.execute(
        "CREATE TABLE IF NOT EXISTS submission_rankings ("
        "sequence INTEGER PRIMARY KEY AUTOINCREMENT, reservation_id TEXT NOT NULL UNIQUE "
        "REFERENCES reservations(reservation_id), arena_id TEXT NOT NULL,target_id TEXT NOT NULL,"
        "baseline_ref TEXT NOT NULL, comparison_context TEXT NOT NULL,"
        "candidate_rate TEXT NOT NULL, required_ratio TEXT NOT NULL,"
        "competitor_id TEXT NOT NULL, stale INTEGER NOT NULL, won INTEGER NOT NULL,"
        "completed_block INTEGER NOT NULL)"
    )


def comparison_context(calibration, witness):
    """Compare equal model, hardware and scoring contracts across baseline advances."""

    if calibration.digest != witness.calibration_context_digest:
        raise ValueError("measurement calibration context differs")
    context = calibration.to_dict()
    # These identities include the commissioned baseline, which is allowed to advance.
    context.pop("arena_digest")
    context.pop("reference_manifest_digest")
    policy = witness.resident_policy.to_dict()
    policy.pop("calibration_digest")
    policy.pop("calibration_context_digest")
    return canonical_digest("cacheon.chain.speed-comparison.v2", {"context": context, "policy": policy})


def measured_speed(qualification, attempt_ref, root):
    """Use the report's exact policy and unrounded scored rate from retained bytes."""

    from cacheon.chain.intake import IntakeError
    from cacheon.eval.qualification_runner import CohortQualificationAttempt, ResidentSpeedWitness
    from cacheon.eval.speed_verdict import speed_grade, SpeedStageDecision

    try:
        payload = json.loads(reopen_evidence(root, attempt_ref))
        if attempt_ref.schema == "cacheon.qualification.operator-quality-correction.v1":
            report = payload["report"]
            if (attempt_ref.domain != "qualification.operator-quality-correction"
                    or payload["report_digest"] != qualification.qualification_report_digest
                    or canonical_digest(f"{attempt_ref.schema}.report", report) != payload["report_digest"]
                    or report["decision"] != "PASS" or report["quality_decision"] != "PASS"
                    or report["selected_delta_digest"] != qualification.selected_delta_digest
                    or payload["reservation_id"] != qualification.reservation_digest):
                raise IntakeError("operator correction differs from accepted qualification")
            witness = ResidentSpeedWitness.from_dict(payload["speed_witness"])
            speedup = report["speedup"]
            if witness.evidence_digest != report["speed_evidence_digest"]:
                raise IntakeError("operator correction changed retained speed evidence")
        else:
            attempt = CohortQualificationAttempt.from_dict(payload)
            report = next(row for row in attempt.reports if row.digest == qualification.qualification_report_digest)
            witness, speedup = report.speed_witness, report.speedup
        verdict, decision = speed_grade(
            witness.resident_policy, [witness.rates[0], witness.rates[2]],
            [witness.rates[1]], concluding=True,
        )
        if decision is not SpeedStageDecision.PASS or speedup != qualification.speedup:
            raise IntakeError("competitive qualification does not retain its PASS")
        rate = Decimal(str(witness.resident_policy.scored_tokens_per_second(witness.rates[1])))
        required = Decimal(str(verdict.required))
        context = qualification.comparison_context_digest or canonical_digest("cacheon.chain.speed-comparison.v1", {
            "arena": qualification.arena_digest,
            "runtime": qualification.incumbent_manifest.runtime_digest,
            "base_engine": qualification.incumbent_manifest.base_engine_digest,
            "workload": witness.workload_digest,
            "policy": witness.resident_policy.digest,
        })
    except (ValueError, KeyError, StopIteration, TypeError) as exc:
        raise IntakeError(f"competitive speed evidence cannot reopen: {exc}") from exc
    if not rate.is_finite() or rate <= 0 or not required.is_finite() or required <= 1:
        raise IntakeError("competitive speed evidence is invalid")
    return rate, required, context


def retain_ranking(store, qualification, attempt_ref, root, completed_block: int) -> bool:
    """Compare against the strongest same-slot winner inside qualification's transaction."""

    from cacheon.chain.intake import IntakeError

    segment = store.reservation_baseline_segment(qualification.reservation_digest)
    if segment is None or (
        segment.manifest.digest != qualification.incumbent_stack_digest
        or segment.tree_digest != qualification.incumbent_tree_digest
    ):
        raise IntakeError("qualification did not evaluate the declared baseline")
    rate, required, context = measured_speed(qualification, attempt_ref, root)
    return _rank(store, qualification, attempt_ref, root, completed_block, rate, required, context)


def _rank(store, qualification, attempt_ref, root, completed_block, rate, required, context,
          *, historical_crown=False):
    from cacheon.chain.intake import IntakeError

    prior = tuple(store._db.execute(
        "SELECT s.* FROM submission_rankings s JOIN reservations r USING(reservation_id) "
        "WHERE (s.comparison_context=? OR s.arena_id=?) "
        "AND s.target_id=? AND s.won=1 AND r.status='qualified'",
        (context, qualification.arena_digest, qualification.target_id),
    ))
    if any(row["comparison_context"] != context for row in prior):
        legacy_context = measured_speed(replace(qualification, comparison_context_digest=""), attempt_ref, root)[2]
        if any(row["comparison_context"] not in (context, legacy_context) for row in prior):
            raise IntakeError("same-slot winner has incompatible measurement context")
    baseline_entry = qualification.incumbent_manifest.entries.get(qualification.target_id)
    baseline_artifact = "" if baseline_entry is None else baseline_entry.artifact_digest
    incorporated_sequence = 0
    for row in prior:
        retained = store._db.execute(
            "SELECT * FROM settlement_candidates WHERE reservation_id=?", (row["reservation_id"],)
        ).fetchone()
        if retained is None:
            raise IntakeError("ranked winner lost its qualification")
        candidate = store._settlement_candidate(retained)
        if candidate.candidate_manifest.entries[qualification.target_id].artifact_digest == baseline_artifact:
            incorporated_sequence = max(incorporated_sequence, row["sequence"])
    # Baseline incorporation subsumes earlier winners. Its direct comparison is
    # authoritative even if absolute machine rates shifted between evaluations.
    unfinalized = tuple(row for row in prior if row["sequence"] > incorporated_sequence)
    competitor = max(unfinalized, key=lambda row: Decimal(row["candidate_rate"]), default=None)
    stale = competitor is not None
    won = historical_crown or competitor is None or rate > Decimal(competitor["candidate_rate"]) * required
    store._db.execute(
        "INSERT INTO submission_rankings(reservation_id,arena_id,target_id,baseline_ref,"
        "comparison_context,candidate_rate,required_ratio,competitor_id,stale,won,completed_block) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (qualification.reservation_digest, qualification.arena_digest, qualification.target_id,
         qualification.incumbent_manifest.digest, context, str(rate), str(required),
         "" if competitor is None else competitor["reservation_id"], int(stale), int(won), completed_block),
    )
    if won:
        # A lease made before this completion must not crown the displaced tip.
        store._db.execute(
            "UPDATE settlement_candidates SET status='held',reason='stronger_unfinalized_winner' "
            "WHERE status IN ('pending','leased') AND reservation_id<>? AND reservation_id IN "
            "(SELECT reservation_id FROM submission_rankings WHERE (comparison_context=? OR arena_id=?) AND target_id=?)",
            (qualification.reservation_digest, context, qualification.arena_digest, qualification.target_id),
        )
    return won


def current_winners(store):
    """Select the strongest accepted candidate in each arena/slot, including unfinalized ones."""

    winners = {}
    for row in store._db.execute("SELECT s.* FROM submission_rankings s JOIN reservations r USING(reservation_id) "
                                 "WHERE s.won=1 AND r.status='qualified' ORDER BY s.sequence"):
        key = row["arena_id"], row["target_id"]
        winners[key] = row
    return tuple(winners.values())


def backfill_rankings(store) -> None:
    """Retain scoring for pre-upgrade accepted PASSes from their archived measurements.

    Historical crowns retain their credit. Other retained PASSes are compared
    in completion order, so a historical A/B PASS that lost to C earns no claim.
    """

    from cacheon.eval.evidence_store import EvidenceArtifactRef
    from cacheon.settlement import SettlementQualification

    rows = tuple(store._db.execute(
        "SELECT sc.* FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
        "WHERE r.status='qualified' AND r.decision='PASS' AND sc.status!='duplicate_proposal' "
        "AND NOT EXISTS (SELECT 1 FROM submission_rankings s WHERE s.reservation_id=sc.reservation_id) "
        "ORDER BY (SELECT MAX(q.retained_block) FROM settlement_qualifications q "
        "WHERE q.reservation_id=r.reservation_id),r.block,r.event_index,r.event_subindex,r.reservation_id"
    ))
    for row in rows:
        candidate = store._settlement_candidate(row)
        if candidate.candidate_manifest is None:
            continue
        store.reopen_settlement_evidence(candidate)
        retained = tuple(store._db.execute(
            "SELECT * FROM settlement_qualifications WHERE reservation_id=? ORDER BY reproduction_index",
            (candidate.reservation_digest,),
        ))
        scores = [measured_speed(
            SettlementQualification.from_dict(json.loads(item["qualification_json"])),
            EvidenceArtifactRef.from_dict(json.loads(item["attempt_ref_json"])), item["evidence_root"],
        ) for item in retained]
        rate, _, context = min(scores, key=lambda score: score[0])
        item = retained[0]
        qualification = SettlementQualification.from_dict(json.loads(item["qualification_json"]))
        won = _rank(store, qualification,
            EvidenceArtifactRef.from_dict(json.loads(item["attempt_ref_json"])), item["evidence_root"],
            max(item["retained_block"] for item in retained), rate, max(score[1] for score in scores), context,
            historical_crown=row["status"] == "crowned")
        if not won:
            store._db.execute(
                "UPDATE settlement_candidates SET status='held',reason='stronger_unfinalized_winner' "
                "WHERE reservation_id=?", (candidate.reservation_digest,),
            )


def lineage_admits_candidate(candidate, lineages, pretransition_reservations) -> bool:
    """Require lineage membership; competitive ranking is retained by intake."""

    from cacheon.settlement import SettlementError

    lineage = lineages.get(candidate.target_id)
    if lineage is None:
        return True
    incumbent = candidate.incumbent_manifest.entries.get(candidate.target_id)
    incumbent_artifact = "" if incumbent is None else incumbent.artifact_digest
    if incumbent_artifact == lineage.artifact_digest:
        return True
    try:
        lineage.threshold_from(incumbent_artifact)
    except SettlementError:
        return False
    # The qualification store already compared retained absolute rates against
    # the strongest same-slot winner at completion. Multiplying ratios here
    # would apply a second, incompatible competitive rule.
    return True
