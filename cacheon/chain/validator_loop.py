"""Finalized chain intake, immutable publication, qualification, and settlement.

This production loop deliberately stops before weight signing.  It reserves
the complete finalized event order before network transport, publishes submitted bytes
into a separate immutable worker tree, optionally invokes the current batch causal
qualification authority, and transactionally adopts its retained PASS projection. The
old shell/CPU fake-score evaluator and JSON Ledger settlement do not exist on this path;
wallet access belongs only to the separate control-plane signer.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cacheon import chain
from cacheon.arena_service import (
    AdmissionDecision,
    ArenaCandidateBinding,
    ArenaQualificationWork,
    ArenaService,
    ArenaServiceRegistry,
)
from cacheon.chain.fetch import FetchError, FetchTransientError, fetch_bundle
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeReservation,
    IntakeScope,
)
from cacheon.chain.eval_cost import (
    EvalCostFetchError,
    EvalCostPolicy,
    EvalCostRequest,
    verify_eval_cost_payment,
)
from cacheon.chain.eval_cost_payment import (
    read_eval_cost_payment,
    read_subnet_owner_coldkey,
)
from cacheon.chain.payload import decode_payload
from cacheon.chain.publication import (
    WorkerBundlePublication,
    WorkerBundlePublicationError,
    WorkerBundleSourceError,
    publish_worker_bundle,
    reopen_worker_bundle,
)
from cacheon.chain.reference_copy_policy import reconcile_reference_copies
from cacheon.copy_fingerprint import fingerprint_submitted_delta
from cacheon.target_servability import unservable_target_reason
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationReservation,
    run_qualification_intake,
)


logger = logging.getLogger("cacheon.chain.validator")
DEFAULT_INTERVAL_S = 60.0
_DISABLED_EVAL_COST_POLICY = EvalCostPolicy(amount_rao=0)


class IntakeControllerError(RuntimeError):
    """Validator-owned intake/qualification authority is inconsistent."""


# Compatibility names for code constructing trusted providers.  The live loop
# accepts only a closed ArenaServiceRegistry, never an arbitrary planner callback.
QualificationWork = ArenaQualificationWork


@dataclass
class PassResult:
    finalized_block: int
    finalized_block_hash: str
    seen: int = 0
    reserved: list[str] = field(default_factory=list)
    published: dict[str, str] = field(default_factory=dict)
    copies: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    held: list[str] = field(default_factory=list)
    settlements: dict[str, str] = field(default_factory=dict)
    screens: dict[str, str] = field(default_factory=dict)


def _finalized_arrivals(
    snapshot,
    *,
    netuid: int,
    eval_cost_policy: EvalCostPolicy,
    payment_lookup=None,
    owner_lookup=None,
) -> tuple[FinalizedArrival, ...]:
    rows: list[FinalizedArrival] = []
    for reveal in snapshot.reveals:
        payload_digest = hashlib.sha256(reveal.data.encode("utf-8")).hexdigest()
        ref = decode_payload(reveal.hotkey, reveal.block, reveal.data)
        if ref is None:
            rows.append(
                FinalizedArrival(
                    reveal.hotkey,
                    "",
                    "",
                    reveal.block,
                    reveal.block_hash.lower(),
                    reveal.event_index,
                    0,
                    payload_digest,
                    "invalid_payload",
                )
            )
            continue
        invalid_reason = ""
        if eval_cost_policy.amount_rao > 0:
            invalid_reason = _eval_cost_invalid_reason(
                ref,
                netuid=netuid,
                policy=eval_cost_policy,
                payment_lookup=payment_lookup,
                owner_lookup=owner_lookup,
            )
        rows.append(
            FinalizedArrival(
                ref.hotkey,
                ref.content_hash,
                ref.url,
                reveal.block,
                reveal.block_hash.lower(),
                reveal.event_index,
                0,
                payload_digest,
                invalid_reason,
                ref.payment_block,
                ref.payment_extrinsic_index,
            )
        )
    return tuple(rows)


def _eval_cost_invalid_reason(
    ref,
    *,
    netuid: int,
    policy: EvalCostPolicy,
    payment_lookup,
    owner_lookup,
) -> str:
    if ref.payment_block <= 0:
        return "missing_eval_cost_payment"
    lookup = payment_lookup or (lambda block, index: None)
    try:
        proof = lookup(ref.payment_block, ref.payment_extrinsic_index)
    except EvalCostFetchError:
        raise
    except Exception as exc:
        raise EvalCostFetchError(
            f"cannot read eval-cost payment at {ref.payment_block}/{ref.payment_extrinsic_index}: {exc}"
        ) from exc
    if proof is None:
        return "eval_cost_payment_invalid"
    if owner_lookup is None:
        raise EvalCostFetchError("eval-cost owner lookup is unavailable")
    try:
        owner = owner_lookup(ref.payment_block)
    except EvalCostFetchError:
        raise
    except Exception as exc:
        raise EvalCostFetchError(
            f"cannot read subnet owner at payment block {ref.payment_block}: {exc}"
        ) from exc
    if not isinstance(owner, str) or not owner:
        raise EvalCostFetchError("subnet owner coldkey is unavailable")
    request = EvalCostRequest(
        netuid=netuid, hotkey=ref.hotkey, content_hash=ref.content_hash
    )
    return verify_eval_cost_payment(
        request=request,
        policy=EvalCostPolicy(
            amount_rao=policy.amount_rao,
            destination=owner,
            payment_window_blocks=policy.payment_window_blocks,
            quote_ttl_blocks=policy.quote_ttl_blocks,
        ),
        proof=proof,
        reveal_block=ref.block,
    )


def _fingerprint_private_bundle(root: Path):
    """Choose a lane by exact parser success; never by miner-provided mode alone."""

    component_error: Exception | None = None
    try:
        return fingerprint_submitted_delta(root)
    except (OSError, TypeError, ValueError) as exc:
        component_error = exc
    try:
        return fingerprint_submitted_delta(root, discovery=True)
    except (OSError, TypeError, ValueError) as discovery_error:
        raise ValueError(
            "submission is neither a registered component nor a closed discovery "
            f"proposal: component={component_error}; discovery={discovery_error}"
        ) from None


def _qualification_reservations(
    reservations: tuple[IntakeReservation, ...],
    publications: tuple[WorkerBundlePublication, ...],
) -> tuple[QualificationReservation, ...]:
    if len(reservations) != len(publications):
        raise IntakeControllerError("qualification publication coverage differs")
    rows: list[QualificationReservation] = []
    for index, (reservation, publication) in enumerate(
        zip(reservations, publications, strict=True)
    ):
        fingerprint = reservation.delta_fingerprint
        if (
            fingerprint is None
            or reservation.publication_digest != publication.digest
            or reservation.arrival.content_hash != publication.content_hash
        ):
            raise IntakeControllerError("qualification intake provenance differs")
        rows.append(
            QualificationReservation(
                reservation.reservation_id,
                publication.digest,
                fingerprint.target_id,
                fingerprint.selected_delta_digest,
                index,
                reservation.arrival.hotkey,
                reservation.arrival.block,
                reservation.arrival.event_index,
                reservation.arrival.event_subindex,
                reservation.target_members,
            )
        )
    return tuple(rows)


def _validate_work(
    work: ArenaQualificationWork,
    expected: tuple[QualificationReservation, ...],
) -> None:
    if type(work) is not ArenaQualificationWork:
        raise IntakeControllerError("qualification planner returned an untyped work item")
    if work.factory.manifest.reservations != expected:
        raise IntakeControllerError("qualification factory changed finalized cohort order")


def _apply_qualification(
    store: FinalizedIntakeStore,
    reservations: tuple[IntakeReservation, ...],
    publications: tuple[WorkerBundlePublication, ...],
    service: ArenaService,
    *,
    minimum_finalized_block: int,
    finalized_block_provider: Callable[[], int],
) -> QualificationIntakeBatch:
    authority_rows = _qualification_reservations(reservations, publications)
    candidates = tuple(
        ArenaCandidateBinding(authority, publication, reservation.screen_attempts)
        for reservation, publication, authority in zip(
            reservations, publications, authority_rows, strict=True
        )
    )
    receipts = tuple(
        store.latest_promoted_screen(row.reservation_id) for row in reservations
    )
    work = service.plan_qualification(candidates, receipts, state=store)
    _validate_work(work, authority_rows)
    prepared = None
    if type(work.factory.manifest) is QualificationAuthorityManifest:
        prepared = work.factory.build()
        arms = tuple(row.arm for row in prepared.prepared.candidates)
        if (
            not arms
            or len({row.baseline_before for row in arms}) != 1
            or any(row.incumbent != arms[0].incumbent for row in arms)
        ):
            raise IntakeControllerError("qualification planner has no single incumbent")
        store.initialize_evaluation_stack(
            arms[0].incumbent,
            tree_digest=arms[0].baseline_before.tree_digest,
        )
    authority_digest = work.factory.manifest.digest
    authority_manifest = work.factory.manifest.to_dict()
    for row in reservations:
        store.mark_qualifying(
            row.reservation_id, authority_digest, authority_manifest
        )
    batch = run_qualification_intake(
        work.factory,
        executor=work.executor,
        resident_baseline_executor=work.resident_baseline_executor,
        entropy_provider=work.entropy_provider,
        hidden_judge=work.hidden_judge,
        deadline=float(work.deadline),
    )
    if (
        type(batch) is not QualificationIntakeBatch
        or batch.authority_manifest_digest != authority_digest
        or tuple(row.reservation_digest for row in batch.outcomes)
        != tuple(row.reservation_id for row in reservations)
        or tuple(row.selected_delta_digest for row in batch.outcomes)
        != tuple(row.selected_delta_digest for row in authority_rows)
    ):
        raise IntakeControllerError("qualification outcomes changed cohort authority")
    # Qualification can occupy the GPU for hours.  Timestamp retained PASS
    # evidence from a finalized head read after the work completes, not from the
    # pass-start reveal snapshot, or the reproduction SLA can be mostly (or
    # entirely) consumed before the first PASS is durable.
    retained_block = finalized_block_provider()
    if (
        type(retained_block) is not int
        or retained_block < minimum_finalized_block
    ):
        raise IntakeControllerError("finalized qualification clock regressed")
    store.apply_qualification_batch(
        batch,
        current_finalized_block=retained_block,
        evidence_root=None if prepared is None else prepared.evidence_root,
    )
    return batch


def _screen_pending(
    store: FinalizedIntakeStore,
    service: ArenaService,
    *,
    current_block: int,
) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for row in store.screenable(limit=store.policy.max_cohort):
        admission = service.admit(
            store.arena_queue_snapshot(current_block=current_block)
        )
        if admission is AdmissionDecision.QUEUE:
            break
        if admission is AdmissionDecision.HOLD:
            store.mark_held(row.reservation_id, "arena_screen_capacity_hold")
            decisions[row.reservation_id] = "hold"
            continue
        publication = reopen_worker_bundle(
            row.publication_root,
            row.arrival.content_hash,
            expected_receipt_digest=row.publication_digest,
        )
        active = store.begin_screen(
            row.reservation_id, service_digest=service.identity
        )
        authority = _qualification_reservations((active,), (publication,))[0]
        candidate = ArenaCandidateBinding(
            authority, publication, active.screen_attempts
        )
        receipt = service.screen(candidate)
        store.apply_screen_receipt(
            active.reservation_id,
            candidate_digest=candidate.digest,
            receipt=receipt,
        )
        decisions[active.reservation_id] = receipt.decision.value
    return decisions


def _settle_pending(
    store: FinalizedIntakeStore,
    *,
    current_block: int,
    finalized_block_provider: Callable[[], int | tuple[int, str]],
) -> dict[str, str]:
    """Settle every causally ready retained PASS without chain or wallet access."""

    from cacheon.settlement import plan_settlement

    def finalized_point() -> tuple[int, str | None]:
        value = finalized_block_provider()
        if type(value) is int:
            if value < 0:
                raise IntakeControllerError("finalized settlement clock is malformed")
            return value, None
        if (
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not int
            or value[0] < 0
            or not isinstance(value[1], str)
            or len(value[1]) != 66
            or not value[1].startswith("0x")
            or any(char not in "0123456789abcdef" for char in value[1][2:])
        ):
            raise IntakeControllerError("finalized settlement point is malformed")
        return value[0], value[1]

    committed: dict[str, str] = {}
    while store.has_pending_settlement():
        observed = finalized_point()
        lease_block = observed[0]
        if lease_block < current_block:
            raise IntakeControllerError("finalized settlement clock regressed")
        current_block = lease_block
        lease = store.lease_settlement_cohort(current_block=current_block)
        if lease is None:
            return committed
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(candidate)
            for candidate in lease.candidates
        )
        refreshed_block = finalized_point()[0]
        if refreshed_block < current_block:
            raise IntakeControllerError("finalized settlement clock regressed")
        store.commit_settlement(lease, plan, evidence, current_block=refreshed_block)
        current_block = refreshed_block
        committed[lease.lease_id] = plan.digest
    return committed


def run_pass(
    subtensor,
    netuid: int,
    *,
    intake_db: str | Path,
    private_root: str | Path,
    publication_root: str | Path,
    policy: IntakePolicy = IntakePolicy(),
    eval_cost_policy: EvalCostPolicy = _DISABLED_EVAL_COST_POLICY,
    arena_registry: ArenaServiceRegistry | None = None,
    arena_id: str | None = None,
    intake_only: bool = False,
    retained_only: bool = False,
) -> PassResult:
    """Run one non-emitting intake/qualification pass.

    ``retained_only`` evaluates the already-durable queue at the current
    finalized head without rereading or advancing reveal history.
    """

    if type(eval_cost_policy) is not EvalCostPolicy:
        raise IntakeControllerError("eval-cost policy is not typed")
    if type(intake_only) is not bool or type(retained_only) is not bool:
        raise IntakeControllerError("pass mode flags must be exact booleans")
    if intake_only and retained_only:
        raise IntakeControllerError("intake-only and retained-only modes conflict")
    if intake_only:
        if arena_registry is not None or arena_id is not None:
            raise IntakeControllerError("intake-only mode cannot receive arena authority")
        service = None
    else:
        if type(arena_registry) is not ArenaServiceRegistry or not arena_id:
            raise IntakeControllerError(
                "live validation requires an injected registered arena service"
            )
        service = arena_registry.require(arena_id)

    scope = IntakeScope(str(subtensor.get_block_hash(0)).lower(), netuid)
    with FinalizedIntakeStore(intake_db, policy, scope=scope) as store:
        cursor = store.finalized_cursor()
        if retained_only:
            if cursor is None:
                raise IntakeControllerError("retained-only pass has no finalized cursor")
            finalized_block, finalized_hash = chain.read_finalized_head(subtensor)
            if finalized_block < cursor[0]:
                raise IntakeControllerError("retained-only finalized head regressed")
            result = PassResult(finalized_block, finalized_hash)
            inserted = ()
        else:
            snapshot = chain.read_finalized_reveal_history(
                subtensor,
                netuid,
                after_block=None if cursor is None else cursor[0],
            )
            result = PassResult(snapshot.finalized_block, snapshot.finalized_block_hash)
            arrivals = _finalized_arrivals(
                snapshot,
                netuid=netuid,
                eval_cost_policy=eval_cost_policy,
                payment_lookup=lambda block, index: read_eval_cost_payment(
                    subtensor, block, index
                ),
                owner_lookup=lambda block: read_subnet_owner_coldkey(
                    subtensor, netuid, block=block
                ),
            )
            result.seen = len(arrivals)
            inserted = store.reserve_finalized(
                arrivals,
                finalized_block=snapshot.finalized_block,
                finalized_block_hash=snapshot.finalized_block_hash.lower(),
                eval_cost_amount_tao_rao=eval_cost_policy.amount_rao,
            )
        # Retained-only operation has no reservation transaction in which to
        # apply the finalized-block SLA.  The call is idempotent for normal
        # intake passes and keeps all downstream screening/settlement bounded.
        store.expire_stale(current_block=result.finalized_block)
        result.reserved.extend(row.reservation_id for row in inserted)

        for pending in store.pending(limit=policy.max_cohort):
            active = store.mark_fetching(pending.reservation_id)
            if active.status != "fetching":
                result.held.append(active.reservation_id)
                continue
            try:
                private = fetch_bundle(
                    active.arrival.url,
                    active.arrival.content_hash,
                    private_root,
                )
            except FetchTransientError as exc:
                store.mark_transport_retry(active.reservation_id, str(exc))
                continue
            except FetchError as exc:
                rejected = store.mark_failed(active.reservation_id, f"fetch:{exc}")
                result.rejected[rejected.reservation_id] = rejected.reason
                continue
            try:
                fingerprint = _fingerprint_private_bundle(private)
            except (OSError, TypeError, ValueError) as exc:
                rejected = store.mark_failed(active.reservation_id, f"manifest:{exc}")
                result.rejected[rejected.reservation_id] = rejected.reason
                continue
            # Fail-closed servability gate: a registered target whose arena
            # chokepoint can never execute must reject deterministically here,
            # not burn an evaluation lease to CandidateNeverExecutedError.
            if fingerprint.product_kind == "component":
                unservable = unservable_target_reason(fingerprint.target_id)
                if unservable is not None:
                    logger.info(
                        "rejecting %s: target %s is unservable: %s",
                        active.reservation_id,
                        fingerprint.target_id,
                        unservable,
                    )
                    rejected = store.mark_failed(
                        active.reservation_id,
                        f"unservable_target:{fingerprint.target_id}",
                    )
                    result.rejected[rejected.reservation_id] = rejected.reason
                    continue
            try:
                publication = publish_worker_bundle(
                    private,
                    publication_root,
                    active.arrival.content_hash,
                )
            except WorkerBundleSourceError as exc:
                rejected = store.mark_failed(
                    active.reservation_id, f"publication_source:{exc}"
                )
                result.rejected[rejected.reservation_id] = rejected.reason
                continue
            except WorkerBundlePublicationError as exc:
                # Publication/storage faults are validator-side NO_DECISION, never a
                # miner loss. The bounded transport retry policy eventually holds it.
                store.mark_transport_retry(active.reservation_id, f"publication:{exc}")
                continue
            published = store.mark_published(
                active.reservation_id,
                delta_fingerprint=fingerprint,
                publication_digest=publication.digest,
                publication_root=publication.root,
            )
            if published.status != "published":
                result.rejected[published.reservation_id] = published.reason
                continue
            result.published[published.reservation_id] = publication.digest

        # Publication and copy disposition are separate durable operations. Run a
        # complete idempotent reconciliation every pass so a crash in that window
        # cannot permanently bypass finalized priority.
        for copied, predecessor in store.reconcile_copies():
            result.copies[copied] = predecessor
            result.published.pop(copied, None)
        for copied, reference in reconcile_reference_copies(store):
            result.copies[copied] = f"validator_reference:{reference}"
            result.published.pop(copied, None)

        if service is not None:
            result.screens.update(
                _screen_pending(
                    store, service, current_block=result.finalized_block
                )
            )
            # Drain only what this arena can seal; otherwise a singleton arena
            # sees an oversized cohort and holds the entire promoted queue.
            cohort = store.promoted(
                limit=min(
                    policy.max_cohort,
                    service.manifest.capacity.max_cohort_size,
                )
            )
            if cohort:
                admission = service.admit_qualification(
                    store.arena_queue_snapshot(
                        current_block=result.finalized_block
                    ),
                    cohort_size=len(cohort),
                )
                if admission is AdmissionDecision.HOLD:
                    for row in cohort:
                        store.mark_held(
                            row.reservation_id, "arena_qualification_capacity_hold"
                        )
                    cohort = ()
                elif admission is AdmissionDecision.QUEUE:
                    cohort = ()
            if cohort:
                publications = tuple(
                    reopen_worker_bundle(
                        row.publication_root,
                        row.arrival.content_hash,
                        expected_receipt_digest=row.publication_digest,
                    )
                    for row in cohort
                )
                batch = _apply_qualification(
                    store,
                    cohort,
                    publications,
                    service,
                    minimum_finalized_block=result.finalized_block,
                    finalized_block_provider=lambda: chain.read_finalized_head(
                        subtensor
                    )[0],
                )
                result.decisions.update(
                    (row.reservation_digest, row.decision.value)
                    for row in batch.outcomes
                )
            result.settlements.update(
                _settle_pending(
                    store,
                    current_block=result.finalized_block,
                    finalized_block_provider=lambda: chain.read_finalized_head(subtensor),
                )
            )
        result.rejected.update(
            (row.reservation_id, row.reason)
            for row in inserted
            if row.status == "failed"
        )
        result.held.extend(
            row.reservation_id for row in store.all() if row.status == "held"
        )
    result.held = sorted(set(result.held))
    return result


def run_validator(
    subtensor,
    netuid: int,
    *,
    intake_db: str | Path,
    private_root: str | Path,
    publication_root: str | Path,
    policy: IntakePolicy = IntakePolicy(),
    eval_cost_policy: EvalCostPolicy = _DISABLED_EVAL_COST_POLICY,
    arena_registry: ArenaServiceRegistry | None = None,
    arena_id: str | None = None,
    intake_only: bool = False,
    retained_only: bool = False,
    interval_s: float = DEFAULT_INTERVAL_S,
    once: bool = False,
    max_consecutive_failures: int = 10,
    audit_log: str | Path | None = None,
) -> Optional[PassResult]:
    """Run finalized intake forever, containing validator-side pass failures."""

    failures = 0
    last: Optional[PassResult] = None
    while True:
        try:
            last = run_pass(
                subtensor,
                netuid,
                intake_db=intake_db,
                private_root=private_root,
                publication_root=publication_root,
                policy=policy,
                eval_cost_policy=eval_cost_policy,
                arena_registry=arena_registry,
                arena_id=arena_id,
                intake_only=intake_only,
                retained_only=retained_only,
            )
            failures = 0
            if audit_log is not None:
                try:
                    from cacheon.chain.audit_log import (
                        ChainAuditLogError,
                        append_chain_audit,
                        pass_audit_record,
                    )

                    append_chain_audit(audit_log, pass_audit_record(last))
                except ChainAuditLogError:
                    # SQLite is the transition authority; the redacted journal is
                    # supplementary observability. Surface loss loudly without
                    # replaying an already-committed pass.
                    logger.exception("validator chain audit append failed")
            logger.info(
                "intake @finalized %d: seen=%d reserved=%d published=%d copies=%d "
                "rejected=%d decisions=%d settlements=%d held=%d",
                last.finalized_block,
                last.seen,
                len(last.reserved),
                len(last.published),
                len(last.copies),
                len(last.rejected),
                len(last.decisions),
                len(last.settlements),
                len(last.held),
            )
        except Exception as exc:  # validator-side fault; supervisor may restart
            failures += 1
            if audit_log is not None:
                try:
                    from cacheon.chain.audit_log import (
                        ChainAuditLogError,
                        append_chain_audit,
                        fault_audit_record,
                    )

                    append_chain_audit(
                        audit_log,
                        fault_audit_record(
                            exc,
                            consecutive_failures=failures,
                        ),
                    )
                except ChainAuditLogError:
                    logger.exception("validator fault audit append failed")
            logger.exception("validator intake pass failed (%d consecutive)", failures)
            if once or failures >= max_consecutive_failures:
                raise
        if once:
            return last
        time.sleep(float(interval_s) * (1 + min(failures, 5)))


__all__ = [
    "IntakeControllerError", "PassResult", "QualificationWork", "run_pass",
    "run_validator",
]
