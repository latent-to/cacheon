"""Finalized chain intake and immutable publication.

This production loop reserves the complete finalized event order before network
transport and publishes submitted bytes into a separate immutable worker tree.
Qualification claims and settlement belong to the standing supervisor, which
imports ``_settle_pending`` from here; wallet access belongs only to the separate
control-plane signer.  The old shell/CPU fake-score evaluator and JSON Ledger
settlement do not exist on this path.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cacheon import chain
from cacheon.chain.fetch import FetchError, FetchTransientError, fetch_bundle
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeError,
    IntakeScope,
    is_lock_collision,
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
    WorkerBundlePublicationError,
    WorkerBundleSourceError,
    publish_worker_bundle,
)
from cacheon.chain.reference_copy_policy import reconcile_reference_copies
from cacheon.copy_fingerprint import fingerprint_submitted_delta


logger = logging.getLogger("cacheon.chain.validator")
DEFAULT_INTERVAL_S = 60.0
_DISABLED_EVAL_COST_POLICY = EvalCostPolicy(amount_rao=0)


class IntakeControllerError(RuntimeError):
    """Validator-owned intake/qualification authority is inconsistent."""


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
    """Fingerprint by exact component-parser success; never by miner-provided mode."""

    try:
        return fingerprint_submitted_delta(root)
    except (OSError, TypeError, ValueError) as component_error:
        raise ValueError(
            f"submission is not a registered component: {component_error}"
        ) from None


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
            lineage_tips=lease.lineage_tips,
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
) -> PassResult:
    """Run one non-emitting intake and publication pass."""

    if type(eval_cost_policy) is not EvalCostPolicy:
        raise IntakeControllerError("eval-cost policy is not typed")

    scope = IntakeScope(str(subtensor.get_block_hash(0)).lower(), netuid)
    with _open_store(intake_db, policy, scope) as store:
        cursor = store.finalized_cursor()
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
        # Idempotent per pass; it keeps all downstream qualification and
        # settlement bounded by the finalized-block SLA.
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
                from cacheon.manifest import load_manifest
                manifest = load_manifest(private)
                selected_arena = "" if manifest.competition is None else manifest.competition.arena
            except (OSError, TypeError, ValueError) as exc:
                rejected = store.mark_failed(active.reservation_id, f"manifest:{exc}")
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
                competition_arena=selected_arena,
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


_LOCK_RETRY_ATTEMPTS = 8
_LOCK_RETRY_PAUSE_S = 0.25


def _open_store(
    intake_db: str | Path,
    policy: IntakePolicy,
    scope: IntakeScope,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> FinalizedIntakeStore:
    """Open the intake store, waiting out a peer controller's short lock hold.

    The standing supervisor takes the same exclusive lock for a fraction of a
    second on every poll. On 2026-09-06 the intake
    pass met that hold on roughly every other pass and counted each one as a
    validator fault, so ten unlucky passes in a row would have stopped intake
    over nothing. The bounded wait spans one dispatcher poll; a collision that
    outlasts it still raises the original error, and no other store error is
    retried.
    """

    for attempt in range(1, _LOCK_RETRY_ATTEMPTS + 1):
        try:
            return FinalizedIntakeStore(intake_db, policy, scope=scope)
        except IntakeError as exc:
            if not is_lock_collision(exc) or attempt == _LOCK_RETRY_ATTEMPTS:
                raise
        sleep(_LOCK_RETRY_PAUSE_S)
    raise AssertionError("unreachable")


def run_validator(
    subtensor,
    netuid: int,
    *,
    intake_db: str | Path,
    private_root: str | Path,
    publication_root: str | Path,
    policy: IntakePolicy = IntakePolicy(),
    eval_cost_policy: EvalCostPolicy = _DISABLED_EVAL_COST_POLICY,
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


__all__ = ["IntakeControllerError", "PassResult", "run_pass", "run_validator"]
