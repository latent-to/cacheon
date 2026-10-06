"""Commit-only follower publication; reveal checks run in a separate monitor."""

from cacheon import chain
from cacheon.chain.weights import (
    StaleWeightProjectionError, WeightPublicationError, WeightPublicationRecord,
    WeightPublicationResult, _advance, _authority_uids_unchanged, _metagraph_digest,
    _reveal_round,
)


def commit_weight_publication(
    subtensor, signer_wallet, projection, journal, *, refresh_blocks, dry_run=False,
):
    """Follow the latest offer; an accepted commit completes publication.

    The journal records attempts and paces unchanged successful offers. Historical
    intent/pending/held records never block this path. No weight-row readback or
    reveal polling is performed, and a failed attempt is retried next pass.
    """
    if type(refresh_blocks) is not int or refresh_blocks <= 0:
        raise WeightPublicationError("weight refresh cadence is malformed")
    if signer_wallet.hotkey.ss58_address != projection.validator_hotkey:
        raise WeightPublicationError("signer wallet differs from projection authority")
    live = chain.fetch_metagraph(subtensor, projection.netuid)
    age = live.block - projection.effective_block
    if not 0 <= age <= refresh_blocks:
        raise StaleWeightProjectionError("shared projection exceeds the initial freshness window")
    bound = (live if age == 0 else chain.fetch_metagraph(
        subtensor, projection.netuid, block=projection.effective_block))
    if _metagraph_digest(projection, bound) != projection.metagraph_digest:
        raise WeightPublicationError("weight projection metagraph binding cannot be reopened")
    if not _authority_uids_unchanged(projection, bound, live):
        raise WeightPublicationError("weight recipient UID mapping changed before signing")
    if dry_run:
        chain.set_weights(subtensor, None, projection.netuid, projection.weights,
                          dry_run=True, metagraph_view=live)
        return WeightPublicationResult(projection.digest, "dry_run", None,
                                       False, False, True, live.block)
    current = journal.load()
    if current is not None:
        previous = journal.retained_projection(current.projection_digest)
        if (previous.validator_hotkey != projection.validator_hotkey
                or previous.chain_scope_digest != projection.chain_scope_digest
                or previous.netuid != projection.netuid):
            raise WeightPublicationError("follower journal belongs to another authority")
        if (current.status == "confirmed" and previous.weights_ppm == projection.weights_ppm
                and live.block < current.confirmed_block + refresh_blocks):
            return WeightPublicationResult(previous.digest, "confirmed", current,
                                           False, False, False, live.block)
    intent = _advance(journal, current, WeightPublicationRecord(
        projection.digest, "intent", submit_block=live.block,
        retry_after_block=live.block, reason="before_sdk_submission"))
    response = None
    try:
        response = chain.set_weights(
            subtensor, signer_wallet, projection.netuid, projection.weights,
            wait_for_inclusion=True, wait_for_finalization=True, metagraph_view=live)
        accepted = response.get("submitted") is True
        failure = str(response.get("message") or response.get("reason") or "commit rejected")
    except Exception as exc:
        accepted = False
        failure = f"{type(exc).__name__}: {exc}"
    if not accepted:
        # Bittensor can replace third-party logging handlers. Keep the original
        # error in process stdout without terminating the follower watch loop.
        print(f"follow-weights commit failed; retry next pass: {failure}", flush=True)
    record = _advance(journal, intent, WeightPublicationRecord(
        projection.digest, "confirmed" if accepted else "released",
        submit_block=live.block, retry_after_block=live.block,
        confirmed_block=live.block if accepted else 0,
        reveal_round=_reveal_round(response),
        reason="block_inclusion" if accepted else "commit_failed_retry_next_pass"))
    return WeightPublicationResult(projection.digest, record.status, record,
                                   False, accepted, False, live.block)
