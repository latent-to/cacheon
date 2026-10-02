"""Follower availability policy: retain authenticated weights for twelve hours.

The source offer and its first observation time live in the existing signer
database. Local refreshes never advance that clock or the source high-water
mark. Every selected vector still goes through the normal publication journal.
"""

from dataclasses import replace
import json
import logging
import time

from cacheon import chain
from cacheon.chain import weight_share
from cacheon.chain.intake import SQLiteFollowerWeightPublicationJournal
from cacheon.chain.weight_share import (
    CurrentWeightOffer, WeightShareError, assert_monotonic_offer_update,
    publish_followed_weights, rebind_offer_signer,
)
from cacheon.chain.weights import WeightProjection, WeightPublicationError
from cacheon.economics import GlobalRewardProjectionContext, MetagraphMember
from cacheon.stack_identity import canonical_digest

MAX_RETAINED_SECONDS = 12 * 60 * 60
_CACHE_KEY = "follower_last_authenticated_offer"
logger = logging.getLogger("cacheon.chain.weight_share")


def _context(scope, signer, metagraph):
    return GlobalRewardProjectionContext(
        scope.digest, signer, metagraph.block, metagraph.block_hash.lower(),
        tuple(MetagraphMember(uid, hotkey) for uid, hotkey in
              zip(metagraph.uids, metagraph.hotkeys, strict=True)),
    )


def _load_source(store, identity):
    row = store._db.execute("SELECT value FROM metadata WHERE key=?", (_CACHE_KEY,)).fetchone()
    if row is None:
        return None
    try:
        cached = json.loads(row[0])
        if cached["identity"] != identity:
            return None
        offer = CurrentWeightOffer.from_dict(cached["offer"])
        if (offer.projection.chain_scope_digest != store.scope.digest
                or offer.projection.netuid != store.scope.netuid
                or type(cached["observed_at"]) not in (int, float)
                or not 0 <= cached["observed_at"] < float("inf")):
            raise ValueError("invalid retained offer authority or age")
        return cached
    except (KeyError, TypeError, ValueError, WeightShareError, WeightPublicationError) as exc:
        logger.warning("follow-weights: saved source unavailable: %s", exc)
        return None


def _burn_offer(subtensor, store, signer):
    target = chain.resolve_subnet_owner_burn_target(subtensor, store.scope.netuid)
    context = _context(store.scope, signer, target.metagraph)
    authority = canonical_digest("cacheon.follower.expired-source-burn.v1", {
        "scope": store.scope.digest, "netuid": store.scope.netuid,
        "owner_coldkey": target.owner_coldkey, "owner_hotkey": target.owner_hotkey,
        "burn_hotkey": target.hotkey, "metagraph": context.metagraph_digest,
    })
    projection = WeightProjection(
        store.scope.digest, store.scope.netuid, signer, authority, authority,
        authority, context.metagraph_digest, (authority,), 0, context.current_block,
        0, (), ((target.hotkey, 1_000_000),),
    )
    return CurrentWeightOffer.from_legacy_projection(projection)


def follow_with_fallback(*, store, subtensor, wallet, url, expected_authority,
                         max_skew_seconds, refresh_blocks, dry_run=False, now=None):
    """Select fresh, retained, or burn weights, then reconcile the exact vector."""
    now = time.time() if now is None else now
    signer = wallet.hotkey.ss58_address
    identity = {"url": url, "authority": expected_authority, "signer": signer}
    cached = _load_source(store, identity)
    live = chain.fetch_metagraph(subtensor, store.scope.netuid)
    context = _context(store.scope, signer, live)
    mode = "source"
    try:
        source = weight_share.fetch_current_weights(
            url, signer=wallet.hotkey, netuid=store.scope.netuid,
            max_skew_seconds=max_skew_seconds, expected_authority=expected_authority,
        )
        projection = source.projection
        if (projection.chain_scope_digest != store.scope.digest
                or projection.netuid != store.scope.netuid):
            raise WeightShareError("fetched projection differs from the connected chain scope")
        if projection.effective_block > live.block:
            live = chain.fetch_metagraph(subtensor, store.scope.netuid)
            context = _context(store.scope, signer, live)
        if not 0 <= live.block - projection.effective_block <= refresh_blocks:
            raise WeightShareError("shared weight projection exceeds the initial freshness window")
        bound = (live if projection.effective_block == live.block else
                 chain.fetch_metagraph(subtensor, store.scope.netuid, block=projection.effective_block))
        if _context(store.scope, signer, bound).metagraph_digest != projection.metagraph_digest:
            raise WeightShareError("shared projection metagraph binding cannot be reopened")
        if not {signer, *dict(projection.weights_ppm)} <= set(live.hotkeys):
            raise WeightShareError("shared projection recipient is no longer registered")
        if cached is not None:
            assert_monotonic_offer_update(CurrentWeightOffer.from_dict(cached["offer"]), source)
        if cached is None or source.to_dict() != cached["offer"]:
            cached = {"identity": identity, "offer": source.to_dict(), "observed_at": now}
            if not dry_run:
                with store._transaction():
                    store._db.execute(
                        "INSERT INTO metadata(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (_CACHE_KEY, json.dumps(cached, sort_keys=True)),
                    )
        if not 0 <= now - cached["observed_at"] <= MAX_RETAINED_SECONDS:
            raise WeightShareError("last distinct source offer is over twelve hours old")
    except Exception as exc:
        # Never use an unauthenticated/error response as a vector. The saved
        # authenticated source is the only candidate for the availability path.
        logger.warning("follow-weights: source unavailable: %s: %s", type(exc).__name__, exc)
        mode = "retained"
        source = None
        if cached is not None and 0 <= now - cached["observed_at"] <= MAX_RETAINED_SECONDS:
            source = CurrentWeightOffer.from_dict(cached["offer"])
            if (source.projection.effective_block > live.block
                    or not {signer, *dict(source.projection.weights_ppm)} <= set(live.hotkeys)):
                source = None
        if source is None:
            mode = "burn"
            source = _burn_offer(subtensor, store, signer)
            context = None  # burn already binds the owner's freshly resolved metagraph
    offer = rebind_offer_signer(source, signer)
    if context is not None:
        # Hotkey amounts stay exact; UIDs are resolved again at the current
        # finalized block. Retained evidence describes the source economics.
        offer = CurrentWeightOffer.from_legacy_projection(replace(
            offer.projection, effective_block=context.current_block,
            metagraph_digest=context.metagraph_digest,
        ))
    journal = SQLiteFollowerWeightPublicationJournal(store, offer)
    head = journal.load()
    if head is not None:
        retained = journal.retained_projection(head.projection_digest)
        if retained.validator_hotkey != signer:
            raise WeightShareError("follower journal belongs to another signer")
        # A new source arriving in the very same finalized block waits one
        # block, preserving the journal's same-block non-equivocation rule.
        if retained.effective_block >= offer.projection.effective_block:
            offer = CurrentWeightOffer.from_legacy_projection(retained)
            journal = SQLiteFollowerWeightPublicationJournal(store, offer)
    result = publish_followed_weights(
        subtensor=subtensor, signer_wallet=wallet, offer=offer, journal=journal,
        refresh_blocks=refresh_blocks, dry_run=dry_run,
    )
    return result, mode
