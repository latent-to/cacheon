"""On-chain I/O: read commitments, push king-of-the-hill weights, preflight checks.

Architecture mirrors the sglang seam. **Pure helpers** (weight-vector math, hotkey↔uid
mapping) carry no SDK and no network, so they are unit-tested directly. **Thin RPC
wrappers** lazily ``import bittensor`` and are the only code that touches the chain —
so the package still imports, and the test suite still runs, with no SDK installed.

The exact SDK methods called here are pinned by ``optima chain-compat``
(optima/chain_canary.py); run it after any bittensor bump, and the wrappers' calls
stay a thin, auditable layer over what that canary asserts.

Submissions ride the chain's NATIVE commit-reveal (SUBNET_BLUEPRINT §3): a miner
posts a timelock-encrypted payload (``set_reveal_commitment``, ≤1024 bytes,
drand-encrypted until the reveal round — nobody can read the bundle URL before
reveal, and the reveal block is the anti-copy priority timestamp). The validator
reads ``get_all_revealed_commitments`` and replays them into the Ledger in chain
order; the Ledger keeps the off-chain half (copy detection + king-of-the-hill).
The older salted-hash transport (``set_commitment``/``get_all_commitments``)
remains for compatibility. The chain is the durable, consensus source of *what was
committed* and *who won*; the Ledger is the scoring half.
"""

from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("optima.chain")

# Yuma-consensus version stamped on set_weights. A coordinated subnet parameter,
# bumped deliberately (like PINNED_SGLANG) so every validator agrees.
WEIGHTS_VERSION_KEY = 1
CHAIN_REVEAL_HISTORY_CAP = 10
MAX_REVEAL_HISTORY_PAGES = 4_096
MAX_REVEAL_HISTORY_ROWS = 1_000_000


class ChainWeightStateError(RuntimeError):
    """The validator's currently active on-chain vector cannot be read safely."""

    validator_fault = True
    retryable = False


class ChainRevealHistoryError(RuntimeError):
    """The bounded chain view cannot prove complete anti-copy ordering."""

    validator_fault = True
    retryable = False


@dataclass
class Commitment:
    """A hotkey's current on-chain commitment — for Optima, the salted commit hash."""
    hotkey: str
    data: str
    block: int = 0


@dataclass
class RevealedCommitment:
    """One revealed (formerly timelock-encrypted) commitment. ``block`` is the reveal
    block the chain recorded — the consensus anti-copy priority timestamp."""
    hotkey: str
    data: str
    block: int


@dataclass
class MetagraphView:
    """A minimal, SDK-free snapshot of the metagraph used for weight-setting + seeding."""
    netuid: int
    block: int
    block_hash: str
    uids: list[int] = field(default_factory=list)
    hotkeys: list[str] = field(default_factory=list)  # index-aligned with uids
    validator_permit: list[bool] = field(default_factory=list)

    def uid_of(self, hotkey: str) -> Optional[int]:
        try:
            return self.uids[self.hotkeys.index(hotkey)]
        except ValueError:
            return None


@dataclass(frozen=True)
class ValidatorWeightSnapshot:
    """Authoritative sparse vector plus its on-chain last-update block."""

    weights: dict[str, float]
    last_update_block: int


# --------------------------------------------------------------------------- #
# Pure helpers — exercised with no chain
# --------------------------------------------------------------------------- #

def normalize(weights: dict[str, float]) -> dict[str, float]:
    """Scale to sum 1.0, dropping non-positive entries. Empty / all-zero -> {}."""
    pos = {k: float(v) for k, v in weights.items() if v and v > 0}
    total = sum(pos.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in pos.items()}


def weights_to_uid_vector(weights_by_hotkey: dict[str, float],
                          metagraph: MetagraphView) -> tuple[list[int], list[float]]:
    """Map ``{hotkey: weight}`` to ``(uids, weights)`` for set_weights, aligned to the
    *live* metagraph.

    A missing positive-weight champion is a safety event, not permission to silently
    redistribute its emission among the remaining titles. The caller must explicitly
    re-settle/neutralize that target before publishing another vector.
    """
    positive: dict[str, float] = {}
    for hotkey, raw_weight in weights_by_hotkey.items():
        if (
            not isinstance(hotkey, str)
            or not hotkey
            or isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
        ):
            raise ChainWeightStateError("weight projection contains an invalid entry")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ChainWeightStateError(
                f"weight projection contains an invalid value for {hotkey!r}"
            )
        if weight > 0:
            positive[hotkey] = weight
    missing = sorted(hotkey for hotkey in positive if metagraph.uid_of(hotkey) is None)
    if missing:
        raise ChainWeightStateError(
            "positive-weight hotkeys are absent from the live metagraph: "
            + ", ".join(missing[:16])
        )
    norm = normalize(positive)
    uids = [metagraph.uid_of(hk) for hk in norm]
    if (
        any(type(uid) is not int or uid < 0 for uid in uids)
        or len(set(uids)) != len(uids)
    ):
        raise ChainWeightStateError(
            "live metagraph maps champion hotkeys to invalid or duplicate UIDs"
        )
    weights = [norm[hk] for hk in norm]
    return uids, weights


def _chain_uint(raw, description: str, *, maximum: int | None = None) -> int:
    """Decode an SDK integer without accepting bools, floats, or truncation."""

    if isinstance(raw, bool):
        raise ChainWeightStateError(f"invalid {description}")
    try:
        value = operator.index(raw)
    except (TypeError, OverflowError):
        raise ChainWeightStateError(f"invalid {description}") from None
    if value < 0 or (maximum is not None and value > maximum):
        raise ChainWeightStateError(f"invalid {description}")
    return value


def read_validator_weight_snapshot(
    subtensor, netuid: int, validator_hotkey: str
) -> ValidatorWeightSnapshot:
    """Read one validator's authoritative vector and last-update block.

    The local weights-state file is only a durable submission cache; it cannot
    prove that a vector is or is not still active after failover/corruption.
    """

    if not isinstance(validator_hotkey, str) or not validator_hotkey:
        raise ChainWeightStateError("validator hotkey must be non-empty")
    try:
        metagraph = subtensor.metagraph(netuid=netuid)
        hotkeys = list(metagraph.hotkeys)
        raw_uids = list(metagraph.uids)
        raw_last_updates = list(metagraph.last_update)
    except Exception as exc:
        raise ChainWeightStateError(
            f"cannot fetch metagraph for active-weight verification: {exc}"
        ) from None

    if len(raw_uids) != len(hotkeys) or len(raw_last_updates) != len(hotkeys):
        raise ChainWeightStateError(
            "metagraph UID/hotkey/last-update widths differ"
        )
    if any(not isinstance(hotkey, str) or not hotkey for hotkey in hotkeys):
        raise ChainWeightStateError("metagraph contains an invalid hotkey")
    if len(set(hotkeys)) != len(hotkeys):
        raise ChainWeightStateError("metagraph contains duplicate hotkeys")

    uids = [
        _chain_uint(raw_uid, "metagraph UID")
        for raw_uid in raw_uids
    ]
    last_updates = [
        _chain_uint(raw_block, "metagraph last-update block")
        for raw_block in raw_last_updates
    ]
    if len(set(uids)) != len(uids):
        raise ChainWeightStateError("metagraph contains duplicate UIDs")

    uid_to_hotkey = dict(zip(uids, hotkeys))
    try:
        validator_index = hotkeys.index(validator_hotkey)
        validator_uid = uids[validator_index]
    except ValueError:
        return ValidatorWeightSnapshot({}, 0)

    # bt 10.3.2's default metagraph is lite=True, so ``metagraph.W`` is an
    # empty shape-(0,) array even when the subnet has neurons. Reading a dense
    # row from it produced the live netuid-307 failure (validator UID 3 indexed
    # into that empty array). The authoritative SDK API is sparse and returns:
    # ``[(source_uid, [(target_uid, uint16_weight), ...]), ...]``.
    try:
        raw_rows = list(subtensor.weights(netuid=netuid))
    except Exception as exc:
        raise ChainWeightStateError(
            f"cannot fetch validator on-chain weights: {exc}"
        ) from None

    rows: dict[int, dict[int, int]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 2:
            raise ChainWeightStateError("chain weight state contains a malformed row")
        raw_source_uid, raw_targets = raw_row
        source_uid = _chain_uint(raw_source_uid, "chain weight source UID")
        if source_uid not in uid_to_hotkey:
            raise ChainWeightStateError(
                "chain weight state contains a source UID absent from the metagraph"
            )
        if source_uid in rows:
            raise ChainWeightStateError("chain weight state contains duplicate source rows")
        if not isinstance(raw_targets, (list, tuple)):
            raise ChainWeightStateError("chain weight state contains malformed targets")

        targets: dict[int, int] = {}
        for raw_target in raw_targets:
            if not isinstance(raw_target, (list, tuple)) or len(raw_target) != 2:
                raise ChainWeightStateError(
                    "chain weight state contains a malformed target row"
                )
            raw_target_uid, raw_weight = raw_target
            target_uid = _chain_uint(raw_target_uid, "chain weight target UID")
            if target_uid not in uid_to_hotkey:
                raise ChainWeightStateError(
                    "chain weight state contains a target UID absent from the metagraph"
                )
            if target_uid in targets:
                raise ChainWeightStateError(
                    "chain weight state contains duplicate target UIDs"
                )
            weight = _chain_uint(
                raw_weight, "uint16 weight", maximum=65_535
            )
            targets[target_uid] = weight
        rows[source_uid] = targets

    result: dict[str, float] = {}
    for target_uid, weight in rows.get(validator_uid, {}).items():
        if weight > 0:
            result[uid_to_hotkey[target_uid]] = float(weight)
    return ValidatorWeightSnapshot(
        normalize(result),
        last_updates[validator_index],
    )


def read_validator_weights(
    subtensor, netuid: int, validator_hotkey: str
) -> dict[str, float]:
    """Compatibility projection of :func:`read_validator_weight_snapshot`."""

    return read_validator_weight_snapshot(
        subtensor, netuid, validator_hotkey
    ).weights


# --------------------------------------------------------------------------- #
# RPC wrappers — lazy bittensor; the only code that touches the chain
# --------------------------------------------------------------------------- #

def connect(network: str = "finney", *, fallback_endpoints: Optional[list[str]] = None,
            retry_forever: bool = False):
    """Open a subtensor client. ``network`` is a named network ('finney', 'test') or an
    explicit ``wss://`` endpoint URL. NOTE: the SDK's 'test' alias resolves to
    ``wss://test.finney.opentensor.ai:443`` — pass the URL explicitly if you mean a
    different testnet endpoint. ``fallback_endpoints``/``retry_forever`` enable the
    SDK's retrying substrate client (auto-reconnect through the fallback list)."""
    import bittensor as bt

    kwargs: dict = {}
    if fallback_endpoints:
        kwargs["fallback_endpoints"] = list(fallback_endpoints)
    if retry_forever:
        kwargs["retry_forever"] = True
    return bt.Subtensor(network=network, **kwargs)


def fetch_metagraph(subtensor, netuid: int) -> MetagraphView:
    mg = subtensor.metagraph(netuid=netuid)
    block = int(subtensor.get_current_block())
    return MetagraphView(
        netuid=netuid,
        block=block,
        block_hash=str(subtensor.get_block_hash(block)),  # chain-compat pins get_block_hash
        uids=[int(u) for u in mg.uids],
        hotkeys=list(mg.hotkeys),
        validator_permit=[bool(p) for p in getattr(mg, "validator_permit", [])],
    )


def read_commitments(subtensor, netuid: int) -> dict[str, Commitment]:
    """Read every hotkey's current commitment. Optima posts the salted commit hash;
    the reveal (bundle + salt) is verified off-chain by the Ledger."""
    block = int(subtensor.get_current_block())
    raw = subtensor.get_all_commitments(netuid=netuid)  # {hotkey: data}
    out: dict[str, Commitment] = {}
    for hotkey, data in dict(raw).items():
        if data is None:
            continue
        out[hotkey] = Commitment(hotkey=hotkey, data=str(data), block=block)
    return out


def read_reveal_history(
    subtensor, netuid: int, *, block: int | None = None
) -> tuple[RevealedCommitment, ...]:
    """Read every available reveal and return one deterministic global history.

    The chain returns each hotkey's reveal history (capped at the 10 most recent);
    discarding older entries loses anti-copy priority after validator downtime. A
    saturated page is first continued *at* its newest oldest boundary block. This
    excludes newer rows while retaining every reveal at the boundary, so multiple
    same-hotkey reveals in one block are not skipped. A subsequent page must move
    strictly backward; ten or more indistinguishable rows at one boundary therefore
    fail closed instead of being silently truncated. Every recovered row is retained,
    de-duplicated, and globally ordered by ``(reveal block, hotkey, payload)``. If
    historical state is unavailable, malformed, non-progressing, or exceeds a hard
    work bound, validation fails closed instead of guessing copy priority.
    """

    rows: dict[tuple[int, str, str], RevealedCommitment] = {}
    if block is not None and (type(block) is not int or block < 0):
        raise ValueError("reveal-history block must be a non-negative integer")
    query_block: int | None = block
    for page in range(MAX_REVEAL_HISTORY_PAGES):
        try:
            raw = subtensor.get_all_revealed_commitments(
                netuid=netuid,
                **({"block": query_block} if query_block is not None else {}),
            )
            page_items = dict(raw).items()
        except Exception as exc:
            raise ChainRevealHistoryError(
                "cannot retrieve complete historical reveal state: "
                f"{type(exc).__name__}: {exc}"
            ) from None

        saturated_oldest: list[int] = []
        for hotkey, history in page_items:
            if not history:
                continue
            if not isinstance(hotkey, str) or not hotkey or len(hotkey) > 256:
                raise ChainRevealHistoryError(
                    "chain reveal history contains an invalid hotkey"
                )
            if not isinstance(history, (list, tuple)):
                raise ChainRevealHistoryError(
                    "chain reveal history contains a malformed hotkey history"
                )
            history_blocks: list[int] = []
            for entry in history:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ChainRevealHistoryError(
                        "chain reveal history contains a malformed row"
                    )
                block, data = entry
                if (
                    type(block) is not int
                    or block < 0
                    or not isinstance(data, str)
                    or (query_block is not None and block > query_block)
                ):
                    raise ChainRevealHistoryError(
                        "chain reveal history contains invalid block/data provenance"
                    )
                history_blocks.append(block)
                row = RevealedCommitment(hotkey=hotkey, data=data, block=block)
                rows[(block, hotkey, data)] = row
                if len(rows) > MAX_REVEAL_HISTORY_ROWS:
                    raise ChainRevealHistoryError(
                        "historical reveal recovery exceeded its bounded row budget"
                    )
            if len(history) >= CHAIN_REVEAL_HISTORY_CAP:
                saturated_oldest.append(min(history_blocks))

        if not saturated_oldest:
            return tuple(rows[key] for key in sorted(rows))
        next_block = max(saturated_oldest)
        if query_block is not None and next_block >= query_block:
            raise ChainRevealHistoryError(
                "historical reveal pagination did not make backward progress; "
                "the chain may contain at least ten same-hotkey reveals at one block"
            )
        query_block = next_block

    raise ChainRevealHistoryError(
        f"historical reveal recovery exceeded {MAX_REVEAL_HISTORY_PAGES} pages"
    )


def read_revealed_commitments(subtensor, netuid: int) -> dict[str, RevealedCommitment]:
    """Compatibility/status view containing only each hotkey's latest reveal.

    Production validation uses :func:`read_reveal_history`; this lossy view remains
    for existing CLI status output and must not drive copy-priority accounting.
    """

    out: dict[str, RevealedCommitment] = {}
    for row in read_reveal_history(subtensor, netuid):
        previous = out.get(row.hotkey)
        if previous is None or (row.block, row.data) > (previous.block, previous.data):
            out[row.hotkey] = row
    return out


def set_weights(subtensor, wallet, netuid: int, weights_by_hotkey: dict[str, float], *,
                version_key: int = WEIGHTS_VERSION_KEY, dry_run: bool = False,
                wait_for_inclusion: bool = True, wait_for_finalization: bool = False) -> dict:
    """Push the king-of-the-hill weights on-chain.

    ``dry_run=True`` builds the ``(uids, weights)`` payload from the live metagraph and
    logs it WITHOUT signing or submitting — so the payload can be eyeballed before going
    live. Returns a structured result either way (never raises on an empty champion).
    """
    mg = fetch_metagraph(subtensor, netuid)
    uids, weights = weights_to_uid_vector(weights_by_hotkey, mg)
    if not uids:
        logger.warning("set_weights: no on-chain hotkeys to weight (champion deregistered?)")
        return {"submitted": False, "reason": "no eligible uids", "uids": [], "weights": []}
    if dry_run:
        logger.info("DRY RUN set_weights netuid=%s version_key=%s uids=%s weights=%s",
                    netuid, version_key, uids, weights)
        return {"submitted": False, "dry_run": True, "uids": uids, "weights": weights}
    result = subtensor.set_weights(
        wallet=wallet, netuid=netuid, uids=uids, weights=weights,
        version_key=version_key, wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # An included extrinsic can still FAIL chain-side (rate limit, permit, CR
    # window) — report that honestly or the caller records weights that never
    # applied. Measured on 307 (2026-07-10): a second commit 24 blocks after the
    # first was accepted by the SDK but never revealed (weights_rate_limit=100
    # applies to CR commits too); the old unconditional submitted=True wrote the
    # state file and suppressed every retry.
    if isinstance(result, tuple):  # older SDKs: (success, message)
        ok, message = bool(result[0]), str(result[1] if len(result) > 1 else "")
    else:
        ok = bool(getattr(result, "success", result))
        message = str(getattr(result, "message", ""))
    if not ok:
        logger.warning("set_weights failed on-chain: %s", message or result)
    return {"submitted": ok, "result": result, "message": message,
            "uids": uids, "weights": weights}


def post_commitment(subtensor, wallet, netuid: int, data: str, *, dry_run: bool = False) -> dict:
    """Miner side: post a commitment (Optima's salted commit hash) on-chain."""
    if dry_run:
        logger.info("DRY RUN set_commitment netuid=%s data=%s", netuid, data)
        return {"submitted": False, "dry_run": True, "data": data}
    result = subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data)
    return {"submitted": True, "result": result}


def post_reveal_commitment(subtensor, wallet, netuid: int, data: str, *,
                           blocks_until_reveal: int = 10, dry_run: bool = False) -> dict:
    """Miner side: post a timelock-encrypted commitment (the submission payload).

    The payload is drand-encrypted by the SDK and auto-revealed by the chain after
    ``blocks_until_reveal`` blocks — a copycat cannot read the bundle URL before the
    reveal, and the reveal block is the consensus priority timestamp. Hotkey-signed
    (no coldkey needed). Chain-side cap: 1024 bytes; budget ~3100 bytes/hotkey/epoch.
    """
    if dry_run:
        logger.info("DRY RUN set_reveal_commitment netuid=%s bytes=%d data=%s",
                    netuid, len(data.encode("utf-8")), data)
        return {"submitted": False, "dry_run": True, "data": data}
    result = subtensor.set_reveal_commitment(
        wallet=wallet, netuid=netuid, data=data, blocks_until_reveal=blocks_until_reveal,
    )
    return {"submitted": True, "result": result}


def preflight(subtensor, wallet, netuid: int) -> list:
    """Cheap pre-checks before scoring/weighting: is this validator registered, and does
    it hold a validator permit? Returns a list of ``Check`` (reuses the canary's type)."""
    from optima.chain_canary import Check

    checks: list[Check] = []
    hotkey = wallet.hotkey.ss58_address
    registered = bool(subtensor.is_hotkey_registered(hotkey_ss58=hotkey, netuid=netuid))
    checks.append(Check(f"hotkey registered on netuid {netuid}", registered, hotkey))
    if registered:
        mg = fetch_metagraph(subtensor, netuid)
        uid = mg.uid_of(hotkey)
        permit = bool(uid is not None and uid < len(mg.validator_permit)
                      and mg.validator_permit[uid])
        checks.append(Check(
            "validator permit", permit,
            f"uid {uid}" if permit else "no permit — weights ignored until you have stake/permit",
        ))
    return checks
