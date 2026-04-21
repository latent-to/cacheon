"""Phase 5 Part A — Chain interaction.

Two kinds of functions live here:

1. Pure helpers (`parse_commitment_data`, `build_commitments`,
   `build_winner_take_all_weights`) — no bittensor dependency, fully
   unit-testable with a fake metagraph object.

2. Thin RPC wrappers (`fetch_metagraph`, `fetch_revealed_commitments`,
   `set_weights`) — call into `bittensor` with retry + logging. The
   `bittensor` import is lazy so the rest of the validator package
   can be imported and tested without the chain lib installed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol

if TYPE_CHECKING:
    import bittensor as bt

from . import config as validator_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommitmentRecord:
    """One miner's most recent on-chain commitment, already parsed.

    On-chain format (encoded with `subtensor.set_reveal_commitment`):
        {"model": "hf-user/repo", "revision": "<git sha>"}
    """
    uid: int
    hotkey: str
    commit_block: int
    model: str
    revision: str
    raw: str  # original JSON string, kept for diagnostics

    def as_eval_key(self) -> tuple[str, int]:
        return (self.hotkey, self.commit_block)


class _MetagraphLike(Protocol):
    """Structural type for a bittensor metagraph. Kept minimal so tests
    can pass a plain object with `hotkeys` attribute."""
    hotkeys: list[str]


# --------------------------------------------------------------------------- #
# Pure parsing helpers
# --------------------------------------------------------------------------- #


def parse_commitment_data(raw: str) -> tuple[str, str] | None:
    """Parse a commitment payload into `(model, revision)` or None.

    Rejects anything that isn't a JSON object with non-empty `model` +
    `revision` strings. Silent on failure — the caller logs.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    model = obj.get("model")
    revision = obj.get("revision")
    if not isinstance(model, str) or not model.strip():
        return None
    if not isinstance(revision, str) or not revision.strip():
        return None
    return model.strip(), revision.strip()


def build_commitments(
    metagraph: _MetagraphLike,
    revealed: dict[str, list[tuple[int, str]]],
) -> dict[int, CommitmentRecord]:
    """Fold per-hotkey revealed commitments into `{uid: CommitmentRecord}`.

    Args:
        metagraph: object with a `hotkeys` list (uid → hotkey ss58).
        revealed: the raw dict returned by
            `subtensor.get_all_revealed_commitments(netuid)`, shaped as
            `{hotkey: [(block, data_str), ...]}`. If a hotkey has multiple
            commitments, we take the one with the highest block.

    Skips:
        - hotkeys with no commitments
        - commitments that fail JSON parsing or don't have `model`+`revision`
    """
    out: dict[int, CommitmentRecord] = {}
    hotkeys = list(metagraph.hotkeys)

    for uid, hotkey in enumerate(hotkeys):
        hotkey_str = str(hotkey)
        reveals = revealed.get(hotkey_str) or []
        if not reveals:
            continue
        block, raw = max(reveals, key=lambda p: p[0])
        parsed = parse_commitment_data(raw)
        if parsed is None:
            logger.debug(
                "UID %d (%s): commitment at block %d is not valid "
                "cacheon JSON — skipping.",
                uid, hotkey_str[:16] + "...", block,
            )
            continue
        model, revision = parsed
        out[uid] = CommitmentRecord(
            uid=uid,
            hotkey=hotkey_str,
            commit_block=int(block),
            model=model,
            revision=revision,
            raw=raw,
        )

    return out


def build_winner_take_all_weights(
    n_uids: int, winner_uid: int
) -> list[float]:
    """One-hot weight vector. All mass on `winner_uid`, everyone else 0."""
    if winner_uid < 0:
        raise ValueError(f"winner_uid must be non-negative, got {winner_uid}")
    size = max(n_uids, winner_uid + 1)
    weights = [0.0] * size
    weights[winner_uid] = 1.0
    return weights


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #


class ChainError(RuntimeError):
    """Raised when a chain RPC fails after all retries."""


def _retry(
    fn: Callable[[], Any],
    *,
    label: str,
    attempts: int,
    delay_s: float,
) -> Any:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s",
                label, i + 1, attempts, exc,
            )
            if i < attempts - 1:
                time.sleep(delay_s)
    raise ChainError(f"{label} failed after {attempts} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# Live RPC wrappers — bittensor imported lazily inside the functions
# --------------------------------------------------------------------------- #


def fetch_metagraph(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> tuple[bt.metagraph, int, str | None]:
    """Fetch metagraph + current block + block hash.

    Block hash is best-effort — RPC can flake under load and we'd rather
    keep going with a None hash than crash the loop.
    """
    def _inner() -> tuple[bt.metagraph, int, str | None]:
        metagraph = subtensor.metagraph(netuid)
        current_block = int(subtensor.block)
        try:
            block_hash = subtensor.substrate.get_block_hash(current_block)
        except Exception as exc:
            logger.warning(
                "Block hash lookup failed: %s — continuing with "
                "block_hash=None .",
                exc,
            )
            block_hash = None
        return metagraph, current_block, block_hash

    return _retry(
        _inner, label="fetch_metagraph",
        attempts=attempts, delay_s=delay_s,
    )


def fetch_revealed_commitments(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> dict[str, list[tuple[int, str]]]:
    """Return `{hotkey: [(block, data_str), ...]}` for the subnet.

    Delegates to whichever bittensor API is available — different
    versions name this differently.
    """
    def _inner() -> dict[str, list[tuple[int, str]]]:
        for method_name in (
            "get_all_revealed_commitments",
            "get_revealed_commitments",
        ):
            fn = getattr(subtensor, method_name, None)
            if callable(fn):
                return fn(netuid) or {}
        raise RuntimeError(
            "subtensor has no get_all_revealed_commitments / "
            "get_revealed_commitments method — bittensor version mismatch?"
        )

    return _retry(
        _inner, label="fetch_revealed_commitments",
        attempts=attempts, delay_s=delay_s,
    )


def set_weights(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    n_uids: int,
    winner_uid: int,
    *,
    version_key: int = validator_config.VERSION_KEY,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> None:
    """Push a one-hot weight vector on-chain. Raises `ChainError` if every
    attempt is rejected — the main loop should sleep and retry next cycle.

    `version_key` tags the weight vector with the validator's mechanism
    version; consensus only trust-weights validators that agree on it.
    """
    weights = build_winner_take_all_weights(n_uids, winner_uid)
    uids = list(range(len(weights)))

    logger.info(
        "Setting weights: winner_uid=%d, n_uids=%d, version_key=%d",
        winner_uid, len(weights), version_key,
    )

    last_reason: str | None = None
    for i in range(attempts):
        try:
            result = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            if isinstance(result, (tuple, list)):
                ok = bool(result[0])
                last_reason = (
                    str(result[1]) if len(result) > 1 else None
                )
            else:
                ok = bool(result)
                last_reason = None
            if ok:
                logger.info("✓ Weights set on-chain (winner_uid=%d)", winner_uid)
                return
            logger.warning(
                "set_weights attempt %d/%d rejected: %s",
                i + 1, attempts, last_reason,
            )
        except Exception as exc:
            last_reason = str(exc)
            logger.error(
                "set_weights attempt %d/%d raised: %s",
                i + 1, attempts, exc,
            )
        if i < attempts - 1:
            time.sleep(delay_s)

    raise ChainError(
        f"set_weights failed after {attempts} attempts: {last_reason}"
    )


def unique_hotkeys(
    commitments: Iterable[CommitmentRecord],
) -> set[str]:
    return {c.hotkey for c in commitments}


# --------------------------------------------------------------------------- #
# Startup preflight
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreflightResult:
    uid: int
    has_validator_permit: bool
    stake: float


class NotRegisteredError(RuntimeError):
    """Validator hotkey is not registered on this subnet — fatal at startup."""


def preflight_check(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
) -> PreflightResult:
    """Fail fast on unregistered hotkeys; warn on missing validator permit.

    Raises `NotRegisteredError` if the wallet hotkey isn't registered on the
    subnet — without this, the loop would happily run then fail every tick at
    `set_weights` with a cryptic substrate error. Permit absence is logged as
    a warning, not fatal: a freshly-staked validator may get permit granted on
    the next epoch without needing a restart.
    """
    hotkey_ss58 = wallet.hotkey.ss58_address

    if not subtensor.is_hotkey_registered(
        netuid=netuid, hotkey_ss58=hotkey_ss58
    ):
        raise NotRegisteredError(
            f"Hotkey {hotkey_ss58} is not registered on netuid {netuid}. "
            f"Register it with: btcli subnet register --netuid {netuid} "
            f"--wallet.name <name> --wallet.hotkey <hotkey>"
        )

    metagraph = subtensor.metagraph(netuid)
    uid = metagraph.hotkeys.index(hotkey_ss58)

    has_permit = False
    try:
        has_permit = bool(metagraph.validator_permit[uid])
    except (AttributeError, IndexError, TypeError):
        # older bittensor or odd metagraph shape — treat as unknown
        logger.debug("Could not read validator_permit[uid=%d]", uid)

    stake = 0.0
    try:
        stake = float(metagraph.S[uid])
    except (AttributeError, IndexError, TypeError):
        logger.debug("Could not read stake S[uid=%d]", uid)

    logger.info(
        "Preflight OK: uid=%d, stake=%.2f, validator_permit=%s",
        uid, stake, has_permit,
    )

    if not has_permit:
        logger.warning(
            "Validator permit not granted for uid=%d — weights will still be "
            "emitted, but may not count toward consensus until the subnet "
            "grants permit (typically next epoch, pending stake threshold).",
            uid,
        )

    return PreflightResult(uid=uid, has_validator_permit=has_permit, stake=stake)
