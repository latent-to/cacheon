"""Everything that talks to Bittensor from the validator.

**Pure helpers** -- `parse_commitment_data`, `build_commitments`,
`build_winner_take_all_weights`, etc. These only need plain data structures
and are easy to unit test with a fake metagraph.

**RPC wrappers** -- `fetch_metagraph`, `fetch_revealed_commitments`,
`set_weights`. They add retries and logging around `bittensor` calls.
The library is imported lazily inside those paths so importing
`validator.*` in tests does not require `bittensor` to be installed.
"""

from __future__ import annotations

import json
import logging
import re
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
        {"image": "registry/repo:tag", "digest": "sha256:<64-char hex>"}

    `image` is a Docker image reference (registry/repo:tag or repo:tag).
    `digest` is the image manifest digest (sha256:...) that pins the exact
    image content regardless of tag mutations.
    """

    uid: int
    hotkey: str
    commit_block: int
    image: str
    digest: str
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


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*$")
_TAG_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_valid_docker_image(image: str) -> bool:
    """Validate a Docker image reference like ``registry:port/repo:tag``.

    The tag is the substring after the last ``:`` only when no ``/``
    follows that colon (so ``registry:5000/repo`` is a port, not a tag).
    """
    if not image:
        return False
    name = image
    last_colon = image.rfind(":")
    if last_colon > 0 and "/" not in image[last_colon:]:
        tag = image[last_colon + 1 :]
        name = image[:last_colon]
        if not tag or not _TAG_RE.match(tag):
            return False
    return bool(_NAME_RE.match(name))


def parse_commitment_data(raw: str) -> tuple[str, str] | None:
    """Parse a commitment payload into `(image, digest)` or None.

    Rejects anything that isn't a JSON object with non-empty `image` +
    `digest` strings. `image` must look like a Docker image reference
    (e.g. `docker.io/user/repo:tag`). `digest` must be a sha256 manifest
    digest (`sha256:<64 hex chars>`). Silent on failure -- caller logs.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    image = obj.get("image")
    digest = obj.get("digest")
    if not isinstance(image, str) or not image.strip():
        return None
    if not isinstance(digest, str) or not digest.strip():
        return None
    image = image.strip()
    digest = digest.strip()
    if not is_valid_docker_image(image):
        return None
    if not DIGEST_RE.match(digest):
        return None
    return image, digest


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
        - commitments that fail JSON parsing or don't have `image`+`digest`
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
                "cacheon JSON -- skipping.",
                uid,
                hotkey_str[:16] + "...",
                block,
            )
            continue
        image, digest = parsed
        out[uid] = CommitmentRecord(
            uid=uid,
            hotkey=hotkey_str,
            commit_block=int(block),
            image=image,
            digest=digest,
            raw=raw,
        )

    return out


def build_winner_take_all_weights(n_uids: int, winner_uid: int) -> list[float]:
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
                label,
                i + 1,
                attempts,
                exc,
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
                "Block hash lookup failed: %s — continuing with block_hash=None .",
                exc,
            )
            block_hash = None
        return metagraph, current_block, block_hash

    return _retry(
        _inner,
        label="fetch_metagraph",
        attempts=attempts,
        delay_s=delay_s,
    )


def _decode_raw_commitment(raw: str | bytes, *, _depth: int = 0) -> str:
    """Normalize a raw on-chain commitment value to a plain JSON string.

    Handles formats observed in the wild, including double-hex encoding
    (SDK hex-encodes, then substrate hex-encodes again):

    1. Plain JSON string (e.g. ``'{"image": ...}'``).
    2. Hex-encoded string with ``0x`` prefix and optional SCALE compact
       length prefix (bittensor SDK stores this way in some versions).
    3. Raw bytes with a SCALE compact length prefix followed by UTF-8
       JSON (substrate library decodes to this in some versions, appears
       as ``'E\\x02{"image": ...}'``).
    4. Double-hex: outer 0x decode yields SCALE prefix + ``0x`` + inner
       hex of JSON. Recurses to unwrap.
    """
    if _depth > 3:
        return str(raw)

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw)

    if s.startswith("0x"):
        try:
            decoded = bytes.fromhex(s[2:])
        except ValueError:
            return s
        text = decoded.decode("utf-8", errors="replace")
        idx_brace = text.find("{")
        idx_0x = text.find("0x")
        if idx_0x >= 0 and (idx_brace < 0 or idx_0x < idx_brace):
            return _decode_raw_commitment(text[idx_0x:], _depth=_depth + 1)
        return text[idx_brace:] if idx_brace >= 0 else text

    idx = s.find("{")
    if idx > 0:
        return s[idx:]
    return s


def _fetch_commitments_raw_substrate(
    subtensor: bt.Subtensor,
    netuid: int,
) -> dict[str, list[tuple[int, str]]]:
    """Fallback: query the substrate storage map directly.

    Bypasses bittensor's hex decoder, which crashes on some commitment
    encodings. Returns the same shape as the SDK methods.
    """
    result = subtensor.substrate.query_map(
        module="Commitments",
        storage_function="RevealedCommitments",
        params=[netuid],
    )
    out: dict[str, list[tuple[int, str]]] = {}
    for key, value in result:
        hotkey = str(key)
        entries = []
        for entry in value or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                data_raw, block = entry[0], entry[1]
            else:
                continue
            entries.append((int(block), _decode_raw_commitment(data_raw)))
        if entries:
            out[hotkey] = entries
    return out


def fetch_revealed_commitments(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> dict[str, list[tuple[int, str]]]:
    """Return `{hotkey: [(block, data_str), ...]}` for the subnet.

    Tries the bittensor SDK first; falls back to a raw substrate query
    if the SDK chokes on hex decoding (observed with mixed commitment
    encodings on chain).
    """

    def _inner() -> dict[str, list[tuple[int, str]]]:
        for method_name in (
            "get_all_revealed_commitments",
            "get_revealed_commitments",
        ):
            fn = getattr(subtensor, method_name, None)
            if callable(fn):
                try:
                    return fn(netuid) or {}
                except ValueError as exc:
                    if "fromhex" in str(exc) or "hexadecimal" in str(exc):
                        logger.warning(
                            "SDK %s hit hex decode error, falling back "
                            "to raw substrate query: %s",
                            method_name,
                            exc,
                        )
                        return _fetch_commitments_raw_substrate(subtensor, netuid)
                    raise
        raise RuntimeError(
            "subtensor has no get_all_revealed_commitments / "
            "get_revealed_commitments method -- bittensor version mismatch?"
        )

    return _retry(
        _inner,
        label="fetch_revealed_commitments",
        attempts=attempts,
        delay_s=delay_s,
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
        winner_uid,
        len(weights),
        version_key,
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
                last_reason = str(result[1]) if len(result) > 1 else None
            else:
                ok = bool(result)
                last_reason = None
            if ok:
                logger.info("✓ Weights set on-chain (winner_uid=%d)", winner_uid)
                return
            logger.warning(
                "set_weights attempt %d/%d rejected: %s",
                i + 1,
                attempts,
                last_reason,
            )
        except Exception as exc:
            last_reason = str(exc)
            logger.error(
                "set_weights attempt %d/%d raised: %s",
                i + 1,
                attempts,
                exc,
            )
        if i < attempts - 1:
            time.sleep(delay_s)

    raise ChainError(f"set_weights failed after {attempts} attempts: {last_reason}")


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

    if not subtensor.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58):
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
        uid,
        stake,
        has_permit,
    )

    if not has_permit:
        logger.warning(
            "Validator permit not granted for uid=%d — weights will still be "
            "emitted, but may not count toward consensus until the subnet "
            "grants permit (typically next epoch, pending stake threshold).",
            uid,
        )

    return PreflightResult(uid=uid, has_validator_permit=has_permit, stake=stake)
