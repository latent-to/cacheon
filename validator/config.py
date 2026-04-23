"""Environment-driven defaults for the validator process.

Paths, poll interval, wallet names, subnet id, and timeouts are read from
``CACHEON_*`` variables so one codebase can target testnet vs mainnet,
different machines, and dry-run mode without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NETUID: int = int(os.environ.get("CACHEON_NETUID", "14"))

SUBTENSOR_NETWORK: str = os.environ.get("CACHEON_NETWORK", "finney")
"""Bittensor network name. `finney` = mainnet, `test` = testnet, or an ws:// URL."""

WALLET_NAME: str = os.environ.get("CACHEON_WALLET_NAME", "default")
WALLET_HOTKEY: str = os.environ.get("CACHEON_WALLET_HOTKEY", "default")

POLL_INTERVAL_S: int = int(os.environ.get("CACHEON_POLL_INTERVAL_S", "360"))
"""Seconds to sleep when there's nothing new to evaluate. GPU eval takes
minutes-to-hours; reacting faster than this buys nothing."""

CHAIN_RETRY_ATTEMPTS: int = int(os.environ.get("CACHEON_CHAIN_RETRY_ATTEMPTS", "3"))
CHAIN_RETRY_DELAY_S: int = int(os.environ.get("CACHEON_CHAIN_RETRY_DELAY_S", "30"))

STATE_DIR: Path = Path(
    os.environ.get("CACHEON_STATE_DIR", str(REPO_ROOT / "state"))
).resolve()
"""Where local JSON state files live. Gitignored."""

SANDBOX_PRECHECK_TIMEOUT_S: int = int(
    os.environ.get("CACHEON_SANDBOX_TIMEOUT_S", "60")
)
"""Hard timeout (seconds) for the static AST sandbox precheck per submission."""

DRY_RUN: bool = os.environ.get("CACHEON_DRY_RUN", "0") == "1"
"""When True, skip `subtensor.set_weights()` — log what would be set instead.
Useful for testing the loop without touching the chain."""

POLICY_CACHE_DIR: Path = Path(
    os.environ.get("CACHEON_POLICY_CACHE_DIR", str(STATE_DIR / "policy-cache"))
).resolve()
"""Where fetched `policy.py` files are cached on disk."""

POLICY_MAX_BYTES: int = int(
    os.environ.get("CACHEON_POLICY_MAX_BYTES", "1048576")
)
"""Hard size cap (bytes) on a single `policy.py` download. Default 1 MB."""

HF_ETAG_TIMEOUT_S: float = float(
    os.environ.get(
        "CACHEON_HF_ETAG_TIMEOUT_S",
        os.environ.get("CACHEON_HF_FETCH_TIMEOUT_S", "30.0"),
    )
)
"""Timeout (seconds) for the HEAD / etag revalidation request inside
``hf_hub_download``.  This does **not** cap the blob download itself —
only the metadata preflight.

``CACHEON_HF_FETCH_TIMEOUT_S`` is still read when
``CACHEON_HF_ETAG_TIMEOUT_S`` is unset (legacy alias)."""

HF_TOKEN: str | None = os.environ.get("CACHEON_HF_TOKEN")
"""Optional HuggingFace access token. Required only for private or gated
repositories; most miner repos are public."""


VERSION_KEY: int = int(os.environ.get("CACHEON_VERSION_KEY", "1"))
"""Version tag passed as `version_key` to `subtensor.set_weights(...)`. Bump
whenever the scoring mechanism, harness semantics, or sandbox rules change in
a way that would produce different king selections on identical commits.

Yuma consensus only trust-weights validators that agree on the version, so
bumping this effectively rolls consensus to the new version once a quorum of
stake has upgraded — validators still running the old code get their weights
ignored until they update."""

# --------------------------------------------------------------------------- #
# King defender-advantage window
# --------------------------------------------------------------------------- #

KING_EPSILON_INITIAL: float = float(
    os.environ.get("CACHEON_KING_EPSILON_INITIAL", "0.01")
)
"""Initial moat a fresh king holds: a challenger must beat
`king.score * (1 + KING_EPSILON_INITIAL)` to dethrone on the block the king
was crowned. Linearly decays to 0 over `KING_EPSILON_DECAY_BLOCKS`.

1% is small enough to not protect truly weak kings, large enough to swallow
float noise and discourage copycat submissions that match king byte-for-byte
(a byte-identical copy also trips the `duplicate_of_king` DQ path in
`state.record_evaluation`; the epsilon covers near-duplicates / scoring noise)."""

KING_EPSILON_DECAY_BLOCKS: int = int(
    os.environ.get("CACHEON_KING_EPSILON_DECAY_BLOCKS", "50400")
)
"""Number of chain blocks over which `KING_EPSILON_INITIAL` decays to 0.
50 400 blocks ≈ 7 days at ~12 s / block. After this window any strict
improvement dethrones — no grandfathering of stale kings."""
