"""Validator configuration constants.

All defaults are overridable via env vars so the validator can run in dev
(4090, testnet) and prod (H100, mainnet) from the same codebase.
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
"""Hard timeout for the Phase 3 static AST precheck per submission."""

DRY_RUN: bool = os.environ.get("CACHEON_DRY_RUN", "0") == "1"
"""When True, skip `subtensor.set_weights()` — log what would be set instead.
Useful for testing the loop without touching the chain."""

VERSION_KEY: int = int(os.environ.get("CACHEON_VERSION_KEY", "1"))
"""Version tag passed as `version_key` to `subtensor.set_weights(...)`. Bump
whenever the scoring mechanism, harness semantics, or sandbox rules change in
a way that would produce different king selections on identical commits.

Yuma consensus only trust-weights validators that agree on the version, so
bumping this effectively rolls consensus to the new version once a quorum of
stake has upgraded — validators still running the old code get their weights
ignored until they update."""
