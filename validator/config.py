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

STATE_SCHEMA_VERSION: int = 1
"""Bump when the on-disk JSON schema changes in a backwards-incompatible way."""

SANDBOX_PRECHECK_TIMEOUT_S: int = int(
    os.environ.get("CACHEON_SANDBOX_TIMEOUT_S", "60")
)
"""Hard timeout for the Phase 3 static AST precheck per submission."""

DRY_RUN: bool = os.environ.get("CACHEON_DRY_RUN", "0") == "1"
"""When True, skip `subtensor.set_weights()` — log what would be set instead.
Useful for testing the loop without touching the chain."""
