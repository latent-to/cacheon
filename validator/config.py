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

POLL_INTERVAL_S: int = int(os.environ.get("CACHEON_POLL_INTERVAL_S", "600"))
"""Seconds to sleep when there's nothing new to evaluate. Docker eval takes
minutes; reacting faster than this buys nothing."""

CHAIN_RETRY_ATTEMPTS: int = int(os.environ.get("CACHEON_CHAIN_RETRY_ATTEMPTS", "3"))
CHAIN_RETRY_DELAY_S: int = int(os.environ.get("CACHEON_CHAIN_RETRY_DELAY_S", "30"))

STATE_DIR: Path = Path(
    os.environ.get("CACHEON_STATE_DIR", str(REPO_ROOT / "state-mainnet"))
).resolve()
"""Where local JSON state files live. Gitignored."""

DRY_RUN: bool = os.environ.get("CACHEON_DRY_RUN", "0") == "1"
"""When True, skip `subtensor.set_weights()` and do not run Docker eval.
Useful for testing the loop without touching the chain."""

VERSION_KEY: int = int(os.environ.get("CACHEON_VERSION_KEY", "1"))
"""Version tag passed as `version_key` to `subtensor.set_weights(...)`. Bump
whenever the scoring mechanism or evaluation rules change in a way that would
produce different winner selections on identical commits.

Yuma consensus only trust-weights validators that agree on the version, so
bumping this effectively rolls consensus to the new version once a quorum of
stake has upgraded."""

MODEL_VOLUME: str = os.environ.get("CACHEON_MODEL_VOLUME", "/models")
"""Host path mounted read-only into miner/baseline containers at ``/models``."""

MODEL_PATH: str = os.environ.get("CACHEON_MODEL_PATH", "/models")
"""Path to read model config.json inside the gpu-eval container (mounted model dir)."""

BASELINE_IMAGE: str = os.environ.get(
    "CACHEON_BASELINE_IMAGE", "vllm/vllm-openai:latest"
)
BASELINE_DIGEST: str = os.environ.get("CACHEON_BASELINE_DIGEST", "")

GPU_COUNT: int = int(os.environ.get("CACHEON_GPU_COUNT", "8"))
"""Number of GPUs on the host. Set to 8 for 8x H200/B200/B300 (the standard eval tier)."""

# --------------------------------------------------------------------------- #
# GPU orchestration (auto-rent)
# --------------------------------------------------------------------------- #

AUTO_RENT: bool = os.environ.get("CACHEON_AUTO_RENT", "0") == "1"
"""When True, the validator automatically rents a GPU pod when challengers
are detected, runs eval, and tears it down."""

PREFERRED_PROVIDER: str = os.environ.get("CACHEON_PREFERRED_PROVIDER", "")
"""If set to 'lium' or 'targon', only that provider is used for GPU rental
even when both API keys are configured. Empty means cheapest-wins."""

LIUM_API_KEY: str = os.environ.get("LIUM_API_KEY", "")
TARGON_API_KEY: str = os.environ.get("TARGON_API_KEY", "")
TARGON_VOLUME_UID: str = os.environ.get("TARGON_VOLUME_UID", "")

MAX_HOURLY_PRICE: int = int(os.environ.get("CACHEON_MAX_HOURLY_PRICE", "2000"))
"""Maximum hourly price in US cents. Refuse to rent above this."""

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
"""Hugging Face token passed to the remote pod for model download."""

HIPPIUS_ACCESS_KEY: str = os.environ.get("HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY: str = os.environ.get("HIPPIUS_SECRET_KEY", "")
S3_BUCKET: str = os.environ.get("CACHEON_S3_BUCKET", "cacheon-validator")
S3_PREFIX: str = os.environ.get("CACHEON_S3_PREFIX", "state-mainnet")

# --------------------------------------------------------------------------- #
# Winner defender-advantage window
# --------------------------------------------------------------------------- #

WINNER_EPSILON_INITIAL: float = float(
    os.environ.get("CACHEON_WINNER_EPSILON_INITIAL", "0.01")
)
"""Initial moat a fresh winner holds: a challenger must beat
`winner.score * (1 + WINNER_EPSILON_INITIAL)` to overtake on the block the
winner won. Linearly decays to 0 over `WINNER_EPSILON_DECAY_BLOCKS`.

1% is small enough to not protect truly weak winners, large enough to swallow
float noise and discourage copycat submissions that match the winner
byte-for-byte (a byte-identical copy also trips the `duplicate_of_winner` DQ
path in `state.record_evaluation`; the epsilon covers near-duplicates /
scoring noise)."""

WINNER_EPSILON_DECAY_BLOCKS: int = int(
    os.environ.get("CACHEON_WINNER_EPSILON_DECAY_BLOCKS", "50400")
)
"""Number of chain blocks over which `WINNER_EPSILON_INITIAL` decays to 0.
50 400 blocks ~ 7 days at ~12 s / block. After this window any strict
improvement overtakes."""

# --------------------------------------------------------------------------- #
# Competition weight distribution
# --------------------------------------------------------------------------- #

WINNER_WEIGHT_SHARE: float = float(
    os.environ.get("CACHEON_WINNER_WEIGHT_SHARE", "0.80")
)
"""Fraction of the competition pool allocated to the winner."""

RUNNER_UP_WEIGHT_SHARE: float = float(
    os.environ.get("CACHEON_RUNNER_UP_WEIGHT_SHARE", "0.20")
)
"""Fraction of the competition pool allocated to the runner-up.
When no runner-up exists, the winner receives 100% of the pool."""

SCORE_EMISSION_TARGET: float = float(
    os.environ.get("CACHEON_SCORE_EMISSION_TARGET", "0.10")
)
"""Improvement score at which the competition pool earns 100% of emission.
Below this threshold, emission scales linearly; the remainder goes to the
burn UID. Example: with target 0.10, a winner scoring 0.05 earns 50% of
emission for the competition pool."""

BURN_UID: int = int(os.environ.get("CACHEON_BURN_UID", "22"))
"""UID that receives the unused portion of emission when the winner's score
is below SCORE_EMISSION_TARGET. Must not collide with the winner or
runner-up UID; the weight builder folds burn weight into the winner on
collision."""

# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #

LOG_RETENTION_DAYS: int = int(os.environ.get("CACHEON_LOG_RETENTION_DAYS", "10"))
"""Delete log files in ``state/logs/`` older than this many days (by filename
timestamp). 0 disables pruning."""
