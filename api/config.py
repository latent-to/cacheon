"""Environment-driven config for the monitoring API."""

import os
from pathlib import Path

STATE_DIR: Path = Path(
    os.environ.get(
        "CACHEON_STATE_DIR",
        str(Path(__file__).resolve().parent.parent / "state-mainnet"),
    )
).resolve()

ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get(
        "CACHEON_ALLOWED_ORIGINS",
        "https://cacheon.ai,https://www.cacheon.ai",
    ).split(",")
    if o.strip()
]
