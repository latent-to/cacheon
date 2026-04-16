"""Deterministic prompt sampling from PG19 seeded by block hash.

Given a Bittensor block hash (hex string), produces a reproducible
list of text passages from PG19's train split.  Same hash always
yields the same passages on any machine.
"""

from __future__ import annotations

import hashlib
import logging
import random

from datasets import load_dataset

logger = logging.getLogger(__name__)

_DATASET_NAME = "pg19"
_DATASET_SPLIT = "train"
_TEXT_COLUMN = "text"


def _seed_from_hash(block_hash: str) -> int:
    digest = hashlib.sha256(block_hash.encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def sample_prompts(block_hash: str, n: int = 10) -> list[str]:
    """Sample *n* PG19 passages deterministically from *block_hash*.

    Parameters
    ----------
    block_hash : str
        Hex string from the Bittensor chain (unique per block).
    n : int
        Number of passages to return.

    Returns
    -------
    list[str]
        Raw PG19 text passages — no truncation applied.
    """
    seed = _seed_from_hash(block_hash)
    rng = random.Random(seed)

    ds = load_dataset(_DATASET_NAME, split=_DATASET_SPLIT, trust_remote_code=True)
    total = len(ds)
    indices = rng.sample(range(total), k=min(n, total))

    texts: list[str] = [ds[i][_TEXT_COLUMN] for i in indices]
    logger.info(
        "Sampled %d prompts from PG19 (seed derived from block_hash=%s…)",
        len(texts),
        block_hash[:12],
    )
    return texts
