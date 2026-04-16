"""Deterministic prompt sampling from PG19 seeded by block hash.

Given a Bittensor block hash (hex string), produces a reproducible
list of text passages from PG19's train split.  Same hash always
yields the same passages on any machine.

PG19 is chosen for its long-form content (full novels).  Long prefill
sequences build large KV caches, which makes the memory and latency
axes in scoring more meaningful — at short context the KV cache is a
rounding error on peak GPU memory and compression gains are invisible.
"""

from __future__ import annotations

import hashlib
import logging
import random

from datasets import load_dataset

logger = logging.getLogger(__name__)

_DATASET_NAME = "pg19"
_DATASET_SPLIT = "train"
_DATASET_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
_TEXT_COLUMN = "text"

DEFAULT_MAX_CHARS = 131_072  # ~32K tokens at ~4 chars/token for English prose
DEFAULT_MIN_CHARS = 1_000


def _seed_from_hash(block_hash: str) -> int:
    digest = hashlib.sha256(block_hash.encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, cutting at a sentence or word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(". ")
    if cut > max_chars // 3:
        return text[: cut + 1]
    last_space = text[:max_chars].rfind(" ")
    if last_space > max_chars // 2:
        return text[:last_space]
    return text[:max_chars]


def sample_prompts(
    block_hash: str,
    n: int = 10,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    """Sample *n* PG19 passages deterministically from *block_hash*.

    Parameters
    ----------
    block_hash : str
        Hex string from the Bittensor chain (unique per block).
    n : int
        Number of passages to return.
    max_chars : int
        Truncate each passage to this many characters, cutting at the
        nearest sentence or word boundary.  Defaults to ~32K tokens
        worth of English prose (131,072 chars).
    min_chars : int
        Skip passages shorter than this (headers, frontmatter, OCR junk).

    Returns
    -------
    list[str]
        PG19 text passages, each between *min_chars* and *max_chars*.
    """
    seed = _seed_from_hash(block_hash)
    rng = random.Random(seed)

    ds = load_dataset(
        _DATASET_NAME,
        split=_DATASET_SPLIT,
        revision=_DATASET_REVISION,
        trust_remote_code=True,
    )
    total = len(ds)

    indices = list(range(total))
    rng.shuffle(indices)

    texts: list[str] = []
    for idx in indices:
        raw = ds[idx][_TEXT_COLUMN]
        if not raw or len(raw) < min_chars:
            continue
        texts.append(_truncate_at_boundary(raw, max_chars))
        if len(texts) >= n:
            break

    logger.info(
        "Sampled %d prompts from PG19 (seed derived from block_hash=%s…)",
        len(texts),
        block_hash[:12],
    )
    return texts
