"""Baseline result cache for containerized evaluation.

The vLLM baseline is run once per prompt set and cached on disk so that
every challenger in the same round is compared against identical baseline
numbers. Cache is keyed by block hash (prompt sets are deterministic
given the block hash).

No Docker, no HTTP -- this module only handles serialization and disk I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BaselinePromptResult:
    """Baseline measurements for a single prompt."""

    tokens: list[str]
    top_logprobs: list[list[dict[str, Any]]]
    ttft_s: float
    throughput_tps: float
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "top_logprobs": self.top_logprobs,
            "ttft_s": self.ttft_s,
            "throughput_tps": self.throughput_tps,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselinePromptResult:
        return cls(
            tokens=list(data["tokens"]),
            top_logprobs=list(data["top_logprobs"]),
            ttft_s=float(data["ttft_s"]),
            throughput_tps=float(data["throughput_tps"]),
            output_tokens=int(data["output_tokens"]),
        )


@dataclass
class BaselineCache:
    """Full baseline run for a prompt set."""

    cache_key: str
    results: list[BaselinePromptResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineCache:
        return cls(
            cache_key=str(data["cache_key"]),
            results=[BaselinePromptResult.from_dict(r) for r in data["results"]],
        )


def derive_cache_key(block_hash: str) -> str:
    """SHA-256 of the block hash, first 16 hex chars."""
    return hashlib.sha256(block_hash.encode()).hexdigest()[:16]


def _cache_file_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"baseline_{cache_key}.json"


def load_cached_baseline(cache_dir: Path, cache_key: str) -> BaselineCache | None:
    """Load cached baseline from disk. Returns None on miss or corrupt file."""
    path = _cache_file_path(cache_dir, cache_key)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return BaselineCache.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
        logger.warning(
            "Corrupt baseline cache at %s (%s) -- treating as miss.", path, exc
        )
        return None


def save_baseline_cache(cache_dir: Path, cache_key: str, cache: BaselineCache) -> None:
    """Atomically write baseline cache to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file_path(cache_dir, cache_key)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(cache_dir),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
