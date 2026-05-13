"""GPU cloud provider abstraction for automated pod rental.

Provides a uniform interface over Lium and Targon so the orchestrator
can search, rent, execute commands, and teardown pods without caring
which provider won the price auction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalized instance (same shape as gpu_search.py)
# ---------------------------------------------------------------------------


@dataclass
class GpuInstance:
    """A single available GPU offering from any provider."""

    provider: str
    instance_id: str
    description: str
    hourly_price_cents: int
    num_gpus: int
    gpu_type: str
    vram_per_gpu_gb: int
    total_vram_gb: int
    storage_gb: int
    memory_gb: int
    vcpus: int
    docker_in_docker: bool
    raw: Any = None


@dataclass
class PodHandle:
    """Opaque handle to a rented pod, returned by ``rent()``."""

    provider: str
    pod_id: str
    gpu_count: int
    hourly_price_cents: int
    raw: Any = None


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GpuProvider(Protocol):
    """Interface every GPU cloud provider must implement."""

    name: str

    def search(self) -> list[GpuInstance]: ...

    def rent(self, instance: GpuInstance) -> PodHandle: ...

    def wait_ready(self, handle: PodHandle, timeout_s: int = 600) -> PodHandle: ...

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]: ...

    def stream_exec(
        self, handle: PodHandle, command: str
    ) -> Generator[dict[str, str], None, None]: ...

    def teardown(self, handle: PodHandle) -> None: ...


# ---------------------------------------------------------------------------
# Tier ranking (shared with gpu_search.py)
# ---------------------------------------------------------------------------

TIER_A: list[dict] = [
    {"label": "1x B300", "gpu_type": "B300", "num_gpus": 1, "min_vram_per_gpu": 288},
    {"label": "2x B200", "gpu_type": "B200", "num_gpus": 2, "min_vram_per_gpu": 180},
    {"label": "2x B300", "gpu_type": "B300", "num_gpus": 2, "min_vram_per_gpu": 288},
    {
        "label": "2x H200 SXM",
        "gpu_type": "H200",
        "num_gpus": 2,
        "min_vram_per_gpu": 141,
    },
    {
        "label": "4x H200 SXM",
        "gpu_type": "H200",
        "num_gpus": 4,
        "min_vram_per_gpu": 141,
    },
    {"label": "4x H100 SXM", "gpu_type": "H100", "num_gpus": 4, "min_vram_per_gpu": 80},
    {
        "label": "4x A100 80GB",
        "gpu_type": "A100",
        "num_gpus": 4,
        "min_vram_per_gpu": 80,
    },
]

TIER_B: list[dict] = [
    {"label": "4x B300", "gpu_type": "B300", "num_gpus": 4, "min_vram_per_gpu": 288},
    {"label": "4x B200", "gpu_type": "B200", "num_gpus": 4, "min_vram_per_gpu": 180},
    {
        "label": "8x H200 SXM",
        "gpu_type": "H200",
        "num_gpus": 8,
        "min_vram_per_gpu": 141,
    },
    {"label": "8x H100 SXM", "gpu_type": "H100", "num_gpus": 8, "min_vram_per_gpu": 80},
    {
        "label": "8x A100 80GB",
        "gpu_type": "A100",
        "num_gpus": 8,
        "min_vram_per_gpu": 80,
    },
    {"label": "8x B200", "gpu_type": "B200", "num_gpus": 8, "min_vram_per_gpu": 180},
    {"label": "8x B300", "gpu_type": "B300", "num_gpus": 8, "min_vram_per_gpu": 288},
]

TIERS = [("A", TIER_A), ("B", TIER_B)]


def _matches_config(instance: GpuInstance, config: dict) -> bool:
    if config["gpu_type"].lower() not in instance.gpu_type.lower():
        return False
    if instance.num_gpus != config["num_gpus"]:
        return False
    if instance.vram_per_gpu_gb < config["min_vram_per_gpu"]:
        return False
    return True


def rank_tiers(candidates: list[GpuInstance]) -> GpuInstance | None:
    """Return the cheapest tier-A match, falling back to tier-B.

    Returns None when no candidate matches any tier.
    """
    for _tier_name, configs in TIERS:
        tier_matches: list[GpuInstance] = []
        for config in configs:
            tier_matches.extend(c for c in candidates if _matches_config(c, config))
        if tier_matches:
            tier_matches.sort(key=lambda m: m.hourly_price_cents)
            return tier_matches[0]
    return None


def _matches_any_tier(instance: GpuInstance) -> bool:
    """Check whether an instance matches at least one tier config."""
    for _tier_name, configs in TIERS:
        for config in configs:
            if _matches_config(instance, config):
                return True
    return False


def search_all_providers(
    providers: list[GpuProvider],
    *,
    max_hourly_price_cents: int = 0,
) -> GpuInstance | None:
    """Query all providers and return the cheapest tier-ranked match."""
    all_candidates: list[GpuInstance] = []
    for prov in providers:
        try:
            instances = prov.search()
            eligible = [i for i in instances if _matches_any_tier(i)]
            all_candidates.extend(eligible)
            logger.info(
                "%s: %d instance(s) found, %d eligible",
                prov.name,
                len(instances),
                len(eligible),
            )
            for inst in eligible:
                logger.info(
                    "\t%s  %dx %s  %dGB VRAM  $%.2f/hr",
                    inst.instance_id,
                    inst.num_gpus,
                    inst.gpu_type,
                    inst.total_vram_gb,
                    inst.hourly_price_cents / 100,
                )
        except Exception as exc:
            logger.error("%s search failed: %s", prov.name, exc)

    if max_hourly_price_cents > 0:
        all_candidates = [
            c for c in all_candidates if c.hourly_price_cents <= max_hourly_price_cents
        ]

    return rank_tiers(all_candidates)
