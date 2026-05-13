#!/usr/bin/env python3
"""Search GPU cloud providers for instances matching Cacheon eval requirements.

Usage:
    python scripts/gpu_search.py                # pretty-print results from all providers
    python scripts/gpu_search.py --json          # raw JSON to stdout
    python scripts/gpu_search.py --best          # print only the single cheapest viable match
    python scripts/gpu_search.py --provider targon  # query only Targon

Importable:
    from scripts.gpu_search import search_all, search_provider
    results = search_all()
    results = search_provider("targon", api_key="...")

Conditions (applied to every provider):
    - VRAM/GPU: B300 >= 288 GB, B200 >= 180 GB, H200 >= 141 GB, H100 >= 80 GB, A100 >= 80 GB

Tier selection (cross-provider, cheapest wins):
    Tier A (preferred): 1x B300, 2x B200, 2x B300, 2x H200, 4x H200, 4x H100, 4x A100
    Tier B (fallback):  4x B300, 4x B200, 8x H200, 8x H100, 8x A100, 8x B200, 8x B300
    Tier B is only considered when Tier A has zero availability.

Providers: targon, lium
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

# ---------------------------------------------------------------------------
# Shared config
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

# ---------------------------------------------------------------------------
# Normalized instance format returned by every provider fetch function:
#
#   provider           str      "targon", "lium"
#   instance_type      str      provider-specific instance name
#   description        str      human-readable description
#   hourly_price_cents int      US cents
#   num_gpus           int
#   gpu_type           str      "H100", "H200", "A100", "B200"
#   vram_per_gpu_gb    int
#   total_vram_gb      int
#   storage_gb         int
#   memory_gb          int
#   vcpus              int
#   available_regions  list[str]
# ---------------------------------------------------------------------------

_VRAM_GB: dict[str, int] = {
    "B300": 288,
    "B200": 180,
    "H200": 141,
    "H100": 80,
    "A100": 80,
}


def _lookup_vram(gpu_type_raw: str) -> tuple[str, int]:
    """Return (canonical_type, vram_gb). Handles variants like 'H200 NVL'."""
    t = gpu_type_raw.upper()
    for canon, vram in _VRAM_GB.items():
        if canon in t:
            return canon, vram
    return "", 0


# ---------------------------------------------------------------------------
# Provider: Targon
# ---------------------------------------------------------------------------


def _targon_normalize_gpu_type(raw: str) -> str:
    """'NVIDIA-H200' -> 'H200', 'NVIDIA-GeForce-RTX-4090' -> 'RTX-4090'."""
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def _fetch_targon(api_key: str) -> list[dict]:
    resp = requests.get(
        "https://api.targon.com/tha/v2/inventory",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"type": "rental", "gpu": "true"},
        timeout=30,
    )
    resp.raise_for_status()

    out: list[dict] = []
    for item in resp.json():
        spec = item.get("spec", {})
        gpu_type_raw = _targon_normalize_gpu_type(spec.get("gpu_type", ""))
        gpu_count = spec.get("gpu_count", 0)
        available = item.get("available", 0)

        if available <= 0:
            continue

        canon, vram = _lookup_vram(gpu_type_raw)
        if not vram:
            continue

        storage_mb = spec.get("storage", 0)
        storage_gb = storage_mb // 1024 if storage_mb else 0
        price_dollars = item.get("cost_per_hour", 0)
        price_cents = int(round(price_dollars * 100))
        memory_mb = spec.get("memory", 0)

        out.append(
            {
                "provider": "targon",
                "instance_type": item.get("name", ""),
                "description": item.get("display_name", ""),
                "hourly_price_cents": price_cents,
                "num_gpus": gpu_count,
                "gpu_type": canon,
                "vram_per_gpu_gb": vram,
                "total_vram_gb": gpu_count * vram,
                "storage_gb": storage_gb,
                "memory_gb": memory_mb // 1024 if memory_mb else 0,
                "vcpus": spec.get("vcpu", 0),
                "available_regions": [f"targon ({available} avail)"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Provider: Lium
# ---------------------------------------------------------------------------


def _fetch_lium(api_key: str) -> list[dict]:
    from lium.sdk import Lium, Config

    client = Lium(config=Config(api_key=api_key))
    executors = client.ls(gpu_count=None)

    out: list[dict] = []
    for ex in executors:
        gpu_type_raw = ex.gpu_type or ""
        canon, vram = _lookup_vram(gpu_type_raw)
        if not vram:
            continue
        if not ex.docker_in_docker:
            continue

        out.append(
            {
                "provider": "lium",
                "instance_type": ex.id,
                "description": ex.machine_name,
                "hourly_price_cents": int(round(ex.price_per_hour * 100)),
                "num_gpus": ex.gpu_count,
                "gpu_type": canon,
                "vram_per_gpu_gb": vram,
                "total_vram_gb": ex.gpu_count * vram,
                "storage_gb": 0,
                "memory_gb": 0,
                "vcpus": 0,
                "available_regions": ["lium"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "targon": {"env_key": "TARGON_API_KEY", "fetch": _fetch_targon},
    "lium": {"env_key": "LIUM_API_KEY", "fetch": _fetch_lium},
}


# ---------------------------------------------------------------------------
# Tier matching (provider-agnostic)
# ---------------------------------------------------------------------------


def _matches_config(instance: dict, config: dict) -> bool:
    if config["gpu_type"].lower() not in instance["gpu_type"].lower():
        return False
    if instance["num_gpus"] != config["num_gpus"]:
        return False
    if instance["vram_per_gpu_gb"] < config["min_vram_per_gpu"]:
        return False
    return True


def _rank_tiers(candidates: list[dict]) -> dict:
    """Apply tier logic to a flat list of normalized candidates."""
    tier_results: list[dict] = []
    best_match: dict | None = None
    best_tier: str | None = None

    for tier_name, configs in TIERS:
        config_results: list[dict] = []
        tier_all: list[dict] = []

        for config in configs:
            matches = [c for c in candidates if _matches_config(c, config)]
            matches.sort(key=lambda m: m["hourly_price_cents"])
            config_results.append({"label": config["label"], "matches": matches})
            tier_all.extend(matches)

        tier_all.sort(key=lambda m: m["hourly_price_cents"])

        tier_results.append(
            {
                "tier": tier_name,
                "configs": config_results,
                "cheapest": tier_all[0] if tier_all else None,
                "total_available": len(tier_all),
            }
        )

        if tier_all and best_match is None:
            best_match = tier_all[0]
            best_tier = tier_name

    return {"tiers": tier_results, "best_match": best_match, "best_tier": best_tier}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_provider(name: str, api_key: str | None = None) -> dict:
    """Query a single provider and return tier-ranked results."""
    prov = PROVIDERS.get(name)
    if not prov:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {', '.join(PROVIDERS)}"
        )

    key = api_key or os.environ.get(prov["env_key"], "")
    if not key:
        raise ValueError(f"{prov['env_key']} not set and no api_key provided")

    candidates = prov["fetch"](key)
    result = _rank_tiers(candidates)
    result["providers_queried"] = [name]
    return result


def search_all(keys: dict[str, str] | None = None) -> dict:
    """Query every provider that has a configured API key, merge, and rank."""
    keys = keys or {}
    candidates: list[dict] = []
    queried: list[str] = []
    errors: dict[str, str] = {}

    for name, prov in PROVIDERS.items():
        key = keys.get(name) or os.environ.get(prov["env_key"], "")
        if not key:
            continue
        try:
            candidates.extend(prov["fetch"](key))
            queried.append(name)
        except Exception as exc:
            errors[name] = str(exc)

    result = _rank_tiers(candidates)
    result["providers_queried"] = queried
    if errors:
        result["provider_errors"] = errors
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_pretty(data: dict) -> None:
    queried = data.get("providers_queried", [])
    errors = data.get("provider_errors", {})
    print(f"\n  Providers queried: {', '.join(queried) or '(none)'}")
    for pname, err in errors.items():
        print(f"  WARNING: {pname} failed: {err}")

    best = data.get("best_match")
    best_tier = data.get("best_tier")
    if best:
        price = best["hourly_price_cents"] / 100
        print(f"\n  BEST MATCH  (Tier {best_tier})")
        print(f"  {best['provider']} / {best['instance_type']}  ${price:.2f}/hr")
        print(
            f"  {best['num_gpus']}x {best['gpu_type']}  "
            f"{best['total_vram_gb']}GB VRAM  {best['storage_gb']}GB disk"
        )
    else:
        print("\n  No matching instances found.")

    for tier_info in data["tiers"]:
        tier_name = tier_info["tier"]
        total = tier_info["total_available"]
        print(f"\n  Tier {tier_name}  [{total} total]")
        for cfg in tier_info["configs"]:
            n = len(cfg["matches"])
            tag = f"[{n} found]" if n else "[none]"
            print(f"    {cfg['label']:16s}  {tag}")
            for m in cfg["matches"]:
                price = m["hourly_price_cents"] / 100
                print(
                    f"          {m['provider']:12s}  {m['instance_type']:28s}"
                    f"  ${price:>6.2f}/hr  {m['storage_gb']}GB disk"
                )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search GPU cloud providers for Cacheon-viable instances",
    )
    parser.add_argument("--json", action="store_true", help="output raw JSON")
    parser.add_argument(
        "--best", action="store_true", help="output only best match JSON"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=f"query only this provider ({', '.join(PROVIDERS)})",
    )
    args = parser.parse_args()

    try:
        if args.provider:
            data = search_provider(args.provider)
        else:
            data = search_all()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as e:
        print(
            f"ERROR: API returned {e.response.status_code}: {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not data.get("providers_queried"):
        env_keys = [p["env_key"] for p in PROVIDERS.values()]
        print(
            f"ERROR: No provider API keys found. Set one of: {', '.join(env_keys)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.best:
        json.dump(data.get("best_match"), sys.stdout, indent=2)
        print()
    elif args.json:
        json.dump(data, sys.stdout, indent=2)
        print()
    else:
        _print_pretty(data)


if __name__ == "__main__":
    main()
