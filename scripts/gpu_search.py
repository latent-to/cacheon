#!/usr/bin/env python3
"""Search GPU cloud providers for instances matching Cacheon eval requirements.

Usage:
    python scripts/gpu_search.py                # pretty-print results from all providers
    python scripts/gpu_search.py --json          # raw JSON to stdout
    python scripts/gpu_search.py --best          # print only the single cheapest viable match
    python scripts/gpu_search.py --provider lambda  # query only Lambda

Importable:
    from scripts.gpu_search import search_all, search_provider
    results = search_all()                       # queries every provider with a configured key
    results = search_provider("lambda", api_key="...")

Conditions (applied to every provider):
    - Storage: >= 400 GB
    - NVLink / SXM: required
    - VRAM/GPU: H200 >= 141 GB, H100 >= 80 GB, A100 >= 80 GB, B200 >= 180 GB

Tier selection (cross-provider, cheapest wins):
    Tier A (preferred): 4x H200, 8x H100, 8x A100 80GB, 8x H200, 2x B200
    Tier B (fallback):  4x H100, 4x A100 80GB, 4x B200, 8x B200
    Tier B is only considered when Tier A has zero availability.

Providers: shadeform, lambda, targon   (extend PROVIDERS to add more)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Callable

import requests

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

TIER_A: list[dict] = [
    {
        "label": "4x H200 SXM",
        "gpu_type": "H200",
        "num_gpus": 4,
        "min_vram_per_gpu": 141,
    },
    {"label": "8x H100 SXM", "gpu_type": "H100", "num_gpus": 8, "min_vram_per_gpu": 80},
    {
        "label": "8x A100 80GB",
        "gpu_type": "A100",
        "num_gpus": 8,
        "min_vram_per_gpu": 80,
    },
    {
        "label": "8x H200 SXM",
        "gpu_type": "H200",
        "num_gpus": 8,
        "min_vram_per_gpu": 141,
    },
    {"label": "2x B200", "gpu_type": "B200", "num_gpus": 2, "min_vram_per_gpu": 180},
]

TIER_B: list[dict] = [
    {
        "label": "2x H200 SXM",
        "gpu_type": "H200",
        "num_gpus": 2,
        "min_vram_per_gpu": 141,
    },
    {"label": "4x H100 SXM", "gpu_type": "H100", "num_gpus": 4, "min_vram_per_gpu": 80},
    {
        "label": "4x A100 80GB",
        "gpu_type": "A100",
        "num_gpus": 4,
        "min_vram_per_gpu": 80,
    },
    {"label": "4x B200", "gpu_type": "B200", "num_gpus": 4, "min_vram_per_gpu": 180},
    {"label": "8x B200", "gpu_type": "B200", "num_gpus": 8, "min_vram_per_gpu": 180},
]

TIERS = [("A", TIER_A), ("B", TIER_B)]

MIN_STORAGE_GB = 400

# ---------------------------------------------------------------------------
# Normalized instance format returned by every provider fetch function:
#
#   provider           str      "shadeform", "lambda", ...
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


# ---------------------------------------------------------------------------
# Provider: Shadeform
# ---------------------------------------------------------------------------


def _fetch_shadeform(api_key: str) -> list[dict]:
    resp = requests.get(
        "https://api.shadeform.ai/v1/instances/types",
        headers={"X-API-KEY": api_key},
        params={"available": "true"},
        timeout=30,
    )
    resp.raise_for_status()

    out: list[dict] = []
    for inst in resp.json().get("instance_types", []):
        cfg = inst.get("configuration", {})

        if not cfg.get("nvlink", False):
            continue
        if cfg.get("storage_in_gb", 0) < MIN_STORAGE_GB:
            continue
        if not any(r.get("available") for r in inst.get("availability", [])):
            continue

        num_gpus = cfg.get("num_gpus", 0)
        vram = cfg.get("vram_per_gpu_in_gb", 0)
        regions = [
            r["region"] for r in inst.get("availability", []) if r.get("available")
        ]

        out.append(
            {
                "provider": "shadeform",
                "instance_type": inst.get("shade_instance_type", ""),
                "description": inst.get("cloud_instance_type", ""),
                "hourly_price_cents": inst.get("hourly_price", 0),
                "num_gpus": num_gpus,
                "gpu_type": cfg.get("gpu_type", ""),
                "vram_per_gpu_gb": vram,
                "total_vram_gb": num_gpus * vram,
                "storage_gb": cfg.get("storage_in_gb", 0),
                "memory_gb": cfg.get("memory_in_gb", 0),
                "vcpus": cfg.get("vcpus", 0),
                "available_regions": regions,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Provider: Lambda
# ---------------------------------------------------------------------------

_VRAM_RE = re.compile(r"\((\d+)\s*GB")


def _lambda_parse_vram(gpu_description: str) -> int:
    m = _VRAM_RE.search(gpu_description)
    return int(m.group(1)) if m else 0


def _lambda_parse_gpu_type(gpu_description: str) -> str:
    token = gpu_description.split("(")[0].strip().split()
    return token[0] if token else ""


def _lambda_is_sxm(gpu_description: str, name: str) -> bool:
    return "sxm" in f"{gpu_description} {name}".lower()


def _fetch_lambda(api_key: str) -> list[dict]:
    resp = requests.get(
        "https://cloud.lambdalabs.com/api/v1/instance-types",
        auth=(api_key, ""),
        timeout=30,
    )
    resp.raise_for_status()

    out: list[dict] = []
    for name, entry in resp.json().get("data", {}).items():
        itype = entry.get("instance_type", {})
        specs = itype.get("specs", {})
        gpu_desc = itype.get("gpu_description", "")
        regions = entry.get("regions_with_capacity_available", [])

        if not _lambda_is_sxm(gpu_desc, name):
            continue
        if specs.get("storage_gib", 0) < MIN_STORAGE_GB:
            continue
        if not regions:
            continue

        num_gpus = specs.get("gpus", 0)
        vram = _lambda_parse_vram(gpu_desc)
        region_names = [r["name"] for r in regions]

        out.append(
            {
                "provider": "lambda",
                "instance_type": name,
                "description": itype.get("description", ""),
                "hourly_price_cents": itype.get("price_cents_per_hour", 0),
                "num_gpus": num_gpus,
                "gpu_type": _lambda_parse_gpu_type(gpu_desc),
                "vram_per_gpu_gb": vram,
                "total_vram_gb": num_gpus * vram,
                "storage_gb": specs.get("storage_gib", 0),
                "memory_gb": specs.get("memory_gib", 0),
                "vcpus": specs.get("vcpus", 0),
                "available_regions": region_names,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Provider: Targon
# ---------------------------------------------------------------------------

# Targon does not expose per-GPU VRAM; use well-known values.
_TARGON_VRAM_GB: dict[str, int] = {
    "H200": 141,
    "H100": 80,
    "A100": 80,
    "B200": 180,
}


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
        gpu_type_raw = spec.get("gpu_type", "")
        gpu_type = _targon_normalize_gpu_type(gpu_type_raw)
        gpu_count = spec.get("gpu_count", 0)
        available = item.get("available", 0)

        if available <= 0:
            continue

        vram = _TARGON_VRAM_GB.get(gpu_type, 0)
        if not vram:
            continue

        # Targon storage comes from attached persistent volumes, not the
        # instance spec (which often reports 0). Skip the storage filter.
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
                "gpu_type": gpu_type,
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
# Provider registry -- add new providers here
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "shadeform": {"env_key": "SHADEFORM_API_KEY", "fetch": _fetch_shadeform},
    "lambda": {"env_key": "LAMBDA_API_KEY", "fetch": _fetch_lambda},
    "targon": {"env_key": "TARGON_API_KEY", "fetch": _fetch_targon},
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
    """Apply tier logic to a flat list of normalized candidates.

    Returns the same shape as search_all / search_provider.
    """
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
    """Query every provider that has a configured API key, merge, and rank.

    Args:
        keys: optional {provider_name: api_key} overrides.
    """
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
