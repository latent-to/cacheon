#!/usr/bin/env python3
"""Aggregate SGLang layerwise NVTX ranges from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import ast
import sqlite3
from collections import defaultdict
from pathlib import Path


def module_kind(module: str) -> str:
    if module.endswith(".mlp.experts"):
        return "mlp.experts"
    if module.endswith(".mlp"):
        return "mlp"
    if module.endswith(".self_attn"):
        return "self_attn"
    if module.endswith(".input_layernorm"):
        return "input_layernorm"
    if module.endswith(".post_attention_layernorm"):
        return "post_attention_layernorm"
    if ".layers." in module:
        return "layer"
    return "other"


def parse_module(text: str) -> str | None:
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        data = ast.literal_eval(text)
    except Exception:
        return None
    module = data.get("Module") if isinstance(data, dict) else None
    return module if isinstance(module, str) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    con = sqlite3.connect(args.sqlite)
    totals = defaultdict(lambda: {"count": 0, "time_ns": 0})
    kind_totals = defaultdict(lambda: {"count": 0, "time_ns": 0})

    query = "SELECT start, end, COALESCE(text, jsonText, '') FROM NVTX_EVENTS WHERE end IS NOT NULL"
    for start, end, text in con.execute(query):
        module = parse_module(str(text))
        if module is None:
            continue
        duration_ns = int(end) - int(start)
        if duration_ns <= 0:
            continue
        totals[module]["count"] += 1
        totals[module]["time_ns"] += duration_ns
        kind = module_kind(module)
        kind_totals[kind]["count"] += 1
        kind_totals[kind]["time_ns"] += duration_ns

    print("kind,pct,time_ms,count")
    total_ns = sum(row["time_ns"] for row in kind_totals.values())
    for kind, row in sorted(kind_totals.items(), key=lambda item: item[1]["time_ns"], reverse=True):
        pct = (row["time_ns"] / total_ns * 100.0) if total_ns else 0.0
        print(f"{kind},{pct:.6f},{row['time_ns'] / 1e6:.6f},{row['count']}")

    print()
    print("module,kind,time_ms,count,mean_ms")
    for module, row in sorted(totals.items(), key=lambda item: item[1]["time_ns"], reverse=True)[: args.limit]:
        count = row["count"]
        time_ms = row["time_ns"] / 1e6
        print(f"{module},{module_kind(module)},{time_ms:.6f},{count},{time_ms / count:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
