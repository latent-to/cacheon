#!/usr/bin/env python3
"""Classify Nsight Systems cuda_gpu_kern_sum CSV rows for Qwen3.5 profiling."""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


RULES: list[tuple[str, re.Pattern[str]]] = [
    ("TP all-reduce fused/multimem", re.compile(r"(multimem|one[_-]?shot|oneshot).*all[_-]?reduce|all[_-]?reduce.*(multimem|one[_-]?shot|oneshot)", re.I)),
    ("TP all-reduce NCCL", re.compile(r"nccl|all[_-]?reduce", re.I)),
    ("GDN decode", re.compile(r"fused_recurrent.*packed_decode|packed_decode.*gated_delta", re.I)),
    ("GDN prefill", re.compile(r"chunk_.*gated_delta|chunk_fwd_o|chunk_scaled_dot_kkt|wy_fast", re.I)),
    ("GDN glue", re.compile(r"fused_gdn_gating|norm_gate|chunk_local_cumsum|cumsum|solve_tril", re.I)),
    ("causal_conv1d", re.compile(r"causal_conv1d|conv1d", re.I)),
    ("Full attention", re.compile(r"trtllm.*mha|mha.*trtllm|attention|flash[_-]?attn|fa4|xqa", re.I)),
    ("MoE", re.compile(r"trtllm_fp4_block_scale_moe|moe|topk|routing|route|gather|scatter|expert", re.I)),
    ("NVFP4 GEMM/projections", re.compile(r"nvfp4|fp4|cutlass|gemm|qkv|o_proj|gate_proj|up_proj|down_proj", re.I)),
    ("RMSNorm/RoPE", re.compile(r"rmsnorm|layernorm|rope|rotary", re.I)),
]


def clean_float(value: str | None) -> float:
    if not value:
        return 0.0
    value = value.strip().replace(",", "").replace("%", "")
    try:
        return float(value)
    except ValueError:
        return 0.0


def classify(name: str) -> str:
    for label, pattern in RULES:
        if pattern.search(name):
            return label
    return "Other"


def pick_field(fields: list[str], candidates: list[str]) -> str:
    lowered = {f.lower(): f for f in fields}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for field in fields:
        low = field.lower()
        if any(cand.lower() in low for cand in candidates):
            return field
    raise KeyError(f"none of {candidates} found in CSV fields: {fields}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: classify_nsys_kernels.py <cuda_gpu_kern_sum.csv> [out_prefix]")
    path = Path(sys.argv[1])
    prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_suffix("")
    with path.open(newline="") as f:
        reader = csv.DictReader(row for row in f if row.strip() and not row.startswith("#"))
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no rows in {path}")

    fields = list(rows[0])
    name_field = pick_field(fields, ["Name", "Kernel Name"])
    time_field = pick_field(fields, ["Total Time (ns)", "Total Time", "Time"])
    inst_field = None
    try:
        inst_field = pick_field(fields, ["Instances", "Calls", "Count"])
    except KeyError:
        pass

    detailed = []
    by_cat: dict[str, dict[str, float]] = defaultdict(lambda: {"time": 0.0, "instances": 0.0})
    total = 0.0
    for row in rows:
        name = row.get(name_field, "")
        category = classify(name)
        time_value = clean_float(row.get(time_field))
        instances = clean_float(row.get(inst_field)) if inst_field else 0.0
        total += time_value
        by_cat[category]["time"] += time_value
        by_cat[category]["instances"] += instances
        out = dict(row)
        out["Optima Category"] = category
        detailed.append(out)

    summary_path = prefix.with_name(prefix.name + "_category_summary.csv")
    detail_path = prefix.with_name(prefix.name + "_categorized.csv")
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["category", "total_time", "pct_gpu_kernel_time", "instances"])
        writer.writeheader()
        for category, values in sorted(by_cat.items(), key=lambda x: x[1]["time"], reverse=True):
            writer.writerow(
                {
                    "category": category,
                    "total_time": values["time"],
                    "pct_gpu_kernel_time": 100.0 * values["time"] / total if total else 0.0,
                    "instances": values["instances"],
                }
            )
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)

    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
