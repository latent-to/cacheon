#!/usr/bin/env python3
"""Print launch dimensions and key NCU metrics from a raw CSV export."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


FIELDS = [
    "Kernel Name",
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__block_dim_y",
    "launch__block_dim_z",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__waves_per_multiprocessor",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_barriers",
    "launch__occupancy_limit_blocks",
]


def short(name: str, limit: int = 110) -> str:
    name = " ".join(name.split())
    return name if len(name) <= limit else name[: limit - 1] + "..."


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_ncu_launch_dims.py <ncu_raw.csv>")
    path = Path(sys.argv[1])
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        print(f"== kernel {i} ==")
        for field in FIELDS:
            value = row.get(field, "")
            if field == "Kernel Name":
                value = short(value)
            print(f"{field}: {value}")


if __name__ == "__main__":
    main()
