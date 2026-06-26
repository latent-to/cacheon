#!/usr/bin/env python3
"""Summarize CUDA kernel time from an Nsight Systems SQLite export."""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _tables(conn: sqlite3.Connection) -> List[str]:
    return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _string_map(conn: sqlite3.Connection) -> Dict[int, str]:
    if "StringIds" not in _tables(conn):
        return {}
    cols = _columns(conn, "StringIds")
    id_col = "id" if "id" in cols else cols[0]
    val_col = "value" if "value" in cols else cols[-1]
    return {int(row[0]): str(row[1]) for row in conn.execute(f"SELECT {id_col}, {val_col} FROM StringIds")}


def _kernel_name(row: sqlite3.Row, strings: Dict[int, str]) -> str:
    for key in ("demangledName", "shortName", "mangledName", "name"):
        if key in row.keys() and row[key] is not None:
            value = row[key]
            if isinstance(value, int):
                return strings.get(value, str(value))
            return str(value)
    return "<unknown>"


def summarize(path: Path, limit: int) -> Iterable[Dict[str, object]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    tables = _tables(conn)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise RuntimeError(f"{path} has no CUPTI_ACTIVITY_KIND_KERNEL table; tables={tables}")

    strings = _string_map(conn)
    totals: Dict[str, Dict[str, float]] = {}
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        name = _kernel_name(row, strings)
        duration_ns = int(row["end"]) - int(row["start"])
        entry = totals.setdefault(name, {"count": 0.0, "time_ns": 0.0})
        entry["count"] += 1
        entry["time_ns"] += duration_ns

    total_ns = sum(v["time_ns"] for v in totals.values())
    rows = []
    for name, data in totals.items():
        rows.append(
            {
                "kernel": name,
                "count": int(data["count"]),
                "time_ms": data["time_ns"] / 1e6,
                "pct": (data["time_ns"] / total_ns * 100.0) if total_ns else 0.0,
            }
        )
    rows.sort(key=lambda item: item["time_ms"], reverse=True)
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    rows = summarize(args.sqlite, args.limit)
    writer = csv.DictWriter(sys.stdout, fieldnames=["pct", "time_ms", "count", "kernel"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
