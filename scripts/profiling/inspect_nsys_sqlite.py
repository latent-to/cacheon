#!/usr/bin/env python3
"""Inspect useful context from an Nsight Systems SQLite export."""

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _tables(conn: sqlite3.Connection) -> List[str]:
    return [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _string_map(conn: sqlite3.Connection) -> Dict[int, str]:
    if "StringIds" not in _tables(conn):
        return {}
    cols = _columns(conn, "StringIds")
    id_col = "id" if "id" in cols else cols[0]
    val_col = "value" if "value" in cols else cols[-1]
    return {int(row[0]): str(row[1]) for row in conn.execute(f"SELECT {id_col}, {val_col} FROM StringIds")}


def _text(value: Any, strings: Dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return strings.get(value, str(value))
    return str(value)


def _kernel_name(row: sqlite3.Row, strings: Dict[int, str]) -> str:
    for key in ("demangledName", "shortName", "mangledName", "name"):
        if key in row.keys() and row[key] is not None:
            return _text(row[key], strings)
    return "<unknown>"


def _pick(cols: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def print_tables(conn: sqlite3.Connection) -> None:
    print("TABLES")
    for table in _tables(conn):
        print(f"  {table}: {', '.join(_columns(conn, table))}")


def print_nvtx(conn: sqlite3.Connection, strings: Dict[int, str]) -> None:
    tables = _tables(conn)
    if "NVTX_EVENTS" not in tables:
        print("NVTX_EVENTS missing")
        return
    cols = _columns(conn, "NVTX_EVENTS")
    text_col = _pick(cols, ("text", "message", "name"))
    start_col = _pick(cols, ("start", "startTime"))
    end_col = _pick(cols, ("end", "endTime"))
    if not text_col or not start_col:
        print(f"NVTX_EVENTS unsupported columns: {cols}")
        return

    print("NVTX_MATCHES")
    limit = 100
    count = 0
    for row in conn.execute("SELECT * FROM NVTX_EVENTS"):
        text = _text(row[text_col], strings)
        if not any(token in text for token in ("TIMED", "decode", "bs", "step", "layer")):
            continue
        start = int(row[start_col])
        end = int(row[end_col]) if end_col and row[end_col] is not None else start
        print(f"  {(end - start) / 1e6:10.3f} ms  {text[:220]}")
        count += 1
        if count >= limit:
            break
    if count == 0:
        print("  <none>")


def kernel_totals(conn: sqlite3.Connection, strings: Dict[int, str], limit: int) -> Iterable[Dict[str, Any]]:
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    pid_col = _pick(cols, ("globalPid", "globalPid", "pid", "processId"))
    device_col = _pick(cols, ("deviceId", "device", "device_id"))
    totals: Dict[tuple[str, int | None, int | None], Dict[str, Any]] = {}
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        name = _kernel_name(row, strings)
        duration_ns = int(row["end"]) - int(row["start"])
        pid = int(row[pid_col]) if pid_col and row[pid_col] is not None else None
        device = int(row[device_col]) if device_col and row[device_col] is not None else None
        key = (name, pid, device)
        entry = totals.setdefault(key, {"kernel": name, "pid": pid, "device": device, "count": 0, "time_ns": 0})
        entry["count"] += 1
        entry["time_ns"] += duration_ns
    total_ns = sum(int(row["time_ns"]) for row in totals.values())
    rows = []
    for entry in totals.values():
        rows.append(
            {
                **entry,
                "time_ms": int(entry["time_ns"]) / 1e6,
                "pct": (int(entry["time_ns"]) / total_ns * 100.0) if total_ns else 0.0,
            }
        )
    rows.sort(key=lambda row: row["time_ms"], reverse=True)
    return rows[:limit]


def print_kernel_totals(conn: sqlite3.Connection, strings: Dict[int, str], limit: int) -> None:
    print("KERNEL_TOTALS_BY_PID_DEVICE")
    print("pct,time_ms,count,pid,device,kernel")
    for row in kernel_totals(conn, strings, limit):
        print(
            f"{row['pct']:.6f},{row['time_ms']:.6f},{row['count']},"
            f"{row['pid']},{row['device']},{row['kernel']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row
    strings = _string_map(conn)
    if args.tables:
        print_tables(conn)
    print_nvtx(conn, strings)
    print_kernel_totals(conn, strings, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
