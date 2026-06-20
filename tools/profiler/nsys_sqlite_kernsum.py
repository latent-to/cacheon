#!/usr/bin/env python3
"""Convert an nsys ``.sqlite`` export into a ``*_kernsum.txt`` the profiler ingests.

The README says "export their text on the GPU box first" — but `nsys stats
--report cuda_gpu_kern_sum` is often NOT run there, while the `.sqlite` (from
`nsys export --type sqlite`) is small enough to pull to the Mac. This closes that
gap with pure stdlib (`sqlite3`): it computes the same whole-run per-kernel
time-share table `parse_nsys_kernsum` expects, using the DEMANGLED kernel name so
the taxonomy in ``ingest.py`` (e.g. `GroupProblemShape` → nvfp4_moe_gemm) matches.

    python3 nsys_sqlite_kernsum.py run.sqlite -o run_kernsum.txt
    # name the output <label>_kernsum.txt; put `prefill`/`decode` in <label> so the
    # findings engine assigns the regime (a prefill big-M kernel must never speak
    # for a decode category — that regime tag is how the report keeps them apart).

Also accepts an nsys `cuda_gpu_kern_sum` CSV (`--from-csv`) and normalizes it to
the same whitespace format (handles the NOTICE/preamble lines nsys prepends).
"""
from __future__ import annotations

import argparse
import csv
import io
import sqlite3
from pathlib import Path


def _row(share: float, total_ns: float, inst: int, name: str) -> str:
    # 9 whitespace fields, name last (parse_nsys_kernsum uses maxsplit=8).
    name = " ".join(str(name).split())
    avg = total_ns / inst if inst else 0.0
    return f"{share:.3f} {total_ns:.0f} {inst} {avg:.1f} {avg:.1f} 0 0 0 {name}"


def from_sqlite(db: Path) -> list[str]:
    con = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        if not cols:
            raise SystemExit(f"{db}: no CUPTI_ACTIVITY_KIND_KERNEL table (not an nsys sqlite export?)")
        namecol = "demangledName" if "demangledName" in cols else "shortName"
        grand = con.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] or 0
        if grand <= 0:
            raise SystemExit(f"{db}: zero total kernel time")
        q = (f"SELECT s.value, COUNT(*), SUM(k.end - k.start) "
             f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.{namecol} "
             f"GROUP BY s.value ORDER BY 3 DESC")
        return [_row(100.0 * tot / grand, float(tot), int(n), name) for name, n, tot in con.execute(q)]
    finally:
        con.close()


def from_csv(path: Path) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    hdr_i = next((i for i, l in enumerate(lines) if l.startswith("Time (%)")), 0)
    out = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[hdr_i:]))):
        try:
            out.append(_row(float(r["Time (%)"]),
                            float(r["Total Time (ns)"].replace(",", "")),
                            int(r["Instances"].replace(",", "")), r["Name"]))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="nsys .sqlite (default) or cuda_gpu_kern_sum CSV with --from-csv")
    ap.add_argument("-o", "--out", type=Path, required=True, help="write <label>_kernsum.txt (label drives regime)")
    ap.add_argument("--from-csv", action="store_true", help="src is an nsys cuda_gpu_kern_sum CSV, not a sqlite")
    args = ap.parse_args()
    rows = from_csv(args.src) if args.from_csv else from_sqlite(args.src)
    args.out.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out} ({len(rows)} kernels)")


if __name__ == "__main__":
    main()
