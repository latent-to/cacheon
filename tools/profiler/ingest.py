#!/usr/bin/env python3
"""Ingest a directory of GPU-profiling artifacts into one normalized dataset.

This is the front half of the profiler pipeline: it knows how to read every
artifact type we produce on a profiling box and flatten them into a single
JSON-serialisable dict. ``findings.py`` turns that dict into verdicts; the
``report.py`` renders it as a self-contained HTML dashboard.

Design constraints (learned the hard way):
  * **stdlib only** — the Mac checkout has no pandas and (usually) no NVIDIA
    CLIs, so ``.ncu-rep`` / ``.nsys-rep`` binaries are opaque. We read only
    their *text exports* (``_details.txt`` / ``_raw.csv`` / ``_kernsum.txt``).
  * **be honest about junk** — some ncu captures come back all-NaN (cluster /
    Cluster-Launch-Control kernels that ncu kernel-replay can't count) and some
    ``_summary.txt`` files are ``==ERROR==`` (wrong --page). We detect and flag
    those rather than silently averaging garbage.
  * **model-agnostic** — the category regexes are a config (``KERNEL_CATS``),
    overridable, so this works on the next model's profiles too, not just
    Qwen3.5.

CLI:  python3 ingest.py <datadir> [-o dataset.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sqlite3
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from taxonomy import DISPLAY, KERNEL_CATS, categorize, taxonomy_json
    from architecture import infer_architecture
except ImportError:  # pragma: no cover - package import fallback
    from .taxonomy import DISPLAY, KERNEL_CATS, categorize, taxonomy_json
    from .architecture import infer_architecture


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _num(tok: str) -> float | None:
    """Parse one ncu metric token; '(!) nan' / 'no data' / 'n/a' -> None."""
    tok = tok.strip()
    if not tok or tok.lower() in ("nan", "no data", "n/a", "(!) nan", "<null>"):
        return None
    m = re.findall(r"[-+]?\d[\d,]*\.?\d*", tok)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def _pct(num: float, den: float) -> float:
    return 0.0 if den <= 0 else 100.0 * num / den


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mtime_iso(paths: Iterable[Path]) -> str | None:
    mtimes = []
    for p in paths:
        try:
            mtimes.append(p.stat().st_mtime)
        except OSError:
            pass
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# 1. torch profiler Chrome traces (*.trace.json.gz) — the clean DECODE source
# --------------------------------------------------------------------------- #
def _trace_label(annotations: collections.Counter, path: Path) -> str:
    names = set(annotations)
    if any("DRAFT_EXTEND" in n or "TARGET_VERIFY" in n for n in names):
        return "mtp_on"
    if any("DECODE" in n for n in names):
        return "mtp_off"
    return path.stem


def _trace_cache_path(path: Path) -> Path:
    st = path.stat()
    cdir = path.parent / ".profiler_cache"
    return cdir / f"{path.name}-{st.st_size}-{int(st.st_mtime)}.json"


def parse_torch_trace(path: Path, kernel_cats=KERNEL_CATS, use_cache: bool = True) -> dict:
    """Bucket GPU kernel time by category for one torch trace.

    Returns a per-trace summary. Kernel ``dur`` in Chrome traces is microseconds.
    Torch traces are huge (~1 GB JSON each); the summary is tiny, so we cache it
    in ``<datadir>/.profiler_cache/`` keyed by (size, mtime) — re-runs are instant
    while iterating on findings/report. Best-effort: cache errors are ignored.
    """
    if use_cache:
        cp = _trace_cache_path(path)
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except (OSError, ValueError):
                pass
    summary = _parse_torch_trace_uncached(path, kernel_cats)
    if use_cache:
        try:
            cp = _trace_cache_path(path)
            cp.parent.mkdir(exist_ok=True)
            cp.write_text(json.dumps(summary))
        except OSError:
            pass
    return summary


def _parse_torch_trace_uncached(path: Path, kernel_cats=KERNEL_CATS) -> dict:
    with gzip.open(path, "rt", errors="replace") as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    anns: collections.Counter = collections.Counter()
    cats: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])   # cat -> [us, count]
    per_kernel: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    total_us = 0.0
    launches = 0
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat_field = str(e.get("cat", "")).lower()
        if cat_field in ("user_annotation", "gpu_user_annotation"):
            anns[str(e.get("name", "?"))] += 1
            continue
        if cat_field not in ("kernel", "gpu_op", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(e.get("dur") or 0.0)
        name = str(e.get("name", "?"))
        c = categorize(name, kernel_cats)
        total_us += dur
        launches += 1
        cats[c][0] += dur
        cats[c][1] += 1
        per_kernel[name][0] += dur
        per_kernel[name][1] += 1

    rank = (re.search(r"TP-?(\d+)", path.name) or [None, "?"])[1]
    cat_rows = {
        c: {"us": v[0], "pct": _pct(v[0], total_us), "count": v[1]}
        for c, v in cats.items()
    }
    top = sorted(
        ({"name": n[:160], "cat": categorize(n, kernel_cats), "us": v[0],
          "pct": _pct(v[0], total_us), "count": v[1]} for n, v in per_kernel.items()),
        key=lambda r: -r["us"],
    )[:60]
    return {
        "file": path.name,
        "label": _trace_label(anns, path),
        "rank": f"TP{rank}",
        "total_us": total_us,
        "launches": launches,
        "cats": cat_rows,
        "top_kernels": top,
    }


# --------------------------------------------------------------------------- #
# 2. ncu text exports (_details.txt primary, _raw.csv for waves/registers)
# --------------------------------------------------------------------------- #
# header line ends with ", Context N, Stream N, Device N, CC X.Y"
_NCU_HDR = re.compile(r"^\s*(\S.*?)\s+\([\d,\s]+\)x\([\d,\s]+\),\s*Context\s+\d+,\s*Stream\s+\d+", re.I)
_NCU_METRICS = {
    "comp": "Compute (SM) Throughput",
    "mem": "Memory Throughput",
    "dram": "DRAM Throughput",
    "dur_us": "Duration",
    "occ": "Achieved Occupancy",
    "warps": "Achieved Active Warps Per SM",
    "l2_hit": "L2 Hit Rate",
    "l1_hit": "L1/TEX Hit Rate",
}


def _ncu_metric_line(line: str, label: str) -> float | None:
    """A metric row is '<Label> <unit> <value>'. Match label as a prefix of the
    stripped line so 'Memory Throughput' doesn't also catch 'L2 ... Throughput'."""
    s = line.strip()
    if not s.startswith(label):
        return None
    rest = s[len(label):]
    toks = rest.split()
    if not toks:
        return None
    val = _num(toks[-1])
    if val is None:
        return None
    if label == "Duration" and toks:
        unit = toks[0].lower()
        if unit == "ns":
            return val / 1000.0
        if unit == "ms":
            return val * 1000.0
        if unit == "s":
            return val * 1_000_000.0
    return val


_CLUSTER_NAME = re.compile(r"ClusterKernel|cluster_launch|_cluster_", re.I)


def parse_ncu_details_text(text: str, kernel_cats=KERNEL_CATS) -> list[dict]:
    """One row per profiled kernel.

    Flags two kinds of untrustworthy capture so the findings layer never
    manufactures a phantom win from them:
      * ``valid=False`` — every metric came back NaN (ncu kernel-replay could
        not count it at all).
      * ``clc=True`` / ``cluster=True`` — the kernel uses thread-block clusters
        / Cluster-Launch-Control, which DEPRESSES the launched cluster/block/
        warp counts, so occupancy & throughput read artificially low (a fake
        "latency-bound, just fuse it" signal). These are vendor cluster kernels.
    """
    rows: list[dict] = []
    cur: dict | None = None

    def _flush():
        if cur is not None:
            cur["valid"] = any(cur.get(k) is not None for k in ("comp", "mem", "dram", "occ"))
            cur["cat"] = categorize(cur["kernel"], kernel_cats)
            cur.setdefault("clc", False)
            # `cluster` is STRUCTURAL (by name): a vendor cluster-dispatch kernel you
            # can't fuse (e.g. routingIndicesClusterKernel). `clc` is the CLC *warning*:
            # it corrupts occupancy/launch counts but NOT throughput — an FP4 cutlass
            # GEMM (2cta clusters) has clc=True yet its 80% DRAM reading is valid.
            # Keep them separate so we don't exclude a real memory-bound floor.
            cur["cluster"] = bool(_CLUSTER_NAME.search(cur["kernel"]))
            rows.append(cur)

    for line in text.splitlines():
        h = _NCU_HDR.match(line)
        if h:
            _flush()
            cur = {"kernel": h.group(1).strip()[:160]}
            continue
        if cur is None:
            continue
        if "Cluster Launch Control" in line or "CLC" in line and "feature enabled" in line:
            cur["clc"] = True
            continue
        for key, label in _NCU_METRICS.items():
            if key not in cur:
                v = _ncu_metric_line(line, label)
                if v is not None:
                    cur[key] = v
    _flush()
    return rows


def parse_ncu_details(path: Path, kernel_cats=KERNEL_CATS) -> list[dict]:
    return parse_ncu_details_text(path.read_text(errors="replace"), kernel_cats)


# raw-CSV SpeedOfLight columns -> the same metric keys parse_ncu_details produces.
# This lets a capture that only exported `--page raw --csv` (no `--page details`
# text) still yield a full bound-type — which is the common case for our gate3
# microbench batteries (MoE / glue / prefill), where only *_raw.csv exists.
_NCU_RAW_SOL = {
    "comp": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dur_us": "gpu__time_duration.sum",
    "warps": "sm__warps_active.avg.per_cycle_active",
}
_DUR_TO_US = {"us": 1.0, "usecond": 1.0, "ns": 1e-3, "nsecond": 1e-3,
              "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}


def parse_ncu_raw_text(text: str) -> list[dict]:
    """Parse an ncu ``--page raw --csv`` export into one row PER KERNEL NAME.

    Beyond the launch metrics ncu *details* omits (waves, registers), this also
    pulls the SpeedOfLight ratios (compute/memory/DRAM throughput, achieved
    occupancy, duration) straight from the raw CSV columns, so a raw-only capture
    is self-sufficient for a bound-type — no ``_details.txt`` required. ncu emits
    one row per profiled launch; we keep the longest-duration instance per kernel
    name as the representative (decode steady state is the same shape every step).
    """
    by_name: dict[str, dict] = {}
    try:
        with io.StringIO(text, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            units = next(reader, [])  # ncu raw csv has a units row
            idx = {name: i for i, name in enumerate(header)}
            kname_i = idx.get("Kernel Name", 4)
            dur_unit = (units[idx["gpu__time_duration.sum"]].strip().lower()
                        if "gpu__time_duration.sum" in idx and idx["gpu__time_duration.sum"] < len(units) else "us")

            def col(row, name):
                i = idx.get(name)
                return _num(row[i]) if i is not None and i < len(row) else None

            for row in reader:
                if not row or kname_i >= len(row) or not row[kname_i].strip():
                    continue
                rec = {
                    "kernel": row[kname_i][:160],
                    "waves": col(row, "launch__waves_per_multiprocessor"),
                    "regs": col(row, "launch__registers_per_thread"),
                    "grid": row[idx["Grid Size"]] if "Grid Size" in idx else "",
                    "block": row[idx["Block Size"]] if "Block Size" in idx else "",
                }
                for key, colname in _NCU_RAW_SOL.items():
                    rec[key] = col(row, colname)
                if rec.get("dur_us") is not None:
                    rec["dur_us"] *= _DUR_TO_US.get(dur_unit, 1.0)
                prev = by_name.get(rec["kernel"])
                if prev is None or (rec.get("dur_us") or 0) > (prev.get("dur_us") or 0):
                    by_name[rec["kernel"]] = rec
    except (OSError, StopIteration):
        return []
    return list(by_name.values())


def parse_ncu_raw(path: Path) -> list[dict]:
    return parse_ncu_raw_text(path.read_text(errors="replace"))


def merge_ncu(details: list[dict], raw: list[dict]) -> list[dict]:
    """Augment details rows with waves/registers from the matching raw row.
    Match on kernel name first; fall back to positional when names line up 1:1."""
    raw_by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for r in raw:
        raw_by_name[r["kernel"]].append(r)
    for i, d in enumerate(details):
        cand = raw_by_name.get(d["kernel"])
        rr = cand.pop(0) if cand else (raw[i] if i < len(raw) and len(raw) == len(details) else None)
        if rr:
            # details (the SoL text page) is authoritative for comp/mem/dram/occ;
            # raw fills anything details lacks plus waves/regs/grid/block.
            for k in ("waves", "regs", "grid", "block", "comp", "mem", "dram", "occ", "dur_us"):
                if rr.get(k) is not None and rr.get(k) != "":
                    d.setdefault(k, rr[k])
    return details


def parse_ncu_log_text(txt: str, file_name: str = "") -> dict:
    exits = re.findall(r"\bEXIT=(\d+)", txt)
    return {
        "file": file_name,
        "profiles": len(re.findall(r"==PROF== Profiling", txt)),
        "launchfailed": "LaunchFailed" in txt,
        "errors": len(re.findall(r"^==ERROR==", txt, re.M)),
        "report": (re.findall(r"==PROF== Report:\s+(\S+)", txt) or [""])[-1],
        "exit": int(exits[-1]) if exits else None,
    }


def parse_ncu_log(path: Path) -> dict:
    return parse_ncu_log_text(path.read_text(errors="replace"), path.name)


# --------------------------------------------------------------------------- #
# 3. nsys kernel summary text exports (*_kernsum.txt) — whole-run (mostly prefill)
# --------------------------------------------------------------------------- #
def _nsys_kernsum_from_rows(rows: list[tuple[float, float, int, str]], file_name: str,
                            kernel_cats=KERNEL_CATS, note: str | None = None) -> dict:
    cats: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    top: list[dict] = []
    total_ns = 0.0
    for share, tot, inst, name in rows:
        c = categorize(name, kernel_cats)
        cats[c][0] += tot
        cats[c][1] += inst
        total_ns += tot
        top.append({"share": share, "count": inst, "cat": c, "name": name[:160]})
    cat_rows = {
        c: {"us": v[0] / 1e3, "pct": _pct(v[0], total_ns), "count": v[1]}
        for c, v in cats.items()
    }
    return {
        "file": file_name,
        "total_ms": total_ns / 1e6,
        "cats": cat_rows,
        "top_kernels": sorted(top, key=lambda r: -r["share"])[:40],
        "note": note or "whole-run (~prefill-dominated); not steady-decode attribution",
    }


def parse_nsys_kernsum_text(text: str, file_name: str, kernel_cats=KERNEL_CATS) -> dict:
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=8)
        if len(parts) < 9:
            continue
        try:
            share = float(parts[0])
            tot = float(parts[1].replace(",", ""))
            inst = int(parts[2].replace(",", ""))
        except ValueError:
            continue
        name = parts[8]
        rows.append((share, tot, inst, name))
    return _nsys_kernsum_from_rows(rows, file_name, kernel_cats)


def parse_nsys_kernsum(path: Path, kernel_cats=KERNEL_CATS) -> dict:
    return parse_nsys_kernsum_text(path.read_text(errors="replace"), path.name, kernel_cats)


def parse_nsys_kernsum_csv(path: Path, kernel_cats=KERNEL_CATS) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    hdr_i = next((i for i, line in enumerate(lines) if line.startswith("Time (%)")), None)
    if hdr_i is None:
        return _nsys_kernsum_from_rows([], path.name, kernel_cats)
    rows = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[hdr_i:]))):
        try:
            rows.append((
                float(row["Time (%)"]),
                float(row["Total Time (ns)"].replace(",", "")),
                int(row["Instances"].replace(",", "")),
                row["Name"],
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return _nsys_kernsum_from_rows(rows, path.name.replace("_cuda_gpu_kern_sum.csv", "_kernsum.txt"),
                                  kernel_cats, note="nsys cuda_gpu_kern_sum CSV export")


def parse_nsys_sqlite(path: Path, kernel_cats=KERNEL_CATS) -> tuple[dict | None, dict | None]:
    """Parse an nsys SQLite export into kernsum + compact timeline summaries."""
    try:
        con = sqlite3.connect(str(path))
    except sqlite3.Error:
        return None, None
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        if not cols:
            return None, None
        namecol = "demangledName" if "demangledName" in cols else "shortName"
        grand = con.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] or 0
        if grand <= 0:
            return None, None
        rows = []
        q = (f"SELECT s.value, COUNT(*), SUM(k.end - k.start) "
             f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.{namecol} "
             f"GROUP BY s.value ORDER BY 3 DESC")
        for name, count, total in con.execute(q):
            rows.append((100.0 * float(total) / grand, float(total), int(count), str(name)))
        kernsum = _nsys_kernsum_from_rows(
            rows, path.name.replace(".sqlite", "_kernsum.txt"), kernel_cats,
            note="derived locally from nsys SQLite export",
        )
        timeline = _parse_nsys_sqlite_timeline(con, path.name, namecol, kernel_cats)
        return kernsum, timeline
    except sqlite3.Error:
        return None, None
    finally:
        con.close()


def _parse_nsys_sqlite_timeline(con: sqlite3.Connection, file_name: str, namecol: str,
                                kernel_cats=KERNEL_CATS) -> dict:
    by_device: dict[str, dict] = {}
    by_stream: dict[str, dict] = {}
    graph_ids: set[int] = set()
    top_dims: dict[str, dict] = {}
    q = (f"SELECT k.start, k.end, k.deviceId, k.streamId, k.globalPid, k.gridX, k.gridY, k.gridZ, "
         f"k.blockX, k.blockY, k.blockZ, k.registersPerThread, k.dynamicSharedMemory, "
         f"k.graphId, s.value "
         f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.{namecol}")
    starts, ends = [], []
    for row in con.execute(q):
        start, end, dev, stream, pid, gx, gy, gz, bx, by, bz, regs, smem, graph_id, name = row
        dur = max(0, int(end) - int(start))
        starts.append(int(start)); ends.append(int(end))
        dev_key = str(dev)
        stream_key = f"d{dev}:s{stream}"
        for bucket, key in ((by_device, dev_key), (by_stream, stream_key)):
            rec = bucket.setdefault(key, {"kernel_ns": 0, "launches": 0})
            rec["kernel_ns"] += dur
            rec["launches"] += 1
        if graph_id is not None:
            graph_ids.add(int(graph_id))
        cat = categorize(str(name), kernel_cats)
        cur = top_dims.get(cat)
        if cur is None or dur > cur.get("dur_ns", 0):
            top_dims[cat] = {
                "cat": cat,
                "kernel": str(name)[:160],
                "dur_ns": dur,
                "grid": [gx, gy, gz],
                "block": [bx, by, bz],
                "regs": regs,
                "dynamic_smem": smem,
                "pid": pid,
                "device": dev,
                "stream": stream,
            }
    span_ns = max(ends) - min(starts) if starts and ends else 0
    total_kernel_ns = sum(v["kernel_ns"] for v in by_device.values())
    def finish(bucket):
        out = []
        for key, rec in sorted(bucket.items()):
            span_gap = max(0, span_ns - rec["kernel_ns"]) if span_ns else 0
            out.append({
                "id": key,
                "kernel_ms": rec["kernel_ns"] / 1e6,
                "launches": rec["launches"],
                "span_gap_ms": span_gap / 1e6,
                "busy_pct_of_span": _pct(rec["kernel_ns"], span_ns),
            })
        return out
    return {
        "file": file_name,
        "span_ms": span_ns / 1e6,
        "kernel_ms_sum": total_kernel_ns / 1e6,
        "devices": finish(by_device),
        "streams": sorted(finish(by_stream), key=lambda r: -r["kernel_ms"])[:40],
        "graph_ids": sorted(graph_ids)[:20],
        "n_graph_ids": len(graph_ids),
        "top_launch_dims": sorted(top_dims.values(), key=lambda r: -r["dur_ns"])[:40],
        "note": "Derived from nsys SQLite; span gaps are per-device/stream wall-clock gaps, not proof of CPU idle by themselves.",
    }


# --------------------------------------------------------------------------- #
# 4. serve_load2 e2e / ceiling logs (e2e_*.txt, ceil_*.txt)
# --------------------------------------------------------------------------- #
_RESULT_RE = re.compile(
    r"\[RESULT\] conc=(?P<conc>\d+) in~(?P<inlen>\d+) out=(?P<outlen>\d+).*?"
    r"AGG output tok/s \(steady\) = (?P<agg>[0-9.]+).*?"
    r"TTFT s: p50=(?P<ttft50>[0-9.]+) p99=(?P<ttft99>[0-9.]+).*?"
    r"per-req decode tok/s: p50=(?P<dec50>[0-9.]+).*?tokens/chunk=(?P<tpc>[0-9.]+).*?"
    r"steady tokens=(?P<toks>[0-9]+) errors=(?P<errs>\d+)",
    re.S,
)

# file-stem -> (config tag, kind)
_E2E_CONFIG = {
    "e2e_mtp_off": ("mtp_off", "sweep"), "e2e_mtp_off2": ("mtp_off_r2", "sweep"),
    "e2e_mtp_on": ("mtp_on", "sweep"), "e2e_noAR": ("no_all_reduce", "sweep"),
    "e2e_nograph": ("no_cuda_graph", "sweep"),
    "ceil_base": ("ceiling_none", "ceiling"), "ceil_moe": ("ceiling_noop_moe", "ceiling"),
    "ceil_gdn": ("ceiling_noop_gdn", "ceiling"), "ceil_attn": ("ceiling_noop_attn", "ceiling"),
}


def _serve_config(stem: str) -> tuple[str, str, str | None]:
    """Map historical and V2 serve_load2 filenames to a normalized config.

    Historical files were named ``ceil_moe.txt``. The V2 pod driver calls the
    generic ``sweep`` helper with tags like ``ceil_moe_r1``, which produces
    ``e2e_ceil_moe_r1.txt``. Normalize both shapes so ceiling analysis does not
    disappear on the next clean run.
    """
    if stem in _E2E_CONFIG:
        cfg, kind = _E2E_CONFIG[stem]
        return cfg, kind, None
    if stem in ("ceil_base2", "e2e_ceil_base2"):
        return "ceiling_none", "ceiling", "r2"
    if stem in ("ceil_base_long", "e2e_ceil_base_long"):
        return "ceiling_none", "ceiling", "long"
    # base/none + Nemotron noop tags (moe/gdn/attn) + MiniMax-M3 noop tags
    # (step3 = decode-attn / A, step1 = indexer / C, topk = B).
    m = re.match(
        r"(?:e2e_)?ceil_(base|none|moe|gdn|attn|step1|step3|topk|indexer|decode_attn)"
        r"(?:_(r\d+))?$", stem)
    if m:
        op, rep = m.groups()
        cfg = "ceiling_none" if op in ("base", "none") else f"ceiling_noop_{op}"
        return cfg, "ceiling", rep
    cfg = stem[4:] if stem.startswith("e2e_") else stem
    return _E2E_CONFIG.get(stem, (cfg, "sweep")) + (None,)


def parse_serve_log(path: Path) -> list[dict]:
    stem = path.stem
    tag, kind, rep = _serve_config(stem)
    rows = []
    for m in _RESULT_RE.finditer(path.read_text(errors="replace")):
        g = m.groupdict()
        rows.append({
            "file": path.name, "config": tag, "kind": kind, "replicate": rep,
            "conc": int(g["conc"]), "in_len": int(g["inlen"]), "out_len": int(g["outlen"]),
            "agg_toks": float(g["agg"]), "ttft_p50": float(g["ttft50"]), "ttft_p99": float(g["ttft99"]),
            "decode_p50": float(g["dec50"]), "tokens_per_chunk": float(g["tpc"]),
            "steady_tokens": int(g["toks"]), "errors": int(g["errs"]),
        })
    return rows


# --------------------------------------------------------------------------- #
# 5. SGLang runtime provenance from logs
# --------------------------------------------------------------------------- #
_LINEAR_ATTN_BACKEND_RE = re.compile(
    r"Linear attention kernel backend:\s*decode=([^,\s]+),\s*prefill=([^\s]+)"
)
_GDN_DISPATCHER_RE = re.compile(
    r"GDN kernel dispatcher:\s*decode=([^,\s]+),\s*extend=([^,\s]+),\s*verify=([^,\s]+)\s+packed_decode=(True|False)"
)
_SERVER_ARG_KEYS = (
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "quantization",
    "kv_cache_dtype",
    "tp_size",
    "ep_size",
    "pp_size",
    "context_length",
    "page_size",
    "chunked_prefill_size",
    "attention_backend",
    "decode_attention_backend",
    "prefill_attention_backend",
    "fp4_gemm_runner_backend",
    "moe_runner_backend",
    "mamba_backend",
    "mamba_ssm_dtype",
    "linear_attn_backend",
    "linear_attn_decode_backend",
    "linear_attn_prefill_backend",
    "enable_flashinfer_allreduce_fusion",
    "enforce_disable_flashinfer_allreduce_fusion",
    "enable_aiter_allreduce_fusion",
    "disable_custom_all_reduce",
    "enable_nccl_nvls",
    "enable_symm_mem",
    "disable_shared_experts_fusion",
    "enforce_shared_experts_fusion",
    "enable_fused_moe_sum_all_reduce",
    "disable_cuda_graph",
    "disable_piecewise_cuda_graph",
    "enable_deterministic_inference",
)
_SERVER_ARG_VALUE_RE = r"(?:'[^']*'|None|True|False|[-+]?\d+(?:\.\d+)?|[A-Za-z0-9_.:/+-]+)"
_NORMALIZED_WARNINGS = (
    (
        "SM100+ defaulted --linear-attn-decode-backend to flashinfer",
        "defaulting --linear-attn-decode-backend to flashinfer",
    ),
    (
        "FlashInfer TRTLLM MoE auto-disabled shared-experts fusion",
        "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set.",
    ),
)


def _arg_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return raw[1:-1]
    return raw


def _counter_entries(counter: dict, key_names: tuple[str, ...] | None = None) -> list[dict]:
    rows = []
    for key, files in counter.items():
        if not isinstance(key, tuple):
            key = (key,)
        row = {"count": len(files), "files": sorted(files)[:6]}
        if key_names:
            row.update({name: val for name, val in zip(key_names, key)})
        else:
            row["value"] = key[0]
        rows.append(row)
    rows.sort(key=lambda r: (-r["count"], tuple(str(v) for k, v in sorted(r.items()) if k != "files")))
    return rows


def _parse_sglang_observations(datadir: Path, files: list[Path] | None = None) -> dict:
    """Extract runtime backend choices from SGLang logs.

    The profile math says what ran; these log lines explain *why* it ran. That
    matters for frontier-style perf work because a 3% bucket can be a real
    kernel opportunity or just a missed SGLang flag.
    """
    linear: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    gdn: dict[tuple[str, str, str, str], set[str]] = collections.defaultdict(set)
    args: dict[str, dict[str, set[str]]] = {
        k: collections.defaultdict(set) for k in _SERVER_ARG_KEYS
    }
    warnings: dict[str, set[str]] = collections.defaultdict(set)

    log_files = [p for p in (files or list(datadir.iterdir())) if p.suffix in (".log", ".txt")]
    for p in sorted(log_files):
        txt = p.read_text(errors="replace")
        for m in _LINEAR_ATTN_BACKEND_RE.finditer(txt):
            linear[(m.group(1), m.group(2))].add(p.name)
        for m in _GDN_DISPATCHER_RE.finditer(txt):
            gdn[(m.group(1), m.group(2), m.group(3), m.group(4))].add(p.name)
        for key in _SERVER_ARG_KEYS:
            rx = re.compile(rf"\b{re.escape(key)}=({_SERVER_ARG_VALUE_RE})")
            for m in rx.finditer(txt):
                args[key][_arg_value(m.group(1))].add(p.name)
        for label, needle in _NORMALIZED_WARNINGS:
            if needle in txt:
                warnings[label].add(p.name)

    server_args = {
        k: _counter_entries(v)
        for k, v in args.items()
        if v
    }
    return {
        "linear_attn_backends": _counter_entries(linear, ("decode", "prefill")),
        "gdn_dispatchers": _counter_entries(
            gdn, ("decode_kernel", "extend_kernel", "verify_kernel", "packed_decode")
        ),
        "server_args": server_args,
        "warnings": _counter_entries(warnings, ("message",)),
    }


# --------------------------------------------------------------------------- #
# 6. v2 metadata: manifests, recursive discovery, telemetry, context inference
# --------------------------------------------------------------------------- #
_PROFILE_ARTIFACT_RE = re.compile(
    r"(\.trace\.json\.gz$|_kernsum\.txt$|_details\.txt$|_raw\.csv$|\.sqlite$|\.nsys-rep$|\.ncu-rep$|^e2e_|^ceil_|gpu_telemetry)",
    re.I,
)
_SKIP_DIRS = {".git", ".profiler_cache", "__pycache__"}


def _load_manifest(datadir: Path) -> dict:
    p = datadir / "profile_manifest.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {"_error": f"could not parse {p.name}"}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _has_profile_artifacts(path: Path) -> bool:
    try:
        return any(p.is_file() and _PROFILE_ARTIFACT_RE.search(p.name) for p in path.iterdir())
    except OSError:
        return False


def _discover_files(datadir: Path) -> list[Path]:
    """Controlled recursive discovery.

    Generated report/cache dirs are skipped. Child dirs that look like complete
    independent profile packs are skipped when the root already has top-level
    artifacts; this prevents ``profiles_b300/nemotron`` from being mixed into a
    root report while still letting export/bundle dirs such as ``m3_export2`` be
    read.
    """
    root_has_artifacts = _has_profile_artifacts(datadir)
    out: list[Path] = []

    def walk(d: Path, is_root: bool = False):
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        for p in entries:
            if p.is_dir():
                if p.name in _SKIP_DIRS or p.name.startswith("profiler_out"):
                    continue
                if not is_root and root_has_artifacts and _has_profile_artifacts(p):
                    if not re.search(r"export|bundle|results|artifacts", p.name, re.I):
                        continue
                walk(p, False)
            elif p.is_file():
                out.append(p)

    walk(datadir, True)
    return out


def _by_name(files: list[Path], suffix: str | None = None, pattern: str | None = None) -> list[Path]:
    out = []
    rx = re.compile(pattern, re.I) if pattern else None
    for p in files:
        if suffix and not p.name.endswith(suffix):
            continue
        if rx and not rx.search(p.name):
            continue
        out.append(p)
    return sorted(out)


def _first_server_arg(sglang_obs: dict, key: str) -> str | None:
    rows = sglang_obs.get("server_args", {}).get(key, [])
    if not rows:
        return None
    return str(rows[0].get("value"))


def _infer_model_id(files: list[Path], sglang_obs: dict, manifest: dict) -> str:
    m = (manifest.get("model") or {}).get("id") or (manifest.get("model") or {}).get("model_id")
    if m:
        return str(m)
    model_path = _first_server_arg(sglang_obs, "model_path")
    if model_path:
        return model_path
    # The server_args regex is intentionally conservative and does not list every
    # possible SGLang field, so use a lightweight log scan for common shapes.
    pats = [
        re.compile(r"model_path='([^']+)'"),
        re.compile(r"--model-path\s+(\S+)"),
        re.compile(r"--model\s+(\S+)"),
        re.compile(r"(?:MiniMaxAI|nvidia|Qwen|deepseek-ai|sgl-project)/[A-Za-z0-9_.:/+-]+"),
    ]
    for p in files:
        if p.suffix not in (".log", ".txt"):
            continue
        txt = p.read_text(errors="replace")[:200_000]
        for rx in pats:
            mm = rx.search(txt)
            if mm:
                return mm.group(1) if mm.lastindex else mm.group(0)
    low = " ".join(x.name for x in files).lower()
    if "nemotron" in low:
        return "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    if "minimax" in low or "m3_" in low:
        return "MiniMaxAI/MiniMax-M3"
    if "qwen35" in low or "397b" in low:
        return "nvidia/Qwen3.5-397B-A17B-NVFP4"
    if "deepseek" in low:
        return "DeepSeek"
    return "unknown"


def _infer_runtime(datadir: Path, sglang_obs: dict, manifest: dict, files: list[Path], telemetry: dict) -> dict:
    server_args = sglang_obs.get("server_args", {})
    backends = {}
    for key, rows in server_args.items():
        if rows:
            backends[key] = rows[0].get("value")
    runtime = {
        "engine": "sglang" if sglang_obs.get("server_args") or sglang_obs.get("warnings") else "unknown",
        "versions": {"sglang": "unknown", "nccl": "unknown"},
        "hardware": {
            "gpus": telemetry.get("gpu_count"),
            "gpu_names": telemetry.get("gpu_names", []),
        },
        "parallelism": {
            "tp": _first_server_arg(sglang_obs, "tp_size") or _first_server_arg(sglang_obs, "tp"),
            "ep": _first_server_arg(sglang_obs, "ep_size"),
            "pp": _first_server_arg(sglang_obs, "pp_size"),
        },
        "graph_mode": "disabled" if _first_server_arg(sglang_obs, "disable_cuda_graph") == "True" else "unknown",
        "backends": backends,
        "source_dir": str(datadir),
    }
    for p in files:
        if p.suffix not in (".log", ".txt"):
            continue
        txt = p.read_text(errors="replace")[:200_000]
        m = re.search(r"sglang(?:==| is using | version[:=]\s*)([A-Za-z0-9_.+-]+)", txt, re.I)
        if m and runtime["versions"]["sglang"] == "unknown":
            runtime["versions"]["sglang"] = m.group(1)
        m = re.search(r"nccl==([A-Za-z0-9_.+-]+)", txt, re.I)
        if m and runtime["versions"]["nccl"] == "unknown":
            runtime["versions"]["nccl"] = m.group(1)
        if "disable_cuda_graph=False" in txt or "graphs_on" in p.name:
            runtime["graph_mode"] = "enabled"
        if "disable_cuda_graph=True" in txt or "graphs_off" in p.name or "nograph" in p.name:
            runtime["graph_mode"] = "disabled" if runtime["graph_mode"] == "unknown" else runtime["graph_mode"]
    return _deep_merge(runtime, manifest.get("runtime") or {})


def parse_gpu_telemetry(path: Path) -> dict | None:
    try:
        with path.open(newline="", errors="replace") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except OSError:
        return None
    if not rows:
        return None

    def key(row, name):
        for k, v in row.items():
            if k.strip() == name:
                return v
        return None

    by_gpu: dict[str, dict] = {}
    for row in rows:
        idx = (key(row, "index") or key(row, " index") or "?").strip()
        rec = by_gpu.setdefault(idx, {
            "index": idx,
            "name": (key(row, "name") or key(row, " name") or "unknown").strip(),
            "samples": 0,
            "sm_clock_mhz": [],
            "mem_clock_mhz": [],
            "power_w": [],
            "temp_c": [],
            "pstates": collections.Counter(),
            "throttle_flags": collections.Counter(),
        })
        rec["samples"] += 1
        for field, out_key in (("clocks.current.sm [MHz]", "sm_clock_mhz"),
                               ("clocks.current.memory [MHz]", "mem_clock_mhz"),
                               ("power.draw [W]", "power_w"),
                               ("temperature.gpu", "temp_c")):
            v = _num(key(row, field) or "")
            if v is not None:
                rec[out_key].append(v)
        pst = (key(row, "pstate") or "").strip()
        if pst:
            rec["pstates"][pst] += 1
        thr = (key(row, "clocks_event_reasons.active") or "").strip()
        if thr:
            rec["throttle_flags"][thr] += 1

    def stats(vals):
        if not vals:
            return {"min": None, "avg": None, "max": None}
        return {"min": min(vals), "avg": sum(vals) / len(vals), "max": max(vals)}

    gpus = []
    for rec in by_gpu.values():
        gpus.append({
            "index": rec["index"],
            "name": rec["name"],
            "samples": rec["samples"],
            "sm_clock_mhz": stats(rec["sm_clock_mhz"]),
            "mem_clock_mhz": stats(rec["mem_clock_mhz"]),
            "power_w": stats(rec["power_w"]),
            "temp_c": stats(rec["temp_c"]),
            "pstates": dict(rec["pstates"]),
            "throttle_flags": dict(rec["throttle_flags"]),
        })
    return {"file": path.name, "gpus": sorted(gpus, key=lambda r: str(r["index"]))}


def _parse_telemetry(files: list[Path]) -> dict:
    rows = []
    names = set()
    for p in _by_name(files, pattern=r"gpu_telemetry.*\.csv$"):
        rec = parse_gpu_telemetry(p)
        if rec:
            rows.append(rec)
            for g in rec.get("gpus", []):
                if g.get("name"):
                    names.add(g["name"])
    gpu_ids = set()
    for rec in rows:
        for g in rec.get("gpus", []):
            gpu_ids.add(str(g.get("index")))
    return {
        "files": rows,
        "gpu_count": len(gpu_ids) or None,
        "gpu_names": sorted(names),
    }


def _workloads_from_dataset(ds: "Dataset") -> list[dict]:
    out = []
    seen = set()
    for r in ds.e2e:
        key = ("e2e", r.get("config"), r.get("conc"), r.get("in_len"), r.get("out_len"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": r.get("file"),
            "kind": r.get("kind", "sweep"),
            "phase": "decode",
            "config": r.get("config"),
            "concurrency": r.get("conc"),
            "input_len": r.get("in_len"),
            "output_len": r.get("out_len"),
            "tokens_per_chunk": r.get("tokens_per_chunk"),
            "graph_mode": "unknown",
        })
    for cap in ds.ncu:
        label = cap.get("label", "")
        bm = re.search(r"(?:^|_)(?:bs|b|m)(\d+)\b", label)
        im = re.search(r"(?:^|_)(?:i|in|ctx)(\d+)\b", label)
        om = re.search(r"(?:^|_)(?:o|out)(\d+)\b", label)
        out.append({
            "source": cap.get("details_file") or cap.get("raw_file") or label,
            "kind": "ncu",
            "phase": cap.get("regime"),
            "config": label,
            "batch": int(bm.group(1)) if bm else cap.get("batch"),
            "input_len": int(im.group(1)) if im else None,
            "output_len": int(om.group(1)) if om else None,
            "graph_mode": "disabled" if "graphs_off" in label else "enabled" if "graphs_on" in label else "unknown",
        })
    for n in ds.nsys:
        fn = n.get("file", "")
        im = re.search(r"(?:^|_)(?:i|in)(\d+)", fn)
        om = re.search(r"(?:^|_)(?:o|out)(\d+)", fn)
        cm = re.search(r"(?:^|_)c(?:onc)?(\d+)", fn)
        out.append({
            "source": fn,
            "kind": "nsys",
            "phase": "prefill" if "prefill" in fn.lower() else "decode",
            "config": fn.replace("_kernsum.txt", ""),
            "concurrency": int(cm.group(1)) if cm else None,
            "input_len": int(im.group(1)) if im else None,
            "output_len": int(om.group(1)) if om else None,
            "graph_mode": "disabled" if "graphs_off" in fn else "enabled" if "graphs_on" in fn else "unknown",
        })
    return out


def _artifact_inventory(datadir: Path, files: list[Path], parsed: set[str], failures: list[dict]) -> dict:
    rows = []
    for p in sorted(files):
        rel = _rel(p, datadir)
        kind = "other"
        if p.name.endswith((".tgz", ".tar.gz", ".tar")):
            kind = "archive"
        elif p.name.endswith(".trace.json.gz"):
            kind = "torch_trace"
        elif p.name.endswith("_kernsum.txt") or p.name.endswith("_cuda_gpu_kern_sum.csv") or p.suffix == ".sqlite":
            kind = "nsys_export"
        elif p.name.startswith("ncu_") or p.name.endswith(".ncu-rep"):
            kind = "ncu"
        elif p.name.endswith(".nsys-rep"):
            kind = "nsys_binary"
        elif p.name.startswith(("e2e_", "ceil_")):
            kind = "e2e"
        elif p.name.startswith("gpu_telemetry"):
            kind = "telemetry"
        rows.append({
            "path": rel,
            "name": p.name,
            "kind": kind,
            "size_bytes": p.stat().st_size if p.exists() else None,
            "status": "parsed" if rel in parsed else "opaque",
        })
    return {
        "root": str(datadir),
        "files": rows,
        "parsed": sorted(parsed),
        "failures": failures,
    }


def _tar_text_members(path: Path) -> dict[str, str]:
    out = {}
    try:
        with tarfile.open(path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = Path(m.name).name
                if not re.search(r"(ncu_.*(_raw\.csv|_details\.txt|\.log)$|gpu_telemetry.*\.csv$)", name, re.I):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                out[name] = fh.read().decode("utf-8", errors="replace")
    except (tarfile.TarError, OSError):
        return {}
    return out


def _build_run(datadir: Path, manifest: dict, files: list[Path]) -> dict:
    rels = sorted(_rel(p, datadir) for p in files)
    h = hashlib.sha1()
    h.update(str(datadir.resolve()).encode())
    for rel in rels[:200]:
        h.update(rel.encode())
    run = {
        "id": h.hexdigest()[:12],
        "name": datadir.name,
        "source_dir": str(datadir),
        "created_at": _mtime_iso(files),
        "tags": [],
        "notes": [],
    }
    man_run = manifest.get("run") or {}
    run = _deep_merge(run, man_run)
    if manifest.get("tags") and not run.get("tags"):
        run["tags"] = manifest.get("tags")
    if manifest.get("notes") and not run.get("notes"):
        run["notes"] = manifest.get("notes") if isinstance(manifest.get("notes"), list) else [manifest.get("notes")]
    return run


# --------------------------------------------------------------------------- #
# top-level ingest
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    meta: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    workloads: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    taxonomy: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    e2e: list = field(default_factory=list)
    decode: list = field(default_factory=list)   # torch traces
    nsys: list = field(default_factory=list)
    timelines: list = field(default_factory=list)
    ncu: list = field(default_factory=list)       # one entry per ncu capture file
    findings: dict = field(default_factory=dict)  # filled by findings.py

    def to_dict(self) -> dict:
        return {
            "meta": self.meta, "run": self.run, "model": self.model,
            "runtime": self.runtime, "workloads": self.workloads,
            "artifacts": self.artifacts, "taxonomy": self.taxonomy,
            "health": self.health, "e2e": self.e2e,
            "decode": self.decode, "nsys": self.nsys, "timelines": self.timelines, "ncu": self.ncu,
            "findings": self.findings, "display": DISPLAY,
        }


def ingest(datadir: Path, kernel_cats=KERNEL_CATS, use_cache: bool = True) -> Dataset:
    datadir = Path(datadir).expanduser()
    ds = Dataset()
    files = _discover_files(datadir)
    manifest = _load_manifest(datadir)
    parsed: set[str] = set()
    failures: list[dict] = []

    # torch decode traces
    for p in _by_name(files, suffix=".trace.json.gz"):
        try:
            ds.decode.append(parse_torch_trace(p, kernel_cats, use_cache=use_cache))
            parsed.add(_rel(p, datadir))
        except Exception as exc:  # large traces should not blank the whole run
            failures.append({"path": _rel(p, datadir), "error": f"trace parse failed: {exc}"})

    # nsys kernel summaries
    seen_nsys_files: set[str] = set()
    for p in _by_name(files, suffix="_kernsum.txt"):
        ds.nsys.append(parse_nsys_kernsum(p, kernel_cats))
        seen_nsys_files.add(p.name)
        parsed.add(_rel(p, datadir))
    for p in _by_name(files, suffix="_cuda_gpu_kern_sum.csv"):
        label_name = p.name.replace("_cuda_gpu_kern_sum.csv", "_kernsum.txt")
        if label_name in seen_nsys_files:
            continue
        ds.nsys.append(parse_nsys_kernsum_csv(p, kernel_cats))
        seen_nsys_files.add(label_name)
        parsed.add(_rel(p, datadir))
    for p in _by_name(files, suffix=".sqlite"):
        kernsum, timeline = parse_nsys_sqlite(p, kernel_cats)
        if kernsum:
            label_name = kernsum.get("file")
            if label_name not in seen_nsys_files:
                ds.nsys.append(kernsum)
                seen_nsys_files.add(label_name)
            parsed.add(_rel(p, datadir))
        if timeline:
            ds.timelines.append(timeline)

    # e2e + ceiling
    for p in sorted([p for p in files if p.name.startswith(("e2e_", "ceil_")) and p.suffix == ".txt"]):
        rows = parse_serve_log(p)
        if rows:
            ds.e2e.extend(rows)
            parsed.add(_rel(p, datadir))

    # ncu captures: group <stem>_details.txt + <stem>_raw.csv + <stem>.log
    ncu_details_files = {p.name[: -len("_details.txt")]: p for p in _by_name(files, suffix="_details.txt")
                         if p.name.startswith("ncu_")}
    ncu_raw_files = {p.name[: -len("_raw.csv")]: p for p in _by_name(files, suffix="_raw.csv")
                     if p.name.startswith("ncu_")}
    ncu_log_files = {p.name[: -len(".log")]: p for p in _by_name(files, suffix=".log")
                     if p.name.startswith("ncu_")}
    tar_members: dict[str, dict[str, str]] = {}
    for archive in [p for p in files if p.suffix in (".tgz", ".gz", ".tar") and "ncu_results" in p.name]:
        members = _tar_text_members(archive)
        if members:
            parsed.add(_rel(archive, datadir))
        for name, text in members.items():
            stem = name
            if stem.endswith("_details.txt"):
                stem = stem[: -len("_details.txt")]
                tar_members.setdefault(stem, {})["details"] = text
            elif stem.endswith("_raw.csv"):
                stem = stem[: -len("_raw.csv")]
                tar_members.setdefault(stem, {})["raw"] = text
            elif stem.endswith(".log"):
                stem = stem[: -len(".log")]
                tar_members.setdefault(stem, {})["log"] = text
    ncu_stems = sorted(set(ncu_details_files) | set(ncu_raw_files) | set(tar_members))
    for stem in ncu_stems:
        details_p = ncu_details_files.get(stem)
        raw_p = ncu_raw_files.get(stem)
        log_p = ncu_log_files.get(stem)
        tar_rec = tar_members.get(stem, {})
        details = (parse_ncu_details(details_p, kernel_cats) if details_p else
                   parse_ncu_details_text(tar_rec["details"], kernel_cats) if "details" in tar_rec else [])
        raw = (parse_ncu_raw(raw_p) if raw_p else
               parse_ncu_raw_text(tar_rec["raw"]) if "raw" in tar_rec else [])
        if details:
            kernels = merge_ncu(details, raw)
        else:
            # raw-CSV-only capture: build full kernel rows from the SoL columns
            # parse_ncu_raw now extracts. `valid` iff we got a real throughput
            # reading (so a bound-type can be computed downstream).
            kernels = []
            for r in raw:
                k = {"kernel": r["kernel"], "cat": categorize(r["kernel"], kernel_cats),
                     "clc": False, "cluster": bool(_CLUSTER_NAME.search(r["kernel"])), **r}
                k["valid"] = any(k.get(m) is not None for m in ("comp", "mem", "dram", "occ"))
                kernels.append(k)
        n_valid = sum(1 for k in kernels if k.get("valid"))
        label = stem.replace("ncu_", "")
        regime = "prefill" if "prefill" in label.lower() else "decode"
        # batch / decode-token count: `bs64`/`b32` (serving batch) or `m64` (gate3
        # decode-M microbench). ctx labels like `ctx131072` must NOT be read as batch.
        bm = re.search(r"(?:^|_)(?:bs|b|m)(\d+)\b", label)
        batch = int(bm.group(1)) if bm else None
        ds.ncu.append({
            "label": label,
            "regime": regime,         # decode captures characterize the decode breakdown; prefill captures don't
            "batch": batch,           # serving batch (>=32) is trustworthy; bs1 shows phantom occupancy headroom
            "details_file": details_p.name if details_p else (f"{stem}_details.txt" if "details" in tar_rec else None),
            "raw_file": raw_p.name if raw_p else (f"{stem}_raw.csv" if "raw" in tar_rec else None),
            "n_kernels": len(kernels),
            "n_valid": n_valid,
            "n_cluster": sum(1 for k in kernels if k.get("cluster")),
            "all_nan": len(kernels) > 0 and n_valid == 0,
            "log": parse_ncu_log(log_p) if log_p else
                   parse_ncu_log_text(tar_rec["log"], f"{stem}.log") if "log" in tar_rec else None,
            "kernels": kernels,
        })
        for p in (details_p, raw_p, log_p):
            if p:
                parsed.add(_rel(p, datadir))

    # health: bogus summary.txt + nan captures + binary-only reps
    bogus_summaries = [p.name for p in _by_name(files, suffix="_summary.txt")
                       if "==ERROR==" in p.read_text(errors="replace")[:200]]
    nan_caps = [c["label"] for c in ds.ncu if c["all_nan"]]
    failed_caps = [c["label"] for c in ds.ncu if (c.get("log") or {}).get("launchfailed")]
    partial_caps = [c["label"] for c in ds.ncu
                    if (c.get("log") or {}).get("launchfailed") and c.get("n_valid", 0) > 0]
    missing_nsys_exports = sorted(
        p.with_suffix("").name for p in _by_name(files, suffix=".nsys-rep")
        if f"{p.with_suffix('').name}_kernsum.txt" not in seen_nsys_files
    )
    sglang_obs = _parse_sglang_observations(datadir, files)
    telemetry = _parse_telemetry(files)
    sglang_notes = []
    for row in sglang_obs.get("linear_attn_backends", []):
        sglang_notes.append(
            f"SGLang linear attention backend observed {row['count']} file(s): "
            f"decode={row['decode']}, prefill={row['prefill']}."
        )
    for row in sglang_obs.get("gdn_dispatchers", []):
        sglang_notes.append(
            f"SGLang GDN dispatcher observed {row['count']} file(s): "
            f"decode={row['decode_kernel']}, extend={row['extend_kernel']}, "
            f"packed_decode={row['packed_decode']}."
        )
    ds.health = {
        "torch_traces": len(ds.decode),
        "nsys_rep_binaries": len(_by_name(files, suffix=".nsys-rep")),
        "nsys_kernsum_exports": len(ds.nsys),
        "nsys_timelines": len(ds.timelines),
        "nsys_missing_kernsum_exports": missing_nsys_exports,
        "ncu_rep_binaries": len(_by_name(files, suffix=".ncu-rep")),
        "ncu_captures_parsed": len(ds.ncu),
        "ncu_all_nan_captures": nan_caps,
        "ncu_launchfailed_captures": failed_caps,
        "ncu_partial_captures": partial_caps,
        "bogus_summary_exports": bogus_summaries,
        "sglang": sglang_obs,
        "telemetry": telemetry,
        "e2e_rows": len(ds.e2e),
        "notes": [
            "Mac-side: .nsys-rep/.ncu-rep binaries are opaque (no local NVIDIA CLI). "
            "Only their text exports are parsed.",
            f"{len(missing_nsys_exports)} nsys report(s) are missing cuda_gpu_kern_sum exports: "
            + (", ".join(missing_nsys_exports) or "none"),
            f"{len(nan_caps)} ncu capture(s) came back all-NaN (cluster/CLC kernels "
            "ncu kernel-replay cannot count): " + (", ".join(nan_caps) or "none"),
            f"{len(failed_caps)} ncu capture(s) reported LaunchFailed: "
            + (", ".join(failed_caps) or "none"),
            f"{len(bogus_summaries)} _summary.txt export(s) are ==ERROR== (wrong --page) "
            "and were ignored.",
        ] + sglang_notes,
    }
    if manifest.get("_error"):
        ds.health["notes"].append(manifest["_error"])
    model_id = _infer_model_id(files, sglang_obs, manifest)
    ds.run = _build_run(datadir, manifest, files)
    ds.model = infer_architecture(model_id, manifest.get("model") or {})
    quant = _first_server_arg(sglang_obs, "quantization")
    if quant and ds.model.get("quantization") in (None, "unknown"):
        ds.model["quantization"] = quant
    kv_dtype = _first_server_arg(sglang_obs, "kv_cache_dtype")
    if kv_dtype:
        ds.model.setdefault("kv_cache", {})["dtype"] = kv_dtype
    ds.runtime = _infer_runtime(datadir, sglang_obs, manifest, files, telemetry)
    ds.workloads = _workloads_from_dataset(ds)
    if manifest.get("workload"):
        ds.workloads.insert(0, _deep_merge({"source": "profile_manifest.json"}, manifest["workload"]))
    ds.taxonomy = taxonomy_json()
    ds.artifacts = _artifact_inventory(datadir, files, parsed, failures)
    ds.meta = {
        "datadir": str(datadir),
        "n_files": len(files),
        "schema_version": 2,
        "schema_compat": [1],
    }
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datadir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write dataset JSON here (default: stdout summary only)")
    args = ap.parse_args()
    ds = ingest(args.datadir)
    d = ds.to_dict()
    if args.out:
        args.out.write_text(json.dumps(d, indent=2))
        print(f"wrote {args.out}")
    print(json.dumps(d["health"], indent=2))


if __name__ == "__main__":
    main()
