#!/usr/bin/env python3
"""Build static visual summaries for the B300 profiler artifacts."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/private/tmp/profiler_b300")
OUT = ROOT / "visuals"
ASSETS = OUT / "assets"

PALETTE = {
    "DeepGEMM MegaMoE": "#1769aa",
    "Grouped MXFP4 MoE": "#117a65",
    "MLA attention": "#8e44ad",
    "TileLang MHC": "#c46a1b",
    "Collectives": "#b03a2e",
    "Dense GEMM": "#2e86c1",
    "Routing and quant": "#6c7a00",
    "Norm and rope": "#7d6608",
    "Memcpy": "#5d6d7e",
    "Other": "#6c757d",
}


def artifact_dir(kind: str) -> Path:
    """Return the profiler artifact directory for old and new local layouts."""
    nested = ROOT / kind / "results"
    if nested.exists():
        return nested
    return ROOT / kind


def artifact_href(kind: str, filename: str) -> str:
    if (ROOT / kind / "results").exists():
        return f"../{kind}/results/{html.escape(filename)}"
    return f"../{kind}/{html.escape(filename)}"


@dataclass
class NcuKernel:
    report: str
    rep_file: str
    label: str
    category: str
    kernel: str
    duration_us: float
    sm_pct: float
    dram_pct: float
    mem_value: float
    mem_unit: str
    mem_gbps: float
    occupancy_pct: float
    regs: float
    smem_kb: float


def clean_number(value: str | int | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip().replace(",", "")
    if not value or value == "no data":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def short_kernel_name(name: str, limit: int = 92) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    replacements = [
        ("void ", ""),
        ("deep_gemm::", ""),
        ("at::native::", ""),
        ("<unnamed>::", ""),
        ("cutlass::", ""),
    ]
    for src, dst in replacements:
        name = name.replace(src, dst)
    if len(name) <= limit:
        return name
    base = name.split("<", 1)[0]
    if 12 < len(base) <= limit:
        return base
    return name[: limit - 1] + "..."


def classify_kernel(name: str) -> str:
    low = name.lower()
    if "mega_moe_impl" in low or "sm100_fp8_fp4_mega_moe_impl" in low:
        return "DeepGEMM MegaMoE"
    if "group_gemm" in low or "gemmuniversal" in low or "mxf4" in low or "mxfp4" in low:
        return "Grouped MXFP4 MoE"
    if "flash_fwd" in low or "mla" in low or "paged_mqa" in low or "metadata" in low and "mla" in low:
        return "MLA attention"
    if "mhc_" in low or "_hc_" in low or "hc_" in low:
        return "TileLang MHC"
    if "nccl" in low or "all_reduce" in low or "allgather" in low or "reduce_scatter" in low:
        return "Collectives"
    if "cublas" in low or "nvjet" in low or "gemm_1d1d" in low or "tf32" in low:
        return "Dense GEMM"
    if "topk" in low or "quant" in low or "moe_fused_gate" in low or "dispatch" in low or "silu" in low:
        return "Routing and quant"
    if "norm" in low or "rope" in low or "rmsnorm" in low:
        return "Norm and rope"
    if "memcpy" in low or "copy" in low:
        return "Memcpy"
    return "Other"


def ncu_label(report: str, kernel: str, idx: int) -> str:
    if "live_deepgemm" in report:
        return "Live DeepGEMM MegaMoE TP1"
    if "groupmm_dsv4_w13" in report and "device_kernel" in kernel:
        return "Grouped MXFP4 MoE W13"
    if "groupmm_dsv4_w2" in report and "device_kernel" in kernel:
        return "Grouped MXFP4 MoE W2"
    if "groupmm" in report and "__get_group_gemm_starts" in kernel:
        return "Grouped MoE start table"
    if "flashmla" in report and "splitkv" in kernel:
        length = "16k" if "16x16k" in report or "128x16k" in report else "8k" if "128x8k" in report else ""
        suffix = f" {length}" if length else ""
        return f"MLA split-KV (FP8 KV){suffix} #{idx + 1}"
    if "flashmla" in report and "combine" in kernel:
        return "MLA combine (FP8 KV)"
    if "tilelang_mhc" in report and "pre" in kernel:
        return "TileLang MHC pre"
    if "tilelang_mhc" in report and "post" in kernel:
        return "TileLang MHC post"
    return short_kernel_name(kernel, 46)


def read_ncu() -> list[NcuKernel]:
    out: list[NcuKernel] = []
    for path in sorted(artifact_dir("ncu").glob("*.raw.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        units = rows[0]
        mem_unit = units.get("dram__bytes.sum.per_second", "")
        rep_name = path.name.replace(".raw.csv", ".ncu-rep")
        report = path.name.replace(".raw.csv", "")
        kernel_idx = 0
        for row in rows:
            kernel = row.get("Kernel Name", "")
            if not kernel:
                continue
            mem_value = clean_number(row.get("dram__bytes.sum.per_second"))
            mem_gbps = mem_value * 1000.0 if mem_unit.startswith("Tbyte") else mem_value
            duration_us = clean_number(row.get("gpu__time_duration.sum"))
            smem = clean_number(row.get("launch__shared_mem_per_block_static")) + clean_number(
                row.get("launch__shared_mem_per_block_dynamic")
            )
            item = NcuKernel(
                report=report,
                rep_file=rep_name,
                label=ncu_label(report, kernel, kernel_idx),
                category=classify_kernel(kernel),
                kernel=kernel,
                duration_us=duration_us,
                sm_pct=clean_number(row.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
                dram_pct=clean_number(row.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")),
                mem_value=mem_value,
                mem_unit=mem_unit,
                mem_gbps=mem_gbps,
                occupancy_pct=clean_number(row.get("sm__warps_active.avg.pct_of_peak_sustained_active")),
                regs=clean_number(row.get("launch__registers_per_thread")),
                smem_kb=smem,
            )
            out.append(item)
            kernel_idx += 1
    return out


def run_label_from_sqlite(path: Path) -> str:
    stem = path.stem
    mode = "CUDA graph" if "cudagraph" in stem else "eager/NVTX"
    if "prefill_exact_128x16k" in stem:
        span = "exact 128x16k prefill"
    elif "decode512_exact_128x16k" in stem:
        span = "exact 128x16k decode512 at 16k KV"
        if "nograph_20260602_161723" in stem:
            span += " (mixed tail-prefill)"
        elif "nograph_delayed" in stem:
            span += " (clean delayed)"
    elif "serving_blackwell" in stem:
        span = "non-exact bench_serving capture"
    elif "long" in stem:
        span = "long steady decode"
    else:
        span = "short serving"
    return f"{mode}, {span}"


def read_nsys() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    run_summaries: list[dict] = []
    category_rows: list[dict] = []
    top_rows: list[dict] = []
    timeline_rows: list[dict] = []
    overall_top: dict[str, dict] = defaultdict(lambda: {"total_ns": 0, "instances": 0, "category": "Other"})

    for db in sorted(artifact_dir("nsys").glob("*.sqlite")):
        run = run_label_from_sqlite(db)
        con = sqlite3.connect(db)
        query = """
            select k.start, k.end, k.deviceId, k.streamId, coalesce(s.value, ss.value, '')
            from CUPTI_ACTIVITY_KIND_KERNEL k
            left join StringIds s on s.id = k.demangledName
            left join StringIds ss on ss.id = k.shortName
        """
        rows = list(con.execute(query))
        con.close()
        if not rows:
            continue
        min_start = min(r[0] for r in rows)
        max_end = max(r[1] for r in rows)
        capture_ns = max(1, max_end - min_start)
        bins = 180
        timeline = {cat: [0 for _ in range(bins)] for cat in PALETTE}
        per_cat: dict[str, dict] = defaultdict(lambda: {"total_ns": 0, "instances": 0})
        per_name: dict[str, dict] = defaultdict(lambda: {"total_ns": 0, "instances": 0, "category": "Other"})
        device_ids = set()
        streams = set()

        for start, end, device_id, stream_id, name in rows:
            dur = max(0, end - start)
            cat = classify_kernel(name)
            key = short_kernel_name(name, 120)
            per_cat[cat]["total_ns"] += dur
            per_cat[cat]["instances"] += 1
            per_name[key]["total_ns"] += dur
            per_name[key]["instances"] += 1
            per_name[key]["category"] = cat
            overall_top[key]["total_ns"] += dur
            overall_top[key]["instances"] += 1
            overall_top[key]["category"] = cat
            device_ids.add(device_id)
            streams.add(stream_id)
            mid = (start + end) / 2 - min_start
            b = min(bins - 1, max(0, int(mid / capture_ns * bins)))
            timeline.setdefault(cat, [0 for _ in range(bins)])[b] += dur

        kernel_ns = sum(v["total_ns"] for v in per_cat.values())
        run_summaries.append(
            {
                "run": run,
                "file": db.name,
                "capture_s": capture_ns / 1e9,
                "kernel_s": kernel_ns / 1e9,
                "launches": len(rows),
                "devices": len(device_ids),
                "streams": len(streams),
            }
        )
        for cat, value in sorted(per_cat.items()):
            category_rows.append(
                {
                    "run": run,
                    "file": db.name,
                    "category": cat,
                    "total_s": value["total_ns"] / 1e9,
                    "instances": value["instances"],
                    "pct_kernel_time": 100.0 * value["total_ns"] / kernel_ns if kernel_ns else 0.0,
                }
            )
        for key, value in sorted(per_name.items(), key=lambda x: x[1]["total_ns"], reverse=True)[:12]:
            top_rows.append(
                {
                    "run": run,
                    "file": db.name,
                    "kernel": key,
                    "category": value["category"],
                    "total_s": value["total_ns"] / 1e9,
                    "instances": value["instances"],
                    "avg_us": value["total_ns"] / max(1, value["instances"]) / 1e3,
                }
            )
        max_bin = max(max(values) for values in timeline.values()) or 1
        for cat, values in timeline.items():
            if not any(values):
                continue
            timeline_rows.append(
                {
                    "run": run,
                    "file": db.name,
                    "category": cat,
                    "values": [v / max_bin for v in values],
                    "total_s": sum(values) / 1e9,
                }
            )

    overall_rows = [
        {
            "kernel": key,
            "category": value["category"],
            "total_s": value["total_ns"] / 1e9,
            "instances": value["instances"],
            "avg_us": value["total_ns"] / max(1, value["instances"]) / 1e3,
        }
        for key, value in sorted(overall_top.items(), key=lambda x: x[1]["total_ns"], reverse=True)[:18]
    ]
    return run_summaries, category_rows, top_rows, timeline_rows + [{"overall_top": overall_rows}]


def read_bench() -> list[dict]:
    rows = []
    for path in sorted(artifact_dir("nsys").glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "results" in data:
            durations = [clean_number(item.get("elapsed_s")) for item in data["results"]]
            rows.append(
                {
                    "file": path.name,
                    "kind": "steady decode stream",
                    "requests": len(durations),
                    "duration_s": max(durations) if durations else 0,
                    "output_tps": 0,
                    "ttft_ms": 0,
                    "tpot_ms": 0,
                }
            )
            continue
        if data.get("total_input_tokens_exact"):
            kind = "exact input_ids profile"
        elif path.name.startswith("bench_"):
            kind = "bench_serving random text"
        else:
            kind = "serving bench"
        rows.append(
            {
                "file": path.name,
                "kind": kind,
                "requests": data.get("completed", 0),
                "duration_s": clean_number(data.get("duration_s", data.get("duration"))),
                "output_tps": clean_number(data.get("output_throughput_tok_s", data.get("output_throughput"))),
                "ttft_ms": clean_number(data.get("mean_ttft_ms")),
                "tpot_ms": clean_number(data.get("mean_tpot_ms")),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def svg_bar_chart(
    rows: list[dict],
    value_key: str,
    label_key: str,
    title: str,
    unit: str,
    path: Path,
    color_key: str | None = None,
    width: int = 1120,
    bar_h: int = 26,
) -> str:
    rows = rows[:18]
    left, right, top = 290, 120, 58
    height = top + len(rows) * (bar_h + 12) + 46
    max_value = max([clean_number(r.get(value_key)) for r in rows] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#1d252c}.muted{fill:#697581}.axis{stroke:#cfd8dc}.grid{stroke:#eef2f4}.small{font-size:12px}.label{font-size:13px}.title{font-size:19px;font-weight:650}</style>',
        f'<text class="title" x="16" y="28">{html.escape(title)}</text>',
    ]
    chart_w = width - left - right
    for i in range(6):
        x = left + chart_w * i / 5
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{height-34}"/>')
        parts.append(f'<text class="small muted" x="{x:.1f}" y="{height-12}" text-anchor="middle">{max_value*i/5:.1f}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * (bar_h + 12)
        value = clean_number(row.get(value_key))
        w = chart_w * value / max_value
        cat = row.get(color_key or "category", "Other")
        color = PALETTE.get(cat, "#1769aa")
        parts.append(f'<text class="label" x="16" y="{y+18}">{html.escape(str(row.get(label_key, ""))[:42])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>')
        parts.append(f'<text class="small" x="{left+w+8:.1f}" y="{y+18}">{value:.2f} {unit}</text>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    path.write_text(svg)
    return path.name


def svg_ncu_scatter(rows: list[NcuKernel], path: Path) -> str:
    width, height = 1120, 680
    left, right, top, bottom = 76, 260, 72, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    max_axis = max(60.0, max([r.sm_pct for r in rows] + [0]), max([r.dram_pct for r in rows] + [0]))
    max_axis = min(100.0, math.ceil(max_axis / 10) * 10)

    def x(v: float) -> float:
        return left + plot_w * v / max_axis

    def y(v: float) -> float:
        return top + plot_h - plot_h * v / max_axis

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#1d252c}.muted{fill:#697581}.grid{stroke:#e9eef2}.axis{stroke:#aeb8c2}.title{font-size:20px;font-weight:650}.small{font-size:12px}.legend{font-size:13px}</style>',
        '<text class="title" x="16" y="30">NCU Speed-of-Light Placement</text>',
        '<text class="small muted" x="16" y="52">X = DRAM throughput percent of peak, Y = SM throughput percent of peak, bubble size = kernel duration</text>',
    ]
    for i in range(7):
        v = max_axis * i / 6
        parts.append(f'<line class="grid" x1="{x(v):.1f}" y1="{top}" x2="{x(v):.1f}" y2="{top+plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{y(v):.1f}" x2="{left+plot_w}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="small muted" x="{x(v):.1f}" y="{height-28}" text-anchor="middle">{v:.0f}</text>')
        parts.append(f'<text class="small muted" x="{left-10}" y="{y(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    parts.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(max_axis)}" y2="{y(max_axis)}" stroke="#b9c4ca" stroke-dasharray="6 6"/>')
    parts.append(f'<text class="small muted" x="{left+plot_w/2}" y="{height-8}" text-anchor="middle">DRAM throughput (% of peak)</text>')
    parts.append(f'<text class="small muted" transform="translate(18 {top+plot_h/2}) rotate(-90)" text-anchor="middle">SM throughput (% of peak)</text>')
    for row in rows:
        color = PALETTE.get(row.category, "#1769aa")
        radius = 6 + math.sqrt(max(row.duration_us, 1)) * 0.7
        parts.append(
            f'<circle cx="{x(row.dram_pct):.1f}" cy="{y(row.sm_pct):.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78" stroke="#17202a" stroke-width="0.7">'
            f"<title>{html.escape(row.label)}\\nSM {row.sm_pct:.1f}% / DRAM {row.dram_pct:.1f}% / {row.duration_us:.1f} us</title></circle>"
        )
    legend_x, legend_y = width - right + 28, top
    for i, cat in enumerate([c for c in PALETTE if any(r.category == c for r in rows)]):
        yy = legend_y + i * 25
        parts.append(f'<circle cx="{legend_x}" cy="{yy}" r="7" fill="{PALETTE[cat]}"/>')
        parts.append(f'<text class="legend" x="{legend_x+16}" y="{yy+4}">{html.escape(cat)}</text>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    path.write_text(svg)
    return path.name


def svg_stacked_categories(category_rows: list[dict], run_summaries: list[dict], path: Path) -> str:
    width, height = 1120, 360
    left, right, top = 230, 80, 72
    bar_h = 34
    plot_w = width - left - right
    runs = [r["run"] for r in run_summaries]
    by_run = defaultdict(list)
    for row in category_rows:
        by_run[row["run"]].append(row)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#1d252c}.muted{fill:#697581}.title{font-size:20px;font-weight:650}.small{font-size:12px}.label{font-size:13px}</style>',
        '<text class="title" x="16" y="30">NSYS GPU Kernel Time Mix</text>',
        '<text class="small muted" x="16" y="52">Stacked by category within each captured trace</text>',
    ]
    for idx, run in enumerate(runs):
        y = top + idx * 58
        parts.append(f'<text class="label" x="16" y="{y+22}">{html.escape(run)}</text>')
        x0 = left
        total = sum(clean_number(r["total_s"]) for r in by_run[run]) or 1
        for row in sorted(by_run[run], key=lambda r: clean_number(r["total_s"]), reverse=True):
            w = plot_w * clean_number(row["total_s"]) / total
            color = PALETTE.get(row["category"], "#6c757d")
            parts.append(
                f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}">'
                f'<title>{html.escape(row["category"])}: {clean_number(row["total_s"]):.2f}s, {clean_number(row["pct_kernel_time"]):.1f}%</title></rect>'
            )
            if w > 58:
                parts.append(
                    f'<text class="small" x="{x0+w/2:.1f}" y="{y+22}" text-anchor="middle" fill="#fff">{clean_number(row["pct_kernel_time"]):.0f}%</text>'
                )
            x0 += w
    legend_x, legend_y = left, top + len(runs) * 58 + 18
    x = legend_x
    for cat in [c for c in PALETTE if any(r["category"] == c for r in category_rows)]:
        parts.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" fill="{PALETTE[cat]}"/>')
        parts.append(f'<text class="small" x="{x+17}" y="{legend_y+11}">{html.escape(cat)}</text>')
        x += 17 + len(cat) * 7 + 18
        if x > width - 160:
            x = legend_x
            legend_y += 22
    parts.append("</svg>")
    svg = "\n".join(parts)
    path.write_text(svg)
    return path.name


def svg_timeline(timeline_rows: list[dict], path: Path) -> str:
    rows = [r for r in timeline_rows if "values" in r]
    runs = []
    for row in rows:
        if row["run"] not in runs:
            runs.append(row["run"])
    width = 1120
    left, top, row_h, run_gap = 220, 70, 12, 24
    categories = [c for c in PALETTE if any(r["category"] == c for r in rows)]
    height = top + len(runs) * (len(categories) * row_h + run_gap) + 52
    plot_w = width - left - 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#1d252c}.muted{fill:#697581}.title{font-size:20px;font-weight:650}.small{font-size:11px}.run{font-size:13px;font-weight:600}</style>',
        '<text class="title" x="16" y="30">NSYS Normalized Launch Activity Timeline</text>',
        '<text class="small muted" x="16" y="52">Each strip is a category; intensity is kernel duration accumulated into normalized time bins</text>',
    ]
    by = {(r["run"], r["category"]): r["values"] for r in rows}
    for ri, run in enumerate(runs):
        base_y = top + ri * (len(categories) * row_h + run_gap)
        parts.append(f'<text class="run" x="16" y="{base_y+12}">{html.escape(run)}</text>')
        for ci, cat in enumerate(categories):
            y = base_y + ci * row_h
            values = by.get((run, cat), [])
            parts.append(f'<text class="small muted" x="{left-10}" y="{y+9}" text-anchor="end">{html.escape(cat[:24])}</text>')
            if not values:
                continue
            bw = plot_w / len(values)
            color = PALETTE.get(cat, "#6c757d")
            for i, value in enumerate(values):
                if value <= 0:
                    continue
                opacity = 0.14 + 0.78 * min(value, 1.0)
                parts.append(f'<rect x="{left+i*bw:.1f}" y="{y}" width="{max(0.8,bw):.2f}" height="{row_h-2}" fill="{color}" opacity="{opacity:.3f}"/>')
        parts.append(f'<text class="small muted" x="{left}" y="{base_y+len(categories)*row_h+13}">0%</text>')
        parts.append(f'<text class="small muted" x="{left+plot_w}" y="{base_y+len(categories)*row_h+13}" text-anchor="end">100%</text>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    path.write_text(svg)
    return path.name


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    ncu = read_ncu()
    run_summaries, category_rows, top_rows, timeline_rows = read_nsys()
    overall_top = timeline_rows[-1]["overall_top"] if timeline_rows and "overall_top" in timeline_rows[-1] else []
    timeline_rows = [r for r in timeline_rows if "values" in r]
    bench = read_bench()

    write_csv(
        OUT / "ncu_kernel_metrics.csv",
        [k.__dict__ for k in ncu],
        [
            "report",
            "rep_file",
            "label",
            "category",
            "duration_us",
            "sm_pct",
            "dram_pct",
            "mem_value",
            "mem_unit",
            "mem_gbps",
            "occupancy_pct",
            "regs",
            "smem_kb",
            "kernel",
        ],
    )
    write_csv(OUT / "nsys_run_summary.csv", run_summaries, ["run", "file", "capture_s", "kernel_s", "launches", "devices", "streams"])
    write_csv(OUT / "nsys_category_mix.csv", category_rows, ["run", "file", "category", "total_s", "instances", "pct_kernel_time"])
    write_csv(OUT / "nsys_top_kernels_by_trace.csv", top_rows, ["run", "file", "kernel", "category", "total_s", "instances", "avg_us"])
    write_csv(OUT / "nsys_overall_top_kernels.csv", overall_top, ["kernel", "category", "total_s", "instances", "avg_us"])
    write_csv(OUT / "bench_summary.csv", bench, ["file", "kind", "requests", "duration_s", "output_tps", "ttft_ms", "tpot_ms"])

    ncu_scatter = svg_ncu_scatter(ncu, ASSETS / "ncu_speed_of_light.svg")
    ncu_bw = svg_bar_chart(
        sorted([k.__dict__ for k in ncu], key=lambda r: r["mem_gbps"], reverse=True),
        "mem_gbps",
        "label",
        "NCU Memory Bandwidth",
        "GB/s",
        ASSETS / "ncu_memory_bandwidth.svg",
    )
    ncu_occ = svg_bar_chart(
        sorted([k.__dict__ for k in ncu], key=lambda r: r["occupancy_pct"], reverse=True),
        "occupancy_pct",
        "label",
        "NCU Achieved Occupancy",
        "%",
        ASSETS / "ncu_occupancy.svg",
    )
    nsys_mix = svg_stacked_categories(category_rows, run_summaries, ASSETS / "nsys_kernel_mix.svg")
    nsys_top = svg_bar_chart(
        overall_top,
        "total_s",
        "kernel",
        "NSYS Top Kernels Across B300 Captures",
        "s",
        ASSETS / "nsys_top_kernels.svg",
    )
    nsys_timeline = svg_timeline(timeline_rows, ASSETS / "nsys_activity_timeline.svg")

    cards = [
        ("NCU reports", str(len(list(artifact_dir("ncu").glob("*.ncu-rep"))))),
        ("NCU kernels", str(len(ncu))),
        ("NSYS traces", str(len(run_summaries))),
        ("NSYS launches", f"{sum(int(r['launches']) for r in run_summaries):,}"),
        ("Total NSYS kernel time", f"{sum(float(r['kernel_s']) for r in run_summaries):.1f}s"),
    ]
    card_html = "\n".join(
        f'<div class="metric"><div class="metric-label">{html.escape(k)}</div><div class="metric-value">{html.escape(v)}</div></div>'
        for k, v in cards
    )

    ncu_rows = []
    for k in sorted(ncu, key=lambda item: item.duration_us, reverse=True):
        ncu_rows.append(
            [
                html.escape(k.label),
                html.escape(k.category),
                f"{k.duration_us:.2f}",
                f"{k.sm_pct:.2f}",
                f"{k.dram_pct:.2f}",
                f"{k.mem_gbps:.1f}",
                f"{k.occupancy_pct:.2f}",
                f'<a href="{artifact_href("ncu", k.rep_file)}">{html.escape(k.rep_file)}</a>',
            ]
        )
    nsys_rows = []
    for r in sorted(run_summaries, key=lambda x: x["run"]):
        nsys_rows.append(
            [
                html.escape(r["run"]),
                f"{r['capture_s']:.2f}",
                f"{r['kernel_s']:.2f}",
                f"{int(r['launches']):,}",
                str(r["devices"]),
                f'<a href="{artifact_href("nsys", r["file"].replace(".sqlite", ".nsys-rep"))}">{html.escape(r["file"].replace(".sqlite", ".nsys-rep"))}</a>',
            ]
        )
    bench_rows = []
    for r in bench:
        bench_rows.append(
            [
                html.escape(r["file"]),
                html.escape(r["kind"]),
                str(r["requests"]),
                f"{r['duration_s']:.2f}",
                f"{r['output_tps']:.2f}" if r["output_tps"] else "",
                f"{r['ttft_ms']:.1f}" if r["ttft_ms"] else "",
                f"{r['tpot_ms']:.1f}" if r["tpot_ms"] else "",
            ]
        )

    css = """
    :root { color-scheme: light; --ink:#17202a; --muted:#66727f; --line:#dbe4ea; --paper:#ffffff; --bg:#f5f7f8; --accent:#1769aa; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { padding:34px 40px 24px; background:#111820; color:white; border-bottom:5px solid #2e86c1; }
    header h1 { margin:0 0 8px; font-size:30px; letter-spacing:0; }
    header p { margin:0; color:#c8d2dc; max-width:980px; line-height:1.45; }
    main { max-width:1240px; margin:0 auto; padding:26px 28px 60px; }
    .metrics { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:18px; }
    .metric, section { background:var(--paper); border:1px solid var(--line); border-radius:8px; }
    .metric { padding:14px 16px; }
    .metric-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    .metric-value { font-size:24px; font-weight:700; margin-top:4px; }
    section { margin:18px 0; padding:20px; box-shadow:0 1px 2px rgba(23,32,42,.04); }
    h2 { margin:0 0 6px; font-size:21px; }
    .section-note { margin:0 0 16px; color:var(--muted); line-height:1.45; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }
    .figure { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; margin:14px 0; }
    .figure img { width:100%; display:block; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { padding:9px 10px; border-bottom:1px solid #e8eef2; text-align:left; vertical-align:top; }
    th { color:#4e5d6c; font-size:12px; text-transform:uppercase; letter-spacing:.04em; background:#f7fafb; position:sticky; top:0; }
    a { color:#1769aa; text-decoration:none; }
    a:hover { text-decoration:underline; }
    .table-wrap { max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:8px; }
    .downloads { display:flex; flex-wrap:wrap; gap:10px; }
    .downloads a { padding:8px 10px; background:#eef5fa; border:1px solid #cfe0ed; border-radius:6px; color:#174a70; }
    @media (max-width:900px) { .metrics, .grid2 { grid-template-columns:1fr; } main { padding:18px; } header { padding:26px 22px; } }
    """
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>B300 DeepSeek V4 Flash FP4 Profiling</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>B300 DeepSeek V4 Flash FP4 Profiling</h1>
  <p>Static visual pack built from local NSYS and NCU artifacts under <code>{html.escape(str(ROOT))}</code>. The model/expert path is FP4/MXFP4; FlashMLA entries labeled FP8 are FP8 E4M3 KV-cache attention, not FP8 model weights. NCU native roofline charts are in the linked <code>.ncu-rep</code> files; this page visualizes the exported scalar metrics and NSYS launch traces.</p>
</header>
<main>
  <div class="metrics">{card_html}</div>

  <section>
    <h2>NCU Kernel Reports</h2>
    <p class="section-note">Speed-of-light placement, normalized memory bandwidth, and achieved occupancy from the captured NCU reports. Memory bandwidth is normalized to GB/s from the NCU unit row.</p>
    <div class="figure"><img src="assets/{ncu_scatter}" alt="NCU SM versus DRAM throughput scatter"></div>
    <div class="grid2">
      <div class="figure"><img src="assets/{ncu_bw}" alt="NCU memory bandwidth bars"></div>
      <div class="figure"><img src="assets/{ncu_occ}" alt="NCU occupancy bars"></div>
    </div>
    <div class="table-wrap">{html_table(["Kernel", "Category", "Duration us", "SM %", "DRAM %", "GB/s", "Occupancy %", "Report"], ncu_rows)}</div>
  </section>

  <section>
    <h2>NSYS End-to-End Captures</h2>
    <p class="section-note">Kernel-time mix and launch-density timelines from the four B300 NSYS traces, including CUDA graph and eager/NVTX captures.</p>
    <div class="figure"><img src="assets/{nsys_mix}" alt="NSYS kernel mix"></div>
    <div class="figure"><img src="assets/{nsys_timeline}" alt="NSYS normalized launch activity timeline"></div>
    <div class="figure"><img src="assets/{nsys_top}" alt="NSYS top kernels"></div>
    <div class="table-wrap">{html_table(["Run", "Capture s", "Kernel s", "Launches", "Devices", "Report"], nsys_rows)}</div>
  </section>

  <section>
    <h2>Bench Metadata</h2>
    <p class="section-note">Benchmark JSON summaries kept with the NSYS artifacts. These are context for the traces, not replacements for the profiler reports.</p>
    <div class="table-wrap">{html_table(["File", "Kind", "Requests", "Duration s", "Output tok/s", "TTFT ms", "TPOT ms"], bench_rows)}</div>
  </section>

  <section>
    <h2>Generated Data</h2>
    <div class="downloads">
      <a href="ncu_kernel_metrics.csv">ncu_kernel_metrics.csv</a>
      <a href="nsys_run_summary.csv">nsys_run_summary.csv</a>
      <a href="nsys_category_mix.csv">nsys_category_mix.csv</a>
      <a href="nsys_top_kernels_by_trace.csv">nsys_top_kernels_by_trace.csv</a>
      <a href="nsys_overall_top_kernels.csv">nsys_overall_top_kernels.csv</a>
      <a href="bench_summary.csv">bench_summary.csv</a>
    </div>
  </section>
</main>
</body>
</html>
"""
    (OUT / "index.html").write_text(html_doc)


if __name__ == "__main__":
    build()
