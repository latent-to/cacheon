#!/usr/bin/env python3
"""Analyze downloaded Qwen3.5 B300 profile artifacts.

The script is deliberately stdlib-only. The Mac checkout may not have local
Nsight Systems/Compute installed, so `.nsys-rep` / `.ncu-rep` files are treated
as opaque unless their text exports were downloaded. Torch Chrome traces and
serve_load2 logs carry enough signal for the first bottleneck pass.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


KERNEL_CATS = {
    "all_gather": re.compile(r"AllGather|_all_gather|all_gather", re.I),
    "all_reduce": re.compile(r"allreduce_fusion|all_reduce|AllReduce|ncclDevKernel_AllReduce|one_shot", re.I),
    "delay_stream": re.compile(r"delayStreamKernel", re.I),
    "fp4_bmm": re.compile(r"^bmm_(?:E2m1|Bfloat16_E2m1)", re.I),
    "dense_gemm_nvjet": re.compile(r"^nvjet_sm103", re.I),
    "splitk_reduce": re.compile(r"splitKreduce", re.I),
    "moe_finalize": re.compile(r"finalizeKernel", re.I),
    "moe_routing": re.compile(r"routingIndices|routingInit|routingCustom|topk|TopK", re.I),
    "gdn_scan": re.compile(r"gdn_wide_vec|gated_delta|fused_recurrent|chunk_gated", re.I),
    "gdn_conv": re.compile(r"causal_conv1d", re.I),
    "attention": re.compile(r"fmha|attention|mha|paged|flash_fwd", re.I),
    "act_mul": re.compile(r"act_and_mul|silu|swiglu", re.I),
    "nvfp4_quant": re.compile(r"nvfp4_quantize|NVFP4Quantize", re.I),
    "fused_qkvzba": re.compile(r"fused_qkvzba", re.I),
    "rmsnorm_layernorm": re.compile(r"rmsnorm|RMSNorm|layer_norm|LayerNorm", re.I),
    "token_gather": re.compile(r"vectorized_gather|gather_kernel|index_select|IndexKernel", re.I),
    "block_scale_interleave": re.compile(r"block_scale_interleave|scale.*interleave", re.I),
    "kv_rope": re.compile(r"mrope|set_kv_buffer|fp8_set_kv", re.I),
    "sampling": re.compile(r"argmax|sample|softmax|penalt|resolve_future_token", re.I),
    "copy_memset": re.compile(r"Memcpy|Memset|memcpy|memset|copy_", re.I),
    "elementwise_misc": re.compile(r"elementwise|triton_poi|sigmoid|FillFunctor|CUDAFunctor_add|float8_copy", re.I),
}

DISPLAY = {
    "fp4_bmm": "FP4 BMM/GEMM family (bmm_*E2m1)",
    "dense_gemm_nvjet": "dense/projection GEMM (nvjet_sm103)",
    "splitk_reduce": "splitKreduce GEMM epilogue",
    "act_mul": "act_and_mul / SiLU",
    "elementwise_misc": "elementwise misc",
    "all_reduce": "all-reduce collective",
    "all_gather": "all-gather collective",
    "attention": "full attention kernel",
    "moe_routing": "MoE routing",
    "gdn_scan": "GDN scan",
    "moe_finalize": "MoE finalize",
    "gdn_conv": "GDN/causal conv",
    "fused_qkvzba": "GDN qkvzba split/reshape",
    "rmsnorm_layernorm": "norm kernels",
    "nvfp4_quant": "NVFP4 quantize",
    "block_scale_interleave": "block-scale interleave",
    "token_gather": "token gather/index",
    "kv_rope": "RoPE / KV write",
    "delay_stream": "delayStreamKernel",
    "sampling": "sampling/misc",
    "copy_memset": "copy/memset",
    "other": "other",
}


@dataclass
class Bucket:
    dur_us: float = 0.0
    count: int = 0

    def add(self, dur_us: float, n: int = 1) -> None:
        self.dur_us += dur_us
        self.count += n


@dataclass
class TraceSummary:
    path: Path
    label: str
    rank: str
    annotations: collections.Counter[str] = field(default_factory=collections.Counter)
    total_us: float = 0.0
    launch_count: int = 0
    cats: dict[str, Bucket] = field(default_factory=lambda: collections.defaultdict(Bucket))
    phase_cats: dict[str, dict[str, Bucket]] = field(default_factory=lambda: collections.defaultdict(lambda: collections.defaultdict(Bucket)))
    top: list[tuple[float, int, str, str]] = field(default_factory=list)
    unknown_top: list[tuple[float, int, str]] = field(default_factory=list)


def category(name: str) -> str:
    for cat, rx in KERNEL_CATS.items():
        if rx.search(name):
            return cat
    return "other"


def pct(num: float, den: float) -> float:
    return 0.0 if den <= 0 else 100.0 * num / den


def parse_rank(path: Path) -> str:
    m = re.search(r"TP-(\d+)", path.name)
    return f"TP{m.group(1)}" if m else "TP?"


def trace_label(annotations: collections.Counter[str], path: Path) -> str:
    names = set(annotations)
    if any("DRAFT_EXTEND" in n or "TARGET_VERIFY" in n for n in names):
        return "mtp_on_bs32"
    if any("DECODE bs=32" in n for n in names):
        return "mtp_off_decode_bs32"
    return path.stem


def phase_name(name: str) -> str:
    if "DRAFT_EXTEND" in name:
        return "draft_extend"
    if "TARGET_VERIFY" in name:
        return "target_verify"
    if "_all_gather" in name or "all_gather" in name:
        return "all_gather_annotation"
    if "CompiledFxGraph" in name:
        return "compiled_fx_graph"
    if "DECODE" in name:
        return "decode"
    return name[:40]


def best_phase_from_active(ts: float, dur: float, active: list[tuple[float, float, str]]) -> str:
    if not active:
        return "all"
    end = ts + dur
    best = ("unannotated", 0.0, math.inf)
    for s, e, name in active:
        ov = min(end, e) - max(ts, s)
        if ov <= 0:
            continue
        width = e - s
        # Prefer more-specific nested ranges when overlap ties.
        if ov > best[1] or (ov == best[1] and width < best[2]):
            best = (name, ov, width)
    return best[0]


def load_trace(path: Path) -> TraceSummary:
    with gzip.open(path, "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data)
    anns: collections.Counter[str] = collections.Counter()
    intervals: list[tuple[float, float, str]] = []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = str(e.get("cat", ""))
        if cat in ("user_annotation", "gpu_user_annotation"):
            name = str(e.get("name", "?"))
            anns[name] += 1
            if cat == "gpu_user_annotation":
                intervals.append((float(e.get("ts", 0.0)), float(e.get("ts", 0.0)) + float(e["dur"]), phase_name(name)))

    summary = TraceSummary(path=path, label=trace_label(anns, path), rank=parse_rank(path), annotations=anns)
    intervals.sort(key=lambda x: x[0])
    kernel_events: list[tuple[float, float, str]] = []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = str(e.get("cat", "")).lower()
        if cat not in ("kernel", "gpu_op", "gpu_memcpy", "gpu_memset"):
            continue
        name = str(e.get("name", "?"))
        dur = float(e["dur"])
        kernel_events.append((float(e.get("ts", 0.0)), dur, name))

    per_kernel: dict[str, Bucket] = collections.defaultdict(Bucket)
    active: list[tuple[float, float, str]] = []
    idx = 0
    for ts, dur, name in sorted(kernel_events, key=lambda x: x[0]):
        end = ts + dur
        while idx < len(intervals) and intervals[idx][0] <= end:
            active.append(intervals[idx])
            idx += 1
        active = [iv for iv in active if iv[1] > ts]
        c = category(name)
        phase = best_phase_from_active(ts, dur, active)
        summary.total_us += dur
        summary.launch_count += 1
        summary.cats[c].add(dur)
        summary.phase_cats[phase][c].add(dur)
        per_kernel[name].add(dur)

    summary.top = sorted(
        [(b.dur_us, b.count, category(name), name) for name, b in per_kernel.items()],
        reverse=True,
    )[:60]
    summary.unknown_top = sorted(
        [(b.dur_us, b.count, name) for name, b in per_kernel.items() if category(name) == "other"],
        reverse=True,
    )[:20]
    return summary


RESULT_RE = re.compile(
    r"\[RESULT\] conc=(?P<conc>\d+) in~(?P<in>\d+) out=(?P<out>\d+).*?"
    r"AGG output tok/s \(steady\) = (?P<agg>[0-9.]+).*?"
    r"TTFT s: p50=(?P<ttft50>[0-9.]+) p99=(?P<ttft99>[0-9.]+).*?"
    r"per-req decode tok/s: p50=(?P<dec50>[0-9.]+).*?tokens/chunk=(?P<tpc>[0-9.]+).*?"
    r"steady tokens=(?P<toks>[0-9.]+) errors=(?P<errs>\d+)",
    re.S,
)


def parse_serve_results(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    txt = path.read_text(errors="replace")
    rows = []
    for m in RESULT_RE.finditer(txt):
        row: dict[str, float | str] = {"file": path.name}
        for k, v in m.groupdict().items():
            row[k] = float(v) if k not in ("conc", "in", "out", "errs") else int(v)
        rows.append(row)
    return rows


def nsys_status(path: Path) -> dict[str, str]:
    txt = path.read_text(errors="replace")
    status = {
        "file": path.name,
        "oom": "yes" if "OutOfMemoryError" in txt else "no",
        "traceback": "yes" if "Traceback" in txt else "no",
        "generated": "yes" if "Generated:" in txt else "no",
        "bench_decode": "yes" if "Decode.  median latency" in txt else "no",
        "bench_prefill": "yes" if "Prefill. latency" in txt else "no",
    }
    m = re.search(r"Prefill\. latency:\s+([0-9.]+) s, throughput:\s+([0-9.]+)", txt)
    if m:
        status["prefill_s"] = m.group(1)
        status["prefill_tok_s"] = m.group(2)
    m = re.search(r"Decode\.  median latency:\s+([0-9.]+) s, median throughput:\s+([0-9.]+)", txt)
    if m:
        status["decode_median_s"] = m.group(1)
        status["decode_tok_s"] = m.group(2)
    return status


def parse_ncu_log(path: Path) -> dict[str, str | int]:
    txt = path.read_text(errors="replace")
    return {
        "file": path.name,
        "profiles": len(re.findall(r"==PROF== Profiling", txt)),
        "exit0": "yes" if re.search(r"\bexit=0\b", txt) else "no",
        "launchfailed": "yes" if "LaunchFailed" in txt else "no",
        "report": (re.findall(r"==PROF== Report:\s+(\S+)", txt) or [""])[-1],
    }


def parse_nsys_kernsum(path: Path) -> tuple[dict[str, Bucket], list[tuple[float, int, str, str]]]:
    buckets: dict[str, Bucket] = collections.defaultdict(Bucket)
    top: list[tuple[float, int, str, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(maxsplit=8)
        if len(parts) < 9:
            continue
        try:
            share = float(parts[0])
            total_ns = float(parts[1].replace(",", ""))
            instances = int(parts[2].replace(",", ""))
        except ValueError:
            continue
        name = parts[8]
        cat = category(name)
        dur_us = total_ns / 1000.0
        buckets[cat].add(dur_us, instances)
        top.append((share, instances, cat, name))
    return buckets, top


def mean_trace_by_label(traces: list[TraceSummary]) -> dict[str, dict[str, Bucket]]:
    grouped: dict[str, list[TraceSummary]] = collections.defaultdict(list)
    for t in traces:
        grouped[t.label].append(t)
    out: dict[str, dict[str, Bucket]] = {}
    for label, items in grouped.items():
        merged: dict[str, Bucket] = collections.defaultdict(Bucket)
        for item in items:
            for cat, b in item.cats.items():
                merged[cat].add(b.dur_us / len(items), int(round(b.count / len(items))))
        out[label] = merged
    return out


def rows_for_buckets(buckets: dict[str, Bucket], total_us: float | None = None, min_pct: float = 0.05) -> list[str]:
    total = total_us if total_us is not None else sum(b.dur_us for b in buckets.values())
    rows = []
    for cat, b in sorted(buckets.items(), key=lambda kv: -kv[1].dur_us):
        p = pct(b.dur_us, total)
        if p < min_pct:
            continue
        rows.append(f"| {DISPLAY.get(cat, cat)} | {p:5.1f}% | {b.dur_us/1000:8.2f} | {b.count} |")
    return rows


def write_report(root: Path, out: Path) -> None:
    traces = [load_trace(p) for p in sorted(root.glob("*.trace.json.gz"))]
    serve_files = [
        "e2e_mtp_off.txt", "e2e_mtp_off2.txt", "e2e_mtp_on.txt", "e2e_noAR.txt",
        "e2e_nograph.txt", "ceil_base.txt", "ceil_moe.txt", "ceil_gdn.txt", "ceil_attn.txt",
    ]
    serve_rows = []
    for name in serve_files:
        serve_rows.extend(parse_serve_results(root / name))

    lines: list[str] = []
    lines.append("# Qwen3.5 B300 Profile Analysis\n")
    lines.append(f"Artifact root: `{root}`\n")
    lines.append("## Artifact Health\n")
    lines.append(f"- Torch traces: {len(traces)}")
    lines.append(f"- NCU reports downloaded: {len(list(root.glob('*.ncu-rep')))}")
    lines.append(f"- NSYS reps downloaded: {len(list(root.glob('*.nsys-rep')))}")
    lines.append(f"- NCU raw CSV exports downloaded: {len(list(root.glob('*_raw.csv')))}")
    lines.append(f"- NCU details exports downloaded: {len(list(root.glob('*_details.txt')))}")
    lines.append("- Local note: this report parses Torch traces and text logs only; local Nsight CLIs were not assumed.\n")

    lines.append("## NCU Artifact Status\n")
    lines.append("| file | size | raw csv | details txt | log status | remote report path |")
    lines.append("|---|---:|---|---|---|---|")
    for rep in sorted(root.glob("*.ncu-rep")):
        stem = rep.with_suffix("").name
        raw = "yes" if (root / f"{stem}_raw.csv").exists() else "no"
        details = "yes" if (root / f"{stem}_details.txt").exists() else "no"
        log = root / f"{stem}.log"
        status = ""
        remote = ""
        if log.exists():
            st = parse_ncu_log(log)
            status = f"profiles={st['profiles']} exit0={st['exit0']} launchfailed={st['launchfailed']}"
            remote = str(st["report"])
        lines.append(f"| {rep.name} | {rep.stat().st_size/1024/1024:.1f} MiB | {raw} | {details} | {status} | `{remote}` |")
    for log in sorted(root.glob("ncu_*.log")):
        if (root / f"{log.with_suffix('').name}.ncu-rep").exists():
            continue
        st = parse_ncu_log(log)
        lines.append(f"| {log.name} | log only | no | no | profiles={st['profiles']} exit0={st['exit0']} launchfailed={st['launchfailed']} | `{st['report']}` |")
    lines.append("")

    lines.append("## E2E Serving Results\n")
    lines.append("| file | conc | agg tok/s | TTFT p50/p99 s | per-req decode p50 | tokens/chunk | errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in serve_rows:
        lines.append(
            f"| {r['file']} | {r['conc']} | {float(r['agg']):.1f} | "
            f"{float(r['ttft50']):.2f}/{float(r['ttft99']):.2f} | {float(r['dec50']):.1f} | "
            f"{float(r['tpc']):.2f} | {r['errs']} |"
        )
    lines.append("")

    lines.append("## NSYS Grid Status\n")
    lines.append("| file | generated | traceback | OOM | bench prefill | bench decode | timing |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in sorted(root.glob("nsys_*.log")):
        st = nsys_status(p)
        timing = []
        if "prefill_tok_s" in st:
            timing.append(f"prefill {st['prefill_tok_s']} tok/s")
        if "decode_tok_s" in st:
            timing.append(f"decode {st['decode_tok_s']} tok/s")
        lines.append(
            f"| {st['file']} | {st['generated']} | {st['traceback']} | {st['oom']} | "
            f"{st['bench_prefill']} | {st['bench_decode']} | {', '.join(timing)} |"
        )
    lines.append("")

    lines.append("## NSYS Kernel Summary Exports\n")
    lines.append("These tables are useful for setup/prefill/failed-window diagnosis, but they are not steady decode attribution unless the underlying run completed the intended decode window.\n")
    for p in sorted(root.glob("*_kernsum.txt")):
        buckets, top = parse_nsys_kernsum(p)
        total = sum(b.dur_us for b in buckets.values())
        lines.append(f"### {p.name}\n")
        lines.append(f"Total parsed GPU-kernel time: {total/1000:.2f} ms\n")
        lines.append("| category | share | ms | launches |")
        lines.append("|---|---:|---:|---:|")
        lines.extend(rows_for_buckets(buckets, total, min_pct=0.5))
        lines.append("")
        lines.append("| nsys share | launches | category | kernel |")
        lines.append("|---:|---:|---|---|")
        for share, instances, cat, name in top[:30]:
            lines.append(f"| {share:.1f}% | {instances} | {DISPLAY.get(cat, cat)} | `{name[:140]}` |")
        lines.append("")

    lines.append("## Torch Trace Decode Breakdown\n")
    grouped = mean_trace_by_label(traces)
    for label, buckets in grouped.items():
        total = sum(b.dur_us for b in buckets.values())
        launch = sum(b.count for b in buckets.values())
        lines.append(f"### {label}\n")
        lines.append(f"Mean per-rank GPU-kernel time: {total/1000:.2f} ms across ~{launch} launches.\n")
        lines.append("| category | share | ms | launches |")
        lines.append("|---|---:|---:|---:|")
        lines.extend(rows_for_buckets(buckets, total, min_pct=0.08))
        lines.append("")

    lines.append("## MTP-On Phase Split\n")
    for tr in traces:
        if tr.label != "mtp_on_bs32" or tr.rank != "TP0":
            continue
        for phase, buckets in sorted(tr.phase_cats.items()):
            total = sum(b.dur_us for b in buckets.values())
            if total <= 0:
                continue
            lines.append(f"### {tr.path.name} phase `{phase}`: {total/1000:.2f} ms\n")
            lines.append("| category | share | ms | launches |")
            lines.append("|---|---:|---:|---:|")
            lines.extend(rows_for_buckets(buckets, total, min_pct=0.5))
            lines.append("")

    lines.append("## Top Kernels By Trace\n")
    for tr in traces:
        lines.append(f"### {tr.path.name} ({tr.label}, {tr.rank})\n")
        lines.append("| share | ms | launches | category | kernel |")
        lines.append("|---:|---:|---:|---|---|")
        for dur, n, cat, name in tr.top[:35]:
            lines.append(f"| {pct(dur, tr.total_us):.2f}% | {dur/1000:.2f} | {n} | {DISPLAY.get(cat, cat)} | `{name[:140]}` |")
        lines.append("")

    lines.append("## Largest Uncategorized Kernels\n")
    for tr in traces:
        if not tr.unknown_top:
            continue
        lines.append(f"### {tr.path.name}\n")
        lines.append("| share | ms | launches | kernel |")
        lines.append("|---:|---:|---:|---|")
        for dur, n, name in tr.unknown_top[:12]:
            lines.append(f"| {pct(dur, tr.total_us):.2f}% | {dur/1000:.2f} | {n} | `{name[:160]}` |")
        lines.append("")

    lines.append("## Immediate Interpretation\n")
    lines.append("- The reliable steady decode breakdown is the Torch `step[DECODE bs=32]` trace. The `nsys_decode_b32` kernel summary is still useful, but as a setup/prefill/failed-extend profile: it shows graph capture, first-step, token-gather, block-scale-interleave, delayStream, and OOM-window behavior, not clean steady decode shares.")
    lines.append("- The MTP-off trace is compute-heavy and graph-covered: FP4 BMM/GEMM family plus nvjet dense/projection GEMMs dominate. Token gather and block-scale interleave are not meaningful steady-decode targets in this trace.")
    lines.append("- `e2e_nograph.txt` is the strongest runtime result: disabling CUDA graph collapses conc32 from ~1.5k tok/s to ~642 tok/s and conc64 from ~2.85k to ~625 tok/s. Any candidate that breaks graph capture is dead on arrival.")
    lines.append("- `ncu_fp4gemm_b32.ncu-rep` is downloaded and its log shows 8 FP4 GEMM launches captured successfully, but raw/details exports are still missing on this Mac because local `ncu` is not installed. The downloaded `ncu_glue_gdn.ncu-rep` captured one routing kernel then hit LaunchFailed; use it cautiously.")
    lines.append("")

    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=Path("~/Downloads/github/temp/profiles_b300").expanduser())
    ap.add_argument("--out", type=Path, default=Path("/private/tmp/qwen35_b300_profile_analysis.md"))
    args = ap.parse_args()
    write_report(args.root, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
