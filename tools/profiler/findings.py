#!/usr/bin/env python3
"""Turn a normalized profile dataset into ranked, defensible findings.

This is the insight layer. ``ingest.py`` gives us *what ran and how fast*;
this module answers *where is the win, and how big can it possibly be* — and,
just as importantly, *where there is no win* (so we don't chase a vendor floor).

The core join: decode kernel-time **share** (torch trace) × bound-type
**verdict** (ncu) → per-category winnability, then an **Amdahl ceiling** on the
realistic end-to-end gain. Everything is derived from the raw numbers; the
thresholds are explicit constants you can argue with, not magic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- tunable thresholds (all explicit; argue with these, don't hide them) ---
MEM_BOUND_PCT = 60.0      # mem/dram throughput at/above this => memory-bound
COMPUTE_BOUND_PCT = 60.0  # compute throughput at/above this => compute-bound
DOMINANCE_GAP = 10.0      # one axis must beat the other by this to "win" the label
LATENCY_CEIL = 50.0       # if max(comp,mem) below this => under-utilised launch
LOW_WAVES = 0.5           # waves/SM below this => serialized tiny launch (fuse it)
FUSION_EFFICIENCY = 0.6   # fraction of glue time fusion realistically recovers
MIN_DECODE_PCT = 0.3      # ignore decode categories below this share


# --------------------------------------------------------------------------- #
# per-kernel bound-type verdict
# --------------------------------------------------------------------------- #
def bound_type(k: dict) -> tuple[str, str, bool]:
    """Return (bound_type, verdict_text, winnable_by_rewrite).

    Classification is throughput-based (DRAM/compute %) — those readings stay
    valid even under CLC. Occupancy/waves are only *supporting* evidence and are
    treated as unreliable when ``clc`` is set.
    """
    comp, mem, dram = k.get("comp"), k.get("mem"), k.get("dram")
    occ, waves, clc = k.get("occ"), k.get("waves"), k.get("clc")
    if comp is None or mem is None:
        return "unknown", "no valid ncu metrics (all-NaN / not captured)", False
    memx = max(mem, dram or 0)
    hi = max(comp, memx)
    if memx >= MEM_BOUND_PCT and memx - comp > DOMINANCE_GAP:
        return ("memory", f"MEMORY/BW-bound ({memx:.0f}% mem vs {comp:.0f}% compute) — "
                "vendor floor; lever is fewer bytes / fuse / bigger batch, NOT faster math", False)
    if comp >= COMPUTE_BOUND_PCT and comp - memx > DOMINANCE_GAP:
        if occ is not None and occ < 20 and not clc:
            return ("compute", f"compute {comp:.0f}% but LOW occupancy ({occ:.0f}%) — "
                    "occupancy-limited at this decode shape, not a clean wall; uncertain", False)
        return ("compute", f"COMPUTE-bound ({comp:.0f}%) — near vendor peak; don't rewrite", False)
    if hi < LATENCY_CEIL:
        # Both pipes idle => latency/launch-bound. The conclusion holds under CLC
        # (throughputs are real); only call it a FUSE win when not CLC-corrupted
        # and the launch is genuinely tiny.
        extra = f", waves={waves:.2f}" if waves is not None else ""
        if clc:
            return ("latency", f"low util ({hi:.0f}%{extra}) but CLC corrupts occupancy — "
                    "verify without thread-block clusters before trusting a fuse win", False)
        tag = "FUSE into adjacent GEMM" if (waves is not None and waves < LOW_WAVES) else "fuse / more parallelism"
        return ("latency", f"LATENCY/OCCUPANCY-bound (peak util {hi:.0f}%{extra}) — {tag}", True)
    return ("mixed", f"mixed (compute {comp:.0f}% / mem {memx:.0f}%) — inspect before committing", False)


def _decode_ncu_for_cat(ncu: list[dict]) -> tuple[dict[str, dict], set[str]]:
    """Pick a representative ncu kernel per category to characterize *decode*.

    Soundness rules (each one fixes a real phantom-win we hit):
      * only **decode-regime** captures (a prefill big-M GEMM is compute-bound;
        the decode skinny-M GEMM of the *same name* is memory-bound — never let
        the prefill capture speak for decode);
      * skip **cluster / CLC** kernels (their counters are depressed → fake
        "latency-bound"); those categories are flagged as cluster floors instead;
      * de-prioritise **bs=1** captures (phantom occupancy headroom that vanishes
        at the serving batch), then prefer the longest-duration kernel.

    Returns (representative-per-category, set-of-categories-that-are-cluster-floors).
    """
    cands: dict[str, list[dict]] = {}
    cluster_cats: set[str] = set()
    for cap in ncu:
        if cap.get("regime") != "decode":
            continue
        bs1 = cap.get("batch") == 1
        for k in cap.get("kernels", []):
            c = k.get("cat", "other")
            if k.get("cluster"):
                cluster_cats.add(c)
                continue
            if not k.get("valid"):
                continue
            cands.setdefault(c, []).append({**k, "capture": cap["label"], "_bs1": bs1})
    best: dict[str, dict] = {}
    for c, lst in cands.items():
        # non-bs1 first, then by duration desc
        lst.sort(key=lambda k: (k["_bs1"], -(k.get("dur_us") or 0)))
        best[c] = lst[0]
    return best, cluster_cats


# --------------------------------------------------------------------------- #
# canonical decode breakdown
# --------------------------------------------------------------------------- #
def _canonical_decode(decode: list[dict]) -> dict | None:
    """Prefer the clean steady-decode trace: mtp_off, rank TP0."""
    if not decode:
        return None
    pri = sorted(decode, key=lambda t: (
        0 if t.get("label") == "mtp_off" else 1,
        0 if t.get("rank") == "TP0" else 1,
        -t.get("total_us", 0),
    ))
    return pri[0]


# --------------------------------------------------------------------------- #
# e2e levers
# --------------------------------------------------------------------------- #
def _by_config(e2e: list[dict]) -> dict[str, dict[int, dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for r in e2e:
        grouped.setdefault((r["config"], r["conc"]), []).append(r)
    out: dict[str, dict[int, dict]] = {}
    for (cfg, conc), rows in grouped.items():
        row = dict(rows[-1])
        row["n"] = len(rows)
        for key in ("agg_toks", "ttft_p50", "ttft_p99", "decode_p50",
                    "tokens_per_chunk", "steady_tokens", "errors"):
            vals = [x.get(key) for x in rows if isinstance(x.get(key), (int, float))]
            if vals:
                row[key] = sum(vals) / len(vals)
        if len(rows) > 1:
            vals = [x.get("agg_toks") for x in rows if isinstance(x.get("agg_toks"), (int, float))]
            if vals:
                mu = sum(vals) / len(vals)
                row["agg_toks_min"] = min(vals)
                row["agg_toks_max"] = max(vals)
                row["agg_toks_spread_pct"] = round(100 * (max(vals) - min(vals)) / mu, 2) if mu else None
        out.setdefault(cfg, {})[conc] = row
    return out


def _levers(e2e: list[dict]) -> dict:
    cfg = _by_config(e2e)
    base = cfg.get("mtp_off", {})
    lev: dict = {"mtp": [], "cuda_graph": None, "all_reduce": [], "ceiling": []}

    on = cfg.get("mtp_on", {})
    for conc in sorted(set(base) & set(on)):
        b, o = base[conc]["agg_toks"], on[conc]["agg_toks"]
        lev["mtp"].append({"conc": conc, "mtp_off": b, "mtp_on": o,
                           "delta_pct": round(100 * (o - b) / b, 1) if b else None})

    ng = cfg.get("no_cuda_graph", {})
    pairs = [(c, base[c]["agg_toks"], ng[c]["agg_toks"]) for c in sorted(set(base) & set(ng))]
    if pairs:
        factor = max((g / n) for _, g, n in pairs if n)
        # The serving-regime factor (highest concurrency tested) is the honest
        # headline; the conc=1 ratio is much larger (launch overhead dominates a
        # single stream) but isn't how the box is served.
        hi_conc = max(c for c, _, n in pairs if n)
        serving = next(g / n for c, g, n in pairs if c == hi_conc and n)
        lev["cuda_graph"] = {
            "pairs": [{"conc": c, "graph": g, "no_graph": n,
                       "speedup": round(g / n, 2) if n else None} for c, g, n in pairs],
            "max_speedup": round(factor, 2),
            "serving_speedup": round(serving, 2),
            "serving_conc": hi_conc,
        }

    nar = cfg.get("no_all_reduce", {})
    for conc in sorted(set(base) & set(nar)):
        b, n = base[conc]["agg_toks"], nar[conc]["agg_toks"]
        lev["all_reduce"].append({"conc": conc, "with_fused_ar": b, "without": n,
                                  "delta_pct": round(100 * (n - b) / b, 1) if b else None})

    base_ceil = cfg.get("ceiling_none", {})
    for name, c in cfg.items():
        if not name.startswith("ceiling_noop_"):
            continue
        op = name.replace("ceiling_noop_", "")
        for conc, row in c.items():
            ref = base_ceil.get(conc)
            if ref:
                r0, r1 = ref["agg_toks"], row["agg_toks"]
                lev["ceiling"].append({
                    "op": op, "conc": conc, "baseline": r0, "noop": r1,
                    "apparent_share_pct": round(100 * (r1 - r0) / r1, 1) if r1 else None,
                })
    return lev


def _server_arg_values(dataset: dict, key: str) -> set[str]:
    rows = dataset.get("health", {}).get("sglang", {}).get("server_args", {}).get(key, [])
    return {str(r.get("value")) for r in rows}


def _has_server_arg(dataset: dict, key: str, value: str) -> bool:
    return value in _server_arg_values(dataset, key)


def _gdn_flashinfer_without_packed_decode(dataset: dict) -> dict | None:
    for row in dataset.get("health", {}).get("sglang", {}).get("gdn_dispatchers", []):
        if row.get("decode_kernel") == "FlashInferGDNKernel" and str(row.get("packed_decode")) == "False":
            return row
    return None


def _unknown_opportunity(e: dict, dataset: dict) -> dict:
    """Specialize unknown buckets with SGLang/runtime provenance when available."""
    cat = e["cat"]
    if cat == "gdn_scan":
        gdn = _gdn_flashinfer_without_packed_decode(dataset)
        if gdn:
            return {
                "title": f"A/B GDN decode backend: FlashInfer vs Triton packed decode ({e['pct']:.1f}% of decode)",
                "category": cat,
                "est_decode_gain_pct": None,
                "evidence": (
                    "profile logs show decode=FlashInferGDNKernel with packed_decode=False; "
                    "SGLang's TritonGDNKernel has a packed decode fast path, but this run did not exercise it"
                ),
                "action": (
                    "run interleaved e2e with explicit --linear-attn-decode-backend triton vs flashinfer "
                    "(same prefill/backend flags), then trace for the packed-decode kernel and apply KL/task gates"
                ),
            }
    if cat == "elementwise":
        return {
            "title": f"Attribute elementwise misc by callsite ({e['pct']:.1f}% of decode)",
            "category": cat,
            "est_decode_gain_pct": None,
            "evidence": "kernel names collapse real work into generic PyTorch elementwise/copy/add/sigmoid buckets",
            "action": (
                "add layer/function NVTX ranges around GDN, MoE, residual/norm, scheduler-state, and sampler paths; "
                "name-only taxonomy is not enough to decide fuse-vs-floor"
            ),
        }
    if cat == "splitk_reduce":
        return {
            "title": f"Profile splitKreduce ownership ({e['pct']:.1f}% of decode)",
            "category": cat,
            "est_decode_gain_pct": None,
            "evidence": "splitKreduce is a cublasLt/nvjet epilogue/reduction, not a standalone op slot by default",
            "action": (
                "join each splitKreduce launch to its parent GEMM shape and test cublasLt/nvjet algo knobs; "
                "do not count it as miner-fusible until that attribution is done"
            ),
        }
    if cat == "all_reduce":
        evidence = "no ncu capture; could be comm wall, overlap artifact, or batch-dependent"
        if _has_server_arg(dataset, "enable_flashinfer_allreduce_fusion", "True"):
            evidence = "SGLang logs show enable_flashinfer_allreduce_fusion=True, so the current path is already the FlashInfer fused all-reduce path"
        return {
            "title": f"Repeat all-reduce A/B under controlled conditions ({e['pct']:.1f}% of decode)",
            "category": cat,
            "est_decode_gain_pct": None,
            "evidence": evidence,
            "action": (
                "run interleaved with/without custom/fused all-reduce at the target TP and concurrency; "
                "only test symm_mem/NVLS/MSCCL variants if the rented pod topology supports them"
            ),
        }
    if cat == "attention":
        return {
            "title": f"NCU TRT-LLM attention at real context lengths ({e['pct']:.1f}% of decode)",
            "category": cat,
            "est_decode_gain_pct": None,
            "evidence": "trace shows vendor fmha/trtllm_mha kernels; no clean decode ncu capture in this artifact set",
            "action": (
                "capture serving-batch NCU for 16k and 64k contexts, then test attention-backend alternatives e2e; "
                "treat as a vendor floor until counters or a flag A/B prove otherwise"
            ),
        }
    if cat == "gdn_conv":
        return {
            "title": f"Profile GDN causal conv and surrounding copies ({e['pct']:.1f}% of decode)",
            "category": cat,
            "est_decode_gain_pct": None,
            "evidence": "causal_conv1d_update is separate from the GDN recurrence; backend A/B may shift this boundary",
            "action": (
                "profile after the GDN decode-backend A/B; if it remains, inspect conv state layout and graph-captured launch cost"
            ),
        }
    return {
        "title": f"PROFILE {e['display']} ({e['pct']:.1f}% of decode) — bound-type unknown",
        "category": cat,
        "est_decode_gain_pct": None,
        "evidence": "no ncu capture; could be a floor or a fusion target",
        "action": "rent a 1h ncu box at the serving batch (>=32) + a bs=1 control before committing",
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def derive(dataset: dict) -> dict:
    decode = dataset.get("decode", [])
    ncu = dataset.get("ncu", [])
    e2e = dataset.get("e2e", [])
    display = dataset.get("display", {})

    canon = _canonical_decode(decode)
    best_ncu, cluster_cats = _decode_ncu_for_cat(ncu)

    categories: list[dict] = []
    winnable_pct = floor_pct = unknown_pct = 0.0
    if canon:
        cats = canon.get("cats", {})
        for c, row in sorted(cats.items(), key=lambda kv: -kv[1]["pct"]):
            if row["pct"] < MIN_DECODE_PCT:
                continue
            k = best_ncu.get(c)
            if k:
                bt, verdict, win = bound_type(k)
                ev = {"capture": k.get("capture"), "comp": k.get("comp"), "mem": k.get("mem"),
                      "dram": k.get("dram"), "occ": k.get("occ"), "waves": k.get("waves"),
                      "kernel": k.get("kernel")}
            elif c in cluster_cats:
                bt, win, ev = "cluster", False, None
                verdict = ("vendor cluster kernel (thread-block clusters / Cluster-Launch-Control) — "
                           "ncu can't count it reliably and you can't fuse it into a GEMM; treat as a vendor floor")
            else:
                bt, verdict, win, ev = "unknown", "no clean decode ncu capture — PROFILE IT (serving batch >=32, bs1 control)", None, None
            entry = {
                "cat": c, "display": display.get(c, c), "pct": round(row["pct"], 2),
                "us": round(row["us"], 1), "count": row["count"],
                "bound_type": bt, "verdict": verdict, "winnable": win, "ncu": ev,
            }
            categories.append(entry)
            if win is True:
                winnable_pct += row["pct"]
            elif win is False:
                floor_pct += row["pct"]
            else:
                unknown_pct += row["pct"]

    # Amdahl ceiling
    wp = winnable_pct / 100.0
    max_decode_speedup = 1.0 / (1.0 - wp) if wp < 1 else float("inf")
    realistic_decode_gain = winnable_pct * FUSION_EFFICIENCY      # %, if fusion recovers EFFICIENCY of glue
    amdahl = {
        "winnable_pct": round(winnable_pct, 1),
        "floor_pct": round(floor_pct, 1),
        "unknown_pct": round(unknown_pct, 1),
        "max_decode_speedup_if_winnable_eliminated": round(max_decode_speedup, 3),
        "realistic_decode_gain_pct": round(realistic_decode_gain, 1),
        "assumptions": (
            f"'winnable' = ncu latency/occupancy-bound categories ({winnable_pct:.1f}% of decode kernel time). "
            f"Realistic gain assumes fusion recovers {FUSION_EFFICIENCY:.0%} of that glue time. "
            f"'floor' = memory/compute-bound at a vendor wall ({floor_pct:.1f}%) — not winnable by a kernel rewrite. "
            f"'unknown' = {unknown_pct:.1f}% not yet ncu-profiled."
        ),
    }

    levers = _levers(e2e)

    # peak throughput — over PRIMARY configs only (mtp_off/mtp_on), never an
    # ablation (no_all_reduce / no_cuda_graph), so the headline number is real.
    peak = None
    if e2e:
        primary = [r for r in e2e if r["config"] in ("mtp_off", "mtp_off_r2", "mtp_on")]
        pool = primary or [r for r in e2e if r["kind"] == "sweep"]
        best = max(pool, key=lambda r: r["agg_toks"], default=None)
        if best:
            peak = {"tok_s": best["agg_toks"], "conc": best["conc"], "config": best["config"]}

    # ranked opportunities + hard constraints
    opportunities = []
    for e in categories:
        if e["winnable"] is True:
            opportunities.append({
                "title": f"Fuse {e['display']} ({e['pct']:.1f}% of decode)",
                "category": e["cat"], "est_decode_gain_pct": round(e["pct"] * FUSION_EFFICIENCY, 1),
                "evidence": e["verdict"],
                "action": "eliminate the kernel-launch boundary (fold into adjacent GEMM prologue/epilogue); stay graph-capturable",
            })
    for e in categories:
        if e["winnable"] is None and e["pct"] >= 1.0:
            opportunities.append(_unknown_opportunity(e, dataset))
    opportunities.sort(key=lambda o: (o["est_decode_gain_pct"] is None, -(o["est_decode_gain_pct"] or 0)))

    constraints = []
    if levers.get("cuda_graph"):
        cg = levers["cuda_graph"]
        constraints.append(
            f"CUDA graphs are worth {cg['serving_speedup']}x e2e at serving conc{cg['serving_conc']} "
            f"(up to {cg['max_speedup']}x at conc1) — any kernel/seam MUST stay graph-capturable.")
    if levers.get("mtp"):
        flips = [m for m in levers["mtp"] if m["delta_pct"] is not None and m["delta_pct"] < 0]
        if flips:
            cc = min(m["conc"] for m in flips)
            constraints.append(f"MTP (spec-decode) flips NEGATIVE at conc>={cc} (GPU saturates) — serve MTP-off above it.")
    if levers.get("all_reduce"):
        small = all(abs(a["delta_pct"] or 0) < 5 for a in levers["all_reduce"])
        if small:
            constraints.append("Fused all-reduce ~neutral at this TP — comms is not a big e2e lever here (grows at TP4/8).")
    if _has_server_arg(dataset, "enable_flashinfer_allreduce_fusion", "True"):
        constraints.append("SGLang already had FlashInfer all-reduce fusion enabled in the captured runs — further comm wins need controlled TP/topology A/B, not just turning on the obvious flag.")
    if _gdn_flashinfer_without_packed_decode(dataset):
        constraints.append("GDN decode is NOT closed: captured runs used FlashInferGDNKernel with packed_decode=False, while SGLang has a Triton packed-decode path behind --linear-attn-decode-backend triton.")

    data_quality = list(dataset.get("health", {}).get("notes", []))
    if any(l for l in levers.get("ceiling", [])):
        data_quality.append("Ceiling (noop-op) numbers are single-run and NOISY — treat as directional; "
                            "re-verify interleaved + clock-locked before trusting op-shares.")

    headline = "no e2e data"
    if peak and canon:
        headline = (f"Peak {peak['tok_s']:.0f} tok/s @ conc{peak['conc']} ({peak['config']}). "
                    f"Decode winnable surface ≈ {winnable_pct:.0f}% (fusion), "
                    f"≈ {floor_pct:.0f}% is a vendor floor. "
                    f"Realistic decode gain ≈ {realistic_decode_gain:.0f}%; "
                    f"no 2× (Amdahl-capped at {max_decode_speedup:.2f}× even if all glue vanished).")

    return {
        "headline": headline,
        "peak": peak,
        "decode_canonical": {
            "label": canon.get("label") if canon else None,
            "rank": canon.get("rank") if canon else None,
            "file": canon.get("file") if canon else None,
            "total_us": canon.get("total_us") if canon else None,
            "categories": categories,
        },
        "amdahl": amdahl,
        "levers": levers,
        "opportunities": opportunities,
        "constraints": constraints,
        "data_quality": data_quality,
        "thresholds": {
            "MEM_BOUND_PCT": MEM_BOUND_PCT, "COMPUTE_BOUND_PCT": COMPUTE_BOUND_PCT,
            "DOMINANCE_GAP": DOMINANCE_GAP, "LATENCY_CEIL": LATENCY_CEIL,
            "LOW_WAVES": LOW_WAVES, "FUSION_EFFICIENCY": FUSION_EFFICIENCY,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_json", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    dataset = json.loads(args.dataset_json.read_text())
    f = derive(dataset)
    if args.out:
        args.out.write_text(json.dumps(f, indent=2))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()
