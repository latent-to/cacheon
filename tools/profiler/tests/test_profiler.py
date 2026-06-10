#!/usr/bin/env python3
"""Tests for the profiler pipeline.

These use tiny synthetic fixtures (no GPU, no real artifacts) and deliberately
encode the anti-phantom-win guarantees so they can't silently regress:
  * a prefill capture must NOT characterize a decode category;
  * a cluster / CLC kernel must NOT be reported as a fusion win;
  * a bs=1 capture must NOT be preferred over the serving-batch capture;
  * the Amdahl ceiling math is exact.

Run:  python3 -m pytest tools/profiler/tests/ -q
   or: python3 tools/profiler/tests/test_profiler.py   (no pytest needed)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ingest          # noqa: E402
import findings as fnd  # noqa: E402
import compare as cmp   # noqa: E402
import plan as plan_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# minimal dataset builder for compare.py tests (no dir / no trace parsing)
# --------------------------------------------------------------------------- #
def _mini(peak_tok, cats):
    """cats: list of (cat, pct, count, bound, winnable)."""
    return {
        "display": {c[0]: c[0] for c in cats},
        "meta": {"datadir": f"ds@{peak_tok}"},
        "e2e": [{"config": "mtp_off", "conc": 64, "kind": "sweep", "agg_toks": peak_tok}],
        "findings": {
            "peak": {"tok_s": peak_tok, "conc": 64, "config": "mtp_off"},
            "decode_canonical": {"label": "mtp_off", "rank": "TP0", "categories": [
                {"cat": c, "display": c, "pct": p, "count": n, "us": p * 10,
                 "bound_type": b, "winnable": w, "verdict": "", "ncu": None}
                for (c, p, n, b, w) in cats]},
        },
    }


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_details(p: Path, blocks: list[tuple[str, dict, bool]]):
    """blocks: (kernel_name, {metric_label: value}, clc_flag)."""
    lines = ["[123] python@host"]
    for name, metrics, clc in blocks:
        lines.append(f"  {name} (32, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.3")
        if clc:
            lines.append("    Warning: The result was collected with the Work ID/Cluster Launch Control (CLC) feature enabled.")
        lines.append("    Section: GPU Speed Of Light Throughput")
        for label, val in metrics.items():
            lines.append(f"    {label}    %    {val}")
        lines.append("    Section: Occupancy")
    p.write_text("\n".join(lines))


def _make_dataset(tmp: Path):
    # decode @ bs32: FP4 GEMM is memory-bound (the truth). NOTE clc=True — the FP4
    # cutlass GEMM uses 2cta thread-block clusters, so ncu emits the CLC warning,
    # but its 80% DRAM reading is valid. It must NOT be excluded as "cluster".
    _write_details(tmp / "ncu_fp4gemm_b32_details.txt", [
        ("bmm_E2m1_E2m1E2m1_t128x32x512", {"Compute (SM) Throughput": 49, "Memory Throughput": 80, "DRAM Throughput": 80, "Achieved Occupancy": 25}, True),
    ])
    # decode @ bs1: same kernel reads LOW dram (the phantom) — must be ignored in favour of bs32
    _write_details(tmp / "ncu_fp4gemm_b1_details.txt", [
        ("bmm_E2m1_E2m1E2m1_t128x8x512", {"Compute (SM) Throughput": 14, "Memory Throughput": 22, "DRAM Throughput": 22, "Achieved Occupancy": 20}, True),
    ])
    # PREFILL: same FP4 family but big-M => compute-bound. Must NOT speak for decode.
    _write_details(tmp / "ncu_prefill_details.txt", [
        ("bmm_E2m1_E2m1E2m1_t128x256x512", {"Compute (SM) Throughput": 82, "Memory Throughput": 30, "DRAM Throughput": 30, "Achieved Occupancy": 60}, False),
    ])
    # decode glue: a CLUSTER routing kernel (CLC) reads fake-low util -> must not be a "fuse" win
    _write_details(tmp / "ncu_glue_gdn_details.txt", [
        ("void routingIndicesClusterKernel<KernelParams<512,16>>", {"Compute (SM) Throughput": 4, "Memory Throughput": 6, "DRAM Throughput": 6, "Achieved Occupancy": 12}, True),
        ("void moe::finalizeKernelVecLoad<T>", {"Compute (SM) Throughput": 6, "Memory Throughput": 7, "DRAM Throughput": 7, "Achieved Occupancy": 12}, False),
    ])

    # torch decode trace (mtp_off, TP0)
    events = []

    def kern(name, dur, n):
        for _ in range(n):
            events.append({"ph": "X", "cat": "kernel", "name": name, "dur": dur, "ts": 0})

    kern("bmm_E2m1_E2m1E2m1_t128x32x512_decode", 100, 39)   # 39% FP4 MoE GEMM (memory floor)
    kern("nvjet_sm103_gemm", 100, 25)                       # 25% dense GEMM
    kern("flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel", 100, 4)
    kern("void routingIndicesClusterKernel<...>", 100, 5)   # 5% routing (cluster floor)
    kern("kernel_cutlass_gdn_wide_vec_kernel", 100, 3)       # 3% GDN recurrence (backend provenance needed)
    kern("void moe::finalizeKernelVecLoad<T>", 100, 2)      # 2% finalize (fuse win)
    kern("act_and_mul_kernel", 100, 1)                      # 1% act (no ncu -> unknown)
    with gzip.open(tmp / "run.1234-TP-0.trace.json.gz", "wt") as fh:
        json.dump({"traceEvents": events}, fh)

    # e2e sweeps
    (tmp / "e2e_mtp_off.txt").write_text(_serve(64, 2800) + _serve(32, 1500))
    (tmp / "e2e_mtp_on.txt").write_text(_serve(64, 2400) + _serve(32, 1900))
    (tmp / "e2e_nograph.txt").write_text(_serve(64, 620) + _serve(32, 640))
    (tmp / "e2e_noAR.txt").write_text(_serve(64, 2950))   # ablation: must NOT be the reported peak
    # a bogus ncu summary export (wrong --page)
    (tmp / "ncu_fp4gemm_b32_summary.txt").write_text("==ERROR== the argument for option '--page' is invalid.")
    (tmp / "serve_off.log").write_text(
        "[2026-06-09] SM100+ detected with mamba-ssm-dtype=bfloat16, "
        "defaulting --linear-attn-decode-backend to flashinfer.\n"
        "[2026-06-09] FlashInfer TRTLLM MoE is enabled. "
        "--disable-shared-experts-fusion is automatically set.\n"
        "server_args=ServerArgs(attention_backend='trtllm_mha', "
        "fp4_gemm_runner_backend='flashinfer_cutlass', moe_runner_backend='flashinfer_trtllm', "
        "mamba_backend='triton', mamba_ssm_dtype='bfloat16', linear_attn_backend='triton', "
        "linear_attn_decode_backend='flashinfer', enable_flashinfer_allreduce_fusion=True, "
        "disable_custom_all_reduce=False, disable_shared_experts_fusion=True, "
        "enable_nccl_nvls=False, enable_symm_mem=False)\n"
        "[2026-06-09 TP0] Linear attention kernel backend: decode=flashinfer, prefill=triton\n"
        "[2026-06-09 TP0] GDN kernel dispatcher: decode=FlashInferGDNKernel, "
        "extend=TritonGDNKernel, verify=TritonGDNKernel packed_decode=False\n"
    )


def _serve(conc, agg):
    return (f"----- conc={conc} -----\n[RESULT] conc={conc} in~16384 out=1024 dur=45s steady=23s\n"
            f"  AGG output tok/s (steady) = {agg}   per-stream = {agg/conc:.1f}\n"
            f"  TTFT s: p50=1.00 p99=2.00 (n=8)\n"
            f"  per-req decode tok/s: p50=40.0   tokens/chunk=1.00\n"
            f"  steady tokens=1000 errors=0\nSERVE_LOAD2_DONE\n")


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _run(tmp: Path):
    ds = ingest.ingest(tmp).to_dict()
    return ds, fnd.derive(ds)


def test_clc_cluster_detection(tmp):
    ds, _ = _run(tmp)
    caps = {c["label"]: c for c in ds["ncu"]}
    routing = [k for k in caps["glue_gdn"]["kernels"] if "routing" in k["kernel"].lower()][0]
    assert routing["clc"] is True and routing["cluster"] is True, "CLC/cluster routing kernel not flagged"
    finalize = [k for k in caps["glue_gdn"]["kernels"] if "finalize" in k["kernel"].lower()][0]
    assert finalize["cluster"] is False, "finalize wrongly flagged as cluster"
    # the separation that fixed the FP4 mislabel: CLC warning != structural cluster
    fp4 = caps["fp4gemm_b32"]["kernels"][0]
    assert fp4["clc"] is True and fp4["cluster"] is False, "FP4 GEMM (clc, not name-cluster) misflagged"


def test_regime_and_batch_tags(tmp):
    ds, _ = _run(tmp)
    caps = {c["label"]: c for c in ds["ncu"]}
    assert caps["prefill"]["regime"] == "prefill"
    assert caps["fp4gemm_b32"]["regime"] == "decode" and caps["fp4gemm_b32"]["batch"] == 32
    assert caps["fp4gemm_b1"]["batch"] == 1


def test_fp4_is_memory_bound_from_decode_not_prefill(tmp):
    """The headline correctness check: decode FP4 GEMM must read MEMORY-bound
    (bs32, 80% dram), never compute-bound (which is the prefill capture)."""
    _, f = _run(tmp)
    fp4 = [c for c in f["decode_canonical"]["categories"] if c["cat"] == "fp4_moe_gemm"][0]
    assert fp4["bound_type"] == "memory", f"expected memory-bound, got {fp4['bound_type']}"
    assert fp4["winnable"] is False
    assert fp4["ncu"]["capture"] == "fp4gemm_b32", "did not prefer serving-batch capture"


def test_routing_is_not_a_fusion_win(tmp):
    """The phantom-win guard: a cluster/CLC kernel must never be 'winnable'."""
    _, f = _run(tmp)
    routing = [c for c in f["decode_canonical"]["categories"] if c["cat"] == "moe_routing"][0]
    assert routing["winnable"] is False, "routing cluster kernel reported as winnable!"
    assert routing["bound_type"] == "cluster"
    assert not any("routing" in o["category"] for o in f["opportunities"] if o["est_decode_gain_pct"]), \
        "routing appears as a fusion opportunity"


def test_finalize_is_a_fusion_win(tmp):
    _, f = _run(tmp)
    fin = [c for c in f["decode_canonical"]["categories"] if c["cat"] == "moe_finalize"][0]
    assert fin["winnable"] is True and fin["bound_type"] == "latency"


def test_peak_is_primary_not_ablation(tmp):
    _, f = _run(tmp)
    assert f["peak"]["config"] in ("mtp_off", "mtp_on"), f"peak picked an ablation: {f['peak']}"
    assert f["peak"]["tok_s"] == 2800


def test_amdahl_ceiling_math(tmp):
    _, f = _run(tmp)
    a = f["amdahl"]
    # winnable categories here: finalize(2%) only (act is unknown w/o ncu). floor: fp4 39 + dense? dense has no ncu -> unknown.
    # exact ceiling = 1/(1 - winnable/100)
    expected = round(1.0 / (1.0 - a["winnable_pct"] / 100.0), 3)
    assert abs(a["max_decode_speedup_if_winnable_eliminated"] - expected) <= 0.002
    assert a["winnable_pct"] + a["floor_pct"] + a["unknown_pct"] <= 100.2


def test_bogus_summary_flagged(tmp):
    ds, _ = _run(tmp)
    assert any("summary" in s for s in ds["health"]["bogus_summary_exports"])


def test_trace_cache_roundtrips(tmp):
    """Second parse must hit the cache and return an identical summary."""
    trace = next(tmp.glob("*.trace.json.gz"))
    a = ingest.parse_torch_trace(trace, use_cache=True)
    assert ingest._trace_cache_path(trace).exists(), "cache file not written"
    b = ingest.parse_torch_trace(trace, use_cache=True)
    assert a == b, "cached parse differs from fresh parse"
    # a no-cache parse must still match
    assert ingest.parse_torch_trace(trace, use_cache=False) == a


def test_v2_ceiling_names_are_normalized(tmp):
    p = tmp / "e2e_ceil_moe_r2.txt"
    p.write_text(_serve(32, 1234))
    rows = ingest.parse_serve_log(p)
    assert rows[0]["config"] == "ceiling_noop_moe"
    assert rows[0]["kind"] == "ceiling"
    assert rows[0]["replicate"] == "r2"


def test_replicated_e2e_rows_are_averaged(tmp):
    rows = [
        {"config": "ceiling_none", "conc": 32, "agg_toks": 1000.0, "errors": 0},
        {"config": "ceiling_none", "conc": 32, "agg_toks": 1100.0, "errors": 0},
    ]
    by = fnd._by_config(rows)
    r = by["ceiling_none"][32]
    assert r["agg_toks"] == 1050.0
    assert r["n"] == 2
    assert r["agg_toks_min"] == 1000.0 and r["agg_toks_max"] == 1100.0


def test_sglang_runtime_backend_observations(tmp):
    ds, _ = _run(tmp)
    sg = ds["health"]["sglang"]
    linear = sg["linear_attn_backends"][0]
    assert linear["decode"] == "flashinfer" and linear["prefill"] == "triton"
    gdn = sg["gdn_dispatchers"][0]
    assert gdn["decode_kernel"] == "FlashInferGDNKernel"
    assert gdn["extend_kernel"] == "TritonGDNKernel"
    assert gdn["packed_decode"] == "False"
    assert sg["server_args"]["linear_attn_decode_backend"][0]["value"] == "flashinfer"


def test_gdn_backend_gap_becomes_opportunity(tmp):
    _, f = _run(tmp)
    assert any("GDN decode backend" in o["title"] for o in f["opportunities"]), \
        "FlashInfer/no-packed GDN provenance did not become an explicit opportunity"
    assert any("GDN decode is NOT closed" in c for c in f["constraints"])


def test_capture_plan_closes_grey_buckets(tmp):
    ds, f = _run(tmp)
    ds["findings"] = f
    p = plan_mod.plan(ds)
    ab_names = [x["name"] for x in p["backend_abs"]]
    assert "gdn_decode_backend" in ab_names
    labels = {x["label"] for x in p["ncu_rows"]}
    assert "nvjet" in labels
    assert "gdnscan" in labels
    assert "finalize" not in labels, "already-captured NCU labels should not be re-planned"
    assert p["comm_actions"], "all-reduce should become e2e/nsys action, not NCU replay"


def test_compare_win_with_fusion(tmp):
    base = _mini(2800, [("fp4_moe_gemm", 39, 39, "memory", False), ("moe_finalize", 2.0, 40, "latency", True)])
    patched = _mini(3000, [("fp4_moe_gemm", 41, 39, "memory", False), ("moe_finalize", 0.1, 2, "latency", True)])
    c = cmp.compare(base, patched, noise_pct=2.0)
    assert c["win"] is True, c["headline"]
    assert c["fused"] and c["fused"][0]["cat"] == "moe_finalize", "fusion not detected"
    assert "WIN" in c["headline"] and "corroborated" in c["headline"]


def test_compare_inconclusive_within_noise(tmp):
    base = _mini(2800, [("fp4_moe_gemm", 39, 39, "memory", False)])
    patched = _mini(2830, [("fp4_moe_gemm", 39, 39, "memory", False)])  # +1.07% < 2% noise
    c = cmp.compare(base, patched, noise_pct=2.0)
    assert c["win"] is None and "INCONCLUSIVE" in c["headline"], c["headline"]


def test_compare_regression(tmp):
    base = _mini(2800, [("fp4_moe_gemm", 39, 39, "memory", False)])
    patched = _mini(2600, [("fp4_moe_gemm", 39, 39, "memory", False)])
    c = cmp.compare(base, patched, noise_pct=2.0)
    assert c["win"] is False and "REGRESSION" in c["headline"], c["headline"]


def test_compare_apparent_win_no_structure(tmp):
    """e2e up but no glue category fused → flagged as possible clock noise, not a clean win."""
    base = _mini(2800, [("fp4_moe_gemm", 39, 39, "memory", False), ("moe_finalize", 2.0, 40, "latency", True)])
    patched = _mini(3050, [("fp4_moe_gemm", 39, 39, "memory", False), ("moe_finalize", 2.0, 40, "latency", True)])
    c = cmp.compare(base, patched, noise_pct=2.0)
    assert "APPARENT WIN" in c["headline"] and c["cautions"], c["headline"]


def main():
    import tempfile
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _make_dataset(tmp)
            try:
                t(tmp)
                print(f"  PASS  {t.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {t.__name__}: {e}")
                raise
    print(f"\n{passed}/{len(tests)} passed")


# pytest fixture
try:
    import pytest

    @pytest.fixture
    def tmp(tmp_path):
        _make_dataset(tmp_path)
        return tmp_path
except ImportError:
    pass


if __name__ == "__main__":
    main()
