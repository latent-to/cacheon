# 🟢 H200 PROFILING SESSION — START HERE (next Claude: read this top-to-bottom, then act)

> Local working log, written 2026-06-02 from the CC-blind Targon B200 box. Purpose: a fresh
> agent on the new **8×H200** box should know in 2 minutes what to do and why. The B200 box
> (Targon) runs Confidential-Computing mode which **blocks CUPTI** → no nsys/ncu/torch-profiler,
> so we've been profiling blind (CUDA-events + ablation). This H200 box exists to get the
> **ground-truth, time-weighted per-kernel breakdown** we couldn't get, identify the exact
> kernels in the attention/NSA-indexer cluster to fuse, then go back to B200 to build+validate.
>
> Read `WORKLOG.md` → the "🎯 PERFORMANCE PLAYBOOK" + "🔬 PROFILING WISHLIST" sections for the
> full hard-won context. This file is the H200-specific action plan.

---

## 0. THE MISSION (one sentence)
Profile V4-Flash decode on H200 with **nsys + ncu** to produce a ranked table
`(kernel, % of decode step, % of HBM roofline, bottleneck reason, fix idea)` for the
**attention / NSA-indexer / host-dispatch** path — then bring it back to B200 to write fused kernels.

## 1. FIRST 15 MINUTES — verify this box can actually profile (the one GO/NO-GO)
Targon was CC-mode (CUPTI blocked). Standard H200 is NOT, but **confirm before anything else**:
```bash
# in the sglang container, single GPU:
python3 /opt/optima/_kineto_check.py     # already in the repo
# MUST print "KINETO_OK" with non-zero device_time. If "KINETO_EMPTY" -> this box is ALSO CC-mode;
# stop, tell the user, get a non-CC instance. Also try: nsys status -e   (look for CUPTI: available)
```
If KINETO_OK → CUPTI works → nsys/ncu/torch-profiler all work → proceed.

## 2. SETUP — serve V4-Flash on H200 (confirmed working; sources at bottom)
- **Checkpoint:** `sgl-project/DeepSeek-V4-Flash-FP8` (~284 GB FP8; Hopper has NO FP4 tensor cores, so
  the B200 FP4 experts are run as FP8 here). Alternative: `deepseek-ai/DeepSeek-V4-Flash` +
  `--moe-runner-backend marlin` (FP4 weights, W4A16, TP-only).
- **Image:** `lmsysorg/sglang:latest` (NOT `lmsysorg/sglang:deepseek-v4-blackwell` — that's sm100-only).
- **sglang version:** ≥ 0.5.12.
- **Offline engine for profiling** (TP=8 across the 8 H200s; mirror our Targon bench):
```python
e = sgl.Engine(model_path="sgl-project/DeepSeek-V4-Flash-FP8", tp_size=8, trust_remote_code=True,
               mem_fraction_static=0.85, disable_cuda_graph=False)   # NO EAGLE for throughput profiling
```
- Docker flags: `--privileged --gpus all --shm-size 32g --ulimit memlock=-1 -v <repo>:/opt/optima`.
- **Reuse `_decode_bench_ab.py`** (it's in the repo) as the load generator; just swap the model path
  to the FP8 checkpoint and tp_size=8. Warm up hard before timing (clocks ramp; same lesson as B200).

## 3. THE PROFILING PLAN (prioritized; ~2–3 h of GPU) — do in order
Decode is **memory-bound**; the deliverable is **time-weighted** per-kernel + **GPU-busy-vs-host-gap**.
1. **(30 min) nsys a decode step**, graphed, no-EAGLE, batch ∈ {1, 8, 32, 128} **and one 16k-context run.**
   - Easiest path: sglang's built-in torch-profiler — `export SGLANG_TORCH_PROFILER_DIR=/opt/optima/_trace`,
     start server, hit `/start_profile` during a decode load, `/stop_profile`, parse the per-rank chrome
     traces. OR `nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true -o /opt/optima/dec python3 _decode_bench_ab.py`
     (nsys must trace the spawned TP workers — `--trace-fork-before-exec` matters).
   - **Deliverable:** the per-kernel time-weighted breakdown + the host/dispatch gap %. This single
     capture replaces every ablation + eager-vprof guess we made on B200 (which were kernel-count-weighted
     and misled us — see WORKLOG trap #1).
2. **(45 min) ncu the top-5 time kernels** from step 1:
   `ncu --set full --launch-count 1 --kernel-name-base demangled -k <regex> -o /opt/optima/<name> python3 ...`
   Pull: **memory throughput (% of peak HBM BW)**, achieved occupancy, **warp-stall reasons**, the roofline.
   - **Deliverable:** the ranked table `(kernel, time%, HBM-roofline%, stall reason, fix idea)`.
3. **(30 min) ncu the research-named suspects specifically** (these are WHY we're here):
   - the **NSA indexer prep cluster** — `weights_proj` GEMM, `compute_q` (wq_b + rope + rotate), the FP8
     quant, the reshapes — file `layers/attention/compressed/indexer.py:309 forward_c4_indexer`.
   - `flashmla_decode` (maintainers say it underperforms on B200; check it on H200 too).
   - the MoE grouped-GEMM at low tokens/expert (note: H200 = Marlin/FP8, ≠ B200 FP4 — numbers won't
     transfer, but confirm whether it's launch-bound at low batch).
4. **(remaining) Long-context (16k) nsys** — the indexer is **O(N)/step**; confirm it dominates at long
   context (it reads ~0.98 GB of the 1.1 GB/step at 131k) and that comm overlaps compute.

## 4. WHAT WE ALREADY MEASURED ON B200 (so you know what to confirm/refute)
- **MBU ≈ 8–17%** (batch 8→128). 3–6× off the memory roofline; it's a MIX, not one kernel.
- **Ablation (graphed, real): removing the whole attention block = 1.35–1.76× → attention is
  26–43% of the in-graph decode step** (b8 37%, b32 26%, b128 43%). The FlashMLA *core* is only ~2.5%;
  the rest of that 26–43% is the **NSA indexer + compressor + MLA projections + ropes + glue** = the
  fusable target. (Confirm this split with nsys time-weighting; refine the indexer's own share.)
- **MoE FP4 GEMM is class-leading OSS / tuned** (don't try to beat it). At low batch the win is
  dispatch/host overhead (SGLang got 1.79× at concurrency-4 from host-overhead cuts alone).
- **Confirmed dead ends (don't repeat):** attention-backend flags, `wo_a` einsum→bmm (null e2e),
  hc_pre, racing FlashMLA/cublas/the MoE GEMM.

## 5. WHAT TO BRING BACK TO B200 (the output of this session)
A ranked list: **for each hot kernel — its % of the step, its % of HBM roofline, why it's slow, and the
fix** (fuse with neighbor / persistent kernel / reduce launches / better layout). Highest-value expected:
the **indexer `_forward_prepare` cluster fused into one kernel** (cuts ~6–8 launches), and any kernel ncu
shows far below HBM roofline. Then: **write the fused kernel on B200, route via the seam, validate with a
PAIRED/interleaved end-to-end A/B** (paired comparison works fine in CC; absolute kernel timing doesn't).

## 6. CAVEATS (do not get fooled)
- **MoE differs Hopper↔Blackwell** (FP8/Marlin vs FP4 CUTLASS). MoE-kernel numbers from H200 do NOT
  transfer to B200. But the **attention / NSA indexer / host-dispatch structure is identical** → that's
  our target → H200 profiling is valid for it. We are NOT optimizing the MoE (it's tuned).
- **Absolute throughput differs** (H200 ~4.8 TB/s vs B200 ~8 TB/s HBM; FP8 vs FP4). Use H200 to find
  WHICH kernels are far from roofline and WHY (structure), not for absolute B200 numbers.
- **Warmup noise** still applies (±6–17% at low batch); but on H200 you can likely **lock clocks**
  (`nvidia-smi -lgc`, which Targon denied) — DO IT if permitted, it kills the noise.
- Confirm non-CC (step 1) before trusting any profiler output.

## 7. HARNESS ALREADY IN THE REPO (works on H200 unchanged)
`_kineto_check.py` (CUPTI probe), `_decode_bench_ab.py` (warmup-controlled bench), `optima/vprof.py`
(CUDA-event eager profiler), `optima/ablate.py` (in-graph region stubbing), `_decode_bench_long.py`
(16k-context bench). The recompose/seam pattern is in `examples/miner_dsv4_hc_pre_recompose`.

---

### Sources (confirmed 2026-06-02)
- SGLang DeepSeek-V4 cookbook (Hopper checkpoints + serve flags): https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- LMSYS DeepSeek-V4 day-0 (benchmarks V4-Flash on H200, TP=4): https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- "Hopper has no FP4 hardware → use FP8 build `sgl-project/DeepSeek-V4-Flash-FP8` (~284 GB)" — multiple (SGLang docs, Verdent/Lushbinary/Runpod guides).
- FP8/W4A16 paths on Hopper (Marlin / flashinfer_mxfp4): SGLang docs + canada-quant/DeepSeek-V4-Flash-W4A16-FP8 (HF).

