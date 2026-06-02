"""A/B decode-throughput harness for sglang's TP all-reduce backend — the *collective ceiling*.

Measures DECODE tok/s for ONE all-reduce backend, warmup-controlled and prefill-isolated:

* warmup-controlled — a hard pre-warm ramps GPU clocks to steady state before any timing
  (the warmup-artifact lesson: low-batch tok/s swings ±6-17% with clock ramp; a "win" that
  is really warmup drift has burned us before).
* prefill-isolated — each batch is timed at two output lengths and the prefill is subtracted,
  so the reported number is decode throughput, not prefill+decode. (Comm is ~34% of *decode*
  for the marlin champion; we must measure decode, not the whole request.)

The *dimension* swept is the tensor-parallel all-reduce implementation, selected purely via
stock sglang flags — no source patch, one pinned package (the consensus invariant). The goal:
find out whether ANY stock backend beats the default custom all-reduce (`two_shot`) at decode
shapes BEFORE we invest in a collective seam + a custom kernel. If none does and the default is
already near the bandwidth floor, the standalone-reduce lever is closed and only compute-comm
overlap (the escape-hatch tier) remains.

Driven by env so sweep.sh can orchestrate the matrix:
    ALLREDUCE_BACKEND  default|nccl|nccl_nvls|symm_mem|torch_symm_mem|mscclpp
    MODEL_PATH         default: deepseek-ai/DeepSeek-V4-Flash
    TP                 default: 4
    MOE_BACKEND        default: marlin (H200 champion); use flashinfer_mxfp4 on B200/sm100
    BATCHES            default: 32,128   (the steady decode regime / cuda-graph-max-bs)
    OUT_SHORT,OUT_LONG default: 64,320   (decode window measured = LONG-SHORT = 256 tokens)
    MEM_FRACTION       default: 0.85

NOTE: a flag *requests* a backend; sglang's per-message ``should_*()`` predicate decides which
kernel actually runs each call. ALWAYS confirm the kernel that ran with parse_allreduce_latency.py
(run the sweep with NSYS=1) — do not trust the flag alone.
"""

from __future__ import annotations

import os
import time

import sglang as sgl

# backend -> the stock sglang Engine kwargs that select that TP all-reduce path.
BACKENDS: dict[str, dict] = {
    "default": {},  # custom all-reduce (the `two_shot` kernel we measured at ~333us avg) — baseline
    "nccl": {"disable_custom_all_reduce": True},  # drop custom AR -> NCCL ring / LL
    "nccl_nvls": {"disable_custom_all_reduce": True, "enable_nccl_nvls": True},  # NCCL NVSwitch in-network reduce
    "symm_mem": {"enable_symm_mem": True},  # pynccl symmetric-memory (NVLS) path
    "torch_symm_mem": {"enable_torch_symm_mem": True},  # PyTorch multimem all-reduce
    "mscclpp": {"enable_mscclpp": True},  # mscclpp small-message AR (falls back to NCCL)
}

_PROMPT = "Explain step by step how an out-of-order superscalar CPU executes instructions."


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def main() -> None:
    backend = _env("ALLREDUCE_BACKEND", "default")
    if backend not in BACKENDS:
        print(f"RESULT backend={backend} : BAD_BACKEND (known: {','.join(BACKENDS)})", flush=True)
        return

    batches = [int(b) for b in _env("BATCHES", "32,128").split(",")]
    out_short, out_long = int(_env("OUT_SHORT", "64")), int(_env("OUT_LONG", "320"))
    window = out_long - out_short

    kw = dict(
        model_path=_env("MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        tp_size=int(_env("TP", "4")),
        trust_remote_code=True,
        mem_fraction_static=float(_env("MEM_FRACTION", "0.85")),
        moe_runner_backend=_env("MOE_BACKEND", "marlin"),
        chunked_prefill_size=4096,
        disable_flashinfer_autotune=True,
    )
    # Per-box base engine config merged over the defaults — e.g. B200 V4-Flash needs
    # swa_full_tokens_ratio + moe_runner_backend=flashinfer_mxfp4 to even init. Point
    # ENGINE_KWARGS_JSON at the same json the working run used; the swept all-reduce
    # backend is applied LAST so it always wins.
    ek_path = _env("ENGINE_KWARGS_JSON", "")
    if ek_path:
        import json

        with open(ek_path) as f:
            kw.update(json.load(f))
    kw.update(BACKENDS[backend])
    print(
        f"CONFIG backend={backend} applied={BACKENDS[backend]} | "
        f"model={kw['model_path']} tp={kw['tp_size']} moe={kw['moe_runner_backend']} "
        f"batches={batches} decode_window={window}",
        flush=True,
    )

    try:
        engine = sgl.Engine(**kw)
    except Exception as ex:  # noqa: BLE001 - a backend may be unavailable on this box; record and move on
        print(f"RESULT backend={backend} : ENGINE_FAILED {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        return

    def run(batch: int, n_tokens: int) -> float:
        sp = {"temperature": 0.0, "max_new_tokens": n_tokens, "ignore_eos": True}
        t0 = time.time()
        engine.generate([_PROMPT] * batch, sp)
        return time.time() - t0

    # HARD warmup: batch 32 x 512 tokens ramps the GPU to steady clocks before timing.
    run(32, 512)

    for batch in batches:
        try:
            t_short = run(batch, out_short)
            t_long = run(batch, out_long)
            decode_dt = t_long - t_short  # prefill + the first out_short tokens cancel out
            decode_tokens = window * batch  # exact: ignore_eos forces every seq to emit max_new_tokens
            decode_tps = decode_tokens / decode_dt if decode_dt > 0 else float("nan")
            raw_tps = out_long * batch / t_long
            print(
                f"RESULT backend={backend:<14} batch={batch:>3} : "
                f"decode {decode_tps:8.1f} tok/s | raw {raw_tps:8.1f} tok/s | per-seq {decode_tps / batch:6.2f}",
                flush=True,
            )
        except Exception as ex:  # noqa: BLE001
            print(f"RESULT backend={backend:<14} batch={batch:>3} : FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)

    engine.shutdown()


if __name__ == "__main__":
    main()
