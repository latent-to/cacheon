# Distributed output projection

This bundle targets `collective.dp_output_projection_norm`. It contains the
V1 CUDA algorithm with corrected E4M3 scale rounding, its declared native build
and a prepared entry.
It does not modify SGLang modules. The validator's Cacheon adapter supplies the
fused call on stock SGLang 0.5.18.

The published capability domain is BF16 on four SM103 GPUs, with 1–32 equally
padded local rows, attention width 16384 and hidden width 6144. The algorithm
gathers attention rows, runs the vendor projection on each rank's output-column
weight shard, then gathers columns while adding residuals, normalizing and
optionally preparing linear-layout NVFP4 data for routed experts. The router and
shared expert retain the BF16 output. Each process group owns its communication
workspace, initialized before CUDA graph capture and reused by serialized calls.

NVFP4 block scales use double-precision multiplication and division, then round
through FP32 to nearest-even E4M3, matching the pinned PyTorch reference conversion.
An approximate reciprocal changes some midpoint ties and must not replace division.

The data-as-arrival exchange protocol follows the vLLM/TRT-LLM Lamport approach
also used by the incumbent DP exchange. The redistribution and fused payload are
the contribution; no claim of inventing that communication protocol is made.

From a Cacheon revision that contains this slot and adapter:

```bash
python -m cacheon.cli scan bundles/dp_output_projection
python -m cacheon.cli verify bundles/dp_output_projection \
  --device cuda --dtype bfloat16 --world-size 4
```

The scale-rounding regression uses the same native bundle entry on four GPUs:

```bash
python -m torch.distributed.run --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=4 \
  -m pytest -q tests/test_dp_output_projection.py -k native_quantizer
```

The verifier checks the declared tensor math, independent full-matrix reference,
changed-input CUDA graph replays and temporal cases. Serving performance and model
quality require the commissioned end-to-end consumer afterward. The earlier V1
private-fork measurement is development evidence; it is not a qualification of
this newly packaged bundle or a registration in an existing arena.

The registered adapter replaces the selected post-attention projection/preparation
call. Other dense, normalization and DP-exchange call sites retain their current
contributions. Use a runtime whose target catalog preserves those contributions;
the earlier whole-target displacement removed optimizations outside this ABI.
