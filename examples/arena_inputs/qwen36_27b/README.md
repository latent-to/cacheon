# Qwen3.6-27B FP8 development inputs

This separate arena profile targets `Qwen/Qwen3.6-27B` using the official
[`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
checkpoint at the revision in `model-revision.json`. It uses one NVIDIA RTX
PRO 6000 Blackwell Server Edition (96 GB, SM120) per engine, TP1, and a native
262,144-token context including generated output.

The proposed selector is `qwen36-27b-rtxpro6000-fp8-tp1`, with `forward_pass`
contributions, a **0.15 TAO upload fee** (150,000,000 RAO), and a **7% allocation**
(70,000 ppm). These are commissioning inputs, not an active registration.
The complete intended future split is DeepSeek 83%, Qwen3.8 10%, and this arena
7%, once each source is independently qualified. The existing economics
authority must preserve historical submission terms when adding sources.

## Runtime and development check

Use the reviewed SGLang 0.5.20 runtime with Cacheon's native recurrent-state
audits. The official FP8 checkpoint has E4M3 weights in 128x128 blocks and
dynamic activation quantization. `dtype=bfloat16` leaves non-quantized operations
in BF16; it does not load the quantized linear weights as BF16. KV precision
uses the runtime default. Prefix caching and CUDA graphs remain enabled.

Mount the checkpoint read-only at `/model`, these inputs read-only at `/arena`,
and a writable result/cache directory at `/work`. Keep this folder outside
the submitted bundle. Inside the exact arena image:

```bash
python -m cacheon.cli compat --sglang-version 0.5.20
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The public synthetic requests exercise different prefixes, reuse and decode.
They are development inputs; the context limit does not make them a 256k
benchmark. Use a fresh output directory and the default four-window audit
minimum. The checker starts a separate graph engine after the audit passes.
See the [dated GPU results](../../../docs/results/qwen36-27b-pro6000.md)
for the whole-model controls and full-context capacity test.

## Commissioning boundary

`commission-engine-template.json` is the `engine_config` object consumed by
`cacheon.eval.b300_arena_definition.engine_template` in a sealed prompt
authority. Production derives request concurrency from its workload. The flat
development config allows eight running requests; this does not promise eight
simultaneous full-context requests. Both configurations share TP1, FP8,
262,144-token context, 4,096-token chunked prefill and 0.85 static memory fraction.

Policy-17 qualification uses two disjoint TP1 lanes for concurrent baseline and
candidate replay, then swaps physical orientations. Each engine uses one GPU;
the paired evaluator needs two. Reserve an explicit pair and drain its worker
before assigning either GPU to inference. Use this model's tokenizer and quality
authority, the current worker-bootstrap producers, and the final reviewed
image/source to generate identities. Enable the upload fee and allocation only
after paired replay, audit, pristine quality and recovery checks pass. New
intake must set `accept_legacy_bundles=false`. This profile supersedes no arena.
