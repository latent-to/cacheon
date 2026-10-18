# Muse-Spark-Open development check inputs

Use these inputs with the `Muse-Spark-Open-816B-A42B-NVFP4` checkpoint and the
`RadixArk/Muse-Spark-Open-DFlash2` draft on four B300 GPUs (TP4 with expert
parallelism 4). The ungated `unsloth/Muse-Spark-Open-816B-A42B-NVFP4` mirror
(revision `fc6908f53c6882d4ec48fca5ba144e41d21e1f92`) is file-for-file identical
to Meta's gated `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` (revision
`4239e4db19376904f2e82c167e78f70abb5455fe`): the same 128 LFS objects by SHA-256
and size. The draft revision is `46c90f412549c472941a68382d91a99e5e6372a0`.
Keep this folder outside the submitted bundle. Mount it read-only at `/arena`,
the model at `/cacheon/input/model` with the draft download inside it as
`/cacheon/input/model/dflash2` (a draft is sealed as one directory under the
model root, so the model identity covers its bytes and nothing else is
mounted), and a writable result/cache directory at `/work`.

The engine image is SGLang's day-0 Muse Spark image, pinned by digest in the
`Dockerfile` (`lmsysorg/sglang:dev-muse-spark-open@sha256:8cfc5a72…`, a
v0.5.21 base with the model runtime that reached SGLang main on 2026-10-18).
The configuration is the cookbook's B300 NVFP4 cell plus the capacity flags
InferenceX published for the same lane: `trtllm_mha` attention, page size 128,
FP8 KV cache, `swa-full-tokens-ratio` 0.1 and `mem-fraction-static` 0.85 (the
cookbook defaults, page size 64 and ratio 0.8, saturate the full-attention KV
pool at 24 sessions), 48 running requests, 48-row decode graphs, 16K breakable
prefill graphs, layernorm sequence parallel with MegaMoE prefill, and DFlash2
speculation with eight draft tokens and block verification. `engine_env`
carries `SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=1`, `TVM_FFI_CUDA_ARCH_LIST=10.3a`
and `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1`: SGLang's Muse hook
forces that last switch off, and with it off the sliding-window KV of cached
prefixes fills the SWA pool, the prefix hit rate at 24 sessions fell to 0.34 and
per-user decode to 15.7 tok/s; on it restores a 0.95 hit rate (measured on the
Verda 8xB300 node, 2026-10-18).
The hierarchical cache stays off; the first measured pool report decides
whether the arena seals it. `deterministic` is false, as on DeepSeek-V4.1-Flash.

A four-GPU lane makes one pair the whole eight-GPU node, so this arena runs one
qualification per node, as the GLM arenas did.

Build the qualification image from this repository root:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build -t cacheon-musespark -f examples/arena_inputs/musespark/Dockerfile "$build_dir"
```

Pass `--build-arg SGLANG_IMAGE=…` to build on the fork image instead. The
image build installs cacheon beside the image's SGLang and checks imports as
UID 65534. Inside that image, run
`python -m cacheon.cli compat --sglang-version "$(python3 -c 'import sglang; print(sglang.__version__)')"`
before the development check; the commissioned arena's runtime authority
carries the exact version it seals, as it did for the DeepSeek-V4.1-Flash fork
nightly.

Route the bundle to this arena in its existing competition table once the
arena is commissioned (the identifier below is provisional):

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "musespark-b300-node-v1"
```

## Which node addresses execute

Nothing has been read from a running engine yet. Confirm the per-node call
counts in the development check's audit receipts before paying for a
submission; DeepSeek-V4.1-Flash showed that a model's layer loop can bypass its
decoder layers' `forward`, so the executing addresses are an observation, not
an assumption.

`development-requests.json` follows the DeepSeek-V4.1-Flash recipe with this
model's tokenizer: the same eight public synthetic prompts, decoded from the
DeepSeek file with its tokenizer and re-tokenized with the Muse tokenizer (each
gives 8,236-8,263 tokens, cut to exactly 8,192), then two batches of the same
eight 512-token prefixes, eight output tokens each at zero temperature with EOS
stopping disabled. No special tokens are added, as in the DeepSeek file.

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /cacheon/input/model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

Use the default four-window minimum. Measured on the Verda 8xB300 node on
2026-10-18: a cold boot from network storage with an empty runtime cache took
933 s, and boots from a warm runtime seed took 243-283 s, so the 1,800-second
limit per engine holds only with a seed (`--runtime-seed`), as on
DeepSeek-V4.1-Flash.
