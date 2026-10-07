# DeepSeek-V4.1-Flash development check inputs

Use these inputs with the commissioned `deepseek-ai/DeepSeek-V4.1-Flash`
checkpoint (revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`) and the SGLang
0.5.21 image on four B300 GPUs (TP4 with expert parallelism 4). Keep this folder
outside the submitted bundle. Mount it read-only at `/arena`, the model at
`/model`, and a writable result/cache directory at `/work`.

The runtime selects its attention, MoE and FP8 backends from the model; the
`dsv4` attention backend refuses deterministic inference, so the arena seals
`deterministic` as false and the audit relies on the paired replay and the row
tolerance instead. DSpark is the checkpoint's own speculative decoder: block
size five, no separate draft model and no step count. Prefix caching stays
enabled, matching the agent-replay runtime. The development config caps running
requests at 24 per lane to bound graph setup. Qualification owns its sealed
workload and configuration; this check does not produce a speed score.

Everything in this folder was derived from the 0.5.21 source and the published
recipes before the image was run; the first boot on the commissioned lane
confirms the image layout the Dockerfile assumes, the memory fraction, and the
page geometry of the index cache (64 or 128 slots, by the installed DeepGEMM).

Build the pinned image from the repository root with the selected revision:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build -t cacheon-dsv41flash \
  -f examples/arena_inputs/dsv41flash/Dockerfile "$build_dir"
```

Inside that image, run `python -m cacheon.cli compat --sglang-version 0.5.21`
before the development check.

Route the bundle to this arena in its existing competition table:

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "dsv41flash-b300-node-v1"
```

## Which node addresses execute

Addresses come from the served model's `named_modules()`, but this model's
layer loop does not call its decoder layers' `forward`: a bundle that names
`model.layers.*` binds and never runs, so it cannot qualify. The nodes that
execute are `model`, `model.layers.*.self_attn`, `model.layers.*.mlp`, the
hash-memory nodes `model.layers.1.engram` and `model.layers.14.engram`, and
`logits_processor`. The attention compressor and indexer are not modules with a
`forward` and cannot be named. Confirm the per-node call counts in the
development check's audit receipts before paying for a submission.

A `tree_cache` bundle uses `target = "prefix_cache"` with the same `mode` and
arena. The runtime factory is described in
[the cache contract](../../../docs/architecture/slot-contract.md#the-prefix-cache).
Development support does not advertise that target on a live arena; use the
arena's published target availability before submitting.

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The first batch contains eight public synthetic prompts, each tokenized to
exactly 8,192 tokens with this model's tokenizer. A second batch reuses 512-token
prefixes of those prompts, twice each. Both generate eight deterministic output
tokens with EOS stopping disabled. The 24 requests exercise prefill, DSpark
verification and prefix reuse while keeping the untimed, per-call audit bounded.
The text describes summing a list of integers and dividing by its length; each
prompt has a distinct numbered prefix. No hidden quality questions are included.

Use the default four-window minimum and 1,800-second limit per engine. The
checker runs every supplied request, then starts a fresh graph engine only
after the audit passes. A development PASS covers these inputs; the arena
still qualifies candidates on its complete served workload and quality set.
Larger batches, long context, and uneven request lengths remain necessary
when those regimes are part of the candidate's intended optimization.
