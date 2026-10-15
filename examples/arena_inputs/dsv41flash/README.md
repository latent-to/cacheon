# DeepSeek-V4.1-Flash development check inputs

Use these inputs with the commissioned `nvidia/DeepSeek-V4.1-Flash-NVFP4`
checkpoint (revision `3431dde3247c13b5957f682b1e3c6fcae2566079`) and the
[SGLang fork at `4f335dadf136c65619aa070cb0719a5f330d8659`](https://github.com/latent-to/sglang/commit/4f335dadf136c65619aa070cb0719a5f330d8659)
on two B300 GPUs (TP2 with expert parallelism 1). Keep this folder
outside the submitted bundle. Mount it read-only at `/arena`, the model at
`/model`, and a writable result/cache directory at `/work`.

The runtime selects its attention, MoE and FP8 backends from the model; the
`dsv4` attention backend refuses deterministic inference, so the arena seals
`deterministic` as false and the audit relies on the paired replay and the row
tolerance instead. DSpark is the checkpoint's own speculative decoder: block
size five, no separate draft model and no step count. Prefix caching stays
enabled, matching the agent-replay runtime. The development config permits 48
running requests, uses 16K breakable prefill graphs, bounded decoder replay,
64-row decode graphs and prefill/decode interval 96. The commissioned arena can
seal `max_running_requests` independently of its replay concurrency; omitting it
retains the engine default. Qualification owns its sealed
workload and configuration; this check does not produce a speed score.

The checkpoint's Engram hash-memory tables do not fit beside the weights at TP2,
so the engine keeps them in pinned host memory: `engine-config.json` carries an
`engine_env` object setting `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and
`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`, which `cacheon check` exports
to the engine and its ranks before construction. Each rank pins its own shard,
about 95 GiB, so one TP2 engine needs about 190 GiB of free host memory plus the
huge pages SGLang requests; a paired qualification runs two engines, and a box
with two pairs needs twice that again. The sealed arena configuration carries the
same two variables.

Build the fork's Dockerfile at the selected commit and set `FORK_IMAGE` to its
image reference. Then build the qualification image from this repository root:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build --build-arg SGLANG_IMAGE="${FORK_IMAGE:?Set the built fork image}" -t cacheon-dsv41flash \
  -f examples/arena_inputs/dsv41flash/Dockerfile "$build_dir"
```

Inside that image, run
`python -m cacheon.cli compat --sglang-version 0.0.0.dev1+g684fea8af`
before the development check.

Retain its source revision and inspected image ID with the build. The image
build makes the copied SGLang package readable and checks imports as UID 65534;
a root-only import check does not establish runtime access. The new worker must pass
commissioning against its installed SGLang version and receive fresh image,
runtime and service identities; prior runtime results do not qualify it.

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
Both `forward_pass` and `prefix_cache` are open on this arena.

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The first batch contains eight public synthetic prompts, each tokenized to
exactly 8,192 tokens with this model's tokenizer. Two subsequent batches each
contain the same eight 512-token prefixes. The last batch starts after the
preceding batch finishes, allowing prefix reuse. All three generate eight output
tokens with zero temperature and EOS stopping disabled. The 24 requests exercise
prefill, DSpark verification and prefix reuse while keeping the untimed,
per-call audit bounded.
The text describes summing a list of integers and dividing by its length; each
prompt has a distinct numbered prefix. No hidden quality questions are included.

Use the default four-window minimum and 1,800-second limit per engine. The
checker runs every supplied request, then starts a fresh graph engine only
after the audit passes. A development PASS covers these inputs; the arena
still qualifies candidates on its complete served workload and quality set.
Larger batches, long context, and uneven request lengths remain necessary
when those regimes are part of the candidate's intended optimization.
