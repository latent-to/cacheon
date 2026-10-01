# Local quickstart

This quickstart exercises bundle parsing, static policy, and the node interface smoke on
CPU. It does not reproduce production qualification and cannot create a crown.

## Prerequisites

- Python 3.10 or newer
- Git
- enough local storage for a CPU PyTorch installation

Clone the source and create an isolated environment:

```bash
git clone https://github.com/latent-to/cacheon.git
cd cacheon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev]"
```

## Read the two targets

A bundle replaces one of two registered targets. `forward_pass` covers the served
model: the bundle names the modules it replaces by address, such as
`model.layers.*.mlp`, and each replacement receives the stock arguments and returns
what stock returns. `prefix_cache` replaces the scheduler's prefix cache object. See
the [target catalog](../reference/target-catalog.md).

## Scan without executing code

```bash
python -m cacheon.cli scan examples/miner_node_identity
```

`scan` parses the manifest and applies the recursive static policy. A clean scan is an
admission signal, not a sandbox and not correctness evidence.

Expected output:

```text
bundle: node-identity-control  abi: cacheon-op-abi-v0  ops: 1
  [clean] model.layers.*.mlp <- kernels/forward.py
```

- `bundle` and `abi` came from `manifest.toml`;
- `ops: 1` means one declared implementation row;
- `[clean]` means the declared source passed static policy; it does not mean the callable
  was imported, numerically tested, or graph-captured.

If `scan` reports `VIOLATIONS`, fix the named source or recursive-tree finding first.
Typical causes are forbidden file/process/network APIs, an undeclared native file, a
compiled artifact, or executable Python vendored outside the declared entry.

## Smoke the interface

```bash
python -m cacheon.cli verify examples/miner_node_identity
```

`verify` repeats the scan, then imports each entry in a fresh child process and checks
that it accepts the node's arguments:

```text
  [INTERFACE OK] model.layers.*.mlp variant='default' entry(module, *args, **kwargs)
Scan and import/signature smoke passed. Forward math, preparation and graphs require cacheon check in the arena image.
```

`examples/miner_node_wrong` passes the same smoke although it scales the MLP result by
1.5: numerical truth is the stock module in the running engine, so only
`cacheon check` in the published arena image, and qualification, can fail it. See
[Your first bundle](../miner-guide/your-first-kernel.md).

## Run the test suite

```bash
pytest -q
```

The suite covers much more than the CPU tutorial: manifests, target resolution, stack
assembly, hostile transport, OCI policy, qualification evidence, settlement, emissions,
and compatibility guards. GPU-specific tests skip when their
runtime is unavailable.

## What this did not prove

The quickstart did not exercise:

- SGLang model execution and the node audit against stock;
- CUDA graph capture and replay;
- no-egress OCI candidate execution;
- production qualification (paired replay windows), the registered eager audit,
  and pristine T;
- audited qualification or settlement; or
- integration and serving.

Those boundaries require validator-owned hardware, runtime identities, policies, and
evidence stores. Developer GPU experiments remain non-authoritative regardless of their
local output.

## Continue

| If you want to… | Read… |
|---|---|
| write your first bundle | [Your first kernel](../miner-guide/your-first-kernel.md) |
| prepare a GPU development machine | [GPU setup](../dev/gpu-setup.md) |
| understand production qualification | [Qualification](../validator-guide/qualification.md) |
| operate finalized intake first | [Deployment readiness](../validator-guide/first-hour.md) |
