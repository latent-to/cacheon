# Bundle manifest

Every contribution bundle has a `manifest.toml`. The manifest is parsed as
data before any contribution module is loaded. Its current ABI identifier is
`cacheon-op-abi-v0`. Branding does not version this hash-bound protocol
identifier, so Cacheon authors continue to emit the published spelling.

## What parsing does—and does not do

A manifest passes through distinct layers:

```mermaid
flowchart LR
    T["TOML + contained paths"] --> M["Manifest object"]
    M --> F["Observed bundle features"]
    F --> C["Target-catalog resolution"]
    C --> S["Static scan"]
    S --> V["Component verification"]
    V --> Q["Arena qualification"]
```

`load_manifest()` covers only the first step: required syntax, identifier
shape, path containment/existence, variant uniqueness, and structural CUDA or
patch declarations. It does not import source, prove that the requested target
exists, or verify numerical behavior. `scan`, target resolution, `verify`, and
production qualification are separate gates so a structurally valid document
cannot grant itself authority.

## Minimal singleton example

```toml
bundle_id = "example-silu-v1"
abi_version = "cacheon-op-abi-v0"

[competition]
target = "activation.silu_and_mul"
mode = "slot"

[[ops]]
slot = "activation.silu_and_mul"
source = "kernel.py"
entry = "silu_and_mul"
dtypes = ["float32"]
architectures = ["cpu"]
```

`bundle_id` and identifiers accept letters, numbers, `.`, `_`, and `-`.
Paths are relative to the bundle and must resolve to regular contained files.

## Top-level fields

| Field | Required | Meaning |
|---|---:|---|
| `bundle_id` | yes | Human-readable bundle identifier; not the content identity |
| `abi_version` | yes | New bundles must emit `cacheon-op-abi-v0`; the reader also accepts the exact hash-bound pre-cutover spelling described below |
| `[competition]` | recommended | Explicit requested target and `slot`/`atomic` mode |
| `[[ops]]` | yes | One or more implementation rows |
| `[[dep_patches]]` | no | Declared text patches for a validator-approved dependency lane |

New bundles must use `cacheon-op-abi-v0`. Validators also retain one narrowly
scoped reader-only compatibility spelling, `optima-op-abi-v0`, for finalized
bundles committed before the rename. Readers preserve the submitted spelling
and bytes; they do not rewrite a publication because the spelling participates
in the committed bundle identity. Every other ABI spelling is refused.

The canonical bundle hash, not `bundle_id`, is the proposal's content identity.

### Identity versus display name

The content hash walks the bundle's own regular files in sorted relative-path
order and hashes length-prefixed path and byte sequences. Git and Python cache
noise is excluded; symlinks are not part of the hashed file set and the bundle
loader/scanner rejects unsafe tree structure. Editing source, metadata, a
patch, or even the manifest produces a new identity. Renaming only the outer
directory does not.

`bundle_id` is useful in logs and diagnostics but never proves that two reveals
contain the same artifact. Commit/reveal, immutable publication, copy checks,
and qualification use digest-bound content.

### Competition table

| Field | Values |
|---|---|
| `target` | A validator-registered target ID |
| `mode` | `slot` or `atomic`; `system` parses only for legacy migration |

The table is a request, not policy. Intake resolves it against the frozen
[target catalog](target-catalog.md) and complete observed feature set. The
current resolver can infer a target when an exact singleton or registered
atomic member set is unambiguous, which preserves older bundles. New
competitive bundles should declare `[competition]` explicitly.

## Operation rows

| Field | Required | Meaning |
|---|---:|---|
| `slot` | yes | Registered execution slot |
| `source` | yes | Python source module within the bundle |
| `entry` | yes | Entry callable name |
| `variant` | conditional | Capability variant; required on every row when a slot repeats |
| `prepare` | no | Weight-preparation callable for a registered prepare/forward ABI |
| `setup` | development only | Legacy parser/direct-framework diagnostic hook; every registered component target forbids it |
| `dtypes` | no | Declared dtype capability domain; an empty array adds no dtype restriction |
| `architectures` | no | Declared architecture capability domain; an empty array adds no architecture restriction |
| `metadata` | no | Eligibility/capability JSON within the bundle |
| `base_kernel` | no | Validator-owned base-kernel identifier for an override submission |
| `override_point` | no | Typed hole in that base; requires `base_kernel` |
| `cuda_sources` | no | Inspectable `.cu`/`.cuh` inputs for a reviewed builder |

Unknown operation fields are retained for observation. They do not grant a
capability; complete feature resolution can reject a bundle that asks for more
than its target admits.

### How an operation row is selected

The `slot` chooses a validator-owned semantic ABI, not an arbitrary import
hook. The validator resolves `source` inside the bundle and looks up the named
Python identifier only after structural and static gates. It allocates outputs
and passes arguments in the registered slot order. Candidate code fills those
outputs; it does not redefine shapes, references, tolerances, or the call site.

For prepare/forward slots, `prepare` names the registered one-time weight
transformation while `entry` names the runtime call. `setup` is a legacy direct
framework hook and every current component target forbids it, so new
competitive manifests should not use it.

### Variants

Several rows may implement the same semantic slot only when every row has a
unique explicit `variant`. Their validator-parsed capability domains must not
overlap. Runtime selection is fail-closed: exactly one applicable variant is
required, otherwise the trusted baseline is used or qualification refuses the
candidate according to the registered boundary.

Empty or omitted `dtypes` and `architectures` do not mean “supports nothing”;
they add no restriction at the manifest-parser layer. Eligibility metadata and
validator observation still constrain the real capability domain. Conversely,
listing `sm103` does not prove the source builds or runs there.

```toml
[[ops]]
slot = "norm.rmsnorm"
variant = "sm90-bf16"
source = "rms_sm90.py"
entry = "rmsnorm"
dtypes = ["bfloat16"]
architectures = ["sm90"]

[[ops]]
slot = "norm.rmsnorm"
variant = "sm103-bf16"
source = "rms_sm103.py"
entry = "rmsnorm"
dtypes = ["bfloat16"]
architectures = ["sm103"]
```

This example demonstrates variant syntax only. `norm.rmsnorm` is unavailable
for paid submission in the current MiniMax-M3 mainnet arena because that model
does not execute the registered RMSNorm callsite. See
[Current MiniMax-M3 availability](../miner-guide/slots.md#current-minimax-m3-availability).

## Dependency patches

```toml
[[dep_patches]]
target = "flashinfer"
path = "patches/change.patch"
```

Only UTF-8 `.patch`/`.diff` files are structurally accepted. Binary, rename,
and deletion patches are refused. Admission still requires a target that
permits the observed dependency-patch feature and a validator-reviewed applier
with a bounded destination policy.

CUDA source declarations behave similarly. Listing `.cu`/`.cuh` paths makes
them inspectable inputs to the sanctioned build lane; it is not permission to
ship a prebuilt binary or execute an arbitrary compiler command. Rebuild
operations come from reviewed validator policy.

## What the manifest cannot do

A bundle cannot choose:

- its reward family, overlap, or displacement;
- arena hardware, workload, thresholds, or hidden tasks;
- the incumbent, reference, or release stack;
- isolation or network policy;
- qualification, reproduction, or settlement outcomes.

## Failure diagnosis

| Error class | Typical cause | Fix the right layer |
|---|---|---|
| TOML/required-field error | Missing `[[ops]]`, wrong ABI string, invalid identifier | Correct manifest syntax |
| Path error | Absolute path, `..` escape, missing file, unsafe symlink | Keep every declared input as a regular contained file |
| Duplicate-slot error | Multiple rows without explicit unique variants | Name every variant and make domains disjoint |
| Competition error | Unknown target, wrong `slot`/`atomic` mode, legacy `system` title | Choose a registered target from validator output |
| Feature-admission error | CUDA, patch, override, setup, or extra capability outside target policy | Remove the feature or choose the registered lane |
| CUDA-source declaration error | Missing, uncontained, or non-`.cu`/`.cuh` input | Declare contained source paths in `cuda_sources`; build admission remains validator-owned |
| Static scan error | Forbidden import/operation or uninspectable tree content | Rewrite the source; scanning is not a sandbox exception list |
| Verification error | Callable/signature/output/reference mismatch | Debug the registered tensor and correctness contract |
| Qualification failure | Complete engine misses timing, drift, quality, fidelity, or resource gates | Inspect retained arena evidence; do not relabel the outcome |

## Pre-submission checklist

- Run `slots` and select the economic target separately from its execution
  slot members.
- Declare `[competition]` explicitly for new singleton or atomic work.
- Keep `bundle_id` descriptive but assume only the content hash is identity.
- Declare every source, metadata, CUDA, and dependency-patch input with a
  contained relative path.
- Give repeated slot rows unique variants with non-overlapping domains.
- Avoid `setup`; use only registered `prepare` and entry contracts.
- Run `scan`, then the appropriate local or collective `verify` command.
- Hash and package the exact verified tree; any later byte change is a new
  proposal.

Source: [`cacheon/manifest.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/manifest.py).
