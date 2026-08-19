# Why submissions fail

This page is the measured record, not advice. The table below is the terminal
disposition of every failed reservation on subnet 14 as of 2026-08-18.

| Cause | Count | Share |
|---|---|---|
| Copy of another miner's bundle | 107 | 36.3% |
| Kernel does not compile | 83 | 28.1% |
| Slower than the baseline | 81 | 27.5% |
| CUDA-graph contract | 17 | 5.8% |
| Rejected at screen | 3 | 1.0% |
| Copy of validator library code | 3 | 1.0% |

Only the third row is an engineering loss. Roughly two thirds of all failures
are copies or code that never compiled — neither reaches a GPU, and both are
free to avoid before submitting.

## Copies are detected by containment, not by hash

A copy is not caught by comparing file hashes. The validator fingerprints
bundles per definition and by normalized whole file, then tests containment
against every prior submission **and** against `cacheon_kernels/`, the
validator's own public reference library. Renaming, reformatting, or reordering
does not defeat it.

Two consequences worth stating plainly:

- Submitting a bundle derived from another miner's proposal fails as
  `copy_of:<predecessor>`, and the predecessor keeps the credit.
- Submitting code taken from `cacheon_kernels/` fails as
  `copy_of:validator_reference:library-<file>`. That library is the baseline you
  are being measured against; resubmitting it cannot beat it.

## Compile your kernel before you submit

83 bundles — the second largest group — failed with
`candidate_kernel_does_not_compile`. They never executed, so they could not be
scored. Every one of these was detectable by invoking the declared entry in a
matching local Triton/CUDA environment before submission.

One error accounted for the great majority of them, across 81 distinct hotkeys:

```text
triton.compiler.errors.CompilationError
  AttributeError("'constexpr' object has no attribute 'bit_length'")
```

The cause is calling Triton's **host-side** helper from inside a `@triton.jit`
function on a `tl.constexpr` parameter:

```python
@triton.jit
def _kernel(..., D: tl.constexpr, ...):
    col_offsets = tl.arange(0, triton.next_power_of_2(D))   # fails at trace time
```

`triton.next_power_of_2` is ordinary Python and reaches `(n - 1).bit_length()`,
which needs a real `int`. At trace time `D` is a `constexpr` wrapper object, so
tracing aborts and the kernel is never built. Assigning it first
(`POW2: tl.constexpr = triton.next_power_of_2(D)`) fails identically.

There is no device-side replacement to swap in: `triton.language` does not
export `next_power_of_2`. The fix is a small refactor — compute the bound on the
host at the launch site and pass it in as its own `tl.constexpr` argument:

```python
BLOCK_D = triton.next_power_of_2(D)                    # host, at the launch site
_kernel[grid](..., D=D, BLOCK_D=BLOCK_D, ...)

@triton.jit
def _kernel(..., D: tl.constexpr, BLOCK_D: tl.constexpr, ...):
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D
```

This is the idiom in Triton's official fused-softmax tutorial. `tl.arange`
requires a compile-time power-of-two bound in any case, so the bound has to be
computed where real Python integers exist.

Run the bundle checks before submitting. `scan` reports this known source shape,
but its compilability check is advisory because it cannot tell whether a flagged
kernel is reachable; it may return exit 2 for dead code that production never
invokes. Inspect the finding, then actually exercise the declared entry on CUDA:

```bash
python -m cacheon.cli scan path/to/your_bundle
```

```bash
python -m cacheon.cli verify path/to/your_bundle --device cuda --dtype bfloat16 \
  --model <registered-model-key>
```

Only the sandboxed production build/execution path can issue an attributable
compile `FAIL`. The local checks are there to prevent paying for an obvious
failure, not to recreate validator authority.

## Losing on speed is a real result

81 bundles were correct, compiled, graph-safe, and simply not faster than the
baseline they were measured against. That is the subnet working as intended.
The baseline is a tuned production stack; see
[Finding an improvement](finding-a-win.md) before choosing a target, and
[Choose a target](slots.md) for what is registered.

## CUDA graphs are part of the contract

17 failures were graph-contract failures, split between
`graph_member_not_applicable` (the declared applicability did not hold for the
members actually captured) and `graph_eager_failed`. A kernel that is correct in
eager mode but cannot be captured and replayed is not shippable here. See
[Graph correctness](graph-safety.md).

## Identical bytes are not re-evaluated

If the validator has already produced a terminal verdict for exactly your
publication bytes, under exactly the same arena, that verdict is replayed
instead of re-running the evaluation. Resubmitting an unchanged bundle
therefore changes nothing. Change one byte and you get a new content identity
and a fresh evaluation.

A `FAIL` replays. A `PASS` does not: a first `PASS` is `reproduction_pending`,
and settlement requires an independently bound `PASS` pair, so replaying one
would manufacture the second half of that pair from the first.

## Things that are not valid submissions

- **Patches to SGLang.** The engine is pinned and consensus-critical. Submit
  kernels, not engine changes. See [Dependency patches](dep-patches.md).
- **Engine-wide setup.** A bundle that installs process-wide setup belongs in
  the fenced [discovery lane](discovery-lane.md), not a registered target.

## Before you submit — checklist

1. `python -m cacheon.cli scan` and `verify` both pass locally.
2. The kernel compiles on your machine, in the mode you expect it to run.
3. It is your own work, and not derived from `cacheon_kernels/`.
4. It captures and replays under CUDA graphs.
5. You have a reason to believe it is faster than the baseline, not merely
   different.
6. You have paid the submission cost — see [Submit on-chain](submitting.md).
