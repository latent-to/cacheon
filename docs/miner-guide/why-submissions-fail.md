# Why submissions fail

Most failed submissions have one of a few causes: a copy of earlier work, a
kernel that does not compile, a CUDA-graph contract failure, an audit mismatch
against stock, or a candidate that is simply not faster than the incumbent. Only
the last is an engineering loss. Copies and compile failures are free to avoid
before submitting.

## Copies are detected by containment, not by hash

A copy is not caught by comparing file hashes. The validator fingerprints
bundles per definition and by normalized whole file, then tests containment
against every prior submission, the repository's example bundles and test
fixtures, **and** `cacheon_kernels/`, the validator's own public reference
library. Renaming, reformatting, or reordering does not defeat it.

Two consequences worth stating plainly:

- Submitting a bundle derived from another miner's proposal fails as
  `copy_of:<predecessor>`, and the predecessor keeps the credit.
- Submitting code taken from `cacheon_kernels/` fails as
  `copy_of:validator_reference:library-<file>`. That library is the validator's
  public reference code, not an original contribution. The baseline you are
  measured against is the arena's commissioned incumbent stack.

## Compile your kernel before you submit

A kernel that cannot be traced or compiled never executes, so it cannot be
scored; the engine stops with the candidate's original error. This is detectable
by invoking the declared entry in a matching local Triton/CUDA environment before
submission.

One Triton error is by far the most common:

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
invokes. Inspect the finding, then actually exercise the declared entry in the arena image:

```bash
python -m cacheon.cli scan path/to/your_bundle
```

```bash
python -m cacheon.cli check path/to/your_bundle --model /model \
  --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

Only the sandboxed production build/execution path can issue an attributable
compile `FAIL`. The local checks are there to prevent paying for an obvious
failure, not to recreate validator authority.

## Losing on speed is a real result

A correct, compiled, graph-safe bundle that is not faster than the incumbent
fails on speed. That is the subnet working as intended. The baseline is a tuned
production stack; see [Finding an improvement](finding-a-win.md) before choosing
a target, and [Choose a target](slots.md) for what is registered.

The speed grade pools elapsed serving cost over paired windows in two lane
orientations and requires a statistical lower bound on the gain above one. There
is no fixed minimum margin; the bar is the sealed noise of the arena. A speed
FAIL states which of two things was measured. `speed_threshold_not_met` means the
gain did not clear that bound by the last sealed window, and the candidate was not
measurably slower either — an ordinary competitive miss, not a regression.
`candidate_slower` means the estimate fell below the mirrored bound. A first lane
orientation that reads slower than the arena's sealed futility margin fails
without the swapped orientation, and its detail names that margin. A candidate
that clears the speed bound but misses service attainment fails with
`service_contract_not_met`. Verdicts settled under older policies may carry the
combined code `speed_regression`.

## CUDA graphs are part of the contract

A kernel that is correct in eager mode but cannot be captured and replayed is
not shippable here. See [Graph correctness](graph-safety.md).

A [node-address](../architecture/slot-contract.md#node-addresses) bundle is proven
on the timed run itself. Every claimed module must have run inside a captured
graph there; one that only ever ran eagerly fails with `never invoked inside a
CUDA-graph capture`. `prepare` and `entry` must also leave the live module's
methods alone: `a method inside node ... was rebound after binding (Class: attr)`
names the attribute the bundle changed.

## The audit compares your kernel with stock

After the speed stage, an untimed eager run samples live calls to each claimed
node and compares each output with stock SGLang on the same inputs, at the
registered tolerance.

- `slot_audit_failed` — compared calls show a different function: a call far
  outside tolerance, or more than one compared call in 100 just outside it.
  Rare near misses from low-precision rounding pass. Compare against stock at
  the live shapes and dtypes before resubmitting.
- `audit_not_covered` — the audit compared too few calls to grade the kernel.
  This is not a judgement on the bundle: the disposition is `NO_DECISION` and
  the validator re-evaluates it.

The exact rule is in [Audit outcomes](../validator-guide/fidelity.md#audit-outcomes).

A prefix-cache bundle is judged by content instead. On both arms, the validator
checks that every page a served prefix points to holds the bytes the engine
computed for that prefix. Serving other bytes, keeping pages across a flush,
moving a request's own slots or claiming more tokens than the key stops the
engine as the candidate's failure. See
[the prefix cache](../architecture/slot-contract.md#the-prefix-cache).

## Identical bytes are not re-evaluated

If the validator has already failed exactly your publication bytes under exactly
the same commissioned arena service, that `FAIL` is replayed instead of re-running
the evaluation. Resubmitting an unchanged failed bundle therefore changes nothing;
a new commissioned baseline measures it fresh. A `PASS` is never replayed, and
resubmitting a passed bundle earns nothing new: reward skips a duplicate contribution.

A retained report reopens under its original authority. A complete audited
`PASS` is sufficient for qualification. Reopening its retained bytes verifies
that authority; it does not create a new measurement or a duplicate reward.

## A closed family is parked, not failed

The commissioned arena workload may be unable to measure some registered
target families. A submission for one is parked at intake with reason
`target_unavailable:<target>` before any evaluation runs. This is not a
judgement on the bundle and it is not charged — the cited eval-cost payment
stays spendable. The submission closes as `NO_DECISION` and is not re-queued:
resubmit the same bytes when the family reopens. The operator's announcement
names the open targets; see [GLM availability](slots.md#current-glm-53-availability).

## Things that are not valid submissions

- **Patches to SGLang.** The engine is pinned and consensus-critical. Submit
  kernels or a prefix cache, not engine changes.
- **Engine-wide setup.** A bundle that installs process-wide setup is not a
  registered target and is rejected at resolution.

## Before you submit — checklist

1. `python -m cacheon.cli scan` and `verify` both pass locally.
2. The kernel compiles on your machine, in the mode you expect it to run.
3. It is your own work, and not derived from `cacheon_kernels/`.
4. `check` in the arena image passes its audit and, for node bundles, shows
   captured execution on every claimed address and rank.
5. You have a reason to believe it is faster than the baseline, not merely
   different.
6. You have paid the submission cost — see [Submit on-chain](submitting.md).
