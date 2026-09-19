# Graph evidence

CUDA-graph behavior is part of the serving contract, not a performance toggle
the candidate may opt out of. A kernel that works eagerly but fails capture or
replay cannot replace an incumbent graph path.

## Capture and replay in plain language

Normal eager execution lets Python and the runtime decide what to launch on every call.
A CUDA graph records one launch sequence against fixed storage addresses, then replays
that sequence with new tensor contents. Serving uses this to remove repeated CPU launch
overhead and stabilize scheduling.

For a candidate, the important phases are:

1. **Warmup:** imports, legal lazy initialization, and approved compilation complete
   before evidence-bearing capture.
2. **Eager check:** the callable produces the correct result outside a graph. A failure
   here is ordinary ABI/math failure, not a graph problem.
3. **Capture:** operations issued by the callable must be legal on the capture stream.
   Host synchronization or data-dependent Python decisions break this phase.
4. **Replay:** the validator changes logical inputs, poisons outputs, replays without
   rerunning candidate Python, and compares the new result. A kernel which captured a
   stale pointer or first-call value fails here.

Graph safety is not synonymous with “uses CUDA.” It means the implementation continues
to compute the slot contract when Python launch logic is frozen and only device work is
replayed.

## There is nothing to declare

A bundle does not state whether it is graph-safe, and there is no metadata key
for it. Whether a slot serves from inside the captured region is a property of
that slot's live seam, owned by the validator. Every slot whose seam is captured
routes the candidate under capture exactly as it routes it eagerly, and a kernel
that cannot be captured fails there loudly instead of being quietly skipped while
the captured graph keeps serving stock.

## How qualification proves it

Qualification has no separate graph test. The proof is taken from the timed run
and the two checks that already surround it:

- **Captured completion.** The dispatcher, not the candidate, records whether
  each invocation happened inside a CUDA-graph capture. On a graphs-on run every
  claimed slot on every rank must have completed inside one. A candidate that
  only ever ran eagerly, for example because its declared domain excludes the
  captured shapes, would have been timed as stock; it fails with
  `never invoked inside a CUDA-graph capture`.
- **Audit.** The eager, untimed audit role checks each slot's output against the
  stock computation on live calls.
- **Quality.** The pristine reference scores the text the timed run produced. A
  kernel that is captured but replays its capture-time answer passes the first
  check and fails this one
  ([measured](../results/qwen-h100-node-slots.md#stale-under-capture)).

In short: every selected implementation path must be evidenced. A fallback that
silently makes the candidate N/A cannot create a crown. The execution check is in
[engine_worker.py](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/engine_worker.py).

## Local graph diagnostics

On CUDA, `verify` graph-tests every slot whose registered live seam is captured:

```bash
python -m cacheon.cli verify my_bundle \
  --device cuda --dtype bfloat16
```

For a collective:

```bash
python -m cacheon.cli verify my_collective \
  --device cuda --dtype bfloat16 \
  --world-size 4 --tp-size 4
```

The collective verifier creates the actual rank group, captures each applicable
clean-room shape, poisons outputs between replays, and grades every rank. Run it
on homogeneous GPUs matching the arena architecture.

A CPU `verify` can prove eager numerical behavior, but it cannot produce CUDA
capture evidence. Likewise, a local CUDA pass is a developer diagnostic—not
the evidence production qualification retains from its own timed run.

When reading local output:

- `graph=not-required` says this run did not request graph proof; it does not certify an
  eager-only implementation for an arena that requires graphs;
- `graph=verified` plus positive `graph_replays` says the local profiles passed capture
  and checked replay;
- `NUMERICAL_PASS ... graph=NOT_VERIFIED` means the math passed but the requested graph
  contract did not; and
- N/A profiles add no graph evidence because the candidate was not applicable.

Production takes its proof from the served shapes of its own timed run, so even a local
`graph=verified` is preparation rather than a qualification receipt.

## Common capture failures

Graph-safe serving code must avoid:

- `.item()`, device-to-host reads, or synchronizing `.cpu()` calls;
- Python branches whose path depends on live tensor values;
- dynamic allocation or compilation during capture/replay;
- retaining a pointer to one output and writing it on later calls;
- reading stale tensor values captured from a previous invocation;
- writing only part of a poisoned output;
- changing collective order across ranks;
- creating or destroying process groups in the entry callable;
- using a different stream or event protocol without capture-safe ownership.

Prepare layouts, compile approved artifacts, and allocate persistent workspace
before serving capture. During replay, consume the current dynamic inputs and
write the current validator-provided outputs.

One subtle failure pattern is a kernel that appears stable because the shape stays fixed
while values change. CUDA graphs intentionally support that regime: storage addresses
remain fixed, but sequence lengths, routing IDs, scores, and tensor data can differ on
each replay. Treat any such value as live device input. Reading it into Python during
capture freezes one branch and is semantically wrong even if the first replay happens to
match.

## Variable shapes and variants

CUDA graphs fix storage addresses, not semantic values. The verifier replays
fresh logical input values through the captured path. A kernel that bakes in
first-replay data can appear correct at capture and fail replay.

If algorithms genuinely differ by shape, use explicit disjoint variants and
capability domains. Do not branch on a host read of a runtime tensor. Every
claimed slot must complete inside a capture on every rank, whichever variant
served it.

An eager-only seam, when one is registered by an arena, is a validator-owned
descriptor fact rather than an exemption a bundle can claim. No slot is
registered eager today, so every claimed slot must complete inside a capture.

## Do not benchmark a different regime

Disabling CUDA graphs in a contributor-controlled launch can help debug startup or math,
but it changes the incumbent execution regime and therefore cannot establish a production
speedup. A graph-on candidate must be compared with the graph-on incumbent under the same
evaluation stack.

When graph verification fails, use the stage-specific guidance in
[Diagnostics](diagnostics.md). Narrowing metadata until the candidate never
executes is not a workaround: a candidate that does not run cannot be crowned.
