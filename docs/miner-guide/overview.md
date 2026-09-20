# Miner guide

Cacheon accepts source proposals that make a validator-owned inference engine
faster without changing its required behavior. A miner contributes one
registered delta, the validator evaluates that delta inside the exact incumbent
stack, and only fully qualified wins can become a crown.

This is not a contest for self-reported microbenchmarks. The validator owns the
model, runtime, prompts, reference behavior, build, launch policy, measurements,
and final decision.

## Why participate

A miner has an opportunity to earn a share of validator weight by becoming the
independently verified performance frontier for one target in one published
evaluation arena. Submission itself earns nothing. If one complete validator
qualification proves the improvement and settlement crowns it, settlement records the
reward claim in the same transaction. The validator later combines eligible claims
into a weight vector and publishes it on-chain.

```text
proposal -> one complete audited PASS -> settlement crown -> reward claim -> confirmed weights
```

[Read how miner rewards work in plain English →](incentives.md)

## The job in one sentence

Find one validator-registered boundary that matters to the published workload,
implement exactly that boundary, and show that substituting only your delta into the
current incumbent produces a reproducible end-to-end improvement without weakening
behavior.

That sentence contains the whole discipline:

- **registered boundary** rules out inventing an economic target in the manifest;
- **matters to the workload** rules out optimizing a fast but irrelevant microkernel;
- **exactly that boundary** preserves a causal marginal comparison;
- **current incumbent** rules out benchmarking against a convenient stock baseline;
- **end-to-end** accounts for dispatch, synchronization, graph mode, and downstream work;
- **reproducible** requires retained counts, intervals, and correctness evidence that reopen under the sealed policy; and
- **without weakening behavior** keeps pristine quality evidence outside candidate
  control.

## Choose your path before writing code

For a node arena, name one supported module or several disjoint modules in the
served model. The bundle replaces each module's `forward`, accepts its stock
arguments and returns its result structure. Use an enclosing module when the
optimization crosses internal calls. Scheduler policy and unrelated serving
configuration remain outside the contribution.

## The three identities to keep separate

The **arena** identifies the commissioned model/runtime and workload. A **node
address** identifies an execution boundary such as `model.layers.*.mlp`. The
registered **target** admits the bundle's declared node addresses and ties them
to evaluation and reward policy.

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "<published-arena-id>"
```

The manifest requests an existing target; it does not grant new authority. See
[Slots and targets](slots.md) and [Kernel ABI](kernel-abi.md). Retained catalog
examples follow their old `SlotSpec` contracts; their signatures are not node
interfaces.

For node bundles, `verify` is scanning plus import/signature smoke. Use
[`check` in the published image](your-first-kernel.md#6-move-to-the-matching-gpu-environment)
for the live model binder, audit and captured execution.

## What happens to a submission

Cacheon keeps two objects distinct, and keeps both away from serving:

1. A **proposal** is the source archive you publish and commit on-chain. It is
   untrusted input, not an engine dependency.
2. A **crown** is a fully qualified marginal win for one registered
   target in one evaluation stack. It can receive standing reward under the
   active emissions policy.

Nothing after a crown is automatic. Integrating crowned source into maintained code and
any release are maintainer decisions outside this repository; a crown is not permission to
ship miner code.

That separation is part of the product contract, not release ceremony. Read
the full [product model](../architecture/product-model.md)
before working on an advanced target.

## How qualification works

For a registered target, the validator constructs an exact marginal comparison:

- **B**: the opening read from the exact incumbent on the baseline lane;
- **C**: the read from the one-target-transition candidate on the disjoint
  candidate lane;
- **B′**: a mandatory second incumbent read, the quality gate's stock-drift
  control;
- **A**: a registered eager, untimed audit role for the candidate delta; and
- **T**: a candidate-free pristine reference used after candidate teardown.

The candidate does not choose the rest of the stack. The validator materializes
the exact incumbent and candidate engines on the two-process substrate and
serializes timed work.
Bookending detects drift, A supplies the
registered sampled slot regrade, and T prevents “fast because behavior changed”
from becoming a win. Static, build, ABI, graph, and abbreviated-serving checks
are admission screens only; they cannot crown a proposal.

A promoted proposal must pass two complete, independent qualification attempts.
The crown records the accepted qualification speedup. After the complete audited attempt,
the durable intake state is `qualified`; settlement and confirmed weights remain separate.

The evaluation design and evidence objects live in
[qualification.py](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification.py),
[qualification_runner.py](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py),
and [arena_service.py](https://github.com/latent-to/cacheon/blob/main/cacheon/arena_service.py).

## How rewards work

Reward details depend on the policy announced by the operator; a local speedup is
never enough to estimate income. Read [How miners earn rewards](incentives.md).

## Your development loop

The local CLI provides static and component diagnostics:

```bash
python -m cacheon.cli scan my_bundle
python -m cacheon.cli verify my_bundle
```

These commands find manifest, static-policy, ABI, correctness, routing, and graph
problems. Profile and bracket serving performance in an environment matching the
published arena contract. Neither the CLI checks nor a contributor-controlled A/B run
reproduces crown authority: local work does not possess the finalized intake record,
validator stack manifest, hidden inputs, calibrated policies, immutable publications,
isolated service, or second independent qualification attempt.

Use the technical guide in this order:

1. [Slots and targets](slots.md)
2. [Bundle format](bundle-format.md)
3. [Kernel ABI](kernel-abi.md)
4. [Your first kernel](your-first-kernel.md)
5. [Finding a win](finding-a-win.md)
6. [Submitting](submitting.md) — copy-paste chain-submit sequence, including eval-cost
7. [Diagnostics](diagnostics.md)

Read [Override points](override-points.md) only when the registered target requires one.

At the end of the sequence you should be able to answer, with concrete identities:

1. Which arena, stack generation, target, model, architecture, topology, phase, and dtype
   does the proposal address?
2. Which exact files and capability domain form the selected delta?
3. Which verifier shapes and graph replays exercised every applicable variant?
4. Why can this slot-level mechanism move end-to-end critical-path time?
5. Which local result is diagnostic, and which operator receipt is the last
   authoritative state?

If any answer is still “whatever the validator chooses,” obtain the published arena
contract before submitting. A content-addressed proposal cannot be repaired in place
after reveal.
