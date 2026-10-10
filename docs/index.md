<div class="cacheon-hero" markdown>
<div class="cacheon-eyebrow">Bounded proposals · isolated evaluation · reviewed release boundary</div>

# From untrusted GPU proposals to measured contributions.

Cacheon measures untrusted GPU optimization proposals as attributable contributions
to **Cacheon Engine**. SGLang owns the serving control plane; a contribution replaces
modules of the served model or the scheduler's prefix cache. Integration and release
are maintainer decisions outside this repository.

<div class="cacheon-actions" markdown>
[Build a kernel](miner-guide/overview.md){ .md-button .md-button--primary }
[How miners earn rewards](miner-guide/incentives.md){ .md-button }
[Operate the referee](validator-guide/overview.md){ .md-button }
[Understand the architecture](architecture/product-model.md){ .md-button }
</div>
</div>

## Why miners participate

Cacheon rewards fully qualified performance improvements, not uploads or
self-reported benchmarks. A proposal earns credit when one complete audited
qualification passes and beats the best earlier rewarded PASS against the same arena
and incumbent by the required margin; a crown is not required. The validator later
combines eligible credit into a weight vector and publishes it on-chain; realized token
emission still depends on the wider Bittensor network.

[See the reward lifecycle in plain English →](miner-guide/incentives.md)

At the highest level, Cacheon has two systems: the chain-independent product and
the market that improves it. Operationally, the market side separates the
subnet control plane from the hostile-code referee, giving readers three
cooperating surfaces with different trust boundaries.

<div class="cacheon-grid" markdown>
<a class="cacheon-card" href="engine/model-provision/">
<strong>Cacheon Engine</strong>
<span>The chain-independent product: how the model bytes an engine serves are sealed and provisioned. A crown never ships by itself.</span>
</a>
<a class="cacheon-card" href="architecture/pipeline/">
<strong>The referee</strong>
<span>Isolated, evidence-producing evaluation of one marginal target delta against a validator-owned incumbent stack.</span>
</a>
<a class="cacheon-card" href="validator-guide/chain-loop/">
<strong>The subnet</strong>
<span>Finalized proposal ordering, attribution, economic settlement, and weight publication. It is not part of the serving product.</span>
</a>
</div>

## One idea, four objects

A miner submits a **proposal** for one registered target delta; the referee may
establish a **crown** after one complete audited passing qualification. Integration into
maintained source and release are separate authorities outside this repository, so a
crown is never permission to run miner source in production. Work that does not fit a
registered target is not a valid proposal.

[Learn the product model →](architecture/product-model.md)

## Why the architecture is composable

Every candidate runs as a complete isolated engine but is rewarded only for the smallest
validator-controlled delta it contributes: the incumbent evaluation stack and the same
stack with one registered target replaced replay the same sealed agent workload
concurrently on two isolated lanes, swap lanes and boot fresh, and a candidate-free
pristine reference grades the sealed trajectories afterwards. The execution unit is a
disposable engine; the economic unit is one registered target, so a new optimization
builds on previous wins without copying them.

[Follow a proposal through the system →](architecture/pipeline.md)

## Choose your path

| Goal | Start here |
|---|---|
| Write a Triton, CuTeDSL, or Python kernel, or a prefix cache | [Miner guide](miner-guide/overview.md) |
| Validate the repository locally without a GPU | [Local quickstart](get-started/quickstart.md) |
| Deploy intake, an arena provider, and qualification workers | [Validator guide](validator-guide/overview.md) |
| Audit trust boundaries and failure behavior | [Security model](security/threat-model.md) |

## Evidence is scoped, not blended

Every performance or authority claim is scoped to the exact runtime, hardware, arena,
stack, identities, and procedure that produced it. Diagnostic measurements cannot
authorize a crown, and crown evidence cannot authorize reviewed source or serving.

!!! note "Source of executable truth"
    This [Cacheon repository](https://github.com/latent-to/cacheon) owns both
    executable contracts and this documentation. Content-addressed production
    evidence and immutable publications live in separate operator-owned stores.
    When prose and code disagree, the code, schemas, and tests in the same
    revision take precedence.
