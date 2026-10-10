# What Cacheon is

Cacheon is an open inference-acceleration system built around a pinned SGLang runtime.
It uses a permissionless market to discover GPU optimizations, a hostile-code referee
to measure them, and a separate integration and release process to turn selected work
into a product.

The chain is useful for proposal ordering, attribution, and rewards. It is not a runtime
dependency of Cacheon Engine.

## A useful mental model

Cacheon is a compiler contest whose winning patches must still pass through a product
release process. That separation answers three questions that otherwise get mixed
together:

- **Who arrived first?** Finalized chain ordering and durable intake answer this.
- **What was actually faster and faithful?** Validator-owned qualification evidence
  answers this.
- **What may run in production?** Review, integration, signing, and release verification
  answer this.

## Two systems, three operational surfaces

At the highest level, Cacheon separates the chain-independent product from the
market that improves it. Inside the market system, the subnet control plane and
the hostile-code referee remain distinct operational surfaces.

### Cacheon Engine

Cacheon Engine is the chain-independent serving product. Integration into maintained
source, release signing, and serving are maintainer decisions outside this repository,
which contains no release tooling. Miner URLs, wallets, chain state, and the live
evaluation incumbent are not serving dependencies.

### The subnet and referee

The subnet is a market for proposals. The referee is the measurement system that:

1. preserves finalized proposal priority;
2. fetches and republishes hostile bytes immutably;
3. rejects invalid, unregistered, and copied work without creating economic authority;
4. evaluates each admitted proposal as one exact marginal substitution;
5. retains and reopens the evidence;
6. reopens the complete audited qualification before settlement; and
7. projects rewards for active, verifiable contributions.

The referee may update the untrusted evaluation incumbent. It cannot update a signed
serving release.

## The two objects

A **proposal** is hostile input: a miner asks the validator to evaluate one target-scoped
delta. A **crown** is retained measurement evidence that one complete audited
qualification improved one registered arena and target. Integration into maintained
source, release, and serving are separate authorities outside this repository; the
[product model](../architecture/product-model.md) is the normative statement of each.

No transition is implicit. In particular, a crown is not permission to run miner source
in production.

## A proposal, worked end to end

Suppose a miner profiles a published arena and finds that the MLP blocks are on the
critical path. They build a source-only bundle that names `model.layers.*.mlp` under
the `forward_pass` target and declares that it applies to BF16 calls on `sm90`.

1. Locally, `scan` checks the tree without importing the candidate, and `verify` imports
   each entry in a fresh child process and checks its signature. `check`, in the
   published arena image, binds the node in a real engine and audits every sampled call
   against the stock module. These are development checks, so even a clean GPU result
   does not establish priority or a win.
2. The miner packages the tree. The proposal identity is a SHA-256 over the sorted
   identity-bearing relative paths and bytes, not over gzip timestamps or archive
   compression. They host that archive on public HTTPS and commit the hash and URL with
   a hotkey-signed timelock submission.
3. After finalized reveal, the validator fetches the archive, safely extracts it,
   recomputes the hash, republishes an immutable tree, and resolves the claimed target
   against the active catalog. It observes the actual proposal features rather than
   trusting the manifest to grant itself permissions.
4. Version-3 qualification measures every candidate with the paired replay:
   speed policy 17 launches separate baseline and candidate engines that replay
   the same sealed agent workload concurrently on two isolated lanes, then
   swaps the lanes and boots both engines fresh. A registered eager, untimed audit role
   (**A**) then checks the candidate delta; after candidate teardown, the
   pristine reference (**T**) supplies candidate-free quality evidence.
5. One complete audited PASS becomes `qualified` and eligible for settlement.
   Settlement reopens its retained evidence and may create the target crown;
   it does not schedule another qualification.
6. Independently of the crown, the PASS earns credit for the miner's hotkey if it
   beats the best earlier rewarded PASS against the same arena and incumbent by the
   required margin. The active incentive policy determines how that credit
   contributes to the validator's weight vector, which a separate publisher later
   submits and confirms on-chain. Product integration remains a separate maintainer
   decision. See [How miners earn rewards](../miner-guide/incentives.md).

If the implementation is wrong at step 1, the miner changes the bundle. If its HTTPS
server times out at step 3, the validator may retry the same identity. If the candidate
is slower at step 4, that exact identity fails; a revised kernel is a new proposal. This
is why the last authoritative lifecycle state matters more than the last local command.

## Slots, targets, and engines

These terms describe different layers:

- A **slot** is the address a manifest row replaces: a node address such as
  `model.layers.*.mlp`, or `tree_cache` for the prefix cache.
- A **target** assigns economic identity: `forward_pass` for model nodes,
  `prefix_cache` for the cache.
- A **materialized engine** is the complete content-addressed tree executed for an arm.

The candidate engine can contain the full incumbent stack, but the validator constructs
it. The miner contributes only the selected delta.

The distinction also explains fallback. A narrowly specialized proposal does not replace
the entire engine. For a live call outside its declared capability domain, stock SGLang
serves the call. That is safe, but an implementation which never routes onto material
arena calls cannot contribute measurable speedup.

## One stack and a reference

The referee's `EvaluationStackManifest` may carry crowned but unintegrated hostile
proposals and runs only inside isolation; the pristine `ReferenceManifest` contains no
proposal and never competes for speed, so the fastest current proposal is never its own
correctness oracle. See [Stacks and manifests](../architecture/stacks.md).

## Authority levels

Cacheon exposes contributor diagnostics that do not create crowns:

- `scan` checks static policy without loading a bundle;
- `verify` imports each entry and checks its signature, without running it;
- `check`, in the published arena image, audits the bundle against stock in a real
  engine and checks captured execution; and
- contributor-controlled matched A/B profiling can investigate full-engine behavior and
  throughput without producing validator evidence.

Production authority begins only after finalized intake, immutable publication, a closed
arena service, isolated qualification, retained evidence, and audited qualification.
See [Proposal to release](../architecture/pipeline.md) for the complete sequence.

Use this rule when reading the rest of the documentation: a command is not authoritative
merely because it performs similar math. Authority comes from the identities, isolation,
policies, evidence, and audited qualification bound to the production path.

## Read next

- [See why miners participate and how rewards work](../miner-guide/incentives.md)
- [Run the CPU quickstart](quickstart.md)
- [Understand the product model](../architecture/product-model.md)
- [Choose a registered target](../miner-guide/slots.md)
