# Emissions policy

Cacheon separates economic accounting from chain publication. Settlement creates
content-addressed claims. A policy projects those claims into an exact
1,000,000-part weight vector. A separate signer journals, submits, reads back, and
confirms that vector.

Policy bytes are validator consensus configuration. Miners cannot supply or
override them. A policy change requires validator authorization and an explicit
migration; it does not alter the thresholds used to grade retained results.

## Policy generations

Two generations coexist so retained evidence remains reopenable:

| Generation | Claim model | Publication command | Status |
|---|---|---|---|
| Legacy V1 | Decaying standing credit plus bounded discovery claims | `cacheon set-weights` | Retained and operational |
| Finite-debt V2 | Finite registered-CROWN principal plus reviewed discovery bounty | none | Design retained; implementation extracted from the tree on 2026-08-09 |

Only legacy V1 can publish weights. The extracted V2 implementation and its
durable evidence remain reopenable from Git history and the reserved schema.

## Legacy V1

Policy `cacheon.emissions.v1.8` awards new measured performance records a fixed
share of the available emission budget. A complete audited PASS supplies the
evidence; a crown is not required to start the award.
Qualification keeps using the manually commissioned incumbent when another
contribution crowns. Changing that comparison baseline requires an explicit
operator commission; a crown alone does not change it.
Duplicate packaging or recommissioning of the same contribution cannot renew
its award. Within each arena and exact measured incumbent stack, let `s` be the
new PASS speedup and `best` the largest previously accepted speedup, initially
`1`. Only `s > best` creates progress. A second PASS against an old incumbent
does not earn the improvement already established by a stronger PASS.

For acceptance block `a`, current block `n`, and half-life `h`, shares of total
emission are:

```text
q = max(0, ln(s / best) / ln(1.01))
R = max(0, 1 - discovery_reserve - outstanding_standing_shares_at_acceptance)
A = R × (1 - 2^(-q))
share(n) = A × 2^(-(n - a) / h)
```

The 1% unit is anchored to the meaningful-improvement gate. The policy choice is
that one such unit earns half the remaining standing budget. A 1.82% record
earns about 71.5% of that budget, not 1.82% of total emission. The exponential
form follows from requiring compounded improvements accepted at the same time
to leave the same reserve whether submitted together or in stages:
`2^(-(q1 + q2)) = 2^(-q1) × 2^(-q2)`. The 50% anchor is an explicit incentive
choice, not a theorem of economic fairness.

Awards are replayed by acceptance block and the insertion order of their
retained qualification records. A later PASS in the same block cannot move
ahead of an earlier award. The discovery reserve is always the configured
`discovery_pool_ppm / 10^6`; when unused it goes to the validator. Holding that
capacity aside prevents later discovery bounties from diluting standing awards.
Outstanding awards remain reserved even while their recipients are absent or
excluded. A new award cannot spend that temporarily redirected money.

Each award retains its acceptance clock and initial amount when later winners
arrive or the operator changes the incumbent. New awards have no submission-age
or stall multiplier. Without new awards, one half-life halves each standing
share and returns the difference to the validator; after four half-lives it is
one sixteenth of its initial amount. Sum each hotkey's fixed-point credits and
convert to integer ppm by flooring against the fixed `10^12` unit. Never divide
by total miner credit. The validator receives all unallocated, decayed, absent,
excluded, unused-discovery, and rounding remainder.

Two limits matter. A new winner can claim only the uncommitted budget; it cannot
take another miner's outstanding award. Also, splitting progress across time
is not neutral because the reserve replenishes through decay: weak competition
can still encourage waiting. The rule does not claim complete strategy-proofness.
Separate commissioned baselines have separate measurement records; changing a
baseline must remain an authorized operation, never a miner-controlled way to
reset a record.

#### Preserving existing awards during migration

`frontier_awards_from_block` is an immutable acceptance-block boundary in the
policy. PASSes accepted before it retain the v1.7 absolute-credit formula, their
saved speedup and acceptance time. For those claims only, submission block `b`
and the preceding pre-boundary PASS submission block `p` in the arena determine:

```text
credit = floor(ln(s) × (1 + sqrt((b - p) / 1800)) × 2^(-(n - a) / h) × 10^12)
```

Their remaining absolute shares consume reserve before new awards are issued.
They also establish each comparison baseline's previous record. If old
liabilities exhaust capacity, the initial new award is zero; it does not reduce
old payouts or acquire a deferred entitlement. A projection exceeding the total emission budget stops instead of
rescaling claims. The boundary defaults to `0` for a new deployment; set it
explicitly when preserving older awards. The same boundary must be supplied to
`set-weights`, `push-weight-offer`, and the offer producer's configuration.

Existing v1.1/v1.3/v1.4/v1.5/v1.6/v1.7 bindings can advance once to v1.8 with the
same half-life, discovery lifetime and discovery pool. The new boundary becomes
bound with the policy digest and cannot subsequently be changed. This reuses
retained qualification evidence; it neither reruns evaluations nor rewrites
claims. Missing acceptance or comparison-baseline evidence stops a new award.

The active standing claim validates its evaluation stack against that stack's
sealed catalog and target-spec bytes. Historical v1 composition and v2 exclusion
rules retain their active meaning; installing another model's catalog does not
reinterpret or invalidate an earned claim. A composed crown may carry an
incumbent contribution whose exact PASS belongs to another arena; that original
claim keeps its age and earns once, without a duplicate claim in the new arena. Reward
history is derived from existing settlement candidates and their retained
PASS records; there is no parallel accepted-history table. The existing metadata
stores only the count and digest of the history already used for awards. Each
projection must preserve that prefix before appending new PASSes. Removal,
reordering, changed acceptance times, or changed measurement inputs stop the
projection rather than silently repricing old awards. Evidence corrections need
an explicit accounting migration; they never authorize a paid rerun.
Missing or corrupt evidence holds the projection. If a claimant leaves the metagraph, its share
goes to the validator for that tick and returns if the claimant re-registers.

A discovery qualification can create one non-renewable bounded claim. It does not
install an evaluation-stack contribution or create a standing family. Duplicate
packaging, promotion, integration, or release cannot renew that claim.

The projector reopens every earning accepted PASS, active stack, standing claim, and
discovery claim, binds finalized chain scope and membership, aggregates by hotkey,
and includes the validator remainder in one positive integer-ppm vector totaling
1,000,000. Completing the vector does not increase any miner's incentive.

### All-uncrowned bootstrap

Normal V1 projection refuses to publish without a real crown. An operator may
explicitly direct the complete vector to a registered burn hotkey:

```bash
cacheon set-weights \
  --intake-db chain_intake/intake.sqlite3 \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --wallet default \
  --hotkey validator \
  --half-life-blocks <BLOCKS> \
  --discovery-lifetime-blocks <BLOCKS> \
  --discovery-pool-ppm <PPM> \
  --refresh-blocks <BLOCKS> \
  --burn-hotkey <REGISTERED_BURN_HOTKEY> \
  --dry-run
```

The burn path is valid only when all of these are true:

- there is no active standing or discovery claim;
- no evaluation arena has a crowned generation;
- V2 composition has not been activated; and
- the burn hotkey belongs to the exact projection metagraph.

The same command fails closed as soon as real economic authority exists.

### Subnet-owner burn bootstrap

`--burn-to-subnet-owner` is the chain-resolved variant of the same
all-uncrowned bootstrap: instead of naming a burn hotkey it resolves the
subnet owner's registration from the finalized metagraph and publishes the
identical crownless full-pool projection through the durable
intent/pending/confirmed journal, with the same refusals the moment any
active claim, crowned arena, or activated composition exists. Each pass
resolves the burn sink fresh (or stops before signing with `--dry-run`):

```bash
cacheon set-weights \
  --burn-to-subnet-owner \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --intake-db chain_intake/intake.sqlite3 \
  --half-life-blocks <BLOCKS> \
  --discovery-lifetime-blocks <BLOCKS> \
  --discovery-pool-ppm <PPM> \
  --refresh-blocks <BLOCKS> \
  --wallet default \
  --hotkey validator \
  --dry-run
```

Resolution uses the finalized metagraph RuntimeAPI fields `owner_coldkey`,
`owner_hotkey`, and per-UID coldkeys from that same block-bound snapshot (no
unpinned `subnet()` / storage fallback). Candidates are UIDs whose coldkey
equals the subnet owner. Prefer `owner_hotkey` when it is among those
candidates; otherwise choose the lowest matching UID. Fail closed when the
owner coldkey is missing or no owned neuron is registered. Publication runs
through the standard journaled reconciler with `require_current_crown=False`,
so intent, pending, confirmed, and held states are durable in the intake
database, a foreign in-flight journal head is refused, and settlement-state
refusals surface as nonretryable publication faults that stop a `--watch`
loop.

### V1 publication loop

`set-weights` supports one reconciliation or a continuous operator loop:

```bash
cacheon set-weights <POLICY_AND_SIGNER_ARGUMENTS> \
  --watch \
  --interval <SECONDS>
```

Watch mode reruns the complete authority refresh and reconciliation. It uses bounded
retry for retryable transport or chain faults and does not retry a nonretryable
publication fault. It cannot be combined with `--dry-run`, `--reconcile-only`, or
`--release-hold`.

Before signing, the reconciler refreshes finalized authority. A later finalized
head is acceptable only when the validator UID and every weighted recipient UID
remain unchanged. UID reassignment before signing aborts publication; reassignment
after submission prevents confirmation and retains a hold.

The publication journal distinguishes `intent`, `pending`, `confirmed`, `held`, and
`released`. An SDK success response is not confirmation. Confirmation requires an
exact finalized readback of the intended recipient set and values within the fixed
verifier tolerance.

Signer-free modes reopen journal state without submitting:

- `--reconcile-only` grades retained publication state against chain authority.
- `--release-hold "reason"` appends an audited release; it does not approve or
  submit the old vector.

## Finite-debt V2

V2 finite debt is a retained design, not implemented code. Its implementation
— fixed-point finite registered-CROWN principal, a separate bounded
reviewed-discovery bounty class, content-addressed one-campaign composition
policies, wallet-free atomic activation, and gapless confirmed-boundary debt
publication — was extracted from the tree on 2026-08-09 without ever being
activated. The complete implementation, its policy arithmetic, the reviewed
selection-report identities, and the deterministic load study remain in Git
history at
[`dc158fb4`](https://github.com/latent-to/cacheon/commit/dc158fb4).

The design intent retained for a future reintroduction:

- an eligible post-activation crown receives a **finite** bounded claim that
  is paid down over later confirmed epochs — no perpetual royalty;
- a later crown does not erase an unpaid balance;
- reviewed discovery pays a separate bounded bounty class; and
- activation is an explicit, independently approved one-way cutover, never an
  inference from implemented arithmetic.

One compatibility artifact remains in the tree: the shared-weight offer wire
schema keeps its `lane`/`debt_binding` fields with debt-lane payloads
rejected. The reserved schema-4/5/6 migrations and V2 table DDL were retired on
2026-09-05; intake still accepts the metadata stamps 3 through 6, so databases
created earlier open unchanged with their V2 tables untouched, and a fresh
database now stops at stamp 3. Reintroducing V2 is a new reviewed change with
its own design and security review, not a revert switch.

## Operational invariants

- Preserve one writer for a validator/database authority.
- Back up SQLite with WAL-aware tooling and retain every referenced evidence root.
- Treat policy, campaign, reserve, membership, and activation digests as immutable.
- Treat claimant departure and UID reassignment differently. A currently absent
  claimant's family share goes to the validator for that tick under the bound
  projection; unexplained UID reassignment still halts publication. Never rewrite
  a historical boundary from a later metagraph snapshot.
- Never repair a hold by deleting journal rows, editing debt, or replacing evidence
  with a summary.
- Keep evaluator containers separate from wallet and signer authority.

## Current evidence limits

- No live V2 activation or debt-publication receipt ever existed, and the V2
  implementation is no longer in the tree.
- Registered discovery promotion remains unsupported.
- Historical signer-free shadows and synthetic load sweeps established
  accounting behavior only; they authorized no activation or chain mutation.
- Production still requires exact campaign/reserve manifests, retained historical
  membership authority, independently graded review and invalidation authority,
  and accepted production audit-canary evidence.

See [Settlement and weights](../validator-guide/settlement-and-weights.md) for the
operator flow.

## Source anchors

- [Legacy economics](https://github.com/latent-to/cacheon/blob/main/cacheon/economics.py)
- [V1 publication](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/weights.py)
- [CLI](https://github.com/latent-to/cacheon/blob/main/cacheon/cli.py)
