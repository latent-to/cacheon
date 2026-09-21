# Emissions policy

Cacheon separates economic accounting from chain publication. Settlement creates
content-addressed claims. A policy projects those claims into an exact
1,000,000-part weight vector. A separate signer journals, submits, reads back, and
confirms that vector.

Policy bytes are validator consensus configuration. They are not supplied by a
miner, inferred from a bundle, or changed by an operator after observing a result.

## Policy generations

Two generations coexist so retained evidence remains reopenable:

| Generation | Claim model | Publication command | Status |
|---|---|---|---|
| Legacy V1 | Decaying standing credit plus bounded discovery claims | `cacheon set-weights` | Retained and operational |
| Finite-debt V2 | Finite registered-CROWN principal plus reviewed discovery bounty | none | Design retained; implementation extracted from the tree on 2026-08-09 |

Only legacy V1 can publish weights. The extracted V2 implementation and its
durable evidence remain reopenable from Git history and the reserved schema.

## Legacy V1

Every distinct registered contribution with a complete audited PASS
earns from its retained qualification, whether or not it becomes the crown.
Qualification keeps using the manually commissioned incumbent when another
contribution crowns. Changing that comparison baseline requires an explicit
operator commission; a crown alone does not change it.
Duplicate packaging of the same contribution earns once.

For speedup `s > 1`, finalized submission block `b`, the preceding distinct PASS
block `p` in the same arena (or `b` for the first), current block `n`, decay start
`d`, half-life `h`, and fixed stall scale `1,800` blocks:

```text
credit = floor(ln(s) × (1 + sqrt((b - p) / 1800)) × 2^(-(n - d) / h) × 10^12)
```

This is the retained full-strength rule. Static allocation can assign a reduced
waiting bonus to future submissions as described below; changing code alone
does not rescore earlier claims.

Submission time controls stall credit. New PASS claims hold their decay factor
at one until a matching weight vector containing their evidence and a positive
share for their hotkey is confirmed. The first qualifying confirmation fixes
`d`; retries and restarts cannot move it. An older identical vector or an
inclusion receipt without sufficiently recent active-weight readback does not
start the clock. Existing deployments can retain their original `d = b` clocks
through the recorded legacy-claim set.

Publication starts and recovery adjustments are append-only intake metadata,
bound into the projection's evidence and policy identity. They never rewrite
submission blocks, retained PASS records, or claim identities.
Logarithmic units make compounded gains path-independent. Policy version
`cacheon.emissions.v1.5` projects every accepted qualification and replaces v1.4's
CROWN-only restriction. Existing v1.1/v1.3/v1.4 bindings move forward only
when all numeric policy fields match.

The active standing claim validates its evaluation stack against that stack's
sealed catalog and target-spec bytes. Historical v1 composition and v2 exclusion
rules retain their active meaning; installing another model's catalog does not
reinterpret or invalidate an earned claim. Reward
history is derived from existing settlement candidates and their retained
PASS records; there is no parallel accepted-history table. Missing or corrupt
evidence holds the projection. If a claimant leaves the metagraph, its share
goes to the validator for that tick and returns if the claimant re-registers.

A validator rebuild may change a carried contribution's selected payload. Its
original PASS remains payable when the reopened source artifact, attribution,
target and target contract still match. A rebuild creates no second claim;
changing any of those identities requires its own earned qualification.

A discovery qualification can create one non-renewable bounded claim. It does not
install an evaluation-stack contribution or create a standing family. Duplicate
packaging, promotion, integration, or release cannot renew that claim.

The projector reopens every earning accepted PASS, active stack, standing claim, and
discovery claim, binds finalized chain scope and membership, aggregates by hotkey,
and normalizes one positive integer-ppm vector totaling 1,000,000.

### Static arena percentages

The existing offer producer can combine any configured number of arena control
planes into one V1 vector. Operators set percentages manually; there is no
auction, demand adjustment, or automatic change to those settings. A named
source groups the historical evaluation generations in one intake database.

Settings use integer parts per million (ppm): `600000` means 60%. Normalize
the complete configured set **only if its sum exceeds 1000000**. Thus
40/40/40 becomes approximately 33⅓/33⅓/33⅓, while 20/30 stays 20/30.
Largest remainders, with source names breaking ties, resolve integer rounding.
Each submission receives the normalized terms effective at its finalized
arrival block. Qualification time does not change those terms. Appending a
new settings version cannot reprice older submissions.

Each settings row can also specify `stall_bonus_ppm`: an integer from zero to
1000000 controlling only the additional waiting bonus. The quarter-strength
rule uses `250000` and gives each affected submission this credit:

```text
credit = floor(ln(s) × (1 + 0.25 × sqrt((b - p) / 1800)) × 2^(-(n - d) / h) × 10^12)
```

The base multiplier remains one; this does not divide the entire reward by four.
A gap of 1800 blocks changes the multiplier from 2 to 1.25; a gap of 7200 blocks
changes it from 3 to 1.5. The logarithmic speedup factor, decay half-life and
publication-triggered decay start remain unchanged.

Bonus strength is frozen at finalized submission block `b`, just like the arena
percentage. A submission before the new boundary retains its full bonus even
if it qualifies afterward. Appending a future rule alone does not change current
payouts. New winners can still dilute older payouts. Rows without
`stall_bonus_ppm` mean full strength (`1000000`), preserving existing configuration
bytes and digests; they do not inherit a preceding row's bonus. Include `250000`
explicitly in every later settings row that should continue the reduced rule.
Preactivation rows must retain full strength. Bonus strength is not normalized
with arena percentages.

The time-without-improvement (stall) bonus is internal to each source. For its
pool, average the submissions' frozen percentages using their speed/decay credit
with the stall multiplier set to one. Split that pool among its submissions in
proportion to frozen percentage times full credit, including the stall bonus.
Changing only the stall bonus can redistribute that source's pool but cannot
increase its share relative to another source. When every submission in a source
has the same terms, its pool is exactly that percentage regardless of total credit.
Zero-percentage submissions neither earn nor dilute positive-percentage submissions.
Missing positive base credit and unassigned capacity go to the configured burn
hotkey. A PASS can earn before its source has a crown; retained qualification
evidence remains mandatory.

Frozen terms are percentages used in accounting, not guaranteed token amounts.
Settings alone cannot change existing rewards. New earning submissions can dilute
older payouts across sources while preserving their recorded terms. When historical
pools together exceed 100%, the existing shared weight-vector allocation divides
the available emission among them; it does not rewrite the accepted settings.
For example, old primary terms of 100% plus new secondary terms of 40% produce
approximately 71.43% / 28.57% payouts. With both sources on uniform 60% / 40%
terms, they receive 60% / 40% even if one source has more credit or a larger stall
bonus. Underallocated requests retain their remainder as burn. Dynamic pricing
based on an arena's time without improvement is not enabled.

The producer combines recipients across sources before integer rounding and
retains an allocation report containing source snapshots, submission terms,
actual source shares, burn, and final weights. The existing `WeightProjection`
adds `allocation_evidence` and `rewarded_evidence_digests` only for this path;
legacy projection bytes remain unchanged. Confirmed publication starts decay
only for rewarded claims, including when one hotkey also owns a zero-offer claim.
Discovery bounties without source allocation are refused by this path.

When a retained submission uses a reduced bonus, the allocation report also
records `submission_stall_bonus_ppm` for every earning claim. Its immutable
allocation history binds those strengths into the offer's policy identity.
Unmodified full-strength configurations keep the previous report encoding.

See [Static allocation configuration](../validator-guide/chain-loop.md#static-allocation-configuration)
for activation and updates through the single existing producer.

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
