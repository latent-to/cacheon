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

Every registered target still defines one current evaluation family: the active
standing claim is the incumbent used by settlement and promotion. Reward credit is
no longer taken only from that incumbent. Every accepted `CROWN` is appended to an
immutable history and keeps earning a decaying share.

For a retained accepted crown with `speedup_ppm > 1_000_000`, predecessor block
`p` (the previous accepted crown in that arena, or this crown's own
`crowned_block` if it is first), age `a`, stall scale `s`, and half-life `h`:

```text
units      = ln(speedup_ppm / 1_000_000)
elapsed    = crowned_block - p
M          = 1 + sqrt(elapsed / s)
credit     = floor(units × M × 2^(-a / h) × 10^12)
```

`crowned_block` and `p` are finalized proposal submission blocks, so validator
evaluation delay cannot inflate the stall bounty. The first accepted crown in an
arena uses `M = 1`. Later accepted crowns in the same arena reset that arena's
clock for the next contribution; discovery bounties do not. Compounded
multiplicative gains are path-independent because the units are logarithmic.

Accepted history is not retired when a later crown replaces the evaluation
incumbent. All positive decayed scores are pooled globally, summed by hotkey,
and normalized exactly to 1,000,000 ppm. There is no 80% champion floor. This
rule ships as `policy_version` `cacheon.emissions.v1.3` (the claim model remains
the legacy V1 generation). The policy digest includes
`time_multiplier_scale_blocks` plus optional `excluded_hotkeys` and
`excluded_claim_digests`; the bound `emissions_policy_digest` in intake
metadata must be rotated deliberately when deploying. Old publication journals
remain reopenable but refuse mixed old/new policy authority.

Retirement or neutralization removes only the current-stack standing claim. A
stale, incompatible, missing, or unreopenable accepted or standing crown holds
the complete projection; its share is not silently redistributed. An active
claimant absent from the bound metagraph does not hold the vector: that
family's allocated standing or live-discovery share is published to the
validator hotkey for this tick. Other families keep the ppm they would have
received if the claimant were still registered. If the hotkey returns, the next
projection pays it again at the then-current decayed credit.

Operator exclusions are different from that burn. `--exclude-hotkey` and
`--exclude-claim-digest` (and the matching sealed weights-stage lists) omit the
named miner or accepted-history row from credit, then renormalize the remaining
positive standing and live-discovery credit as if the excluded rows were never
in the pool. The evaluation incumbent and retained evidence still reopen; a
drop from weights does not unseat the kernel. An excluded claim digest that is
not among retained accepted crowns, or an exclusion that leaves no positive
standing credit, fails closed. Changing exclusions rotates `policy.digest` and
requires the same deliberate `emissions_policy_digest` metadata rotation as any
other policy change.

A discovery qualification can create one non-renewable bounded claim. It does not
install an evaluation-stack contribution or create a standing family. Duplicate
packaging, promotion, integration, or release cannot renew that claim.

The V1 projector reopens every active stack and claim, binds exact finalized chain
scope and metagraph membership, aggregates credit by hotkey, and normalizes one
positive integer-ppm vector totaling 1,000,000.

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
  --time-multiplier-scale-blocks 1800 \
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
  --time-multiplier-scale-blocks 1800 \
  --refresh-blocks <BLOCKS> \
  --wallet default \
  --hotkey validator \
  --dry-run
```

```bash
cacheon set-weights \
  --burn-to-subnet-owner \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --intake-db chain_intake/intake.sqlite3 \
  --half-life-blocks <BLOCKS> \
  --discovery-lifetime-blocks <BLOCKS> \
  --discovery-pool-ppm <PPM> \
  --time-multiplier-scale-blocks 1800 \
  --refresh-blocks <BLOCKS> \
  --wallet default \
  --hotkey validator \
  --watch \
  --interval <SECONDS>
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

Two compatibility artifacts remain in the tree:
[`chain/reserved_schema.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/reserved_schema.py)
preserves the schema-4/5/6 migrations and V2 table DDL verbatim so existing
intake databases keep validating, and the shared-weight offer wire schema
keeps its `lane`/`debt_binding` fields with debt-lane payloads rejected.
Reintroducing V2 is a new reviewed change with its own design and security
review, not a revert switch.

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

See [Current status](state-of-record.md) for the maintained evidence ledger and
[Settlement and weights](../validator-guide/settlement-and-weights.md) for the
operator flow.

## Source anchors

- [Legacy economics](https://github.com/latent-to/cacheon/blob/main/cacheon/economics.py)
- [Reserved V2 schema](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/reserved_schema.py)
- [V1 publication](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/weights.py)
- [CLI](https://github.com/latent-to/cacheon/blob/main/cacheon/cli.py)
