# Emissions policy

Optima projects rewards from independently verified marginal improvements in the
active evaluation stack. Settlement and chain publication use a content-addressed
policy manifest; policy parameters are validator consensus configuration and are not
inferred from submissions or historical ledger state.

## Authority status

There are currently three deliberately separate policy authorities:

- **Legacy V1 authority:** the standing-claim policy described later in this
  document remains reopenable for retained testnet projections and publication
  journals. Its bytes and digest are not reinterpreted.
- **Selected finite-debt V2:** “V2” names the emissions-generation migration from
  standing legacy claims; the serialized registered-CROWN manifest itself correctly
  says `policy_version="optima.finite-debt.v2"` and schema 2. D-015's pure
  campaign-sized arithmetic and additive durable validation are implemented but
  inactive. D-012 and its signer-free chain shadow are immutable historical evidence
  for the superseded family-share manifest; they do not authorize D-015 bytes.
- **Selected reviewed-discovery composition:** D-013 adds a separately versioned
  `optima.incentive-composition.v1` manifest, pure two-class arithmetic, schema-5
  review-pending/bounty-only durable state, and signer-free
  `chain-incentive-composition-shadow`. Those bytes
  are implemented but inactive. A live signer-free testnet shadow passed over
  explicitly synthetic states with `submitted=false`; its exact bounded receipt is
  recorded below and carries no review, settlement, publication, or activation authority.

Legacy V1 remains the sole wired publisher. Activation still lacks the exact
MiniMax-M3 campaign identity, production family map, and reserve identity plus a
fresh campaign-policy shadow; an
atomic or quiescent V1→core→composition cutover; retained-boundary publication,
confirmation, catch-up, and debt debit; independently graded review and
runtime-invalidation authority; an atomic successor protocol for later campaign
rotation or one-to-two expansion; retained membership-departure history rather than
only a current snapshot; durable promotion transport/linkage; and production audit
evidence transport. The current `review_digest` is controller-supplied and
content-bound, not independently reopened and graded. See
[FIDELITY.md](FIDELITY.md) for the audit gap.

Policy migration creates no retroactive V2 debt for legacy crowns. An activation
record has its own policy digest and exact finalized block/hash; it never rewrites
the legacy `emissions_policy_digest` needed to reopen old projections.

The miner-facing explanation is [INCENTIVES.md](INCENTIVES.md).

## Selected finite-debt policy

After activation, only an independently reproduced registered-lane `CROWN` can
issue **registered-CROWN principal**. The lower reproduced relative speedup is expressed in
multiplicative 1%-log units:

```text
G = floor(1_000_000 * ln(speedup) / ln(1.01))
```

With no retained or seeded prior-family clock, the first crown has multiplier
`1_000_000`. Otherwise the accepted finalized-crown block gap `D` is used:

```text
M = 1_000_000 + floor(100_000 * D / (D + 648_000))
```

Every accepted crown resets its family clock; no other disposition does. The
bonus therefore reaches 5% at 648,000 blocks (about 90 days) and is capped below
10%.

One policy epoch is 7,200 blocks and contains 1,000,000 accounting units. After
the 10% minimum reserve, the reference claim capacity is 900,000 units. Every
validator-owned family maps once to model campaign `c`. For campaign claim-sizing
share `B_c` in ppm and `k=1_000_000`, principal uses this
exact flooring order:

```text
F_c = floor(900_000 * B_c / 1_000_000)
Q   = floor(F_c * k * G * M / 1_000_000^3)
```

The manifest enforces 1,000,000-ppm claim sizing for a sole campaign or
500,000-ppm claim sizing for claims in either of two campaigns, and rejects more
than two. Every family maps exactly once and every
campaign has a family. Target count never divides a campaign share: adding 1, 10,
or 100 unused families changes principal by zero, while each family retains its
own elapsed-time clock. Campaign shares size claims rather than creating hard
epoch silos; all registered claims later share global `P_c` pro rata.

D-015's 14 preregistered screens all passed. At `k=1`, the weekly 4.4%/5% normal
envelope paid 100%, expired zero, drained to zero outstanding, and had five-day
maximum latency under both zero and saturated discovery; worst utilization was
77.4136%. Five-day 4.4% cadence was marginal and four-day cadence overloaded.
The sensitivity sweep made `k=1.25` marginal. At `k=1.5` the worst rows crossed
into overload while other rows remained marginal; `k=2` was plainly overloaded,
retaining `k=1`. Report semantic digest:
`7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.

Registered-CROWN claims start aging at settlement. Discovery bounty age instead
starts at the retained qualified-win block, so a later review consumes rather than
refreshes its 648,000-block window and review at or after expiry cannot mint. Both
classes conserve `principal = paid + forfeited + remaining`. Under the selected
composition, one confirmed epoch allocates the two classes in this order:

```text
P_d     = min(50_000, live_discovery_debt)
P_c     = min(900_000 - P_d, live_registered_CROWN_debt)
reserve = 1_000_000 - P_d - P_c
```

Discovery claims split `P_d` and registered-CROWN claims split `P_c`, each in its
own claim-digest largest-remainder class and pro rata by remaining principal. Each
class exhausts its integer payout before hotkey aggregation, so it has no unassigned
rounding residue. The explicit reserve receives its 100,000-unit floor plus unused
capacity and receives the complete epoch when there is no payable debt.
Expired/cancelled principal is forfeited accounting debt, not a one-time transfer
to the reserve. Projection alone never debits a claim.

The accounting denomination is **confirmed weight-ppm epochs**, not a guaranteed
amount of TAO or alpha. A token-denominated promise would require an additional
finalized chain receipt and conversion policy that does not exist today.

### Selected reviewed-discovery composition

D-013 deliberately does not auto-price discovery-only work in log units. Its pure
policy intends one reviewed discovery win to take exactly one economic path:

1. `registered_promotion`: issue no discovery debt, register the boundary, then
   require fresh qualification/reproduction and a CROWN before registered debt can
   be issued; or
2. `bounty_only`: issue one non-renewable finite discovery claim and no registered
   title.

The selected bounty has a 50,000-ppm epoch cap, a per-award principal cap of one
discovery-pool epoch (50,000 weight-ppm epoch units), and a 648,000-block lifetime
anchored to the retained qualified win. It has no campaign share, family clock, time
bonus, renewal, or permanent title.

That two-branch statement is policy intent, not the current durable execution
surface. Schema 5 atomically retains qualified discovery candidates/evidence as
`review_pending` and can issue a unique bounded `bounty_only` claim. It deliberately
rejects `registered_promotion`; the database disposition constraint is bounty-only.
Promotion remains fail-closed until existing typed `DiscoveryWinRecord` and
`DiscoveryPromotion` authority are transported and reopened, the target is
registered, fresh qualification/reproduction/CROWN is linked, and the same work has
one identity across discovery and registered lanes. Current uniqueness therefore
prevents duplicate durable bounties for retained identities, but does not yet enforce
“never both” for the same underlying work across both lanes.

Review delay consumes bounty life. A review at or after
`win_block + discovery_lifetime_blocks` cannot mint. The landed finalized
`expire_review_pending_discovery_wins` API terminalizes overdue pending wins as
`review_expired` and appends `discovery_review_expired`; production orchestration
must still run it reliably.

The selected cell is
`8561028c943738da2fe622e5f5c9fd43ebec16fdd59feab3561de25fbfa450d9`;
the selection-report digest is
`6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc`.
The deterministic sweep was 9 cells × 36 scenarios × 10 seeds = 3,240 rows and
replayed byte-identically locally and on the RTX pod. The selected cell paid all
273,000,000 units of non-departed principal, with zero expiry or outstanding debt
and 100% worst-run payout; 9,000,000 units of departed debt were
forfeited/cancelled. Analytic and measured saturated registered-CROWN capacity
dilution was 55,555 ppm (5.5555%), and saturated tapes eventually paid 100% of
CROWN principal. This is synthetic accounting evidence, not miner-equilibrium,
token-value, or GPU-performance evidence.

D-014 subsequently held this policy fixed for a 288-row review-delay sensitivity
matrix, byte-identically replayed on arm64/Python 3.11 and x86_64/Python 3.12.
All 108 preregistered 0/1/7-day SLA rows passed: 100% discovery principal paid,
zero expiry/unissued principal, maximum instantaneous CROWN-capacity dilution of
55,555 ppm, and zero CROWN paid-fraction regression versus zero delay. The
90/120-day cases issued no stale debt; 30/60/89 days were diagnostic only. Report
digest: `f0939d67241dffa49aac95c035c43dd7ea14b51eb2671fe106cb09347511b7ef`.
This is synthetic accounting sensitivity evidence only; it supplies no external
review, publication, activation, durable-state-completion, or GPU-performance
authority.

Before D-015, the signer-free composed shadow passed on testnet netuid 307 at
finalized block 7,586,146 with metagraph size 6. Explicitly synthetic states
projected 850,000 ppm registered-CROWN payout, 50,000 ppm reviewed-discovery
payout, and 100,000 ppm reserve, conserving 1,000,000 ppm. The receipt records
`submitted=false`; semantic digest
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`,
file SHA-256
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`.
It constructed no wallet and did not supply or exercise review, settlement,
publication, debt-debit, D-015 policy, or activation authority. Its core policy is
the historical D-012 family-share generation; a campaign-policy shadow is still
required before activation.

## Migration and activation invariants

- V1 standing claims continue to identify the active evaluation-stack title;
  V2 finite claims represent historical payment debt. They are not the same row,
  and Legacy V1 remains the only publication path until an explicit cutover.
- Family clocks and claim issuance commit atomically with the corresponding
  registered-family CROWN. Failed, held, discovery-only, copied, retirement, or
  neutralization events create no CROWN principal and advance no family clock.
  A reviewed `bounty_only` disposition is the sole separate authority that can
  issue discovery principal.
- Activation may explicitly seed a family clock from a retained pre-activation
  accepted crown. A seed is digest-bound activation authority, creates no claim or
  retroactive principal, and means the first post-activation claim need not use
  the no-prior-clock multiplier.
- Reveal/finalized order controls priority and elapsed-time input. Registered CROWN
  age starts at settlement; discovery bounty age starts when the qualified win is
  retained, not when review later mints the claim.
- A reserve anchor is a policy-bound hotkey, not a UID. Missing reserve authority
  fails closed, and the reserve cannot own a miner claim.
- Policy upgrades with open debt are forbidden in the first implementation.
  The current composition generation also permits only one activation, so model
  rotation or one-to-two campaign expansion requires an atomic successor protocol.
- Schema-4→5 migration creates only empty immutable composition tables: it imports
  no historical CROWNs or legacy discovery awards and creates no retroactive debt.
  Composition activation additionally requires clean open-debt state and fails if
  any legacy discovery claim is active.
- Once composition is active, legacy settlement cannot auto-award a discovery
  bounty. Active settlement can retain a review-pending discovery win and the
  durable disposition ledger can issue only `bounty_only`; registered promotion is
  intentionally rejected pending its typed cross-lane authority.
- The finalized registered-family invalidation API can cancel that family's open
  debt and insert a clock marker so its next CROWN behaves like a first CROWN. It
  consumes an external `invalidation_digest`; it does not independently decide or
  grade runtime invalidity, so that authority remains an activation blocker.
- A projection build, dry run, restart, SDK success, or retry never reduces debt.
  Only an exact confirmed completed-epoch record may do so once. The durable API
  currently binds an externally supplied publication-record digest but does not
  reopen and grade a V2 confirmation journal. Epochs are gapless from activation,
  so a missed or slowly confirmed retained boundary currently wedges later closure;
  production needs retained-boundary catch-up rather than skipping it.
- V1 and V2 must never publish concurrently. Core activation and composition
  activation are two distinct finalized steps; before the second step, legacy
  settlement/publication can still race. Activation requires an atomic handoff or
  operator-enforced quiescence and proof that failure cannot leave both (or neither)
  publishing silently.
- Lifecycle reconciliation currently consumes an eligible-hotkey snapshot. Production
  must retain boundary-specific membership/departure history so a later snapshot is
  not misread as the authoritative historical departure event.
- The MiniMax-M3 campaign identity, its real production family map, and reserve
  identity must be frozen into exact manifests and a campaign-policy shadow run
  against those bytes before activation.

## Legacy V1 standing reward families

Each active registered target defines one reward family. A singleton target defines
one family for its slot. An atomic target defines one family for the complete target
and suppresses every overlapping singleton family while it is active. Packaging,
integration, release, and engine-stack records do not create additional reward
families.

A crown contributes credit only while its exact contribution is active in the current
evaluation stack and its retained qualification authority can be reopened. Credit is
derived from the measured marginal speedup above unity, expressed as integer parts per
million. It is never derived from an argmax or from the number of slots in a target.

For a crown with `speedup_ppm > 1_000_000`, age `a` blocks, and policy half-life `h`,
the standing credit is:

```text
improvement = speedup_ppm - 1_000_000
credit      = floor(improvement * h / (h + a))
```

This reciprocal decay is deterministic under integer arithmetic and reaches half of
the original credit at one half-life. A retired or neutralized crown contributes no
credit. A stale, incompatible, missing, or unverifiable crown holds the complete
projection; its share is not redistributed silently.

## Legacy V1 discovery bounties

A discovery qualification does not create a standing family or modify the evaluation
stack. It may create one non-renewable bounty claim, identified by the discovery
proposal and retained evidence. The policy manifest bounds both the bounty lifetime
in blocks and the aggregate discovery share in parts per million. Duplicate packaging,
promotion, integration, or release of the same discovery cannot create another claim.

## Legacy V1 projection and publication

Standing credit is summed by miner hotkey after every active family has reopened.
When live discovery claims exist, their aggregate is normalized inside the bounded
discovery pool and standing credit receives the remainder. Otherwise standing credit
receives the complete vector. The final vector is normalized only after chain scope,
validator identity, current stack generations, retained evidence, and current
metagraph membership have been bound into one projection.

Only the control-plane signer may publish a projection. It refreshes chain authority
immediately before reconciliation and again after submission. Publication is journaled
as `intent`, `pending`, `held`, `released`, or `confirmed`; an SDK return value alone
never confirms weights. A held publication requires an explicit audited release event
before another intent. A real extrinsic requires at least one genuine current-schema
crown. Dry runs may exercise projection and reconciliation but cannot create publication
intent.
