# Optima incentives

> **Activation status:** this is the selected but inactive V2 composition. D-012's
> registered-CROWN finite-debt arithmetic, durable state, and signer-free shadow are
> implemented; that class's live testnet shadow passed with `submitted=false`.
> D-013's separately reviewed discovery-bounty arithmetic, schema-5
> review-pending/bounty-only durable state, and signer-free composed-shadow command
> are also implemented. A live testnet
> shadow passed over explicitly synthetic states with `submitted=false`; exact
> receipt facts are retained below. It supplied no wallet, review, settlement,
> publication, or activation authority. Legacy V1 remains the sole wired publisher.
> Meaningful V2 emissions still require the actual production family catalog and
> reserve identity followed by an exact rerun/manifest; an atomic or quiescent
> two-step V1→core→composition cutover; retained-boundary publication confirmation,
> catch-up, and debt debit; independently graded review and runtime-invalidation
> authority; membership-departure history rather than only a current snapshot;
> durable discovery-promotion transport/linkage; and production audit-evidence
> transport.

For registered-family rewards, this curve pays for an independently reproduced
improvement over the current validator-controlled frontier. A submission earns
nothing merely for being fast, novel, or the newest upload. It must pass
correctness and fidelity, beat the exact incumbent under the noise-aware scorer,
pass a second independent reproduction, and settle as a `CROWN` for one canonical
reward family. The separately reviewed discovery boundary is described below.

## The short version

Once this rule is activated, a settling crown creates a finite claim whose size is
determined by:

1. the multiplicative throughput improvement over the prior frontier;
2. a small bonus for how long that reward family had gone without an accepted
   crown; and
3. that family's explicit share of the global claim budget.

The claim is paid down from later confirmed weight epochs. It cannot receive more
than its issued principal and expires after 90 days. A tiny lone claim is therefore
paid only what it earned; it is never normalized into the entire emission vector.

The selected pure policy intends a separately reviewed discovery win to take exactly
one economic path: promotion into a registered target followed by fresh
requalification/CROWN, or one bounded finite bounty. Durable schema-5 currently
implements review-pending retention plus the `bounty_only` branch and deliberately
rejects `registered_promotion`. Until typed promotion transport and cross-lane work
identity exist, “never both” is policy intent, not end-to-end same-work enforcement.

## Exact launch curve

The conservative settled speedup `s` is the slower of the two independently
passing measurements. Production represents its improvement in parts per
million of a 1%-log unit:

```text
G_ppm = floor(1,000,000 * ln(s) / ln(1.01))
```

`G_ppm = 1,000,000` is one multiplicative 1% improvement. Log units make
compounding path-independent: two successive 1% improvements create the same
base credit as one 2.01% improvement, apart from the documented fixed-point
flooring and any elapsed-time bonus.

For every crown after the first one in its family, let `D` be the number of blocks
since the previous accepted crown in that same family. The time multiplier also
uses fixed-point integer arithmetic:

```text
M_ppm = 1,000,000 + floor(100,000 * D / (D + 648,000))
```

With no retained prior-family clock, the first crown has `M_ppm = 1,000,000`;
chain age alone does not create a windfall. An activation may explicitly seed a
family clock from a retained pre-activation accepted crown, so the first
post-activation claim can legitimately have a bonus; every seed is part of the
activation authority and creates no retroactive principal. Every accepted crown
then resets the family clock. Failed,
rejected, held, discovery-only, copied, or arbitrary submissions do not.

The curve is deliberately mild:

| Time since prior crown | Multiplier |
|---:|---:|
| 0 days | 1.000x |
| 30 days | 1.025x |
| 90 days | 1.050x |
| 180 days | 1.0667x |
| 365 days | 1.0802x |
| Infinite limit | 1.100x |

## Claim size

One confirmed payout epoch is 7,200 blocks (approximately one day) and contains
`E = 1,000,000` weight-parts. The minimum reserve is 10%, so the reference miner
claim pool is `C = 900,000` parts per epoch.

If reward family `f` has normalized policy share `B_f_ppm`, the exact issuance
order is:

```text
F_f = floor(C * B_f_ppm / 1,000,000)
Q   = floor(F_f * k_ppm * G_ppm * M_ppm / 1,000,000^3)
```

Here `k_ppm = 1,000,000`: one family-budget share of a claim-pool epoch per
1%-log unit. Family shares are explicit, content-addressed policy data and sum
to 100%. The selected launch rule assigns
shares as equally as integer ppm permits to whichever eligible catalog is approved
at activation: each family gets the quotient, then any one-ppm remainder goes to
the lexicographically earliest family digests. The catalog has not yet been frozen.
We will not invent unequal "commercial value" weights without separately retained
demand and governance evidence.

The frozen selection sweep used four exactly equal synthetic families; the
digest-ordered remainder rule is the production-safe generalization for a real
catalog whose size may not divide 1,000,000.

For intuition, with four equal families:

- a first 1% crown issues `900,000 * 25% = 225,000` weight-part epochs;
- the same 1% crown after a 90-day family drought issues `236,250`;
- a first 5% crown has about `4.90` log units and issues about `1.10 million`,
  equal to about `1.23` claim-pool epochs of capacity; because payout epochs are
  discrete, an otherwise uncontended claim finishes in the second epoch.

With ten equal families, the first example is `90,000`, not `225,000`. After
activation, miners must read the exact active policy manifest rather than assume a
family count. There is no active finite-debt manifest today.

An improvement smaller than 1% can still earn proportionally if it clears the
validator's measured confidence bar and reproduces. The 1% figure is an accounting
unit, not a minimum accepted speedup.

## How claims are paid

At each confirmed composed epoch, let `D_live` be open discovery principal and
`C_live` be open registered-CROWN principal, both measured in weight-ppm epoch
units. The two classes receive independent capacities:

```text
P_d     = min(50,000, D_live)
P_c     = min(900,000 - P_d, C_live)
reserve = 1,000,000 - P_d - P_c
```

- discovery claims share `P_d` pro rata by remaining principal, using
  claim-digest largest-remainder ordering;
- registered-CROWN claims independently share `P_c` by the same rule;
- claims are allocated before their hotkeys are aggregated, so one hotkey owning
  several claims cannot change a class's rounding order;
- unused discovery quota returns to registered-CROWN capacity; only capacity
  left after both classes' actual payouts goes to the explicit reserve.
  Discovery can consume at most 50,000 ppm, so saturated discovery reduces the
  otherwise 900,000-ppm CROWN capacity to 850,000 ppm;
- the activation design requires a claim to be debited only after the exact
  projected vector is confirmed for the completed epoch; rebuilding or dry-running
  a projection does not pay it. That publication-confirmation bridge is not wired,
  and the gapless epoch rule currently means a missed or slowly confirmed retained
  boundary cannot simply be skipped to a later one;
- registered-CROWN principal expires 648,000 blocks after its settlement. A discovery
  bounty's same-length window starts at the retained qualified-win block, not review:
  delayed review consumes the window and review at or after expiry cannot mint;
- deregistration
  forfeits the remaining balance; the resulting unused capacity in later epochs
  flows to the reserve. A finalized durable API can cancel one registered family's
  open debt and reset its next-CROWN clock for runtime invalidation, but the authority
  that decides and signs that invalidation remains external to the API and is not
  independently graded or wired.

The active kernel title and the payment claim are separate. Being superseded
does not erase already earned finite principal, but it also does not create a
perpetual royalty. Expiry places a hard bound on historical liability.

## Discovery-lane boundary

The D-012 curve covers registered singleton or atomic reward-family CROWNs. A
cross-cutting discovery prototype does not automatically mint log-relative
principal or reset a family clock. D-013 selected the separate reviewed rule:

- discovery payout capacity is capped at 50,000 ppm per epoch;
- one award can issue at most one discovery-pool epoch of principal, exactly
  50,000 weight-ppm epoch units under the selected policy;
- the 648,000-block lifetime (about 90 days) starts at the retained qualified-win
  block, so delayed review consumes the available payout window and review at or
  after expiry cannot mint;
- it has no family share, family clock, elapsed-time bonus, renewal, or permanent
  title; and
- the pure disposition type expresses `registered_promotion` versus `bounty_only`
  as mutually exclusive policy choices.

Pure-policy promotion issues no discovery debt and intends a route to a registered
target followed by fresh qualification, reproduction, and CROWN. The durable store
does **not** execute that route today. It atomically retains a qualified discovery as
`ReviewPendingDiscoveryWin`, can later issue one unique bounded `bounty_only` claim,
and rejects `registered_promotion` until existing typed `DiscoveryWinRecord` and
`DiscoveryPromotion` authority are transported/reopened, the target is registered,
fresh requalification/CROWN is linked, and the same work has one identity across
discovery and registered lanes. Consequently the bounty ledger prevents duplicate
bounties for its retained identities, but it cannot yet prove that repackaged work
did not later earn through the registered lane.

The durable `expire_review_pending_discovery_wins` path terminalizes an unreviewed
win at its deadline as `review_expired` and appends `discovery_review_expired`.
Production still needs to schedule that finalized expiry reconciliation reliably.

The selected D-013 cell is
`8561028c943738da2fe622e5f5c9fd43ebec16fdd59feab3561de25fbfa450d9`;
the report digest is
`6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc`.
The deterministic matrix contained 9 cells × 36 scenarios × 10 seeds = 3,240
rows and replayed byte-identically locally and on the RTX pod. In the selected
cell, non-departed principal paid 273,000,000/273,000,000 units, no such principal
expired or remained outstanding, and the worst run still paid 100%. Departed debt
forfeited/cancelled 9,000,000 units. Analytic and measured saturated CROWN-capacity
dilution was 55,555 ppm (5.5555%), while saturated tapes eventually paid 100% of
CROWN principal. These are synthetic accounting results, not evidence about miner
equilibrium, token value, or GPU performance.

### Review-delay sensitivity

D-014 held the selected policy fixed and varied only review delay and review-service
mode. Its deterministic matrix contained 8 delays × 3 modes × 4 scenarios × 3 seeds
= 288 rows and replayed byte-identically on arm64/Python 3.11 and
x86_64/Python 3.12. The preregistered review-SLA screen covered delays of 0, 1, and
7 days across every mode, scenario, and seed: all 108 rows passed. Within that
screen discovery paid 100%, expiry/unissued principal was zero, maximum
instantaneous CROWN-capacity dilution was 55,555 ppm, and CROWN paid-fraction
regression versus the zero-delay case was zero percentage points.

The 90- and 120-day cases issued no stale discovery debt, as required by the
win-anchored 90-day lifetime. Delays of 30, 60, and 89 days were diagnostic only;
they did not widen the preregistered review SLA. The report digest is
`f0939d67241dffa49aac95c035c43dd7ea14b51eb2671fe106cb09347511b7ef`.
This establishes deterministic synthetic accounting behavior under the tested
review delays. It does not provide an external review authority, activate V2,
publish weights, by itself prove durable-state hardening, or measure GPU
performance.

The signer-free composed shadow subsequently passed on testnet netuid 307 at
finalized block 7,586,146 with metagraph size 6. Its explicitly synthetic states
projected 850,000 ppm of registered-CROWN payout, 50,000 ppm of reviewed-discovery
payout, and 100,000 ppm of reserve, exactly 1,000,000 ppm total. It wrote
`submitted=false`; receipt semantic digest
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`,
receipt-file SHA-256
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`.
This is read-only projection/membership feasibility evidence. It constructed no
wallet and provides no review, settlement, publication, debt-debit, or activation
authority.

A separate multi-pass restart audit then exercised claim/event/cardinality
substitution cases. Reopen now derives exact paired qualification/evidence/CROWN
speed, principal, family clocks, discovery lifecycle, and all balance transitions
from their immutable journals before filtering status or allowing an upgrade. The
reproduced cases are retained regressions; final results were 98/98 focused,
2,135 passed with 19 skips repository-wide, and 111/111 on the pod. This hardens
the inactive state implementation; it does not close the production authorities
listed below.

The implementation retains a controller-supplied, content-bound `review_digest`,
but does not independently reopen or grade an external review system. That review
authority is therefore still an activation blocker, not an enforced governance fact.

## What miners should optimize for

- Submit a real, reproducible frontier improvement as soon as it is ready. The
  time bonus is capped at 10%, while waiting risks being pre-empted and delays all
  payout.
- Optimize one canonical target well. Packaging the same work into more bundles,
  hotkeys, singleton targets, or overlapping atomic targets cannot create extra
  reward families.
- Do not split a gain merely to manufacture credit. Log units remove the base
  split advantage; every accepted piece also resets the family clock.
- Do not count on noisy borderline measurements. A crown requires two independent
  passes and uses the lower measured speedup.
- Do not assume that labeling work “discovery” earns a bounty. The validator-owned
  review chooses the disposition, and the current implementation is inactive.
- The intended rule forbids bounty-plus-promotion for the same work, but do not treat
  that as fully enforced yet: promotion transport and cross-lane work identity are
  still missing.
- Keep the hotkey registered while a balance is open.

## What the numbers do—and do not—promise

Claims are denominated in confirmed validator **weight-part epochs**, not in a
fixed amount of TAO or alpha. Actual token emission also depends on Bittensor
consensus, the validator's realized influence, subnet state, and chain mechanics.
Optima can state exactly what weight share its accounting owes; it cannot promise
a token conversion rate it does not control.

Each activation policy is versioned and content-addressed. Schema-5 migration creates
empty composition tables and no retroactive debt for legacy crowns or discovery
awards. Activation fails if any legacy discovery row is retained—V1 has no journal
that proves a mutable terminal flag—and requires validated, clean open-debt state;
once composition is active, the legacy automatic discovery
award path is disabled. Those guards do not make the cutover atomic: core and
composition activate in two steps, leaving a race unless the operator quiesces legacy
settlement/publication. Production must also retain and catch up the exact missed
publication boundary before later gapless epochs, retain membership departure history
instead of applying only a latest snapshot, bind independently graded review and
runtime-invalidation authority, freeze the real family catalog/reserve and rerun the
selection/shadow, complete promotion linkage, and transport the production audit.
Parameter changes cannot silently rewrite existing claims.

For the retained technical authority and migration boundary, see
[EMISSIONS_POLICY.md](EMISSIONS_POLICY.md). For the evaluation gates that must be
passed before any claim exists, see [HOW_OPTIMA_WORKS.md](HOW_OPTIMA_WORKS.md) and
[MINER_GUIDE.md](MINER_GUIDE.md).
