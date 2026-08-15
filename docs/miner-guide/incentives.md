# How miners earn rewards

Cacheon does not reward the act of uploading a kernel. It rewards a measured
improvement that survives independent reproduction and settlement. When the
operator enables the eval-cost gate, each admitted proposal must also transfer
the published TAO amount to the current subnet owner coldkey; that transfer is an
anti-spam admission cost, not a reward.
`--pay` freezes the quoted amount for about one hour (300 blocks) so the price cannot
move while the transfer is in flight.

!!! tip "The 30-second answer"
    You submit one optimization for a target in a validator-published evaluation
    arena. The validator compares it with that arena's exact evaluation incumbent
    on the registered workload and checks that it remains within the required
    behavior and quality limits. If the same optimization passes two independently
    bound qualification attempts, settlement may name it the new **crown** for that
    target and record the corresponding reward claim in the same transaction.

    The active policy determines how that claim contributes to validator weights.
    A separate publisher later combines all eligible claims into a weight vector and
    confirms that vector on-chain. Merely submitting code, reporting a local
    benchmark, or passing once earns nothing.

## Why participate?

The opportunity is simple: make one published part of the validator's inference
workload faster. If the improvement wins and is independently reproduced, your
miner hotkey—the Bittensor identity used to submit—can receive a share of that
validator's on-chain weight. Bittensor uses network weights to determine token
emissions.

Cacheon is designed so that a participant can contribute one valuable part of an
inference engine rather than build or operate the whole system. The validator
supplies the model, workload, current stack, reference behavior, and measurement
environment. The miner focuses on one published optimization target.

The value proposition has three parts:

- **Results, not promises.** Reward eligibility follows an end-to-end improvement
  measured by the validator and reproduced independently. It does not depend on a
  self-reported benchmark or how large the submission is.
- **One focused improvement.** A miner can improve one published kernel or
  multi-kernel target. Later candidates are measured on top of the current
  frontier, so contributors do not need to resubmit the rest of the engine to
  receive credit for one change.
- **Verifiable credit.** The chain records which hotkey submitted each exact
  bundle and when. Qualification, settlement, and the reward claim retain that
  identity, creating an auditable record of the measured contribution.

This is an opportunity, not a guaranteed payout. It is economically sensible
only when your expected share under the operator's active policy justifies your
development and compute costs. A failed, unreproduced, or unsettled proposal has
no reward claim, and an unconfirmed weight publication does not realize a
projection on-chain.

## From proposal to possible emission

```text
submit -> pass twice -> settlement crowns proposal and records claim
       -> validator publishes confirmed weights -> network determines emission
```

The stages have different meanings:

| Stage | What it establishes | Reward status |
|---|---|---|
| Finalized reveal accepted into intake | Exact proposal identity, miner hotkey, and finalized arrival order | No reward |
| First qualification `PASS` | One complete attempt succeeded | No reward; the proposal is `reproduction_pending` |
| Qualified | A fresh independently bound attempt reproduced the result | No reward yet; settlement is still pending |
| Crown settled | Settlement selected the proposal and recorded its crown and reward claim together | The claim is eligible under the active policy |
| Weight publication confirmed | The intended recipients and weight values were read back from finalized chain state within the verifier tolerance | Cacheon's projection is realized; token income remains network-dependent |
| Crown retired or neutralized | The standing claim is no longer active | V1 standing credit stops |
| Active claimant becomes ineligible or evidence cannot reopen | Reward authority cannot be projected safely | Publication is held; the missing share is not redistributed |

Passing twice makes a proposal eligible for settlement; it does not guarantee a
win. Settlement rechecks the evidence and compares competing proposals for the
same or overlapping work. It crowns at most one registered candidate before the
incumbent changes, and other current candidates are held for fresh qualification
against the new incumbent. The winning transition may also retire or neutralize
claims displaced by registered target overlap.

The speedup used for settlement is deliberately conservative:

```text
settled speedup = min(primary speedup, reproduction speedup)
```

This prevents one unusually favorable run from setting the reward basis.

## What “validator weight” means

Settlement records who earned credit. A separate publisher calculates how the
validator should divide its weight among all eligible hotkeys and submits that
division on-chain.

That confirmed vector is Cacheon's economic output, not a fixed cash or token
prize. Bittensor combines it with the weights and stake of other active validators
at the subnet's epoch. Network consensus, clipping, subnet emission, and the
validator's realized influence determine the final miner emission. A Cacheon
claim therefore does **not** promise a fixed amount of TAO, alpha, fiat value, or
even a fixed percentage of subnet emission.

The terms mean:

- a **claim** is the validator's record that a hotkey earned credit for an
  accepted improvement;
- a **projection** is the calculation that divides this validator's weight among
  eligible hotkeys;
- a **confirmed publication** means the intended recipients and weights were read
  back from finalized chain state within the verifier tolerance; and
- **realized emission** is the token amount the wider network ultimately
  allocates.

### Why this page does not quote an alpha amount

A Cacheon weight share is one validator's intended allocation. It is not the
miner's final Bittensor incentive share. At an epoch, Bittensor filters active
permitted validators, weights their rows by stake, computes consensus, clips
outlying weights, and only then derives each miner's incentive. The miner's alpha
also depends on how much alpha is available to miners in that particular epoch.

That means a statement such as “this Cacheon claim has 10% of one validator's
weight” is not enough to calculate “this miner receives 10% of alpha.” A defensible
alpha estimate needs a specific live subnet and epoch, its accumulated alpha,
tempo and halving state, the complete active validator stake-and-weight matrix,
the subnet's consensus parameters, and the resulting miner incentive. Those
inputs can change until the epoch runs.

Cacheon has no retained mainnet epoch that binds all of those inputs to a crowned
miner, so this page does not invent a numeric alpha example. Once such a finalized
epoch exists, the honest example is a historical calculation from that exact chain
state—not a conversion from local speedup alone. See Bittensor's official
[emissions and Yuma Consensus explanation](https://www.bittensor.com/docs/concepts/emissions).

## Which reward policy applies?

The repository contains two intentionally separate policy generations. A miner
cannot choose between them. The operator must announce the active policy,
including its exact digest, chain scope, arena, target catalog, and publication
cadence.

| Policy | Plain-English model | Current status |
|---|---|---|
| **Legacy V1 standing rewards** | The current crown for each active target receives standing credit based on its reproduced improvement. That credit decays with age and is normalized relative to all other live claims. | Implemented and exercised end to end on testnet; this does not establish mainnet economics. Check the operator announcement for the deployment you intend to join. |
| **V2 finite debt** | An eligible post-activation crown receives a bounded claim that is paid down over later confirmed epochs. A later crown does not erase the unpaid balance, but the old crown receives no perpetual royalty. | Design retained; the implementation was extracted from the tree on 2026-08-09 and would return as a new reviewed change. It creates no claim and pays nothing today. |

Only legacy V1 can publish weights. Do not estimate a current reward with the
V2 formula; until a future release reintroduces and an operator activates it,
V2 creates no claim and pays no principal.

## How V1 standing rewards work

V1 is relative rather than fixed:

1. A settled crown creates one active standing claim for its registered target.
2. The claim's starting credit comes from the conservative improvement above the
   previous incumbent, not from total code size or effort.
3. Credit decays with claim age. Age starts at the proposal's finalized
   submission block, so qualification or settlement delay does not restart the
   clock.
4. The projector reopens every live claim, aggregates credit by miner hotkey,
   and normalizes all eligible credit into one 1,000,000-part weight vector.
5. A separate signer journals, submits, reads back, and confirms that vector.
6. A later crown for the same target retires the previous standing claim.

If live legacy discovery claims exist, they share a separately configured,
bounded discovery pool; otherwise that capacity remains with standing claims.
An invalid target set, missing evidence, or an ineligible active claimant prevents
the complete projection. Submission or readback failure leaves publication
pending or held. Neither path silently assigns a missing share elsewhere.

### A simplified V1 example

Suppose a candidate records `1.040x` in its first passing attempt and `1.034x`
in independent reproduction. Settlement can use at most `1.034x`.

Under V1, the `3.4%` marginal improvement becomes the input to the claim's
standing-credit calculation. It does **not** mean the miner receives 3.4% of
tokens or alpha. The final Cacheon weight share depends on that claim's age,
every other live standing claim, any live discovery pool, claimant eligibility,
and successful publication. If a later accepted contribution replaces this
crown, its V1 standing credit ends.

The exact integer formula, flooring order, failure rules, and operator commands
are in [Legacy V1 emissions policy](../reference/emissions-policy.md#legacy-v1).

## What actually determines a miner's reward?

At minimum:

- the exact policy and parameters announced for the deployment;
- the lower speedup from the two accepted qualification attempts;
- whether settlement crowns or holds the proposal, and whether its target
  displaces an overlapping incumbent family;
- the proposal's finalized block and, under V1, its age;
- other live standing and discovery claims;
- whether a newer crown replaces the contribution;
- whether the claimant hotkey remains eligible in the bound metagraph;
- whether the validator publishes and confirms the calculated vector; and
- Bittensor's consensus, subnet emission, and chain mechanics after publication.

This is why a local speedup cannot be converted honestly into a token estimate
by itself.

## Check this before spending money

Before renting GPUs or submitting, obtain the operator's current announcement
and verify:

- network and netuid;
- active arena and target catalog;
- eligible targets and current evaluation stack;
- active incentive generation and exact policy digest;
- policy parameters and weight-publication cadence;
- claimant-registration requirements; and
- the status or receipt surface used after reveal.

If those facts are missing, there is no defensible payout estimate. Repository
support for a policy is not evidence that a particular deployment has activated
it, announced matching policy authority, or produced confirmed publication
evidence.

## What about discovery work?

A cross-cutting idea that does not fit a registered target uses the separate
[discovery lane](discovery-lane.md). Discovery does not receive a standing target
crown.

Under V1, a qualifying discovery that completes settlement as
`DISCOVERY_BOUNTY` may receive one bounded, non-renewable claim under the active
policy. If V2 were activated, the implemented discovery path would support only
a one-time bounded `bounty_only` claim. Turning discovery work into a permanent
registered target is not implemented, and inactive V2 cannot currently create a
live reward.

## How inactive V2 would differ

V2 uses finite accounting principal for eligible post-activation crowns instead
of projecting them through V1 standing-credit arithmetic. Activation does not
convert an existing V1 claim or create retroactive principal.

- an independently reproduced, post-activation `CROWN` that completes settlement
  issues bounded principal based on its conservative marginal improvement;
- later crowns do not erase already issued principal;
- confirmed epochs pay down live claims subject to the epoch capacity;
- unpaid claims can expire or be forfeited, so issuance does not guarantee that
  the full amount will be collected; and
- principal changes only after finalized weight readback and retained intake
  catch-up.

The implementation currently accepts one immutable MiniMax-M3 campaign. Model
rotation, a second live campaign, and successor activation need a new protocol.
The deterministic load study tests accounting behavior, not live miner income or
token value.

See [Finite-debt V2](../reference/emissions-policy.md#finite-debt-v2) for the
exact policy and [Current status](../reference/state-of-record.md#inactive-v2-finite-debt)
for the maintained activation boundary.

## Read the exact rules

This page explains the participant-facing mechanism. The normative arithmetic
and operational boundaries live in:

- [Emissions policy](../reference/emissions-policy.md) — exact V1 and V2 formulas;
- [Settlement and weights](../validator-guide/settlement-and-weights.md) — validator
  settlement, signing, publication, and recovery;
- [Current status](../reference/state-of-record.md) — what has actually been
  exercised and what remains inactive.
