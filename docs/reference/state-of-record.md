# State of record

This page is the dated capability and evidence ledger for Optima. Evergreen
pages define contracts and procedures; this page identifies the implementation
revision, evidence class, and unresolved limits behind readiness claims.

Passing tests, completing an empirical qualification, activating an incentive
policy, publishing weights, and producing a deployable engine release are
different events. Evidence for one does not authorize another.

## Source snapshot

Snapshot date: **2026-07-26**

| Item | Value |
|---|---|
| Repository | [`latent-to/cacheon`](https://github.com/latent-to/cacheon) |
| Implementation baseline | [`745930474e3b721b30f9e02c2944aa94449d9253`](https://github.com/latent-to/cacheon/commit/745930474e3b721b30f9e02c2944aa94449d9253) |
| Production Python | 123 files and 99,279 lines under `optima/` |
| Tests | 112 Python files and 59,820 lines under `tests/` |
| Complete local suite | 2,310 passed, 19 skipped, 0 failed |
| Test command | `PYENV_VERSION=sn120 python -m pytest -q tests` in an unrestricted local environment |
| SGLang pin | `0.5.13.post1` in `optima/compat.py` |
| Bittensor raw-reveal storage ABI | `10.3.2` in `optima/chain_canary.py` |
| Public CLI | 19 commands |

The documentation-only migration branch does not change runtime source,
examples, kernels, or operational behavior. File and line counts describe the
implementation baseline; they are not quality metrics. The suite is
CPU/non-empirical validation and does not establish GPU performance,
container-runtime isolation, chain finality, or serving readiness.

## Authority order

When sources disagree, apply this order:

1. executable code, closed registries, schemas, and tests in the referenced
   revision;
2. the normative [product model](../architecture/product-model.md),
   [slot contract](../architecture/slot-contract.md), and
   [emissions policy](emissions-policy.md);
3. authenticated retained evidence and immutable publications;
4. this dated ledger; and
5. campaign notes, plans, console output, and historical narrative.

The evidence classes are intentionally non-substitutable:

| Evidence | Establishes | Does not establish |
|---|---|---|
| Unit/focused suite | Implemented invariants under test fixtures | Real GPU, chain, or serving behavior |
| `scan` / `verify` | Static policy and component correctness | Complete-engine speed, pristine quality, or settlement |
| Resident screen | Registered routing decision | Qualification PASS, crown, or reward speedup |
| One qualification PASS | Decision under one frozen authority | Crown or release readiness |
| Two independently bound PASSes | Settlement eligibility for one exact contribution/context | Integration or serving authorization |
| Confirmed weight journal | Exact vector read back under the publisher's policy | Qualification, activation, or release authority |
| Signed release verification | Descriptor, artifact, model, and signature consistency | Successful registry build or production serving |

## Implemented surfaces

### Submission, intake, and transport

- Finalized timelock commit-reveal establishes ordering and content identity.
- The fetcher enforces HTTPS/network policy and bounded gzip/tar preflight,
  including PAX/GNU extension payloads, before extraction.
- Current bounds include 64 MiB compressed, 256 MiB extracted, 4,096 members,
  16 MiB per file, 8 MiB per inspectable file, and 32 MiB aggregate inspectable
  content. Extension headers are bounded separately.
- Deterministic re-hash, cumulative copy disposition, immutable worker
  publication, and reopen-before-use remain required.
- Replayed discovery proposals are terminally disposed or deduplicated before
  screening. Legacy schema-3 single-PASS migration holds are non-crownable and
  have an evidence-preserving archive command.

### Slots, targets, and direct artifacts

The executable catalog contains 11 slots and one registered atomic target:

| Kind | Registered identifiers |
|---|---|
| Op | `activation.silu_and_mul`, `norm.rmsnorm` |
| Block | `attention.sdpa`, `attention.decode`, `attention.msa_block_score`, `attention.msa_prefill_block_score`, `moe.fused_experts` |
| Collective | `moe.fused_experts_reduce`, `collective.all_reduce`, `collective.ar_residual_rmsnorm`, `collective.moe_finalize_ar_rmsnorm` |
| Atomic target | `collective.moe_epilogue.v1` over the two MoE epilogue collective members |

The closed direct-artifact registry has one crownable provider,
`cutlass.cute.cubin.v1`. Candidate compiler-factory code runs in a GPU-hidden,
no-network child and may publish one sealed CUBIN. Validator code owns ABI
admission, ordinal binding, pointer/scalar/TMA materialization, launch, storage,
cleanup, and evidence. The schema exposes collective vocabulary, but the
standard provider does not supply arbitrary group/peer resolvers; unsupported
plans fail closed.

### Routing-only resident screen

The abbreviated-serving stage may keep a stock engine resident and hot-swap a
bounded candidate queue. Each swap is generation-bound, triggers graph
recapture, and is checked by shared stock brackets and contamination canaries.
The calibrated screen policy is retained by the arena provider.

Direct AOT artifacts, dependency patches, native rebuilds, and setup hooks are
not safely hot-swappable. They receive a typed screen waiver and proceed to
dedicated qualification. A waiver and a screen promotion are routing products,
not qualification evidence.

### Resident adaptive qualification

Production providers select qualification policy version 3:

1. two isolated resident TP lanes are assigned incumbent and candidate roles;
2. speed begins with B/C/B′;
3. borderline evidence adds C′/B″ under the frozen escalation rule;
4. a registered sampled slot audit runs in a separate eager, untimed candidate
   role and is regraded by the host;
5. candidate lifetimes are destroyed before candidate-free pristine T grades
   the sealed trajectory; and
6. an eligible reproduction exchanges the physical incumbent and candidate
   lane roles.

Version 1 and version 2 evidence remain readable for historical compatibility.
Screen measurements do not enter this authority. Candidate-attributable
failure can produce `FAIL`; infrastructure, drift, missing evidence, or broken
authority produces `NO_DECISION`.

From 2026-07-25 the resident speed policy of record was version 2: the scored
rate is the
steady-state timed window (`timed_tokens / timed_seconds`), conditioning stays
bounded by the sealed operational timing budget rather than entering the scored
rate, and the calibrated maximum baseline disagreement must be at most 2%.
Version-1 witnesses graded the charged rate and regrade only under their own
sealed arithmetic; the policy version is digest-bound and cross-version
splicing is refused. The motivating evidence is the 2026-07-24 stage-exit
described under empirical evidence, retained verbatim as the repository test
fixture `tests/fixtures/speed_stage_exit_45cbcc04.json`.

Policy version 3 is implemented, tested, and as of 2026-07-25 has produced its
first settled production program (described under empirical evidence): the
scored rate becomes the
median over per-batch timed windows retained in the witness rows, each read
carries a sealed window-scatter bound that refuses grading of an unfit
measurement everywhere (live and on reopen), version-3 timed reads
request no log-probability collection so evaluation work never shares the
clock with a speed measurement (`top_logprobs_num` 0 is now expressible
through the session protocol, worker, and binary evidence codec), and a
sealed conditioning slowdown bound fails a candidate whose unscored
conditioning span (the host-visible prefill surface) regresses past the
baseline's at equal warmth position — graded from spans already sealed in
every read, adding no measurement time. The companion quality mode
(decided 2026-07-25) is teacher-NLL-only: a zero top-k width in the
qualification profile and raw quality binding selects it end to end —
empty support rows through the reference protocol and worker, explicit
null distribution/KL evidence with uniformity enforced at every layer,
and typed refusal of any threshold policy that names a distribution
metric against it. Distribution-level numerics coverage remains with the
in-engine slot audit stage. The
supporting instrument authority, `optima/eval/box_certificate.py`, seals
per-session stock-vs-stock null floors, a resolved known-magnitude effect,
double-bounded expiry, and typed instrument-invalidation records. Overnight
2026-07-24/25 measurement context: two version-2 joined primaries passed
clearly while the timed noise floor of the box deteriorated 0.72% to 3.09%
across the night and the final calibration honestly refused; version 3 is
the structural response.

The audit gate is Torch-free, checks exact slot × TP-rank/process coverage, and
canonicalizes floating-point facts before durable receipt identity. Audit is
authoritative only when the frozen plan registers the matching requirement.

### Settlement

Settlement requires two complete PASS attempts over the same economic identity
with distinct authority/evidence commitments and, for version 3, the required
physical-lane role swap. It uses the lower accepted speedup, reopens exact
evidence, and commits the candidate disposition, hash-chained events, claims,
and optional evaluation-stack transition transactionally.

Held reservations require a typed evidence-preserving disposition. Lease expiry
is not arena retirement, and the repository does not implement a generic typed
arena-retirement transition.

### Legacy V1 weights

`set-weights` is a separate signer control plane with intent-before-submit,
readback, pending, held, released, and confirmed states. It supports:

- signer-free dry-run and reconciliation;
- an all-uncrowned bootstrap projection to a registered `--burn-hotkey`;
- stable-UID finality catch-up when authority and weighted-recipient mappings
  remain unchanged; and
- continuous `--watch` operation with bounded retry behavior.

Burn becomes invalid when a claim, crown, or active V2 composition exists.
Submission success is never inferred from absence of an SDK exception; exact
recipient/value readback and `last_update` govern confirmation.

### Inactive V2 finite debt

The implemented V2 path contains:

- fixed-point finite registered-CROWN debt;
- a separate bounded reviewed-discovery bounty class;
- content-addressed campaign and composition policies;
- a wallet-free atomic activation command; and
- gapless, cadence-bounded `set-debt-weights` publication that debits only
  after exact boundary/vector/policy/readback confirmation.

Pure arithmetic supports one 100% campaign or two 50% research campaigns. The
implemented activation path accepts exactly one immutable MiniMax-M3 campaign
at 100% sizing. Rotation, a second live campaign, and successor activation are
unsupported.

V2 is **inactive** in this snapshot. There is no retained live activation or
V2 publication receipt. Durable reviewed discovery can remain
`review_pending` and can issue `bounty_only` debt, but registered promotion
fails closed until typed promotion transport, target registration, fresh
requalification/CROWN linkage, and cross-lane work identity exist.

### Engine release

Evaluation and serving remain separate products. The release model includes
reviewed integration records, sealed model/native identities, deterministic
source/wheel products, SBOM/provenance, Ed25519 signatures, OCI context, host
policy, registry types, and serving receipts.

Current release authority is incomplete:

- the serving wheel does not close every manifest/direct-artifact runtime
  import;
- release preparation does not provider-specifically rebuild and reopen the
  complete CuTe index/compile-profile authority;
- `release_runtime.py` does not propagate the signed CuTe compile-profile
  digest into the engine process;
- builder output, effective runtime arguments, management-route policy, and
  complete release/session receipt binding still require end-to-end closure;
  and
- no final deterministic registry pair, authorized image, or complete all-rank
  serving receipt set is claimed for this revision.

Loading sealed native artifacts inside evaluation OCI proves evaluation
runtime support. It does not close the serving release.

## Empirical evidence

### Retained B200 qualification and settlement

The strongest retained production-shaped crown evidence predating the resident
version-3 path is a TP4 joined block-score qualification on an 8×B200 host:

| Field | Primary | Reproduction |
|---|---:|---:|
| Charged-basis speedup | 1.0561× | 1.0487× |
| Timed-section diagnostic | 1.0932× | 1.0866× |
| Decision | PASS | PASS |

Settlement used the lower value, 1.0487×, committed a generation-1 crown, and
reopened successfully after restart. The attempts had distinct required
authority/evidence digests. They do not prove distinct operators or failure
domains.

This evidence used an earlier SM100 worker image and SGLang source build
`0.0.0.dev1+g56e290315`, not the repository pin `0.5.13.post1`. It predates
the resident version-3 schedule and current audit transport. It therefore
demonstrates the earlier bound qualification/settlement mechanism, not a
current-revision production canary.

### MiniMax-M3 fused-epilogue evidence

Earlier 4×B300 runs measured shallow and deep fused-epilogue submissions through
the historical referee and later through the testnet intake loop. Those runs
remain mechanism and performance evidence for their exact runtime, policy, and
hardware. They are not retroactive current-schema crowns and do not qualify the
resident version-3 path. See the
[MiniMax-M3 evidence note](../results/minimax-m3.md).

### Current audit-path canary status

A bounded 4×B300 run on 2026-07-19 did not satisfy the current launch gate. The
sabotage control was rejected. The honest primary produced no verdict after
concurrent legs shared an executor label and invalidated quiescence authority.
The honest reproduction passed graph and pristine-T quality, but its deep slot
had only four audited calls per rank against the required 32 and its speed gate
failed at 1.005507×. Zero observed audit comparison violations did not repair
insufficient coverage.

This is retained failure evidence. It is not an activation canary, PASS, or
performance authority. The subsequent resident screen and two-lane adaptive
qualification implementation are test-covered and informed by GPU calibration,
but no retained end-to-end current-revision version-3 primary/reproduction
canary is claimed here.

### First production version-3 speed verdict (2026-07-24)

A 4×B300 joined primary on 2026-07-24 executed the resident version-3 path end
to end for the first time: prepare, intake hygiene, graph, and the slot audit
all passed, and the speed stage produced the first production version-3 speed
verdict. That verdict was `FAIL speed_regression` under version-1 charged-basis
arithmetic: the second baseline read ran as a warm continuation of the first
baseline session while the candidate read ran cold, so the conditioning-
inclusive scored rate turned a positional split into 6.3% apparent baseline
noise and a 1.126 required bar, while the candidate was faster than both
baseline reads on every timed window. The sealed stage-exit is retained
verbatim as `tests/fixtures/speed_stage_exit_45cbcc04.json`, and a regression
test pins both readings: version-1 arithmetic reproduces the shipped verdict
exactly, and version-2 timed-basis arithmetic grades the same sealed reads as a
clear pass. The reservation's terminal disposition stands; no verdict was
altered after the fact, and any future attempt requires a fresh submission
under the version-2 policy.

### First settled crown under speed policy version 3 (2026-07-25)

A 4×B300 program on 2026-07-25 ran the resident version-3 policy through every
production phase for the first time and settled the first crown: intake, graph
verification, controller snapshot, two per-lane calibrations, a joined
primary, a joined lane-swapped reproduction, a restart proof, and a signer-free
weight projection all passed in one continuous program against testnet
finalized intake (reservation
`c7713892…`, target `collective.ar_residual_rmsnorm`, candidate content
`747405b4…`, source revision `07c032ed`). The calibrations sealed per-lane
timed noise floors of 0.30% (primary lane) and 0.125% (reproduction lane)
against the 2% ceiling, with negative, positive, and stock controls sealed
before any timed arm was observed. The joined primary graded a timed speedup
of 1.0212 and the reproduction, with candidate and baseline physically
swapped across lanes, graded 1.0278; baseline bracket disagreement inside the
timed phases was 0.21% and 0.03% respectively. Settlement bound the pair and
accepted the lower speedup, 1.0212. This was also the first production
execution of the teacher-NLL-only quality mode (zero top-k width) and of the
joined reproduction, restart, and weights orchestration. The evidence was
relocated to validator-owned storage with every retained artifact re-verified
against its bound content digest, and a signer-free weight projection built
from the settled state (crown count 1, full pool to the crowned hotkey).
This establishes measurement and settlement under the version-3 policy. It
does not establish on-chain weight publication (the signed submission is a
separate, operator-reviewed act), incentive activation, integration review, or
serving readiness.

### Incentive evidence

The tracked one-campaign load report contains 64 matrix rows and four burst
controls and replays to semantic digest
`505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.
It is deterministic accounting sensitivity, not chain, GPU, token-value,
miner-equilibrium, activation, or publication evidence. See
[Incentive load validation](../results/incentive-load-validation.md).

Historical signer-free shadows exercised synthetic policy fixtures against
exact testnet membership and submitted no weights. They do not authorize the
current one-campaign V2 bytes. No live V2 activation or debit-confirming
publication receipt exists in this snapshot.

## Public CLI

The live command inventory is:

```text
slots  compat  chain-compat  scan  verify
chain-package  chain-submit  chain-status  chain-register  chain-validate
chain-archive-schema3-hold
model-provision  release-verify  release-context
chain-incentive-shadow  chain-incentive-composition-shadow
chain-activate-incentives
set-weights  set-debt-weights
```

The local miner loop is `scan` plus `verify`. Complete-engine performance and
quality authority begins with a deployment-injected arena provider.

## Deployment boundary

The source implements the mainnet-shaped intake, evaluation, settlement, and
publication control planes. Moving from a completed production-shaped testnet
exercise to a live subnet still requires deployment-owned inputs and evidence:
endpoint/netuid, registrations, permit/stake, validator/miner/burn identities,
wallet/key custody, immutable hosted bundles, GPU capacity, backups,
monitoring, and the required current-version GPU canary. This page does not
claim that a live mainnet deployment or receipt exists.

## Source anchors

- [Slot catalog](https://github.com/latent-to/cacheon/blob/main/optima/slots.py)
- [Target catalog](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py)
- [Hardened fetch](https://github.com/latent-to/cacheon/blob/main/optima/chain/fetch.py)
- [Resident screening](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_screen_lane.py)
- [Adaptive resident runtime](https://github.com/latent-to/cacheon/blob/main/optima/eval/crossover_runtime.py)
- [Qualification](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_runner.py)
- [Audit gate](https://github.com/latent-to/cacheon/blob/main/optima/audit_gate.py)
- [Settlement](https://github.com/latent-to/cacheon/blob/main/optima/settlement.py)
- [Legacy publication](https://github.com/latent-to/cacheon/blob/main/optima/chain/weights.py)
- [V2 activation](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_activation.py)
- [V2 debt publication](https://github.com/latent-to/cacheon/blob/main/optima/chain/debt_publication.py)
- [Release construction](https://github.com/latent-to/cacheon/blob/main/optima/release.py)
