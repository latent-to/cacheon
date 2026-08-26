# Testnet runbook

This runbook exercises native timelock submission and the current finalized intake path.
It does not claim that the repository ships a turnkey production arena provider.

Testnet validates chain integration; it does not magically make off-chain GPU evidence or
release operations authoritative. Keep the plane boundaries explicit:

| Plane | Testnet can exercise | Still supplied off chain |
|---|---|---|
| Submission | Registration, commit/reveal, finality, canonical event order | Stable HTTPS archive origin |
| Intake | Finalized history, reservation, fetch, re-hash, publication, restart | Validator-owned storage and policy |
| Arena | Selection of a registered arena ID after injection | Production provider, worker fleet, prompts, topology, calibration |
| Qualification | Persistence/settlement joins when a real provider executes them | Resident crossover, audit/T GPU execution, and retained evidence |
| Emissions | Finalized metagraph/readback, V1 dry run | Genuine crown, approved activation bundle, signer operations |
| Release | Nothing inherently chain-hosted | Integration review, release key, registry, build and serving fleet |

## What is on chain

A miner commits one canonical payload:

```json
{"v":1,"h":"<64 lowercase hex SHA-256>","u":"https://host/bundle.tar.gz"}
```

The payload is limited to 1024 UTF-8 bytes and posted through the chain's native timelock
commit-reveal interface. The validator uses finalized reveal event order as priority.
Proposal bytes, evaluation evidence, stack state, and release artifacts remain off chain.

Priority is the finalized `(block, event index, event subindex, hotkey, content hash)`
arrival key. The reservation identity is a separate canonical digest that also binds the
block hash, URL, and finalized-payload digest. `chain-status` can show revealed
commitments before the validator has accepted them as finalized history; wait for
finality rather than using display order or local arrival time as authority.

## Prerequisites

- An explicitly configured WebSocket endpoint and netuid.
- A compatible Bittensor SDK; run `cacheon chain-compat` after installation or upgrade.
- A wallet and hotkey for registration/submission.
- Faucet funds or other testnet funding required by that network.
- An HTTPS origin that serves the exact archive bytes stably.
- Durable local storage for the intake database and both storage roots.

Registration may require coldkey authorization. Recurring submission and weight calls use
the hotkey. Keep the coldkey off long-lived validator and evaluator hosts.

## Set explicit network variables

Do not rely on a mutable SDK alias when you intend a particular endpoint:

```bash
export CACHEON_NET="wss://test.chain.opentensor.ai:443"
export CACHEON_NETUID="307"
```

Confirm the endpoint and netuid for your deployment; the values above are examples, not a
promise that a public test subnet remains available.

## Register a hotkey

```bash
cacheon chain-register \
  --netuid "$CACHEON_NETUID" \
  --network "$CACHEON_NET" \
  --wallet default \
  --hotkey miner
```

The command is idempotent for an already registered hotkey and prints chain preflight
results.

## Package and host a proposal

```bash
cacheon chain-package ./my_bundle --out ./hosted/my_bundle.tar.gz
```

Upload that archive without changing it. The production payload and fetcher accept
**HTTPS only**. `file://` is available solely through explicit hermetic-test helpers, and
plain HTTP is not supported. Intake preflights the raw gzip/tar stream, including bounded
PAX/GNU extension payloads, before extraction.

## Submit

Inspect the exact payload without signing:

```bash
cacheon chain-submit ./my_bundle \
  --url https://example.invalid/my_bundle.tar.gz \
  --netuid "$CACHEON_NETUID" \
  --network "$CACHEON_NET" \
  --wallet default \
  --hotkey miner \
  --dry-run
```

Replace the URL with the real stable HTTPS location, then remove `--dry-run`:

```bash
cacheon chain-submit ./my_bundle \
  --url https://bundles.example.org/my_bundle.tar.gz \
  --netuid "$CACHEON_NETUID" \
  --network "$CACHEON_NET" \
  --wallet default \
  --hotkey miner \
  --blocks-until-reveal 10
```

The committed content hash is rederived from the local bundle and later from the fetched
and extracted tree. Hosting different bytes causes a durable rejection.

Eval-cost defaults off. When the operator enables it, add
`--eval-cost-tao-rao 1000000000` to `chain-validate` and follow the miner `--pay`
sequence in [Submitting a proposal](../miner-guide/submitting.md#step-by-step-commands).

## Inspect chain state

```bash
cacheon chain-status \
  --netuid "$CACHEON_NETUID" \
  --network "$CACHEON_NET" \
  --wallet default \
  --hotkey miner
```

Wait for the reveal to become finalized before expecting intake to reserve it.

## Run one finalized intake pass

```bash
cacheon chain-validate \
  --netuid "$CACHEON_NETUID" \
  --network "$CACHEON_NET" \
  --intake-db chain_intake/intake.sqlite3 \
  --private-root chain_intake/private \
  --publication-root chain_intake/worker \
  --intake-only \
  --once
```

Expected output summarizes finalized block, arrivals, reservations, immutable
publications, copies, rejections, screens, decisions, settlements, and holds. In
intake-only mode the screen, qualification, and settlement counts remain disabled.

Run the same command again. No new reveal should be republished; durable identity and the
finalized cursor make the pass idempotent.

Retain both summaries plus the SQLite/database scope, immutable publication receipt,
committed content hash, finalized block/hash/event position, and HTTPS response metadata.
A successful fetch without those joins is not a restart proof.

Then perform controlled restart drills on a disposable testnet deployment:

1. stop between finalized reservation and fetch; the row must remain pending;
2. interrupt an active fetch; restart must retain `NO_DECISION`/hold rather than claim a
   candidate failure;
3. stop after publication but before copy reconciliation; the next pass must reconcile
   authoritative later copies idempotently; and
4. attempt a second controller against the same database; it must fail on exclusive
   ownership rather than run concurrently.

Do not perform these drills against the only copy of standing production evidence.

## Full qualification is a deployment integration

The console entry point cannot construct a production `ArenaServiceRegistry`. Full
validation requires reviewed Python deployment code to inject a registry/provider and
select the arena. See [Arena service](arena-service.md) and
[Qualification](qualification.md).

`chain-validate` accepts only its documented intake and arena controls. External evaluator
commands, scoring policy, chain-signing credentials, and weight publication belong to
separate authorities and must not be added to the intake service definition.

With an injected deployment registry, the same testnet loop may screen, qualify, retain a
two-PASS pair, and settle. That does not move qualification onto chain: the chain supplies
arrival and current metagraph authority, while the registered OCI/referee fleet produces
and retains the evidence. Production version-3 qualification uses current v7
resident B/C with conditional B′ or v8 two-process B/C/B′, then audit and
pristine T. Verify the primary and exact physical-lane-swapped reproduction
separately before daemon mode.

## Weight dry run

Only after the SQLite store contains genuine current-schema crowns and complete active
arena state can the separate signer build a projection:

```bash
cacheon set-weights \
  --intake-db chain_intake/intake.sqlite3 \
  --netuid "$CACHEON_NETUID" \
  --network "$CACHEON_NET" \
  --wallet default \
  --hotkey validator \
  --half-life-blocks <BLOCKS> \
  --discovery-lifetime-blocks <BLOCKS> \
  --discovery-pool-ppm <PPM> \
  --refresh-blocks <BLOCKS> \
  --dry-run
```

Dry run refreshes live authority and computes the projection without creating a
publication intent. Stable CLI output reports the projection digest, dry-run status,
chain-match result, and `submitted=False`; it does not emit the projected UID/weight
vector or every projection input. An empty intake-only test database cannot be turned
into weights by supplying arbitrary CLI values.

The sole V1 bootstrap exception is an explicit `--burn-hotkey` projection. It directs
the complete vector to one registered hotkey only while there is no claim, crown, or
activated V2 composition. Exercise it with `--dry-run` before any real submission.

The dry run still opens the exclusive SQLite authority and reads a live metagraph. Stop or
coordinate the validator loop so it does not own the database at that moment. Preserve
the printed projection digest and status. If an operational audit also requires the
effective block and hash, membership and policy digests, stack generations, or output
vector, capture the typed projection and its inputs through reviewed programmatic
instrumentation; the stock CLI does not serialize those fields to stdout.
`status=dry_run` and `submitted=False` are the expected non-emitting result.

A normal real weight test is a materially different operation: it requires a genuine crown,
persists journal intent before signing, and may submit a hotkey extrinsic. Do not remove
`--dry-run` merely to prove connectivity, and do not treat an SDK submission response as
confirmation; the exact recipient set, fixed-tolerance normalized values, and a
sufficiently new chain `last_update` are required.

For a continuously supervised V1 signer, `--watch --interval <SECONDS>` repeats complete
authority refresh and reconciliation with bounded retry. It cannot be combined with
dry-run or signer-free journal modes.

## V2 status

The V2 shadow, activation, and debt-publication commands were extracted from
the tree on 2026-08-09 without ever producing a live activation or
publication receipt. Legacy V1 `set-weights` is the only publication path.
See [Emissions policy](../reference/emissions-policy.md#finite-debt-v2).

## Evidence scope

Executing this runbook authorizes only the products that are successfully retained and
reopened. Do not infer a stronger boundary from testnet connectivity:

| Retained product | What it can establish | What it cannot establish |
|---|---|---|
| Finalized reveal, committed-tree re-hash, immutable publication, and restart reconciliation | Chain intake, proposal identity, and durable cursor behavior | GPU qualification, settlement, or release readiness |
| Metagraph-backed weight dry run | Projection construction against live chain state without an extrinsic | Signing, submission, inclusion, or confirmation |
| Structural two-pass fixture | State-machine transitions, independence checks, evidence reopening, and settlement plumbing | Empirical GPU speedup, production calibration, or arena-provider readiness |
| Builder-authenticated reproducible OCI pair plus release/session-bound serving receipts | Release-image identity and execution through the approved serving seams for that exact session; include AOT coverage for sealed direct artifacts | Qualification authority unless separate resident crossover, audit, and T evidence exists; clean-wheel, native-provider, and effective-policy gates remain separate prerequisites |

The [state of record](../reference/state-of-record.md) identifies which of these evidence
products have been completed. Deployment must supply and commission its production arena
provider independently of the command-line intake path.

## Testnet exit criteria

Before treating a deployment as production-capable, require evidence for all of:

- endpoint/genesis/netuid scope and finality behavior;
- HTTPS intake limits, committed hash, publication reopen, copy ordering, and restart;
- registered arena manifest/provider identity and queue behavior;
- faithful and broken non-crown screen controls;
- real isolated primary and independent reproduction attempts with evidence restore;
- settlement lease expiry, blocker, atomic-commit, and stale-incumbent controls;
- live V1 dry-run projection plus signer journal/readback drills under approved policy;
  and
- independently verified release build/serve/rollback if operating the release plane.

Passing only the first two bullets proves intake, not a launch-ready validator.

## Failure triage

| Symptom | Check |
|---|---|
| Payload refused before signing | HTTPS URL, lowercase 64-hex hash generation, payload size |
| Reveal not visible | Timelock, finality, endpoint/netuid, hotkey registration |
| Fetch rejected | DNS/global routing, TLS certificate, redirect chain, archive limits, committed hash |
| Reservation held | Store reason, retry count, controller restart, queue/arena capacity |
| No qualification | `--intake-only`, or missing injected registry/provider |
| Weight projection refused | No genuine crown, incomplete/stale family, metagraph change, held publication journal |
| Burn projection refused | A claim/crown/V2 activation exists, or burn hotkey is absent from the exact metagraph |
| Database ownership error | Another validator pass or signer owns the exclusive store; coordinate, never remove the lock |
| `pending` weight publication persists | Wait until retry block, inspect authoritative readback, preserve the journal |
| Weight publication becomes `held` | Audit projection/readback/signer state; append a reasoned release only after review |
| Release smoke missing receipts | This is off-chain release validation; stop rollout rather than treating testnet intake as proof |

Never “fix” a test by editing SQLite rows, weakening HTTPS, or treating
`NO_DECISION` as a candidate failure.

## Source anchors

- [Chain submission helper](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/submit.py)
- [Payload contract](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/payload.py)
- [Current validator loop](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/validator_loop.py)
- [Finalized intake store](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/intake.py)
