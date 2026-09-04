# CLI reference

The CLI is a thin operator and contributor surface over Cacheon's typed APIs. A
command's output has only the authority listed here; human-readable output is not a
qualification, settlement, or release receipt.

```bash
python -m cacheon.cli <command> [options]
python -m cacheon.cli <command> --help
```

Use module invocation on GPU hosts because SGLang starts child processes. The
installed `cacheon` console script resolves to the same parser.

## Command inventory

| Command | Audience | Authority | Purpose |
|---|---|---|---|
| `slots` | all | read-only | Print the registered slot ABI |
| `compat` | operator | diagnostic | Check the installed SGLang seam surface against the pin |
| `chain-compat` | operator | diagnostic | Check the installed Bittensor SDK surface without chain access |
| `scan` | contributor | local gate | Parse a bundle and apply recursive static policy |
| `verify` | contributor | local gate | Check declared slot behavior against validator-owned references |
| `explain` | all | read-only | Say in plain language what an evaluation product records about a bundle |
| `chain-package` | contributor | packaging | Build a canonical archive and print its content hash |
| `chain-publish` | contributor | object-store mutation | Package, publish, and anonymously verify a content-addressed proposal archive |
| `chain-eval-cost` | contributor | read-only | Print the published eval-cost quote (v1 is a fixed TAO transfer) |
| `chain-eval-cost-credit` | validator operator | private intake mutation | Grant (or list) one artificial eval-cost credit admitting one unpaid reveal from a hotkey |
| `chain-submit` | contributor | chain mutation | Commit a bundle hash and HTTPS fetch location through timelock reveal |
| `chain-status` | all | read-only | Inspect public subnet, registration, and reveal state |
| `chain-reservation-status` | validator operator | read-only private diagnostics | Explain one retained reservation without taking the validator write lock |
| `chain-miner-report` | validator operator | read-only private diagnostics | Report every retained submission for one miner hotkey with its stated cause and next step |
| `chain-evaluation-lease` | validator operator | one-shot evaluation lease transition | Preview, claim, heartbeat, or infrastructure-release store-selected work from sealed file authority |
| `chain-register` | operator | chain mutation | Burn-register a hotkey and run the SDK preflight |
| `chain-validate` | validator | production intake | Consume finalized reveals; a deployment may inject qualification services |
| `chain-snapshot` | validator | private object-store mutation | Publish and reopen a consistent validator recovery snapshot |
| `chain-snapshot-verify` | validator | recovery verification | Download and semantically reopen one snapshot, optionally into fresh staging |
| `chain-archive-schema3-hold` | validator | durable state transition | Terminally archive one exact legacy schema-3 reproduction hold |
| `chain-release-hold` | validator | durable state transition | Return one held or no-decision reservation to its queue under a stated reason |
| `chain-reopen-qualification` | validator | durable state transition | Return one unsettled two-PASS reservation to the screen queue for a fresh pair when retained evidence shows its credited half read the baseline lane under the arena band |
| `chain-backfill-lineage` | validator | durable state transition | Rebuild the per-target lineage ledger from the newest recorded crown; idempotent |
| `set-weights` | signer | legacy production control plane | Reconcile the journaled V1 projection, including bounded burn bootstrap/watch operation, or run the subnet-owner burn bypass |
| `mint-push-credentials` | operator | weight-share push auth | Create/rotate HMAC secrets for eval → serve-weights |
| `mint-weight-gateway` | operator | weight-share deploy | Create a dedicated HTTP authority wallet (and optional push credentials) for serve-weights |
| `push-weight-offer` | eval | peer weight distribution | Build the legacy V1 offer and HTTP-push; never chain-publishes |
| `serve-weights` | weights gateway | peer weight distribution | Serve/store the offer; optional authenticated PUT from eval |
| `follow-weights` | signer | peer weight publication | Fetch the shared offer and publish through the commit-reveal reconciler |
| `model-provision` | validator operator | production artifact | Seal model bytes into a content-addressed publication and receipt |

There is no local command that grants qualification or settlement authority. Complete
engine qualification begins at the deployment-injected arena boundary; a local evaluator
or JSON ledger is not an alternate production interface.

## Contribution commands

### `slots`

```bash
python -m cacheon.cli slots
```

Reads the registered `SLOTS` table without importing contribution code or requiring a
GPU. See the [slot catalog](slots-table.md).

### `scan`

```bash
python -m cacheon.cli scan path/to/bundle
```

Loads `manifest.toml` as data and recursively applies the Python policy to declared and
vendored `.py` files. Manifest-declared CUDA sources and dependency patches are admitted
as separate reviewed-build tiers; undeclared executable files, binaries, and symlinks are
rejected. Static scanning is defense in depth; a clean result does not make contribution
code trusted.

`scan` also runs a separate Triton compilability heuristic over each declared
op source. A known host-only helper inside `@triton.jit` prints
`BROKEN KERNEL` and makes this local command exit 2. That is deliberately not a
production verdict: Triton compiles each kernel only when first invoked, and the
static heuristic cannot tell whether a flagged helper is reachable from the
declared entry. In the retained 2026-08-18 corpus it also flagged dead kernels
inside bundles that had demonstrably compiled. Only sandboxed execution of the
reachable entry can support an attributable compile `FAIL`; treat the local
finding as an inexpensive reason to inspect or fix the source before submission.

### `verify`

```bash
# CPU contract smoke
python -m cacheon.cli verify examples/miner_silu_torch \
  --device cpu --dtype float32

# CUDA contract and graph verification
python -m cacheon.cli verify path/to/bundle \
  --device cuda --dtype bfloat16 --model MiniMax-M3

# Distributed verification at the arena topology
python -m cacheon.cli verify path/to/bundle \
  --device cuda --world-size 4
```

Options are `--dtype`, `--device`, `--seed`, `--world-size`, `--tp-size`, and
`--model`. The verifier rejects ambiguous variant domains, applies the model-specific
slot profile, and spawns workers for candidate execution. Collective slots use the
requested rank count; a host without enough CUDA devices falls back to CPU/Gloo unless
`--device cuda` makes the requirement explicit.

Verification proves only the exercised component contract. It does not establish model
integration, serving throughput, pristine quality, isolation, independent reproduction,
or settlement.

### `explain`

```bash
python -m cacheon.cli explain qualification-product.json
python -m cacheon.cli explain collected-result.json --log worker-stderr.log
python -m cacheon.cli explain --evidence-dir retained-failed-run/
```

Renders one evaluation product — or a validator's collected result, which
embeds it — as plain language: whether the kernel ran, whether it was correct,
and whether it was faster, with the disqualifying fact stated before any
number. What the GPUs did comes from the product's `qualification.execution`
artifact when the run published one; `--log` supplies a retained worker stderr
for runs that did not, and is the only source of the per-shape kernel trace.
When a failed run produced no product, `--evidence-dir` reads its retained
`*.stderr` artifacts directly and reports the candidate bundle hash, source
file, line, callable phase, exception, affected GPU/ranks, and failing call
chain.

## Submission commands

### Package

```bash
python -m cacheon.cli chain-package path/to/bundle --out bundle.tar.gz
```

The command writes one canonical wrapper archive and prints the deterministic content
hash of the extracted bundle tree. Host the exact archive at a stable HTTPS URL. The URL
is transport; the hash is proposal identity.

### Publish

```bash
set -a
source .env
set +a

python -m pip install -e ".[object-store]"
python -m cacheon.cli chain-publish path/to/bundle \
  --out bundle.tar.gz
```

The command reads `CACHEON_OBJECT_STORE_ACCESS_KEY_ID`,
`CACHEON_OBJECT_STORE_SECRET_ACCESS_KEY`, and
`CACHEON_OBJECT_STORE_BUCKET`. It uses generic S3 by default. Set
`CACHEON_OBJECT_STORE_ENDPOINT_URL` for any S3-compatible service; that endpoint
selects the service without a provider name and defaults to path-style
addressing. `CACHEON_OBJECT_STORE_REGION` and
`CACHEON_OBJECT_STORE_ADDRESSING_STYLE` override its signing region and URL
style. Only the `CACHEON_OBJECT_STORE_*` names are read; the pre-rename
`OPTIMA_OBJECT_STORE_*` fallback was removed on 2026-08-03.
`CACHEON_OBJECT_STORE_PROVIDER=hippius` and `minio` are optional presets,
not validator protocol identities. The miner's credentials authorize the
upload only; they are not written into the archive, URL, on-chain payload, or
validator configuration.

The object key defaults to the stable storage-compatibility prefix
`cacheon/miner-bundles/sha256/<content_hash>.tar.gz`; the prefix is not rewritten
as product branding. Publication refuses an
existing object that does not extract to the same committed hash, makes the
object anonymously readable, and verifies the resulting URL with the same
production HTTPS fetcher used by validator intake. `--dry-run` packages and
prints the planned key and URL without a remote change. `--create-bucket` is an
explicit bucket mutation; otherwise the miner must create the bucket first.
Custom S3-compatible services use `--object-store-endpoint` and optionally
`--object-store-region` / `--object-store-addressing`; a known
`--object-store-provider` preset merely fills those defaults. Public validator
URLs must still resolve to canonical HTTPS.

### Submit

```bash
python -m cacheon.cli chain-submit path/to/bundle \
  --url https://artifacts.example/bundle.tar.gz \
  --network <network> --netuid <netuid> \
  --wallet <wallet> --hotkey <miner-hotkey> \
  --blocks-until-reveal 10
```

`chain-submit` re-hashes the local bundle before constructing the payload. Use
`--dry-run` to print it without signing. `--pay` transfers the quoted TAO amount
to the current subnet owner and commits a v2 pointer. If that transfer lands but
the reveal fails, retry with `--eval-cost-payment-block` and
`--eval-cost-payment-extrinsic-index` instead of `--pay`. Full commands:
[Submitting a proposal](../miner-guide/submitting.md#step-by-step-commands).
When the operator sets `--eval-cost-tao-rao` above zero, unpaid v1 reveals fail
admission.

```bash
python -m cacheon.cli chain-eval-cost --netuid <netuid> --network <network>
```

### Inspect public chain state

```bash
python -m cacheon.cli chain-status \
  --network <network> --netuid <netuid> \
  --wallet <wallet> --hotkey <hotkey>
```

The wallet arguments add registration information. This command does not expose the
validator's private intake, screening, qualification, or settlement database.

### Register a hotkey

```bash
python -m cacheon.cli chain-register \
  --network <network> --netuid <netuid> \
  --wallet <wallet> --hotkey <hotkey>
```

This is a chain mutation and may burn registration cost. It checks for existing
registration first, then runs the SDK preflight.

## Validator commands

### `chain-eval-cost-credit`

Grant one artificial eval-cost credit: the next fee-gated reveal from the
granted hotkey that arrives **without** a payment pointer is admitted exactly
as if a submission payment had been consumed. Use it as a make-good when a
paid submission died for a validator-side reason (for example, a paid row
expired under validator backpressure, which does not release the consumed
payment).

```bash
python -m cacheon.cli chain-eval-cost-credit \
  --intake-db /srv/cacheon/state/intake.sqlite3 \
  --hotkey <miner-hotkey-ss58> \
  --coldkey <miner-coldkey-ss58> \
  --note "reservation <id>: expired under validator backpressure"
```

After granting, the miner re-submits with a plain `chain-submit` — no `--pay`
and no payment pointer. Semantics:

- One credit admits exactly one reveal; a second unpaid reveal fails
  `missing_eval_cost_payment` as usual. Credits are matched by hotkey, oldest
  grant first; `--coldkey` and `--note` are recorded for audit only.
- A credit is consumed only when the reveal is actually admitted (`reserved`
  or `deferred`). A reveal that fails for any other reason (malformed payload,
  hotkey epoch limit, ...) leaves the credit unspent.
- A credit never rescues a reveal that cited a payment pointer: an invalid or
  already-used payment keeps its verdict.
- Safe to run while `chain-validate` is live. The command writes through its
  own SQLite connection and never takes the intake controller lock; the credit
  is claimed inside the controller's own admission transaction.

Audit the trail with `--list` (optionally filtered by `--hotkey`); spent
credits print the consuming reservation and block:

```bash
python -m cacheon.cli chain-eval-cost-credit \
  --intake-db /srv/cacheon/state/intake.sqlite3 --list
```

### `chain-reservation-status`

Run this command on the CPU validator host while `chain-validate` is running:

```bash
python -m cacheon.cli chain-reservation-status \
  --intake-db /srv/cacheon/state/intake.sqlite3 \
  --audit-log /srv/cacheon/state/chain-audit.jsonl \
  --reservation-id <64-HEX-RESERVATION-ID>
```

`--content-hash` and `--miner-hotkey` are alternative exact selectors. The command
refuses either selector when it matches more than one row; use the reservation ID to
remove that ambiguity. Add `--json` for a machine-readable support record.

### `chain-miner-report`

Answers the question a miner actually asks — what happened to everything I sent —
which `chain-reservation-status` cannot, because it refuses a selector matching
more than one row:

```bash
python -m cacheon.cli chain-miner-report \
  --intake-db /srv/cacheon/state/intake.sqlite3 \
  --miner-hotkey <SS58> \
  --remote-spool-root /srv/cacheon/remote-worker
```

Each submission is reported with its typed outcome, the persisted reason code, a
stated cause, and a next step. Reasons that are not the candidate's fault —
queue-window expiry and validator-side infrastructure holds — say so explicitly
rather than reading as a verdict. A screen rejection names the stage that
stopped the bundle and the reason that stage recorded, which travels inside the
signed screen receipt. Add `--json` for the machine-readable record.

Pass `--evidence-root <dir>` (repeatable, one per worker generation) to reopen
the retained qualification evidence behind each attempt. The report then renders
what the attempt measured and, for resident-lane runs, what every GPU did with
the kernel: whether it loaded, how often each slot was called, whether those
calls were inside the CUDA graph the timed windows replay, whether it raised,
and why any call routed to SGLang's kernel instead. Those rows are published by
the run as the unsealed `qualification.execution` artifact and matched to the
submission by its publication digest; an attempt whose store is not configured
is reported as not retained rather than omitted.

Pass `--remote-spool-root <dir>` once per retained worker epoch to join the
immutable lease/recovery history with request transport events and the result's
request-scoped `worker_log`. That artifact contains adapter phases and the
bounded engine/native output routed to OCI stderr, plus sidecar identity for
every OCI lifetime in the request. A crash report therefore names the last completed
phase, observed component, exception chain, file and line, and retained raw-log
hash without searching VM or pod logs.

The submissions dashboard consumes this same reader. Its detail drawer shows the
plain-English explanation and offers one raw `.log` containing the exact
request-scoped adapter diagnostics and retained OCI diagnostic streams. The OCI
worker reserves stdout for framed protocol and redirects ordinary Python/native
stdout into that retained stream. Section headers mark missing or truncated bytes;
a bounded prefix is never labeled complete.

A candidate-owned load or invocation error is retained as its own qualification
failure product. With the corresponding `--evidence-root`, this command prints
the offending reservation and candidate arm, exact error, and diagnostic
stream hash. Fresh registered qualification is singleton-bound, so a
proved candidate exception is terminal `FAIL` without a retry or a cohort guess.

The report never derives a decision. A row carrying no typed decision is reported
as having none. It reads the same durable rows through the same read-only
snapshot and redaction helpers as `chain-reservation-status`, so the two views
cannot disagree about a verdict.

### `chain-reservation-status` internals

The command opens the live WAL database read-only and takes one consistent SQLite read
snapshot. It does not acquire the `FinalizedIntakeStore` process lock, mutate a row, or
interrupt the intake daemon. Its queue position is a position among work that is
actually selectable: reproduction work precedes new primary work, both lanes retain
finalized arrival order, and active screening or qualification is reported as active
rather than assigned a fictitious queue rank. Work held by an active remote evaluation
lease is reported as `leased`, is excluded from waiting position and depth, and includes
the lease ID, stage, generation, cohort position, and expiry block. The private worker
owner is not printed. Promoted qualification work may be selected as an indivisible
retry group or bounded cohort, so the command does not invent a per-row numeric rank for
that phase.

Human and JSON output omit the submitted URL, private publication paths, evidence-root
paths, and raw exception text. A redacted reason includes its digest so an operator can
correlate it with private incident material without disclosing that material. The
record includes the finalized arrival key, typed screen stages and grades, qualification
decisions and artifact references, retained settlement qualification references, and
whether referenced evidence is available on the host.

This is an evidence reader, not a promise that every historical failure has complete
evidence. Some infrastructure `NO_DECISION` paths in the current durable schema retain
only a failure digest. Those rows are labeled
`qualification_failure_retained_by_digest_only`; the digest proves which failure
product was recorded but cannot reconstruct its contents. Do not turn such a row into a
candidate `FAIL` or invent a cause. See the
[operator diagnostics procedure](../validator-guide/chain-loop.md#operator-reservation-diagnostics)
for the exact support interpretation and current retention boundary.

### `chain-evaluation-lease`

```bash
python -m cacheon.cli chain-evaluation-lease \
  --config /srv/cacheon/sealed/evaluation-lease.json preview
```

The owner-controlled JSON file is the sole authority for the absolute intake database
path, exact intake policy and chain scope, owner, stage, lease duration, qualification
cohort maximum, and bounded lock-retry policy. The command has no environment fallback
or target, model, hotkey, netuid, mission, endpoint, or path default.

Each invocation performs exactly one operator request and exits. It is not an
evaluation worker, daemon, or scheduler.

The operations are `preview`, `claim`, `heartbeat <lease-id>`,
`release <lease-id> --reason <reason> [--result-digest <sha256>]`, and
`requeue-expired --authority <SEALED_JSON>`. Preview is non-mutating. Claim
ordering, reproduction priority, qualification cohorts, lease generation,
heartbeat CAS, infrastructure release, and the finalized block clock remain
owned by `FinalizedIntakeStore`; this command does not evaluate or settle work.
Every success prints canonical JSON bound to the retained finalized cursor.

`requeue-expired` is the narrow validator-downtime recovery. Its owner-only
authority file must use the closed
`cacheon-validator-downtime-requeue-authority-v1` schema, the exact reason
`validator_worker_unavailable`, a nonempty reservation-ID cohort, and a disjoint
list of retained-result reservation IDs. It atomically restores only exact
expired rows to their durable pre-expiry `published` or `promoted` lane and
starts a fresh finalized-block SLA without erasing prior evidence. One ordinary
refresh is allowed if the cohort expires again; a further refresh requires the
authority to set the explicit boolean `allow_repeat_refresh`. This is not a
generic resurrection or a way to rerun a favorable terminal result.

For claim, heartbeat, and release, the durable store mutation and canonical JSON
emission are not one transaction. A process or output failure after the store commits
can therefore leave the result ambiguous. These mutating verbs have no process-level
idempotency promise: inspect the authoritative durable lease state before deciding
whether to issue another exact operation.

### `chain-validate`

The stock entrypoint supports complete finalized intake without a GPU service:

```bash
python -m cacheon.cli chain-validate \
  --network <network> --netuid <netuid> \
  --intake-only --once
```

Intake mode persists finalized order, hardened fetch and re-hash results, private
retention, immutable worker publication, and copy disposition. Storage and loop controls
are `--intake-db`, `--private-root`, `--publication-root`, `--audit-log`,
`--interval`, and `--once`. `--eval-cost-tao-rao` defaults to `0` (gate off);
set `1000000000` to require the published 1 TAO `transfer_keep_alive` to the
current subnet owner coldkey per admission, and
`--eval-cost-payment-window-blocks` to bound how old that transfer may be relative
to the reveal. `--eval-cost-quote-ttl-blocks` (default 300, about one hour) is how long a quoted
amount stays valid until the transfer is included. The transfer may precede the
reveal; unused payments remain reusable until a reserved or deferred admission
consumes them. The audit file is a redacted, fsynced JSONL chronology;
it does not contain URLs, hotkeys, candidate bytes, or exception messages, and it does
not replace SQLite as transition authority.

Qualification requires a deployment-owned `ArenaServiceRegistry` plus a registered
`--arena-id`. The repository does not construct a production provider from shell text.
A reviewed deployment wrapper calls `cmd_chain_validate(args,
arena_registry=registry)` or `run_validator(...)` after creating the registry. Invoking
the stock module without `--intake-only` refuses to run because no registry was injected.

### `chain-snapshot` and `chain-snapshot-verify`

Install the optional S3 client on the validator host, then publish a private snapshot:

```bash
python -m pip install -e ".[object-store]"

cacheon chain-snapshot \
  --intake-db chain_intake/intake.sqlite3 \
  --audit-log chain_intake/chain-audit.jsonl \
  --object-store-bucket <PRIVATE_BUCKET> \
  --object-store-endpoint <S3_COMPATIBLE_HTTPS_ENDPOINT> \
  --object-store-region <SIGNING_REGION> \
  --sealed-input qualification-inputs=/srv/cacheon/sealed-inputs
```

Credentials come from `CACHEON_OBJECT_STORE_ACCESS_KEY_ID` and
`CACHEON_OBJECT_STORE_SECRET_ACCESS_KEY` (or the equivalent flags). Generic S3 is
the default. `--object-store-provider hippius` or `minio` is only a convenience
preset for endpoint, region, and addressing defaults; a custom endpoint needs no
provider name. Archive keys default below the stable private
`cacheon/validator-archive/v1` compatibility prefix, overridable with
`CACHEON_VALIDATOR_ARCHIVE_PREFIX` or `--object-store-prefix`.

The command uses SQLite's online backup API and uploads digest-addressed blobs plus
a closed snapshot manifest. It automatically includes:

- the consistent SQLite image and redacted audit journal;
- every immutable worker publication referenced by the database;
- each retained qualification artifact referenced by
  `settlement_qualifications`; and
- only sealed-input roots explicitly named with repeatable
  `--sealed-input NAME=PATH`.

It does not discover or upload models, OCI images, wallets, credentials, caches,
unredacted logs, or unrelated evidence directories. Use a private bucket—or an
explicitly policy-isolated private prefix—not the anonymous miner-publication
namespace. Bucket encryption, versioning/object lock, lifecycle, and access policy
remain deployment responsibilities.

Verify every scheduled backup with a temporary semantic reopen:

```bash
cacheon chain-snapshot-verify \
  --manifest-key <KEY_PRINTED_BY_CHAIN_SNAPSHOT> \
  --object-store-bucket <PRIVATE_BUCKET> \
  --object-store-endpoint <S3_COMPATIBLE_HTTPS_ENDPOINT> \
  --object-store-region <SIGNING_REGION>
```

Add `--restore-root /fresh/private/staging/path` for a retained restore drill.
The destination must not exist. Restore verifies the SQLite image, audit journal,
worker publication receipts/content hashes, and evidence references, then writes a
private `restore-map.json`. It never replaces the live database or original
absolute evidence/publication paths. The staged database and restored files become
authoritative only through a separately reviewed recovery cutover.

### `chain-archive-schema3-hold`

```bash
python -m cacheon.cli chain-archive-schema3-hold \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --reservation-id <reservation-id> \
  --reason "reviewed migration disposition"
```

This is a terminal, evidence-preserving disposition for one exact legacy
schema-3 single-PASS hold. It does not qualify, reproduce, release, crown, or
publish weights. Current-schema work must use the normal authority path.

### `chain-release-hold`

```bash
python -m cacheon.cli chain-release-hold \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --reservation-id <reservation-id> \
  --reason "lane repaired; rerun"
```

Returns one `held` or `no_decision` reservation to the queue position its
retained state supports: `reproduction_pending` when one settlement
qualification exists, otherwise `published`, otherwise `transport_retry`. The
operator reason is persisted on the row. It never signs, settles, crowns, or
touches evidence, and it refuses a legacy schema-3 migration hold, which has
its own terminal command above.

### `chain-reopen-qualification`

```bash
python -m cacheon.cli chain-reopen-qualification \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --reservation-id <reservation-id> \
  --evidence-state-dir <worker-state-dir> \
  --dry-run
```

Returns one `qualified` two-PASS reservation whose settlement candidate is
still `pending` to the screen queue as `published`, exactly like a fresh
submission: the live worker screens it again, binds it to the live stack, and
measures a new independent pair against the current incumbent. The retained
candidate and both halves move to `settlement_reopenings`, so the pair stops
earning the moment it leaves `qualified`. The command refuses unless the
retained stage-exit artifacts show that the half which set the credited (lower)
speedup read the baseline lane under the arena band — the median of every
retained baseline-role read in that arena minus five percent, over at least six
reads — and it prints that evidence either way. `--dry-run` prints the evidence
and changes nothing. Crowned or otherwise settled candidates are lineage and
are refused. A reopened row binds to the stack whose service re-screens it,
never to the stack current at its original arrival; running the command again
on a reopened row that is still waiting for its fresh pair repairs that
binding (or leaves the row unbound for the screen to bind) and changes nothing
else. It never signs, settles, or crowns.

### `chain-backfill-lineage`

```bash
python -m cacheon.cli chain-backfill-lineage \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3
```

Rebuilds the per-target lineage ledger (`target_lineage_tips` and its nodes)
from the newest `CROWN` recorded per target and prints each target's tip,
parent, and winning speedup. A store that settled before the ledger existed
needs this once, or the ancestor-fork guard stays inert. It is idempotent,
recomputes the same rows from the same journal, and never signs, settles, or
crowns. Historical journals carry no transition-time snapshot, so the backfill
marks only reservations no later than the winning submission as pre-transition.

### `set-weights`

```bash
python -m cacheon.cli set-weights \
  --network <network> --netuid <netuid> \
  --half-life-blocks <blocks> \
  --discovery-lifetime-blocks <blocks> \
  --discovery-pool-ppm <ppm> \
  --refresh-blocks <blocks> \
  --dry-run
```

The command reopens settled state and computes a pure global projection. `--dry-run`
creates no publication intent and submits nothing; stable stdout reports the projection
digest, `status=dry_run`, `chain_matches`, and `submitted=False`. It does not print the
projected UID/weight vector or the complete projection inputs. A live reconciliation
journals intent before submission and confirms it only through later chain observation.
`--release-hold REASON` appends an audited release of the held publication and does not
submit. `--reconcile-only --validator-hotkey <hotkey>` confirms or releases without
constructing a signer.

`--burn-hotkey <hotkey>` is available only while retained authority is completely
uncrowned and has no active reward claim or V2 composition. It projects the complete pool
to that registered bootstrap identity and fails closed as soon as normal reward authority
exists. `--watch --interval <seconds>` runs repeated reconciliations with bounded retry
rules; it cannot be combined with dry-run, reconcile-only, or hold release. Remove the
burn hotkey before restarting after the first CROWN.

`--burn-to-subnet-owner` is the **chain-resolved all-uncrowned bootstrap**.
Each pass resolves the burn sink from the finalized metagraph RuntimeAPI
(owner coldkey matches; prefer `SubnetOwnerHotkey`, else lowest UID), builds
the crownless `{hotkey: 1.0}` projection against the intake database's empty
settlement state, and publishes it through the durable
intent/pending/confirmed journal with `require_current_crown=False` (or stops
before signing with `--dry-run`). It refuses the moment any active reward
claim, crowned evaluation arena, or activated V2 composition exists, and it
refuses a foreign in-flight journal head (its own in-flight head — same
scope, signer, policy, and vector — is resumed and reconciled instead);
those refusals are nonretryable, so a `--watch` loop stops with a typed
error at the first CROWN — restart without the flag to publish settlement
weights. It does not publish shared
weight offers. It cannot be combined with `--burn-hotkey`,
`--reconcile-only`, `--release-hold`, `--weight-offer-path`, or an
object-store provider. Emissions policy flags and `--intake-db` are required.

```bash
python -m cacheon.cli set-weights \
  --burn-to-subnet-owner \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --half-life-blocks <blocks> --discovery-lifetime-blocks <blocks> \
  --discovery-pool-ppm <ppm> --refresh-blocks <blocks> \
  --wallet default --hotkey validator \
  [--wallet-path <wallets-root>] \
  --dry-run
```

```bash
python -m cacheon.cli set-weights \
  --burn-to-subnet-owner \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --half-life-blocks <blocks> --discovery-lifetime-blocks <blocks> \
  --discovery-pool-ppm <ppm> --refresh-blocks <blocks> \
  --wallet default --hotkey validator \
  [--wallet-path <wallets-root>] \
  --watch --interval 60
```

Only a live signer pass whose reconciliation returns `pending` or `confirmed`
writes the exact publishable projection to `<intake-db>.current_weights.json`
(or `--weight-offer-path`) and, when configured, to a swappable object store
(`--object-store-provider hippius|s3|minio|local`). Dry-run, reconcile-only,
and held passes do not advance either shared offer. The object-store write is
synchronous while the intake database's exclusive controller lock is held, so
a later controller process cannot be overtaken by an older background upload.
Prefer the eval/serve/follow split below when eval must not hold a chain-signing
weight path: `push-weight-offer` → `serve-weights` → `follow-weights`.

### `mint-push-credentials` / `mint-weight-gateway` / `push-weight-offer` / `serve-weights` / `follow-weights`

Provision a **dedicated** gateway hotkey for HTTP response signatures. Do not
reuse a follower / `set_weights` hotkey:

```bash
python -m cacheon.cli mint-weight-gateway \
  --wallet-path /var/lib/cacheon/wallets \
  --wallet gateway \
  --hotkey authority \
  --push-credentials /secret/push-credentials.json
```

That writes mode-0600 secrets (hotkey mnemonic only) + `AUTHORITY.json`, prints
`authority_ss58`, and does not print mnemonics. Existing push-credential files
are refused unless `--force` (which retires active secrets and appends).
The `cacheon.weight-gateway-secrets.v1` and
`cacheon.weight-gateway-authority.v1` schema labels are write-only informational
output. Pre-rename `optima.*` mnemonic and `AUTHORITY.json` files remain valid
operator records; Cacheon does not consume either form as input, so no record
rewrite or dual reader is required.
Followers pin that ss58 with `--expected-authority` (or omit the flag to
auto-pin the on-chain subnet-owner hotkey).

```bash
python -m cacheon.cli mint-push-credentials --path /secret/push-credentials.json

python -m cacheon.cli serve-weights \
  --object-store-provider hippius \
  --object-store-bucket cacheon-weights \
  --push-credentials /secret/push-credentials.json \
  --network <network> --netuid <netuid> \
  --wallet gateway --hotkey authority \
  --wallet-path /var/lib/cacheon/wallets \
  --host 0.0.0.0 --port 8080

python -m cacheon.cli push-weight-offer \
  --intake-db chain_intake/intake.sqlite3 \
  --network <network> --netuid <netuid> \
  --url http://weights-gateway:8080 \
  --push-credentials /secret/push-credentials.json \
  --attribution-hotkey <placeholder-ss58> \
  --half-life-blocks <blocks> \
  --discovery-lifetime-blocks <blocks> \
  --discovery-pool-ppm <ppm>

python -m cacheon.cli follow-weights \
  --url http://weights-gateway:8080 \
  --network <network> --netuid <netuid> \
  --wallet default --hotkey follower \
  --refresh-blocks <blocks> \
  --expected-authority <weights-gateway-hotkey> \
  --watch
```

`push-weight-offer` is the eval path: it builds the legacy V1 offer and
HTTP-PUTs it. It never opens a weight-signing wallet or calls `set_weights`. Credentials resolve from
`--push-credentials`, else `CACHEON_WEIGHT_PUSH_CREDENTIALS` (JSON path), else
`CACHEON_WEIGHT_PUSH_KEY` (+ optional `CACHEON_WEIGHT_PUSH_CREDENTIAL_ID`).
The corresponding `CACHEON_WEIGHT_PUSH_*` names remain last-precedence transition
aliases. Cacheon variables win when both forms are present.
`serve-weights` exposes `GET /v1/current-weights` (permit + hotkey signature)
and optional `PUT /v1/current-weights` (same credential resolution). A
credentialed PUT stores an HMAC-authenticated envelope, and a push-enabled
server verifies that envelope before signing any GET response. The push client
accepts only a fresh HMAC-authenticated acknowledgement binding its request
timestamp, credential, offer, and projection digests; an HTTP intermediary
cannot manufacture success without the push secret. Server-side
storage/transport failures remain retryable.

Cacheon verifies both the complete legacy `X-Cacheon-*`/`cacheon.*` transport
dialect and the distinct `X-Cacheon-*`/`cacheon.*` dialect. Headers, schemas,
HMAC domains, offer bytes, and acknowledgements must all select the same
dialect; mixed forms fail closed. Existing authenticated objects are reopened
without rewriting their bytes.

`follow-weights` rebinds the offer to the follower hotkey and publishes through
`reconcile_weight_publication` / commit-reveal. A fresh follower accepts an
older projection only while its age is at most `--refresh-blocks` and every
authority-relevant UID is unchanged. It retains publication chronology in a
signer-only journal, so a V2 follower does not need the evaluator's settlement
or composition database. Use one `--journal-db` per signer. Debt-lane offers
carry the full `DebtWeightPublicationBinding` so follower `weights_ppm` match
the economic projection.

`set-weights` and `follow-weights` accept a repeatable `--fallback-endpoint`
(`wss://` URL) that the SDK client fails over to when `--network` is
unavailable, and that also serves as an archive endpoint when the primary node
has discarded a requested historical block; `--watch` additionally turns on
the client's retry-forever reconnect. Every submission stamps the version key
the chain will accept: `set_weights` reads the subnet's `WeightsVersionKey`
hyperparameter, takes the maximum of it and the pinned floor, refuses to sign
when the hyperparameter is unreadable (commit-reveal drops a low key silently),
and prints one `SET-WEIGHTS-PAYLOAD` line with the exact UID/weight vector to
stdout.

Provider swap is config-only via `--object-store-provider` /
`CACHEON_OBJECT_STORE_*`; an environment-only
`CACHEON_OBJECT_STORE_PROVIDER` is sufficient, while explicit flags take
precedence over environment values. Corresponding `CACHEON_OBJECT_STORE_*`
variables are accepted only as last-precedence transition aliases. The optional
S3-compatible dependency is
`pip install -e ".[object-store]"` (boto3, Apache-2.0). See
[Settlement and weights](../validator-guide/settlement-and-weights.md#shared-current-weights-endpoint).

## Environment checks

```bash
python -m cacheon.cli compat
python -m cacheon.cli chain-compat
```

`compat` checks the exact pinned SGLang version plus registered imports and signatures. A
version mismatch is a failing result. `chain-compat` checks the Bittensor SDK API used by
Cacheon without connecting to a network.

## Release commands

### Provision model bytes

```bash
python -m cacheon.cli model-provision \
  /srv/models/model /srv/cacheon/model-publication \
  --expected-content-digest <sha256> --workers <n>
```

The result is an immutable content-addressed model tree and receipt.

### Verify a release

```bash
python -m cacheon.cli release-verify /srv/cacheon/releases/<digest> \
  --expected-public-key <ed25519-public-key> \
  --descriptor-digest <expected-digest>
```

The expected public key is an external trust input. A key discovered only inside the
release cannot authenticate its signer.

### Materialize a container context

```bash
python -m cacheon.cli release-context \
  /srv/cacheon/releases/<digest> ./context \
  --expected-public-key <ed25519-public-key> \
  --descriptor-digest <expected-digest>
```

The command reopens the complete signed publication before writing a deterministic OCI
context. Release construction and signing remain programmatic APIs; there is no public
`release-create` command.

## Exit behavior

Commands use a non-zero status for parser errors, explicit refusals, and failed local
checks, but exit `0` is only command completion. It is not always business-state success:
for example, `set-weights` can complete with a durable `pending` publication that still
requires later chain observation. Automation must inspect the typed status and retain the
receipts or durable records emitted by the authoritative subsystem. Console prose and
process status alone are never settlement or publication evidence.

Source: [`cacheon/cli.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/cli.py).
