# Cacheon submissions dashboard

Read-only API + web dashboard over the live netuid-14 intake database
(`/data/mainnet14-cacheon-h3-m4i-pre-crown/state/intake.sqlite3`).

## Run

```bash
/root/cacheon-ops/bin/cacheon-dashboard      # pm2 process "cacheon-dashboard"
# or in the foreground:
/root/cacheon-ops/dashboard/run.sh
```

Logs: `pm2 logs cacheon-dashboard` (files under `~/.pm2/logs/`).

Then open http://127.0.0.1:8788/ (interactive API docs at `/docs`).

Uses the prod conda python (`/root/miniconda3/envs/prod/bin/python`), which
already has fastapi, uvicorn, bittensor 10.3.2, and async-substrate-interface.
From a local repository checkout, install `python -m pip install -e ".[dashboard,dev]"`
to run the dashboard and its API tests.

## What it shows

| Tab | Content |
|-----|---------|
| Overview | Status counts, submissions/day sparkline, failure reasons, currently running eval |
| Queue | Pending submissions in queue order with wait times, running evals with wall clock + lease countdown, GPU spool requests, supervisor/heartbeat |
| Submissions | All reservations: status, hotkey, submit time/block, fee tx, decisions, full detail drawer (qualification attempts, leases, settlement, plain-English worker forensics, downloadable logs) |
| Payments | Eval-cost payments (0.5 τ minimum): tx ref block-extrinsic with tao.app link, paying **coldkey** (resolved from chain), applied/consumed status, submission outcome; operator credits |
| Winners | Retained PASS settlement candidates: credited gain, the replay result (decode tok/s per user and first-token time, baseline → candidate), baseline kind, served weight share, settlement status, and current on-chain emission |
| Miners | Per-hotkey leaderboard sorted by served weight share: submissions, crowns, qualified/failed, fees paid, registration + emission |
| Timeline | Settlement events (CROWN/ADOPTION/HOLD/…), the served weight offer's vector, and this validator's follower journal (intent/pending/held/confirmed) |
| System | DB/chain/process/heartbeat health, intake lag |

Queue, submission-list and detail fee labels include the credit spent on that
specific submission. Available credits and credits spent on another submission
do not mark it as covered. On-chain payments retain their transaction links;
credit-funded details show the covered amount without inventing a transaction.

The Winners tab places completed PASSes still awaiting earlier queue entries in
a separate, greyed-out **Potential winners** table above retained rewards. Their
measured gains remain visible, but payouts and final winner selection are pending.
Once queue eligibility is finalized, qualifying winners appear normally on the
next refresh; PASSes below the winning threshold do not enter retained rewards.
Resolution uses the shared reward comparison from the weight producer: scoring
gain is relative to the best earlier PASS in queue order within the same arena
and measured baseline that itself earned.
Later submissions never change an earlier submission's reference. Retained
baseline measurements and PASS evidence are unchanged.
Already finalized rewards retain their existing eligibility if earlier work is reopened.
`/api/winners` returns these pending results in `waiting_items` / `waiting_total`,
separate from finalized `items` / `pass_total`. Waiting rows have
`waiting_for_queue: true`, no weight share, and no finalized queue comparison.

## Design notes

- **Never writes the intake DB.** Opens it `mode=ro` with WAL visibility. Safe to run alongside intake/supervisor.
- Chain enrichment (block timestamps, payment coldkey signers, metagraph
  emissions) runs on a background thread against
  `wss://archive.sub.latent.to` and caches results in
  `dashboard/state/enrichment.sqlite3`. If the chain is unreachable the API
  still serves everything from the DB; times fall back to block-number
  estimates (dotted underline in the UI).
  A failed enrichment connection is closed before reconnecting so its client
  caches and websocket resources can be released.
  Before loading the chain client, enrichment defaults its runtime metadata cache
  to two versions. Historical lookups reload evicted metadata; the SQLite results
  remain cached. This trades additional archive requests for lower memory use.
- Times are sent as unix seconds; the browser renders them in your locale.
- Explorer links: tao.app `/block/{n}`, `/blocks/{n}/extrinsics/{i}`,
  `/portfolio/{ss58}` (plus taostats fallback for extrinsics).

## Config (env)

| Var | Default |
|-----|---------|
| `CACHEON_DASH_HOST` | `127.0.0.1` |
| `CACHEON_DASH_PORT` | `8788` |
| `CACHEON_DASH_DB` | mission intake.sqlite3 |
| `CACHEON_DASH_SPOOL` | `/root/cacheon-ops/remote-worker/spool` |
| `CACHEON_DASH_NETWORK` | `wss://archive.sub.latent.to` |
| `CACHEON_DASH_ENRICH` | `1` (set `0` to disable chain lookups) |
| `SUBSTRATE_RUNTIME_CACHE_SIZE` | `2` for enrichment (explicit values override; set before starting Python because the chain client reads it at import time) |
| `MALLOC_ARENA_MAX` | `2` in `run.sh`; limits glibc allocator arenas for concurrent reads, trading allocator concurrency for lower retained memory. Direct Python launches must set it before starting Python. |
| `CACHEON_DASH_OFFER` | `/var/lib/cacheon/current_weights.json` (the file the weight-offer service serves) |
| `CACHEON_DASH_FOLLOW_JOURNAL` | unset; the follower journal SQLite named by the follow-weights lane's `--journal-db`. When unset the Timeline says so instead of showing a stale journal. |
| `CACHEON_DASH_EXCLUSIONS` | unset; the `exclusions.json` the running weight producer applies. A submission named in its `claims`, or a hotkey in its `records` that the served offer pays nothing, shows the operator's recorded decision. When unset no decision is shown. |

## API

`/api/overview`, `/api/queue`, `/api/submissions` (filters: `status`,
`hotkey`, `q`, `active`, `limit`, `offset`, `order`),
`/api/submissions/{id}`, `/api/payments`, `/api/winners`, `/api/miners`,
`/api/events`, `/api/weights`, `/api/hotkey/{ss58}`, `/api/health`.

Submission list and detail responses expose `evaluation_recovery` when an
operator has linked a payment rejection to a corrected submission. The existing
intake `metadata` key `evaluation_recoveries` stores a JSON object mapping original
reservation IDs to evaluated reservation IDs. The reader requires the same
hotkey and, when already resolved before rejection, target. It reads current status and active lease stage from the
linked reservation. The UI links to that evaluation beside the original payment
error; it preserves both submissions' identities, errors and evaluation history.

Bundle downloads and public raw-log downloads are withheld until **eight hours after
the terminal evaluation result**, not eight hours after submission. Queued,
running and held submissions remain withheld. The detail API returns an empty
`url` and `bundle_visibility` with `available`, `release_at` (Unix seconds or
null), and `result_block`. It uses the latest completed evaluation/retained PASS
block and its exact cached chain timestamp; missing or estimated result times
do not release source. Results and performance measurements remain visible.
Raw logs and source-bearing exception messages follow the same delay because
compiler output can contain source. Direct raw-log requests return HTTP 403
until release; validator intake and private diagnostic access are unchanged.

Updated miner clients encrypt archives before uploading. The revealed chain URL
points to ciphertext; `/api/bundle-encryption-key` supplies only the recipient
public key. Set `CACHEON_BUNDLE_DECRYPTION_KEY` for both intake and the dashboard
to the validator-owned mode-0600 file containing the 32-byte private key as hex.
Back up this key privately and retain it while submissions encrypted to it remain
pending. The dashboard reads checked bundles from `CACHEON_DASH_MISSION/private`.
After release, detail `url` names `/api/submissions/{id}/bundle.tar.gz`, which
rechecks the retained bundle hash before download. Existing plaintext uploads
remain readable by intake and cannot be made confidential retroactively.

The evaluation baseline card links to the submission recorded by the evaluated
artifact's lineage transition, preserving the selected arena. A base-engine
baseline has no submission; missing historical links are labelled explicitly.
The card and the Submissions list also name the best retained PASS on that same
arena and baseline stack: the result a later PASS must beat by the reward margin
to earn. The crown lineage decides adoption, not pay, and is not shown.
The previous-best scoring comparison also links to that submission's details.

A replay-graded submission leads with its `result` in the Submissions list, the
Winners table and the detail: the verdict with the measured gain, the gain that
was needed and the passes run, then pooled decode tok/s and mean first-token
time, baseline → candidate. `reward` is its share of the served offer and the
ordered comparison that decided it (the detail carries the same fields under
`settlement`). Both are null where nothing is retained to report, which
includes every batch-cell attempt: its raw lane ratio is not the credited gain.

The submission detail renders the signed evaluation records in full. Each
qualification attempt carries
`speed` — the measurements from the retained stage-exit artifact: the paired
replay windows and the retained grade; `speed` is null when no local evidence
store retains that attempt's artifact, and for an attempt graded by a retired
batch-cell policy, whose stored lane rates are not rendered. It also reuses the validator's `worker_log` explanation.
Each request with retained forensics links to
`/api/submissions/{reservation_id}/forensics/{request_id}.log`. The response is
built read-only from the hash-verified result and contains the exact
request-scoped adapter diagnostics and retained OCI diagnostic streams. The OCI
worker reserves stdout for framed protocol and redirects ordinary Python/native
stdout into the retained stderr stream, so miner prints and crash diagnostics are
both present. Section headers state byte counts, hashes, and whether the 16 MiB
stream bound truncated the output.

Each qualification attempt also has a **Performance** section:

- Completed `NO_DECISION` attempts held by the remote dispatcher remain visible
  from their retained result, even without a qualification-disposition row.
  Their measurements and original decision remain separate from the reservation's
  current status, including a later operator rejection. Importing the same
  attempt does not duplicate it in the history.
- **Agent replay:** the measured and required gain, then one row per pass. A
  pass pairs the baseline and the candidate that ran at the same moment: gain,
  time for the same work, decode tok/s pooled over the arm, mean and p95
  first-token time, and service attainment.
- A retained batch-cell attempt (speed policies 8–15) shows that its
  measurements are unavailable; its stored lane rates are not rendered.

The reader follows database-recorded and staged
evidence roots and selects the submission's target from historical multi-target
reports, so changing worker generations does not hide retained measurements.
An unreadable retained response or grading artifact is shown as an evidence
error. Execution summaries join PID identities to completed rank records, so
repeated snapshots do not become extra GPUs or lose the recorded graph capture.
Worker connection status reflects the CPU relay's last successful pod check.
It does not treat the relay's default adapter flag as a failed GPU process;
the pod starts an adapter when work arrives and retires it after qualification.

Metagraph emission is denominated in the subnet's own alpha token, so
`/api/winners` and `/api/miners` report `emission_alpha_per_day` beside an
`emission_symbol` read from the netuid's on-chain symbol (`ㄷ` for netuid 14).
The installed bittensor unit table can disagree with the chain, so the UI
renders the served symbol and never a local one. Only the eval-cost fee is
in TAO.

Reward shares come from the served weight offer, not from settlement status:
a complete audited PASS earns only when its credited speedup clears the best earlier
PASS by the sealed minimum margin. Pre-policy runtime generations retain their
existing eligibility; a crown is not required. The weight producer and Winners
view share this arrival-ordered filter.
A pass that settlement held as `stale_incumbent` (timed against an earlier
baseline than the champion) or `lost_potential` (did not exceed the crown
record) is shown as `passed` in `settlement_status`: it earns like any other
pass and only the crown was withheld. The detail's **Reward and crown** block
states the two separately. A PASS named in the `claims` of
`CACHEON_DASH_EXCLUSIONS` reads `excluded` and is never shown as the result to beat.
Historical retained pairs keep their identities. `/api/winners` carries each
reservation's `weight_share` from the served offer's allocation evidence
(`submission_weights_ppm`), including the allocator's integer rounding. A
submission absent from an available breakdown has zero share; pending rows and
offers without a readable breakdown have null shares. Older offers without this
field show `attribution_unavailable` until the producer publishes a new breakdown.
`/api/miners` continues to carry the hotkey's total fraction of the served vector,
null when the offer file is unavailable. Both include an `offer` summary (projection digest, effective block,
standing-claim count, named `crown_count` on the wire). `/api/weights` returns that vector with UIDs and on-chain
incentive beside the follower journal rows, so the lag between the served
offer and chain consensus is visible rather than mistaken for a wrong number.

For producers that redistribute weights after static allocation, set
`CACHEON_DASH_SUBMISSION_SHARES` to their final submission-breakdown JSON file.
It must contain `projection_digest` matching the served offer and
`submission_weights_ppm` keyed by reservation ID after redistribution. This
explicitly selects that report instead of pre-redistribution allocation evidence;
a missing or mismatched snapshot shows unavailable attribution. Miners always
uses the served hotkey vector.

`/api/winners` keeps settlement credit (`improvement_pct`) separate from measured
candidate and baseline throughput. `baseline_kind` identifies stock, incumbent,
or unknown; missing retained measurements stay null.

## Multiple arenas

Submission links include server-rendered Open Graph and Twitter metadata, with a
1200×630 PNG at `/api/submissions/<id>/preview.png?arena=<key>`. Crawlers do not
need JavaScript. Cards show the model, target, evaluation status, throughput gain,
submission and stock SGLang tok/s and TTFT, plus the SGLang version tag and
seven-character Git commit. The dashboard extra includes Pillow; the bundled Ubuntu font keeps
rendering independent of host fonts.

An explicitly empty incumbent stack supplies a paired stock comparison.
The card uses the lower accepted qualification when a historical
PASS pair exists, and reads performance from that same attempt. Batch throughput
uses the fastest B/B-prime stock observation and the conservative candidate rate;
replay cards use mean decode throughput and mean TTFT. The headline is
`(submission tok/s / stock tok/s - 1) × 100`, not the qualification score.
Incumbent comparisons never become stock gains. Without stock measurements,
the card shows centered submission tok/s and TTFT, with no comparison section.
Missing measurements and commits are labelled unavailable. Pending and failed submissions
keep their actual status. HTML is not cached; PNGs are cached for 60 seconds so
new results can replace pending cards (social platforms may cache independently).

For replay submissions evaluated against an optimized incumbent, set
`CACHEON_DASH_STOCK_REFERENCES` to an absolute JSON file containing a list of
`{"summary": "/absolute/stock-run/summary.json", "inputs": "/absolute/stock-run/A-inputs.json"}`
entries. These are the saved stock-run summary and original lane inputs, not
submission-specific numbers. The inputs must contain an empty `stock_manifest`
(`stock_manifest._entries: []` in the saved dataclass inputs) and its
`workload_digest`; the summary supplies `kind: "stock_sglang_reference"`,
`workload_digest`, `load`, `mean_decode_tps`, `mean_ttft_s`, and `completed_unix`.
Keep both files with the retained raw measurements. The stock manifest's arena,
runtime, and base-engine digests must match the candidate; both workload digests
and concurrency must match its retained replay. The arena binds model and topology;
the workload binds engine configuration and replay inputs. One reference can then
serve future submissions with the same identities, across targets. Configure only
one reference per identity; ambiguous or unreadable configured references raise an
error rather than selecting a favorable run. Different identities remain unpaired.

Cards with a separate reference use the saved mean decode throughput and mean
TTFT, and visibly label the stock measurement date and **separate runs** in both
the PNG and crawler metadata. The headline still compares the displayed decode
tok/s, not replay turns/s or settlement credit. A paired stock measurement takes
precedence. This affects social presentation only; it does not regrade submissions
or change rewards.

Build metadata is fetched automatically from the commissioned image's
`ai.sglang.build.commit` and `ai.sglang.image.tag` Docker labels. The submission's
runtime must match the configured registration; its READY receipt is read from
the source's `stage` directory or one of its immediate subdirectories. The READY
digest must match the registration before its immutable `worker_image` is inspected
over SSH using that registration's host, port, user and known-hosts file. The
dashboard process needs the existing operator SSH authentication. No container is
started and no candidate code is executed. Lookups time out after five seconds
and are cached for five minutes; failures are logged. There is no manually
maintained commit/version map. Missing metadata or a historical runtime that no
longer matches the registration displays `SGLang commit unavailable`.

Set `CACHEON_DASH_SOURCES` to an absolute JSON config path. It has `default`
(the selected source key) and a `sources` array. Every source explicitly names:

```json
{
  "key": "secondary",
  "slug": "secondary-1.0",
  "label": "Secondary arena",
  "model": "Commissioned model name",
  "paths": {
    "db": "/arena/secondary/state/intake.sqlite3",
    "mission": "/arena/secondary",
    "audit": "/arena/secondary/state/chain-audit.jsonl",
    "spool": "/arena/secondary/remote-worker/spool",
    "heartbeat": "/arena/secondary/remote-worker/spool/state/heartbeat.json",
    "registration": "/arena/secondary/registration.json",
    "logs": "/arena/secondary/logs",
    "evidence_state": "/arena/secondary/remote-worker/state",
    "stage": "/arena/secondary/stage"
  },
  "cache": "/arena/dashboard-cache/secondary.sqlite3",
  "evidence_roots": ["/arena/secondary/qualification-evidence"],
  "cutoff_reservation": "",
  "weights_included": false,
  "checkpoint": null,
  "processes": {
    "intake": ["chain-validate", "--intake-only", "/arena/secondary/state/intake.sqlite3"],
    "supervisor": ["cacheon.chain.standing_cpu_supervisor", "/arena/secondary/supervisor.json"],
    "relay": ["cpu-serve", "/arena/secondary/remote-worker/spool"]
  }
}
```

Paths are absolute; private bundles remain under `mission/private`. Each source
needs distinct intake, enrichment cache and private roots. Caches must never
point to an intake DB. Configure separate spool, log and evidence roots too.
The dashboard reads current WAL contents and reports an unreadable DB; it does
not substitute an immutable snapshot. Config changes take effect after restart.

The page's arena selector scopes every data tab, detail link, and delayed
bundle/log download. Share a competition at `/<slug>#<tab>`, where `slug` is
the versioned path name each source declares, for example `/glm-5.3#winners`
or `/dsv41-flash#winners`. Submission links use `/glm-5.3?submission=<id>#winners`.
Only the declared slugs are accepted as paths; a bare key such as `/glm`
returns 404. Query parameters accept both the slug and the `?arena=<key>` key.
The root page selects the configured default; `?arena=<key>` page
links also work and become path links in the address bar without losing the
tab or submission. A path takes precedence over an `arena` query parameter.
Unknown arena paths return 404. The reverse proxy must forward these paths to
the dashboard app, as it does `/`; domain configuration is unchanged.
API clients continue to pass `?arena=<key>` (or the versioned name); omission
selects `default`. `/api/arenas` exposes each source's URL `slug` alongside its key.
Unknown keys return 404 and unavailable selected databases 503. Each source uses
its registered arena namespace; empty-namespace history remains visible through
the database's legacy arena alias. An unpublished legacy observation is hidden
when another configured database published that reservation for a different arena.
Its original rejection remains in the database; a local publication retains its
own history. Legacy ownership checks read peer databases in read-only mode and
return 503 if a required peer cannot be read. Replicated published bundles are
displayed only in their selected arena. Queries and downloads share
this scope without changing the intake database. `/api/arenas` reports each
source's health independently. `/api/arena-events` combines events by retained
block, using source and local sequence only for ties, and names unavailable
sources. The Timeline offers an all-arena toggle.

Process health matches command arguments plus a configured source path, rather
than global module substrings. The configured `heartbeat` path is the CPU relay's
local pulse. Successful SSH polls retain the worker's verified heartbeat unchanged
in adjacent `worker-heartbeat.json`; failed polls never advance its timestamp.
The health and queue APIs expose these separately as `relay_heartbeat` and
`gpu_heartbeat`, including `fresh`. A missing worker observation is unknown, one
older than 120 seconds is stale, and a registration-binding mismatch is explicit.
Stale or mismatched observations cannot report a current adapter or active request.
Older relay deployments without the worker file show unknown worker health.
The Queue, System and arena badges use worker observations for worker status;
CPU relay freshness and active evaluation leases do not establish worker liveness.
Public health omits operator paths and SSH errors.

`model` supplies the source's display label; leave it empty to retain historical
legacy labels. Optional `checkpoint` maps each retained engine digest to its
`repo`, `revision`, `content_digest`, and `url`. Only an exact engine match is shown,
so older checkpoints can remain visible alongside the current one.
Configured sources never search another source's checkpoint cache or legacy
operations roots. Labels and commercial presentation do not change service,
settlement or weight identities.

`/api/weights` and its journal remain global, ignoring the arena selector.
`weights_included=false` displays “weights off / not yet in served vector” and
null source-row shares, including when a miner earns elsewhere. Set it true only
after the source is included in the real producer. Displayed miner shares are
still global hotkey shares, not an inferred per-arena split. Future breakdowns
must consume retained producer allocation evidence. The status pills separately
show **target weights**, not actual reward shares. Set the optional top-level
`weight_producer_config` in the dashboard sources JSON to the absolute path of
the active weight-offer service configuration. The dashboard follows its
`weights_stage_config` and `arena_allocation_path` on every refresh, selecting
the latest settings row effective at the observed finalized intake block.
Source keys must match the allocation's keys. Overcommitted targets normalize
to 100% using the existing allocator; totals below 100% stay unchanged. Future
rows do not display early, and earnings or historical submission terms do not
alter the displayed target. A single-source producer without an allocation
schedule shows 100% for its configured intake database and 0% for other sources.
Missing configuration or unreadable settings display “target unavailable”.
`/api/arenas` exposes the result as `target_weight_ppm`; this does not enable
weights for an arena marked weights-off. For live process tracking, set
`weight_producer_pidfile` to the active service manager's absolute PID-file path.
On every refresh the dashboard resolves that PID's current `--config` argument;
this takes precedence over the fixed `weight_producer_config` path and follows
producer restarts or configuration relocations without a dashboard restart.
A stopped producer or unreadable PID/command displays “target unavailable”;
it never falls back to a stale fixed configuration. If the service is deleted
and recreated with a different PID-file path, update the dashboard setting.
The header shows its last successful target check and the 15-second refresh
interval. The dashboard never changes producer settings. Sponsorship revenue,
evaluation fees/operator credits, and miner alpha rewards remain separate.
