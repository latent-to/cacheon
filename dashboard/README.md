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
| Submissions | All reservations: status, hotkey, submit time/block, fee tx, screen state, decisions, full detail drawer (screen/qual attempts, leases, settlement, plain-English worker forensics, downloadable logs) |
| Payments | Eval-cost payments (0.5 τ minimum): tx ref block-extrinsic with tao.app link, paying **coldkey** (resolved from chain), applied/consumed status, submission outcome; operator credits |
| Winners | Retained PASS settlement candidates: credited gain, measured candidate and baseline tok/s, prefill gain, served weight share, settlement status, and current on-chain emission |
| Miners | Per-hotkey leaderboard sorted by served weight share: submissions, crowns, qualified/failed, fees paid, registration + emission |
| Timeline | Settlement events (CROWN/ADOPTION/HOLD/…), the served weight offer's vector, and this validator's follower journal (intent/pending/held/confirmed) |
| System | DB/chain/process/heartbeat health, intake lag |

## Design notes

- **Never writes the intake DB.** Opens it `mode=ro` (falls back to
  `immutable=1`). Safe to run alongside intake/supervisor.
- Chain enrichment (block timestamps, payment coldkey signers, metagraph
  emissions) runs on a background thread against
  `wss://archive.sub.latent.to` and caches results in
  `dashboard/state/enrichment.sqlite3`. If the chain is unreachable the API
  still serves everything from the DB; times fall back to block-number
  estimates (dotted underline in the UI).
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
| `CACHEON_DASH_OFFER` | `/var/lib/cacheon/current_weights.json` (the file the weight-offer service serves) |
| `CACHEON_DASH_FOLLOW_JOURNAL` | unset; the follower journal SQLite named by the follow-weights lane's `--journal-db`. When unset the Timeline says so instead of showing a stale journal. |

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

The submission detail renders the signed evaluation records in full. Each
screen attempt carries `stages` — every graded check from the signed receipt
(stage, grade, reason, elapsed time), so a `screen_rejected` names the exact
failing check and its measured margin. Each qualification attempt carries
`speed` — the lane rates from the retained stage-exit artifact (per-role
tokens/second, timed windows, window scatter, conditioning ratio, and the
C/B speedup); `speed` is null when no local evidence store retains that
attempt's artifact. It also reuses the validator's `worker_log` explanation.
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
- **Why this result:** the shared grader explains the retained policy and reads,
  including gains against both baselines, required gain and measured baseline
  drift. A borderline speed result is distinguishable from invalid measurement.
- **Output throughput:** B/C/B′ output tok/s and total timed batch seconds.
  This includes prompt processing and generation; it is not isolated decode time.
- **Prefill:** v12 prompt-pass throughput in **prompts/s**, total timed batch
  seconds, observed candidate gain over the faster baseline read, and the
  retained prefill margin. Each prompt pass generates one output token, so the
  output-token count is a request count, not an input-token throughput measure.
  The comparison describes the measurements; it does not replace the verdict.
- **TTFT / TPOT by workload:** mean first-token latency, mean time per subsequent
  token, and output throughput, separated by input tokens, output tokens and
  request concurrency. Cells are recomputed from retained host timing windows.
  Evaluations without these timings explicitly show **Not measured**.

The Winners table includes conservative observed prefill gain when retained
passing attempts contain prompt passes. Missing historical measurements remain
absent. `session.measure_phase_latency` must have been enabled in the evaluation
to display TTFT/TPOT; enabling a dashboard panel does not enable measurement or
reconstruct timings for old runs.

Credited gains are labelled separately from measured throughput. A v12 prefill
credit is not an output-throughput ratio, so the dashboard does not divide
candidate tok/s by that credit to invent a stock tok/s estimate.

The API keeps ordinary lane `tokens_per_second` and adds `timed_seconds` and
`cells`. Prefill lanes set `tokens_per_second` to null and expose
`prompts_per_second` instead. `speed.prefill` contains the observed `speedup`
and retained `min_margin`. The reader also follows database-recorded and staged
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
reproduced PASS pairs earn under the retained-pair policy with time decay, and
a crown is not required. `/api/winners` and `/api/miners` carry `weight_share`
(the hotkey's fraction of the served vector, null when the offer file is
unavailable) plus an `offer` summary (projection digest, effective block,
crown count). `/api/weights` returns that vector with UIDs and on-chain
incentive beside the follower journal rows, so the lag between the served
offer and chain consensus is visible rather than mistaken for a wrong number.

`/api/winners` keeps settlement credit (`improvement_pct`) separate from measured
candidate and baseline throughput. `baseline_kind` identifies stock, incumbent,
or unknown; missing retained measurements stay null. The former `sglang_*` and
`cumulative_*_over_sglang` estimates are removed: weighted prefill credits and
different competition epochs cannot reconstruct a measured stock throughput.
