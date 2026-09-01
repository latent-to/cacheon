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

## What it shows

| Tab | Content |
|-----|---------|
| Overview | Status counts, submissions/day sparkline, failure reasons, currently running eval |
| Queue | Pending submissions in queue order with wait times, running evals with wall clock + lease countdown, GPU spool requests, supervisor/heartbeat |
| Submissions | All reservations: status, hotkey, submit time/block, fee tx, screen state, decisions, full detail drawer (screen/qual attempts, leases, settlement, plain-English worker forensics, downloadable logs) |
| Payments | Eval-cost payments (1 τ each): tx ref block-extrinsic with tao.app link, paying **coldkey** (resolved from chain), applied/consumed status, submission outcome; operator credits |
| Winners | Every retained two-PASS improvement: conservative speedup over the incumbent, compounded stack speedup over retained SGLang stock with a serving tok/s estimate, settlement status, and current miner emission |
| Miners | Per-hotkey leaderboard: submissions, crowns, qualified/failed, fees paid, registration + emission |
| Timeline | Settlement events (CROWN/ADOPTION/HOLD/…) and weight publications journal |
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

## API

`/api/overview`, `/api/queue`, `/api/submissions` (filters: `status`,
`hotkey`, `q`, `active`, `limit`, `offset`, `order`),
`/api/submissions/{id}`, `/api/payments`, `/api/winners`, `/api/miners`,
`/api/events`, `/api/weights`, `/api/hotkey/{ss58}`, `/api/health`.

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

`/api/winners` reports each contribution twice over: `improvement_pct` is the
conservative gain over the incumbent it displaced, while
`cumulative_speedup_over_sglang` compounds the settled gains of a target's
CROWN lineage in settlement-event order, so it reads as the stack's position
against retained SGLang stock at the moment that contribution was crowned.
`tokens_per_second` is the slower independently passing candidate lane from the
qualification artifacts, and `sglang_tokens_per_second` divides it by the
cumulative speedup. The SGLang-relative fields are null for a retained PASS
that never entered the stack, and the tok/s pair is null when no local evidence
store retains the attempt's artifact.
