# Mainnet FIFO — remaining TODO

Companion to [`CONTEXT_MAINNET_FIFO.md`](CONTEXT_MAINNET_FIFO.md) and
[`HANDOVER_ops_mainnet_fifo_screen_hardening.md`](HANDOVER_ops_mainnet_fifo_screen_hardening.md).

§10 proof is done (epoch `4d3df000…`). Screening currently **stopped** for FE eval prep.
Not green.

## Done

- [x] Watchdog / intake perms / swap error / ABI image / launcher image rotate / authority rebind
- [x] §10 two durable screens + ledger outcome (`adapter_start_count=1`)
- [x] Branch/PR: `ops/mainnet-fifo-screen-hardening` / #75 (`a6fa7ac`)
- [x] Authority-rebind runbook + live path validation
- [x] Pause screening; rebind repro to `cfc0c7a…`; submit+publish FE v9 `747405b4…` (`dd382ab6…`)

## Next

1. **FE m4n one-shot** — finish pod mission rsync, new program roots, launch qual
   (details in handover). Do not reopen `a9a258` / `314cf17c`.
2. Optional: commit handover + this TODO refresh onto the branch.
3. After FE program finishes (or abort): recommission **one** screen dispatcher on a
   new epoch if screening should resume.

## Do not

- Second dispatcher on same epoch · chain-walk / raw SQL lease recovery · call subnet
  green after screen-only · treat screen promotion as crown · reuse spent m4l roots.
