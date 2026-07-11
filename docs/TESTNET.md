# Running Optima on the Bittensor testnet

How the chain layer (`optima/chain/`) runs against a real subnet, and the exact
loop validated on testnet netuid 307 (2026-07-08). See `docs/SUBNET_BLUEPRINT.md`
for the architecture this implements; this doc is the operator's runbook.

## What rides the chain

The chain is the trust anchor ONLY (commitments + weights). A miner's whole
on-chain footprint is one commitment carrying
`{"v":1,"h":"<sha256 content hash>","u":"<fetch url>"}` posted via the chain's
NATIVE timelock commit-reveal (`set_reveal_commitment`):

- the payload is drand-encrypted until the reveal round — nobody (validators
  included) can read the bundle URL before the reveal, so there is no
  pre-evaluation copy window;
- the reveal block is the consensus anti-copy priority timestamp — the validator
  replays reveals into the Ledger in chain order, so "earliest committer wins"
  is decided by the chain, not by any off-chain clock;
- chain-side caps: 1024 bytes per timelock payload, ~3100 payload-bytes per
  hotkey per epoch (each commit costs `max(100, bytes)` of that budget).

Everything else — fetching, hash verification, copy detection, the gate chain,
settlement — happens on the validator, recorded in the JSON ledger
(`commit_reveal.Ledger`). Weight policy is read from `Ledger.current_weights()`
— the ONE place emission policy lives (currently per-target king-of-the-hill;
singleton targets are slots, while registered atomic multi-slot targets receive
one indivisible title; NOT frozen winner-take-all).

## Prerequisites

- `bittensor` SDK (validated: 10.3.2; note it pins `bittensor-drand<2.0.0` — a
  drand 2.x wheel breaks the import). Run `optima chain-compat` after ANY SDK
  bump — it introspects the installed SDK for every method we call.
- A wallet (`btcli wallet create`). Key roles, verified against SDK source:
  - registration (`burned_register`): coldkey + hotkey
  - staking: coldkey
  - **commitments and set_weights: hotkey only** — the recurring loop never
    touches the coldkey. Keep the coldkey off the validator box.
- Testnet TAO (the faucet is Discord-manual these days).

## The loop, step by step

```bash
NET="wss://test.chain.opentensor.ai:443"   # pass the URL explicitly: the SDK's
                                            # 'test' alias is a DIFFERENT host
                                            # (test.finney.opentensor.ai)

# 0. one-time: join the subnet (validator hotkey, and each miner hotkey)
optima chain-register --netuid 307 --network "$NET" --wallet default --hotkey default

# 1. miner: package a bundle -> tar.gz + the content hash that will be committed
optima chain-package examples/miner_silu_torch --out hosted/bundle.tar.gz

# 2. miner: host the tarball on authenticated HTTPS, then commit hash+URL
#    on-chain (timelock). Production payloads reject file:// and plaintext HTTP.
optima chain-submit examples/miner_silu_torch --url https://example.com/bundle.tar.gz \
    --netuid 307 --network "$NET" --wallet default --hotkey miner
#    (--dry-run prints the exact payload without signing)

# 3. inspect the subnet: neurons, permits, revealed submissions
optima chain-status --netuid 307 --network "$NET" --wallet default --hotkey default

# 4. validator: the referee loop (single pass with --once; daemon without)
optima chain-validate --netuid 307 --network "$NET" --wallet default --hotkey default \
    --arena minimax-m3-b300-tp4-decode-v1 \
    --ledger chain_ledger.json --bundles-dir chain_bundles \
    --once --dry-run-weights
```

One `chain-validate` pass = read revealed commitments (chain order) → fetch each
new artifact (size-capped; hostile-archive-safe extraction: no symlinks /
hardlinks / path escapes) → **re-hash the extracted tree and refuse a mismatch
with the committed hash** → fingerprint + Ledger reveal (copies are demoted, not
evaluated) → independently resolve the fetched manifest's validator-owned
competition target → derive a one-shot prompt seed from post-commit block-hash
entropy → evaluate originals out-of-process → require the report's bundle hash,
arena fingerprint/bracket/regime, prompt seed/engine, validator image, sglang pin,
and target/mode/canonical members to match → record scores → settle per target
→ push weights (the SDK routes through the drand commit-reveal weight path
automatically when the subnet enables it). Every submission gets an EvalRecord,
so restarts skip known work and dead URLs are not refetched.

### Evaluators

- Default (no flag): **verify-mode plumbing** — runs `optima verify` on CPU. Its
  result is explicitly non-crownable even at margin zero. Never use this for real emissions.
- Production: `--oci-gpus 0,1,2,3` runs the registered validator-owned OCI
  controller and publishes retained host/device/qualification evidence.
- Development only: `--eval-cmd 'optima-eval.sh {bundle} {report} {arena} {prompt_seed}'` — the
  command must exit zero **and** atomically write the typed qualification report
  produced by `optima evaluate {bundle} --arena {arena} --prompt-seed
  {prompt_seed} --report {report}`. The report separately
  carries completion, quality, speed confidence, crownability, and the resolved
  target/mode/canonical member slots; the chain independently derives the same
  identity from the fetched manifest and requires an exact match. Missing,
  malformed, incomplete, stale, mismatched, or internally inconsistent reports fail closed.
  A command report is never crownable.

All arena workers for one validator/netuid must share the same ledger path,
retained-artifact root, and whole-pass lock. The ledger computes one global
equal-per-target vector across registered arenas; independent per-arena ledgers
cannot safely compose on-chain weights. The native reveal-history RPC exposes a
bounded ten-row view per hotkey. Optima paginates saturated windows backward with
historical `block=` queries and fails closed if the configured RPC cannot serve
that archive state or pagination does not make bounded progress.

Ledger compatibility is additive for historical singleton rows: an old
`slot = s` score loads as target `s`, mode `slot`, members `[s]`. Historical
multi-op rows are not safely inferable because the old ledger persisted only
`ops[0]`; re-evaluate and re-crown those bundles under their registered atomic
target rather than guessing membership during load. All pre-arena scores and
champions remain in a quarantined legacy namespace; they are never current in a
registered arena and must also be re-evaluated there.

## Validated on netuid 307 (2026-07-08)

- Registered validator (uid 3) + miner (uid 4) hotkeys; burn ≈ 0.0005 tTAO each.
- Committed `miner_silu_torch` (178-byte payload, 5-block timelock) from the
  miner hotkey; the reveal appeared at block 7509374.
- One historical validator plumbing pass picked it up through the now-test-only
  local-file transport, hash-verified,
  CPU-verified, crowned it, and produced `{miner_hotkey: 1.0}` targeting uid 4;
  a second pass did zero new work (idempotence).
- Subnet-307 caveats (someone else's unstarted subnet): its alpha pool rejects
  stake (`SubtokenDisabled`), and validator permits are the gate for actually
  landing `set_weights` — check `chain-status` for your permit before expecting
  weights to apply. Hyperparams there: tempo 360, commit-reveal weights ENABLED,
  weights_rate_limit 100 (skipped under commit-reveal).

## Validated on netuid 307, round 2 (2026-07-10) — real weights

Everything the 07-08 pass left dry-run, landed for real (the subnet owner
start-called 307 around 07-08, so `SubtokenEnabled` and permits now work there):

- Validator hotkey holds a validator permit (stake-weight via alpha stake; the
  permit is top-`max_validators` by stake-weight, recalculated per epoch).
- `chain-validate --once --margin 0` WITHOUT `--dry-run-weights`:
  `set_weights` SUBMITTED (the SDK auto-routes through drand commit-reveal on
  this subnet; the weights become visible in the metagraph after the reveal at
  the next epoch boundary — check `Subtensor.weights(netuid)`).
- Multi-miner emission split: a second miner hotkey committed a bundle for a
  second singleton target → per-target settle → weights
  `{miner: 2/3, miner2: 1/3}` pushed.
- Copy demotion through the chain: the same bundle committed later by another
  hotkey was demoted (`copies=1`, never evaluated), and the loop skipped the
  redundant weight push (weights unchanged, not stale).
- Broken bundle through the chain: failed the gate chain (`passed=False`,
  score 0), crown unchanged.
- Daemon mode + mid-epoch restart: kill and restart → next pass `new=0`
  (EvalRecords suppress replay), weights stable.

Operational gotchas (learned here):

- **Reveal history is settlement input, not a latest-value register.** The loop
  replays every row in chain order. Ten-row windows are paginated through
  historical block state, so multiple submissions between validator passes are
  preserved; use an archive-capable RPC or validation fails closed.
- **bittensor's import reconfigures global logging** — it sets pre-existing
  loggers to CRITICAL, which silenced daemon mode entirely (the ledger advanced
  while the log stayed empty). `chain-validate` now takes ownership of the
  `optima.chain` logger subtree after connecting. A silent validator is an
  unoperable validator; keep this in mind for any new entry point.
- **dTAO staking slippage is real.** 307's pool was alpha-drained (constant
  product: `alpha_in ~0.01`); 100 tTAO bought ~0.005 alpha — the measured
  numbers match `x·y=k` exactly. Check the pool before assuming stake buys
  stake-weight at par on mainnet.

## Threat-model notes

- The validator never imports miner code: evaluation is subprocess-only, same as
  `optima verify`. Bundle fetch treats archives as hostile (member-type and
  path checks, archive/extracted/member-count budgets).
- Chain keys stay on the control box; the GPU box only ever sees the bundle
  directory (SUBNET_BLUEPRINT §8). Wire `--eval-cmd` over SSH accordingly.
- A miner lying about the hash (committing X, hosting Y) is rejected at the
  re-hash step and recorded, so the lie is not retried every pass.
