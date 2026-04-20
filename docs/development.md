# Cacheon — Development & Testing Notes

Operational notes for the subnet owner. Not miner-facing.

## Networks

| Network            | NETUID  | Purpose                                   |
| ------------------ | ------- | ----------------------------------------- |
| `finney` (mainnet) | **14**  | Production — real emissions, real miners  |
| `test` (testnet)   | **460** | Validator development and dry-run testing |

Testnet subnet 460 was registered 2026-04-20 by the subnet owner. It is an empty sandbox with no miners or active commitments — used only to verify the validator loop against a live chain without touching mainnet.

## Dry-run validator smoke test (testnet)

Runs the full Part A loop (chain scan → commitment parsing → challenger selection → state persistence) without calling `set_weights` or triggering GPU eval:

```bash
pip install bittensor   # if not already installed

python scripts/remote_validator.py \
    --network test \
    --netuid 460 \
    --wallet-name default \
    --wallet-hotkey default \
    --dry-run \
    -v
```

Expected output on a clean run (no miners, no commitments):

```
Scan @ block XXXXX: 1 hotkey(s), 0 valid commitment(s)
Challenger selection: 0 new, 0 pre-rejected, 0 already known
No king yet — skipping set_weights ...
Tick OK @ block XXXXX in X.Xs (king_changed=False, new_evals=0)
```

State written to `state/state.json`. The loop then sleeps 360s and repeats.

## Simulating a challenger (testnet)

To exercise the challenger selection path, register a miner hotkey on netuid 460 and submit a commitment:

```bash
# 1. Register a second hotkey on testnet SN460
btcli subnet register --netuid 460 --network test --wallet.name default --wallet.hotkey miner

# 2. Submit a commitment pointing at any public HF repo
python - <<'EOF'
import bittensor as bt, json
wallet = bt.wallet(name="default", hotkey="miner")
subtensor = bt.subtensor(network="test")
data = json.dumps({"model": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "main"})
subtensor.set_reveal_commitment(wallet=wallet, netuid=460, data=data, blocks_until_reveal=1)
print("committed")
EOF
```

On the next dry-run tick the validator will classify that hotkey:block as a new challenger, call `eval_fn`, and hit `NotImplementedError` (expected — Part B not wired yet). This confirms commitment parsing and challenger selection work end-to-end.

## Fixing the SyntaxError in chain.py

If you see `SyntaxError: parameter without a default follows parameter with a default` on line 248 of `validator/chain.py`, the `set_weights` signature has `netuid: int = 14` with a default before positional params. Fix: remove the `= 14` default so it reads `netuid: int`.

## Environment variables

All validator constants can be overridden without touching code:

```bash
export CACHEON_NETUID=460
export CACHEON_NETWORK=test
export CACHEON_WALLET_NAME=default
export CACHEON_WALLET_HOTKEY=default
export CACHEON_STATE_DIR=/tmp/cacheon_state
export CACHEON_DRY_RUN=1
export CACHEON_POLL_INTERVAL_S=60   # shorter for dev
```

Then just run:

```bash
python scripts/remote_validator.py -v
```
