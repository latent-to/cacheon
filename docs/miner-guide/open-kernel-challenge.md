# Open Kernel Challenge

The [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) asks for
faster serving of Meta's Muse Spark Open on H200 GPUs. Its kernel track (five
single-H200 problems) runs through November 4, 11:59 pm PT; its serving track
(SGLang patches scored on a whole 8×H200 node) runs November 5–19. The
challenge pays its prizes on the serving board; the top 50 on its overall
kernel board get serving-queue priority.

Cacheon pays 10% of its emissions to miners who hold the best official results
and publish the code they submitted. Everyone enters the challenge under their
own account and builds on each other's published code.

## Enter

1. Install the challenge CLI and sign in with your GitHub account:
   `curl -fsSL https://openkernelchallenge.dev/install.sh | sh`, then
   `okc login`. Your board name is your GitHub login.
2. Start from a problem's kit (`okc init <problem>`) and submit with
   `okc submit`. The challenge's [rules](https://openkernelchallenge.dev/rules)
   and daily limits apply.
3. Register a hotkey on the Cacheon subnet; only registered hotkeys can receive
   weight.

## Claim a board

After an official run lands on a board, publish a **public gist from the same
GitHub account** containing the exact file you submitted (the kernel file, or
the serving `patch.diff`) and `cacheon-okc-claim.json`:

```json
{
  "board": "muse_moe_w4a16",
  "job": "9c8b7e9f45234a99",
  "hotkey": "YOUR_HOTKEY_SS58",
  "code_sha256": "SHA256_OF_THE_SUBMITTED_FILE",
  "signature": "HOTKEY_SIGNATURE_HEX"
}
```

`board` is the problem name, or `serving`. The signature is your hotkey's
signature over
`cacheon-okc-claim:v1:<board>:<job>:<github login>:<code_sha256>`:

```python
import hashlib

import bittensor as bt

code = open("submission.py", "rb").read()
digest = hashlib.sha256(code).hexdigest()
wallet = bt.Wallet(name="COLDKEY_NAME", hotkey="HOTKEY_NAME")
message = f"cacheon-okc-claim:v1:muse_moe_w4a16:9c8b7e9f45234a99:your-login:{digest}"
print(digest, wallet.hotkey.sign(message.encode()).hex())
```

Keep one claim gist per board. A claim counts only while its job is still your
row on that board, so when a better run replaces your row, update the gist with
the new file, job and signature.

## How the share is paid

- Each counted board has a crown: the best claimed official result. A new claim
  takes a held crown only by beating the holder's result by 1% on a kernel board
  or 2% on the serving score, so resubmitting shared code cannot win on judge
  noise. If the holder's claim lapses (their row changes without a matching
  claim, or the run is withdrawn or removed), the best current claim takes it.
- Until a serving crown exists, `muse_bf16_proj`, `muse_attention_decode`,
  `muse_moe_w4a16` and `muse_decode_layer` (the boards that decide serving
  priority) each pay a quarter of the share to their crown. `muse_prefill_layer`
  is not paid.
- Once a serving crown exists, it takes the whole share.
- Unheld parts are burned. Crowns refresh every 15 minutes; on-chain weights
  follow after commit-reveal and validator consensus.

## Build on each other

Every claim is a public gist holding the exact winning code. The validator
publishes the current crowns, with their gist links, on the Cacheon dashboard.
Start from the best published code, improve it, submit it under your own
account, and claim the board when your official result clears the margin.
