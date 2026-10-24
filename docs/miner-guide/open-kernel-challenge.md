# Open Kernel Challenge

The Cacheon OKC arena pays qualifying claims using OKC's public ranks.
The [arena dashboard](https://dash.cacheon.ai/okc) shows the current round's
start block, claims, and reward shares.

The [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) runs five
single-H200 kernel boards and an eight-H200 serving board. Cacheon's allocation
is 10% of emissions for qualifying top-three claims.

Every miner must register their username and submit their claim in the current
round, at or after its start block. Older registrations and claims are archived
and do not count. An existing qualifying OKC job can be claimed again after a
new username registration; the round does not require a new OKC evaluation.

## Register your exact username

Install the updated miner CLI from the competition's `kestrel` branch:

```bash
python -m pip install --upgrade 'git+https://github.com/latent-to/cacheon.git@kestrel'
```

The new username requirement is:

- The OKC username contains the literal, case-sensitive string `Cacheon`.
- It contains the first five lowercase hex characters of the miner hotkey's
  decoded 32-byte public key, without `0x`. These are hex characters, not the
  beginning of the SS58 address, and not the coldkey.
- The same hotkey commits the entire exact username on-chain before submitting
  the Cacheon claim. Registration may follow the OKC run or its top-three result.
- The committed username must match OKC's public `team` string exactly. No
  lowercasing, trimming, or other normalization is applied.

Set `NET`, `NETUID`, `WALLET`, and `HOTKEY` as in
[Submitting a proposal](submitting.md#step-by-step-commands), then register:

```bash
python -m cacheon.cli chain-okc-register 'Cacheon-<five-hex-hotkey-prefix>-name' \
  --netuid "$NETUID" --network "$NET" --wallet "$WALLET" --hotkey "$HOTKEY"
```

This is a normal plaintext, hotkey-signed commitment. It does not upload a bundle
or charge an evaluation fee. The command waits for finality and verifies the
exact name from chain state. Save the returned registration block. Use
`--dry-run` to validate and inspect the name without submitting.

A later bundle commitment replaces the live commitment slot, so the claim must
reference the saved registration block. Verification reads that historical
block under the claiming hotkey, rather than trusting a username supplied in
the URL or relying on the hotkey's latest commitment.

The oldest valid on-chain username commitment within the round wins the binding.
Each hotkey may bind one exact username, and each exact username may bind one
hotkey. A later commitment cannot change either side of an existing binding.
Registering the same pair again keeps its original priority; it does not reset
it. Claim arrival order does not establish or change username ownership. Within one block,
chain event order breaks ties. Later overwrites or deregistration do not release
an accepted binding within the round.

## Claim your OKC run

Use the ordinary challenge CLI to sign in, initialize a problem, and submit a
run. Keep the exact username registered above and note the qualifying job ID.
Publish the exact submitted code as an ordinary Cacheon bundle, then make an
unpaid claim with the registration block:

```bash
python -m cacheon.cli chain-submit okc_bundle \
  --url "<bundle URL>?okc=<board>:<job>&okc_user_block=<registration block>" \
  --netuid "$NETUID" --network "$NET" --wallet "$WALLET" --hotkey "$HOTKEY"
```

The boards remain `muse_bf16_proj`, `muse_attention_decode`, `muse_moe_w4a16`,
`muse_decode_layer`, `muse_prefill_layer`, and `serving`. The host serving the
bundle must tolerate the added query parameters.

## Existing reward rules

After username verification, the existing rules remain: a job must currently
rank in its board's top three; the bundle must fetch and match its committed
hash; a job may count only once; and code identical to an earlier
claim, ignoring whitespace and file names, pays only its earlier claimant.
The same username and hotkey binding may claim distinct jobs under these rules.
The share is split equally among counting claims, or goes to the configured
default hotkey if none qualify. Failed fetches require a new claim. Claimed code
remains private until its track closes.

Username registration and a five-character key marker do not prove code
authorship or that the submitted bundle matches the code evaluated by OKC.

Registration alone does not authorize rewards. Claims are checked against the
round's finalized commitment-history index, described in the
[validator chain loop](../validator-guide/chain-loop.md#static-allocation-configuration).
