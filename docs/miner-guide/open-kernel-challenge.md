# Open Kernel Challenge

The [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) asks for
faster serving of Meta's Muse Spark Open on H200 GPUs. Its kernel track (five
single-H200 problems) runs through November 4, 11:59 pm PT; its serving track
(SGLang patches scored on a whole 8×H200 node) runs November 5–19. The
challenge pays its prizes on the serving board; the top 50 on its overall
kernel board get serving-queue priority.

Cacheon pays 10% of its emissions to miners whose challenge runs are in the top
3 of a board and who submit that code to Cacheon.

## Enter

1. Install the challenge CLI and sign in: `curl -fsSL
   https://openkernelchallenge.dev/install.sh | sh`, then `okc login`. You accept
   the challenge's [rules](https://openkernelchallenge.dev/rules) yourself.
2. Start from a problem's kit (`okc init <problem>`), develop on any Hopper GPU
   (the judge runs one H200), and submit with `okc submit`. Note the job id it
   prints; `okc jobs` lists them again.
3. Register a hotkey on the Cacheon subnet; only registered hotkeys can receive
   weight.

## Submit your code to Cacheon

Put the exact file you submitted (the kernel file, or the serving `patch.diff`)
alone in a directory, publish it as for any Cacheon submission, and commit it
**without paying** (no `--pay`), adding `?okc=<board>:<job>` to the URL. Set
`NET`, `NETUID`, `WALLET` and `HOTKEY` as in
[Submitting a proposal](submitting.md#step-by-step-commands):

```bash
mkdir okc_moe && cp submission.py okc_moe/
python -m cacheon.cli chain-publish okc_moe --out dist/okc_moe.tar.gz
python -m cacheon.cli chain-submit okc_moe \
  --url "<URL chain-publish printed>?okc=muse_moe_w4a16:9c8b7e9f45234a99" \
  --netuid "$NETUID" --network "$NET" --wallet "$WALLET" --hotkey "$HOTKEY"
```

`board` is the problem name (`muse_bf16_proj`, `muse_attention_decode`,
`muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer`) or `serving`; `job`
is your run's job id on that board. The claim is not evaluated and costs no eval
fee. If your host serves the archive only at its exact URL, host it where extra
query parameters are ignored (S3, R2 and most static hosts do).

## How the share is paid

- Every 15 minutes the validator reads the six boards' top 3 and the claims. A
  claim counts when its job is in its board's top 3 and its archive fetched and
  matched the committed hash. The first counting claim of a job owns it.
- The share is split equally among counting claims. With none, it goes to
  UID 215.
- When a better run replaces your row, its new job needs a new claim.
- The [arena page](https://dashboard.cacheon.ai/okc) lists every claim, its
  standing and the code behind it. On-chain weights follow after commit-reveal
  and validator consensus.

## Build on each other

The code behind every paid claim is public on the arena page. Start from the
best of it, improve it, submit it under your own account, and claim your row.
