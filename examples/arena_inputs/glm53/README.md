# GLM-5.3 development check inputs

Use these inputs with the commissioned GLM-5.3 NVFP4 model and SGLang 0.5.18
image on four B300 GPUs (TP4 and attention DP4). Keep this folder outside
the submitted bundle. Mount it read-only at `/arena`, the model at `/model`,
and a writable result/cache directory at `/work`.

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The set contains eight public synthetic prompts, each tokenized to exactly
8,192 tokens with this GLM tokenizer, and eight deterministic output tokens
with EOS stopping disabled. It exercises a substantial prefill and decode on
every rank while keeping the untimed, per-call development audit bounded.
The text describes summing a list of integers and dividing by its length; each
prompt has a distinct numbered prefix. No hidden quality questions are included.

Use the default four-window minimum and 1,800-second limit per engine. The
checker runs every supplied request, then starts a fresh graph engine only
after the audit passes. A development PASS covers these inputs; the arena
still qualifies candidates on its complete served workload and quality set.
Larger batches, long context, and uneven request lengths remain necessary
when those regimes are part of the candidate's intended optimization.
