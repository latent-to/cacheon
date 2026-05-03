# Mining on Cacheon

Full docs: https://cacheon.io/docs/miners/overview

## Quick start

```bash
export HF_TOKEN=hf_...

python miner/commit.py \
  --policy-file policy.py \
  --repo you/my-kv-policy \
  --wallet-name my-miner \
  --wallet-hotkey default \
  --network test --netuid 460
```

This uploads your `policy.py` to Hugging Face and commits the repo + revision SHA on-chain in one command. The validator picks it up within ~6 minutes.

## What's a policy?

A single Python class (`KVCachePolicy`) that controls how the KV cache is stored and retrieved during transformer inference. See the [full docs](https://cacheon.io/docs/miners/policy-interface) for the interface spec and [example policies](https://cacheon.io/docs/miners/example-policies).
