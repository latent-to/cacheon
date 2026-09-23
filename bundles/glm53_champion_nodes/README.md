# GLM-5.3 champion node bundle

This bundle carries the retained projection, DP exchange, routed MoE and
normalization implementations through `model.layers.*` and `model.norm`.
`metadata/provenance.json` records each original bundle and the checksum of every
unchanged kernel source. `glm53_baseline/layer.py` provides the node preparation
and forward interface, preserving the stock layer's collective sequence and
shared-expert join.

The supported runtime is the commissioned GLM-5.3 NVFP4 model on four B300 GPUs,
with TP4, attention DP4 and attention TP1, BF16 activations and the pinned
SGLang 0.5.20 DSA runtime with FlashInfer 0.6.18. The wrapper rejects a different topology. It preserves
each original implementation's shape and phase selection: the projection fusion
covers 1–32 local decode rows, and routed MoE keeps its small-row finalizer and
large-prefill tactic. Other shapes follow the original serving implementation.

The bundle uses the installed Cacheon NVFP4 weight-view helper and declares both
native translation units through the ordinary bundle build recipe. Run the
supported checks inside the arena image with its published engine options and
development requests:

```bash
python -m cacheon.cli scan bundles/glm53_champion_nodes
python -m cacheon.cli verify bundles/glm53_champion_nodes
python -m cacheon.cli check bundles/glm53_champion_nodes \
  --model /model --engine-config engine-options.json \
  --requests development-requests.json --output check-results
```

`verify` checks scanning and interface imports. `check` runs the real model audit
and then a fresh graph engine. Qualification and served throughput require the
arena's paired workload and quality gates. This packaging does not create a new
reward claim or replace the original submissions' attribution and evidence.
