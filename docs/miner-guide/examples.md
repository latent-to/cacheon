# Example bundles

Examples are development controls, not crowns or performance claims.

## Node controls

| Example | Expected behavior |
|---|---|
| [miner_node_identity](https://github.com/latent-to/cacheon/tree/main/examples/miner_node_identity) | Returns the original MLP result; import smoke passes and engine audit should match stock. |
| [miner_node_wrong](https://github.com/latent-to/cacheon/tree/main/examples/miner_node_wrong) | Scales a tensor-valued MLP result by 1.5; import smoke passes and engine audit should fail. |

Both name `model.layers.*.mlp`. Use them only where that node exists and is
supported. Set the actual `competition.arena` before submitting; the examples
omit it so local checks do not invent an arena identity.

Follow [Your first bundle](your-first-kernel.md) to copy, scan, smoke-test and
run the real engine check. Keep results and compiler caches outside the bundle.

## Qwen development inputs

The [Qwen3.6-35B-A3B input kit](https://github.com/latent-to/cacheon/tree/main/examples/arena_inputs/qwen36)
has short- and long-context checks, the H100 image recipe, engine settings,
measured check times and routing lines. It does not change the qualification workload.

## GLM baseline

The [October 1 GLM baseline source release](https://github.com/latent-to/cacheon/releases/tag/glm-baseline-20261001)
contains the crowned bundle used by the operator's new commission, with its content
hash and archive checksum. Start from those sources for the `glm53-b300-node-v1`
arena, then follow [Submitting](submitting.md) with your own checked improvements.

The [GLM champion node bundle](https://github.com/latent-to/cacheon/tree/main/bundles/glm53_champion_nodes)
packages the four retained GLM implementations behind decoder-layer and final-norm
nodes. Its README specifies the B300 topology and checker inputs; its provenance
file identifies the unchanged kernel sources. It is the model-specific baseline
assembly, while the controls above demonstrate the minimal node interface.

No registered target permits an engine-wide setup hook. A candidate optimizes only
its declared node, not unrelated model-serving policy.
