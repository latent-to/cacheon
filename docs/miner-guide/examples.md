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

## DeepSeek-V4.1-Flash development inputs

The [DeepSeek-V4.1-Flash input kit](https://github.com/latent-to/cacheon/tree/main/examples/arena_inputs/dsv41flash)
has the B300 image recipe, the TP2 engine settings with the Engram host-table
environment, 24 development requests and the node addresses that execute on
this model. It does not change the qualification workload.

## GLM baseline

The [GLM champion node bundle](https://github.com/latent-to/cacheon/tree/main/bundles/glm53_champion_nodes)
packages the four retained GLM implementations behind decoder-layer and final-norm
nodes. Its README specifies the B300 topology and checker inputs; its provenance
file identifies the unchanged kernel sources. It is the model-specific baseline
assembly, while the controls above demonstrate the minimal node interface.

No registered target permits an engine-wide setup hook. A candidate optimizes only
its declared node, not unrelated model-serving policy.
