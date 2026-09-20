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

## Retained catalog fixtures

Other examples in the tree exercise older `SlotSpec` contracts, adversarial
controls, native-build paths and identity tests. Their output-buffer signatures
are not the node ABI. The presence of a fixture does not open a serving lane or
establish that it beats the current incumbent. Use the node controls for the
current miner workflow and the [source catalog](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py)
when maintaining a retained reference fixture.

No registered target permits the old `miner_setup_demo` engine-wide setup hook.
A candidate optimizes only its declared node, not unrelated model-serving policy.
