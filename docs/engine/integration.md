# After a crown

A crown is retained measurement evidence. It updates the arena's evaluation
stack transactionally and can earn standing reward under the active emissions
policy. It does not ship anything.

Integration into maintained source, release signing, and serving are separate
authorities from evaluation-stack settlement, and this repository does not
implement them. There is no manifest type a crown can be promoted into, no
review record, and no release descriptor. Turning a crowned proposal into
maintained code is a maintainer decision made in source control, with its own
provenance, license, security, compatibility, and test review. Nothing in the
intake, referee, or settlement path performs or records that decision.

What the repository keeps:

- `EvaluationStackManifest`, the referee's content-addressed incumbent. Its
  entries are hostile proposal references and are executed only inside the
  isolated evaluation boundary. See [Stack manifests](../reference/stack-manifests.md).
- Deterministic materialization of an evaluation stack from proposal sources
  in the engine tree, with the rebuild and native-artifact identities that
  make a launched engine reopenable.
- Model provisioning, which seals the bytes an engine serves. See
  [Model provisioning](model-provision.md).

A crown is never permission to run miner code outside the referee, and no
supported path leads from a miner URL, a chain record, an evaluation bundle,
or a crown to a serving host.

Source: [`cacheon/stack_manifest.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/stack_manifest.py) and
[`cacheon/engine_tree.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/engine_tree.py).
