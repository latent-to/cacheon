# Stack manifests

Cacheon represents the referee's incumbent and the pristine reference with
separate canonical manifest types. They are deliberately not interchangeable.

## The two questions they answer

| Manifest | Question | Proposal source? | Arena identity? | Timed? |
|---|---|---:|---:|---:|
| `EvaluationStackManifest` | What complete incumbent does this arena evaluate? | allowed | yes | as the policy-required B and optional/required B′ reads |
| `ReferenceManifest` | What pristine candidate-free engine grades sealed trajectories? | rejected | yes, with reference identities | no |

## Evaluation stack

`EvaluationStackManifest` identifies the complete incumbent used by a specific
arena. It binds:

- runtime and base-engine digests;
- the arena digest;
- the full target-catalog snapshot and digest; and
- one content-addressed contribution reference per active target.

Its entries are hostile `ProposalContributionRef` values, executed only inside
the referee's isolation. `with_contribution()` creates a new immutable identity;
it never edits the old one.

Conceptual shape:

```json
{
  "type": "evaluation_stack",
  "schema_version": 1,
  "stack_policy_version": "...",
  "runtime_digest": "<sha256>",
  "base_engine_digest": "<sha256>",
  "arena_digest": "<sha256>",
  "catalog_digest": "<sha256>",
  "catalog_snapshot": {},
  "entries": {
    "moe.fused_experts": { "type": "proposal", "...": "..." }
  }
}
```

### Read an evaluation identity in layers

- `runtime_digest` fixes the executable runtime context.
- `base_engine_digest` fixes the engine before active contribution deltas.
- `arena_digest` fixes the hardware/workload/policy comparison domain.
- `catalog_snapshot` explains the target policy used at that time, while
  `catalog_digest` authenticates those exact bytes.
- `entries` names the active contribution for each economic target.

Each entry separates the whole artifact digest, selected-payload digest,
target-spec digest, and attribution digest. Two archives are therefore not the
same marginal delta merely because they share a `bundle_id`, and historical
evidence cannot be reinterpreted through a newer catalog.

### A marginal transition

Suppose the incumbent already contains a `moe.fused_experts` contribution and a
new proposal wins `norm.rmsnorm`. C is not a two-file bundle: the validator
materializes a complete engine equal to the incumbent everywhere except the
resolved RMSNorm target. After two matching PASS attempts, settlement can
derive a new evaluation stack by adding that reference and applying any
catalog-defined displacement. The old manifest remains a content-addressed
rollback point.

That transition changes the stack digest even when runtime, base engine,
arena, and every unrelated contribution remain fixed.

## Pristine reference

`ReferenceManifest` is the separate quality authority used by pristine T. It
binds an empty, candidate-free stack and its materialized tree, launch,
runtime, base engine, arena, catalog, controller/worker distributions, exact
model bytes, logical hardware, workload, tokenizer, hidden corpus commitment,
hidden judge, and selection policy.

T is untimed. It grades trajectories sealed by the current v7 B/C/[B′] or v8
B/C/B′ schedule only after any registered eager audit A has completed and
candidate engines have been destroyed. Neither the incumbent evaluation stack nor a
candidate's self-reported scores can substitute for this authority.

Reference identity is intentionally broader than a kernel catalog. It binds
the controller and worker distributions, launch contract, sealed model bytes,
logical hardware, tokenizer, workload, hidden-corpus commitment, judge, and
selection policy. If any of those changes, old quality evidence cannot be
silently attached to the new reference.

## Proposal references

A proposal reference separates the whole artifact from the economic core:

```text
selected_delta_digest = H(
  target_id,
  target_spec_digest,
  selected_payload_digest
)
```

`ProposalContributionRef` names the hostile artifact digest beside the
selected-payload, target-spec, and attribution digests, so padding or
re-attributing an archive changes the artifact identity without changing the
selected delta.

## Canonical identity rules

- Parsing is exact-schema and rejects unknown or mistyped structure.
- Catalog snapshots travel with their digests; installed policy cannot silently
  reinterpret retained evidence.
- Entry keys must equal each reference's target ID.
- Target-spec digests must match the bound catalog context.
- Active-target displacement and conflicts are revalidated.
- Serialized ordering is canonical before a digest or signature is computed.

## Common rejection cases

| Symptom | Meaning |
|---|---|
| Unknown field during reopen | The object does not match the exact current schema |
| Catalog digest and snapshot disagree | The supplied context is inconsistent or forged |
| Entry key differs from `ref.target_id` | The mapping tries to relabel a contribution |
| Target-spec digest differs | Qualification used another semantic contract |
| Overlapping active targets | Catalog displacement/conflict exclusion was not applied canonically |
| Runtime/base/arena context mismatch | The stack is being reopened under a different environment |

Treat these as identity failures, not migration hints. The safe response is to
construct a new typed object through the appropriate transition and retain the
old identity for audit and rollback.

## When to use each API

- Construct an `EvaluationStackContext` from validator-owned arena policy and
  call `validate_against()` before planning the versioned speed schedule.
- Use `with_contribution()` only for a catalog-valid evaluation transition; it
  returns a new manifest and never edits the old one.
- Construct and reopen `ReferenceManifest` independently of candidate engines;
  never derive T from the current evaluation stack.

Source: [`cacheon/stack_manifest.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/stack_manifest.py) and
[`cacheon/eval/qualification.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification.py).
