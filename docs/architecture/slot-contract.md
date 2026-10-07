# Slot contract

The slot contract is Cacheon's narrow waist: a validator-owned boundary between
untrusted optimization code and a pinned inference engine. A bundle replaces
modules of the served model, named by [node address](#node-addresses), or the
scheduler's [prefix cache](#the-prefix-cache). Every contribution must satisfy the
invariants on this page. The executable pieces are the seam table
[`cacheon/seams.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/seams.py),
the node binder
[`sglang_nodes.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_nodes.py)
and the cache seam
[`sglang_cache.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_cache.py).

## The invariants

### 1. The validator owns the boundary

The validator owns the call site, the engine, the model and the arguments. A
node candidate receives the stock arguments of the module it replaces and returns
what stock returns; the binder checks the result structure before the engine
consumes it. A cache candidate receives the runtime cache object and returns its
class or a subclass.

### 2. Sampling stays validator-owned

A contribution may replace data-plane computation up to the logits, but it may not
control sampling, token selection or acceptance. Logits a node returns are audited
against stock on the same call like any other result.

### 3. Correctness is judged against stock in the running engine

There is no hand-written reference math. On an audited call the stock module
answers first, an honest twin answers on SGLang's native reference paths, and the
candidate's result and engine-state rows are graded by relative error against
stock, within a multiple of the twin's own error. A cache is graded by the bytes
it serves back for each claimed prefix. This gate is necessary but not
sufficient: production qualification also applies pristine T quality authority to
sealed end-to-end trajectories.

### 4. Miner-reported performance and evidence are not trusted

The host times requests outside the candidate process. The validator owns workloads, role schedules, output storage, reference work, evidence schemas, and verdicts. Candidate logs, self-reported throughput, and self-reported quality cannot mint a score.

A feature that cannot preserve these invariants is not a target. It needs a reviewed catalog or integration change, not a submission.

### 5. Closure is arena-scoped and never inspects an implementation

An arena may close a registered target it cannot currently measure. Closure
keys on the **submitted target name only**: intake parks a submission for a
closed target without judgement and releases its payment. Closing a target
closes its standalone lane and nothing else — implementing that computation
inside any open target's boundary is always legal and can never be a rejection
reason. A fused kernel is judged solely by the contract of the one target it
names.

## CUDA graph contract

Production qualification is graphs-on. A candidate cannot earn authority by
passing only eager execution when the arena serves captured graphs. The binder
binds each node right after the model loads, so both the prefill runner and the
decode graph runner capture the bound forward at every width. Model weights and
prepare-time state are capture-static.

`cacheon check` starts a graphs-on engine after its eager audit passes and
records whether each claimed address executed inside captured graphs on every
required rank. Qualification runs no separate graph stage: its proof is the
captured completions of the timed run, the audit, and the pristine quality gate
([Qualification](../validator-guide/qualification.md#gates-and-three-way-decisions)).

See [Graph safety](../miner-guide/graph-safety.md) for bundle-facing guidance.

## Variants and eligibility

A node may carry several implementation variants for disjoint, validator-observable
capability domains such as dtype or compute capability. Variants do not create new
reward units: every row resolves to the bundle's one target.

Eligibility is evaluated before candidate selection. Unknown capability fields,
overlapping ambiguous variants, unsupported topology, and missing prerequisites fail
closed or route to stock according to the registered pre-selection policy. Node rows
may not constrain token counts, because DP ranks hold different local batch sizes and
a split selection could desynchronize a collective. The miner cannot introduce a new
capability vocabulary through manifest extras.

## Adding a target

A boundary that is a module of the served model needs no validator change: it is a
node address. A runtime object that is not a module, as the prefix cache was, needs a
reviewed change: its row in [`seams.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/seams.py),
its adapter, its target and contract in the catalog, compatibility canaries against
the pinned runtime, and its documentation.

## Node addresses

Every slot name other than [`tree_cache`](#the-prefix-cache) is a node address: a dotted name from
`named_modules()` of the served model, where `*` stands for exactly one segment. `model.layers.*.mlp` is every MoE block, `model.layers.3` one decoder
layer, `model` the whole decoder stack. One adapter,
[`sglang_nodes.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_nodes.py),
serves every width, and a bundle that lists several addresses replaces several
nodes at once. Granularity is the model's own module tree: a span that is not a
module (on Qwen3.5 the attention block has no module of its own) is reached
through the nearest enclosing node.

The candidate is a drop-in for the node's stock `forward`.
`entry(prepared, *args, **kwargs)` receives the stock arguments and returns what
stock returns. `prepare(module)` runs once per bound node; a bundle without one
receives the module itself. While a candidate runs, the nodes beneath it serve
stock, so a wide candidate may call the stock children it does not replace.

Correctness has no declared reference math and therefore no offline verification.
Truth is the stock node in the running engine, on the same call:

1. on an audited eager call the stock node runs first;
2. its result tensors are kept, together with the engine-state rows the batch may
   write: the cache rows at `out_cache_loc` (whole pages where the pool stores
   pages) and, on hybrid models, the recurrent state rows of the batch's requests;
3. the arguments stock changed and the state rows are put back;
4. the honest twin answers the same call and is put back the same way: the stock
   node with supported fused ops on SGLang's native reference paths, giving the
   same math with different rounding. Ordinary residual RMSNorm uses an unfused
   addition in the input dtype before native normalization; explicit FP32-residual
   and other semantic overrides retain their native path. The DSA indexer has no native implementation
   and retains hardware dispatch; its children and surrounding ops still use
   native paths where called;
5. the candidate runs on the same call, and each row of its result and state rows
   (a token, a cache row, a request's state) is graded by its relative error
   against stock's.

Packed DSA MLA records are decoded as FP8 latent values with FP32 scales and BF16
rotary values. The separate index cache is decoded as FP8 keys with FP32 scales;
its touched pages are preserved and restored as raw bytes. DeepSeek-V4 pages are
decoded by the layout the pool declares: E4M3 values with one UE8M0 exponent per
tile and a BF16 rotary tail, or packed E2M1 values with E4M3 scales; its FP4
index pages are decoded the same way, and a kv-source layer's compressed pages,
index pages and per-request pending-pair ring are graded with its window pages.
An unrecognized packed layout raises instead of being interpreted as homogeneous
FP8. The first
failed window of each bound node is logged with its concrete name and tensor
position, including when the contribution claims a wildcard address.

Under GDN ReplaySSM speculative verification, a target-verify call also writes
replay rings, keyed by the batch's request slots, and per-draft conv windows,
keyed by verify scratch row. Both are preserved, restored and graded; the conv
windows through the pool's physical buffers. A BF16 ring's residual
(`rawv`/`rawk`) is graded summed with its high part (`d`/`k`), the one number
the pair holds. Any other ReplaySSM state layout raises.

A row passes within the larger of 2% and three times the twin's
90th-percentile row error on that node, taking the larger of the current call
and the recent-call estimate. Quiet decode history therefore cannot suppress
the reference noise of a later prefill call. The limit never exceeds 40%: the widest honest node
measured needed 33% and the wrong controls sat at 50%. Rows stock itself left
non-finite (an idle data-parallel rank's padding) are not graded. The tolerance is measured because honest
rounding grows with the width of the node: the twin sits 0.4% from stock at a block
and 4–11% at the whole stack. It is kept per bound node, not per address, because
the same address is quieter at layer 0 than at layer 39. Rows are graded rather
than tensors because a mixture-of-experts routing flip moves one whole token and
nothing else. Rows pool across calls into windows of 256 per node and graded
tensor, and a window passes when 75% of its rows do, so a one-token decode call is
never a verdict by itself. A wide node is therefore held only as tightly as honest
BF16 rounding allows at that width; a narrow node inside it is held tighter.

A whole-number or true/false result (expert ids, selected token indices, a mask) is
a choice, not a magnitude. Its row error is the share of the row's entries that
differ from stock's, position by position, under the same tolerance and the same
75% bar. Router weights are given in the order of the expert ids, so those paired
outputs retain their order. A kernel that changes their order belongs in a claim
on the enclosing module, whose activations are graded as numbers.

DSA's selected-token indices are an unordered collection: sparse attention reads
the selected cache positions without accompanying per-position weights. The
pinned Indexer, DSA attention and decoder-layer index outputs are compared by
membership and multiplicity. Permutation alone is not an error; missing choices,
duplicated IDs and changed padding counts still are. The tolerance and window
bar stay the same. This interpretation is confined to those model-defined index
outputs; it does not reorder execution results, router pairs or masks.

What stock leaves in its own arguments is not graded. The fused RMSNorm overwrites
its arguments and returns them, and a decoder layer leaves normed intermediates in
its dead input; values reach the caller through the result and the engine state.
A result that is a record rather than a tuple is graded through its fields.

`prepare` and `entry` receive the live module and must leave its methods alone.
Every callable on the node's modules and their classes is recorded at binding,
before any candidate code has run, and compared before each audited reference: a
candidate that rebinds a `forward` makes stock agree with it, and one that rebinds
a native path makes the twin noisy. A change raises, names the module and
attribute, and is receipted as the candidate's. One change is the engine's own:
SGLang's fused ops leave their dispatch target empty until the first call, so an
attribute that was empty at binding may be filled with one of that module's own
recorded methods and with nothing else. Weights are not recorded; a rewritten
weight changes the served model itself, which the end-to-end quality gate grades
against the pristine reference.

The audit draw and both reference passes happen before the dispatcher looks at the
candidate's eligibility. Eligibility can differ by rank (data-parallel ranks hold
different batches), and a node that contains a collective hangs unless every rank
runs it the same number of times.

A bundle's addresses must not contain one another, and an address must name at
least one module. Either failure raises at binding and is receipted as the
candidate's. Binding happens once, after `ModelRunner.load_model`, which is early
enough for the prefill and decode CUDA graph runners to capture the bound
`forward` at every width.

The graph proof for a node bundle is taken from the timed run itself. The
dispatcher, not the candidate, records whether each invocation happened inside a
CUDA-graph capture, and a graphs-on run requires that of every claimed node on
every rank: a candidate whose declared domain excludes every captured shape would
otherwise be timed as stock. A candidate that is captured but returns a stale
answer on replay is caught by the end-to-end quality gate: with one decoder layer
or one MoE block of forty returning its warm-up answer, the pristine reference's
NLL of the output went from 0.09 to 15.5 and 9.0
([runs](../results/qwen-h100-node-slots.md#stale-under-capture)).

A node-address bundle resolves to the
[`forward_pass` target](../reference/target-catalog.md#registered-targets), and
its reservation carries the addresses it declared. The
check separates honest from wrong at every width from one activation to the whole
forward pass; the runs are in
[Qwen H100 node slots](../results/qwen-h100-node-slots.md).

## The prefix cache

The address `tree_cache` names the scheduler's prefix cache: the object the
scheduler keeps as `tree_cache`, which matches a request's leading tokens to KV
slots already written, takes finished and chunked requests in, locks, evicts and,
with the hierarchical cache on, moves KV between device and host memory. It is a
separate `prefix_cache` target, with no sub-addresses. A cache replacement retains
the commissioned `forward_pass` contribution. Combining addresses in a manifest
is not a substitute for preserving independently owned stack entries.
[`sglang_cache.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_cache.py)
serves it; the node adapter does not.

There is one contract across models: `entry(cache)` receives the initialized
runtime cache and returns a subclass of `type(cache)`. The factory may initialize
state on that object; the validator binds the returned methods onto it. Its
components and transfer workers retain their references to the same object.
There is no model name, cache implementation name, dimension, or dtype in this
ABI, and no separate `prepare`. SGLang constructs the object, its components,
host tier and transfer counters before the factory runs. The replacement keeps
the engine's KV allocator and request pool, through which the scheduler allocates
and evicts the validator's KV memory. It must preserve the runtime object's
interfaces and state guarantees; returning a subclass is not correctness proof.
The choice is made once, at engine start, with an empty call descriptor: an op declaring
dtypes, architectures or eligibility never matches, and the run fails as `candidate_never_executed`.

The same content checks run for stock and candidate. Stock checking failures are
infrastructure failures; they are not attributed to a miner's cache. Storage
validation is adapter work under this common contract. It recognizes full-attention
KV, sliding-window KV and its index state (by slot, or by page where the window pool
stores whole pages), compressed KV and index pages, per-request sliding-window rings,
and the pending-pair ring a DeepSeek-V4 kv-source layer holds per request, which a
cache handoff must leave unchanged. A request-local ring is never stored in the tree,
so a prefix hit that skips its trailing window is refused. Recurrent checkpoints are
refused: no commissioned arena caches them. A new model does not require a new
miner contract; an unfamiliar storage layout requires validator support before a
contribution can use it.

The [GLM GPU checks](../results/prefix-cache.md) exercise prefix reuse,
native host restoration, reset, graph execution, audit import and rejection of
corrupted cached state under this contract. The GPU results cover GLM's
full-attention layout; they do not establish GPU coverage for every recognized
state layout.

Cache versions replace one another within this target. Iterative improvement
means the next implementation retains the useful behavior of the current winner
and beats that complete winner. The validator does not merge arbitrary cache
algorithms or infer source inheritance. After recommissioning, the next
qualification measures the additional gain over that cache and the retained
kernel stack; it does not repay their inherited speedup.

What a cache can fake is a hit: a served prefix whose slots do not hold what the
engine computed for it skips that prefill and returns wrong tokens fast. The
validator checks content, not the path the bytes took, so a host-memory tier,
stock's or the bundle's own, passes whenever it brings back the exact bytes.
Whenever the scheduler hands a request to the cache, before the cache sees it,
the validator hashes each complete page of KV the request's own forward passes
computed and records the pair of that hash and a digest of the prefix through the
page: the request's `extra_key` and `cache_salt`, its tokens, and under EAGLE the
token after the page, which the draft KV reads. At the same handoff it hashes up
to 64 randomly chosen pages the request read from the cache and requires each
pair to be on record. Hashes cover four layers, drawn at engine start, of each KV
buffer kind in the target and draft pools, the DSA indexer's pages and the shorter
DeepSeek-V4 index pages of each full page included. The pairs
live in a 16 MB table on the device, and a flush forgets them. After the cache
handles an unfinished request, the request's own slots beyond what the cache now
protects must be unmoved, and the row the next forward pass reads must agree with
the prefix left on the request. A match is SGLang's `MatchResult` and claims no more
tokens than its key. Served and protected lengths are whole pages, and the class the
factory returns must be concrete.

The check runs on the scheduler's stream behind the forward pass that wrote the
bytes; the host reads each verdict at a later handoff, and only a flush or an
audited request waits for one. A page is checked after the forward pass that read
it, so bytes moved into a served slot after that pass are not told apart from bytes
placed before. The check does not bound memory a cache allocates beyond the
engine's pools. Sliding-window KV may be overwritten during forward, so its
additional checks run in the existing untimed audit role: record computed state
before handing it to the cache, verify device hits before use, and verify host
restores after the native transfer stream completes. Unfinished requests retain
their live ring state. These checks sample up to four layers per transferable
state field. The handoffs
`match_prefix`, `cache_unfinished_req`, `cache_finished_req`,
`ready_to_load_host_cache` and `reset` may be overridden in the class but not replaced on
the instance or class later.

Every candidate refusal and every raise in a method the bundle defines stops the
engine as the candidate's failure. An unsupported cache or state layout is refused
before the factory runs. The cache runs in the scheduler,
never in a CUDA graph, so its completions count on a graphs-on run without a
capture; in the audit role each audited request waits for its verdict and adds one
unit to the address's audit receipt.

## Escape hatches

Normal target submissions cannot request arbitrary engine-wide setup or framework mutation. Cross-cutting proposals are not submittable; dependency patches are refused, and native builds use the validator-shipped, policy-constrained build step. Successful work should be resolved into a registered target or reviewed product source without relabeling changed selected payload bytes under old evidence.

With MTP enabled, contributions still optimize the registered target computation.
The validator owns the draft model, speculative schedule, sampling and acceptance
rules, and request batching. A contribution may not change those controls or
manipulate draft proposals or acceptance decisions to manufacture throughput.

This keeps experimentation possible without widening every ordinary submission's authority.

## Failure behavior by phase

| Phase | Example | Required behavior |
|---|---|---|
| Manifest resolution | Address outside the roots, overlapping nodes, ambiguous variant, stale contract digest | Reject before candidate execution |
| Pre-selection live routing | Shape or topology is outside a registered variant | Use the stock path when policy permits; do not count a candidate firing |
| Selected call | Candidate raises or returns a different result structure | In strict qualification, invalidate the candidate execution; a silent stock retry cannot produce crown evidence |
| Audit | Results or state rows differ from stock beyond the twin-scaled bound | Candidate `FAIL` |
| Graph replay | Output reflects capture-time input | Fail the pristine quality gate in qualification |
| End-to-end quality | Node audits pass but sealed trajectory regresses | Fail under pristine T quality authority |
| Infrastructure | Worker, device, or evidence authority cannot establish a valid result | `NO_DECISION`, not an attributable candidate loss |

This distinction explains why “fallback exists” and “the candidate qualifies” are
different statements. Fallback can preserve availability in a non-strict serving or
development context. Crownable evidence must prove that the selected candidate route
actually fired and completed.

## Target reviewer checklist

A new or changed target is ready only when a reviewer can answer yes to all of the
following:

- Is the boundary a runtime object the engine exposes at one place, and does
  sampling stay validator-owned?
- Is truth the stock object in the running engine, graded on the same call?
- Are capability domains finite, unambiguous, and observable before selection?
- Is there a real pinned-runtime chokepoint with a compatibility canary?
- Do strict-mode receipts and end-to-end qualification prove that the candidate path
  fired without fallback?
- Are its roots disjoint from every other target's, so no two targets own one region?

Passing a unit test without these properties is not sufficient to extend the narrow waist.

## Source map

- This page — normative invariants
- [`seams.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/seams.py) — the engine chokepoints
- [`sglang_nodes.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_nodes.py) — node binding and audit
- [`target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py) — economic target projection
- [`sglang_cache.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_cache.py) — prefix-cache seam and its claim ledger
