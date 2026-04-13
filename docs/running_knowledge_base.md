# LLM Knowledge Base

_Running notes on transformer internals — written for someone who knows software but is new to ML._

---

## Attention Mechanism

### Q, K, V — what they actually are

Every token produces **three vectors**, all projected from the same input vector via learned weight matrices:

| Vector        | Ask yourself                           | Role                                   |
| ------------- | -------------------------------------- | -------------------------------------- |
| **Q** (Query) | _What am I looking for?_               | Sent out to match against other tokens |
| **K** (Key)   | _What do I offer to others?_           | Receives queries from other tokens     |
| **V** (Value) | _What do I contribute if attended to?_ | The actual content that gets summed    |

```
Q = input @ W_q
K = input @ W_k
V = input @ W_v
```

These are **not** separate properties of a word — they are three independent learned projections of the same vector. Every token generates all three.

**The computation:**

```
scores  = Q @ K.T          # dot product → raw relevance score for every pair
weights = softmax(scores)  # normalize → probabilities summing to 1
output  = weights @ V      # weighted average of all V vectors
```

**Why O(n²):** For each of the n tokens, you compute a dot product against every other token's K — that's n × n operations. A 10k-token sequence = 100M dot products. This is the central scaling problem of transformers.

---

### Multi-Head Attention

Instead of one attention operation, run **H independent heads in parallel**, each in a smaller subspace:

```
# Each head projects into d_model/H dimensions
Q_i = input @ W_qi   # smaller slice of the space
K_i = input @ W_ki
V_i = input @ W_vi

output = concat(head_1, ..., head_H) @ W_o  # concatenate and project back up
```

**Cost stays the same** — H small attentions ≈ 1 big attention.

**Why it matters:** Each head learns to track a different _type_ of relationship. In practice you find heads specializing in:

- **Syntactic** — subject→verb, verb→object dependencies
- **Coreference** — "the" at pos 5 pointing back to "The" at pos 0
- **Positional** — attending mostly to nearby tokens
- **Semantic** — broader meaning similarity across the sentence

No single head learns all of this. The model distributes the work.

**KV cache implication:** Each head has its own K and V matrices, so cache size scales by H. A model like GPT-3 (96 layers × 96 heads) has an enormous cache. Head-level sparsity patterns differ — positional heads are locally redundant, coreference heads have rare but critical long-range links. Relevant for eviction strategy design.

---

### RoPE (Rotary Position Embedding)

**The problem:** Attention is purely content-based. Without positional info, "the cat sat on the mat" and "the mat sat on the cat" look identical to the model.

**Old approach:** Additive positional embeddings — add a learned position vector to each token before attention. Works, but encodes _absolute_ position (token is at slot 7) rather than _relative_ distance (these two tokens are 3 apart). Relative distance is usually what matters.

**What RoPE does:** Instead of adding position to the token, it _rotates_ the Q and K vectors by an angle proportional to their position:

```
token at position m → Q rotated by m·θ
token at position n → K rotated by n·θ
```

When you compute Q·K, the rotation math causes the score to depend only on `(m - n)·θ` — **pure relative distance**, regardless of absolute position.

**Why it won** (used in LLaMA, Mistral, basically all modern models):

- Generalizes to longer sequences — rotation extrapolates naturally; learned absolute embeddings don't
- Zero extra parameters — it's a fixed formula applied at runtime
- Applied directly to Q and K, doesn't touch token embeddings

**KV cache implication:** RoPE rotation is baked into cached K vectors at their original position. You can't re-insert an evicted K row at a different position — its rotation is tied to where it came from. Eviction policies need to account for this.

---

## Prefill vs Decode

Transformer inference runs in two stages:

**Prefill:** the entire input prompt is processed in one parallel forward pass. All tokens go through QKV projections simultaneously, populating the KV cache. No output token is sampled yet.

**Decode:** one new token is generated per step. Its Q attends to all cached K/Vs from prefill plus any previously decoded tokens. Output distribution → sample next token → repeat until done.

```
Prefill:  tokens [0..499] → QKV → write K/V for all 500 to cache (parallel)
Decode 1: token  [500]    → QKV → write K/V[500], attend to [0..500] → "The"
Decode 2: token  [501]    → QKV → write K/V[501], attend to [0..501] → "capital"
```

The KV cache exists so decode steps don't recompute K/V for the prompt on every token. This is why cache size is the bottleneck — it grows by one row per decode step, per layer, per head.

**Interface implication:** `write()` receives `seq_len = full_prompt_length` during prefill and `seq_len = 1` during decode. Same method, different shapes — no special-casing needed.

---

## Monkey-patching attention in this stack

**What it is:** Replacing a method or attribute on an existing class at runtime, without editing the library source. Example: `Foo.bar = new_bar` makes every `Foo` instance use `new_bar`.

**What HuggingFace attention does:** Each layer’s `self_attn.forward()` projects to Q, K, V; applies RoPE to Q and K; runs softmax(Q @ K.T) @ V; returns through the output projection.

**What the harness does:** It monkey-patches each layer’s `forward` with a wrapper that keeps the Q/K/V projections and RoPE (RoPE must run before any KV write), then delegates attention to the miner’s policy (`write` K/V, `attend` from Q) instead of the built-in matmul path, and finishes with `o_proj` on the policy output.

**Why not vLLM:** vLLM’s PagedAttention lives in C++/CUDA and isn’t cleanly interceptable from Python. HuggingFace attention is ordinary Python `forward` methods, so `model.layers[i].self_attn.forward = patched_forward` works with no extra build. Tradeoff: HF is slower than vLLM in production, but for an evaluator where baseline and miner share the same stack, relative comparison stays fair.
