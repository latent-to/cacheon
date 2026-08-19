# Your first component bundle

This walkthrough builds a registered singleton proposal and runs the cheap
developer diagnostics. It needs no GPU. The result is a valid learning bundle,
not evidence of a competitive win.

## 1. Install a development checkout

```bash
git clone https://github.com/latent-to/cacheon.git
cd cacheon
python -m pip install -e '.[cpu,dev]'
```

On a GPU host, install the Torch build matched to the arena's pinned
CUDA/SGLang environment first, then install Cacheon without replacing it. The
[GPU setup guide](../dev/gpu-setup.md) explains the current development
boundary. The operator's frozen arena image and runtime identities are the
source of truth for an authoritative environment.

Use `python -m cacheon.cli` in commands below. It is explicit about the active
checkout and behaves correctly when engine diagnostics spawn worker processes.

## 2. Copy the CPU example

```bash
cp -R examples/miner_silu_torch my_silu
```

The committed
[example bundle](https://github.com/latent-to/cacheon/tree/main/examples/miner_silu_torch)
contains a source implementation that fills the supplied output:

```python
def silu_and_mul(x, out):
    d = x.shape[-1] // 2
    result = torch.nn.functional.silu(x[..., :d].float()).to(x.dtype)
    out.copy_(result * x[..., d:])
```

Replace `my_silu/manifest.toml` with an explicitly targeted manifest:

```toml
bundle_id = "my-silu-v1"
abi_version = "cacheon-op-abi-v0"

[competition]
target = "activation.silu_and_mul"
mode = "slot"

[[ops]]
slot = "activation.silu_and_mul"
source = "kernels/silu_and_mul.py"
entry = "silu_and_mul"
dtypes = ["float32", "bfloat16", "float16"]
metadata = "metadata/silu_and_mul.json"
```

The `[competition]` table asks for the registered singleton target. The op row
supplies its implementation. Those are separate identities even though both
currently use the string `activation.silu_and_mul`.

## 3. Scan the source tree

```bash
python -m cacheon.cli scan my_silu
```

`scan` checks manifest/path structure and performs the development static-policy
scan. Fix every reported item. A clean scan does not make code safe or
crownable; production still fetches, republishes, builds, and runs the proposal
inside validator-owned isolation.

`scan` also reports broken Triton kernels, marked `[BROKEN KERNEL]`, and exits
non-zero. **Run it before every submission.**

Triton is a JIT and compiles *per kernel, on that kernel's first invocation*. A
report means the kernel raises at trace time **if it is ever invoked**. If it is
never reached it is latent dead code and your bundle still evaluates — so this
is a warning about your kernel, not a prediction that your bundle fails. Fix it
regardless: a kernel that crashes the moment a shape reaches it is a defect
waiting for the workload that triggers it.

The most common instance is calling a host-side Triton helper from inside a
`@triton.jit` body:

```python
@triton.jit
def _kernel(out_ptr, D: tl.constexpr):
    col_offsets = tl.arange(0, triton.next_power_of_2(D))   # fails to compile
```

At trace time `D` is a `tl.constexpr` wrapper object, not a Python `int`, so the
host-side helper raises and compilation aborts. There is no in-language
replacement to swap to — `triton.language` does not export `next_power_of_2`.
Compute the value at the launch site and pass it in as its own `tl.constexpr`
parameter:

```python
@triton.jit
def _kernel(out_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D

BLOCK_D = triton.next_power_of_2(D)          # host, at the launch site
_kernel[grid](out, D=D, BLOCK_D=BLOCK_D)
```

`tl.arange` requires a compile-time power-of-two bound in any case, so passing it
as a `constexpr` argument is the required shape rather than a workaround.

## 4. Verify the callable contract

```bash
python -m cacheon.cli verify my_silu --device cpu --dtype float32
```

This diagnostic constructs validator-owned inputs and poisoned outputs, invokes
the bundle over the slot's profiles, detects input mutation and incomplete
writes, and compares the result with the trusted reference. The applicable
shape rows should be `ok`. Because the example declares graph-safe operation,
the CPU headline is `NUMERICAL_PASS ... graph=NOT_VERIFIED`, not a CUDA graph
pass.

A CPU pass proves only the local numerical ABI. It does not prove:

- CUDA compilation or architecture eligibility;
- CUDA-graph capture and replay;
- performance in the incumbent engine stack;
- serving quality on the arena workload;
- authoritative qualification or a crown.

For an intentional failure, run the committed wrong implementation:

```bash
python -m cacheon.cli verify \
  examples/miner_silu_broken_torch --device cpu --dtype float32
```

That bundle computes different math. It should exit nonzero with failed shape
results. Use it to confirm that your environment is exercising the gate you
think it is.

## 5. Add a real specialization

Once you replace the Torch body with a Triton, CUDA, or other target-approved
implementation, declare only the domain you actually support. For example:

```toml
[[ops]]
slot = "activation.silu_and_mul"
variant = "sm90-bf16"
source = "kernels/silu_sm90.py"
entry = "silu_and_mul"
dtypes = ["bfloat16"]
architectures = ["sm90"]
metadata = "metadata/silu_sm90.json"
```

```json
{
  "graph_safe": true,
  "capabilities": {
    "num_tokens": {"min": 1, "max": 4096}
  }
}
```

An architecture mismatch is N/A, not a pass. A capability domain matching none
of the verifier's applicable shapes also fails verification. If you add a
second variant, give every row a unique `variant` and make the domains provably
disjoint; there is no manifest-order priority.

## 6. Move to the matching GPU environment

First rerun ABI verification on the real dtype and architecture:

```bash
python -m cacheon.cli verify my_silu --device cuda --dtype bfloat16
```

For a collective target, use the arena's topology:

```bash
python -m cacheon.cli verify my_collective \
  --device cuda --dtype bfloat16 --world-size 4 --tp-size 4
```

Then follow the canonical
[performance-development procedure](../validator-guide/running-evals.md#performance-development)
in an environment matching the published arena contract. No repository command
materializes the incumbent/candidate engines for this local experiment. Bracket the
candidate with identical incumbent runs:

```text
B  -> C -> B′
speedup = candidate_rate / mean(baseline_before_rate, baseline_after_rate)
```

Keep the arena's CUDA-graph state, topology, model, dtype, workload, and charged-work
definition fixed. Reject a result when B/B′ drift is comparable to the claimed gain.
For a long-prefill target, make the workload genuinely prefill-heavy; repeated prompts
with a live radix cache can silently turn later iterations into decode/cache-hit work.

This local bracket is a performance hypothesis, not crown authority. The validator binds
resident B/C/B′, conditional C′/B″, registered eager audit A, then pristine T, together
with resources, graph evidence, hidden inputs, and calibrated policies, and requires a
separately bound reproduction.

## 7. Decide whether the target is worth pursuing

A correct kernel is the starting line. Profile the full incumbent engine and
ask whether this exact slot has enough wall-time share for your measured kernel
gain to matter. Then compare the complete candidate delta against the current
incumbent stack, not against a convenient stock or standalone baseline.

Continue with [Finding a win](finding-a-win.md), [Graph evidence](graph-safety.md),
and [Submitting](submitting.md).
