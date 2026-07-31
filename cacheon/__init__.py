"""Cacheon: a validator harness for an inference-throughput competition on SGLang.

The competition swaps an untrusted, miner-supplied kernel into a *fixed* model
graph at a small number of typed op-slots, then scores it on (a) end-to-end
throughput and (b) output-distribution fidelity (KL) against a frozen reference
run of the same model. The validator owns the model, the graph, the dispatch
seam, the build, the timing, and the reference.

This package is deliberately split so the parts that do NOT need a GPU (manifest
parsing, static policy scanning, registry/eligibility, scoring math) can run and
be tested anywhere, while the GPU-only pieces (kernel correctness on device,
end-to-end SGLang eval) are isolated behind lazy imports.

Threat model in one sentence: with Triton/CuteDSL the miner's kernel is *Python
that runs in-process*, so artifact-level safety is impossible and the real
boundary is process + GPU-context isolation; everything here is designed to
shrink the miner's host surface and to keep timing/scoring outside the miner's
reach. See ``cacheon/sandbox.py`` and the README for what is and isn't enforced.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = [
    "__version__",
]
