"""Entrypoint for `python -m inference_engine`.

Runs the harness smoke test: verify passthrough correctness, then collect
baseline metrics.
"""

import logging

from .harness import Harness
from .passthrough import PassthroughPolicy

logging.basicConfig(level=logging.INFO, format="%(message)s")

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a linked list.",
    "Summarize the theory of relativity in three sentences.",
    "What are the main differences between TCP and UDP?",
]

harness = Harness()

ok = harness.verify(TEST_PROMPTS)
print(f"\nVerification: {'PASS' if ok else 'FAIL'}")
if not ok:
    raise SystemExit(1)

policy = PassthroughPolicy()
result = harness.run(policy, TEST_PROMPTS)
print(f"Latency:        {result.latency_s:.2f}s")
print(f"Peak GPU:       {result.peak_memory_bytes / 1e9:.2f} GB")
print(f"Policy memory:  {result.policy_memory_bytes / 1e9:.4f} GB")
print(f"\nOutputs:")
for prompt, text in zip(TEST_PROMPTS, result.output_texts):
    print(f"  Q: {prompt[:60]}")
    print(f"  A: {text[:120]}…\n")
