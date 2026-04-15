"""Smoke test — runs the harness with Qwen2.5-0.5B on CPU or MPS.

Same architecture as Qwen2.5-7B-Instruct, just smaller (~1GB).
Verifies monkey-patch mechanics, generate loop, and verification logic
before you touch the H100.

Usage:
    python scripts/smoke_test.py               # auto-detects MPS or CPU
    python scripts/smoke_test.py --device cpu
    python scripts/smoke_test.py --device mps
"""

import argparse
import logging
import sys
from pathlib import Path

# Make sure the repo root is on the path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of Japan.",
    "What color is the sky?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on (default: auto-detect cuda > mps > cpu)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max tokens to generate per prompt (default: 32)",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA (%s)", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("Using CPU")
    else:
        device = args.device
        logger.info("Using %s (user-specified)", device)

    # Import here so logging is set up first
    from inference_engine.harness import Harness
    from inference_engine.passthrough import PassthroughPolicy

    logger.info("Loading %s on %s…", SMOKE_MODEL, device)
    harness = Harness(
        model_name=SMOKE_MODEL,
        device=device,
        dtype=torch.float32,   # float32 for CPU/MPS stability
    )

    # --- Step 1: unit-level monkey-patch check ---
    logger.info("\n=== Step 1: Monkey-patch sanity ===")
    call_log = []

    class TracePolicy(PassthroughPolicy):
        def write(self, keys, values, layer_idx, positions):
            call_log.append(("write", layer_idx, keys.shape))
            super().write(keys, values, layer_idx, positions)

        def attend(self, query, layer_idx, **kwargs):
            call_log.append(("attend", layer_idx))
            return super().attend(query, layer_idx, **kwargs)

    policy = TracePolicy()
    policy.setup(harness._cache_config)
    harness.run(policy, [PROMPTS[0]], max_new_tokens=3)

    num_layers = harness._cache_config.num_layers
    writes = [e for e in call_log if e[0] == "write"]
    attends = [e for e in call_log if e[0] == "attend"]
    logger.info(
        "  write() calls: %d  (expected ≥ %d — one per layer per step)",
        len(writes), num_layers,
    )
    logger.info(
        "  attend() calls: %d (expected ≥ %d)", len(attends), num_layers
    )
    assert len(writes) >= num_layers, "write() not called for every layer"
    assert len(attends) >= num_layers, "attend() not called for every layer"
    logger.info("  PASS")

    # --- Step 2: verify passthrough == unpatched ---
    logger.info("\n=== Step 2: Verification (passthrough == unpatched HF) ===")
    ok = harness.verify(PROMPTS, max_new_tokens=args.max_new_tokens)
    if not ok:
        logger.error("FAIL — passthrough output does not match unpatched model.")
        sys.exit(1)

    # --- Step 3: baseline metrics ---
    logger.info("\n=== Step 3: Baseline metrics ===")
    result = harness.run(
        PassthroughPolicy(), PROMPTS, max_new_tokens=args.max_new_tokens
    )
    logger.info("  Latency:       %.2fs total", result.latency_s)
    logger.info(
        "  Peak memory:   %.2f MB",
        result.peak_memory_bytes / 1e6 if result.peak_memory_bytes else 0,
    )
    logger.info(
        "  Policy memory: %.4f MB", result.policy_memory_bytes / 1e6
    )
    for prompt, text in zip(PROMPTS, result.output_texts):
        logger.info("  Q: %s", prompt)
        logger.info("  A: %s\n", text)

    logger.info("All smoke tests passed.")


if __name__ == "__main__":
    main()
