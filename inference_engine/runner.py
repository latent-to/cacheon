"""Phase 3 — Subprocess runner for miner policy submissions.

Executes a miner policy in an isolated child process, deserialises
the output, and validates shape / dtype / NaN / range.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid

import torch

from .policy import CacheConfig
from .sandbox import CheckResult, check as static_check

logger = logging.getLogger(__name__)

VALUE_RANGE = (-100.0, 100.0)
TIMEOUT_SECONDS = 300
ATTN_WEIGHT_SUM_TOL = 0.05


def _write_worker(workdir: str, policy_source: str, config: CacheConfig) -> str:
    """Write the worker script that the subprocess will execute.

    Returns the path to the worker script.
    """
    config_dict = {
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "max_seq_len": config.max_seq_len,
    }

    policy_path = os.path.join(workdir, "policy_submission.py")
    with open(policy_path, "w") as f:
        f.write(policy_source)

    worker_source = textwrap.dedent(f"""\
        import json, sys, os, importlib.util, torch
        from pathlib import Path

        config_json = {json.dumps(config_dict)!r}
        config = json.loads(config_json)

        spec = importlib.util.spec_from_file_location(
            "policy_submission", {policy_path!r},
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find the KVCachePolicy subclass
        policy_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (isinstance(obj, type)
                and hasattr(obj, "write") and hasattr(obj, "attend")
                and obj.__name__ != "KVCachePolicy"):
                policy_cls = obj
                break

        if policy_cls is None:
            print(json.dumps({{"error": "no KVCachePolicy subclass found"}}))
            sys.exit(1)

        # Minimal synthetic run: 1 layer, small shapes
        from inference_engine.policy import CacheConfig as CC, AttentionOutput

        cfg = CC(
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            max_seq_len=config["max_seq_len"],
            dtype=torch.float32,
        )

        policy = policy_cls()
        policy.setup(cfg)

        batch = 1
        seq_len = 4
        kv_heads = config["num_kv_heads"]
        q_heads = config["num_heads"]
        head_dim = config["head_dim"]

        k = torch.randn(batch, kv_heads, seq_len, head_dim)
        v = torch.randn(batch, kv_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)
        policy.write(k, v, layer_idx=0, positions=positions)

        q = torch.randn(batch, q_heads, 1, head_dim)
        out = policy.attend(q, layer_idx=0)

        result = {{
            "output_shape": list(out.output.shape),
            "output_dtype": str(out.output.dtype),
            "output_has_nan": bool(torch.isnan(out.output).any()),
            "output_has_inf": bool(torch.isinf(out.output).any()),
            "output_min": float(out.output.min()),
            "output_max": float(out.output.max()),
            "memory_bytes": policy.memory_bytes(),
        }}

        if out.attention_weights is not None:
            w = out.attention_weights
            result["attn_weights_shape"] = list(w.shape)
            result["attn_weights_sum_last_dim"] = float(w.sum(dim=-1).mean())
        else:
            result["attn_weights_shape"] = None

        output_path = os.path.join({workdir!r}, "result.json")
        with open(output_path, "w") as f:
            json.dump(result, f)
    """)

    worker_path = os.path.join(workdir, "worker.py")
    with open(worker_path, "w") as f:
        f.write(worker_source)

    return worker_path


def _validate_output(
    result: dict,
    config: CacheConfig,
) -> str | None:
    """Validate the deserialized worker output. Returns a reason string on
    failure, None on success."""

    if "error" in result:
        return f"worker error: {result['error']}"

    expected_shape = [1, config.num_heads, 1, config.head_dim]
    if result["output_shape"] != expected_shape:
        return (
            f"output shape {result['output_shape']} != "
            f"expected {expected_shape}"
        )

    if result["output_has_nan"]:
        return "NaN in output tensor"
    if result["output_has_inf"]:
        return "Inf in output tensor"

    lo, hi = VALUE_RANGE
    if result["output_min"] < lo or result["output_max"] > hi:
        return (
            f"output values out of range [{lo}, {hi}]: "
            f"min={result['output_min']:.2f}, max={result['output_max']:.2f}"
        )

    if result.get("attn_weights_shape") is not None:
        expected_kv_len = 4  # seq_len used in synthetic run
        expected_attn_shape = [1, config.num_heads, 1, expected_kv_len]
        if result["attn_weights_shape"] != expected_attn_shape:
            return (
                f"attention_weights shape {result['attn_weights_shape']} != "
                f"expected {expected_attn_shape}"
            )
        s = result.get("attn_weights_sum_last_dim", 0.0)
        if abs(s - 1.0) > ATTN_WEIGHT_SUM_TOL:
            return (
                f"attention weights don't sum to 1 (mean sum={s:.4f})"
            )

    return None


def run_check(
    source: str,
    config: CacheConfig,
    timeout: int = TIMEOUT_SECONDS,
) -> CheckResult:
    """Run the policy in a subprocess and validate its output.

    Layer 1 (static AST) runs first. If it passes, the policy is executed
    in an isolated child process with a hard timeout.
    """

    static = static_check(source)
    if not static.ok:
        return static

    workdir = os.path.join(tempfile.gettempdir(), f"cacheon_{uuid.uuid4().hex}")
    os.makedirs(workdir, exist_ok=True)

    try:
        worker_path = _write_worker(workdir, source, config)
        result_path = os.path.join(workdir, "result.json")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {**os.environ, "PYTHONPATH": repo_root}

        try:
            proc = subprocess.run(
                [sys.executable, "-u", worker_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                ok=False,
                reason=f"policy execution timed out after {timeout}s",
            )

        if proc.returncode != 0:
            stderr = proc.stderr[-500:] if proc.stderr else "(no stderr)"
            return CheckResult(
                ok=False,
                reason=f"policy subprocess crashed (exit {proc.returncode}): {stderr}",
            )

        if not os.path.exists(result_path):
            return CheckResult(
                ok=False,
                reason="worker did not produce result.json",
            )

        with open(result_path) as f:
            result = json.load(f)

        err = _validate_output(result, config)
        if err is not None:
            return CheckResult(ok=False, reason=err)

        return CheckResult(ok=True)

    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
