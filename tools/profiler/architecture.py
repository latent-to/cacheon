#!/usr/bin/env python3
"""Small built-in architecture descriptors for profiler metadata.

These descriptors are intentionally descriptive, not authoritative. They help the
UI and capture planner group evidence by architecture family; measured traces and
counters always win over inferred metadata.
"""
from __future__ import annotations

import re


UNKNOWN_ARCH = {
    "decoder_type": "unknown",
    "attention": "unknown",
    "layer_mix": "unknown",
    "features": {
        "dense": None,
        "moe": None,
        "gqa": None,
        "mla": None,
        "linear_attention": None,
        "ssm": None,
        "sparse_attention": None,
        "mtp": None,
    },
    "kv_cache": {"dtype": "unknown", "bytes_per_token_bf16": None, "note": "unknown"},
}


_DESCRIPTORS: list[tuple[re.Pattern, dict]] = [
    (re.compile(r"nemotron.*3.*ultra", re.I), {
        "family": "Nemotron 3",
        "scale": "550B total / 55B active",
        "decoder_type": "Hybrid sparse MoE",
        "attention": "Hybrid Mamba-2 SSD + attention",
        "layer_mix": "Mamba-2 SSD + attention + LatentMoE + MTP",
        "context_tokens": 1_000_000,
        "features": {"moe": True, "ssm": True, "linear_attention": False, "gqa": None,
                     "mla": False, "sparse_attention": False, "mtp": True, "dense": False},
        "kv_cache": {"dtype": "fp8", "bytes_per_token_bf16": None,
                     "note": "Hybrid state cache plus attention KV; exact footprint is runtime dependent."},
    }),
    (re.compile(r"minimax.*m3|mini.?max-m3", re.I), {
        "family": "MiniMax M3",
        "scale": "428B",
        "decoder_type": "Sparse MoE",
        "attention": "MSA sparse attention",
        "layer_mix": "MSA sparse attention + MoE",
        "features": {"moe": True, "ssm": False, "linear_attention": False, "gqa": None,
                     "mla": False, "sparse_attention": True, "mtp": None, "dense": False},
        "kv_cache": {"dtype": "bf16/fp8 depending runtime", "bytes_per_token_bf16": None,
                     "note": "Sparse-attention index/KV traffic is a first-class perf surface."},
    }),
    (re.compile(r"qwen3\.5|qwen35|397b", re.I), {
        "family": "Qwen3.5",
        "scale": "397B total / 17B active",
        "decoder_type": "Hybrid sparse MoE",
        "attention": "GDN linear attention + full attention",
        "layer_mix": "GDN/linear-attention blocks + MoE + full attention",
        "features": {"moe": True, "ssm": False, "linear_attention": True, "gqa": True,
                     "mla": False, "sparse_attention": False, "mtp": None, "dense": False},
        "kv_cache": {"dtype": "fp8_e4m3 in current scripts", "bytes_per_token_bf16": None,
                     "note": "GDN state and full-attention KV should be separated in analysis."},
    }),
    (re.compile(r"deepseek.*v[34]|deepseek.*r1", re.I), {
        "family": "DeepSeek",
        "decoder_type": "Sparse MoE",
        "attention": "MLA",
        "layer_mix": "MLA + MoE",
        "features": {"moe": True, "ssm": False, "linear_attention": False, "gqa": False,
                     "mla": True, "sparse_attention": False, "mtp": True, "dense": False},
        "kv_cache": {"dtype": "runtime dependent", "bytes_per_token_bf16": None,
                     "note": "MLA reduces KV-cache bytes/token; long-context attention remains architecture-specific."},
    }),
    (re.compile(r"gpt.?oss", re.I), {
        "family": "GPT-OSS",
        "decoder_type": "Sparse MoE",
        "attention": "GQA with RoPE",
        "layer_mix": "MoE decoder blocks",
        "features": {"moe": True, "ssm": False, "linear_attention": False, "gqa": True,
                     "mla": False, "sparse_attention": False, "mtp": None, "dense": False},
        "kv_cache": {"dtype": "runtime dependent", "bytes_per_token_bf16": None, "note": "GQA KV cache."},
    }),
    (re.compile(r"llama|mistral|gemma|olmo|phi|smollm", re.I), {
        "decoder_type": "Dense",
        "attention": "GQA/MHA with RoPE or variant",
        "layer_mix": "Dense attention + MLP blocks",
        "features": {"moe": False, "ssm": False, "linear_attention": False, "gqa": True,
                     "mla": False, "sparse_attention": False, "mtp": False, "dense": True},
        "kv_cache": {"dtype": "runtime dependent", "bytes_per_token_bf16": None, "note": "Dense decoder KV cache."},
    }),
]


def infer_architecture(model_id: str | None, manifest_model: dict | None = None) -> dict:
    model_id = model_id or ""
    out = {
        "id": model_id or "unknown",
        "family": "unknown",
        "quantization": "unknown",
        "context_tokens": None,
        **UNKNOWN_ARCH,
    }
    for rx, desc in _DESCRIPTORS:
        if rx.search(model_id):
            merged = dict(out)
            merged.update(desc)
            merged["features"] = {**UNKNOWN_ARCH["features"], **desc.get("features", {})}
            merged["kv_cache"] = {**UNKNOWN_ARCH["kv_cache"], **desc.get("kv_cache", {})}
            out = merged
            break
    if manifest_model:
        out = _deep_merge(out, manifest_model)
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out
