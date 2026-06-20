#!/usr/bin/env python3
"""Declarative kernel taxonomy for the profiler.

The ingest layer still exposes the historical ``KERNEL_CATS`` and ``DISPLAY``
objects for compatibility, but the source of truth is ``TAXONOMY``: ordered
category specs with UI/planning metadata. Order matters because the first regex
match wins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CategorySpec:
    id: str
    display: str
    regex: str
    component: str
    ownership: str = "unknown"       # ours | vendor | collective | runtime | unknown
    phase: str = "both"              # decode | prefill | both
    description: str = ""

    def compile(self) -> re.Pattern:
        return re.compile(self.regex, re.I)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display": self.display,
            "regex": self.regex,
            "component": self.component,
            "ownership": self.ownership,
            "phase": self.phase,
            "description": self.description,
        }


TAXONOMY: list[CategorySpec] = [
    CategorySpec("fp4_moe_gemm", "FP4 MoE GEMM (bmm_*E2m1)",
                 r"^bmm_(?:E2m1|Bfloat16_E2m1)", "moe", "vendor", "both"),
    CategorySpec("nvfp4_moe_gemm", "NVFP4 MoE grouped GEMM (cutlass GemmUniversal)",
                 r"GroupProblemShape|GemmUniversal<[^>]*Group|cutlass_moe_fp4|(?:nvfp4|nvf4|mxfp4).*grouped.*gemm",
                 "moe", "vendor", "both"),
    CategorySpec("dense_gemm", "dense/projection GEMM (nvjet)",
                 r"nvjet|cutlass.*gemm|gemm_", "dense_gemm", "vendor", "both"),
    CategorySpec("splitk_reduce", "splitKreduce GEMM epilogue",
                 r"splitKreduce", "dense_gemm", "vendor", "decode"),
    CategorySpec("moe_finalize", "MoE finalize",
                 r"finalizeKernel", "moe", "vendor", "both"),
    CategorySpec("moe_routing", "MoE routing",
                 r"routingIndices|routingInit|routingCustom|moe.*topk", "moe", "vendor", "both"),
    CategorySpec("all_reduce", "all-reduce (collective)",
                 r"allreduce_fusion|all_reduce|AllReduce|ncclDevKernel_AllReduce|one_shot|lamport",
                 "collective", "collective", "both"),
    CategorySpec("all_gather", "all-gather (collective)",
                 r"AllGather|_all_gather|all_gather", "collective", "collective", "both"),
    CategorySpec("msa_decode_attn", "MSA sparse attn (gqa_share / merge)",
                 r"_gqa_share_sparse_(?:decode|fwd)_kernel|_merge_topk_attn_out_kernel",
                 "attention", "ours", "both"),
    CategorySpec("msa_indexer_score", "MSA indexer score / block-score attn",
                 r"_decode_score_kernel|_decode_score_attn_kernel|_merge_attn_out_kernel|_flash_attn_fwd_with_block_sco",
                 "attention", "ours", "both"),
    CategorySpec("msa_topk", "MSA top-k (radix / bitonic)",
                 r"minimax_decode_topk|_topk_index(?:_partial|_merge)?_kernel",
                 "attention", "ours", "decode"),
    CategorySpec("msa_qknorm_rope", "MSA fused qknorm+RoPE glue",
                 r"fused_gemma_qknorm_rope|fused_qk_norm_rope|fused_parallel_qknorm",
                 "attention", "ours", "decode"),
    CategorySpec("msa_kv_insert", "MSA KV / index-K cache insert",
                 r"store_kv_index_kernel|fused_store_kv", "kv_cache", "ours", "decode"),
    CategorySpec("msa_moe_gemm", "MoE grouped GEMM (mxfp8/deep_gemm/bf16)",
                 r"m_grouped_gemm|fp8_gemm|deep_gemm|mxfp8.*gemm|grouped_gemm_.*fp8|fused_moe_kernel",
                 "moe", "vendor", "both"),
    CategorySpec("delay_stream", "delayStreamKernel",
                 r"delayStreamKernel", "runtime", "runtime", "both"),
    CategorySpec("attention", "attention",
                 r"fmha|attention|mha|paged|flash_fwd|trtllm.*mha", "attention", "vendor", "both"),
    CategorySpec("ssm_scan_decode", "SSM scan - decode recurrence (selective_scan_update)",
                 r"selective_scan_update|selective_state_update", "ssm", "ours", "decode"),
    CategorySpec("ssm_chunk_prefill", "SSM chunk - prefill scan (chunk_*)",
                 r"chunk_scan|chunk_state|state_passing|chunk_cumsum|bmm_chunk", "ssm", "ours", "prefill"),
    CategorySpec("ssm_conv", "SSM causal conv1d",
                 r"causal_conv1d", "ssm", "ours", "both"),
    CategorySpec("gdn_scan", "GDN scan (recurrence)",
                 r"gdn_wide_vec|gated_delta|fused_recurrent|chunk_gated", "linear_attention", "ours", "both"),
    CategorySpec("gdn_conv", "GDN causal conv",
                 r"short_conv|conv1d_update", "linear_attention", "ours", "both"),
    CategorySpec("fused_qkvzba", "GDN qkvzba split/reshape",
                 r"fused_qkvzba|qkvzba", "linear_attention", "ours", "decode"),
    CategorySpec("act_mul", "act_and_mul / SiLU",
                 r"act_and_mul|silu|swiglu|sigmoid_gate", "activation", "ours", "both"),
    CategorySpec("rmsnorm", "norm (rms/layer)",
                 r"rmsnorm|RMSNorm|layer_norm|LayerNorm|norm_fwd", "normalization", "ours", "both"),
    CategorySpec("nvfp4_quant", "NVFP4 quant / scale-interleave",
                 r"nvfp4_quantize|NVFP4Quantize|quantize.*fp4|block_scale_interleave|scale.*interleave",
                 "quantization", "ours", "both"),
    CategorySpec("token_gather", "token gather / index",
                 r"vectorized_gather|gather_kernel|index_select|IndexKernel|gather", "runtime", "runtime", "both"),
    CategorySpec("kv_rope", "RoPE / KV write",
                 r"mrope|rope|set_kv_buffer|fp8_set_kv|reshape_and_cache", "kv_cache", "ours", "both"),
    CategorySpec("sampling", "sampling / logits",
                 r"argmax|sample|softmax|penalt|resolve_future_token|logits", "sampling", "runtime", "decode"),
    CategorySpec("copy_memset", "copy / memset",
                 r"Memcpy|Memset|memcpy|memset|\bcopy_|elementwise_copy", "memory", "runtime", "both"),
    CategorySpec("elementwise", "elementwise misc",
                 r"elementwise|triton_poi|FillFunctor|CUDAFunctor|float8_copy|add_kernel|mul_kernel",
                 "elementwise", "runtime", "both"),
]


KERNEL_CATS: dict[str, re.Pattern] = {spec.id: spec.compile() for spec in TAXONOMY}
DISPLAY: dict[str, str] = {spec.id: spec.display for spec in TAXONOMY}
DISPLAY["other"] = "other / uncategorised"


def categorize(name: str, kernel_cats: dict[str, re.Pattern] = KERNEL_CATS) -> str:
    for cat, rx in kernel_cats.items():
        if rx.search(name):
            return cat
    return "other"


def taxonomy_json() -> dict:
    by_id = {spec.id: spec.to_dict() for spec in TAXONOMY}
    by_id["other"] = {
        "id": "other",
        "display": DISPLAY["other"],
        "regex": "",
        "component": "unknown",
        "ownership": "unknown",
        "phase": "both",
        "description": "Uncategorised kernels; add taxonomy only after measurement shows the bucket matters.",
    }
    return {
        "version": 1,
        "ordered_categories": [spec.id for spec in TAXONOMY] + ["other"],
        "categories": by_id,
    }
