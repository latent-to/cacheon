"""Validator-owned serving options accepted by the isolated engine protocol.

Commissioned profiles select these options; candidate bundles cannot change
the engine configuration carried by their sealed launch.
"""

from typing import Mapping


ENGINE_KWARG_KINDS: Mapping[str, str] = {
    **dict.fromkeys("""
        chunked_prefill_size context_length dp_size ep_size max_prefill_tokens page_size
        speculative_num_steps speculative_eagle_topk speculative_num_draft_tokens hicache_size
        speculative_dspark_block_size cuda_graph_max_bs_decode swa_prefix_tails prefill_decode_interval
        cuda_graph_max_bs_prefill random_seed
    """.split(), "positive_int"),
    **dict.fromkeys("""
        cuda_graph_backend_prefill kv_cache_dtype quantization speculative_algorithm
        hicache_mem_layout hicache_io_backend dsv4_attn_backend
    """.split(), "token"),
    **dict.fromkeys("""
        disable_radix_cache enable_dp_attention enable_flashinfer_allreduce_fusion
        trust_remote_code enable_hierarchical_cache enable_decoder_swa_bounded_replay
        weight_loader_prefetch_checkpoints
    """.split(), "bool"),
    **dict.fromkeys(("cuda_graph_bs", "cuda_graph_bs_decode"), "int_list"),
    # Resident sessions recapture CUDA graphs on a LIVE scheduler loop; the
    # default 300s watchdog kills the rank mid-capture (measured 2026-07-20).
    "watchdog_timeout": "positive_int",
}
# SGLang reads the DeepSeek-V4.1 Engram host-table switches from the environment,
# not ServerArgs; at TP2 the tables do not fit beside the weights in HBM.
ENGINE_ENV_KINDS: Mapping[str, str] = dict.fromkeys(
    ("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT",
     "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL", "SGLANG_SPEC_TP_SYNC",
     "SGLANG_DEFAULT_THINKING", "SGLANG_DSV41_REASONING_EFFORT"), "token")
