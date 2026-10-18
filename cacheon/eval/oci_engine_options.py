"""Validator-owned serving options accepted by the isolated engine protocol.

Commissioned profiles select these options; candidate bundles cannot change
the engine configuration carried by their sealed launch. The value checkers
live here with the kinds table they serve; the session protocol binds them to
its own error type.
"""

import math
import re
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
        hicache_mem_layout hicache_io_backend dsv4_attn_backend mm_feature_transport
        reasoning_parser tool_call_parser speculative_draft_attention_backend
        speculative_draft_model_path
    """.split(), "token"),
    **dict.fromkeys("""
        disable_radix_cache enable_dp_attention enable_flashinfer_allreduce_fusion
        trust_remote_code enable_hierarchical_cache enable_decoder_swa_bounded_replay
        weight_loader_prefetch_checkpoints enable_layernorm_sp enable_megamoe_for_layernorm_sp
        speculative_use_block_verification
    """.split(), "bool"),
    **dict.fromkeys(("cuda_graph_bs", "cuda_graph_bs_decode"), "int_list"),
    # Resident sessions recapture CUDA graphs on a LIVE scheduler loop; the
    # default 300s watchdog kills the rank mid-capture (measured 2026-07-20).
    "watchdog_timeout": "positive_int",
    # Hybrid sliding-window models size their SWA KV pool as a ratio of the
    # full-attention pool; SGLang's fallback 0.8 starved the full pool at 24
    # sessions on 4xB300 Muse Spark (InferenceX run 38825818513, 2026-10-16).
    # The only float kwarg; the config digest spells it like mem_fraction_static.
    "swa_full_tokens_ratio": "unit_float",
}
# SGLang reads the DeepSeek-V4.1 Engram host-table switches from the environment,
# not ServerArgs; at TP2 the tables do not fit beside the weights in HBM. The
# Muse Spark DFlash2 draft sampler and the TVM FFI JIT architecture list are
# environment switches the same way. SGLang's Muse hook forces the unified
# cache's out-of-window sliding-window KV release off; with it off the SWA pool
# holds about three sessions of cached context, the prefix hit rate at 24
# sessions fell to 0.34 and per-user decode to 15.7 tok/s (Verda, 2026-10-18).
ENGINE_ENV_KINDS: Mapping[str, str] = dict.fromkeys(
    ("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT",
     "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL", "SGLANG_SPEC_TP_SYNC",
     "SGLANG_DEFAULT_THINKING", "SGLANG_DSV41_REASONING_EFFORT",
     "SGLANG_DFLASH_EAGER_DRAFT_SAMPLER", "TVM_FFI_CUDA_ARCH_LIST",
     "SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS"), "token")

_TOKEN = re.compile(r"[A-Za-z0-9_.:+/@-]{1,256}\Z")
_MODEL_SUBDIR = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}\Z")
_FLOAT_FIELDS = ("mem_fraction_static", "swa_full_tokens_ratio")


def _bool(value: object, *, field_name: str, error: type[Exception]) -> bool:
    if type(value) is not bool:
        raise error(f"{field_name} must be boolean")
    return value


def _bounded_int(value: object, *, field_name: str, minimum: int, maximum: int,
                 error: type[Exception]) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise error(f"{field_name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_float(value: object, *, field_name: str, minimum: float, maximum: float,
                   error: type[Exception]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise error(f"{field_name} must be finite and in [{minimum}, {maximum}]")
    return result


def _optional_token(value: object, *, field_name: str, error: type[Exception]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise error(f"{field_name} must be null or a bounded runtime token")
    return value


def _validate_options(value: object, kinds: Mapping[str, str], *, label: str,
                      error: type[Exception]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise error(f"{label} must be an object")
    unknown = set(value) - set(kinds)
    if unknown:
        raise error(f"{label} contains unsupported keys: {sorted(unknown)!r}")
    result: dict[str, object] = {}
    for key in sorted(value):
        item = value[key]
        kind = kinds[key]
        field_name = f"{label}.{key}"
        if kind == "bool":
            result[key] = _bool(item, field_name=field_name, error=error)
        elif kind == "positive_int":
            result[key] = _bounded_int(
                item, field_name=field_name, minimum=1, maximum=16_777_216, error=error
            )
        elif kind == "token":
            result[key] = _optional_token(item, field_name=field_name, error=error)
            if result[key] is None:
                raise error(f"{field_name} must not be null")
        elif kind == "unit_float":
            result[key] = _bounded_float(
                item, field_name=field_name, minimum=0.0, maximum=1.0, error=error
            )
            if result[key] == 0.0:
                raise error(f"{field_name} must be in (0, 1]")
        elif kind == "int_list":
            if (
                not isinstance(item, (list, tuple))
                or not 1 <= len(item) <= 128
                or any(
                    type(value) is not int or not 1 <= value <= 16_777_216
                    for value in item
                )
                or list(item) != sorted(set(item))
            ):
                raise error(f"{field_name} must be a strictly increasing bounded int array")
            result[key] = list(item)
        else:  # pragma: no cover - validator-owned table invariant
            raise AssertionError(f"unknown engine kwarg kind {kind!r}")
    return result


def _sealed_draft_path(value: object, *, root: str) -> bool:
    """A draft model is one directory under the model root, so the model
    manifest and content digest cover its bytes and no second mount exists."""
    prefix = root + "/"
    return isinstance(value, str) and value.startswith(prefix) and (
        _MODEL_SUBDIR.fullmatch(value[len(prefix):]) is not None
    )


def _digest_identity(identity: dict[str, object]) -> dict[str, object]:
    """Stack identities deliberately reject JSON floats: bind each binary64
    value through its shortest round-trippable decimal spelling."""
    identity["mem_fraction_static"] = format(identity["mem_fraction_static"], ".17g")  # type: ignore[arg-type]
    kwargs = identity["engine_kwargs"]
    if isinstance(kwargs, dict) and "swa_full_tokens_ratio" in kwargs:
        kwargs["swa_full_tokens_ratio"] = format(kwargs["swa_full_tokens_ratio"], ".17g")  # type: ignore[arg-type]
    return identity
