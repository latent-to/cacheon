"""Baseline Harness — monkey-patched inference with KVCachePolicy.

Loads Qwen2.5-7B-Instruct and replaces each attention layer's forward()
with a shim that routes K/V storage and attention computation through a
KVCachePolicy (policy.write() + policy.attend()). The rest of the model
— embeddings, MLP, LayerNorm, RoPE, output projection — is untouched.

Three execution paths:

  _generate_reference()         Unpatched HF model, full-sequence
                                recompute each decode step (no KV cache).
                                Produces ground-truth token ids + logits.
                                Only used by verify().

  _score_sequence_patched()     Patched model, SINGLE forward pass over
                                the reference's token sequence. Returns
                                logits aligned 1:1 with the reference.
                                Both paths see identical tokens, so KL
                                is valid across all positions. (distil-
                                style evaluation.)

  run(policy, prompts)          Patched model, prefill + single-token
                                decode loop. Production path for baseline
                                (PassthroughPolicy) and miner evaluation.

PassthroughPolicy is NOT "no cache" — it is a Python reimplementation
of standard FP16 attention that satisfies the KVCachePolicy interface.
It is the identity baseline: same math as HF, just wired through the
policy hooks. Miners replace it with compressed/evicted alternatives.

Production scoring flow (Phase 3+):
  baseline = run(PassthroughPolicy, prompts)   # reference logits/memory/latency
  miner    = run(MinerPolicy,       prompts)   # miner logits/memory/latency
  score    = scoring.score(baseline, miner)    # KL gate + weighted delta
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from .passthrough import PassthroughPolicy
from .policy import AttentionOutput, CacheConfig, KVCachePolicy

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    output_texts: list[str]
    output_ids: list[list[int]]
    all_logits: list[torch.Tensor]  # per-prompt [num_generated, vocab_size]
    latency_s: float
    peak_memory_bytes: int
    policy_memory_bytes: int


# ---------------------------------------------------------------------------
# Monkey-patch factory
# ---------------------------------------------------------------------------


def _make_patched_forward(
    attn_module,
    policy: KVCachePolicy,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Return a replacement forward() for one attention layer.

    Keeps: Q/K/V projections, RoPE, output projection.
    Replaces: the attention matmul — routed through policy.write() + policy.attend().

    head counts are passed explicitly because newer transformers removed
    num_heads / head_dim as instance attributes on Qwen2Attention.
    """

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,  # ignored — policy handles masking
        position_ids=None,
        past_key_value=None,  # ignored — policy IS the cache
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # --- Q / K / V projections (unchanged) ---
        q = attn_module.q_proj(hidden_states)
        k = attn_module.k_proj(hidden_states)
        v = attn_module.v_proj(hidden_states)

        q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # --- RoPE — MUST happen before policy.write() ---
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # --- Positions for the policy ---
        positions = (
            position_ids.squeeze(0) if position_ids is not None else cache_position
        )

        # --- Policy: write K/V, attend with Q ---
        policy.write(k, v, layer_idx, positions)

        # During prefill (q_len > 1) the HF mask encodes causal + sliding-window
        # constraints and must be forwarded.  During single-token decode (q_len == 1)
        # HF builds a [1,1,1,1] mask that doesn't cover the policy's accumulated
        # KV cache, so we pass None and let the policy attend to all cached positions.
        mask_for_policy = attention_mask if q_len > 1 else None
        attn_out: AttentionOutput = policy.attend(
            q, layer_idx, attention_mask=mask_for_policy
        )

        # --- Reshape + output projection (unchanged) ---
        output = attn_out.output  # [bsz, heads, q_len, head_dim]
        output = output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        output = attn_module.o_proj(output)

        # newer transformers Qwen2DecoderLayer unpacks 2 values: (hidden_states, _)
        return output, None

    return patched_forward


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Harness:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        logger.info("Loading model %s on %s (%s)", model_name, device, dtype)
        self.device = torch.device(device)
        self.dtype = dtype

        # Use eager attention so the unpatched reference produces numerically
        # identical results to the passthrough policy's Python attention.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        cfg = self.model.config
        self._cache_config = CacheConfig(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            max_seq_len=getattr(cfg, "max_position_embeddings", 32768),
            dtype=dtype,
        )
        self._original_forwards: list = []

    # ---- patch / unpatch -------------------------------------------------

    def _patch_attention(self, policy: KVCachePolicy) -> None:
        self._original_forwards = []
        cfg = self._cache_config
        for idx, layer in enumerate(self.model.model.layers):
            self._original_forwards.append(layer.self_attn.forward)
            layer.self_attn.forward = _make_patched_forward(
                layer.self_attn,
                policy,
                idx,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
            )

    def _unpatch_attention(self) -> None:
        for idx, layer in enumerate(self.model.model.layers):
            layer.self_attn.forward = self._original_forwards[idx]
        self._original_forwards = []

    # ---- prompt formatting ------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        """Wrap raw text in the model's chat template so instruct-tuned
        models produce coherent outputs instead of garbage tokens."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---- single-prompt generation ----------------------------------------

    @torch.inference_mode()
    def _generate_one(self, prompt: str, max_new_tokens: int) -> dict:
        """Run prefill + decode for one prompt. Returns text, token ids, logits."""

        formatted = self._format_prompt(prompt)
        input_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"].to(
            self.device
        )
        prompt_len = input_ids.shape[1]

        all_logits: list[torch.Tensor] = []
        generated_ids: list[int] = []

        # --- Prefill: full prompt in one forward pass ---
        outputs = self.model(input_ids=input_ids, use_cache=False)
        next_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        all_logits.append(next_logits.squeeze(0))
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        # --- Decode: one token at a time ---
        # At step=0 we feed generated_ids[0] which sits at position prompt_len.
        for step in range(max_new_tokens - 1):
            if generated_ids[-1] == self.tokenizer.eos_token_id:
                break

            position_ids = torch.tensor([[prompt_len + step]], device=self.device)
            outputs = self.model(
                input_ids=next_token,
                position_ids=position_ids,
                use_cache=False,
            )
            next_logits = outputs.logits[:, -1, :]
            all_logits.append(next_logits.squeeze(0))
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text,
            "token_ids": generated_ids,
            "logits": torch.stack(
                all_logits, dim=0
            ).cpu(),  # [num_generated, vocab_size]
        }

    # ---- public API ------------------------------------------------------

    def run(
        self,
        policy: KVCachePolicy,
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> RunResult:
        """Run inference with the given policy across all prompts."""

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            gc.collect()
            mem_before = torch.cuda.memory_allocated(self.device)
        else:
            mem_before = 0

        self._patch_attention(policy)
        try:
            start = time.perf_counter()
            results = []
            for prompt in prompts:
                policy.setup(self._cache_config)
                result = self._generate_one(prompt, max_new_tokens)
                results.append(result)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
        finally:
            self._unpatch_attention()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            gc.collect()
            mem_after = torch.cuda.memory_allocated(self.device)
            peak_mem = torch.cuda.max_memory_allocated(self.device)
        else:
            mem_after = 0
            peak_mem = 0

        # Delta isolates what the policy added to GPU memory (KV cache).
        # Only the last prompt's cache survives (setup() resets per prompt),
        # but both baseline and miner see the same prompts in the same order.
        policy_mem = max(0, mem_after - mem_before)

        return RunResult(
            output_texts=[r["text"] for r in results],
            output_ids=[r["token_ids"] for r in results],
            all_logits=[r["logits"] for r in results],
            latency_s=elapsed,
            peak_memory_bytes=peak_mem,
            policy_memory_bytes=policy_mem,
        )

    def measure_latency_interleaved(
        self,
        policy_a: KVCachePolicy,
        policy_b: KVCachePolicy,
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> tuple[float, float]:
        """Measure latency fairly by interleaving per-prompt runs.

        For each prompt, policy_a and policy_b run back-to-back so both
        see the same CUDA allocator state.  Generated outputs are discarded
        — this is timing-only.  Memory and logits come from separate
        ``run()`` calls.

        Returns ``(policy_a_total_s, policy_b_total_s)``.
        """
        a_times: list[float] = []
        b_times: list[float] = []

        for prompt in prompts:
            # --- policy A ---
            policy_a.setup(self._cache_config)
            self._patch_attention(policy_a)
            try:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                self._generate_one(prompt, max_new_tokens)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                a_times.append(time.perf_counter() - t0)
            finally:
                self._unpatch_attention()

            # --- policy B ---
            policy_b.setup(self._cache_config)
            self._patch_attention(policy_b)
            try:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                self._generate_one(prompt, max_new_tokens)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                b_times.append(time.perf_counter() - t0)
            finally:
                self._unpatch_attention()

        return sum(a_times), sum(b_times)

    @torch.inference_mode()
    def _generate_reference(self, prompt: str, max_new_tokens: int) -> dict:
        """Reference generation: unpatched model, full sequence recomputed each step.

        The unpatched model has no KV cache, so we feed the entire growing
        sequence every step.  This is slower than single-token decode but
        is the only correct approach without a cache.

        Returns a dict with 'token_ids' and 'logits' (matching _generate_one).
        """
        formatted = self._format_prompt(prompt)
        all_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"].to(
            self.device
        )
        generated = []
        all_logits: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=all_ids, use_cache=False)
            next_logit = outputs.logits[:, -1, :]
            all_logits.append(next_logit.squeeze(0))
            next_id = next_logit.argmax(dim=-1)
            generated.append(next_id.item())
            if next_id.item() == self.tokenizer.eos_token_id:
                break
            all_ids = torch.cat([all_ids, next_id.unsqueeze(0)], dim=1)

        return {
            "token_ids": generated,
            "logits": torch.stack(all_logits, dim=0) if all_logits else torch.empty(0),
        }

    @torch.inference_mode()
    def _score_sequence_patched(
        self,
        prompt: str,
        generated_ids: list[int],
        policy: KVCachePolicy,
    ) -> torch.Tensor:
        """Run [prompt + generated_ids] through the patched model in one
        forward pass and return logits for the generated positions.

        Both paths see identical input tokens, so KL divergence is valid
        across ALL positions — no truncation at first-divergence needed.
        """
        formatted = self._format_prompt(prompt)
        prompt_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"].to(
            self.device
        )
        prompt_len = prompt_ids.shape[1]

        gen_tensor = torch.tensor([generated_ids], device=self.device)
        full_ids = torch.cat([prompt_ids, gen_tensor], dim=1)

        policy.setup(self._cache_config)
        self._patch_attention(policy)
        try:
            outputs = self.model(input_ids=full_ids, use_cache=False)
        finally:
            self._unpatch_attention()

        # logits[t] predicts token t+1.
        # Continuation logits at positions [prompt_len-1 .. prompt_len+gen_len-2]
        # predict the generated tokens at positions [prompt_len .. prompt_len+gen_len-1].
        gen_len = len(generated_ids)
        return outputs.logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :].squeeze(
            0
        )

    def score_policy_on_sequence(
        self,
        policy: KVCachePolicy,
        prompts: list[str],
        reference_ids: list[list[int]],
    ) -> list[torch.Tensor]:
        """Teacher-forced logits: run each prompt + reference tokens through
        the patched model in a single forward pass.

        Returns one ``[gen_len, vocab_size]`` tensor per prompt.  These
        logits are computed on the *exact same token sequence* as the
        baseline, eliminating autoregressive drift from KL measurement.
        """
        logits: list[torch.Tensor] = []
        for prompt, ref_ids in zip(prompts, reference_ids):
            lg = self._score_sequence_patched(prompt, ref_ids, policy)
            logits.append(lg.cpu())
        return logits

    def verify(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        kl_threshold: float = 1e-4,
    ) -> bool:
        """Verify passthrough policy matches unpatched HuggingFace output.

        Uses the distil-style approach: generate a token sequence with the
        unpatched reference, then feed that SAME sequence through the patched
        model in one forward pass.  Both paths see identical input tokens so
        KL divergence is valid across ALL positions — no truncation needed.
        """
        import torch.nn.functional as F

        logger.info("Verifying passthrough on %d prompts…", len(prompts))

        for i, prompt in enumerate(prompts):
            ref = self._generate_reference(prompt, max_new_tokens)
            ref_ids = ref["token_ids"]
            ref_logits = ref["logits"]

            if ref_logits.shape[0] == 0:
                logger.warning("  prompt %d: no tokens generated", i)
                return False

            policy = PassthroughPolicy()
            pat_logits = self._score_sequence_patched(prompt, ref_ids, policy)

            n = min(ref_logits.shape[0], pat_logits.shape[0])
            ref_f32 = torch.nan_to_num(ref_logits[:n].float(), nan=-1e4)
            pat_f32 = torch.nan_to_num(pat_logits[:n].float(), nan=-1e4)
            p = F.softmax(ref_f32, dim=-1).clamp(min=1e-12)
            q = F.softmax(pat_f32, dim=-1).clamp(min=1e-12)
            kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()

            ok = not (math.isnan(kl) or kl >= kl_threshold)
            status = "PASS" if ok else "FAIL"
            logger.info(
                "  prompt %d: %s  (KL=%.6f, %d tokens compared)",
                i,
                status,
                kl,
                n,
            )

            if not ok:
                logger.error(
                    "    KL %.6f (threshold %.6f) — monkey-patch has a bug",
                    kl,
                    kl_threshold,
                )
                return False

        logger.info(
            "Verification passed on all %d prompts (KL < %.1e).",
            len(prompts),
            kl_threshold,
        )
        return True
