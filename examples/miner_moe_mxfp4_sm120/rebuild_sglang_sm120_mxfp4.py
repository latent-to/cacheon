#!/usr/bin/env python3
"""Patch SGLang 0.5.12.post1 for SM120 FlashInfer MXFP4 MoE.

This rebuild helper is bundled with the example miner so a fresh checkout can
reproduce the RTX Blackwell GPT-OSS MXFP4 win without committing the full
``experiments/`` worklog tree. It assumes FlashInfer 0.6.12 already has stock
SM120 ``cutlass_fused_moe`` support and only patches SGLang's GPT-OSS MXFP4
routing:

  - allow ``moe_runner_backend=flashinfer_mxfp4`` on SM120,
  - keep packed FP4 weight bytes plain,
  - interleave only MXFP4 block scales,
  - allocate GPT-OSS TP shards with MXFP4 block-ceil intermediate padding,
  - reorder GPT-OSS interleaved W13 rows into CUTLASS halved ``[up; gate]``,
  - use MXFP8 activation quantization with linear scale layout,
  - disable PDL for the RTX PRO 6000 Blackwell path.

Tested target:
  - sglang==0.5.12.post1
  - flashinfer-python==0.6.12
  - torch==2.12.0+cu130
  - RTX PRO 6000 Blackwell / sm_120a
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


def package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot find installed package: {package}")
    return Path(spec.origin).resolve().parent


def make_backup(path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve()).lstrip("/").replace("/", "__")
    backup = backup_dir / f"{key}.bak"
    if not backup.exists():
        shutil.copy2(path, backup)


def replace_once(text: str, old: str, new: str, *, path: Path, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{path}: replacement point not found for {label}")
    return text.replace(old, new, 1)


INIT_GATE_OLD = '''        # When `flashinfer_mxfp4` is enabled, dispatch to one of two FlashInfer
        # entry points depending on the GPU:
        #   - SM100 (Blackwell)  -> trtllm_fp4_block_scale_moe (existing)
        #   - SM90  (Hopper)     -> cutlass_fused_moe(use_w4_group_scaling=True)
        #                           (FlashInfer PR #3084, post-0.6.10)
        self._fi_kernel: Optional[str] = None
        if self.use_flashinfer:
            if is_sm100_supported():
                self._fi_kernel = "trtllm_sm100"
            elif is_sm90_supported():
                if not _FI_HAS_SM90_CUTLASS_MXFP4:
                    raise RuntimeError(
                        "moe_runner_backend=flashinfer_mxfp4 on SM90 requires the "
                        "interleave_moe_{weights,scales}_for_sm90_mixed_gemm helpers "
                        "from FlashInfer PR #3084 (>= 0.6.11). Upgrade flashinfer-python "
                        "or pick a different backend (e.g. marlin / triton_kernel)."
                    )
                self._fi_kernel = "cutlass_sm90"
            else:
                raise NotImplementedError(
                    "moe_runner_backend=flashinfer_mxfp4 requires SM90 or SM100."
                )
'''


INIT_GATE_NEW = '''        # When `flashinfer_mxfp4` is enabled, dispatch to a FlashInfer
        # entry point depending on the GPU:
        #   - SM120 (RTX Blackwell) -> cutlass_fused_moe(MXFP8 act + MXFP4 weight)
        #   - SM100 (datacenter BW) -> trtllm_fp4_block_scale_moe
        #   - SM90  (Hopper)        -> cutlass_fused_moe(use_w4_group_scaling=True)
        self._fi_kernel: Optional[str] = None
        if self.use_flashinfer:
            if is_sm120_supported():
                self._fi_kernel = "cutlass_sm120_mxfp8"
            elif is_sm100_supported():
                self._fi_kernel = "trtllm_sm100"
            elif is_sm90_supported():
                if not _FI_HAS_SM90_CUTLASS_MXFP4:
                    raise RuntimeError(
                        "moe_runner_backend=flashinfer_mxfp4 on SM90 requires the "
                        "interleave_moe_{weights,scales}_for_sm90_mixed_gemm helpers "
                        "from FlashInfer PR #3084 (>= 0.6.11). Upgrade flashinfer-python "
                        "or pick a different backend (e.g. marlin / triton_kernel)."
                    )
                self._fi_kernel = "cutlass_sm90"
            else:
                raise NotImplementedError(
                    "moe_runner_backend=flashinfer_mxfp4 requires SM90, SM100, or SM120."
                )
'''


SM120_PROCESS_METHOD = '''    def _process_weights_for_sm120_cutlass_mxfp8(self, layer):
        """Prepare GPT-OSS MXFP4 weights for SM120 FlashInfer CUTLASS MoE.

        FlashInfer 0.6.12's working SM120 path consumes MXFP8 activations and
        plain packed MXFP4 weights. It still needs MXFP4 block scales in the
        FlashInfer interleaved scale layout. GPT-OSS stores W13 rows interleaved
        as [gate_0, up_0, gate_1, up_1, ...], while this CUTLASS SwiGLU path
        consumes halved [up; gate].
        """
        sf_block_size = 32

        N_un = layer.w13_weight.shape[1] // 2
        K_un = layer.w13_weight.shape[2] * 2
        N_pad = self._padded_intermediate
        K_pad = self._padded_hidden
        E = layer.num_local_experts
        device = layer.w13_weight.device
        bias_dtype = layer.w13_weight_bias.dtype

        def _stack_up_gate_w13(unpadded, last_pad, last_un):
            gate_rows = unpadded[:, 0::2, :last_un]
            up_rows = unpadded[:, 1::2, :last_un]
            out = torch.zeros(E, 2 * N_pad, last_pad, dtype=unpadded.dtype, device=device)
            out[:, :N_un, :last_un] = up_rows
            out[:, N_pad : N_pad + N_un, :last_un] = gate_rows
            return out

        def _pad_w2_3d(unpadded, last_pad, last_un):
            out = torch.zeros(E, K_pad, last_pad, dtype=unpadded.dtype, device=device)
            out[:, :K_un, :last_un] = unpadded[:, :K_un, :last_un]
            return out

        def _interleave_scales(scale):
            return torch.stack(
                [
                    nvfp4_block_scale_interleave(scale[i].view(torch.uint8))
                    .reshape_as(scale[i])
                    .contiguous()
                    for i in range(scale.shape[0])
                ]
            ).contiguous()

        w13_padded = _stack_up_gate_w13(
            layer.w13_weight.data.view(torch.uint8), K_pad // 2, K_un // 2
        )
        w13_scale_padded = _stack_up_gate_w13(
            layer.w13_weight_scale.data.view(torch.uint8),
            K_pad // sf_block_size,
            K_un // sf_block_size,
        )

        w13_bias = torch.zeros(E, 2 * N_pad, dtype=bias_dtype, device=device)
        w13_bias[:, :N_un] = layer.w13_weight_bias.data[:, 1 : 2 * N_un : 2]
        w13_bias[:, N_pad : N_pad + N_un] = layer.w13_weight_bias.data[
            :, 0 : 2 * N_un : 2
        ]

        w2_padded = _pad_w2_3d(
            layer.w2_weight.data.view(torch.uint8), N_pad // 2, N_un // 2
        )
        w2_scale_padded = _pad_w2_3d(
            layer.w2_weight_scale.data.view(torch.uint8),
            N_pad // sf_block_size,
            N_un // sf_block_size,
        )
        w2_bias = torch.zeros(E, K_pad, dtype=bias_dtype, device=device)
        w2_bias[:, :K_un] = layer.w2_weight_bias.data[:, :K_un]

        layer.w13_weight = Parameter(w13_padded, requires_grad=False)
        layer.w2_weight = Parameter(w2_padded, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            _interleave_scales(w13_scale_padded), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            _interleave_scales(w2_scale_padded), requires_grad=False
        )
        layer.w13_weight_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_weight_bias = Parameter(w2_bias, requires_grad=False)

        layer.swiglu_alpha = Parameter(
            torch.full((E,), 1.702, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        layer.swiglu_beta = Parameter(
            torch.ones((E,), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        layer.swiglu_limit = Parameter(
            torch.full((E,), 7.0, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        layer.gemm1_global = Parameter(
            torch.ones((E,), dtype=torch.float32, device=device), requires_grad=False
        )
        layer.gemm2_global = Parameter(
            torch.ones((E,), dtype=torch.float32, device=device), requires_grad=False
        )

        torch.cuda.empty_cache()

'''


SM120_APPLY_METHOD = '''    def _apply_sm120_cutlass_mxfp8(self, layer, x, topk_output):
        """Run SM120 FlashInfer CUTLASS MoE with MXFP8 activations and MXFP4 weights."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker, select_experts

        try:
            from tvm_ffi import use_torch_stream
        except ImportError:
            from contextlib import nullcontext as use_torch_stream

        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = select_experts(
                hidden_states=x,
                router_logits=topk_output.router_logits,
                topk_config=topk_output.topk_config,
                num_token_non_padded=topk_output.num_token_non_padded,
                expert_location_dispatch_info=topk_output.expert_location_dispatch_info,
            )
        topk_weights = topk_output.topk_weights.to(torch.float32)
        topk_ids = topk_output.topk_ids

        origin_hidden_states_dim = x.shape[-1]
        with use_torch_stream():
            x_quant, x_scale = mxfp8_quantize(x, False, alignment=self._padded_hidden)
        x_scale = x_scale.reshape(x.shape[0], -1)
        assert x_quant.shape[-1] == self._padded_hidden

        with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
            symm_output = torch.empty(
                x_quant.shape[0],
                self._padded_hidden,
                dtype=torch.bfloat16,
                device=x_quant.device,
            )

        with use_torch_stream():
            output = flashinfer_cutlass_fused_moe(
                input=x_quant,
                input_sf=x_scale,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=layer.w13_weight.view(torch.long),
                fc2_expert_weights=layer.w2_weight.view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=[
                    layer.w13_weight_scale.view(torch.int32),
                    layer.gemm1_global,
                    layer.w2_weight_scale.view(torch.int32),
                    layer.gemm2_global,
                ],
                fc1_expert_biases=layer.w13_weight_bias,
                fc2_expert_biases=layer.w2_weight_bias,
                swiglu_alpha=layer.swiglu_alpha,
                swiglu_beta=layer.swiglu_beta,
                swiglu_limit=layer.swiglu_limit,
                swizzled_input_sf=False,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                use_mxfp8_act_scaling=True,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                enable_pdl=False,
                output=symm_output,
            )[0]

        return StandardCombineInput(
            hidden_states=output[:, :origin_hidden_states_dim].contiguous()
        )

'''


CREATE_CUTLASS_BLOCK_ORIGINAL = '''        elif self._fi_kernel == "cutlass_sm90":
            # cutlass mixed-input GEMM contraction dim K must be % 128 == 0
            # (interleave factor for MXFP4 group_size=32 is 4). The kernel
            # also expects ``fc1_expert_weights`` in halved ``[up; gate]``
            # layout, which means the padding boundary must fall on the
            # gate / up split.
            #
            # The mxfp4 weight loader (FusedMoE.weight_loader fast path) does
            # a NAIVE copy of HF's ``[2*intermediate_size, hidden_packed]``
            # tensor into the buffer's ``[:dim1, :dim2]`` slice. Padding the
            # buffer here would push the gate/up boundary, so HF's "up"
            # rows would land in the buffer's "gate" half and vice versa.
            # Marlin sidesteps this by not padding; we do the same and
            # rebuild a properly-padded buffer in
            # ``_process_weights_for_sm90_cutlass`` after the load completes.
            self._padded_intermediate = round_up(intermediate_size_per_partition, 128)
            self._padded_hidden = round_up(hidden_size, 128)
            # create_weights below uses the *unpadded* sizes so the loader's
            # naive-copy fast path is correct.
            intermediate_size_per_partition_after_pad = intermediate_size_per_partition
'''


CREATE_CUTLASS_BLOCK_V1 = '''        elif self._fi_kernel in ("cutlass_sm90", "cutlass_sm120_mxfp8"):
            # cutlass mixed-input GEMM contraction dim K must be % 128 == 0
            # (interleave factor for MXFP4 group_size=32 is 4). The kernel
            # also expects ``fc1_expert_weights`` in halved ``[up; gate]``
            # layout, which means the padding boundary must fall on the
            # gate / up split.
            #
            # The mxfp4 weight loader (FusedMoE.weight_loader fast path) does
            # a NAIVE copy of HF's ``[2*intermediate_size, hidden_packed]``
            # tensor into the buffer's ``[:dim1, :dim2]`` slice. Padding the
            # buffer here would push the gate/up boundary, so HF's "up"
            # rows would land in the buffer's "gate" half and vice versa.
            # Marlin sidesteps this by not padding; we do the same and
            # rebuild a properly-padded buffer in
            # ``_process_weights_for_sm90_cutlass`` after the load completes.
            self._padded_intermediate = round_up(intermediate_size_per_partition, 128)
            self._padded_hidden = round_up(hidden_size, 128)
            # create_weights below uses the *unpadded* sizes so the loader's
            # naive-copy fast path is correct.
            intermediate_size_per_partition_after_pad = intermediate_size_per_partition
'''


CREATE_CUTLASS_BLOCK_V2 = '''        elif self._fi_kernel in ("cutlass_sm90", "cutlass_sm120_mxfp8"):
            # cutlass mixed-input GEMM contraction dim K must be % 128 == 0
            # (interleave factor for MXFP4 group_size=32 is 4). The kernel
            # also expects ``fc1_expert_weights`` in halved ``[up; gate]``
            # layout, which means the padding boundary must fall on the
            # gate / up split.
            #
            # GPT-OSS MXFP4 checkpoints are stored in 32-value blocks. Under
            # TP=4, 2880 / 4 is 720, but the loader slices by ceil(blocks/tp),
            # so ranks need room for 23 blocks = 736 intermediate values. Use
            # that loader-padded size for the initial buffer, then rebuild a
            # 128-aligned CUTLASS buffer after load. This keeps the naive copy
            # correct while avoiding the original 360-vs-368 packed mismatch.
            if self._fi_kernel == "cutlass_sm120_mxfp8":
                moe_intermediate_size = extra_weight_attrs.get("moe_intermediate_size")
                moe_tp_size = max(int(getattr(layer, "moe_tp_size", 1)), 1)
                if moe_intermediate_size is not None:
                    total_blocks = int(moe_intermediate_size) // mxfp4_block
                    load_intermediate = (
                        (total_blocks + moe_tp_size - 1) // moe_tp_size
                    ) * mxfp4_block
                else:
                    load_intermediate = round_up(
                        intermediate_size_per_partition, mxfp4_block
                    )
                self._loaded_intermediate = load_intermediate
                self._padded_intermediate = round_up(load_intermediate, 128)
                self._padded_hidden = round_up(hidden_size, 128)
                intermediate_size_per_partition_after_pad = load_intermediate
            else:
                # SM90 keeps upstream's unpadded loader buffer and only pads in
                # ``_process_weights_for_sm90_cutlass`` after the load completes.
                self._padded_intermediate = round_up(
                    intermediate_size_per_partition, 128
                )
                self._padded_hidden = round_up(hidden_size, 128)
                intermediate_size_per_partition_after_pad = (
                    intermediate_size_per_partition
                )
'''


def patch_sglang_mxfp4(sglang_root: Path, backup_dir: Path) -> None:
    path = sglang_root / "srt/layers/quantization/mxfp4.py"
    make_backup(path, backup_dir)
    text = path.read_text()

    if (
        "cutlass_sm120_mxfp8" in text
        and "def _apply_sm120_cutlass_mxfp8" in text
        and "self._loaded_intermediate = load_intermediate" in text
    ):
        print(f"{path}: latest SM120 branch already patched")
        return

    if "cutlass_sm120_mxfp8" not in text:
        text = replace_once(
            text, INIT_GATE_OLD, INIT_GATE_NEW, path=path, label="SM120 init gate"
        )

    if CREATE_CUTLASS_BLOCK_V2 not in text:
        if CREATE_CUTLASS_BLOCK_V1 in text:
            text = text.replace(CREATE_CUTLASS_BLOCK_V1, CREATE_CUTLASS_BLOCK_V2, 1)
        else:
            text = replace_once(
                text,
                CREATE_CUTLASS_BLOCK_ORIGINAL,
                CREATE_CUTLASS_BLOCK_V2,
                path=path,
                label="SM120 create_weights TP load padding",
            )

    if "self._process_weights_for_sm120_cutlass_mxfp8(layer)" not in text:
        text = replace_once(
            text,
            '        if self._fi_kernel == "cutlass_sm90":\n'
            "            self._process_weights_for_sm90_cutlass(layer)\n"
            "            return\n",
            '        if self._fi_kernel == "cutlass_sm120_mxfp8":\n'
            "            self._process_weights_for_sm120_cutlass_mxfp8(layer)\n"
            "            return\n"
            '        if self._fi_kernel == "cutlass_sm90":\n'
            "            self._process_weights_for_sm90_cutlass(layer)\n"
            "            return\n",
            path=path,
            label="SM120 process branch",
        )

    if "def _process_weights_for_sm120_cutlass_mxfp8" not in text:
        text = replace_once(
            text,
            "    def _process_weights_for_sm90_cutlass(self, layer):\n",
            SM120_PROCESS_METHOD
            + "    def _process_weights_for_sm90_cutlass(self, layer):\n",
            path=path,
            label="SM120 process method",
        )

    if "def _apply_sm120_cutlass_mxfp8" not in text:
        text = replace_once(
            text,
            "    def _apply_sm90_cutlass(self, layer, x, topk_output):\n",
            SM120_APPLY_METHOD + "    def _apply_sm90_cutlass(self, layer, x, topk_output):\n",
            path=path,
            label="SM120 apply method",
        )

    if "return self._apply_sm120_cutlass_mxfp8(layer, x, topk_output)" not in text:
        text = replace_once(
            text,
            '        if self._fi_kernel == "cutlass_sm90":\n'
            "            return self._apply_sm90_cutlass(layer, x, topk_output)\n"
            "        if self.use_flashinfer:\n",
            '        if self._fi_kernel == "cutlass_sm120_mxfp8":\n'
            "            return self._apply_sm120_cutlass_mxfp8(layer, x, topk_output)\n"
            '        if self._fi_kernel == "cutlass_sm90":\n'
            "            return self._apply_sm90_cutlass(layer, x, topk_output)\n"
            "        if self.use_flashinfer:\n",
            path=path,
            label="SM120 apply branch",
        )

    path.write_text(text)
    print(f"{path}: patched latest SGLang SM120 FlashInfer MXFP4 path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("/tmp/optima_latest_sglang_sm120_patch_backups"),
        help="directory for one-time file backups before mutation",
    )
    args = parser.parse_args()

    sglang_root = package_root("sglang")
    print(f"sglang root: {sglang_root}")
    print(f"backup dir: {args.backup_dir}")
    patch_sglang_mxfp4(sglang_root, args.backup_dir)
    print("done")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"patch failed: {exc}", file=sys.stderr)
        raise
