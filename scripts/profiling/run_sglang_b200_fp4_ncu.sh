#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-cudagraph}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-optima_sglang_b200_fp4_ncu_${MODE}_${STAMP}}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
RESULT_NAME="${RESULT_NAME:-ncu_serving_b200_fp4_${MODE}_${STAMP}}"
USE_MEGAMOE="${USE_MEGAMOE:-1}"
MEGAMOE_W4A4="${MEGAMOE_W4A4:-1}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-megamoe}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-${FP4_GEMM_RUNNER_BACKEND:-auto}}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.075}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MEGAMOE_MAX_TOKENS_PER_RANK="${MEGAMOE_MAX_TOKENS_PER_RANK:-8320}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

NCU_DEVICES="${NCU_DEVICES:-0}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-cutlass::device_kernel.*(float_e2m1|float_ue8m0|MixedInput|mixed)}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
NCU_SET="${NCU_SET:-roofline}"
NCU_SECTIONS="${NCU_SECTIONS:-MemoryWorkloadAnalysis,Occupancy}"
NCU_PROFILE_FROM_START="${NCU_PROFILE_FROM_START:-off}"
NCU_RANGE_FILTER="${NCU_RANGE_FILTER:-}"
NCU_CHECK_EXIT_CODE="${NCU_CHECK_EXIT_CODE:-no}"

if [[ "${USE_MEGAMOE}" == "1" ]]; then
  MOE_ARGS=(--moe-a2a-backend "${MOE_A2A_BACKEND}")
else
  MOE_ARGS=(--moe-runner-backend "${MOE_RUNNER_BACKEND}")
fi

case "${MODE}" in
  cudagraph)
    SGLANG_PROFILE_ARGS=()
    ;;
  nograph_nvtx)
    SGLANG_PROFILE_ARGS=(--disable-cuda-graph --enable-layerwise-nvtx-marker)
    ;;
  *)
    echo "MODE must be cudagraph or nograph_nvtx, got: ${MODE}" >&2
    exit 2
    ;;
esac

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size 64g \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e CUDA_HOME=/usr/local/cuda \
  -e HF_HOME=/root/.cache/huggingface \
  -e RESULT_NAME="${RESULT_NAME}" \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 \
  -e SGLANG_OPT_USE_JIT_NORM=1 \
  -e SGLANG_OPT_USE_JIT_INDEXER_METADATA=1 \
  -e SGLANG_OPT_USE_TOPK_V2=1 \
  -e SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1 \
  -e SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 \
  -e SGLANG_OPT_USE_FAST_MASK_EP=1 \
  -e SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1 \
  -e SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1 \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0 \
  -e NVSHMEM_DISABLE_IB=1 \
  -e SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1 \
  -e SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE="${USE_MEGAMOE}" \
  -e SGLANG_OPT_FIX_HASH_MEGA_MOE="${USE_MEGAMOE}" \
  -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="${MEGAMOE_MAX_TOKENS_PER_RANK}" \
  -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS="${MEGAMOE_W4A4}" \
  -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND="${MEGAMOE_W4A4}" \
  -e NCU_DEVICES="${NCU_DEVICES}" \
  -e NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX}" \
  -e NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP}" \
  -e NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT}" \
  -e NCU_SET="${NCU_SET}" \
  -e NCU_SECTIONS="${NCU_SECTIONS}" \
  -e NCU_PROFILE_FROM_START="${NCU_PROFILE_FROM_START}" \
  -e NCU_RANGE_FILTER="${NCU_RANGE_FILTER}" \
  -e NCU_CHECK_EXIT_CODE="${NCU_CHECK_EXIT_CODE}" \
  -v /root/models:/root/models \
  -v /root/.cache:/root/.cache \
  -v /root/optima_profile:/opt/optima_profile \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
mkdir -p /opt/optima_profile/results
NCU_ARGS=(
  --force-overwrite
  --target-processes all
  --devices "${NCU_DEVICES}"
  --profile-from-start "${NCU_PROFILE_FROM_START}"
  --graph-profiling node
  --check-exit-code "${NCU_CHECK_EXIT_CODE}"
  --kernel-name-base demangled
  --kernel-name "regex:${NCU_KERNEL_REGEX}"
  --launch-skip "${NCU_LAUNCH_SKIP}"
  --launch-count "${NCU_LAUNCH_COUNT}"
  --print-summary per-kernel
  --export "/opt/optima_profile/results/${RESULT_NAME}"
)
if [[ -n "${NCU_RANGE_FILTER}" ]]; then
  NCU_ARGS+=(--range-filter "${NCU_RANGE_FILTER}")
fi
if [[ -n "${NCU_SET}" ]]; then
  NCU_ARGS+=(--set "${NCU_SET}")
fi
IFS="," read -r -a sections <<< "${NCU_SECTIONS}"
for section in "${sections[@]}"; do
  if [[ -n "${section}" ]]; then
    NCU_ARGS+=(--section "${section}")
  fi
done
exec /usr/local/cuda/bin/ncu "${NCU_ARGS[@]}" \
  sglang serve \
    --trust-remote-code \
    --model-path '"${MODEL_PATH}"' \
    --tp '"${TP_SIZE}"' \
    '"${MOE_ARGS[*]}"' \
    --fp4-gemm-backend '"${FP4_GEMM_BACKEND}"' \
    --mem-fraction-static '"${MEM_FRACTION_STATIC}"' \
    --swa-full-tokens-ratio '"${SWA_FULL_TOKENS_RATIO}"' \
    --chunked-prefill-size '"${CHUNKED_PREFILL_SIZE}"' \
    --cuda-graph-max-bs '"${CUDA_GRAPH_MAX_BS}"' \
    --max-running-requests '"${MAX_RUNNING_REQUESTS}"' \
    --host 0.0.0.0 \
    --port '"${PORT}"' \
    '"${SGLANG_PROFILE_ARGS[*]}"' \
    '"${EXTRA_SGLANG_ARGS}"''
