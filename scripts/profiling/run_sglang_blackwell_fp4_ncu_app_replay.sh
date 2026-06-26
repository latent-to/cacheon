#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-optima_ncu_app_replay_${STAMP}}"
RESULT_NAME="${RESULT_NAME:-ncu_app_replay_${STAMP}}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13}"
MODE="${MODE:-nograph_nvtx}"

NCU_DEVICES="${NCU_DEVICES:-0}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-deep_gemm::sm100_fp8_fp4_mega_moe_impl}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
NCU_SECTIONS="${NCU_SECTIONS:-SpeedOfLight,SpeedOfLight_RooflineChart,MemoryWorkloadAnalysis,Occupancy}"
NCU_CHECK_EXIT_CODE="${NCU_CHECK_EXIT_CODE:-no}"

USE_MEGAMOE="${USE_MEGAMOE:-1}"
MEGAMOE_W4A4="${MEGAMOE_W4A4:-1}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-megamoe}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-auto}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.075}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MEGAMOE_MAX_TOKENS_PER_RANK="${MEGAMOE_MAX_TOKENS_PER_RANK:-8320}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

SHORT_NUM_PROMPTS="${SHORT_NUM_PROMPTS:-16}"
SHORT_CONCURRENCY="${SHORT_CONCURRENCY:-16}"
SHORT_INPUT_LEN="${SHORT_INPUT_LEN:-512}"
SHORT_OUTPUT_LEN="${SHORT_OUTPUT_LEN:-64}"
PROFILE_START_STEP="${PROFILE_START_STEP:-2}"
PROFILE_STEPS="${PROFILE_STEPS:-8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run \
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
  -e PORT="${PORT}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e MODE="${MODE}" \
  -e USE_MEGAMOE="${USE_MEGAMOE}" \
  -e MEGAMOE_W4A4="${MEGAMOE_W4A4}" \
  -e MOE_A2A_BACKEND="${MOE_A2A_BACKEND}" \
  -e MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND}" \
  -e FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND}" \
  -e CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS}" \
  -e MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
  -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  -e SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO}" \
  -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}" \
  -e EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS}" \
  -e SHORT_NUM_PROMPTS="${SHORT_NUM_PROMPTS}" \
  -e SHORT_CONCURRENCY="${SHORT_CONCURRENCY}" \
  -e SHORT_INPUT_LEN="${SHORT_INPUT_LEN}" \
  -e SHORT_OUTPUT_LEN="${SHORT_OUTPUT_LEN}" \
  -e PROFILE_START_STEP="${PROFILE_START_STEP}" \
  -e PROFILE_STEPS="${PROFILE_STEPS}" \
  -e WARMUP_REQUESTS="${WARMUP_REQUESTS}" \
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
  -v /root/models:/root/models \
  -v /root/.cache:/root/.cache \
  -v /root/optima_profile:/opt/optima_profile \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
mkdir -p /opt/optima_profile/results /opt/optima_profile/logs
NCU_ARGS=(
  --force-overwrite
  --target-processes all
  --devices "'"${NCU_DEVICES}"'"
  --replay-mode application
  --profile-from-start off
  --graph-profiling node
  --check-exit-code "'"${NCU_CHECK_EXIT_CODE}"'"
  --kernel-name-base demangled
  --kernel-name "regex:'"${NCU_KERNEL_REGEX}"'"
  --launch-skip "'"${NCU_LAUNCH_SKIP}"'"
  --launch-count "'"${NCU_LAUNCH_COUNT}"'"
  --print-summary per-kernel
  --export "/opt/optima_profile/results/'"${RESULT_NAME}"'"
)
IFS="," read -r -a sections <<< "'"${NCU_SECTIONS}"'"
for section in "${sections[@]}"; do
  if [[ -n "${section}" ]]; then
    NCU_ARGS+=(--section "${section}")
  fi
done
exec /usr/local/cuda/bin/ncu "${NCU_ARGS[@]}" /opt/optima_profile/scripts/profiling/ncu_bounded_sglang_target.sh'

docker logs "${NAME}" > "/root/optima_profile/logs/${NAME}.docker.log" 2>&1 || true
docker rm "${NAME}" >/dev/null 2>&1 || true
echo "NCU_APP_REPLAY_DONE name=${NAME} result=${RESULT_NAME}"
