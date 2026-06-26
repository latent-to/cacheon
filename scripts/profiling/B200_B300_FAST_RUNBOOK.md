# B200/B300 DeepSeek-V4-Flash Profiling Runbook

This is the fast path for a rented Blackwell pod. The first gate is Nsight
Compute counter access. If that fails, stop before downloading the model.

Sources checked on 2026-06-02:
- NVIDIA ERR_NVGPUCTRPERM guidance: counters require host enablement, or the
  profiled container must run with CAP_SYS_ADMIN when admin-only counters are
  enabled: https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters
- NVIDIA HGX AI Factory table: B200 SXM is listed at 180 GB/GPU and B300 SXM at
  288 GB/GPU; verify the rented pod with `nvidia-smi`.
- SGLang DeepSeek-V4 docs: `DeepSeek-V4-Flash` is 284B total / 13B active and
  the published single-node recipe is 4 GPUs for B200 / GB200 / GB300 / H200.
  The Blackwell max-throughput path uses MegaMoE W4A4.
- SGLang install docs recommend CUDA 13 images for B300/GB300: use the full
  `lmsysorg/sglang:dev-cu13` image for profiling unless it is broken on the host.
- Verda's B200/B300 software-stack blog says their VMs support Docker/NVIDIA
  container toolkit, allow HW counter profiling with NCU/Nsight/CUPTI, and set
  `NVreg_RestrictProfilingToAdminUsers=0`.

## Shadeform / Verda Notes

Shadeform launches VMs; its container-on-instance feature just starts a Docker
container on the VM. For profiling, prefer a plain VM and run Docker manually
over SSH so we control `--gpus`, `--ipc=host`, `--network=host`, and profiling
arguments. True Docker-in-Docker is not required.

For the Verda `2B300.60V` / Shadeform `B300x2` offer, expect the visible VRAM to
be whatever `nvidia-smi` reports. Verda's B300 page lists the `2B300.60V` VM as
about 525 GB aggregate VRAM even though B300 hardware is marketed as 288 GB/GPU;
the screenshot may show 576 GB. Trust `nvidia-smi` for memory budgeting.

Fast provider check after SSH:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
grep -E 'RmProfilingAdminOnly|RestrictProfiling' /proc/driver/nvidia/params || true
docker version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If Docker is not installed, stop and switch to an OS/template with Docker already
installed if available. Installing Docker plus NVIDIA container toolkit is possible
but burns rental time.

## 0. Copy Helpers

From local:

```bash
rsync -az --progress -e 'ssh -p <PORT>' scripts/profiling/ root@<HOST>:/root/optima_profile/scripts/profiling/
```

On the pod:

```bash
mkdir -p /root/optima_profile/{results,logs}
cd /root/optima_profile
chmod +x scripts/profiling/*.sh scripts/profiling/*.py
```

## 1. Abort Gate: NCU Counters

Run this before model download.

```bash
IMAGE=lmsysorg/sglang:dev-cu13 MODE=cap_sys_admin \
  /root/optima_profile/scripts/profiling/ncu_counter_preflight.sh
```

If it fails, try the stronger container permission once:

```bash
IMAGE=lmsysorg/sglang:dev-cu13 MODE=privileged \
  /root/optima_profile/scripts/profiling/ncu_counter_preflight.sh
```

If both fail with `ERR_NVGPUCTRPERM`, destroy the pod. That means the host/provider
has not exposed usable NVIDIA performance counters to this environment.

Also record:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
grep -E 'RmProfilingAdminOnly|RestrictProfiling' /proc/driver/nvidia/params || true
```

## 2. Model Download

Only after the NCU gate passes:

```bash
IMAGE=lmsysorg/sglang:dev-cu13 TOKEN_FILE=/home/shadeform/token \
  /root/optima_profile/scripts/profiling/download_deepseek_v4_flash.sh
```

## 3. Start Nsight Systems Server

Defaults are TP=2 on GPUs `0,1`, CUDA graph enabled, FP4, Blackwell MegaMoE W4A4,
and a 128 request CUDA graph size.

```bash
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=30000 MODE=cudagraph \
  IMAGE=lmsysorg/sglang:dev-cu13 \
  MODEL_PATH=/root/models/DeepSeek-V4-Flash-FP4 \
  /root/optima_profile/scripts/profiling/run_sglang_blackwell_fp4_nsys.sh
```

Wait until ready:

```bash
until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null; do sleep 10; done
```

If the server OOMs, retry once with:

```bash
MEM_FRACTION_STATIC=0.82 SWA_FULL_TOKENS_RATIO=0.05 CHUNKED_PREFILL_SIZE=4096
```

If a 2xB200 cannot load after that, stop. A 2xB300 should have substantially
more memory headroom.

## 4. Short End-to-End NSYS Capture

```bash
CONTAINER="$(docker ps --format '{{.Names}}' | grep optima_sglang_blackwell_fp4_nsys | head -1)"
CONTAINER="${CONTAINER}" PORT=30000 NUM_PROMPTS=128 MAX_CONCURRENCY=128 \
  RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=512 PROFILE_START_STEP=20 PROFILE_STEPS=40 \
  /root/optima_profile/scripts/profiling/bench_sglang_profile.sh
```

For the intended long-output steady-decode profile, capture only a bounded decode
window instead of waiting for all 128 requests to finish:

```bash
docker cp /root/optima_profile/scripts/profiling/steady_decode_profile.py \
  "${CONTAINER}:/opt/optima_profile/scripts/profiling/steady_decode_profile.py"
docker exec "${CONTAINER}" python3 /opt/optima_profile/scripts/profiling/steady_decode_profile.py \
  --base-url http://127.0.0.1:30000 \
  --model /root/models/DeepSeek-V4-Flash-FP4 \
  --num-requests 128 \
  --max-workers 128 \
  --output-tokens 16384 \
  --settle-s 90 \
  --capture-s 30 \
  --json-out /opt/optima_profile/results/steady_decode_128x16k.json
```

Stop the server after the profile is written:

```bash
docker stop "${CONTAINER}"
```

## 5. Targeted NCU Roofline / Occupancy / Memory

Start one profiled server. Keep NCU narrow or it will be slow.

```bash
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=30000 MODE=cudagraph \
  IMAGE=lmsysorg/sglang:dev-cu13 \
  MODEL_PATH=/root/models/DeepSeek-V4-Flash-FP4 \
  NCU_DEVICES=0 \
  NCU_SET=roofline \
  NCU_SECTIONS=MemoryWorkloadAnalysis,Occupancy \
  NCU_LAUNCH_COUNT=1 \
  /root/optima_profile/scripts/profiling/run_sglang_blackwell_fp4_ncu.sh
```

Trigger the same short workload:

```bash
CONTAINER="$(docker ps --format '{{.Names}}' | grep optima_sglang_blackwell_fp4_ncu | head -1)"
CONTAINER="${CONTAINER}" PORT=30000 NUM_PROMPTS=128 MAX_CONCURRENCY=128 \
  RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=512 PROFILE_START_STEP=20 PROFILE_STEPS=10 \
  /root/optima_profile/scripts/profiling/bench_sglang_profile.sh
```

If NCU captures the wrong kernel, rerun with a narrower `NCU_KERNEL_REGEX` based
on the NSYS top-kernel summary.

## 6. Export Everything Before Deleting

```bash
cd /root/optima_profile/results
for rep in *.nsys-rep; do
  nsys export --type sqlite --force-overwrite=true "${rep}" -o "${rep%.nsys-rep}.sqlite" || true
  nsys stats --force-export=true --report cuda_gpu_kern_sum "${rep}" > "${rep%.nsys-rep}_kernel_summary.txt" || true
done
OUT="/root/optima_profile_blackwell_$(date +%Y%m%d_%H%M%S).tar.gz"
if command -v pigz >/dev/null; then
  tar -I 'pigz -1' -cf "${OUT}" /root/optima_profile
else
  tar -czf "${OUT}" /root/optima_profile
fi
echo "${OUT}"
```

Download with rsync:

```bash
rsync -az --progress -e 'ssh -p <PORT>' root@<HOST>:/root/optima_profile_blackwell_*.tar.gz /tmp/
```

## Hardware Decision

2xB200 is worth trying if NCU counters pass, but treat it as experimental: SGLang
publishes 4-GPU V4-Flash recipes, not a guaranteed 2-GPU recipe. The model should
be memory-plausible in FP4/MXFP4, but the actual 128x16k token-pool headroom must
be proven by loading and starting the steady-decode workload.

2xB300 is more useful for this profiling goal. It has much more HBM per GPU than
B200, should tolerate larger token pools and MegaMoE workspace better, and still
profiles the Blackwell FP4/MegaMoE path you care about.
