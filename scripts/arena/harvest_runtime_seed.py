"""Harvest a warm runtime seed from one stock boot of the exact worker image on a free lane.

Every qualification lease mounts an empty tmpfs runtime cache and seeds it from a
validator-provisioned tree (``cacheon.eval.oci_backend._seed_runtime_cache``), so compiled kernels
never survive a lease, and a cold boot that recompiles them outran the per-engine boot bound (24 to
30 minutes on DeepSeek-V4.1-Flash, 933 s on Muse Spark). This tool boots the stock engine once in
the image with the arena's sealed flags and the production cache environment, a host directory
bound at the container cache root, waits for the engine to answer, stops it, and seals the
directory's regular files as the seed: symlinks and other non-regular entries are dropped (the
commissioner refuses them), the top-level families must be the ones the backend installs, every
entry is made read-only, and the receipt is written exactly as ``commission_worker.py`` recomputes
it, so a later commission accepts the seed unchanged. With ``--verify`` the engine boots a second
time from the seed and both boot times land in the harvest record beside the receipt.

The engine flags are the sealed engine config translated the way ``cacheon.eval.engine_worker``
translates it for ``sglang.Engine`` (same keys, same precedence); ``launch_server`` takes the same
fields as flags.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

CONTAINER_MODEL, CONTAINER_CACHE = "/cacheon/input/model", "/cacheon/runtime-cache"
# cacheon.eval.oci_backend._SEED_CACHE_FAMILIES, the families a lease installs; anything else is refused there.
FAMILIES = {"cuda", "flashinfer", "home", "huggingface", "sglang", "torch-extensions", "torchinductor",
            "triton", "triton-home", "xdg"}
CACHE_ENV = {"HOME": "home", "CUDA_CACHE_PATH": "cuda", "FLASHINFER_WORKSPACE_BASE": "flashinfer",
             "HF_HOME": "huggingface", "SGLANG_CACHE_DIR": "sglang", "TORCH_EXTENSIONS_DIR": "torch-extensions",
             "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton", "TRITON_HOME": "triton-home",
             "XDG_CACHE_HOME": "xdg"}
FIXED_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONUNBUFFERED": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "900", "TILELANG_DEFAULT_TARGET": "cuda"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def engine_kwargs(config: dict) -> dict:
    """Mirror cacheon.eval.engine_worker.engine_kwargs over the sealed engine config."""
    kwargs = {"model_path": CONTAINER_MODEL, "dtype": config["dtype"],
              "mem_fraction_static": config["mem_fraction_static"], "log_level": config["log_level"]}
    if config.get("attention_backend"):
        kwargs["attention_backend"] = config["attention_backend"]
    if config.get("disable_cuda_graph"):
        kwargs["disable_cuda_graph"] = True
    if config.get("deterministic"):
        kwargs["enable_deterministic_inference"] = True
    if config.get("tp_size"):
        kwargs["tp_size"] = int(config["tp_size"])
    if config.get("max_running_requests"):
        kwargs["max_running_requests"] = int(config["max_running_requests"])
    if config.get("moe_runner_backend"):
        kwargs["moe_runner_backend"] = config["moe_runner_backend"]
    if config.get("disable_custom_all_reduce"):
        kwargs["disable_custom_all_reduce"] = True
    owned = {"attention_backend", "deterministic", "disable_custom_all_reduce", "disable_cuda_graph", "dtype",
             "engine_env", "log_level", "mem_fraction_static", "moe_runner_backend", "tp_size", "max_running_requests"}
    kwargs.update({key: value for key, value in config.items() if key not in owned})
    return kwargs


def engine_flags(kwargs: dict) -> list[str]:
    """ServerArgs fields are the Engine kwargs; False and None are the defaults the Engine would also take."""
    flags: list[str] = []
    for key, value in kwargs.items():
        if key == "model_path" or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        flags += [flag] if value is True else [flag, str(value)]
    return flags


def gpus_free(gpus: list[str]) -> bool:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                   text=True).splitlines()
    used = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in rows if line.strip()}
    return all(used[gpu] <= 64 for gpu in gpus)


def boot(args, cache: Path, name: str, log: Path, flags: list[str], env: dict[str, str]) -> float:
    """One stock boot with the cache directory bound; returns the seconds to the first healthy answer."""
    work = cache.parent / (name + "-work")
    work.mkdir(mode=0o700)
    command = ["docker", "run", "-d", "--init", f"--name={name}", "--network=host", "--ipc=private",
               f"--shm-size={args.shm}", "--ulimit=memlock=-1:-1", f"--memory={args.memory}",
               f"--memory-swap={args.memory}", "--gpus", f"\"device={args.gpus}\"", "--workdir=/work"]
    if args.cpuset:
        command.append(f"--cpuset-cpus={args.cpuset}")
    command += [f"--env={key}={value}" for key, value in sorted(env.items())]
    command += ["--mount", f"type=bind,src={args.model},dst={CONTAINER_MODEL},readonly",
                "--mount", f"type=bind,src={cache},dst={CONTAINER_CACHE}",
                "--mount", f"type=bind,src={work},dst=/work",
                "--entrypoint=python3", args.image, "-m", "sglang.launch_server", "--model-path", CONTAINER_MODEL,
                "--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", "arena", *flags]
    (log.with_suffix(".argv.json")).write_text(json.dumps(command, indent=1) + "\n")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    with log.open("xb") as handle:
        follower = subprocess.Popen(["docker", "logs", "--follow", name], stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.boot_limit_seconds:
            state = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", name],
                                   capture_output=True, text=True).stdout.strip()
            if state != "true":
                raise SystemExit(f"{name} exited during boot; see {log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5) as response:
                    if response.status == 200:
                        return round(time.monotonic() - start, 1)
            except Exception:
                pass
            time.sleep(10)
        raise SystemExit(f"{name} not ready after {args.boot_limit_seconds} s; see {log}")
    finally:
        subprocess.run(["docker", "stop", "--time", "120", name], stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        follower.wait(timeout=60)
        for _ in range(60):
            if gpus_free(args.gpus.split(",")):
                break
            time.sleep(5)


def seal(cache: Path, seed: Path, image_id: str) -> tuple[dict, dict]:
    """Copy the regular files as the seed and write the receipt commission_worker recomputes."""
    dropped = {"symlinks": 0, "other": 0}
    for row in sorted(cache.rglob("*")):
        target = seed / row.relative_to(cache)
        if row.is_symlink():
            dropped["symlinks"] += 1
        elif row.is_dir():
            target.mkdir(mode=0o755, parents=True, exist_ok=True)
        elif row.is_file():
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            shutil.copyfile(row, target)
        else:
            dropped["other"] += 1
    families = sorted(path.name for path in seed.iterdir())
    unknown = set(families) - FAMILIES
    if unknown:
        raise SystemExit(f"the boot wrote cache families the backend does not install: {sorted(unknown)}")
    for path in [seed, *seed.rglob("*")]:
        path.chmod(path.stat().st_mode & ~0o222)
    files = sorted(path for path in seed.rglob("*") if path.is_file())
    if not files:
        raise SystemExit("the boot left no cache files to seed")
    tree = hashlib.sha256()
    for path in files:
        tree.update(f"{path.relative_to(seed)}\0{sha256(path)}\n".encode())
    receipt = {"image_id": image_id, "tree_sha256": tree.hexdigest(), "files": len(files),
               "bytes": sum(path.stat().st_size for path in files), "families": families,
               "kind": "harvested-stock-boot", "compiled_model_cache": True}
    return receipt, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-inputs", required=True, type=Path, help="examples/arena_inputs/<arena>")
    parser.add_argument("--image-proof", required=True, type=Path, help="durable-images.json of the worker image")
    parser.add_argument("--image", help="local image reference to boot; default: the proof's retained image")
    parser.add_argument("--model", required=True, type=Path, help="the commissioned checkpoint root on this host")
    parser.add_argument("--gpus", required=True, help="the free lane, e.g. 0,1,2,3")
    parser.add_argument("--cpuset", default="", help="optional host cpuset for the boot")
    parser.add_argument("--memory", default="900g")
    parser.add_argument("--shm", default="128g")
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--boot-limit-seconds", type=int, default=7200)
    parser.add_argument("--work", required=True, type=Path, help="new directory for the cache, logs and argv")
    parser.add_argument("--out", required=True, type=Path, help="new seed directory; <out>.json is its receipt")
    parser.add_argument("--verify", action="store_true", help="boot once more from the seed and record the time")
    args = parser.parse_args()
    os.umask(0o022)
    for path in (args.work, args.out, args.out.with_name(args.out.name + ".json")):
        if path.exists():
            raise SystemExit(f"{path} exists")
    if not args.out.is_absolute():
        raise SystemExit("--out must be absolute: the commissioner binds the seed by absolute path")
    proof = json.loads(args.image_proof.read_bytes())
    args.image = args.image or proof["images"][0]["image"]
    image_id = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
                                       text=True).strip()  # what bootstrap_worker records as image_id
    if not gpus_free(args.gpus.split(",")):
        raise SystemExit(f"GPUs {args.gpus} are not free")
    config = json.loads((args.arena_inputs / "engine-config.json").read_bytes())
    flags = engine_flags(engine_kwargs(config))
    env = {key: f"{CONTAINER_CACHE}/{family}" for key, family in CACHE_ENV.items()}
    env.update(FIXED_ENV)
    env.update(config.get("engine_env", {}))
    args.work.mkdir(mode=0o700, parents=True)
    cache = args.work / "cache"
    cache.mkdir()
    cold = boot(args, cache, f"cacheon-seed-harvest-{args.port}", args.work / "engine-cold.log", flags, env)
    args.out.mkdir(mode=0o755, parents=True)
    receipt, dropped = seal(cache, args.out, image_id)
    record = {"schema": "cacheon-runtime-seed-harvest-v1", "image": args.image, "image_proof": str(args.image_proof),
              "engine_config_sha256": sha256(args.arena_inputs / "engine-config.json"), "gpus": args.gpus,
              "cold_boot_seconds": cold, "dropped": dropped, "receipt": receipt}
    if args.verify:
        seeded = args.work / "cache-seeded"
        shutil.copytree(args.out, seeded, symlinks=False)
        for path in [seeded, *seeded.rglob("*")]:
            path.chmod(path.stat().st_mode | 0o200)
        record["seeded_boot_seconds"] = boot(args, seeded, f"cacheon-seed-verify-{args.port}",
                                             args.work / "engine-seeded.log", flags, env)
    with args.out.with_name(args.out.name + ".json").open("x") as handle:
        handle.write(json.dumps(receipt) + "\n")
    args.out.with_name(args.out.name + ".json").chmod(0o444)
    (args.work / "harvest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"seed": str(args.out), "files": receipt["files"], "bytes": receipt["bytes"],
                      "families": receipt["families"], "tree_sha256": receipt["tree_sha256"], "cold_boot_seconds": cold,
                      "seeded_boot_seconds": record.get("seeded_boot_seconds"), "dropped": dropped}))


if __name__ == "__main__":
    main()
