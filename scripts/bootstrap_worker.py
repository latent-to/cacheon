"""Prepare a replacement host's model, pinned image and controller environments.

Run from an exported Cacheon source tree on the worker. Commissioning remains
the existing deployment entrypoint and consumes the paths printed here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


_DOWNLOAD = """import json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo,revision,model,token,receipt=sys.argv[1:]
root=Path(model)
files=json.loads(Path(receipt).read_bytes())['files']
if not all((root/r['path']).is_file() and (root/r['path']).stat().st_size==r['size'] for r in files):
    snapshot_download(repo_id=repo,revision=revision,local_dir=model,
                      token=Path(token).read_text().strip(),max_workers=16)
"""


def run(argv: list[str], log: Path, *, env: dict[str, str], timeout: int) -> None:
    """Keep complete subprocess output without putting token contents in argv."""
    with log.open("ab", buffering=0) as output:
        subprocess.run(argv, env=env, stdout=output, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True, timeout=timeout)


def environment(uv: str, root: Path, name: str, env: dict[str, str]) -> str:
    """Reuse the interpreter while uv reconciles the requested package pins."""
    python = root / name / "bin/python"
    if not python.is_file():
        run([uv, "venv", "--python", "/usr/bin/python3.12", str(root / name)],
            root / "logs" / f"{name}.log", env=env, timeout=120)
    return str(python)


def provision_model(args: argparse.Namespace, uv: str, env: dict[str, str]) -> None:
    """Resume incomplete downloads; full content verification runs before completion."""
    python = environment(uv, args.root, "download-venv", env)
    log = args.root / "logs/download.log"
    run([uv, "pip", "install", "--python", python, "huggingface_hub==2.2.0"],
        log, env=env, timeout=600)
    run([python, "-c", _DOWNLOAD, args.model_repo, args.model_revision,
         str(args.model_dir), str(args.token_file), str(args.model_receipt)],
        log, env=env, timeout=args.download_timeout)
    cache = args.model_dir / ".cache"
    if cache.exists():
        destination = args.root / "model-download-cache"
        if destination.exists():
            raise RuntimeError(f"download cache destination already exists: {destination}")
        shutil.move(str(cache), destination)
    for path in (args.model_dir, *args.model_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"model tree contains a symlink: {path}")
        path.chmod(path.stat().st_mode & ~0o222)


def mount_model_ram(args: argparse.Namespace, expected: dict) -> None:
    """Keep model reconstruction off the disk used to unpack Docker layers."""
    size = sum(row["size"] for row in expected["files"])
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if available < 2 * size:
        raise RuntimeError("RAM model staging needs twice the model size available before loading")
    args.model_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    if args.model_dir.is_mount():
        kind = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target",
                                        str(args.model_dir)], text=True).strip()
        if kind != "tmpfs":
            raise RuntimeError("--ram-model requires an existing tmpfs or an empty directory")
    else:
        if any(args.model_dir.iterdir()):
            raise RuntimeError("refusing to hide existing model files under a RAM mount")
        subprocess.run(["mount", "-t", "tmpfs", "-o",
                        f"size={size + (1 << 30)},nosuid,nodev,mode=0755",
                        "tmpfs", str(args.model_dir)], check=True, timeout=30)


def provision_image(args: argparse.Namespace, env: dict[str, str]) -> str:
    """Pull the requested immutable image only when Docker lacks it locally."""
    subprocess.run(["docker", "info", "--format", "{{.ID}}"], check=True,
                   stdout=subprocess.DEVNULL, timeout=30, env=env)
    command = ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"]
    existing = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
    if existing.returncode:
        run(["docker", "pull", args.image], args.root / "logs/image.log",
            env=env, timeout=args.download_timeout)
    return subprocess.check_output(command, text=True, timeout=30, env=env).strip()


def provision_environments(args: argparse.Namespace, uv: str, env: dict[str, str]) -> str:
    """Install the existing CPU controller and sealed replay client's dependencies."""
    python = environment(uv, args.root, "controller-venv", env)
    log = args.root / "logs/controller-venv.log"
    run([uv, "pip", "install", "--python", python, "torch==2.8.0",
         "--index-url", "https://download.pytorch.org/whl/cpu"], log, env=env, timeout=900)
    run([uv, "pip", "install", "--python", python, "PyNaCl>=1.6.2,<2",
         "aiohttp>=3.9,<4", "transformers==5.19.0", "setuptools>=68"],
        log, env=env, timeout=600)
    replay = environment(uv, args.root, "aiperf-venv", env)
    run([uv, "pip", "install", "--python", replay, "aiperf==0.13.0"],
        args.root / "logs/aiperf-venv.log", env=env, timeout=900)
    return python


def main() -> None:
    """Populate host inputs and print their verified identities for commissioning."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model-dir", "model-receipt", "token-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model-repo", "model-revision", "image"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--download-timeout", type=int, default=7200)
    parser.add_argument("--ram-model", action="store_true",
                        help="download directly into a model-sized tmpfs (requires root)")
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        parser.error("--image must be the commissioned immutable repository digest")
    for name in ("root", "model_dir", "model_receipt", "token_file"):
        setattr(args, name, getattr(args, name).resolve())
    args.root.mkdir(parents=True, exist_ok=True, mode=0o755)
    (args.root / "logs").mkdir(exist_ok=True, mode=0o700)
    probe = args.root / ".bootstrap-write-probe"
    with probe.open("xb") as output:
        output.write(b"bootstrap"); output.flush(); os.fsync(output.fileno())
    if probe.read_bytes() != b"bootstrap":
        raise RuntimeError("bootstrap filesystem probe changed")
    probe.unlink()
    expected = json.loads(args.model_receipt.read_bytes())
    if args.ram_model:
        mount_model_ram(args, expected)
    env = {**os.environ, "UV_CACHE_DIR": str(args.root / "uv-cache"),
           "HF_HOME": str(args.root / "hf-cache"), "PYTHONDONTWRITEBYTECODE": "1"}
    uv = args.root / "bin/uv"
    if not uv.is_file():
        installer = args.root / "uv-installer.sh"
        urllib.request.urlretrieve("https://astral.sh/uv/0.11.32/install.sh", installer)
        run(["sh", str(installer)], args.root / "logs/uv.log",
            env={**env, "UV_UNMANAGED_INSTALL": str(uv.parent), "UV_NO_MODIFY_PATH": "1"},
            timeout=300)
    with ThreadPoolExecutor(max_workers=3) as pool:
        model = pool.submit(provision_model, args, str(uv), env)
        image = pool.submit(provision_image, args, env)
        controller = pool.submit(provision_environments, args, str(uv), env)
        model.result()
        image_id, python = image.result(), controller.result()
    source = Path(__file__).resolve().parents[1]
    publication = args.root / "model-publication"
    publication.mkdir(exist_ok=True, mode=0o700)
    run([python, "-B", "-m", "cacheon.cli", "model-provision", str(args.model_dir),
         str(publication), "--expected-content-digest", expected["content_digest"],
         "--workers", "8"], args.root / "logs/model-provision.log",
        env={**env, "PYTHONPATH": str(source)}, timeout=1800)
    result = {"image": args.image, "image_id": image_id, "model_root": str(args.model_dir),
              "model_content_digest": expected["content_digest"],
              "model_publication": str(publication), "controller_python": python,
              "aiperf": str(args.root / "aiperf-venv/bin/aiperf")}
    complete = args.root / "bootstrap-complete.json"
    with complete.open("w") as output:
        json.dump(result, output, indent=2); output.write("\n")
        output.flush(); os.fsync(output.fileno())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
