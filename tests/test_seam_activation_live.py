"""Opt-in live proof that a bundle kernel executes in the serving path.

This tier boots a real sglang server three times inside a validator worker
image — null-armed baseline, armed with the exact-math silu bundle, armed
with the deliberately broken silu bundle — and compares one temperature-0
completion across the boots. Activation is log-silent by design, so the
behavioral probe is the only honest detector: the exact bundle must leave
the output byte-identical and the broken bundle must corrupt it. If the
broken arm matches baseline, bundles are not reaching the serving path at
all, which is the regression this test exists to catch.

The tier never runs by default — not in CI and not on GPU hosts. Arm it
explicitly:

    CACHEON_LIVE_SERVE_TESTS=1        opt-in master switch (required)
    CACHEON_SERVE_IMAGE=...           worker image ref (sglang + .pth seam)
    CACHEON_SERVE_MODEL=/path         small local model directory to mount
    CACHEON_SERVE_REPO=/path          docker-mountable copy of this repo
                                      (default: this checkout; set it when
                                      the checkout sits on a FUSE mount
                                      docker cannot bind from)
    CACHEON_SERVE_GPU=0               GPU index for --gpus device=N
    CACHEON_SERVE_BOOT_TIMEOUT_S=320  per-boot readiness budget

With the master switch set, a missing or broken prerequisite is a loud
failure, never a silent skip. Runtime is roughly ten minutes for the three
boots; the container runs with --network none and is probed in-namespace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPT = "List the first five prime numbers."
_CONTAINER_PORT = 31001

pytestmark = pytest.mark.skipif(
    os.environ.get("CACHEON_LIVE_SERVE_TESTS") != "1",
    reason="opt-in live serving tier: set CACHEON_LIVE_SERVE_TESTS=1 to run",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        pytest.fail(
            f"CACHEON_LIVE_SERVE_TESTS=1 is set but {name} is not; "
            "the live tier fails loudly on misconfiguration"
        )
    return value


def _run(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _serve_repo() -> Path:
    configured = os.environ.get("CACHEON_SERVE_REPO", "")
    repo = Path(configured) if configured else _REPO_ROOT
    if not (repo / "cacheon" / "seam.py").is_file():
        pytest.fail(f"CACHEON_SERVE_REPO={repo} does not look like a cacheon repo")
    if repo != _REPO_ROOT:
        # The mounted copy is what actually executes in-container; a stale
        # copy would test the wrong tree. content_hash ignores __pycache__
        # and .git, so this is an exact source-identity comparison.
        for rel in ("cacheon", "examples/miner_silu_torch",
                    "examples/miner_silu_broken_torch"):
            if content_hash(_REPO_ROOT / rel) != content_hash(repo / rel):
                pytest.fail(
                    f"CACHEON_SERVE_REPO copy is stale: {rel} differs from "
                    "this checkout; refresh the mountable copy"
                )
    for bundle in ("miner_silu_torch", "miner_silu_broken_torch"):
        stray = list((repo / "examples" / bundle).rglob("__pycache__"))
        if stray:
            pytest.fail(
                f"remove {stray[0]} before running: the bundle scan refuses "
                "unexpected directories"
            )
    return repo


def _boot(name: str, image: str, model: str, repo: Path, gpu: str,
          arming: dict[str, str]) -> None:
    args = [
        "docker", "run", "-d", "--name", name,
        "--gpus", f"device={gpu}", "--network", "none",
        "-v", f"{model}:/model:ro", "-v", f"{repo}:/repo:ro",
        "-e", "PYTHONPATH=/repo",
    ]
    for key, value in arming.items():
        args += ["-e", f"{key}={value}"]
    args += [
        image, "python3", "-m", "sglang.launch_server",
        "--model-path", "/model", "--host", "127.0.0.1",
        "--port", str(_CONTAINER_PORT),
        "--mem-fraction-static", "0.5", "--context-length", "2048",
        "--chunked-prefill-size", "2048", "--max-prefill-tokens", "2048",
        "--cuda-graph-max-bs", "2", "--max-running-requests", "4",
    ]
    started = _run(args)
    if started.returncode != 0:
        pytest.fail(f"docker run failed for {name}: {started.stderr.strip()}")

    budget = int(os.environ.get("CACHEON_SERVE_BOOT_TIMEOUT_S", "320"))
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        logs = _run(["docker", "logs", name])
        if "ready to roll" in logs.stdout + logs.stderr:
            return
        time.sleep(8)
    tail = "\n".join((_run(["docker", "logs", name]).stderr).splitlines()[-8:])
    pytest.fail(f"{name} did not become ready within {budget}s; log tail:\n{tail}")


_PROBE_CODE = """
import json, urllib.request
payload = {{
    "model": "/model",
    "messages": [{{"role": "user", "content": {prompt!r}}}],
    "temperature": 0,
    "max_tokens": 48,
}}
req = urllib.request.Request(
    "http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={{"Content-Type": "application/json"}},
)
body = json.loads(urllib.request.urlopen(req, timeout=60).read())
print(json.dumps(body["choices"][0]["message"]["content"]))
"""


def _probe(name: str) -> str:
    code = _PROBE_CODE.format(prompt=_PROMPT, port=_CONTAINER_PORT)
    result = _run(["docker", "exec", name, "python3", "-c", code], timeout=90)
    if result.returncode != 0:
        pytest.fail(f"probe failed in {name}: {result.stderr.strip()}")
    text = json.loads(result.stdout)
    if not isinstance(text, str) or not text.strip():
        pytest.fail(f"probe returned an empty completion in {name}")
    return text


def test_bundle_kernel_executes_in_the_serving_path() -> None:
    if shutil.which("docker") is None:
        pytest.fail("live serving tier is armed but docker is not on PATH")
    image = _required_env("CACHEON_SERVE_IMAGE")
    model = _required_env("CACHEON_SERVE_MODEL")
    gpu = os.environ.get("CACHEON_SERVE_GPU", "0")
    repo = _serve_repo()

    modes = (
        ("baseline", {}),
        ("armed_exact", {
            "CACHEON_BUNDLE_PATH": "/repo/examples/miner_silu_torch",
            "CACHEON_ACTIVE": "1",
        }),
        ("armed_broken", {
            "CACHEON_BUNDLE_PATH": "/repo/examples/miner_silu_broken_torch",
            "CACHEON_ACTIVE": "1",
        }),
    )
    outputs: dict[str, str] = {}
    for mode, arming in modes:
        name = f"cacheon-seam-live-{mode}-{uuid.uuid4().hex[:8]}"
        try:
            _boot(name, image, model, repo, gpu, arming)
            outputs[mode] = _probe(name)
        finally:
            _run(["docker", "rm", "-f", name])

    assert outputs["armed_exact"] == outputs["baseline"], (
        "the exact-math bundle changed temperature-0 output; the seam is "
        "altering results it must reproduce byte-identically:\n"
        f"baseline: {outputs['baseline']!r}\n"
        f"armed:    {outputs['armed_exact']!r}"
    )
    assert outputs["armed_broken"] != outputs["baseline"], (
        "the broken bundle left output unchanged, so bundle kernels are NOT "
        "executing in the serving path; activation is silently dead"
    )
