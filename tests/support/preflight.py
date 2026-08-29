"""Runtime preflight builders shared across the OCI and B300 suites.

Five suites each carried a verbatim ~35-line ``RuntimePreflightReceipt``
builder and three more carried the matching ``RuntimePreflightFacts``
builder. Every divergence between the copies is fixture data — registry
name, worker byte counts, uid source, a digest label — so each stays a
keyword argument here; nothing below is a behavior mode. Suites whose
builders carry genuinely different data (the char-repeat digests in
``test_oci_session_worker_order`` and the manifest-sourced fields in
``test_oci_reference_session``) keep their own definitions on purpose.
"""

from __future__ import annotations

import os

from cacheon.eval.oci_session_protocol import RuntimePreflightFacts
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt

from tests.support.b300 import sha


def preflight_receipt(
    *,
    image: str,
    platform: str,
    worker: str,
    registry: str = "cacheon",
    uid: int | None = None,
    gid: int | None = None,
    python_executable: str = "/usr/local/bin/python3",
    sglang_version: str = "0.0.0.dev1+g56e290315",
    worker_file_count: int = 200,
    worker_total_bytes: int = 1_000_000,
    argv_label: str = "preflight-argv",
) -> RuntimePreflightReceipt:
    reference = f"registry.example/{registry}@sha256:" + image
    return RuntimePreflightReceipt(
        schema="cacheon-runtime-preflight-v2",
        requested_image=reference,
        image_digest=image,
        local_image_id="sha256:" + "a" * 64,
        repo_digests=(reference,),
        oci_platform="linux/amd64",
        platform_digest=platform,
        docker_binary="/usr/bin/docker",
        uid=max(1, os.getuid()) if uid is None else uid,
        gid=max(1, os.getgid()) if gid is None else gid,
        sglang_version=sglang_version,
        worker_distribution="cacheon-harness",
        worker_version="0.0.1",
        worker_distribution_digest=worker,
        worker_file_count=worker_file_count,
        worker_total_bytes=worker_total_bytes,
        python_implementation="cpython",
        python_executable=python_executable,
        python_version="3.12.0",
        python_abi="cpython-312-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
        package_versions=(),
        cudart_library="libcudart.so.13",
        cuda_visible_devices="",
        nvidia_visible_devices="void",
        security_argv_sha256=sha(argv_label),
    )


def preflight_facts(
    *, launch_digest: str, engine_config_digest: str, **changes: object
) -> RuntimePreflightFacts:
    values: dict[str, object] = {
        "launch_digest": launch_digest,
        "runtime_digest": sha("runtime"),
        "stack_digest": sha("stack"),
        "tree_digest": sha("tree"),
        "engine_config_digest": engine_config_digest,
        "worker_distribution_digest": sha("worker"),
        "model_revision_digest": sha("revision"),
        "model_manifest_digest": sha("manifest"),
        "model_content_digest": sha("content"),
        "sglang_version": "0.0.0.dev1+g56e290315",
        "gpu_architectures": ("sm120",),
        "topology_digest": sha("topology"),
        "loopback_only": True,
        "read_only_inputs": True,
        "private_writable_cache": True,
    }
    values.update(changes)
    return RuntimePreflightFacts(**values)  # type: ignore[arg-type]
