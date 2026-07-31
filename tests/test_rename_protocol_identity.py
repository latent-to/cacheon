from __future__ import annotations

import ast
from pathlib import Path

from cacheon.artifact_device_launch import (
    DEVICE_ARTIFACT_ADMISSION_SCHEMA,
    DEVICE_LAUNCH_PLAN_SCHEMA,
)
from cacheon.artifact_identity import (
    DIRECT_ARTIFACT_ENTRY,
    DIRECT_ARTIFACT_IDENTITY_SCHEMA,
)
from cacheon.artifact_provider import CUTE_CUBIN_PATCHER_ID
from cacheon.cuda_cubin import CUDA_CUBIN_ABI_SCHEMA, CUDA_CUBIN_CONTRACT_SCHEMA
from cacheon.discovery import (
    DISCOVERY_ABI_VERSION,
    DISCOVERY_OVERLAY_SCHEMA,
)
from cacheon.eval.qualification import (
    GRAPH_EVIDENCE_MEDIA_TYPE,
    GRAPH_EVIDENCE_SCHEMA,
)
from cacheon.eval.runtime_preflight import (
    CONTAINER_RECEIPT_SCHEMA,
    HOST_RECEIPT_SCHEMA,
    PLATFORM_DIGEST_SCHEMA,
    WORKER_DIGEST_SCHEMA,
    WORKER_DISTRIBUTION,
)
from cacheon.eval.oci_session_protocol import SESSION_SCHEMA


ROOT = Path(__file__).resolve().parents[1]


def test_rename_keeps_legacy_protocol_and_runtime_abi_vocabulary() -> None:
    assert DIRECT_ARTIFACT_ENTRY == "_optima_direct_artifact"
    assert DIRECT_ARTIFACT_IDENTITY_SCHEMA == "optima.direct-artifact-execution.v1"
    assert CUTE_CUBIN_PATCHER_ID == "optima.build-cute-cubin.v1"
    assert DEVICE_LAUNCH_PLAN_SCHEMA == "optima.device-launch-plan.v1"
    assert DEVICE_ARTIFACT_ADMISSION_SCHEMA == "optima.device-artifact-admission.v1"
    assert CUDA_CUBIN_ABI_SCHEMA == "optima.cuda-cubin-abi.v1"
    assert CUDA_CUBIN_CONTRACT_SCHEMA == "optima.cuda-cubin-contract.v1"
    assert DISCOVERY_ABI_VERSION == "optima-discovery-abi-v1"
    assert DISCOVERY_OVERLAY_SCHEMA == "optima.discovery-overlay.v1"
    assert GRAPH_EVIDENCE_MEDIA_TYPE == "application/vnd.optima.graph-verification+json"
    assert GRAPH_EVIDENCE_SCHEMA == "optima.qualification.graph-raw-evidence.v1"
    assert SESSION_SCHEMA == "optima-isolated-engine-session-v1"
    assert CONTAINER_RECEIPT_SCHEMA == "optima-runtime-container-preflight-v2"
    assert HOST_RECEIPT_SCHEMA == "optima-runtime-preflight-v2"
    assert WORKER_DIGEST_SCHEMA == "optima-installed-distribution-v1"
    assert PLATFORM_DIGEST_SCHEMA == "optima-runtime-platform-v1"

    # Distribution and Python package names are branding, not protocol vocabulary.
    assert WORKER_DISTRIBUTION == "cacheon-harness"


def test_rename_does_not_rotate_literal_canonical_digest_domains() -> None:
    rotated: list[str] = []
    for path in sorted((ROOT / "cacheon").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            else:
                continue
            domain = node.args[0]
            if (
                function_name == "canonical_digest"
                and isinstance(domain, ast.Constant)
                and isinstance(domain.value, str)
                and domain.value.startswith("cacheon.")
            ):
                rotated.append(f"{path.relative_to(ROOT)}:{domain.lineno}: {domain.value}")
    assert rotated == []


def test_rename_keeps_serialized_native_names_and_cryptographic_salts() -> None:
    engine_tree = (ROOT / "cacheon" / "engine_tree.py").read_text(encoding="utf-8")
    marginal_runtime = (ROOT / "cacheon" / "eval" / "marginal_runtime.py").read_text(
        encoding="utf-8"
    )
    seam = (ROOT / "cacheon" / "seam.py").read_text(encoding="utf-8")
    cute_aot = (ROOT / "cacheon" / "cute_aot.py").read_text(encoding="utf-8")
    cuda_ext = (ROOT / "cacheon" / "patchers" / "build_cuda_ext.py").read_text(
        encoding="utf-8"
    )
    qualification = (ROOT / "cacheon" / "eval" / "qualification.py").read_bytes()
    oci_process = (ROOT / "cacheon" / "eval" / "oci_process.py").read_bytes()

    assert "optima_c_{" in engine_tree
    assert "cacheon_c_{" not in engine_tree
    assert "optima_c_{" in marginal_runtime
    assert "optima_c_[0-9a-f]{64}" in seam
    assert "optima_cute_{digest}" in cute_aot
    assert "optima_cuda_{artifact_id}" in cuda_ext
    assert b"optima-selection-secret-v1\\0" in qualification
    assert b"optima.oci-quiescence.v1\\0" in oci_process
    assert b"optima.oci-namespace.v1\\0" in oci_process


def test_rename_keeps_signed_release_evidence_vocabulary() -> None:
    source = (ROOT / "cacheon" / "release.py").read_text(encoding="utf-8")

    for marker in (
        '"creators": ["Tool: optima-release-v1"]',
        '"documentNamespace": "urn:optima:engine-release:"',
        '"name": "Optima Engine "',
        '"name": "optima-engine-tree"',
        '"name": "optima-native-artifact"',
        '"buildType": "https://optima.engine/build/v1"',
        '"id": "https://optima.engine/builder/v1"',
        '"uri": "optima:model-provision-receipt"',
        '"uri": "optima:native-artifact"',
    ):
        assert marker in source

    # The generated runtime itself remains renamed.
    assert 'RUNTIME_DISTRIBUTION = "cacheon-engine"' in source
    assert b"import cacheon.bootstrap" in source.encode("utf-8")
