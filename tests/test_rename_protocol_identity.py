"""The Cacheon rename is total: protocol vocabulary, runtime ABI, and salts.

Subnet 14 launched with the Cacheon vocabulary as the only protocol identity.
The retired Optima spellings are not accepted, emitted, or present in shipping
source. These pins keep an accidental revert from silently rotating hash
domains or wire schemas.
"""

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
from cacheon.chain.publish import DEFAULT_BUNDLE_KEY_PREFIX
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
from cacheon.manifest import ABI_VERSION, SUPPORTED_ABI_VERSIONS


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_runtime_abi_vocabulary_is_cacheon() -> None:
    assert DIRECT_ARTIFACT_ENTRY == "_cacheon_direct_artifact"
    assert DIRECT_ARTIFACT_IDENTITY_SCHEMA == "cacheon.direct-artifact-execution.v1"
    assert CUTE_CUBIN_PATCHER_ID == "cacheon.build-cute-cubin.v1"
    assert DEVICE_LAUNCH_PLAN_SCHEMA == "cacheon.device-launch-plan.v1"
    assert DEVICE_ARTIFACT_ADMISSION_SCHEMA == "cacheon.device-artifact-admission.v1"
    assert CUDA_CUBIN_ABI_SCHEMA == "cacheon.cuda-cubin-abi.v1"
    assert CUDA_CUBIN_CONTRACT_SCHEMA == "cacheon.cuda-cubin-contract.v1"
    assert DISCOVERY_ABI_VERSION == "cacheon-discovery-abi-v1"
    assert DISCOVERY_OVERLAY_SCHEMA == "cacheon.discovery-overlay.v1"
    assert GRAPH_EVIDENCE_MEDIA_TYPE == "application/vnd.cacheon.graph-verification+json"
    assert GRAPH_EVIDENCE_SCHEMA == "cacheon.qualification.graph-raw-evidence.v1"
    assert SESSION_SCHEMA == "cacheon-isolated-engine-session-v1"
    assert CONTAINER_RECEIPT_SCHEMA == "cacheon-runtime-container-preflight-v2"
    assert HOST_RECEIPT_SCHEMA == "cacheon-runtime-preflight-v2"
    assert WORKER_DIGEST_SCHEMA == "cacheon-installed-distribution-v1"
    assert PLATFORM_DIGEST_SCHEMA == "cacheon-runtime-platform-v1"
    assert WORKER_DISTRIBUTION == "cacheon-harness"
    assert ABI_VERSION == "cacheon-op-abi-v0"
    assert SUPPORTED_ABI_VERSIONS == frozenset({"cacheon-op-abi-v0"})
    assert DEFAULT_BUNDLE_KEY_PREFIX == "cacheon/miner-bundles/sha256"


def test_no_canonical_digest_domain_uses_the_retired_name() -> None:
    retired: list[str] = []
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
                and domain.value.lower().startswith("optima")
            ):
                retired.append(f"{path.relative_to(ROOT)}:{domain.lineno}: {domain.value}")
    assert retired == []


def test_serialized_native_names_and_cryptographic_salts_are_cacheon() -> None:
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

    assert "cacheon_c_{" in engine_tree
    assert "optima_c_{" not in engine_tree
    assert "cacheon_c_{" in marginal_runtime
    assert "cacheon_c_[0-9a-f]{64}" in seam
    assert "cacheon_cute_{digest}" in cute_aot
    assert "cacheon_cuda_{artifact_id}" in cuda_ext
    assert b"cacheon-selection-secret-v1\\0" in qualification
    assert b"cacheon.oci-quiescence.v1\\0" in oci_process
    assert b"cacheon.oci-namespace.v1\\0" in oci_process


def test_signed_release_evidence_vocabulary_is_cacheon() -> None:
    source = (ROOT / "cacheon" / "release.py").read_text(encoding="utf-8")

    for marker in (
        '"creators": ["Tool: cacheon-release-v1"]',
        '"documentNamespace": "urn:cacheon:engine-release:"',
        '"name": "Cacheon Engine "',
        '"name": "cacheon-engine-tree"',
        '"name": "cacheon-native-artifact"',
        '"buildType": "https://cacheon.engine/build/v1"',
        '"id": "https://cacheon.engine/builder/v1"',
        '"uri": "cacheon:model-provision-receipt"',
        '"uri": "cacheon:native-artifact"',
    ):
        assert marker in source

    assert 'RUNTIME_DISTRIBUTION = "cacheon-engine"' in source
    assert "import cacheon.bootstrap" in source


def test_retired_name_is_absent_from_shipping_source() -> None:
    """No spelling of the retired name survives in package or bundle source."""

    offenders: list[str] = []
    for base in ("cacheon", "cacheon_kernels", "examples"):
        for path in sorted((ROOT / base).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if b"optima" in path.read_bytes().lower():
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
