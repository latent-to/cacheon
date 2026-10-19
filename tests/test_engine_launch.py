"""The native-build identity bytes that retained launch digests were computed over.

The controller's import closure is proven end to end in
``tests/test_validator_end_to_end.py``; launch resolution runs on the pod. This
golden stays because every launch digest a retained PASS carries binds a native
build digest, and no other check freezes those payload bytes.
"""

from __future__ import annotations

import hashlib

from cacheon.eval.engine_launch import (
    NATIVE_BUILD_SCHEMA_VERSION,
    NativeBuildSpec,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
)
from cacheon.stack_identity import canonical_digest, canonical_json_bytes


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _native(tree_digest: str, **changes: object) -> NativeBuildSpec:
    values: dict[str, object] = {
        "tree_digest": tree_digest,
        "image_digest": _digest("image"),
        "platform_digest": _digest("platform"),
        "worker_distribution_digest": _digest("worker-dist"),
        "target_architecture": "sm120",
        "dependency_policy_digest": _digest("dependency-policy"),
    }
    values.update(changes)
    values.setdefault("toolchain_digest", native_toolchain_digest(
        image_digest=values["image_digest"],  # type: ignore[arg-type]
        platform_digest=values["platform_digest"],  # type: ignore[arg-type]
    ))
    values.setdefault("patcher_digest", native_patcher_digest(
        worker_distribution_digest=values["worker_distribution_digest"],  # type: ignore[arg-type]
    ))
    values.setdefault("compiler_flags_digest", native_compiler_policy_digest(
        image_digest=values["image_digest"],  # type: ignore[arg-type]
        worker_distribution_digest=values["worker_distribution_digest"],  # type: ignore[arg-type]
        dependency_policy_digest=values["dependency_policy_digest"],  # type: ignore[arg-type]
        target_architecture=values["target_architecture"],  # type: ignore[arg-type]
    ))
    return NativeBuildSpec(**values)  # type: ignore[arg-type]


def test_legacy_native_build_schema1_payload_bytes_and_digest_are_unchanged() -> None:
    native = _native(_digest("tree"))
    expected_policy_payload = {
        "dependency_policy_digest": native.dependency_policy_digest,
        "image_digest": native.image_digest,
        "target_architecture": native.target_architecture,
        "worker_distribution_digest": native.worker_distribution_digest,
    }
    assert native.compiler_flags_digest == canonical_digest(
        "cacheon.eval.native-compiler-policy", expected_policy_payload
    )

    expected_build_payload = {
        "compiler_flags_digest": native.compiler_flags_digest,
        "dependency_policy_digest": native.dependency_policy_digest,
        "image_digest": native.image_digest,
        "patcher_digest": native.patcher_digest,
        "platform_digest": native.platform_digest,
        "schema_version": NATIVE_BUILD_SCHEMA_VERSION,
        "target_architecture": native.target_architecture,
        "toolchain_digest": native.toolchain_digest,
        "tree_digest": native.tree_digest,
        "type": "native_build",
        "worker_distribution_digest": native.worker_distribution_digest,
    }
    assert native.schema_version == NATIVE_BUILD_SCHEMA_VERSION
    assert native.to_dict() == expected_build_payload
    assert native.canonical_bytes == canonical_json_bytes(expected_build_payload)
    assert native.digest == canonical_digest(
        "cacheon.eval.native-build", expected_build_payload
    )

    # schema_version was historically the tenth positional argument.  Adding
    # the optional profile must not reinterpret existing positional callers.
    positional = NativeBuildSpec(
        native.tree_digest,
        native.image_digest,
        native.platform_digest,
        native.worker_distribution_digest,
        native.toolchain_digest,
        native.patcher_digest,
        native.compiler_flags_digest,
        native.target_architecture,
        native.dependency_policy_digest,
        NATIVE_BUILD_SCHEMA_VERSION,
    )
    assert positional == native
