"""Nonce-bound durable storage for commissioned graph evidence."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import os
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from cacheon.eval.b300_qualification_graph_provider import (
    ARTIFACT_DOMAIN,
    ARTIFACT_MEDIA_TYPE,
    ARTIFACT_SCHEMA,
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
    B300QualificationGraphProviderError,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStoreError,
    absolute_deadline as _deadline,
    absolute_path as _absolute,
    canonical_object as _canonical_object,
    check_deadline as _check_deadline,
    directory_identity as _directory_identity,
    fsync_directory as _fsync_directory,
    mkdir_private as _mkdir,
    owner_uid as _owner,
    publish_sealed as _publish_sealed_file,
    read_regular as _read_regular_file,
)
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    prepare_evidence_root,
    publish_evidence,
    reopen_evidence,
)
from cacheon.stack_identity import canonical_json_bytes, require_sha256_hex


INDEX_SCHEMA = "cacheon.eval.b300-qualification-graph-evidence-index.v1"
ATTEMPT_SCHEMA = "cacheon.eval.b300-qualification-graph-evidence-attempt.v2"
MAX_INDEX_BYTES = 64 * 1024
MAX_ATTEMPT_BYTES = 64 * 1024
_LOCK_POLL_SECONDS = 0.01


class B300QualificationGraphPreEntryFailure(Exception):
    """Compatibility exception that never authorizes an armed-attempt retry."""

    def __init__(self, proof_digest: str) -> None:
        try:
            self.proof_digest = require_sha256_hex(proof_digest, field="pre-entry failure proof digest")
        except (TypeError, ValueError) as exc:
            raise B300QualificationGraphEvidenceStoreError(str(exc)) from None
        super().__init__("untrusted pre-entry failure claim")


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise B300QualificationGraphEvidenceStoreError(str(exc)) from None


@dataclass(frozen=True)
class B300QualificationGraphAttemptToken:
    """Exact authority naming one durable armed generation."""

    generation: int
    nonce: str
    binding_digest: str
    verification_policy_digest: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation != 1:
            raise B300QualificationGraphEvidenceStoreError("graph evidence attempt generation must be exactly one")
        object.__setattr__(self, "nonce", _digest(self.nonce, "graph evidence attempt nonce"))
        object.__setattr__(
            self,
            "binding_digest",
            _digest(self.binding_digest, "graph evidence attempt binding digest"),
        )
        object.__setattr__(
            self,
            "verification_policy_digest",
            _digest(
                self.verification_policy_digest,
                "graph evidence attempt verification policy digest",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_digest": self.binding_digest,
            "generation": self.generation,
            "nonce": self.nonce,
            "verification_policy_digest": self.verification_policy_digest,
        }


@dataclass(frozen=True)
class B300QualificationGraphGenerationOutput:
    """Artifact output explicitly bound to the generation that produced it."""

    token: B300QualificationGraphAttemptToken
    artifact: B300QualificationGraphArtifact

    def __post_init__(self) -> None:
        if type(self.token) is not B300QualificationGraphAttemptToken:
            raise B300QualificationGraphEvidenceStoreError("graph generation output token must be exact and typed")
        if type(self.artifact) is not B300QualificationGraphArtifact:
            raise B300QualificationGraphEvidenceStoreError("graph generation output artifact must be exact and typed")


GraphArtifactProducer = Callable[
    [B300QualificationGraphBinding, B300QualificationGraphAttemptToken, float],
    B300QualificationGraphGenerationOutput,
]


@dataclass(frozen=True)
class _AttemptState:
    outcome: str
    token: B300QualificationGraphAttemptToken | None = None
    reference: EvidenceArtifactRef | None = None


def _publication_boundary(_kind: str, _phase: str) -> None:
    """Private deterministic crash-injection seam used by durability tests."""


class B300QualificationGraphEvidenceStore:
    """Create-once graph-artifact index for one verification policy."""

    def __init__(self, root: str | Path, verification_policy_digest: str) -> None:
        self.root = _absolute(root)
        self.verification_policy_digest = _digest(verification_policy_digest, "verification policy digest")
        created_root = False
        try:
            self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
            created_root = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise B300QualificationGraphEvidenceStoreError(f"cannot create graph evidence root: {exc}") from None
        self._identities: dict[Path, tuple[int, int]] = {self.root: _directory_identity(self.root)}
        self._identity_lock = threading.RLock()
        _fsync_directory(self.root)
        if created_root:
            _fsync_directory(self.root.parent)

        self.evidence_root = self.root / "artifacts"
        self.index_root = self.root / "indexes"
        self.lock_root = self.root / "locks"
        self.attempt_root = self.root / "attempts"
        self.staging_root = self.root / "staging"
        for path in (
            self.evidence_root,
            self.index_root,
            self.lock_root,
            self.attempt_root,
            self.staging_root,
        ):
            self._identities[path] = _mkdir(path)
        try:
            prepare_evidence_root(self.evidence_root)
        except EvidenceStoreError as exc:
            raise B300QualificationGraphEvidenceStoreError(str(exc)) from None

        policy = self.verification_policy_digest
        self._policy_index_root = self.index_root / policy
        self._policy_lock_root = self.lock_root / policy
        self._policy_attempt_root = self.attempt_root / policy
        self._policy_staging_root = self.staging_root / policy
        for path in (
            self._policy_index_root,
            self._policy_lock_root,
            self._policy_attempt_root,
            self._policy_staging_root,
        ):
            self._identities[path] = _mkdir(path)

    def _validate_directories(self) -> None:
        with self._identity_lock:
            identities = tuple(self._identities.items())
        for path, expected in identities:
            if _directory_identity(path) != expected:
                raise B300QualificationGraphEvidenceStoreError("graph evidence directory identity changed")

    @staticmethod
    def _binding(value: object) -> B300QualificationGraphBinding:
        if type(value) is not B300QualificationGraphBinding:
            raise B300QualificationGraphEvidenceStoreError("graph binding must be exact and typed")
        return value

    def _index_path(self, binding: B300QualificationGraphBinding) -> Path:
        return self._policy_index_root / f"{binding.digest}.json"

    def _lock_path(self, binding: B300QualificationGraphBinding) -> Path:
        return self._policy_lock_root / f"{binding.digest}.lock"

    @contextlib.contextmanager
    def _locked(self, binding: B300QualificationGraphBinding, deadline: float) -> Iterator[None]:
        _check_deadline(deadline)
        self._validate_directories()
        path = self._lock_path(binding)
        common = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        common |= getattr(os, "O_NOFOLLOW", 0)
        created = False
        descriptor: int | None = None
        locked = False
        try:
            try:
                descriptor = os.open(path, common | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
            except FileExistsError:
                descriptor = os.open(path, common)
            if created:
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                _fsync_directory(path.parent)
            before = path.lstat()
            opened = os.fstat(descriptor)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or opened.st_nlink != 1
                or opened.st_uid != _owner()
                or stat.S_IMODE(opened.st_mode) != 0o600
            ):
                raise B300QualificationGraphEvidenceStoreError("graph evidence key lock is unsafe")
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EAGAIN, errno.EWOULDBLOCK}:
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise B300QualificationGraphEvidenceHold("graph evidence key lock deadline expired") from None
                    time.sleep(min(_LOCK_POLL_SECONDS, remaining))
            _check_deadline(deadline)
            locked_path = path.lstat()
            locked_fd = os.fstat(descriptor)
            if (
                (locked_path.st_dev, locked_path.st_ino) != (locked_fd.st_dev, locked_fd.st_ino)
                or locked_fd.st_nlink != 1
                or locked_fd.st_uid != _owner()
                or stat.S_IMODE(locked_fd.st_mode) != 0o600
            ):
                raise B300QualificationGraphEvidenceStoreError("graph evidence key lock changed while waiting")
            self._validate_directories()
            yield
            _check_deadline(deadline)
        except (B300QualificationGraphEvidenceStoreError, OSError) as exc:
            if isinstance(exc, B300QualificationGraphEvidenceStoreError):
                raise
            raise B300QualificationGraphEvidenceStoreError(f"graph evidence key lock failed: {exc}") from None
        finally:
            if descriptor is not None:
                if locked:
                    with contextlib.suppress(OSError):
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                with contextlib.suppress(OSError):
                    os.close(descriptor)

    @staticmethod
    def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
        return _read_regular_file(path, label=label, max_bytes=max_bytes)

    def _publish_sealed(
        self,
        target: Path,
        payload: bytes,
        *,
        kind: str,
        label: str,
        max_bytes: int,
        deadline: float,
    ) -> None:
        _publish_sealed_file(
            target,
            payload,
            kind=kind,
            label=label,
            max_bytes=max_bytes,
            deadline=deadline,
            staging_root=self._policy_staging_root,
            boundary=_publication_boundary,
        )

    def _attempt_directory(self, binding: B300QualificationGraphBinding) -> Path:
        path = self._policy_attempt_root / binding.digest
        with self._identity_lock:
            expected = self._identities.get(path)
            if expected is None:
                self._identities[path] = _mkdir(path)
            elif _directory_identity(path) != expected:
                raise B300QualificationGraphEvidenceStoreError(
                    "graph evidence attempt directory identity changed"
                )
        return path

    @staticmethod
    def _record_name(record_type: str) -> str:
        if record_type not in {"armed", "output", "terminal"}:
            raise B300QualificationGraphEvidenceStoreError("graph evidence attempt record type is invalid")
        return f"{1:016d}.{record_type}.json"

    def _token(self, binding: B300QualificationGraphBinding) -> B300QualificationGraphAttemptToken:
        return B300QualificationGraphAttemptToken(
            1,
            secrets.token_hex(32),
            binding.digest,
            self.verification_policy_digest,
        )

    def _attempt_common(
        self,
        binding: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        record_type: str,
    ) -> dict[str, object]:
        return {
            "binding": binding.to_dict(),
            "binding_digest": binding.digest,
            "generation": token.generation,
            "generation_nonce": token.nonce,
            "record_type": record_type,
            "schema": ATTEMPT_SCHEMA,
            "verification_policy_digest": self.verification_policy_digest,
        }

    def _append_arm(
        self,
        binding: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        deadline: float,
    ) -> None:
        directory = self._attempt_directory(binding)
        self._publish_sealed(
            directory / self._record_name("armed"),
            canonical_json_bytes(self._attempt_common(binding, token, "armed")),
            kind="arm",
            label="graph evidence attempt arm",
            max_bytes=MAX_ATTEMPT_BYTES,
            deadline=deadline,
        )

    def _append_success(
        self,
        binding: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        reference: EvidenceArtifactRef,
        deadline: float,
    ) -> None:
        exact = self._validate_reference(binding, reference)
        value = {
            **self._attempt_common(binding, token, "terminal"),
            "artifact_reference": exact.to_dict(),
            "outcome": "success",
        }
        directory = self._attempt_directory(binding)
        self._publish_sealed(
            directory / self._record_name("terminal"),
            canonical_json_bytes(value),
            kind="terminal",
            label="graph evidence attempt terminal",
            max_bytes=MAX_ATTEMPT_BYTES,
            deadline=deadline,
        )

    def _append_output(
        self,
        binding: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        reference: EvidenceArtifactRef,
        deadline: float,
    ) -> None:
        exact = self._validate_reference(binding, reference)
        value = {
            **self._attempt_common(binding, token, "output"),
            "artifact_reference": exact.to_dict(),
        }
        self._publish_sealed(
            self._attempt_directory(binding) / self._record_name("output"),
            canonical_json_bytes(value),
            kind="output",
            label="graph evidence generation output",
            max_bytes=MAX_ATTEMPT_BYTES,
            deadline=deadline,
        )

    def _parse_attempt_record(
        self,
        binding: B300QualificationGraphBinding,
        path: Path,
        generation: int,
        filename_type: str,
    ) -> tuple[B300QualificationGraphAttemptToken, EvidenceArtifactRef | None]:
        payload = self._read_regular(path, label="graph evidence attempt record", max_bytes=MAX_ATTEMPT_BYTES)
        row = _canonical_object(payload, label="graph evidence attempt record")
        common = {
            "binding",
            "binding_digest",
            "generation",
            "generation_nonce",
            "record_type",
            "schema",
            "verification_policy_digest",
        }
        if filename_type == "armed":
            expected_fields = common
        elif filename_type == "output":
            expected_fields = common | {"artifact_reference"}
        else:
            expected_fields = common | {"artifact_reference", "outcome"}
        if set(row) != expected_fields:
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence attempt record fields do not match the closed schema"
            )
        try:
            indexed_binding = B300QualificationGraphBinding.from_dict(row["binding"])
            token = B300QualificationGraphAttemptToken(
                row["generation"],
                row["generation_nonce"],
                row["binding_digest"],
                row["verification_policy_digest"],
            )
        except (B300QualificationGraphProviderError, TypeError) as exc:
            raise B300QualificationGraphEvidenceStoreError(
                f"graph evidence attempt binding or token is not exact: {exc}"
            ) from None
        if (
            row["schema"] != ATTEMPT_SCHEMA
            or row["record_type"] != filename_type
            or row["generation"] != generation
            or token.binding_digest != binding.digest
            or token.verification_policy_digest != self.verification_policy_digest
            or indexed_binding != binding
            or indexed_binding.digest != binding.digest
        ):
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence attempt record differs from its generation, key, token, or policy"
            )
        if filename_type == "armed":
            return token, None
        if filename_type == "terminal" and row["outcome"] != "success":
            raise B300QualificationGraphEvidenceStoreError("graph evidence attempt terminal outcome is unsupported")
        try:
            reference = EvidenceArtifactRef.from_dict(row["artifact_reference"])
        except EvidenceStoreError as exc:
            raise B300QualificationGraphEvidenceStoreError(
                f"graph evidence success reference is not exact: {exc}"
            ) from None
        return token, self._validate_reference(binding, reference)

    def _attempt_state(self, binding: B300QualificationGraphBinding) -> _AttemptState:
        directory = self._attempt_directory(binding)
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            raise B300QualificationGraphEvidenceStoreError(
                f"cannot enumerate graph evidence attempt history: {exc}"
            ) from None
        records: dict[int, dict[str, Path]] = {}
        for path in children:
            parts = path.name.split(".")
            if (
                len(parts) != 3
                or len(parts[0]) != 16
                or not parts[0].isdigit()
                or parts[1] not in {"armed", "output", "terminal"}
                or parts[2] != "json"
                or int(parts[0]) < 1
            ):
                raise B300QualificationGraphEvidenceStoreError(
                    "graph evidence attempt history contains an unexpected entry"
                )
            generation = int(parts[0])
            records.setdefault(generation, {})[parts[1]] = path
        if not records:
            return _AttemptState("none")
        if set(records) != {1}:
            if 1 not in records:
                raise B300QualificationGraphEvidenceStoreError("graph evidence attempt history has a generation gap")
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence attempt history continues after its only authorized generation"
            )
        group = records[1]
        armed_path = group.get("armed")
        if armed_path is None:
            raise B300QualificationGraphEvidenceStoreError("graph evidence attempt terminal has no matching arm")
        arm_token, _ = self._parse_attempt_record(binding, armed_path, 1, "armed")
        output_path = group.get("output")
        output_reference = None
        if output_path is not None:
            output_token, output_reference = self._parse_attempt_record(
                binding, output_path, 1, "output"
            )
            if output_token != arm_token or type(output_reference) is not EvidenceArtifactRef:
                raise B300QualificationGraphEvidenceStoreError(
                    "graph evidence output is not bound to its exact armed generation"
                )
        terminal_path = group.get("terminal")
        if terminal_path is None:
            return _AttemptState(
                "output" if output_reference is not None else "armed",
                arm_token,
                output_reference,
            )
        if output_reference is None:
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence terminal has no durable generation output"
            )
        terminal_token, reference = self._parse_attempt_record(binding, terminal_path, 1, "terminal")
        if (
            terminal_token != arm_token
            or type(reference) is not EvidenceArtifactRef
            or reference != output_reference
        ):
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence terminal is not bound to its exact armed generation"
            )
        return _AttemptState("success", arm_token, reference)

    def _validate_artifact(
        self, binding: B300QualificationGraphBinding, artifact: object
    ) -> B300QualificationGraphArtifact:
        if type(artifact) is not B300QualificationGraphArtifact:
            raise B300QualificationGraphEvidenceStoreError(
                "graph generation output did not contain an exact typed artifact"
            )
        if artifact.binding != binding:
            raise B300QualificationGraphEvidenceStoreError("graph generation output differs from the exact binding")
        if artifact.verification_policy_digest != self.verification_policy_digest:
            raise B300QualificationGraphEvidenceStoreError(
                "graph generation output differs from the exact verification policy"
            )
        return artifact

    def _validate_output(
        self,
        binding: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        output: object,
    ) -> B300QualificationGraphArtifact:
        if type(output) is not B300QualificationGraphGenerationOutput:
            raise B300QualificationGraphEvidenceStoreError(
                "graph callback output is not bound to an exact armed generation"
            )
        if output.token != token:
            raise B300QualificationGraphEvidenceStoreError(
                "graph callback output token differs from the exact armed generation"
            )
        return self._validate_artifact(binding, output.artifact)

    def _expected_reference(self, artifact: B300QualificationGraphArtifact) -> EvidenceArtifactRef:
        payload = artifact.canonical_bytes
        return EvidenceArtifactRef(
            ARTIFACT_DOMAIN,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            ARTIFACT_MEDIA_TYPE,
            ARTIFACT_SCHEMA,
        )

    def _publish_artifact(
        self,
        binding: B300QualificationGraphBinding,
        artifact: B300QualificationGraphArtifact,
        deadline: float,
    ) -> EvidenceArtifactRef:
        exact = self._validate_artifact(binding, artifact)
        _check_deadline(deadline)
        try:
            reference = publish_evidence(
                self.evidence_root,
                exact.canonical_bytes,
                domain=ARTIFACT_DOMAIN,
                media_type=ARTIFACT_MEDIA_TYPE,
                schema=ARTIFACT_SCHEMA,
                deadline=deadline,
            )
        except EvidenceStoreError as exc:
            raise B300QualificationGraphEvidenceStoreError(f"cannot publish graph evidence artifact: {exc}") from None
        exact_reference = self._validate_reference(binding, reference)
        _check_deadline(deadline)
        return exact_reference

    def _validate_reference(
        self,
        binding: B300QualificationGraphBinding,
        reference: EvidenceArtifactRef,
    ) -> EvidenceArtifactRef:
        if (
            type(reference) is not EvidenceArtifactRef
            or reference.domain != ARTIFACT_DOMAIN
            or reference.media_type != ARTIFACT_MEDIA_TYPE
            or reference.schema != ARTIFACT_SCHEMA
            or reference.size < 1
        ):
            raise B300QualificationGraphEvidenceStoreError(
                "indexed graph evidence reference differs from the closed artifact schema"
            )
        try:
            payload = reopen_evidence(self.evidence_root, reference)
            artifact = B300QualificationGraphArtifact.from_canonical_bytes(payload)
        except (EvidenceStoreError, B300QualificationGraphProviderError) as exc:
            raise B300QualificationGraphEvidenceStoreError(
                f"indexed graph evidence did not reopen exactly: {exc}"
            ) from None
        if artifact.binding != binding:
            raise B300QualificationGraphEvidenceStoreError("indexed graph artifact differs from the exact binding")
        if artifact.verification_policy_digest != self.verification_policy_digest:
            raise B300QualificationGraphEvidenceStoreError(
                "indexed graph artifact differs from the exact verification policy"
            )
        return reference

    def _index_value(
        self,
        binding: B300QualificationGraphBinding,
        reference: EvidenceArtifactRef,
    ) -> dict[str, object]:
        return {
            "artifact_reference": reference.to_dict(),
            "binding": binding.to_dict(),
            "binding_digest": binding.digest,
            "schema": INDEX_SCHEMA,
            "verification_policy_digest": self.verification_policy_digest,
        }

    def _read_index(self, binding: B300QualificationGraphBinding) -> EvidenceArtifactRef:
        payload = self._read_regular(
            self._index_path(binding),
            label="graph evidence index",
            max_bytes=MAX_INDEX_BYTES,
        )
        row = _canonical_object(payload, label="graph evidence index")
        fields = {
            "artifact_reference",
            "binding",
            "binding_digest",
            "schema",
            "verification_policy_digest",
        }
        if set(row) != fields:
            raise B300QualificationGraphEvidenceStoreError("graph evidence index fields do not match the closed schema")
        if row["schema"] != INDEX_SCHEMA:
            raise B300QualificationGraphEvidenceStoreError("graph evidence index schema is unsupported")
        try:
            indexed_binding = B300QualificationGraphBinding.from_dict(row["binding"])
            reference = EvidenceArtifactRef.from_dict(row["artifact_reference"])
        except (B300QualificationGraphProviderError, EvidenceStoreError) as exc:
            raise B300QualificationGraphEvidenceStoreError(
                f"graph evidence index is not exactly typed: {exc}"
            ) from None
        if (
            row["verification_policy_digest"] != self.verification_policy_digest
            or row["binding_digest"] != binding.digest
            or indexed_binding != binding
            or indexed_binding.digest != binding.digest
        ):
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence index differs from its exact policy or binding key"
            )
        return self._validate_reference(binding, reference)

    def _publish_index(
        self,
        binding: B300QualificationGraphBinding,
        reference: EvidenceArtifactRef,
        deadline: float,
    ) -> None:
        payload = canonical_json_bytes(self._index_value(binding, reference))
        self._publish_sealed(
            self._index_path(binding),
            payload,
            kind="index",
            label="graph evidence index",
            max_bytes=MAX_INDEX_BYTES,
            deadline=deadline,
        )
        if self._read_index(binding) != reference:
            raise B300QualificationGraphEvidenceStoreError("published graph evidence index did not reopen exactly")
        _check_deadline(deadline)

    def _resolve_success_locked(
        self,
        binding: B300QualificationGraphBinding,
        state: _AttemptState,
        deadline: float,
    ) -> EvidenceArtifactRef:
        if state.outcome != "success" or type(state.reference) is not EvidenceArtifactRef:
            raise B300QualificationGraphEvidenceStoreError("graph evidence success state lacks an exact reference")
        self._publish_index(binding, state.reference, deadline)
        reference = self._read_index(binding)
        _check_deadline(deadline)
        return reference

    def _reject_unbacked_index(
        self,
        binding: B300QualificationGraphBinding,
        state: _AttemptState,
    ) -> None:
        if state.outcome != "success" and os.path.lexists(self._index_path(binding)):
            raise B300QualificationGraphEvidenceStoreError(
                "graph evidence index has no authenticated terminal attempt"
            )

    def arm(self, binding: B300QualificationGraphBinding, *, deadline: float) -> B300QualificationGraphAttemptToken:
        """Durably arm one generation and return its exact nonce-bearing token."""

        exact = self._binding(binding)
        bound = _deadline(deadline)
        with self._locked(exact, bound):
            state = self._attempt_state(exact)
            self._reject_unbacked_index(exact, state)
            if state.outcome in {"armed", "output"}:
                raise B300QualificationGraphEvidenceHold(
                    "graph evidence attempt is already armed without a terminal record"
                )
            if state.outcome == "success":
                self._resolve_success_locked(exact, state, bound)
                raise B300QualificationGraphEvidenceStoreError("graph evidence binding already has a terminal mapping")
            token = self._token(exact)
            self._append_arm(exact, token, bound)
            _check_deadline(bound)
            return token

    def finalize(
        self,
        binding: B300QualificationGraphBinding,
        output: B300QualificationGraphGenerationOutput,
        *,
        deadline: float,
    ) -> EvidenceArtifactRef:
        """Finalize only output bound to the exact durable armed generation."""

        exact = self._binding(binding)
        bound = _deadline(deadline)
        if type(output) is not B300QualificationGraphGenerationOutput:
            raise B300QualificationGraphEvidenceStoreError(
                "graph finalization output must be exact and generation-bound"
            )
        with self._locked(exact, bound):
            state = self._attempt_state(exact)
            self._reject_unbacked_index(exact, state)
            if state.outcome == "none" or state.token is None:
                raise B300QualificationGraphEvidenceStoreError("graph finalization has no durable matching arm")
            artifact = self._validate_output(exact, state.token, output)
            if state.outcome == "success":
                if self._expected_reference(artifact) != state.reference:
                    raise B300QualificationGraphEvidenceStoreError("graph finalization diverges from terminal success")
                reference = self._resolve_success_locked(exact, state, bound)
                _check_deadline(bound)
                return reference
            if state.outcome == "output":
                reference = state.reference
                if (
                    type(reference) is not EvidenceArtifactRef
                    or self._expected_reference(artifact) != reference
                ):
                    raise B300QualificationGraphEvidenceStoreError(
                        "graph finalization diverges from durable generation output"
                    )
            else:
                reference = self._publish_artifact(exact, artifact, bound)
                _check_deadline(bound)
                self._append_output(exact, state.token, reference, bound)
            assert type(reference) is EvidenceArtifactRef
            self._append_success(exact, state.token, reference, bound)
            self._publish_index(exact, reference, bound)
            indexed = self._read_index(exact)
            _check_deadline(bound)
            return indexed

    def reopen(self, binding: B300QualificationGraphBinding, *, deadline: float) -> EvidenceArtifactRef:
        """Reopen and fully authenticate one exact durable mapping."""

        exact = self._binding(binding)
        bound = _deadline(deadline)
        with self._locked(exact, bound):
            state = self._attempt_state(exact)
            self._reject_unbacked_index(exact, state)
            if state.outcome == "success":
                reference = self._resolve_success_locked(exact, state, bound)
                _check_deadline(bound)
                return reference
            if state.outcome in {"armed", "output"}:
                raise B300QualificationGraphEvidenceHold("graph evidence attempt is armed without a terminal record")
            raise B300QualificationGraphEvidenceStoreError("graph evidence index is unavailable")

    def probe_once(
        self,
        binding: B300QualificationGraphBinding,
        producer: GraphArtifactProducer,
        *,
        deadline: float,
    ) -> EvidenceArtifactRef:
        """Arm once; recovery consumes only store-owned durable output."""

        exact = self._binding(binding)
        bound = _deadline(deadline)
        if not callable(producer):
            raise B300QualificationGraphEvidenceStoreError("graph artifact producer must be callable before arming")

        with self._locked(exact, bound):
            state = self._attempt_state(exact)
            self._reject_unbacked_index(exact, state)
            if state.outcome == "success":
                reference = self._resolve_success_locked(exact, state, bound)
                _check_deadline(bound)
                return reference
            if state.outcome == "output":
                token = state.token
                reference = state.reference
                if type(token) is not B300QualificationGraphAttemptToken or type(
                    reference
                ) is not EvidenceArtifactRef:
                    raise B300QualificationGraphEvidenceStoreError(
                        "graph durable output state is incomplete"
                    )
                self._append_success(exact, token, reference, bound)
                self._publish_index(exact, reference, bound)
                indexed = self._read_index(exact)
                _check_deadline(bound)
                return indexed
            if state.outcome == "armed":
                raise B300QualificationGraphEvidenceHold(
                    "graph evidence attempt is armed without durable generation output"
                )
            else:
                token = self._token(exact)
                self._append_arm(exact, token, bound)
        if type(token) is not B300QualificationGraphAttemptToken:
            raise B300QualificationGraphEvidenceStoreError("graph evidence armed state lacks an exact token")

        _check_deadline(bound)
        try:
            output = producer(exact, token, bound)
        except Exception as exc:
            raise B300QualificationGraphEvidenceHold(
                f"graph artifact producer outcome is ambiguous and remains on HOLD: {exc}"
            ) from None
        _check_deadline(bound)
        try:
            return self.finalize(exact, output, deadline=bound)
        except B300QualificationGraphEvidenceHold:
            raise
        except B300QualificationGraphEvidenceStoreError as exc:
            raise B300QualificationGraphEvidenceHold(
                f"graph artifact producer was not exact and remains on HOLD: {exc}"
            ) from None


__all__ = [
    "ATTEMPT_SCHEMA", "B300QualificationGraphAttemptToken", "B300QualificationGraphEvidenceHold",
    "B300QualificationGraphEvidenceStore", "B300QualificationGraphEvidenceStoreError",
    "B300QualificationGraphGenerationOutput", "B300QualificationGraphPreEntryFailure",
    "GraphArtifactProducer", "INDEX_SCHEMA",
]
