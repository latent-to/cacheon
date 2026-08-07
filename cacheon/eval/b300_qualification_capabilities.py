"""Explicit reusable B300 capabilities; no deployment discovery, launch, or grading."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cacheon.engine_tree import EngineTreeError, integrated_source_tree_digest
from cacheon.eval.b300_registered_qualification_inputs import B300FocusedGraphFacts
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification import (
    GraphVariantRequirement,
    SelectionCommitment,
    SelectionEntropyReceipt,
)
from cacheon.eval.qualification_intake import GraphShapeObservation, GraphVariantObservation
from cacheon.eval.qualification_runner import (
    HiddenJudgeBinding,
    HiddenJudgeReceipt,
    hidden_judge_output_digest,
)
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)

SECRET_SCHEMA = "cacheon.eval.selection-secret-record.v1"
ENTROPY_SCHEMA = "cacheon.eval.selection-entropy-record.v1"
HIDDEN_JUDGE_SCHEMA = "cacheon.eval.accepted-token-subsequence-judge.v1"
RESOLVER_SCHEMA = "cacheon.eval.closed-contribution-source-resolver.v1"
_MAX_SEALED_BYTES = 16 * 1024 * 1024
_MAX_SECRET_OR_ENTROPY_BYTES = 4096


class B300QualificationCapabilityError(RuntimeError):
    """A reusable qualification capability failed closed."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise B300QualificationCapabilityError(str(exc)) from None


def _canonical_object(payload: bytes, *, field: str) -> dict[str, object]:
    def reject(value: str) -> object:
        raise B300QualificationCapabilityError(f"{field} contains {value}")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise B300QualificationCapabilityError(f"{field} repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject,
            parse_constant=reject,
            object_pairs_hook=pairs,
        )
        if type(value) is not dict or canonical_json_bytes(value) != payload:
            raise B300QualificationCapabilityError(f"{field} is not canonical JSON")
    except B300QualificationCapabilityError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise B300QualificationCapabilityError(f"{field} is malformed: {exc}") from None
    return value


def _private_root(value: object) -> tuple[Path, tuple[int, int]]:
    try:
        root = Path(value)  # type: ignore[arg-type]
    except TypeError:
        raise B300QualificationCapabilityError("private store root is not path-like") from None
    if not root.is_absolute():
        raise B300QualificationCapabilityError("private store root must be absolute")
    try:
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        before = root.lstat()
        resolved = root.resolve(strict=True)
        after = resolved.stat()
    except OSError as exc:
        raise B300QualificationCapabilityError(f"private store root is unavailable: {exc}") from None
    owner = os.geteuid() if hasattr(os, "geteuid") else after.st_uid
    if (
        resolved != root
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or after.st_uid != owner
        or stat.S_IMODE(after.st_mode) != 0o700
    ):
        raise B300QualificationCapabilityError(
            "private store root must be owner-controlled, nonsymlink mode 0700"
        )
    return root, (after.st_dev, after.st_ino)


class _PrivateStore:
    def __init__(self, root: object) -> None:
        self.root, self._root_identity = _private_root(root)

    def _validate_root(self) -> None:
        try:
            before = self.root.lstat()
            after = self.root.resolve(strict=True).stat()
        except OSError as exc:
            raise B300QualificationCapabilityError(
                f"private store root changed: {exc}"
            ) from None
        owner = os.geteuid() if hasattr(os, "geteuid") else after.st_uid
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISDIR(after.st_mode)
            or (after.st_dev, after.st_ino) != self._root_identity
            or after.st_uid != owner
            or stat.S_IMODE(after.st_mode) != 0o700
        ):
            raise B300QualificationCapabilityError("private store root identity changed")

    @contextlib.contextmanager
    def _locked(self):
        self._validate_root()
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.root / ".lock", flags, 0o600)
            os.fchmod(descriptor, 0o600)
            info = os.fstat(descriptor)
            owner = os.geteuid() if hasattr(os, "geteuid") else info.st_uid
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != owner
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise B300QualificationCapabilityError("private store lock is unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except B300QualificationCapabilityError:
            raise
        except OSError as exc:
            raise B300QualificationCapabilityError(
                f"private store lock failed: {exc}"
            ) from None
        finally:
            if "descriptor" in locals():
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read(self, path: Path, *, field: str) -> dict[str, object]:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            info = os.fstat(descriptor)
            owner = os.geteuid() if hasattr(os, "geteuid") else info.st_uid
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != owner
                or stat.S_IMODE(info.st_mode) != 0o400
                or info.st_size > _MAX_SEALED_BYTES
            ):
                raise B300QualificationCapabilityError(f"{field} file is unsafe")
            with os.fdopen(descriptor, "rb") as handle:
                payload = handle.read(_MAX_SEALED_BYTES + 1)
                descriptor = -1
            if len(payload) > _MAX_SEALED_BYTES:
                raise B300QualificationCapabilityError(f"{field} file is oversized")
        except B300QualificationCapabilityError:
            raise
        except FileNotFoundError:
            raise B300QualificationCapabilityError(f"{field} is unavailable") from None
        except OSError as exc:
            raise B300QualificationCapabilityError(f"cannot read {field}: {exc}") from None
        finally:
            if "descriptor" in locals() and descriptor >= 0:
                os.close(descriptor)
        return _canonical_object(payload, field=field)

    def _partials(self, path: Path) -> tuple[Path, ...]:
        return tuple(sorted(self.root.glob(f".{path.name}.*")))

    def _publish(self, path: Path, value: object) -> None:
        payload = canonical_json_bytes(value)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=self.root
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o400)
            os.link(temporary, path)
            os.unlink(temporary)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError:
            raise B300QualificationCapabilityError(
                f"sealed record {path.name!r} appeared concurrently"
            ) from None
        except OSError as exc:
            raise B300QualificationCapabilityError(
                f"cannot publish sealed record {path.name!r}: {exc}"
            ) from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _secret_bytes(value: object, *, field: str) -> bytes:
    if (
        type(value) is not bytes
        or len(value) < 32
        or len(value) > _MAX_SECRET_OR_ENTROPY_BYTES
    ):
        raise B300QualificationCapabilityError(
            f"{field} must be 32..{_MAX_SECRET_OR_ENTROPY_BYTES} exact bytes"
        )
    return value


class DurableSelectionSecretStore(_PrivateStore):
    """Atomic exact-reference selection-secret storage and loader."""

    @staticmethod
    def _path(root: Path, reference: str) -> Path:
        return root / f"secret-{reference}.json"

    def put(self, reference: str, secret: bytes) -> None:
        reference = _digest(reference, "selection secret reference")
        secret = _secret_bytes(secret, field="selection secret")
        path = self._path(self.root, reference)
        with self._locked():
            if path.exists():
                if self._reopen(path, reference) != secret:
                    raise B300QualificationCapabilityError(
                        "selection secret reference already names different bytes"
                    )
                return
            if self._partials(path):
                raise B300QualificationCapabilityError(
                    "partial selection-secret publication exists"
                )
            self._publish(
                path,
                {
                    "reference": reference,
                    "schema": SECRET_SCHEMA,
                    "secret_hex": secret.hex(),
                    "secret_sha256": sha256_hex(secret),
                },
            )

    def _reopen(self, path: Path, reference: str) -> bytes:
        value = self._read(path, field="selection secret record")
        if set(value) != {"reference", "schema", "secret_hex", "secret_sha256"}:
            raise B300QualificationCapabilityError(
                "selection secret record fields are not closed"
            )
        if value["schema"] != SECRET_SCHEMA or value["reference"] != reference:
            raise B300QualificationCapabilityError(
                "selection secret record names another authority"
            )
        encoded = value["secret_hex"]
        if not isinstance(encoded, str):
            raise B300QualificationCapabilityError("selection secret encoding is malformed")
        try:
            secret = bytes.fromhex(encoded)
        except ValueError:
            raise B300QualificationCapabilityError(
                "selection secret encoding is malformed"
            ) from None
        secret = _secret_bytes(secret, field="selection secret")
        if secret.hex() != encoded or sha256_hex(secret) != value["secret_sha256"]:
            raise B300QualificationCapabilityError("selection secret record is corrupt")
        return secret

    def __call__(self, reference: str) -> bytes:
        reference = _digest(reference, "selection secret reference")
        with self._locked():
            return self._reopen(self._path(self.root, reference), reference)

    @property
    def inventory_digest(self) -> str:
        with self._locked():
            rows: list[dict[str, str]] = []
            for path in sorted(self.root.glob("secret-*.json")):
                reference = path.name.removeprefix("secret-").removesuffix(".json")
                reference = _digest(reference, "selection secret filename")
                secret = self._reopen(path, reference)
                rows.append({"reference": reference, "secret_sha256": sha256_hex(secret)})
            return canonical_digest(
                "cacheon.eval.selection-secret-inventory.v1", {"records": rows}
            )


def _teardown_payload(value: OCIQuiescenceReceipt) -> dict[str, object]:
    return {
        "container_ids": list(value.container_ids),
        "executor_id": value.executor_id,
        "lease_records": list(value.lease_records),
        "manager_instance_id": value.manager_instance_id,
        "namespace_digest": value.namespace_digest,
        "observed_monotonic_s": format(value.observed_monotonic_s, ".17g"),
        "resource_entries": list(value.resource_entries),
        "schema": value.schema,
        "sequence": value.sequence,
    }


def _teardown_from_payload(value: object) -> OCIQuiescenceReceipt:
    fields = {
        "container_ids",
        "executor_id",
        "lease_records",
        "manager_instance_id",
        "namespace_digest",
        "observed_monotonic_s",
        "resource_entries",
        "schema",
        "sequence",
    }
    if type(value) is not dict or set(value) != fields:
        raise B300QualificationCapabilityError("stored teardown fields are not closed")
    if any(type(value[name]) is not list for name in (
        "container_ids", "lease_records", "resource_entries"
    )):
        raise B300QualificationCapabilityError("stored teardown arrays are malformed")
    observed = value["observed_monotonic_s"]
    if not isinstance(observed, str):
        raise B300QualificationCapabilityError("stored teardown time is malformed")
    try:
        return OCIQuiescenceReceipt(
            schema=value["schema"],  # type: ignore[arg-type]
            executor_id=value["executor_id"],  # type: ignore[arg-type]
            manager_instance_id=value["manager_instance_id"],  # type: ignore[arg-type]
            namespace_digest=value["namespace_digest"],  # type: ignore[arg-type]
            sequence=value["sequence"],  # type: ignore[arg-type]
            observed_monotonic_s=float(observed),
            lease_records=tuple(value["lease_records"]),  # type: ignore[arg-type]
            resource_entries=tuple(value["resource_entries"]),  # type: ignore[arg-type]
            container_ids=tuple(value["container_ids"]),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise B300QualificationCapabilityError(
            f"stored teardown is malformed: {exc}"
        ) from None


def _compatible_teardown(
    stored: OCIQuiescenceReceipt, current: OCIQuiescenceReceipt
) -> bool:
    if current.digest == stored.digest:
        return True
    if (
        current.executor_id != stored.executor_id
        or current.namespace_digest != stored.namespace_digest
        or current.observed_monotonic_s < stored.observed_monotonic_s
    ):
        return False
    return (
        current.manager_instance_id != stored.manager_instance_id
        or current.sequence >= stored.sequence
    )


class DurableSelectionEntropyProvider(_PrivateStore):
    """Persist one post-commit entropy value and its first teardown binding."""

    def __init__(
        self,
        root: object,
        *,
        source_digest: str,
        entropy_source: Callable[[SelectionCommitment, OCIQuiescenceReceipt], bytes],
    ) -> None:
        super().__init__(root)
        self.source_digest = _digest(source_digest, "entropy source digest")
        if not callable(entropy_source):
            raise B300QualificationCapabilityError("entropy source is not callable")
        self._source = entropy_source

    @staticmethod
    def _path(root: Path, commitment_digest: str) -> Path:
        return root / f"entropy-{commitment_digest}.json"

    def _reopen(
        self,
        path: Path,
        commitment: SelectionCommitment,
        current_teardown: OCIQuiescenceReceipt,
    ) -> SelectionEntropyReceipt:
        value = self._read(path, field="selection entropy record")
        if set(value) != {"commitment", "entropy_hex", "receipt", "schema", "teardown"}:
            raise B300QualificationCapabilityError(
                "selection entropy record fields are not closed"
            )
        if value["schema"] != ENTROPY_SCHEMA:
            raise B300QualificationCapabilityError("selection entropy schema changed")
        try:
            stored_commitment = SelectionCommitment.from_dict(value["commitment"])
            receipt = SelectionEntropyReceipt.from_dict(value["receipt"])
        except (TypeError, ValueError) as exc:
            raise B300QualificationCapabilityError(
                f"selection entropy identity is malformed: {exc}"
            ) from None
        stored_teardown = _teardown_from_payload(value["teardown"])
        encoded = value["entropy_hex"]
        if not isinstance(encoded, str):
            raise B300QualificationCapabilityError("selection entropy encoding is malformed")
        try:
            entropy = bytes.fromhex(encoded)
        except ValueError:
            raise B300QualificationCapabilityError(
                "selection entropy encoding is malformed"
            ) from None
        entropy = _secret_bytes(entropy, field="selection entropy")
        authority_receipt = canonical_digest(
            "cacheon.eval.selection-entropy-authority-receipt.v1",
            {
                "commitment_digest": stored_commitment.digest,
                "entropy_digest": sha256_hex(entropy),
                "source_digest": self.source_digest,
                "teardown_digest": stored_teardown.digest,
            },
        )
        expected = SelectionEntropyReceipt(
            self.source_digest,
            stored_commitment.digest,
            sha256_hex(entropy),
            authority_receipt,
        )
        if (
            entropy.hex() != encoded
            or stored_commitment != commitment
            or receipt != expected
            or not _compatible_teardown(stored_teardown, current_teardown)
        ):
            raise B300QualificationCapabilityError(
                "selection entropy record differs from its commitment or teardown"
            )
        return receipt

    def __call__(
        self, commitment: SelectionCommitment, teardown: OCIQuiescenceReceipt
    ) -> SelectionEntropyReceipt:
        if type(commitment) is not SelectionCommitment:
            raise B300QualificationCapabilityError("selection commitment is not exact")
        if type(teardown) is not OCIQuiescenceReceipt:
            raise B300QualificationCapabilityError("teardown receipt is not exact")
        if commitment.entropy_source_digest != self.source_digest:
            raise B300QualificationCapabilityError(
                "selection commitment names another entropy source"
            )
        path = self._path(self.root, commitment.digest)
        with self._locked():
            if path.exists():
                return self._reopen(path, commitment, teardown)
            if self._partials(path):
                raise B300QualificationCapabilityError(
                    "partial selection-entropy publication exists"
                )
            try:
                entropy = self._source(commitment, teardown)
            except Exception as exc:
                raise B300QualificationCapabilityError(
                    f"selection entropy source failed: {exc}"
                ) from None
            entropy = _secret_bytes(entropy, field="selection entropy")
            receipt = SelectionEntropyReceipt(
                self.source_digest,
                commitment.digest,
                sha256_hex(entropy),
                canonical_digest(
                    "cacheon.eval.selection-entropy-authority-receipt.v1",
                    {
                        "commitment_digest": commitment.digest,
                        "entropy_digest": sha256_hex(entropy),
                        "source_digest": self.source_digest,
                        "teardown_digest": teardown.digest,
                    },
                ),
            )
            self._publish(
                path,
                {
                    "commitment": commitment.to_dict(),
                    "entropy_hex": entropy.hex(),
                    "receipt": receipt.to_dict(),
                    "schema": ENTROPY_SCHEMA,
                    "teardown": _teardown_payload(teardown),
                },
            )
            return self._reopen(path, commitment, teardown)


@dataclass(frozen=True)
class AcceptedTokenTask:
    task_digest: str
    accepted_token_subsequences: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_digest", _digest(self.task_digest, "hidden task"))
        sequences = tuple(tuple(row) for row in self.accepted_token_subsequences)
        if (
            not sequences
            or sequences != tuple(sorted(set(sequences)))
            or any(
                not row or any(type(token) is not int or token < 0 for token in row)
                for row in sequences
            )
        ):
            raise B300QualificationCapabilityError(
                "accepted token subsequences must be canonical nonempty token tuples"
            )
        object.__setattr__(self, "accepted_token_subsequences", sequences)

    def identity_data(self) -> dict[str, object]:
        return {
            "accepted_token_subsequences": [list(row) for row in self.accepted_token_subsequences],
            "task_digest": self.task_digest,
        }


@dataclass(frozen=True)
class AcceptedTokenPrompt:
    prompt_digest: str
    tasks: tuple[AcceptedTokenTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_digest", _digest(self.prompt_digest, "prompt"))
        tasks = tuple(self.tasks)
        if (
            not tasks
            or any(type(row) is not AcceptedTokenTask for row in tasks)
            or tuple(row.task_digest for row in tasks)
            != tuple(sorted(set(row.task_digest for row in tasks)))
        ):
            raise B300QualificationCapabilityError(
                "accepted-token prompt tasks are not canonical"
            )
        object.__setattr__(self, "tasks", tasks)

    def identity_data(self) -> dict[str, object]:
        return {
            "prompt_digest": self.prompt_digest,
            "tasks": [row.identity_data() for row in self.tasks],
        }


class ExactAcceptedTokenSubsequenceJudge:
    """Exact contiguous-token hidden judge with a self-bound sealed identity."""

    @staticmethod
    def hidden_judge_digest(
        *,
        hidden_corpus_commitment: str,
        hidden_task_policy_digest: str,
        prompts: tuple[AcceptedTokenPrompt, ...],
    ) -> str:
        corpus = _digest(hidden_corpus_commitment, "hidden corpus commitment")
        policy = _digest(hidden_task_policy_digest, "hidden task policy")
        rows = tuple(prompts)
        if (
            not rows
            or any(type(row) is not AcceptedTokenPrompt for row in rows)
            or tuple(row.prompt_digest for row in rows)
            != tuple(sorted(set(row.prompt_digest for row in rows)))
        ):
            raise B300QualificationCapabilityError(
                "hidden judge prompts are not canonical and complete"
            )
        return canonical_digest(
            HIDDEN_JUDGE_SCHEMA,
            {
                "hidden_corpus_commitment": corpus,
                "hidden_task_policy_digest": policy,
                "match_policy": "contiguous-exact-any-v1",
                "prompts": [row.identity_data() for row in rows],
            },
        )

    def __init__(
        self,
        binding: HiddenJudgeBinding,
        prompts: tuple[AcceptedTokenPrompt, ...],
    ) -> None:
        if type(binding) is not HiddenJudgeBinding:
            raise B300QualificationCapabilityError("hidden judge binding is not exact")
        digest = self.hidden_judge_digest(
            hidden_corpus_commitment=binding.hidden_corpus_commitment,
            hidden_task_policy_digest=binding.hidden_task_policy_digest,
            prompts=prompts,
        )
        if digest != binding.hidden_judge_digest:
            raise B300QualificationCapabilityError(
                "accepted-token authority differs from its hidden-judge binding"
            )
        self.binding = binding
        self._prompts = {row.prompt_digest: row for row in prompts}

    @staticmethod
    def _contains(output: tuple[int, ...], accepted: tuple[int, ...]) -> bool:
        width = len(accepted)
        return any(
            output[index : index + width] == accepted
            for index in range(len(output) - width + 1)
        )

    def __call__(
        self,
        *,
        prompt_digest: str,
        output_ids: tuple[int, ...],
        task_digests: tuple[str, ...],
    ) -> HiddenJudgeReceipt:
        prompt_digest = _digest(prompt_digest, "hidden judge prompt")
        if type(output_ids) is not tuple or any(
            type(token) is not int or token < 0 for token in output_ids
        ):
            raise B300QualificationCapabilityError("hidden judge output IDs are malformed")
        if type(task_digests) is not tuple:
            raise B300QualificationCapabilityError("hidden task digests are not exact")
        prompt = self._prompts.get(prompt_digest)
        if prompt is None:
            raise B300QualificationCapabilityError("hidden judge prompt is not sealed")
        expected = tuple(row.task_digest for row in prompt.tasks)
        if task_digests != expected:
            raise B300QualificationCapabilityError(
                "hidden task digests differ from the sealed prompt"
            )
        passed = tuple(
            any(self._contains(output_ids, accepted) for accepted in task.accepted_token_subsequences)
            for task in prompt.tasks
        )
        return HiddenJudgeReceipt(
            self.binding.digest,
            prompt_digest,
            hidden_judge_output_digest(prompt_digest, output_ids),
            task_digests,
            passed,
        )


@dataclass(frozen=True)
class _SourceSnapshot:
    identity: str
    path: Path
    tree_digest: str
    root_device: int
    root_inode: int

    def identity_data(self, kind: str) -> dict[str, object]:
        return {
            "identity": self.identity,
            "kind": kind,
            "path": self.path.as_posix(),
            "root_device": self.root_device,
            "root_inode": self.root_inode,
            "tree_digest": self.tree_digest,
        }


def _source_tree(path_value: object) -> tuple[Path, str, tuple[int, int]]:
    try:
        path = Path(path_value)  # type: ignore[arg-type]
    except TypeError:
        raise B300QualificationCapabilityError("contribution source is not path-like") from None
    if not path.is_absolute():
        raise B300QualificationCapabilityError("contribution source must be absolute")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        if (
            resolved != path
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISDIR(before.st_mode)
        ):
            raise B300QualificationCapabilityError(
                "contribution source must be a canonical nonsymlink directory"
            )
        tree_digest = integrated_source_tree_digest(path)
        after = path.lstat()
    except B300QualificationCapabilityError:
        raise
    except (EngineTreeError, OSError) as exc:
        raise B300QualificationCapabilityError(
            f"cannot snapshot contribution source: {exc}"
        ) from None
    identity = (before.st_dev, before.st_ino)
    if identity != (after.st_dev, after.st_ino):
        raise B300QualificationCapabilityError("contribution source changed while sealing")
    return path, tree_digest, identity


class ClosedContributionSourceResolver:
    """Closed digest-to-source mapping with drift detection and an empty mode."""

    def __init__(
        self,
        proposal_sources: Mapping[str, object],
        integrated_sources: Mapping[str, object],
        *,
        empty_incumbent: bool = False,
    ) -> None:
        if not isinstance(proposal_sources, Mapping) or not isinstance(integrated_sources, Mapping):
            raise B300QualificationCapabilityError("resolver mappings are not explicit mappings")
        if empty_incumbent and (proposal_sources or integrated_sources):
            raise B300QualificationCapabilityError(
                "empty-incumbent resolver cannot contain source mappings"
            )
        if not empty_incumbent and not (proposal_sources or integrated_sources):
            raise B300QualificationCapabilityError(
                "an empty resolver requires explicit empty-incumbent mode"
            )
        if set(proposal_sources) & set(integrated_sources):
            raise B300QualificationCapabilityError(
                "proposal and integrated resolver identities are ambiguous"
            )
        self._proposal = self._snapshots(proposal_sources, field="proposal")
        self._integrated = self._snapshots(integrated_sources, field="integrated")
        policy = "empty-incumbent-fail-closed" if empty_incumbent else "closed-explicit-mappings"
        self.digest = canonical_digest(
            RESOLVER_SCHEMA,
            {
                "integrated": [row.identity_data("integrated") for row in self._integrated.values()],
                "policy": policy,
                "proposal": [row.identity_data("proposal") for row in self._proposal.values()],
            },
        )

    @classmethod
    def empty_incumbent(cls) -> "ClosedContributionSourceResolver":
        return cls({}, {}, empty_incumbent=True)

    @staticmethod
    def _snapshots(
        values: Mapping[str, object], *, field: str
    ) -> dict[str, _SourceSnapshot]:
        result: dict[str, _SourceSnapshot] = {}
        for raw_identity, raw_path in sorted(values.items()):
            identity = _digest(raw_identity, f"{field} source identity")
            path, tree_digest, root_identity = _source_tree(raw_path)
            result[identity] = _SourceSnapshot(
                identity, path, tree_digest, root_identity[0], root_identity[1]
            )
        return result

    @staticmethod
    def _resolve(values: Mapping[str, _SourceSnapshot], identity: str, kind: str) -> Path:
        identity = _digest(identity, f"{kind} source identity")
        row = values.get(identity)
        if row is None:
            raise B300QualificationCapabilityError(
                f"{kind} contribution source is not in the closed resolver"
            )
        path, tree_digest, root_identity = _source_tree(row.path)
        if (
            path != row.path
            or tree_digest != row.tree_digest
            or root_identity != (row.root_device, row.root_inode)
        ):
            raise B300QualificationCapabilityError(
                f"{kind} contribution source changed after resolver sealing"
            )
        return path

    def resolve_proposal(self, artifact_digest: str) -> Path:
        return self._resolve(self._proposal, artifact_digest, "proposal")

    def resolve_integrated(self, source_tree_digest: str) -> Path:
        return self._resolve(self._integrated, source_tree_digest, "integrated")


@dataclass(frozen=True)
class StructuredGraphShapeRecord:
    descriptor_digest: str
    applicable: bool
    eager_passed: bool
    capture_succeeded: bool
    replay_count: int
    replay_passed: bool
    observation_complete: bool
    failure_is_candidate_attributable: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "descriptor_digest", _digest(self.descriptor_digest, "shape descriptor")
        )
        for field in (
            "applicable",
            "eager_passed",
            "capture_succeeded",
            "replay_passed",
            "observation_complete",
            "failure_is_candidate_attributable",
        ):
            if type(getattr(self, field)) is not bool:
                raise B300QualificationCapabilityError(f"{field} is not an exact boolean")
        if type(self.replay_count) is not int or self.replay_count < 0:
            raise B300QualificationCapabilityError("replay_count is not a nonnegative integer")
        try:
            GraphShapeObservation(
                self.descriptor_digest,
                self.applicable,
                self.eager_passed,
                self.capture_succeeded,
                self.replay_count,
                self.replay_passed,
            )
        except (TypeError, ValueError) as exc:
            raise B300QualificationCapabilityError(
                f"structured graph shape is inconsistent: {exc}"
            ) from None

    @property
    def failed(self) -> bool:
        return self.applicable and not (
            self.eager_passed and self.capture_succeeded and self.replay_passed
        )

    def observation(self) -> GraphShapeObservation:
        return GraphShapeObservation(
            self.descriptor_digest,
            self.applicable,
            self.eager_passed,
            self.capture_succeeded,
            self.replay_count,
            self.replay_passed,
        )


@dataclass(frozen=True)
class StructuredGraphVariantRecord:
    slot_id: str
    variant_id: str
    context_applicable: bool
    domain_coverage_complete: bool
    shapes: tuple[StructuredGraphShapeRecord, ...]

    def __post_init__(self) -> None:
        shapes = tuple(self.shapes)
        if (
            not shapes
            or any(type(row) is not StructuredGraphShapeRecord for row in shapes)
            or tuple(row.descriptor_digest for row in shapes)
            != tuple(sorted(set(row.descriptor_digest for row in shapes)))
        ):
            raise B300QualificationCapabilityError(
                "structured graph shapes are not canonical and complete"
            )
        if type(self.context_applicable) is not bool or type(self.domain_coverage_complete) is not bool:
            raise B300QualificationCapabilityError(
                "structured graph variant flags are not exact booleans"
            )
        try:
            checked = GraphVariantObservation(
                self.slot_id,
                self.variant_id,
                self.context_applicable,
                self.domain_coverage_complete,
                tuple(row.observation() for row in shapes),
            )
        except (TypeError, ValueError) as exc:
            raise B300QualificationCapabilityError(
                f"structured graph variant is inconsistent: {exc}"
            ) from None
        object.__setattr__(self, "slot_id", checked.slot_id)
        object.__setattr__(self, "variant_id", checked.variant_id)
        object.__setattr__(self, "shapes", shapes)


def structured_focused_graph_facts(
    expected_graph_replays: int,
    records: tuple[StructuredGraphVariantRecord, ...],
) -> B300FocusedGraphFacts:
    """Convert complete typed raw facts without launching or grading work."""

    if type(expected_graph_replays) is not int or expected_graph_replays < 2:
        raise B300QualificationCapabilityError(
            "expected graph replays must be an integer >= 2"
        )
    if type(records) is not tuple or not records or any(
        type(row) is not StructuredGraphVariantRecord for row in records
    ):
        raise B300QualificationCapabilityError(
            "graph conversion requires exact structured variant records"
        )
    keys = tuple((row.slot_id, row.variant_id) for row in records)
    if keys != tuple(sorted(set(keys))):
        raise B300QualificationCapabilityError(
            "structured graph variants are not canonical and complete"
        )
    requirements = []
    observations = []
    for row in records:
        if not row.domain_coverage_complete:
            raise B300QualificationCapabilityError(
                "incomplete graph-domain evidence cannot become candidate evidence"
            )
        applicable = tuple(shape.descriptor_digest for shape in row.shapes if shape.applicable)
        if bool(applicable) != row.context_applicable:
            raise B300QualificationCapabilityError(
                "graph context applicability is ambiguous"
            )
        for shape in row.shapes:
            if not shape.observation_complete:
                raise B300QualificationCapabilityError(
                    "partial graph-shape evidence cannot become candidate evidence"
                )
            if shape.failed and not shape.failure_is_candidate_attributable:
                raise B300QualificationCapabilityError(
                    "infrastructure-scoped graph failure cannot become candidate evidence"
                )
            if not shape.failed and shape.failure_is_candidate_attributable:
                raise B300QualificationCapabilityError(
                    "successful graph observation cannot claim failure attribution"
                )
            if shape.replay_count > expected_graph_replays or (
                shape.replay_passed and shape.replay_count != expected_graph_replays
            ):
                raise B300QualificationCapabilityError(
                    "graph replay coverage is incomplete or exceeds authority"
                )
        requirements.append(
            GraphVariantRequirement(
                row.slot_id,
                row.variant_id,
                tuple(shape.descriptor_digest for shape in row.shapes),
                row.context_applicable,
                applicable,
            )
        )
        observations.append(
            GraphVariantObservation(
                row.slot_id,
                row.variant_id,
                row.context_applicable,
                True,
                tuple(shape.observation() for shape in row.shapes),
            )
        )
    return B300FocusedGraphFacts(
        expected_graph_replays, tuple(requirements), tuple(observations)
    )


__all__ = [
    "AcceptedTokenPrompt",
    "AcceptedTokenTask",
    "B300QualificationCapabilityError",
    "ClosedContributionSourceResolver",
    "DurableSelectionEntropyProvider",
    "DurableSelectionSecretStore",
    "ExactAcceptedTokenSubsequenceJudge",
    "StructuredGraphShapeRecord",
    "StructuredGraphVariantRecord",
    "structured_focused_graph_facts",
]
