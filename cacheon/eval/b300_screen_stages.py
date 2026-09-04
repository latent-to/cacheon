"""Closed current-Cacheon screen stages for registered B300 arena lanes.

The static stage reopens the immutable worker publication, projects only its
miner-owned inventory, and reruns manifest, target-catalog, contribution, and
recursive source policy.  It never imports candidate Python.

Build, ABI, and graph stages share one bounded candidate-bound carrier.  Build
uses :func:`run_oci_prebuild` directly and reopens the native product.  ABI and
graph keep their fixed stage positions but only reopen that carrier: their GPU
work is deferred to the resident all-rank swap, whose acknowledgement proves
slot registration and whose read forces graph recapture/replay.

An exception or missing/mutated carrier is infrastructure ``NO_DECISION``.
Only deterministic policy rejection from immutable submitted bytes can produce
candidate ``FAIL`` here.  No candidate field selects a module, command,
argument, or loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Iterator, Protocol

from cacheon.arena_service import (
    MAX_SCREEN_REASON_CHARS,
    ArenaCandidateBinding,
    ArenaServiceManifest,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import (
    WorkerBundlePublication,
    WorkerBundlePublicationError,
    reopen_worker_bundle,
)
from cacheon.engine_tree import (
    EngineTreeError,
    InspectedContribution,
    MaterializedEngineTree,
    inspect_contribution,
)
from cacheon.eval.b300_arena_provider import B300ScreenStageHandler
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    ResolvedEngineLaunch,
    TrustedLaunchBinding,
    resolve_engine_launch,
)
from cacheon.eval.evidence_store import (
    prepare_evidence_root,
)
from cacheon.eval.native_artifact import (
    NativeArtifactPublication,
    reopen_native_artifact,
)
from cacheon.eval.oci_backend import (
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
)
from cacheon.eval.oci_prebuild import (
    OCIPrebuildResult,
    run_oci_prebuild,
)
from cacheon.manifest import (
    ManifestError,
    all_declared_cuda_sources,
    all_declared_dep_patches,
    load_manifest,
)
from cacheon.rebuild import RebuildError
from cacheon.registry import eligibility_from_metadata
from cacheon.sandbox import scan_tree
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)
from cacheon.target_catalog import (
    TargetCatalog,
    TargetCatalogError,
    TargetResolutionError,
)


STATIC_SCREEN_SCHEMA = "cacheon.eval.b300-static-screen.v1"
PIPELINE_SCREEN_SCHEMA = "cacheon.eval.b300-build-abi-graph-screen.v1"
SCREEN_EVIDENCE_SCHEMA = "cacheon.eval.b300-screen-stage-evidence.v1"

_PIPELINE_STAGES = ("build", "abi", "graph")
# Retained verbatim in the coordinator identity digest: the isolated eager
# execution mode was deleted, and sealed screen deployments must replay.
_PIPELINE_EXECUTION_MODE = "resident"
_TREE_METADATA = "metadata/cacheon_engine_tree.json"


class B300ScreenStagesError(RuntimeError):
    """A screen request or sealed deployment authority is inconsistent."""


class _CandidateStaticFailure(ValueError):
    pass


class B300ScreenPlanResolver(Protocol):
    """Deployment-owned construction of exact closed OCI screen inputs."""

    def __call__(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
    ) -> "B300ScreenExecutionPlan": ...


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise B300ScreenStagesError(str(exc)) from None


def _elapsed_ms(started: float) -> int:
    return max(1, round((time.monotonic() - started) * 1_000))


def _stage_result(
    *,
    manifest: ArenaServiceManifest,
    candidate: ArenaCandidateBinding,
    stage: str,
    grade: ScreenGrade,
    reason: str,
    authority_digest: str,
    started: float,
    facts: dict[str, object] | None = None,
) -> ScreenStageResult:
    evidence = canonical_digest(
        SCREEN_EVIDENCE_SCHEMA,
        {
            "authority_digest": authority_digest,
            "candidate_digest": candidate.digest,
            "facts": dict(sorted((facts or {}).items())),
            "grade": grade.value,
            "publication_digest": candidate.publication.digest,
            "reason": reason,
            "screen_attempt": candidate.screen_attempt,
            "service_digest": manifest.digest,
            "stage": stage,
        },
    )
    # The receipt keeps the bounded exact diagnostic in clear next to the digest
    # that seals it. Candidate-controlled text is evidence here, not authority.
    exception_type = (facts or {}).get("exception_type")
    exception_detail = (facts or {}).get("exception_detail")
    stated = (
        f"{reason} ({exception_type}: {exception_detail})"
        if isinstance(exception_type, str)
        and exception_type
        and isinstance(exception_detail, str)
        and exception_detail
        else f"{reason} ({exception_type})"
        if isinstance(exception_type, str) and exception_type
        else reason
    )
    stated = stated.encode("unicode_escape", "backslashreplace").decode("ascii")
    if len(stated) > MAX_SCREEN_REASON_CHARS:
        stated = stated[: MAX_SCREEN_REASON_CHARS - 3] + "..."
    return ScreenStageResult(stage, grade, evidence, _elapsed_ms(started), stated)


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    names = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, name) == getattr(right, name) for name in names)


def _read_inventory_file(
    publication: WorkerBundlePublication,
    relative: str,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    path = publication.root.joinpath(*PurePosixPath(relative).parts)
    descriptor = -1
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if not _same_stat(before, opened):
            raise OSError("worker publication changed while opening")
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise OSError("worker publication was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OSError("worker publication grew while reading")
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        not _same_stat(opened, after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise OSError("worker publication inventory changed")
    return raw


def _reopen_publication(candidate: ArenaCandidateBinding) -> WorkerBundlePublication:
    publication = candidate.publication
    try:
        reopened = reopen_worker_bundle(
            publication.root,
            publication.content_hash,
            expected_receipt_digest=publication.digest,
        )
    except (WorkerBundlePublicationError, OSError, TypeError, ValueError) as exc:
        raise B300ScreenStagesError("worker publication cannot be reopened") from exc
    if reopened != publication:
        raise B300ScreenStagesError("worker publication identity changed")
    return reopened


@contextmanager
def _project_worker_inventory(
    publication: WorkerBundlePublication,
) -> Iterator[Path]:
    """Copy only miner-owned inventory, excluding the host carrier manifest."""

    with tempfile.TemporaryDirectory(prefix="cacheon-b300-static-") as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir(mode=0o700)
        for row in publication.files:
            raw = _read_inventory_file(
                publication,
                row.path,
                row.sha256,
                row.size,
            )
            target = root.joinpath(*PurePosixPath(row.path).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(raw)
            target.chmod(0o600)
        if content_hash(root) != publication.content_hash:
            raise B300ScreenStagesError(
                "projected worker inventory differs from committed content"
            )
        yield root


def _caused_by_oserror(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, OSError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _validate_static_candidate(
    root: Path,
    candidate: ArenaCandidateBinding,
    catalog: TargetCatalog,
) -> InspectedContribution:
    manifest = load_manifest(root)
    for path in sorted(root.rglob("*.py")):
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise _CandidateStaticFailure("candidate Python is not UTF-8") from exc
    inspected = inspect_contribution(root, catalog=catalog)
    declared_cuda = all_declared_cuda_sources(root, manifest)
    declared_patches = all_declared_dep_patches(root, manifest)
    scan = scan_tree(
        root,
        declared_cuda_sources=declared_cuda,
        declared_dep_patches=declared_patches,
    )
    if not scan.ok:
        raise _CandidateStaticFailure("recursive candidate policy rejected bytes")
    reservation = candidate.reservation
    target = catalog.require(reservation.target_id)
    if (
        inspected.manifest != manifest
        or inspected.target_id != reservation.target_id
        or inspected.target_spec_digest
        != catalog.target_spec_digest(reservation.target_id)
        or inspected.selected_delta_digest != reservation.selected_delta_digest
        or target.members != reservation.target_members
    ):
        raise _CandidateStaticFailure(
            "candidate static identity differs from finalized reservation"
        )
    return inspected


class B300StaticScreenAdapter:
    """Exact static stage over immutable publication bytes and a sealed catalog."""

    def __init__(
        self,
        catalog: TargetCatalog,
        *,
        required_slot_quant: tuple[tuple[str, str], ...] = (),
    ) -> None:
        requirements = tuple(required_slot_quant)
        if (
            type(catalog) is not TargetCatalog
            or type(required_slot_quant) is not tuple
            or any(
                type(row) is not tuple
                or len(row) != 2
                or any(not isinstance(value, str) or not value for value in row)
                for row in requirements
            )
        ):
            raise B300ScreenStagesError("static screen catalog is not exact")
        if requirements != tuple(sorted(set(requirements))):
            raise B300ScreenStagesError("static screen quant policy is not canonical")
        try:
            for slot, _quant in requirements:
                catalog.require(slot)
        except TargetCatalogError as exc:
            raise B300ScreenStagesError(
                "static screen quant policy names an unknown target"
            ) from exc
        self.catalog = catalog
        self._catalog_digest = catalog.digest
        self._required_slot_quant = requirements
        self.identity_digest = canonical_digest(
            STATIC_SCREEN_SCHEMA,
            {
                "catalog_digest": catalog.digest,
                "recursive_policy": "cacheon.sandbox.scan_tree",
                "required_slot_quant": [list(row) for row in requirements],
                "source_inspection": "cacheon.engine_tree.inspect_contribution",
                "worker_publication": "cacheon.chain.worker-bundle-publication",
            },
        )

    def _quant_mismatch(
        self,
        inspected: InspectedContribution,
        target_members: tuple[str, ...],
    ) -> tuple[str, str] | None:
        metadata = dict(inspected.metadata)
        for slot, required_quant in self._required_slot_quant:
            if slot not in target_members:
                continue
            variants = inspected.manifest.ops_for(slot)
            if not variants:
                continue
            accepts = False
            for operation in variants:
                raw = None if operation.metadata is None else metadata[operation.metadata]
                value = None if raw is None else json.loads(raw.decode("utf-8"))
                eligibility = eligibility_from_metadata(
                    value, operation.dtypes, operation.architectures
                )
                accepts = accepts or required_quant in eligibility.quant
            if not accepts:
                return slot, required_quant
        return None

    def handler(self) -> B300ScreenStageHandler:
        return B300ScreenStageHandler(
            "static",
            self.identity_digest,
            (),
            self.run_screen,
        )

    def run_screen(
        self,
        manifest: ArenaServiceManifest,
        policy: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
    ) -> ScreenStageResult:
        started = time.monotonic()
        if (
            type(manifest) is not ArenaServiceManifest
            or type(policy) is not ScreenStagePolicy
            or policy.stage != "static"
            or type(candidate) is not ArenaCandidateBinding
        ):
            raise B300ScreenStagesError("static screen request is not exact")
        if self.catalog.digest != self._catalog_digest:
            return _stage_result(
                manifest=manifest,
                candidate=candidate,
                stage="static",
                grade=ScreenGrade.NO_DECISION,
                reason="static_authority_changed",
                authority_digest=self.identity_digest,
                started=started,
            )
        try:
            publication = _reopen_publication(candidate)
            with _project_worker_inventory(publication) as projected:
                inspected = _validate_static_candidate(
                    projected,
                    candidate,
                    self.catalog,
                )
            _reopen_publication(candidate)
        except Exception as exc:
            deterministic = isinstance(
                exc,
                (
                    _CandidateStaticFailure,
                    ManifestError,
                    EngineTreeError,
                    RebuildError,
                    TargetCatalogError,
                    TargetResolutionError,
                ),
            ) and not _caused_by_oserror(exc)
            if deterministic:
                try:
                    _reopen_publication(candidate)
                except Exception:
                    deterministic = False
            return _stage_result(
                manifest=manifest,
                candidate=candidate,
                stage="static",
                grade=(ScreenGrade.FAIL if deterministic else ScreenGrade.NO_DECISION),
                reason=("static_policy" if deterministic else "static_infrastructure"),
                authority_digest=self.identity_digest,
                started=started,
                facts={"exception_type": type(exc).__name__},
            )
        mismatch = self._quant_mismatch(
            inspected, candidate.reservation.target_members
        )
        if mismatch is not None:
            slot, required_quant = mismatch
            return _stage_result(
                manifest=manifest,
                candidate=candidate,
                stage="static",
                grade=ScreenGrade.FAIL,
                reason="static_runtime_quant_mismatch",
                authority_digest=self.identity_digest,
                started=started,
                facts={"required_quant": required_quant, "slot": slot},
            )
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage="static",
            grade=ScreenGrade.PASS,
            reason="static_verified",
            authority_digest=self.identity_digest,
            started=started,
            facts={
                "catalog_digest": self.catalog.digest,
                "selected_delta_digest": inspected.selected_delta_digest,
                "target_id": inspected.target_id,
                "target_spec_digest": inspected.target_spec_digest,
            },
        )


@dataclass(frozen=True)
class B300ScreenExecutionPlan:
    """Candidate-bound graph launch and build binding from one sealed resolver."""

    service_digest: str
    candidate_digest: str
    screen_attempt: int
    selected_delta_digest: str
    graph_launch: EngineLaunchSpec
    binding: TrustedLaunchBinding
    model_mount: TrustedArenaModelMountReceipt
    deadline: float

    def __post_init__(self) -> None:
        for field in (
            "service_digest",
            "candidate_digest",
            "selected_delta_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.screen_attempt) is not int or self.screen_attempt <= 0:
            raise B300ScreenStagesError("screen plan attempt is invalid")
        if (
            type(self.graph_launch) is not EngineLaunchSpec
            or type(self.binding) is not TrustedLaunchBinding
            or type(self.model_mount) is not TrustedArenaModelMountReceipt
            or isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (int, float))
            or not math.isfinite(float(self.deadline))
        ):
            raise B300ScreenStagesError("screen execution plan is not exact")
        object.__setattr__(self, "deadline", float(self.deadline))


def _executor_policy_digest(executor: OCIEngineExecutor) -> str:
    config = executor.config
    limits = {
        field.name: getattr(config.native_limits, field.name)
        for field in fields(config.native_limits)
    }
    return canonical_digest(
        "cacheon.eval.b300-screen-executor-policy.v1",
        {
            "dependency_policy_digest": (
                config.prebuild.policy.dependency_policy_digest
            ),
            "device_configuration_digest": (
                executor.device_policy.configuration_sha256
            ),
            "device_policy_digest": executor.device_policy.policy_sha256,
            "executor_id": config.prebuild.executor_id,
            "native_limits": limits,
            "prebuild_resource_policy_digest": (
                config.prebuild.policy.resource_policy_digest
            ),
            "runtime_policy_digest": config.runtime.digest,
        },
    )


def _read_tree_metadata(tree: MaterializedEngineTree) -> dict[str, object]:
    row = next((item for item in tree.files if item.path == _TREE_METADATA), None)
    if row is None:
        raise B300ScreenStagesError("materialized tree lacks contribution metadata")
    path = tree.root.joinpath(*PurePosixPath(_TREE_METADATA).parts)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise B300ScreenStagesError("materialized tree metadata is unavailable") from exc
    if len(raw) != row.size or hashlib.sha256(raw).hexdigest() != row.sha256:
        raise B300ScreenStagesError("materialized tree metadata changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise B300ScreenStagesError("materialized tree metadata is malformed") from exc
    if type(value) is not dict:
        raise B300ScreenStagesError("materialized tree metadata is not an object")
    return value


def _resolve_candidate_tree(
    plan: B300ScreenExecutionPlan,
    candidate: ArenaCandidateBinding,
    catalog: TargetCatalog,
    *,
    launch: EngineLaunchSpec,
) -> ResolvedEngineLaunch:
    resolved = resolve_engine_launch(launch, plan.binding)
    metadata = _read_tree_metadata(resolved.materialized_tree)
    contributions = metadata.get("contributions")
    if not isinstance(contributions, list):
        raise B300ScreenStagesError("materialized contribution inventory is absent")
    matching = [
        row
        for row in contributions
        if isinstance(row, dict)
        and row.get("target_id") == candidate.reservation.target_id
    ]
    if len(matching) != 1:
        raise B300ScreenStagesError("candidate target is absent from materialized tree")
    row = matching[0]
    if (
        row.get("selected_delta_digest")
        != candidate.reservation.selected_delta_digest
        or row.get("source_digest") != candidate.publication.content_hash
        or row.get("source_kind") != "proposal_artifact"
        or row.get("target_spec_digest")
        != catalog.target_spec_digest(candidate.reservation.target_id)
    ):
        raise B300ScreenStagesError(
            "materialized tree contains another candidate authority"
        )
    return resolved


def _decode_canonical_json(payload: bytes) -> dict[str, object]:
    def reject_float(_value: str) -> None:
        raise B300ScreenStagesError("screen evidence contains a JSON float")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise B300ScreenStagesError("screen evidence repeats a JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject_float,
            parse_constant=reject_float,
            object_pairs_hook=pairs,
        )
    except B300ScreenStagesError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise B300ScreenStagesError("screen evidence is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise B300ScreenStagesError("screen evidence is not canonical")
    return value


@dataclass(frozen=True)
class _PipelineCarrier:
    service_digest: str
    candidate_digest: str
    screen_attempt: int
    publication_digest: str
    plan: B300ScreenExecutionPlan
    prebuild: OCIPrebuildResult
    abi_deferred: bool = False


class B300BuildABIGraphScreenAdapter:
    """Strict single-flight build→ABI→graph screen coordinator.

    Build produces and reopens the candidate-native carrier; the ABI and graph
    stages are explicitly deferred to the final resident stage.  The resident
    swap acknowledgement proves the all-rank slot registration and its read
    forces graph recapture/replay without tearing down the stock model between
    arrivals.  These stages are routing-only and never crown.
    """

    def __init__(
        self,
        *,
        catalog: TargetCatalog,
        executor: OCIEngineExecutor,
        plan_resolver_digest: str,
        plan_resolver: B300ScreenPlanResolver,
        evidence_policy_digest: str,
        evidence_root: str | Path,
    ) -> None:
        if type(catalog) is not TargetCatalog:
            raise B300ScreenStagesError("pipeline catalog is not exact")
        if type(executor) is not OCIEngineExecutor:
            raise B300ScreenStagesError("pipeline executor is not exact")
        if not callable(plan_resolver):
            raise B300ScreenStagesError("pipeline plan resolver is not callable")
        self.catalog = catalog
        self.executor = executor
        self.evidence_root = prepare_evidence_root(Path(evidence_root))
        self._catalog_digest = catalog.digest
        self._executor_digest = _executor_policy_digest(executor)
        self._plan_resolver = plan_resolver
        self._plan_resolver_digest = _digest(
            plan_resolver_digest,
            "plan_resolver_digest",
        )
        self._evidence_policy_digest = _digest(
            evidence_policy_digest,
            "evidence_policy_digest",
        )
        self.identity_digest = canonical_digest(
            PIPELINE_SCREEN_SCHEMA,
            {
                "catalog_digest": self._catalog_digest,
                "evidence_policy_digest": self._evidence_policy_digest,
                "execution_mode": _PIPELINE_EXECUTION_MODE,
                "executor_policy_digest": self._executor_digest,
                "plan_resolver_digest": self._plan_resolver_digest,
                "stages": list(_PIPELINE_STAGES),
            },
        )
        self._active: _PipelineCarrier | None = None
        self._closed = False
        self._lock = threading.RLock()

    def handlers(
        self,
        resource_ids: tuple[str, ...],
    ) -> tuple[B300ScreenStageHandler, ...]:
        return tuple(
            B300ScreenStageHandler(
                stage,
                canonical_digest(
                    PIPELINE_SCREEN_SCHEMA,
                    {"coordinator_digest": self.identity_digest, "stage": stage},
                ),
                resource_ids,
                self.run_screen,
            )
            for stage in _PIPELINE_STAGES
        )

    def close(self) -> None:
        with self._lock:
            self._active = None
            self._closed = True

    def run_screen(
        self,
        manifest: ArenaServiceManifest,
        policy: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
    ) -> ScreenStageResult:
        started = time.monotonic()
        if (
            type(manifest) is not ArenaServiceManifest
            or type(policy) is not ScreenStagePolicy
            or policy.stage not in _PIPELINE_STAGES
            or type(candidate) is not ArenaCandidateBinding
        ):
            raise B300ScreenStagesError("pipeline screen request is not exact")
        with self._lock:
            if self._closed:
                raise B300ScreenStagesError("pipeline screen coordinator is closed")
            if (
                self.catalog.digest != self._catalog_digest
                or _executor_policy_digest(self.executor) != self._executor_digest
            ):
                self._active = None
                return self._no_decision(
                    manifest,
                    candidate,
                    policy.stage,
                    "pipeline_authority_changed",
                    started,
                )
            expected = (
                "build"
                if self._active is None
                else ("graph" if self._active.abi_deferred else "abi")
            )
            if policy.stage != expected or (
                self._active is not None
                and (
                    self._active.service_digest != manifest.digest
                    or self._active.candidate_digest != candidate.digest
                    or self._active.screen_attempt != candidate.screen_attempt
                    or self._active.publication_digest != candidate.publication.digest
                )
            ):
                self._active = None
                return self._no_decision(
                    manifest,
                    candidate,
                    policy.stage,
                    "pipeline_stage_order",
                    started,
                )
            if policy.stage == "build":
                return self._run_build(manifest, candidate, started)
            if policy.stage == "abi":
                return self._defer_abi_to_resident(manifest, candidate, started)
            return self._defer_graph_to_resident(manifest, candidate, started)

    def _no_decision(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        stage: str,
        reason: str,
        started: float,
        exc: Exception | None = None,
    ) -> ScreenStageResult:
        facts = {} if exc is None else {"exception_type": type(exc).__name__}
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage=stage,
            grade=ScreenGrade.NO_DECISION,
            reason=reason,
            authority_digest=self.identity_digest,
            started=started,
            facts=facts,
        )

    def _validate_plan(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        plan: B300ScreenExecutionPlan,
    ) -> None:
        if type(plan) is not B300ScreenExecutionPlan:
            raise B300ScreenStagesError("plan resolver returned an untyped plan")
        runtime = manifest.runtime
        graph = plan.graph_launch
        if (
            plan.service_digest != manifest.digest
            or plan.candidate_digest != candidate.digest
            or plan.screen_attempt != candidate.screen_attempt
            or plan.selected_delta_digest
            != candidate.reservation.selected_delta_digest
            or graph.arena_digest != manifest.digest
            or graph.runtime_digest != runtime.runtime_digest
            or graph.base_engine_digest != runtime.base_engine_digest
            or graph.validator_overlay_digest != runtime.validator_overlay_digest
            or graph.worker_distribution_digest
            != runtime.worker_distribution_digest
            or graph.model_revision_digest != runtime.model_revision_digest
            or graph.model_manifest_digest != runtime.model_manifest_digest
            or graph.model_content_digest != runtime.model_content_digest
            or graph.hardware.architecture != runtime.target_architecture
            or graph.hardware.topology_class != runtime.topology_class
            or graph.hardware.topology_digest != runtime.topology_digest
            or graph.hardware.visible_gpu_count != runtime.gpu_count
            or graph.hardware.tp_size != runtime.tensor_parallel_size
            or graph.hardware.device_policy_digest
            != self.executor.device_policy.policy_sha256
            or plan.model_mount.arena_digest != manifest.digest
            or plan.model_mount.model_revision_digest != runtime.model_revision_digest
            or plan.model_mount.model_manifest_digest != runtime.model_manifest_digest
            or plan.model_mount.model_content_digest != runtime.model_content_digest
            or plan.deadline <= float(self.executor.manager.clock())
        ):
            raise B300ScreenStagesError(
                "screen plan differs from candidate or arena authority"
            )
        _resolve_candidate_tree(plan, candidate, self.catalog, launch=graph)

    def _reopen_carrier(
        self,
        carrier: _PipelineCarrier,
        candidate: ArenaCandidateBinding,
        *,
        launch: EngineLaunchSpec,
    ) -> NativeArtifactPublication:
        _reopen_publication(candidate)
        _resolve_candidate_tree(
            carrier.plan,
            candidate,
            self.catalog,
            launch=launch,
        )
        publication = reopen_native_artifact(
            carrier.prebuild.publication.root,
            expected_build_spec_digest=carrier.prebuild.build_spec_digest,
            expected_publication_digest=(
                carrier.prebuild.publication.publication_digest
            ),
            limits=self.executor.config.native_limits,
        )
        if publication != carrier.prebuild.publication:
            raise B300ScreenStagesError("native build carrier identity changed")
        return publication

    def _run_build(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        started: float,
    ) -> ScreenStageResult:
        try:
            _reopen_publication(candidate)
            plan = self._plan_resolver(manifest, candidate)
            self._validate_plan(manifest, candidate, plan)
            prebuild = run_oci_prebuild(
                plan.graph_launch,
                plan.binding,
                self.executor.config.prebuild,
                manager=self.executor.manager,
                limits=self.executor.config.native_limits,
                deadline=plan.deadline,
            )
            if (
                type(prebuild) is not OCIPrebuildResult
                or prebuild.launch_digest != plan.graph_launch.digest
                or prebuild.build_spec_digest
                != plan.binding.native_build_spec.digest
            ):
                raise B300ScreenStagesError("native prebuild returned another product")
            publication = reopen_native_artifact(
                prebuild.publication.root,
                expected_build_spec_digest=prebuild.build_spec_digest,
                expected_publication_digest=prebuild.publication.publication_digest,
                limits=self.executor.config.native_limits,
            )
            if publication != prebuild.publication:
                raise B300ScreenStagesError("native prebuild publication changed")
            _reopen_publication(candidate)
        except Exception as exc:
            self._active = None
            return self._no_decision(
                manifest,
                candidate,
                "build",
                "build_infrastructure",
                started,
                exc,
            )
        self._active = _PipelineCarrier(
            manifest.digest,
            candidate.digest,
            candidate.screen_attempt,
            candidate.publication.digest,
            plan,
            prebuild,
        )
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage="build",
            grade=ScreenGrade.PASS,
            reason="native_build_reopened",
            authority_digest=self.identity_digest,
            started=started,
            facts={
                "build_spec_digest": prebuild.build_spec_digest,
                "native_publication_digest": publication.publication_digest,
            },
        )

    def _defer_abi_to_resident(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        started: float,
    ) -> ScreenStageResult:
        """Retain the built carrier for the resident all-rank swap check."""

        carrier = self._active
        assert carrier is not None
        try:
            publication = self._reopen_carrier(
                carrier,
                candidate,
                launch=carrier.plan.graph_launch,
            )
        except Exception as exc:
            self._active = None
            return self._no_decision(
                manifest,
                candidate,
                "abi",
                "resident_abi_carrier_infrastructure",
                started,
                exc,
            )
        self._active = replace(carrier, abi_deferred=True)
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage="abi",
            grade=ScreenGrade.PASS,
            reason="resident_abi_deferred",
            authority_digest=self.identity_digest,
            started=started,
            facts={
                "build_spec_digest": carrier.prebuild.build_spec_digest,
                "native_publication_digest": publication.publication_digest,
            },
        )

    def _defer_graph_to_resident(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        started: float,
    ) -> ScreenStageResult:
        """Close the carrier chain before the resident recapture/read stage."""

        carrier = self._active
        assert carrier is not None
        self._active = None
        try:
            if not carrier.abi_deferred:
                raise B300ScreenStagesError(
                    "resident graph stage lacks its deferred ABI carrier"
                )
            publication = self._reopen_carrier(
                carrier,
                candidate,
                launch=carrier.plan.graph_launch,
            )
            _reopen_publication(candidate)
        except Exception as exc:
            return self._no_decision(
                manifest,
                candidate,
                "graph",
                "resident_graph_carrier_infrastructure",
                started,
                exc,
            )
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage="graph",
            grade=ScreenGrade.PASS,
            reason="resident_graph_deferred",
            authority_digest=self.identity_digest,
            started=started,
            facts={
                "build_spec_digest": carrier.prebuild.build_spec_digest,
                "native_publication_digest": publication.publication_digest,
            },
        )

def compose_b300_non_serving_screen_handlers(
    static: B300StaticScreenAdapter,
    pipeline: B300BuildABIGraphScreenAdapter,
    *,
    pipeline_resource_ids: tuple[str, ...],
) -> tuple[B300ScreenStageHandler, ...]:
    """Return the provider's exact static/build/ABI/graph handler sequence."""

    if (
        type(static) is not B300StaticScreenAdapter
        or type(pipeline) is not B300BuildABIGraphScreenAdapter
    ):
        raise B300ScreenStagesError("screen adapters are not exact")
    return (static.handler(), *pipeline.handlers(pipeline_resource_ids))


__all__ = [
    "B300BuildABIGraphScreenAdapter",
    "B300ScreenExecutionPlan",
    "B300ScreenPlanResolver",
    "B300ScreenStagesError",
    "B300StaticScreenAdapter",
    "PIPELINE_SCREEN_SCHEMA",
    "STATIC_SCREEN_SCHEMA",
    "compose_b300_non_serving_screen_handlers",
]
