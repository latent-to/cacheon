"""Closed current-Cacheon screen stages for registered B300 arena lanes.

The static stage reopens the immutable worker publication, projects only its
miner-owned inventory, and reruns manifest, target-catalog, contribution, and
recursive source policy.  It never imports candidate Python.

Build, ABI, and graph stages share one bounded candidate-bound carrier.  Build
uses :func:`run_oci_prebuild` directly.  ABI runs a validator-owned eager OCI
session and independently regrades its raw slot-audit receipts.  Graph runs a
separate graph-enabled OCI session, publishes a verdict-free observation, then
reopens and regrades those bytes before returning a routing-only screen grade.

An exception or missing/mutated carrier is infrastructure ``NO_DECISION``.
Only deterministic policy rejection from immutable submitted bytes, or a
successfully reopened host-regraded audit witness, can produce candidate
``FAIL``.  No candidate field selects a module, command, argument, or loader.
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
from typing import Callable, Iterator, Protocol

from cacheon.arena_service import (
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
    EngineLaunchError,
    EngineLaunchSpec,
    ResolvedEngineLaunch,
    TrustedLaunchBinding,
    resolve_engine_launch,
)
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    prepare_evidence_root,
    publish_evidence,
    reopen_evidence,
)
from cacheon.eval.native_artifact import (
    NativeArtifactError,
    NativeArtifactPublication,
    reopen_native_artifact,
)
from cacheon.eval.oci_backend import (
    EngineExecutionEvidence,
    OCIBackendError,
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
)
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_prebuild import (
    OCIPrebuildError,
    OCIPrebuildResult,
    run_oci_prebuild,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_runner import AuditWitness, QualificationRunnerError
from cacheon.manifest import (
    Manifest,
    ManifestError,
    all_declared_cuda_sources,
    all_declared_dep_patches,
    load_manifest,
)
from cacheon.rebuild import RebuildError
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
ABI_EVIDENCE_DOMAIN = "cacheon.b300-screen-abi"
ABI_EVIDENCE_SCHEMA = "cacheon.b300-screen-abi.v1"
GRAPH_EVIDENCE_DOMAIN = "cacheon.b300-screen-graph"
GRAPH_EVIDENCE_SCHEMA = "cacheon.b300-screen-graph.v1"
EVIDENCE_MEDIA_TYPE = "application/json"

_PIPELINE_STAGES = ("build", "abi", "graph")
_PIPELINE_EXECUTION_MODES = frozenset({"isolated", "resident"})
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
    return ScreenStageResult(stage, grade, evidence, _elapsed_ms(started))


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

    def __init__(self, catalog: TargetCatalog) -> None:
        if type(catalog) is not TargetCatalog:
            raise B300ScreenStagesError("static screen catalog is not exact")
        self.catalog = catalog
        self._catalog_digest = catalog.digest
        self.identity_digest = canonical_digest(
            STATIC_SCREEN_SCHEMA,
            {
                "catalog_digest": catalog.digest,
                "recursive_policy": "cacheon.sandbox.scan_tree",
                "source_inspection": "cacheon.engine_tree.inspect_contribution",
                "worker_publication": "cacheon.chain.worker-bundle-publication",
            },
        )

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
    """Candidate-bound eager and graph launches from one sealed resolver."""

    service_digest: str
    candidate_digest: str
    screen_attempt: int
    selected_delta_digest: str
    eager_launch: EngineLaunchSpec
    graph_launch: EngineLaunchSpec
    binding: TrustedLaunchBinding
    model_mount: TrustedArenaModelMountReceipt
    eager_session: SessionExecutionPlan
    graph_session: SessionExecutionPlan
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
            type(self.eager_launch) is not EngineLaunchSpec
            or type(self.graph_launch) is not EngineLaunchSpec
            or type(self.binding) is not TrustedLaunchBinding
            or type(self.model_mount) is not TrustedArenaModelMountReceipt
            or type(self.eager_session) is not SessionExecutionPlan
            or type(self.graph_session) is not SessionExecutionPlan
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


def _session_plan_digest(plan: SessionExecutionPlan) -> str:
    prompts = [
        [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in batch]
        for batch in plan.prompt_batches
    ]
    return canonical_digest(
        "cacheon.eval.b300-screen-session-plan.v1",
        {
            "audit_policy_digest": (
                None if plan.audit_policy is None else plan.audit_policy.digest
            ),
            "conditioning_count": plan.conditioning_count,
            "engine_config_digest": plan.engine_config.digest,
            "expected_discovery_overlay_identity_digest": (
                plan.expected_discovery_overlay_identity_digest
            ),
            "expected_preflight_digest": plan.expected_preflight.digest,
            "launch_digest": plan.launch_digest,
            "max_new_tokens": plan.max_new_tokens,
            "prompt_digests": prompts,
            "temperature": format(plan.temperature, ".17g"),
            "top_logprobs_num": plan.top_logprobs_num,
            "warmup_count": plan.warmup_count,
        },
    )


def _same_launch_except_engine_config(
    eager: EngineLaunchSpec,
    graph: EngineLaunchSpec,
) -> bool:
    left, right = eager.to_dict(), graph.to_dict()
    left.pop("engine_config_digest")
    right.pop("engine_config_digest")
    return left == right


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


def _publish_and_reopen_witness(
    evidence_root: Path,
    witness: AuditWitness,
) -> tuple[EvidenceArtifactRef, AuditWitness]:
    payload = canonical_json_bytes(witness.to_dict())
    reference = publish_evidence(
        evidence_root,
        payload,
        domain=ABI_EVIDENCE_DOMAIN,
        media_type=EVIDENCE_MEDIA_TYPE,
        schema=ABI_EVIDENCE_SCHEMA,
    )
    reopened = AuditWitness.from_dict(
        _decode_canonical_json(reopen_evidence(evidence_root, reference))
    )
    if reopened != witness or reopened.regrade() != witness.regrade():
        raise B300ScreenStagesError("ABI witness changed after publication")
    return reference, reopened


def _reopen_witness(
    evidence_root: Path,
    reference: EvidenceArtifactRef,
    expected_digest: str,
) -> AuditWitness:
    if (
        reference.domain != ABI_EVIDENCE_DOMAIN
        or reference.media_type != EVIDENCE_MEDIA_TYPE
        or reference.schema != ABI_EVIDENCE_SCHEMA
    ):
        raise B300ScreenStagesError("ABI evidence reference is outside policy")
    witness = AuditWitness.from_dict(
        _decode_canonical_json(reopen_evidence(evidence_root, reference))
    )
    if witness.digest != expected_digest:
        raise B300ScreenStagesError("ABI evidence digest changed")
    witness.regrade()
    return witness


@dataclass(frozen=True)
class B300GraphScreenObservation:
    """Verdict-free facts from one complete graph-enabled audited session."""

    service_digest: str
    candidate_digest: str
    screen_attempt: int
    selected_delta_digest: str
    launch_digest: str
    build_spec_digest: str
    native_publication_digest: str
    session_plan_digest: str
    expected_batches: int
    observed_batches: int
    audit_witness: AuditWitness

    def __post_init__(self) -> None:
        for field in (
            "service_digest",
            "candidate_digest",
            "selected_delta_digest",
            "launch_digest",
            "build_spec_digest",
            "native_publication_digest",
            "session_plan_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if (
            type(self.screen_attempt) is not int
            or self.screen_attempt <= 0
            or type(self.expected_batches) is not int
            or self.expected_batches <= 0
            or type(self.observed_batches) is not int
            or self.observed_batches < 0
            or type(self.audit_witness) is not AuditWitness
        ):
            raise B300ScreenStagesError("graph screen observation is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_witness": self.audit_witness.to_dict(),
            "build_spec_digest": self.build_spec_digest,
            "candidate_digest": self.candidate_digest,
            "expected_batches": self.expected_batches,
            "launch_digest": self.launch_digest,
            "native_publication_digest": self.native_publication_digest,
            "observed_batches": self.observed_batches,
            "screen_attempt": self.screen_attempt,
            "selected_delta_digest": self.selected_delta_digest,
            "service_digest": self.service_digest,
            "session_plan_digest": self.session_plan_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "B300GraphScreenObservation":
        expected = {
            "audit_witness",
            "build_spec_digest",
            "candidate_digest",
            "expected_batches",
            "launch_digest",
            "native_publication_digest",
            "observed_batches",
            "screen_attempt",
            "selected_delta_digest",
            "service_digest",
            "session_plan_digest",
        }
        if type(value) is not dict or set(value) != expected:
            raise B300ScreenStagesError("graph screen evidence fields differ")
        return cls(
            **{
                **value,
                "audit_witness": AuditWitness.from_dict(value["audit_witness"]),
            }
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(GRAPH_EVIDENCE_SCHEMA, self.to_dict())


def _publish_and_reopen_graph(
    evidence_root: Path,
    observation: B300GraphScreenObservation,
) -> tuple[EvidenceArtifactRef, B300GraphScreenObservation]:
    payload = canonical_json_bytes(observation.to_dict())
    reference = publish_evidence(
        evidence_root,
        payload,
        domain=GRAPH_EVIDENCE_DOMAIN,
        media_type=EVIDENCE_MEDIA_TYPE,
        schema=GRAPH_EVIDENCE_SCHEMA,
    )
    reopened = B300GraphScreenObservation.from_dict(
        _decode_canonical_json(reopen_evidence(evidence_root, reference))
    )
    if reopened != observation or reopened.digest != observation.digest:
        raise B300ScreenStagesError("graph observation changed after publication")
    return reference, reopened


@dataclass(frozen=True)
class _PipelineCarrier:
    service_digest: str
    candidate_digest: str
    screen_attempt: int
    publication_digest: str
    plan: B300ScreenExecutionPlan
    prebuild: OCIPrebuildResult
    abi_reference: EvidenceArtifactRef | None = None
    abi_witness_digest: str | None = None
    abi_deferred: bool = False


class B300BuildABIGraphScreenAdapter:
    """Strict single-flight build→ABI→graph screen coordinator.

    ``isolated`` preserves the ordinary eager and graph launch pair.  A
    deployment with a standing resident screen lane uses ``resident``: build
    still produces and reopens the candidate-native carrier, while ABI and
    graph execution are explicitly deferred to the final resident stage.  The
    resident swap acknowledgement proves the all-rank slot registration and
    its read forces graph recapture/replay without tearing down the stock
    model between arrivals.  These stages are routing-only and never crown.
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
        execution_mode: str = "isolated",
    ) -> None:
        if type(catalog) is not TargetCatalog:
            raise B300ScreenStagesError("pipeline catalog is not exact")
        if type(executor) is not OCIEngineExecutor:
            raise B300ScreenStagesError("pipeline executor is not exact")
        if not callable(plan_resolver):
            raise B300ScreenStagesError("pipeline plan resolver is not callable")
        if execution_mode not in _PIPELINE_EXECUTION_MODES:
            raise B300ScreenStagesError("pipeline execution mode is invalid")
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
        self._execution_mode = execution_mode
        self.identity_digest = canonical_digest(
            PIPELINE_SCREEN_SCHEMA,
            {
                "catalog_digest": self._catalog_digest,
                "evidence_policy_digest": self._evidence_policy_digest,
                "execution_mode": self._execution_mode,
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
                else (
                    "abi"
                    if self._active.abi_reference is None
                    and not self._active.abi_deferred
                    else "graph"
                )
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
                if self._execution_mode == "resident":
                    return self._defer_abi_to_resident(
                        manifest, candidate, started
                    )
                return self._run_abi(manifest, candidate, started)
            if self._execution_mode == "resident":
                return self._defer_graph_to_resident(
                    manifest, candidate, started
                )
            return self._run_graph(manifest, candidate, started)

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
        eager, graph = plan.eager_launch, plan.graph_launch
        if (
            plan.service_digest != manifest.digest
            or plan.candidate_digest != candidate.digest
            or plan.screen_attempt != candidate.screen_attempt
            or plan.selected_delta_digest
            != candidate.reservation.selected_delta_digest
            or not _same_launch_except_engine_config(eager, graph)
            or eager.arena_digest != manifest.digest
            or eager.runtime_digest != runtime.runtime_digest
            or eager.base_engine_digest != runtime.base_engine_digest
            or eager.validator_overlay_digest != runtime.validator_overlay_digest
            or eager.worker_distribution_digest
            != runtime.worker_distribution_digest
            or eager.model_revision_digest != runtime.model_revision_digest
            or eager.model_manifest_digest != runtime.model_manifest_digest
            or eager.model_content_digest != runtime.model_content_digest
            or eager.hardware.architecture != runtime.target_architecture
            or eager.hardware.topology_class != runtime.topology_class
            or eager.hardware.topology_digest != runtime.topology_digest
            or eager.hardware.visible_gpu_count != runtime.gpu_count
            or eager.hardware.tp_size != runtime.tensor_parallel_size
            or eager.hardware.device_policy_digest
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
        for launch, session, eager_mode in (
            (eager, plan.eager_session, True),
            (graph, plan.graph_session, False),
        ):
            audit = session.audit_policy
            if (
                session.launch_digest != launch.digest
                or session.expected_engine_config_digest
                != launch.engine_config_digest
                or session.engine_config.digest != launch.engine_config_digest
                or session.engine_config.disable_cuda_graph is not eager_mode
                or session.engine_config.tp_size != runtime.tensor_parallel_size
                or audit is None
                or audit.expected_slots != candidate.reservation.target_members
                or audit.expected_member_count != runtime.tensor_parallel_size
            ):
                raise B300ScreenStagesError(
                    "screen session plan differs from lane authority"
                )
        _resolve_candidate_tree(plan, candidate, self.catalog, launch=graph)
        _resolve_candidate_tree(plan, candidate, self.catalog, launch=eager)

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

    def _validate_execution(
        self,
        execution: EngineExecutionEvidence,
        carrier: _PipelineCarrier,
        launch: EngineLaunchSpec,
        session: SessionExecutionPlan,
    ) -> None:
        if (
            type(execution) is not EngineExecutionEvidence
            or execution.launch_digest != launch.digest
            or execution.resource_policy_digest != self.executor.config.runtime.digest
            or execution.prebuild.build_spec_digest
            != carrier.prebuild.build_spec_digest
            or execution.prebuild.publication.publication_digest
            != carrier.prebuild.publication.publication_digest
            or execution.native_publication_digest
            != carrier.prebuild.publication.publication_digest
            or execution.session.launch_digest != launch.digest
            or execution.session.audit_policy_digest
            != session.audit_policy.digest  # type: ignore[union-attr]
            or len(execution.session.batches) != len(session.prompt_batches)
            or execution.session.warmup_count != session.warmup_count
            or execution.session.conditioning_count != session.conditioning_count
        ):
            raise B300ScreenStagesError("OCI screen execution changed its exact plan")

    def _run_abi(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        started: float,
    ) -> ScreenStageResult:
        carrier = self._active
        assert carrier is not None
        try:
            self._reopen_carrier(
                carrier,
                candidate,
                launch=carrier.plan.eager_launch,
            )
            execution = self.executor.execute(
                carrier.plan.eager_launch,
                carrier.plan.binding,
                carrier.plan.model_mount,
                carrier.plan.eager_session,
                deadline=carrier.plan.deadline,
            )
            self._validate_execution(
                execution,
                carrier,
                carrier.plan.eager_launch,
                carrier.plan.eager_session,
            )
            witness = AuditWitness.from_execution(
                execution,
                selected_delta_digest=carrier.plan.selected_delta_digest,
                policy=carrier.plan.eager_session.audit_policy,
            )
            reference, witness = _publish_and_reopen_witness(
                self.evidence_root,
                witness,
            )
            self._reopen_carrier(
                carrier,
                candidate,
                launch=carrier.plan.eager_launch,
            )
        except Exception as exc:
            self._active = None
            return self._no_decision(
                manifest,
                candidate,
                "abi",
                "abi_infrastructure",
                started,
                exc,
            )
        grade = (
            ScreenGrade.PASS
            if witness.decision is QualificationDecision.PASS
            else ScreenGrade.FAIL
        )
        if grade is ScreenGrade.PASS:
            self._active = replace(
                carrier,
                abi_reference=reference,
                abi_witness_digest=witness.digest,
            )
        else:
            self._active = None
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage="abi",
            grade=grade,
            reason=("eager_abi_verified" if grade is ScreenGrade.PASS else "eager_abi_failed"),
            authority_digest=self.identity_digest,
            started=started,
            facts={
                "artifact_sha256": reference.sha256,
                "audit_witness_digest": witness.digest,
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

    def _run_graph(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
        started: float,
    ) -> ScreenStageResult:
        carrier = self._active
        assert carrier is not None
        self._active = None
        try:
            assert carrier.abi_reference is not None
            assert carrier.abi_witness_digest is not None
            prior = _reopen_witness(
                self.evidence_root,
                carrier.abi_reference,
                carrier.abi_witness_digest,
            )
            if prior.decision is not QualificationDecision.PASS:
                raise B300ScreenStagesError("graph stage lacks a passing ABI witness")
            publication = self._reopen_carrier(
                carrier,
                candidate,
                launch=carrier.plan.graph_launch,
            )
            execution = self.executor.execute(
                carrier.plan.graph_launch,
                carrier.plan.binding,
                carrier.plan.model_mount,
                carrier.plan.graph_session,
                deadline=carrier.plan.deadline,
            )
            self._validate_execution(
                execution,
                carrier,
                carrier.plan.graph_launch,
                carrier.plan.graph_session,
            )
            witness = AuditWitness.from_execution(
                execution,
                selected_delta_digest=carrier.plan.selected_delta_digest,
                policy=carrier.plan.graph_session.audit_policy,
            )
            observation = B300GraphScreenObservation(
                manifest.digest,
                candidate.digest,
                candidate.screen_attempt,
                carrier.plan.selected_delta_digest,
                carrier.plan.graph_launch.digest,
                carrier.prebuild.build_spec_digest,
                publication.publication_digest,
                _session_plan_digest(carrier.plan.graph_session),
                len(carrier.plan.graph_session.prompt_batches),
                len(execution.session.batches),
                witness,
            )
            reference, observation = _publish_and_reopen_graph(
                self.evidence_root,
                observation,
            )
            if (
                observation.service_digest != manifest.digest
                or observation.candidate_digest != candidate.digest
                or observation.screen_attempt != candidate.screen_attempt
                or observation.selected_delta_digest
                != candidate.reservation.selected_delta_digest
                or observation.launch_digest != carrier.plan.graph_launch.digest
                or observation.build_spec_digest
                != carrier.prebuild.build_spec_digest
                or observation.native_publication_digest
                != publication.publication_digest
                or observation.session_plan_digest
                != _session_plan_digest(carrier.plan.graph_session)
                or observation.observed_batches != observation.expected_batches
            ):
                raise B300ScreenStagesError("graph observation failed independent regrade")
            passed, detail = observation.audit_witness.regrade()
            if (observation.audit_witness.decision is QualificationDecision.PASS) != passed:
                raise B300ScreenStagesError("graph audit witness changed on regrade")
            _reopen_publication(candidate)
            self._reopen_carrier(
                carrier,
                candidate,
                launch=carrier.plan.graph_launch,
            )
        except Exception as exc:
            return self._no_decision(
                manifest,
                candidate,
                "graph",
                "graph_infrastructure",
                started,
                exc,
            )
        grade = ScreenGrade.PASS if passed else ScreenGrade.FAIL
        return _stage_result(
            manifest=manifest,
            candidate=candidate,
            stage="graph",
            grade=grade,
            reason=("graph_session_verified" if passed else "graph_session_failed"),
            authority_digest=self.identity_digest,
            started=started,
            facts={
                "artifact_sha256": reference.sha256,
                "graph_observation_digest": observation.digest,
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
    "ABI_EVIDENCE_DOMAIN",
    "ABI_EVIDENCE_SCHEMA",
    "B300BuildABIGraphScreenAdapter",
    "B300GraphScreenObservation",
    "B300ScreenExecutionPlan",
    "B300ScreenPlanResolver",
    "B300ScreenStagesError",
    "B300StaticScreenAdapter",
    "GRAPH_EVIDENCE_DOMAIN",
    "GRAPH_EVIDENCE_SCHEMA",
    "PIPELINE_SCREEN_SCHEMA",
    "STATIC_SCREEN_SCHEMA",
    "compose_b300_non_serving_screen_handlers",
]
