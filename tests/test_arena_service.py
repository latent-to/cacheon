from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaQualificationWork,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceError,
    ArenaServiceManifest,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
    QualificationReservation,
)


def _h(label: str) -> str:
    return (label.encode().hex() + "1" * 64)[:64]


def _manifest(**changes) -> ArenaServiceManifest:
    runtime = ArenaRuntimeIdentity(
        arena_id="minimax-m3-sm120",
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("engine"),
        validator_overlay_digest=_h("overlay"),
        worker_distribution_digest=_h("worker"),
        model_revision_digest=_h("revision"),
        model_manifest_digest=_h("manifest"),
        model_content_digest=_h("model"),
        target_architecture="sm120",
        topology_class="pcie-switch",
        topology_digest=_h("topology"),
        gpu_count=8,
        tensor_parallel_size=8,
    )
    workload = Workload(
        _h("corpus"),
        "finalized-entropy-v1",
        (WorkloadCell("s8", 8192, 1024, 64, 8),),
    )
    values = {
        "runtime": runtime,
        "workload": workload,
        "capacity": ArenaCapacityPolicy(64, 600, 8, 4),
        "qualification_policy_digest": _h("qualification-policy"),
        "provider_digest": _h("provider"),
    }
    values.update(changes)
    return ArenaServiceManifest(**values)


def _publication(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "manifest.toml").write_text("bundle_id = 'candidate'\n")
    source.chmod(0o700)
    (source / "manifest.toml").chmod(0o600)
    committed = content_hash(source)
    return publish_worker_bundle(source, tmp_path / "publications", committed)


def _binding(tmp_path: Path) -> ArenaCandidateBinding:
    publication = _publication(tmp_path)
    reservation = QualificationReservation(
        _h("reservation"),
        publication.digest,
        "attention.msa-prefill",
        _h("delta"),
        0,
        "miner-hotkey",
        10,
        0,
        0,
        ("attention.msa-prefill",),
    )
    return ArenaCandidateBinding(reservation, publication, 1)


class _Provider:
    provider_digest = _h("provider")

    def __init__(self, *, resident_baseline_executor=None):
        self.resident_baseline_executor = resident_baseline_executor
        self.plan_calls = []

    def build_qualification(self, request, state=None):
        self.plan_calls.append(request)
        reservations = tuple(row.reservation for row in request.candidates)
        authority = QualificationAuthorityManifest(
            "registered",
            _h("authority"),
            _h("source"),
            _h("commitment"),
            _h("secret"),
            tuple(row.selected_delta_digest for row in reservations),
            reservations,
        )
        factory = QualificationPlanFactory(
            authority, lambda _ref: b"s" * 32, lambda _s: None
        )
        return ArenaQualificationWork(
            factory,
            object(),
            lambda *_: None,
            lambda **_: None,
            99.0,
            request.qualification_policy_digest,
            self.resident_baseline_executor,
        )


def test_service_identity_binds_every_serving_authority() -> None:
    manifest = _manifest()
    assert len(manifest.digest) == 64
    assert manifest.service_id == f"minimax-m3-sm120@{manifest.digest}"

    variants = (
        dataclasses.replace(
            manifest,
            runtime=dataclasses.replace(manifest.runtime, runtime_digest=_h("runtime2")),
        ),
        dataclasses.replace(
            manifest,
            runtime=dataclasses.replace(
                manifest.runtime, model_content_digest=_h("model2")
            ),
        ),
        dataclasses.replace(
            manifest,
            runtime=dataclasses.replace(
                manifest.runtime, topology_digest=_h("topology2")
            ),
        ),
        dataclasses.replace(
            manifest,
            workload=dataclasses.replace(manifest.workload, prompt_corpus_digest=_h("new")),
        ),
        dataclasses.replace(
            manifest,
            capacity=dataclasses.replace(manifest.capacity, max_queue_depth=65),
        ),
        dataclasses.replace(
            manifest, qualification_policy_digest=_h("qualification-policy2")
        ),
        dataclasses.replace(manifest, provider_digest=_h("provider2")),
        dataclasses.replace(manifest, closed_targets=("attention.msa-prefill",)),
    )
    assert len({manifest.digest, *(row.digest for row in variants)}) == 9


def test_workload_requires_typed_unique_cells() -> None:
    cell = WorkloadCell("s8", 8192, 1024, 64, 8)
    with pytest.raises(ArenaServiceError, match="cells"):
        Workload(_h("corpus"), "seed-v1", ())
    with pytest.raises(ArenaServiceError, match="cells"):
        Workload(_h("corpus"), "seed-v1", (cell, cell))
    with pytest.raises(ArenaServiceError, match="positive integer"):
        WorkloadCell("s8", 8192, 0, 64, 8)


def test_provider_is_validator_supplied_and_digest_bound() -> None:
    provider = _Provider()
    provider.provider_digest = _h("forged")
    with pytest.raises(ArenaServiceError, match="implementation identity"):
        ArenaService(_manifest(), provider)


def test_only_exact_candidates_reach_qualification(tmp_path: Path) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    binding = _binding(tmp_path)
    work = service.plan_qualification((binding,))
    assert type(work) is ArenaQualificationWork
    assert work.factory.manifest.reservations == (binding.reservation,)
    assert work.resident_baseline_executor is None
    assert provider.plan_calls[0].service_digest == service.identity

    with pytest.raises(ArenaServiceError, match="no exact candidates"):
        service.plan_qualification(())
    with pytest.raises(ArenaServiceError, match="no exact candidates"):
        service.plan_qualification((binding.reservation,))
    with pytest.raises(ArenaServiceError, match="exceeds arena capacity"):
        service.plan_qualification((binding,) * 5)


def test_qualification_work_preserves_resident_baseline_executor(
    tmp_path: Path,
) -> None:
    resident_baseline_executor = object()
    provider = _Provider(
        resident_baseline_executor=resident_baseline_executor
    )
    service = ArenaService(_manifest(), provider)

    work = service.plan_qualification((_binding(tmp_path),))

    assert work.resident_baseline_executor is resident_baseline_executor


def test_provider_cannot_change_finalized_order(tmp_path: Path) -> None:
    class WrongProvider(_Provider):
        def build_qualification(self, request, state=None):
            work = super().build_qualification(request, state)
            wrong = dataclasses.replace(
                request.candidates[0].reservation, arrival_order=9
            )
            authority = dataclasses.replace(work.factory.manifest, reservations=(wrong,))
            return dataclasses.replace(
                work,
                factory=QualificationPlanFactory(
                    authority, work.factory.secret_loader, work.factory.plan_builder
                ),
            )

    service = ArenaService(_manifest(), WrongProvider())
    with pytest.raises(ArenaServiceError, match="finalized qualification order"):
        service.plan_qualification((_binding(tmp_path),))


def test_provider_cannot_change_registered_qualification_policy(tmp_path: Path) -> None:
    class WrongPolicyProvider(_Provider):
        def build_qualification(self, request, state=None):
            return dataclasses.replace(
                super().build_qualification(request, state),
                qualification_policy_digest=_h("other-qualification-policy"),
            )

    service = ArenaService(_manifest(), WrongPolicyProvider())
    with pytest.raises(ArenaServiceError, match="qualification policy"):
        service.plan_qualification((_binding(tmp_path),))


def test_arena_service_has_no_dynamic_import_authority() -> None:
    source = Path(__file__).parents[1].joinpath("cacheon", "arena_service.py").read_text()
    assert "importlib" not in source
    assert "import_module" not in source
    assert "entry_point" not in source
