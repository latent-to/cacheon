"""Sealed qualification commission identity and digest prediction.

The load-bearing property: the digests predicted from only the sealed
commission block, the target catalog, and the sealed prompt identity must
equal the digests a real ``B300QualificationConstructionAuthority`` derives
when composed from those same sealed identities.  That equality is what lets
the screen deployment declare the qualification policy digest inside its
manifest before any candidate work exists, without the circular dependency on
the factory's self-derived builder identity.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval import b300_sealed_qualification_commission as sealed
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300QualificationDeploymentError,
    B300RegisteredProfileAuthority,
)
from cacheon.eval.b300_registered_qualification_inputs import (
    B300RegisteredQualificationError,
    registered_b300_member_contract_projection,
)
from tests.support.b300 import (
    GLM53_REGISTERED_TARGET_IDS,
    M3_REGISTERED_TARGET_IDS,
    StubHiddenJudge as _Judge,
)
from cacheon.stack_identity import canonical_json_bytes
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _resident_count_quality(catalog, evidence_root: Path):
    """Build one exact, reopenable count authority for composition tests."""

    from cacheon.eval.registered_resident_count_quality import (
        B300ResidentCountQualityCapability,
    )
    from cacheon.eval.resident_count_quality import (
        publish_resident_count_observation,
        reopen_resident_count_stock,
        seal_resident_count_stock_authority,
    )
    from tests.test_registered_resident_count_quality import _product

    evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence_root.chmod(0o700)
    product = _product(
        evidence_root.parent,
        catalog,
        M3_REGISTERED_TARGET_IDS[0],
        f"{evidence_root.name}-authority",
    )
    stock_observation = reopen_resident_count_stock(
        product.root,
        product.stock,
    )
    stock = seal_resident_count_stock_authority(
        evidence_root,
        publish_resident_count_observation(evidence_root, stock_observation),
        policy=product.stock.policy,
    )
    return B300ResidentCountQualityCapability(
        catalog,
        product.plan.envelope,
        product.plan.prompt_batches,
        product.plan.selected_ordinals,
        product.plan.batch_shape,
        product.plan.admission,
        stock,
        product.judge,
    )


def _resident_pair_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_digest: str,
):
    """Commission a service-bound pair authority and return its executors."""

    import tests.test_b300_resident_pair_factory as pair_fixtures

    harness = pair_fixtures._LifetimeHarness(monkeypatch)
    executors = []
    commissioned = pair_fixtures._commissioned(
        tmp_path / "resident-pair-authority",
        harness,
        executors,
        identity="qualification",
    )
    factory = _rebind_resident_pair_factory(
        commissioned.factory,
        service_digest,
    )
    return factory, tuple(executors)


@pytest.fixture
def resident_pair_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    executors = []
    sequence = 0

    def commission(service_digest: str):
        nonlocal sequence
        sequence += 1
        factory, created = _resident_pair_factory(
            tmp_path / f"pair-{sequence}",
            monkeypatch,
            service_digest,
        )
        executors.extend(created)
        return factory

    yield commission
    for executor in executors:
        executor.manager.close()


def _rebind_resident_pair_factory(factory, service_digest: str):
    from cacheon.eval.b300_resident_pair_factory import (
        B300CommissionedResidentPairFactory,
    )
    from cacheon.eval.oci_backend import expected_runtime_preflight

    model_mount = replace(factory.model_mount, arena_digest=service_digest)
    plans = []
    for row in factory.lane_plans:
        launch = replace(row.stock_launch, arena_digest=service_digest)
        expected = expected_runtime_preflight(
            launch,
            row.stock_binding.runtime_preflight_receipt,
        )
        plans.append(
            replace(
                row,
                stock_launch=launch,
                resident_plan=replace(
                    row.resident_plan,
                    launch_digest=launch.digest,
                    expected_preflight=expected,
                ),
                speed_workload=replace(
                    row.speed_workload,
                    launch_digest=launch.digest,
                    expected_preflight=expected,
                ),
            )
        )
    readiness = replace(factory.readiness, service_digest=service_digest)
    return B300CommissionedResidentPairFactory(
        service_digest=service_digest,
        readiness=readiness,
        lane_pair=factory.lane_pair,
        lane_plans=tuple(plans),
        model_mount=model_mount,
        swap_intake_root=factory.swap_intake_root,
        start_timeout_s=2.0,
        request_timeout_s=2.0,
        close_timeout_s=2.0,
        clock=factory.clock,
    )


def _block() -> dict[str, object]:
    return {
        "schema": sealed.QUALIFICATION_COMMISSION_SCHEMA,
        "builder_source_digest": _h("builder-source"),
        "candidate_binding_builder_digest": _h("binding-builder"),
        "graph_facts_builder_digest": _h("graph-facts"),
        "resident_count_quality_builder_digest": _h("resident-count-builder"),
        "selection_store_digest": _h("selection-store"),
        "source_resolver_digest": _h("source-resolver"),
        "support_policy_digest": _h("support-policy"),
        "verification_policy_digest": _h("verification-policy"),
        "policy": {
            "audit_minimum_calls": 4,
            "hidden_tasks_per_prompt": 2,
            "hidden_tasks_required": True,
            "nll_tail_threshold": "0.35",
            "select_count": 8,
            "tokens_per_prompt": 256,
            "topk_width": 16,
        },
        "session": {
            "conditioning_count": 2,
            "temperature": "0",
            "warmup_count": 1,
        },
        "resident_speed": {
            "max_conditioning_slowdown": "1.35",
            "max_qualification_seconds": 7200,
            "max_stage_seconds": 900,
            "max_window_scatter": "0.25",
            "min_windows": 3,
        },
    }


def test_sealed_commission_block_round_trips() -> None:
    block = _block()
    assert sealed.sealed_qualification_commission(block) is block
    assert canonical_json_bytes(block)


def test_pre_catalog_expansion_commission_schema_is_rejected() -> None:
    block = _block()
    block["schema"] = "cacheon-private-b300-qualification-commission-v1"
    with pytest.raises(
        B300RegisteredQualificationError,
        match="commission block is not closed",
    ):
        sealed.sealed_qualification_commission(block)


def _mutations() -> list[tuple[str, dict[str, object]]]:
    cases: list[tuple[str, dict[str, object]]] = []

    def case(name: str, **overrides: object) -> None:
        block = _block()
        for dotted, value in overrides.items():
            head, _, tail = dotted.partition("__")
            if tail:
                inner = dict(block[head])
                if value is None:
                    inner.pop(tail)
                else:
                    inner[tail] = value
                block[head] = inner
            elif value is None:
                block.pop(head)
            else:
                block[head] = value
        cases.append((name, block))

    case("wrong schema", schema="cacheon-private-b300-qualification-v0")
    case("missing digest field", builder_source_digest=None)
    case("short digest", selection_store_digest="ab" * 8)
    case("uppercase digest", source_resolver_digest=_h("x").upper())
    case("open policy block", policy__surprise=1)
    case("string tokens", policy__tokens_per_prompt="256")
    case("boolean tokens", policy__tokens_per_prompt=True)
    case("select count below pair", policy__select_count=1)
    case("non-string tail threshold", policy__nll_tail_threshold=0.35)
    case("non-bool hidden requirement", policy__hidden_tasks_required=1)
    case("open session block", session__extra=0)
    case("negative warmup", session__warmup_count=-1)
    case("non-string temperature", session__temperature=0.0)
    case("negative temperature", session__temperature="-0.5")
    case("open speed block", resident_speed__extra=0)
    case("zero stage budget", resident_speed__max_stage_seconds=0)
    case("noncanonical scatter", resident_speed__max_window_scatter="0.050")
    return cases


@pytest.mark.parametrize(
    "name,block", _mutations(), ids=[name for name, _ in _mutations()]
)
def test_sealed_commission_block_fails_closed(name: str, block) -> None:
    with pytest.raises(B300RegisteredQualificationError):
        sealed.sealed_qualification_commission(block)

    # Extra top-level fields are also rejected, independent of the mutation.
    widened = _block()
    widened["operator_note"] = "no"
    with pytest.raises(B300RegisteredQualificationError):
        sealed.sealed_qualification_commission(widened)


def test_profile_rows_cover_all_registered_targets_and_bind_member_authority() -> None:
    catalog = default_target_catalog()
    rows = sealed.sealed_qualification_profile_rows(
        catalog,
        registered_target_ids=M3_REGISTERED_TARGET_IDS,
        builder_source_digest=_h("reviewed-one"),
    )
    assert tuple(target for target, _spec, _resolver in rows) == (
        M3_REGISTERED_TARGET_IDS
    )
    assert all(
        spec == catalog.target_spec_digest(target) for target, spec, _ in rows
    )
    assert sealed.predicted_qualification_registry_digest(
        catalog,
        registered_target_ids=GLM53_REGISTERED_TARGET_IDS,
        builder_source_digest=_h("reviewed-one"),
    ) != sealed.predicted_qualification_registry_digest(
        catalog,
        registered_target_ids=M3_REGISTERED_TARGET_IDS,
        builder_source_digest=_h("reviewed-one"),
    )
    other = sealed.sealed_qualification_profile_rows(
        catalog,
        registered_target_ids=M3_REGISTERED_TARGET_IDS,
        builder_source_digest=_h("reviewed-two"),
    )
    assert {resolver for _, _, resolver in rows}.isdisjoint(
        {resolver for _, _, resolver in other}
    )
    projection = registered_b300_member_contract_projection(
        catalog, GLM53_REGISTERED_TARGET_IDS
    )
    atomic = next(row for row in projection if row.kind == "atomic")
    assert len(atomic.members) == 2
    assert tuple(row.slot_id for row in atomic.member_contracts) == atomic.members
    assert "contract_digest" not in atomic.to_dict()
    with pytest.raises(B300RegisteredQualificationError):
        sealed.sealed_qualification_profile_rows(
            catalog,
            registered_target_ids=M3_REGISTERED_TARGET_IDS,
            builder_source_digest="not-a-digest",
        )


def test_predicted_digests_equal_a_real_composed_construction(
    tmp_path: Path,
) -> None:
    catalog = default_target_catalog()
    block = _block()
    judge = _Judge()
    selection_policy_digest = _h("selection-policy")
    rows = sealed.sealed_qualification_profile_rows(
        catalog,
        registered_target_ids=M3_REGISTERED_TARGET_IDS,
        builder_source_digest=block["builder_source_digest"],
    )
    profiles = tuple(
        B300RegisteredProfileAuthority(
            target_id,
            spec_digest,
            resolver_digest,
            lambda _candidate, _prepared: object(),
        )
        for target_id, spec_digest, resolver_digest in rows
    )
    stack = EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base-engine"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    count_quality = _resident_count_quality(
        catalog,
        tmp_path / "evidence",
    )
    construction = B300QualificationConstructionAuthority(
        catalog=catalog,
        registered_target_ids=M3_REGISTERED_TARGET_IDS,
        profiles=profiles,
        incumbent_stack=stack,
        incumbent_tree_digest=_h("incumbent-tree"),
        pristine_stack=stack,
        pristine_tree_digest=_h("incumbent-tree"),
        evidence_root=tmp_path / "evidence",
        evidence_policy_digest=sealed.QUALIFICATION_EVIDENCE_POLICY_DIGEST,
        builder_source_digest=block["builder_source_digest"],
        selection_store_digest=block["selection_store_digest"],
        resident_count_quality_builder_digest=block[
            "resident_count_quality_builder_digest"
        ],
        resident_count_quality=count_quality,
        secret_loader=lambda _reference: b"s" * 32,
        plan_builder=lambda _cohort, _secret: object(),
        entropy_provider_digest=sealed.declared_qualification_entropy_digest(
            selection_policy_digest
        ),
        entropy_provider=lambda *_args: None,
        hidden_judge=judge,
        deadline_policy_digest=sealed.declared_qualification_deadline_digest(),
        deadline_provider=lambda _cohort: time.monotonic() + 600.0,
    )
    assert construction.profile_registry_digest == (
        sealed.predicted_qualification_registry_digest(
            catalog,
            registered_target_ids=M3_REGISTERED_TARGET_IDS,
            builder_source_digest=block["builder_source_digest"],
        )
    )
    assert construction.qualification_builder_digest == (
        sealed.predicted_qualification_builder_digest(
            catalog,
            registered_target_ids=M3_REGISTERED_TARGET_IDS,
            builder_source_digest=block["builder_source_digest"],
            selection_store_digest=block["selection_store_digest"],
            resident_count_quality_builder_digest=block[
                "resident_count_quality_builder_digest"
            ],
        )
    )
    assert construction.qualification_policy_digest == (
        sealed.predicted_qualification_policy_digest(
            catalog,
            registered_target_ids=M3_REGISTERED_TARGET_IDS,
            builder_source_digest=block["builder_source_digest"],
            selection_store_digest=block["selection_store_digest"],
            hidden_judge_binding_digest=judge.binding.digest,
            selection_policy_digest=selection_policy_digest,
            resident_count_quality_builder_digest=block[
                "resident_count_quality_builder_digest"
            ],
        )
    )
    with pytest.raises(B300QualificationDeploymentError, match="exactly cover"):
        replace(construction, profiles=construction.profiles[:-1])
    stale_profiles = list(construction.profiles)
    stale_profiles[0] = replace(
        stale_profiles[0], resolver_digest=_h("stale-member-authority")
    )
    with pytest.raises(B300QualificationDeploymentError, match="authority is stale"):
        replace(construction, profiles=tuple(stale_profiles))


def test_predicted_policy_digest_binds_each_declared_identity() -> None:
    catalog = default_target_catalog()
    block = _block()
    baseline = sealed.predicted_qualification_policy_digest(
        catalog,
        registered_target_ids=M3_REGISTERED_TARGET_IDS,
        builder_source_digest=block["builder_source_digest"],
        selection_store_digest=block["selection_store_digest"],
        hidden_judge_binding_digest=_h("binding-one"),
        selection_policy_digest=_h("selection-one"),
        resident_count_quality_builder_digest=block[
            "resident_count_quality_builder_digest"
        ],
    )
    for overrides in (
        {"builder_source_digest": _h("other-builder")},
        {"selection_store_digest": _h("other-store")},
        {"resident_count_quality_builder_digest": _h("other-count-builder")},
        {"hidden_judge_binding_digest": _h("binding-two")},
        {"selection_policy_digest": _h("selection-two")},
    ):
        variant = sealed.predicted_qualification_policy_digest(
            catalog,
            **{
                "registered_target_ids": M3_REGISTERED_TARGET_IDS,
                "builder_source_digest": block["builder_source_digest"],
                "selection_store_digest": block["selection_store_digest"],
                "hidden_judge_binding_digest": _h("binding-one"),
                "selection_policy_digest": _h("selection-one"),
                "resident_count_quality_builder_digest": block[
                    "resident_count_quality_builder_digest"
                ],
                **overrides,
            },
        )
        assert variant != baseline
