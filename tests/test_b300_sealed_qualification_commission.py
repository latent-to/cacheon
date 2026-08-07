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
from pathlib import Path

import pytest

from cacheon.eval import b300_sealed_qualification_commission as sealed
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300RegisteredProfileAuthority,
)
from cacheon.eval.b300_registered_qualification_inputs import (
    ORDINARY_B300_TARGET_IDS,
    B300RegisteredQualificationError,
)
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _block() -> dict[str, object]:
    return {
        "schema": sealed.QUALIFICATION_COMMISSION_SCHEMA,
        "builder_source_digest": _h("builder-source"),
        "candidate_binding_builder_digest": _h("binding-builder"),
        "graph_facts_builder_digest": _h("graph-facts"),
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
            "temperature": 0.0,
            "warmup_count": 1,
        },
        "resident_speed": {
            "max_conditioning_slowdown": 1.35,
            "max_qualification_seconds": 7200,
            "max_stage_seconds": 900,
            "max_window_scatter": 0.25,
            "min_windows": 3,
        },
    }


class _Judge:
    def __init__(self) -> None:
        self.binding = HiddenJudgeBinding(
            _h("hidden-corpus"), _h("hidden-judge"), _h("hidden-policy")
        )

    def __call__(self, **_kwargs):
        raise AssertionError("identity checks must not execute the hidden judge")


def test_sealed_commission_block_round_trips() -> None:
    block = _block()
    assert sealed.sealed_qualification_commission(block) is block


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
    case("infinite temperature", session__temperature=float("inf"))
    case("negative temperature", session__temperature=-0.5)
    case("open speed block", resident_speed__extra=0)
    case("zero stage budget", resident_speed__max_stage_seconds=0)
    case("nan scatter", resident_speed__max_window_scatter=float("nan"))
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


def test_profile_rows_cover_every_ordinary_target_and_bind_the_identity() -> None:
    catalog = default_target_catalog()
    rows = sealed.sealed_qualification_profile_rows(
        catalog, builder_source_digest=_h("reviewed-one")
    )
    assert tuple(target for target, _spec, _resolver in rows) == (
        ORDINARY_B300_TARGET_IDS
    )
    assert all(
        spec == catalog.target_spec_digest(target) for target, spec, _ in rows
    )
    other = sealed.sealed_qualification_profile_rows(
        catalog, builder_source_digest=_h("reviewed-two")
    )
    assert {resolver for _, _, resolver in rows}.isdisjoint(
        {resolver for _, _, resolver in other}
    )
    with pytest.raises(B300RegisteredQualificationError):
        sealed.sealed_qualification_profile_rows(
            catalog, builder_source_digest="not-a-digest"
        )


def test_predicted_digests_equal_a_real_composed_construction(
    tmp_path: Path,
) -> None:
    catalog = default_target_catalog()
    block = _block()
    judge = _Judge()
    selection_policy_digest = _h("selection-policy")
    rows = sealed.sealed_qualification_profile_rows(
        catalog, builder_source_digest=block["builder_source_digest"]
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
    construction = B300QualificationConstructionAuthority(
        catalog=catalog,
        profiles=profiles,
        incumbent_stack=stack,
        incumbent_tree_digest=_h("incumbent-tree"),
        pristine_stack=stack,
        pristine_tree_digest=_h("incumbent-tree"),
        evidence_root=tmp_path / "evidence",
        evidence_policy_digest=sealed.QUALIFICATION_EVIDENCE_POLICY_DIGEST,
        builder_source_digest=block["builder_source_digest"],
        selection_store_digest=block["selection_store_digest"],
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
            catalog, builder_source_digest=block["builder_source_digest"]
        )
    )
    assert construction.qualification_builder_digest == (
        sealed.predicted_qualification_builder_digest(
            catalog,
            builder_source_digest=block["builder_source_digest"],
            selection_store_digest=block["selection_store_digest"],
        )
    )
    assert construction.qualification_policy_digest == (
        sealed.predicted_qualification_policy_digest(
            catalog,
            builder_source_digest=block["builder_source_digest"],
            selection_store_digest=block["selection_store_digest"],
            hidden_judge_binding_digest=judge.binding.digest,
            selection_policy_digest=selection_policy_digest,
        )
    )


def test_predicted_policy_digest_binds_each_declared_identity() -> None:
    catalog = default_target_catalog()
    block = _block()
    baseline = sealed.predicted_qualification_policy_digest(
        catalog,
        builder_source_digest=block["builder_source_digest"],
        selection_store_digest=block["selection_store_digest"],
        hidden_judge_binding_digest=_h("binding-one"),
        selection_policy_digest=_h("selection-one"),
    )
    for overrides in (
        {"builder_source_digest": _h("other-builder")},
        {"selection_store_digest": _h("other-store")},
        {"hidden_judge_binding_digest": _h("binding-two")},
        {"selection_policy_digest": _h("selection-two")},
    ):
        variant = sealed.predicted_qualification_policy_digest(
            catalog,
            **{
                "builder_source_digest": block["builder_source_digest"],
                "selection_store_digest": block["selection_store_digest"],
                "hidden_judge_binding_digest": _h("binding-one"),
                "selection_policy_digest": _h("selection-one"),
                **overrides,
            },
        )
        assert variant != baseline
