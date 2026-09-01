from __future__ import annotations

from pathlib import Path

import pytest

import cacheon.chain.validator_loop as loop
from cacheon.arena_service import (
    SCREEN_STAGES, AdmissionDecision, ArenaQualificationWork,
    ArenaScreenReceipt, ArenaService, ArenaServiceRegistry, PromotionDecision,
    ScreenGrade, ScreenStageResult,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain import FinalizedRevealSnapshot, RevealedCommitment
from cacheon.chain.eval_cost import (
    EvalCostFetchError,
    EvalCostPaymentProof,
    EvalCostPolicy,
    EvalCostRequest,
    encode_payment_remark,
    quote_eval_cost,
)
from cacheon.chain.intake import FinalizedIntakeStore, IntakePolicy, IntakeScope
from cacheon.chain.payload import encode_payload
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest, QualificationIntakeBatch,
    QualificationIntakeOutcome, QualificationPlanFactory,
)


BLOCK = 90
BLOCK_HASH = "0x" + "9" * 64
SCOPE = IntakeScope("0x" + "0" * 64, 307)


def _bundle(
    root: Path,
    body: str,
    *,
    slot: str = "activation.silu_and_mul",
    entry: str = "silu_and_mul",
) -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "manifest.toml").write_text(
        'bundle_id = "test"\n'
        'abi_version = "cacheon-op-abi-v0"\n\n'
        '[[ops]]\n'
        f'slot = "{slot}"\n'
        'source = "kernels/k.py"\n'
        f'entry = "{entry}"\n'
        'dtypes = ["float32"]\n'
    )
    (root / "kernels/k.py").write_text(body)
    for directory in (root, root / "kernels"):
        directory.chmod(0o700)
    for file in (root / "manifest.toml", root / "kernels/k.py"):
        file.chmod(0o600)
    return root


def _snapshot(rows: list[tuple[str, str]]) -> FinalizedRevealSnapshot:
    reveals = tuple(
        RevealedCommitment(hotkey, payload, BLOCK, BLOCK_HASH, index)
        for index, (hotkey, payload) in enumerate(rows)
    )
    return FinalizedRevealSnapshot(BLOCK, BLOCK_HASH, reveals)


class _NoWeightsSubtensor:
    def get_block_hash(self, block):
        assert block == 0
        return SCOPE.genesis_hash


def _run(
    tmp_path,
    monkeypatch,
    snapshot,
    sources,
    *,
    head_provider=None,
    **changes,
):
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_reveal_history",
        lambda *_, **__: snapshot,
    )
    provider = head_provider or (
        lambda: (snapshot.finalized_block, snapshot.finalized_block_hash)
    )
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_head",
        lambda *_: provider(),
    )
    calls = []

    def fetcher(_url, expected, _root):
        calls.append(expected)
        return sources[expected]

    monkeypatch.setattr(loop, "fetch_bundle", fetcher)

    options = dict(
        intake_db=tmp_path / "state" / "intake.sqlite3",
        private_root=tmp_path / "private-cache",
        publication_root=tmp_path / "worker",
        intake_only=True,
    )
    options.update(changes)
    return loop.run_pass(_NoWeightsSubtensor(), 307, **options), calls, options


def test_finalized_reveal_publishes_once_and_restart_reopens(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])
    result, calls, options = _run(tmp_path, monkeypatch, snapshot, {digest: source})

    assert result.seen == 1 and len(result.reserved) == 1
    assert len(result.published) == 1 and result.decisions == {}
    assert calls == [digest]
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "published"
        assert row.publication_digest == next(iter(result.published.values()))
        assert row.arrival.content_hash == digest

    second, second_calls, _ = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert second.reserved == [] and second.published == {}
    assert second_calls == []


def test_disabled_eval_cost_ignores_v2_pointer_without_consuming_it(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )

    def unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("disabled eval-cost must not read a payment")

    monkeypatch.setattr(loop, "read_eval_cost_payment", unexpected_lookup)
    result, calls, options = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert calls == [digest] and len(result.rejected) == 0
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        assert store.all()[0].status == "published"
        assert store._db.execute(
            "SELECT COUNT(*) FROM eval_cost_payments"
        ).fetchone()[0] == 0


def test_unpaid_v1_is_failed_when_eval_cost_is_required(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [] and len(result.rejected) == 1
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason == "missing_eval_cost_payment"


def _paid_proof(digest: str) -> EvalCostPaymentProof:
    request = EvalCostRequest(netuid=307, hotkey="miner", content_hash=digest)
    quote = quote_eval_cost(
        request,
        policy=EvalCostPolicy(amount_rao=10, destination="treasury"),
        at_block=70,
    )
    return EvalCostPaymentProof(
        block=80,
        extrinsic_index=4,
        signer="coldkey",
        payer="coldkey",
        destination="treasury",
        amount_rao=10,
        remark=encode_payment_remark(request, quote),
    )


def _stub_owner(monkeypatch, dest: str = "treasury") -> None:
    monkeypatch.setattr(
        loop, "read_subnet_owner_coldkey", lambda *_args, **_kwargs: dest
    )


def test_paid_v2_is_admitted_when_eval_cost_is_required(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )
    monkeypatch.setattr(
        loop,
        "read_eval_cost_payment",
        lambda *_args, **_kwargs: _paid_proof(digest),
    )
    _stub_owner(monkeypatch)
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [digest] and len(result.rejected) == 0
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "published"
        assert row.arrival.payment_block == 80


def test_payment_to_a_stale_owner_is_invalid(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )
    monkeypatch.setattr(
        loop,
        "read_eval_cost_payment",
        lambda *_args, **_kwargs: _paid_proof(digest),
    )
    _stub_owner(monkeypatch, "owner-b")
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [] and len(result.rejected) == 1
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason == "eval_cost_payment_invalid"


def test_unrecognizable_payment_pointer_is_invalid(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )
    monkeypatch.setattr(loop, "read_eval_cost_payment", lambda *_args, **_kwargs: None)
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [] and len(result.rejected) == 1
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason == "eval_cost_payment_invalid"


def test_eval_cost_fetch_error_does_not_advance_the_cursor(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )

    def boom(*_args, **_kwargs):
        raise EvalCostFetchError("rpc blip")

    monkeypatch.setattr(loop, "read_eval_cost_payment", boom)
    with pytest.raises(EvalCostFetchError, match="rpc blip"):
        _run(
            tmp_path,
            monkeypatch,
            snapshot,
            {digest: source},
            policy=IntakePolicy(expiry_blocks=100),
            eval_cost_policy=EvalCostPolicy(amount_rao=10),
        )
    with FinalizedIntakeStore(
        tmp_path / "state" / "intake.sqlite3",
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        assert store.finalized_cursor() is None
        assert store.all() == ()


def test_malformed_finalized_payload_is_reserved_and_never_fetched(tmp_path, monkeypatch):
    snapshot = _snapshot([("miner", "not-json")])
    result, calls, options = _run(tmp_path, monkeypatch, snapshot, {})
    assert calls == [] and len(result.reserved) == 1
    assert len(result.rejected) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed" and row.reason == "invalid_payload"


def test_deterministically_unpublishable_submission_is_not_retried(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    reserved = source / ".cacheon-native-artifact.json"
    reserved.write_text("{}\n")
    reserved.chmod(0o600)
    digest = content_hash(source)
    snapshot = _snapshot(
        [("miner", encode_payload(digest, "https://example.com/a"))]
    )
    result, _calls, options = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert len(result.rejected) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason.startswith("publication_source:")


def test_reformatted_later_delta_is_copy_without_any_weight_edge(tmp_path, monkeypatch):
    first = _bundle(
        tmp_path / "first",
        "import torch\n\ndef silu_and_mul(x, out):\n"
        "    d = x.shape[-1] // 2\n"
        "    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])\n",
    )
    second = _bundle(
        tmp_path / "second",
        "import torch\n\n# formatting only\ndef silu_and_mul(x, out):\n"
        "    d = (x.shape[-1] // 2)\n"
        "    out.copy_((torch.nn.functional.silu(x[..., :d]) * x[..., d:]))\n",
    )
    first_hash, second_hash = content_hash(first), content_hash(second)
    assert first_hash != second_hash
    snapshot = _snapshot([
        ("author", encode_payload(first_hash, "https://example.com/a")),
        ("copycat", encode_payload(second_hash, "https://example.com/b")),
    ])
    result, _calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {first_hash: first, second_hash: second},
    )
    assert len(result.published) == 1 and len(result.copies) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        rows = store.all()
        assert [row.status for row in rows] == ["published", "failed"]
        assert rows[1].reason.startswith("copy_of:")


def test_live_loop_calls_batch_qualification_and_retains_fail_outcome(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])

    calls = []
    progress_events = []
    promoted_limits = []
    retained_blocks = []
    resident_baseline_executor = object()
    service = object.__new__(ArenaService)
    service.manifest = type(
        "Manifest",
        (),
        {
            "digest": "e" * 64,
            "qualification_policy_digest": "f" * 64,
            "capacity": type("Capacity", (), {"max_cohort_size": 1})(),
            "closed_targets": (),
        },
    )()
    registry = object.__new__(ArenaServiceRegistry)
    monkeypatch.setattr(ArenaServiceRegistry, "require", lambda *_: service)
    monkeypatch.setattr(ArenaService, "admit", lambda *_: AdmissionDecision.ADMIT)
    monkeypatch.setattr(
        ArenaService, "admit_qualification", lambda *_args, **_kwargs: AdmissionDecision.ADMIT
    )
    monkeypatch.setattr(
        ArenaService,
        "screen",
        lambda self, candidate: ArenaScreenReceipt(
            self.identity,
            candidate.digest,
            candidate.screen_attempt,
            tuple(
                ScreenStageResult(stage, ScreenGrade.PASS, chr(97 + index) * 64, 1)
                for index, stage in enumerate(SCREEN_STAGES)
            ),
            PromotionDecision.PROMOTE,
        ),
    )

    def plan(_self, candidates, _receipts, state=None):
        reservations = tuple(row.reservation for row in candidates)
        authority = QualificationAuthorityManifest(
            "registered", "a" * 64, "b" * 64, "c" * 64, "d" * 64,
            tuple(row.selected_delta_digest for row in reservations), reservations,
        )
        factory = QualificationPlanFactory(
            authority, lambda _ref: b"s" * 32, lambda _secret: None
        )
        return ArenaQualificationWork(
            factory,
            object(),
            lambda *_: None,
            lambda **_: None,
            30.0,
            _self.manifest.qualification_policy_digest,
            resident_baseline_executor,
        )

    monkeypatch.setattr(ArenaService, "plan_qualification", plan)
    # The focused test uses a deliberately non-building plan and a mocked runner.
    monkeypatch.setattr(loop, "QualificationAuthorityManifest", type("NotManifest", (), {}))

    def qualify(factory, **_kwargs):
        assert (
            _kwargs["resident_baseline_executor"] is resident_baseline_executor
        )
        progress_events.append("qualification_complete")
        calls.append(factory.manifest.digest)
        authority = factory.manifest.reservations[0]
        outcome = QualificationIntakeOutcome(
            authority.reservation_digest,
            authority.selected_delta_digest,
            factory.manifest.digest,
            QualificationDecision.FAIL,
            "rejected",
            False,
            attempt_artifact_sha256="b" * 64,
            report_digest="c" * 64,
        )
        ref = EvidenceArtifactRef(
            "qualification.cohort-attempt", "b" * 64, 1,
            "application/json", "cacheon.qualification.cohort-attempt.v1",
        )
        return QualificationIntakeBatch(factory.manifest.digest, (outcome,), ref)

    monkeypatch.setattr(loop, "run_qualification_intake", qualify)
    original_apply = FinalizedIntakeStore.apply_qualification_batch

    def apply_with_progress(self, batch, **kwargs):
        progress_events.append("apply")
        retained_blocks.append(kwargs["current_finalized_block"])
        return original_apply(self, batch, **kwargs)

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "apply_qualification_batch",
        apply_with_progress,
    )
    original_promoted = FinalizedIntakeStore.promoted

    def promoted_with_limit(self, *, limit=None):
        promoted_limits.append(limit)
        return original_promoted(self, limit=limit)

    monkeypatch.setattr(FinalizedIntakeStore, "promoted", promoted_with_limit)

    def refreshed_head():
        progress_events.append("finalized_head")
        return BLOCK + 100, "0x" + "a" * 64

    result, _fetches, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        head_provider=refreshed_head,
        intake_only=False,
        arena_registry=registry,
        arena_id="test-arena",
    )
    assert len(calls) == 1 and len(calls[0]) == 64
    assert promoted_limits == [1]
    assert progress_events == ["qualification_complete", "finalized_head", "apply"]
    assert retained_blocks == [BLOCK + 100]
    assert set(result.decisions.values()) == {"FAIL"}
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed" and row.decision == "FAIL"
        assert store.qualification_dispositions(row.reservation_id)[0]["decision"] == "FAIL"


def test_once_mode_propagates_validator_fault(monkeypatch, tmp_path):
    monkeypatch.setattr(
        loop,
        "run_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("finality failed")),
    )
    with pytest.raises(RuntimeError, match="finality failed"):
        loop.run_validator(
            _NoWeightsSubtensor(),
            307,
            intake_db=tmp_path / "state.sqlite3",
            private_root=tmp_path / "private",
            publication_root=tmp_path / "worker",
            once=True,
        )


def test_intake_only_pass_still_settles_retained_pairs(tmp_path, monkeypatch):
    """The production deployment runs --intake-only; settlement must run there.

    The remote-worker pipeline retains PASS pairs into this store and has no
    other settlement authority: nesting the settle call under the local arena
    service left every retained winner pending until an operator crowned it by
    hand. The wiring, not the settlement internals, is what this pins.
    """

    calls = []

    def recorder(store, *, current_block, finalized_block_provider):
        calls.append(current_block)
        return {"lease-digest": "plan-digest"}

    monkeypatch.setattr(loop, "_settle_pending", recorder)
    snapshot = _snapshot([])
    result, _fetches, _options = _run(
        tmp_path, monkeypatch, snapshot, {}, intake_only=True
    )
    assert calls == [snapshot.finalized_block]
    assert result.settlements == {"lease-digest": "plan-digest"}


def test_settlement_refreshes_stale_pass_height_before_leasing():
    class Store:
        lease_blocks = []

        def has_pending_settlement(self):
            return True

        def lease_settlement_cohort(self, *, current_block):
            self.lease_blocks.append(current_block)
            return None

    store = Store()
    assert loop._settle_pending(
        store,
        current_block=BLOCK,
        finalized_block_provider=lambda: BLOCK + 100,
    ) == {}
    assert store.lease_blocks == [BLOCK + 100]


def test_settlement_head_refresh_failure_cannot_create_a_lease():
    class Store:
        lease_calls = 0

        def has_pending_settlement(self):
            return True

        def lease_settlement_cohort(self, *, current_block):
            self.lease_calls += 1
            return None

    def unavailable_head():
        raise RuntimeError("finalized head unavailable")

    store = Store()
    with pytest.raises(RuntimeError, match="finalized head unavailable"):
        loop._settle_pending(
            store,
            current_block=BLOCK,
            finalized_block_provider=unavailable_head,
        )
    assert store.lease_calls == 0


def test_closed_target_parks_by_name_only_and_fused_closed_slot_math_passes(
    tmp_path, monkeypatch
):
    """Closing a target closes its standalone lane only.

    The closed-family check keys on the SUBMITTED target name. A bundle for an
    open target whose kernel body computes -- and even names -- a closed slot's
    math is never parked for it: what an implementation absorbs inside its own
    boundary is not a rejection surface. Normative statement:
    docs/architecture/slot-contract.md.
    """
    closed = _bundle(
        tmp_path / "closed-src",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    fused = _bundle(
        tmp_path / "fused-src",
        '"""Fuses activation.silu_and_mul into the norm epilogue."""\n'
        "def rmsnorm(x, weight, out):\n"
        "    silu_and_mul = x * x.sigmoid() * weight\n"
        "    out.copy_(silu_and_mul)\n",
        slot="norm.rmsnorm",
        entry="rmsnorm",
    )
    closed_digest = content_hash(closed)
    fused_digest = content_hash(fused)
    snapshot = _snapshot([
        ("miner-closed", encode_payload(closed_digest, "https://example.com/a")),
        ("miner-fused", encode_payload(fused_digest, "https://example.com/b")),
    ])

    service = object.__new__(ArenaService)
    service.manifest = type(
        "Manifest",
        (),
        {
            "digest": "e" * 64,
            "qualification_policy_digest": "f" * 64,
            "capacity": type("Capacity", (), {"max_cohort_size": 1})(),
            "closed_targets": ("activation.silu_and_mul", "attention.sdpa"),
        },
    )()
    registry = object.__new__(ArenaServiceRegistry)
    monkeypatch.setattr(ArenaServiceRegistry, "require", lambda *_: service)
    monkeypatch.setattr(ArenaService, "admit", lambda *_: AdmissionDecision.QUEUE)

    result, _calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {closed_digest: closed, fused_digest: fused},
        intake_only=False,
        arena_registry=registry,
        arena_id="test-arena",
    )

    assert list(result.rejected.values()) == [
        "target_unavailable:activation.silu_and_mul"
    ]
    assert len(result.published) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        by_hotkey = {row.arrival.hotkey: row for row in store.all()}
        parked = by_hotkey["miner-closed"]
        assert parked.status == "expired"
        assert parked.decision == "NO_DECISION"
        passed = by_hotkey["miner-fused"]
        assert passed.status == "published"
        assert passed.reason == ""
