"""The next commission's incumbent authority is derived from the settled crown."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.bundle_hash import CARRIER_RECEIPT_NAME, content_hash
from cacheon.chain.incumbent_authority import (
    AUTHORITY_SCHEMA,
    RECEIPT_FILE,
    SOURCES_DIR,
    STACK_FILE,
    IncumbentAuthorityError,
    derive_incumbent_authority,
)
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.evidence_store import publish_evidence
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
)
from cacheon.settlement import SettlementQualification, plan_settlement
from cacheon.stack_identity import sha256_hex
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import plan_marginal_arm
from cacheon.target_catalog import default_target_catalog

SCOPE = IntakeScope("0x" + "0" * 64, 307)
TARGET = "activation.silu_and_mul"


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _bundle(root: Path) -> Path:
    source = root / "private-bundle"
    kernels = source / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "entry.py").write_text("def run(x, out):\n    return None\n")
    (source / "manifest.toml").write_text(
        "\n".join(
            (
                "bundle_id = 'crowned-fixture'",
                "abi_version = 'cacheon-op-abi-v0'",
                "[[ops]]",
                f"slot = '{TARGET}'",
                "source = 'kernels/entry.py'",
                "entry = 'run'",
                "dtypes = ['bfloat16']",
            )
        )
        + "\n"
    )
    for path in sorted(source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    return source


def _promote(store: FinalizedIntakeStore, reservation_id: str) -> None:
    active = store.begin_screen(reservation_id, service_digest=_h("service"))
    candidate = _h(f"candidate:{reservation_id}:{active.screen_attempts}")
    receipt = ArenaScreenReceipt(
        _h("service"),
        candidate,
        active.screen_attempts,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    store.apply_screen_receipt(reservation_id, candidate_digest=candidate, receipt=receipt)


def _crown(store: FinalizedIntakeStore, tmp_path: Path) -> tuple[str, str]:
    """Admit, publish, qualify twice, and settle one bundle; return its identities."""

    catalog = default_target_catalog()
    source = _bundle(tmp_path)
    committed = content_hash(source)
    publication = publish_worker_bundle(source, tmp_path / "publications", committed)
    incumbent = EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    replacement = ProposalContributionRef(
        target_id=TARGET,
        target_spec_digest=catalog.target_spec_digest(TARGET),
        artifact_digest=committed,
        selected_payload_digest=_h("payload"),
        attribution_digest=_h("attribution"),
    )
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    arm = plan_marginal_arm(
        incumbent,
        replacement,
        catalog=catalog,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h("candidate-tree"),
        expected_context=EvaluationStackContext(
            runtime_digest=_h("runtime"),
            base_engine_digest=_h("base"),
            arena_digest=_h("arena"),
            catalog_snapshot=catalog.snapshot(),
            catalog_digest=catalog.digest,
            target_spec_digests={
                row["target_id"]: catalog.target_spec_digest(row["target_id"])
                for row in targets
            },
        ),
    )
    store.initialize_evaluation_stack(incumbent, tree_digest=arm.baseline_before.tree_digest)
    arrival = FinalizedArrival(
        hotkey="miner",
        content_hash=committed,
        url="https://example.com/crowned.tar.gz",
        block=10,
        block_hash="0x" + f"{10:064x}",
        event_index=0,
    )
    (row,) = store.reserve_finalized(
        (arrival,), finalized_block=10, finalized_block_hash="0x" + f"{10:064x}"
    )
    store.mark_fetching(row.reservation_id)
    store.mark_published(
        row.reservation_id,
        delta_fingerprint=SubmittedDeltaFingerprint(
            "component", TARGET, "1" * 64, (TARGET,), "2" * 64,
            arm.selected_delta_digest, "4" * 64, ("a" * 64,), ("5" * 64,),
        ),
        publication_digest=publication.digest,
        publication_root=publication.root,
    )
    _promote(store, row.reservation_id)
    evidence_root = tmp_path / "evidence"
    attempts = tuple(
        publish_evidence(
            evidence_root,
            f"retained {marker} qualification attempt".encode(),
            domain="qualification.cohort-attempt",
            media_type="application/json",
            schema="cacheon.qualification.cohort-attempt.v1",
        )
        for marker in ("primary", "reproduction")
    )
    for index, (marker, attempt, speedup) in enumerate(
        zip(("primary", "reproduction"), attempts, ("1.05", "1.04"), strict=True)
    ):
        authority = _h(f"{marker}-authority")
        policy = SlotAuditPolicy(_h(f"audit-seed:{marker}")[:32], 100_000, 32, (TARGET,), 1)
        settled = SettlementQualification(
            lane="registered",
            arena_digest=incumbent.arena_digest,
            reservation_digest=row.reservation_id,
            finalized_block=row.arrival.block,
            event_index=row.arrival.event_index,
            event_subindex=row.arrival.event_subindex,
            hotkey=row.arrival.hotkey,
            target_id=TARGET,
            members=(TARGET,),
            selected_delta_digest=arm.selected_delta_digest,
            qualification_authority_digest=authority,
            qualification_plan_digest=_h("plan-" + marker),
            qualification_attempt_digest=attempt.sha256,
            qualification_report_digest=_h("report-" + marker),
            selection_commitment_digest=_h("commitment-" + marker),
            selection_secret_commitment_digest=_h("secret-" + marker),
            selection_evidence_digest=_h("selection-" + marker),
            arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest,
            candidate_tree_digest=arm.challenger.tree_digest,
            speedup=speedup,
            incumbent_manifest=incumbent,
            candidate_manifest=arm.candidate,
            audit_control_digest=policy.control.digest,
            audit_policy=policy,
            audit_evidence_digest=_h("audit-evidence-" + marker),
        )
        if index:
            _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, authority, {"schema": "test-authority"})
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            arm.selected_delta_digest,
            authority,
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=attempt.sha256,
            report_digest=settled.qualification_report_digest,
            settlement_qualification=settled,
        )
        store.apply_qualification_batch(
            QualificationIntakeBatch(authority, (outcome,), attempt),
            current_finalized_block=10,
            evidence_root=evidence_root,
        )
    lease = store.lease_settlement_cohort(current_block=11)
    assert lease is not None
    plan = plan_settlement(
        lease.candidates,
        current_manifest=lease.stack.manifest,
        current_tree_digest=lease.stack.tree_digest,
        initial_event_sequence=lease.initial_event_sequence,
        previous_event_digest=lease.previous_event_digest,
    )
    evidence = tuple(store.reopen_settlement_evidence(item) for item in lease.candidates)
    state = store.commit_settlement(lease, plan, evidence, current_block=11)
    assert state.generation == 1
    return committed, row.reservation_id


def _store(tmp_path: Path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3", IntakePolicy(), scope=SCOPE
    )


def test_authority_is_written_from_the_settled_crown_and_its_retained_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "incumbent-authority"
    with _store(tmp_path) as store:
        committed, reservation_id = _crown(store, tmp_path)
        crowned = store.evaluation_stack(_h("arena"))
        authority = derive_incumbent_authority(store, output)

    assert authority.root == output
    assert (authority.generation, authority.stack_digest, authority.tree_digest) == (
        1, crowned.manifest.digest, crowned.tree_digest
    )
    assert authority.transition_event_id == crowned.transition_event_id
    assert [source.to_dict() for source in authority.sources] == [
        {
            "artifact_digest": committed,
            "publication_digest": authority.sources[0].publication_digest,
            "reservation_id": reservation_id,
            "target_id": TARGET,
        }
    ]
    stack_bytes = (output / STACK_FILE).read_bytes()
    assert EvaluationStackManifest.from_dict(json.loads(stack_bytes)).digest == (
        crowned.manifest.digest
    )
    receipt = json.loads((output / RECEIPT_FILE).read_bytes())
    assert receipt == authority.to_dict()
    assert receipt["schema"] == AUTHORITY_SCHEMA
    assert receipt["stack_sha256"] == sha256_hex(stack_bytes)
    copied = output / SOURCES_DIR / committed
    # The engine hashes every file of a staged incumbent source, so the copy
    # holds the miner's committed bytes and not the carrier receipt.
    assert content_hash(copied) == committed
    assert not (copied / CARRIER_RECEIPT_NAME).exists()
    assert (copied / "manifest.toml").is_file()
    for path in (output, output / SOURCES_DIR, copied):
        assert stat.S_IMODE(path.stat().st_mode) == 0o555
    for path in (output / STACK_FILE, output / RECEIPT_FILE):
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert not (tmp_path / "incumbent-authority.incoming").exists()


def test_authority_refuses_without_a_settled_crown_and_never_writes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "incumbent-authority"
    with _store(tmp_path) as store:
        with pytest.raises(IncumbentAuthorityError, match="no settled crown"):
            derive_incumbent_authority(store, output)
    assert sorted(os.listdir(tmp_path)) == ["private"]


def test_authority_is_append_only_and_refuses_a_tampered_publication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "incumbent-authority"
    with _store(tmp_path) as store:
        committed, reservation_id = _crown(store, tmp_path)
        derive_incumbent_authority(store, output)
        with pytest.raises(IncumbentAuthorityError, match="append-only"):
            derive_incumbent_authority(store, output)
        row = store.get(reservation_id)
        tampered = Path(row.publication_root) / "manifest.toml"
        tampered.chmod(0o600)
        tampered.write_text("bundle_id = 'tampered'\n")
        with pytest.raises(IncumbentAuthorityError, match="cannot reopen"):
            derive_incumbent_authority(store, tmp_path / "second")
    assert not (tmp_path / "second").exists()
    assert not (tmp_path / "second.incoming").exists()
