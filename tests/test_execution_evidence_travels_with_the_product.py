"""What every GPU did with the kernel must reach the miner, off the pod.

On the resident lane the per-rank execution record was printed into a
container log at the NEXT swap and never joined to anything the validator
keeps, so a miner could be told ``FAIL`` by a run whose rows nobody could
find. The rows now ride the swap frame into the host's swap receipt, are
published as one unsealed artifact beside the attempt, travel in the product
like every other artifact, and are rendered by ``chain-miner-report`` from the
evidence store the validator imported them into.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from cacheon.chain.intake import FinalizedArrival, FinalizedIntakeStore, IntakeScope
from cacheon.chain.miner_feedback import format_miner_submissions, miner_submissions
from cacheon.eval.b300_resident_qualification import (
    EXECUTION_EVIDENCE_DOMAIN,
    EXECUTION_EVIDENCE_SCHEMA,
    execution_evidence_refs,
)
from cacheon.eval.evidence_store import (
    prepare_evidence_root,
    publish_canonical_json_evidence,
    reopen_evidence,
)
from cacheon.eval.resident_execution_evidence import (
    EXECUTION_CODEC,
    UNOBSERVED_EVIDENCE,
    RankExecution,
    ResidentExecutionEvidence,
    SlotExecution,
)

SCOPE = IntakeScope("0x" + "0" * 64, 307)
BUNDLE = "7" * 64
SLOT = "attention.msa_block_score"


def _lane_a_ranks() -> tuple[RankExecution, ...]:
    """The four real lane-A rows (2026-08-23), typed."""

    return tuple(
        RankExecution(rank, True, "", (SlotExecution(SLOT, 1140, True),))
        for rank in range(4)
    )


def _swap(generation: int, bundle: str | None, execution: ResidentExecutionEvidence):
    return SimpleNamespace(
        generation=generation, bundle_digest=bundle, execution=execution, expected_ranks=4
    )


def _speed(*ranks_per_candidate_generation: tuple[RankExecution, ...]):
    """A crossover's slices: activation (closes stock), restoration (closes the candidate)."""

    slices = []
    generation = 1
    for index, ranks in enumerate(ranks_per_candidate_generation):
        executed = sum(row.clean(eager_slots=frozenset()) for row in ranks)
        slices.append(
            SimpleNamespace(
                lane_id="A",
                request_id=f"{index + 1:032x}",
                new_swaps=(
                    _swap(generation + 1, BUNDLE, UNOBSERVED_EVIDENCE),
                    _swap(
                        generation + 2, None,
                        ResidentExecutionEvidence(generation + 1, executed, ranks),
                    ),
                ),
            )
        )
        generation += 2
    return SimpleNamespace(candidate_bundle_digest=BUNDLE, request_slices=tuple(slices))


def test_the_rows_are_published_once_per_candidate_generation(tmp_path):
    root = prepare_evidence_root(tmp_path / "evidence")
    (reference,) = execution_evidence_refs(
        _speed(_lane_a_ranks()),
        evidence_root=root,
        request_digest="1" * 64,
        authority_digest="2" * 64,
        source_digest="3" * 64,
    )
    assert (reference.domain, reference.schema) == (
        EXECUTION_EVIDENCE_DOMAIN, EXECUTION_EVIDENCE_SCHEMA
    )
    payload = json.loads(reopen_evidence(root, reference))
    assert payload["bundle_digest"] == BUNDLE
    (swap,) = payload["swaps"]  # the stock generation has nothing to say
    assert (swap["generation"], swap["executed_ranks"], swap["expected_ranks"]) == (2, 4, 4)
    assert [row["value"]["rank"] for row in swap["ranks"]] == [0, 1, 2, 3]
    assert swap["ranks"][0]["value"]["slots"] == [
        {"calls": 1140, "captured": True, "error": "", "skipped": [], "slot": SLOT}
    ]


def test_a_run_with_no_rows_publishes_nothing_and_never_raises(tmp_path):
    assert execution_evidence_refs(
        _speed(),
        evidence_root=tmp_path / "missing",
        request_digest="1" * 64,
        authority_digest="2" * 64,
        source_digest="3" * 64,
    ) == ()
    # An unusable store is logged, not raised: the run's verdict does not
    # depend on a report artifact.
    (tmp_path / "blocked").write_text("not a directory")
    assert execution_evidence_refs(
        _speed(_lane_a_ranks()),
        evidence_root=tmp_path / "blocked",
        request_digest="1" * 64,
        authority_digest="2" * 64,
        source_digest="3" * 64,
    ) == ()


def _qualified_store(tmp_path, evidence_root):
    """A reservation whose attempt artifact and execution rows sit in one store."""

    attempt = publish_canonical_json_evidence(
        evidence_root,
        {"speed_witness": {"rates": []}},
        domain="qualification.stage-exit",
        schema="cacheon.test.stage-exit",
    )
    store = FinalizedIntakeStore(tmp_path / "state" / "intake.sqlite3", scope=SCOPE)
    arrival = FinalizedArrival(
        "miner-0", f"{1:064x}", "https://host.example/b.tar.gz", 40, "0x" + "4" * 64, 0
    )
    store.reserve_finalized(
        (arrival,), finalized_block=40, finalized_block_hash="0x" + "4" * 64
    )
    store._db.execute(
        "UPDATE reservations SET status='failed',decision='FAIL',reason='candidate_slower',"
        "publication_digest=? WHERE reservation_id=?",
        (BUNDLE, arrival.reservation_id),
    )
    store._db.execute(
        "INSERT INTO qualification_dispositions("
        "reservation_id,attempt_index,authority_digest,authority_manifest_json,"
        "evidence_digest,attempt_ref_json,report_digest,failure_digest,decision,reason"
        ") VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            arrival.reservation_id, 0, "2" * 64, "{}", attempt.sha256,
            json.dumps(attempt.to_dict(), sort_keys=True), "", "", "FAIL",
            "candidate_slower",
        ),
    )
    store._db.commit()
    path = store.path
    store.close()
    return path


def test_the_miner_report_renders_the_rows_from_the_imported_store(tmp_path):
    root = prepare_evidence_root(tmp_path / "evidence")
    uncaptured = _lane_a_ranks()[:3] + (
        RankExecution(3, True, "", (SlotExecution(SLOT, 1140, False),)),
    )
    execution_evidence_refs(
        _speed(uncaptured),
        evidence_root=root,
        request_digest="1" * 64,
        authority_digest="2" * 64,
        source_digest="3" * 64,
    )
    # A second bundle's rows in the same store must not be attributed to this one.
    publish_canonical_json_evidence(
        root,
        {"bundle_digest": "8" * 64, "schema": EXECUTION_EVIDENCE_SCHEMA, "swaps": [
            {"generation": 9, "lane_id": "B", "executed_ranks": 4, "expected_ranks": 4,
             "ranks": [EXECUTION_CODEC.encode(
                 RankExecution(0, True, "", (SlotExecution("other.slot", 1, True),))
             )]}
        ]},
        domain=EXECUTION_EVIDENCE_DOMAIN,
        schema=EXECUTION_EVIDENCE_SCHEMA,
    )
    db = _qualified_store(tmp_path, root)

    report = miner_submissions(db, hotkey="miner-0", evidence_roots=(root,))
    (attempt,) = report["submissions"][0]["attempt_evidence"]
    assert attempt["retained"] is True
    text = "\n".join(attempt["explanation"])
    assert "3 of 4 GPU(s) ran your kernel cleanly" in text
    assert f"{SLOT}: called 4,560 times across 4 GPU(s), NOT inside the CUDA graph" in text
    assert "other.slot" not in text
    rendered = format_miner_submissions(report)
    assert "attempt[0] evidence:" in rendered
    assert "did your kernel run" in rendered

    # Without a store the durable record stands alone, and says so.
    bare = miner_submissions(db, hotkey="miner-0")
    assert "attempt_evidence" not in bare["submissions"][0]
    absent = miner_submissions(db, hotkey="miner-0", evidence_roots=(tmp_path / "none",))
    assert absent["submissions"][0]["attempt_evidence"][0]["retained"] is False
