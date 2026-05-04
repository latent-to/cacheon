"""Unit tests for validator.challengers — pure, no chain or GPU."""

from __future__ import annotations

import pytest

from validator.challengers import (
    PrecheckOutcome,
    PrecheckResult,
    allow_all_precheck,
    select_challengers,
)
from validator.chain import CommitmentRecord
from validator.state import ValidatorState

from tests.test_validator_state import _make_eval, _record

pytestmark = pytest.mark.unit


def _commit(
    uid: int,
    hotkey: str,
    block: int,
    image: str = "user/m:latest",
    digest: str = "sha256:" + "a" * 64,
) -> CommitmentRecord:
    return CommitmentRecord(
        uid=uid,
        hotkey=hotkey,
        commit_block=block,
        image=image,
        digest=digest,
        raw="{}",
    )


class TestSelectChallengers:
    def test_all_new_commitments_become_challengers(self):
        state = ValidatorState()
        commits = [
            _commit(0, "hk0", 10),
            _commit(1, "hk1", 20),
            _commit(2, "hk2", 30),
        ]
        result = select_challengers(state, commits)
        assert len(result) == 3
        assert result.newly_rejected == []
        assert result.already_known == []

    def test_already_evaluated_filtered(self):
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk1", commit_block=20))
        commits = [_commit(0, "hk0", 10), _commit(1, "hk1", 20)]
        result = select_challengers(state, commits)
        assert [c.uid for c in result.challengers] == [0]
        assert len(result.already_known) == 1
        assert result.already_known[0].uid == 1

    def test_already_precheck_failed_filtered(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 20, "blocked import: os")
        commits = [_commit(0, "hk0", 10), _commit(1, "hk1", 20)]
        result = select_challengers(state, commits)
        assert [c.uid for c in result.challengers] == [0]
        assert len(result.already_known) == 1
        assert result.already_known[0].uid == 1

    def test_new_commit_block_for_known_hotkey_is_challenger(self):
        """Same hotkey at a new block should re-challenge — miners can
        technically re-commit on-chain even though subnet rule is one-shot."""
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk1", commit_block=20))
        commits = [_commit(1, "hk1", 50)]  # same hotkey, new block
        result = select_challengers(state, commits)
        assert len(result.challengers) == 1
        assert result.challengers[0].commit_block == 50

    def test_precheck_rejects_commitment(self):
        state = ValidatorState()
        commits = [_commit(0, "hk0", 10), _commit(1, "hk1", 20)]

        def reject_odd_uids(com: CommitmentRecord) -> PrecheckResult:
            if com.uid % 2 == 1:
                return PrecheckResult(
                    outcome=PrecheckOutcome.REJECTED,
                    reason="blocked import: os",
                )
            return PrecheckResult(outcome=PrecheckOutcome.OK)

        result = select_challengers(state, commits, precheck=reject_odd_uids)
        assert [c.uid for c in result.challengers] == [0]
        assert len(result.newly_rejected) == 1
        rejected_com, reason = result.newly_rejected[0]
        assert rejected_com.uid == 1
        assert "blocked import" in reason

    def test_precheck_missing_reason_defaults_to_message(self):
        state = ValidatorState()
        commits = [_commit(0, "hk0", 10)]

        def reject_no_reason(_com):
            return PrecheckResult(outcome=PrecheckOutcome.REJECTED, reason=None)

        result = select_challengers(state, commits, precheck=reject_no_reason)
        assert len(result.challengers) == 0
        assert len(result.newly_rejected) == 1
        _, reason = result.newly_rejected[0]
        assert reason  # non-empty default

    def test_allow_all_precheck(self):
        rec = _commit(0, "hk0", 10)
        result = allow_all_precheck(rec)
        assert result.ok is True
        assert result.outcome is PrecheckOutcome.OK

    def test_precheck_defers_commitment(self):
        state = ValidatorState()
        commits = [_commit(0, "hk0", 10), _commit(1, "hk1", 20)]

        def defer_uid_1(com: CommitmentRecord) -> PrecheckResult:
            if com.uid == 1:
                return PrecheckResult(
                    outcome=PrecheckOutcome.DEFERRED,
                    reason="network timeout",
                )
            return PrecheckResult(outcome=PrecheckOutcome.OK)

        result = select_challengers(state, commits, precheck=defer_uid_1)
        assert [c.uid for c in result.challengers] == [0]
        assert len(result.deferred) == 1
        deferred_com, reason = result.deferred[0]
        assert deferred_com.uid == 1
        assert "network timeout" in reason
        assert result.newly_rejected == []

    def test_deferred_not_recorded_in_state(self):
        state = ValidatorState()
        commits = [_commit(0, "hk0", 10)]

        def defer_all(com: CommitmentRecord) -> PrecheckResult:
            return PrecheckResult(outcome=PrecheckOutcome.DEFERRED, reason="retry")

        result = select_challengers(state, commits, precheck=defer_all)
        assert len(result.deferred) == 1
        assert len(result.challengers) == 0
        assert len(result.newly_rejected) == 0
        # Deferred must not touch state
        assert not state.has_precheck_failure("hk0", 10)
        assert not state.is_known("hk0", 10)

    def test_select_does_not_mutate_state(self):
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk1", commit_block=20))
        commits = [_commit(1, "hk1", 20), _commit(2, "hk2", 30)]

        snapshot = state.to_dict()
        _ = select_challengers(
            state,
            commits,
            precheck=lambda c: PrecheckResult(
                outcome=PrecheckOutcome.REJECTED, reason="bad"
            ),
        )
        assert state.to_dict() == snapshot  # selection is side-effect-free

    def test_empty_commitments(self):
        state = ValidatorState()
        result = select_challengers(state, [])
        assert len(result) == 0
        assert result.newly_rejected == []
        assert result.already_known == []

    def test_len_dunder(self):
        state = ValidatorState()
        commits = [_commit(0, "hk0", 10), _commit(1, "hk1", 20)]
        result = select_challengers(state, commits)
        assert len(result) == 2
