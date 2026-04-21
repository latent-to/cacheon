"""Unit tests for validator.state — no chain, no GPU, no bittensor."""

from __future__ import annotations

import json

import pytest

from validator.state import (
    EvaluationRecord,
    KingRecord,
    SCHEMA_VERSION,
    STATE_FILE_NAME,
    ValidatorState,
    _atomic_write_json,
    current_timestamp,
    unknown_commits,
)

pytestmark = pytest.mark.unit


def _make_eval(
    uid: int = 1,
    hotkey: str = "hk_alice",
    commit_block: int = 100,
    score: float = 0.25,
    kl: float = 0.02,
    mem: float = 0.3,
    lat: float = 0.1,
    disqualified: bool = False,
    reason: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=uid,
        hotkey=hotkey,
        commit_block=commit_block,
        model="hf-user/policy",
        revision="abc123",
        score=score,
        kl_divergence=kl,
        memory_reduction=mem,
        latency_improvement=lat,
        disqualified=disqualified,
        disqualify_reason=reason,
        evaluated_at=1700000000.0,
        evaluation_block=commit_block + 10,
    )


class TestEvaluationRecord:
    def test_eval_key(self):
        ev = _make_eval(hotkey="hk_foo", commit_block=42)
        assert ev.eval_key == "hk_foo:42"

    def test_round_trip(self):
        ev = _make_eval()
        restored = EvaluationRecord.from_dict(ev.to_dict())
        assert restored == ev

    def test_from_dict_ignores_unknown_keys(self):
        ev = _make_eval()
        data = ev.to_dict()
        data["extra_future_field"] = "ignored"
        restored = EvaluationRecord.from_dict(data)
        assert restored == ev


class TestKingRecord:
    def test_from_evaluation(self):
        ev = _make_eval(score=0.5)
        king = KingRecord.from_evaluation(ev)
        assert king.uid == ev.uid
        assert king.score == ev.score
        assert king.hotkey == ev.hotkey

    def test_round_trip(self):
        ev = _make_eval()
        king = KingRecord.from_evaluation(ev)
        restored = KingRecord.from_dict(king.to_dict())
        assert restored == king


class TestValidatorStateRecording:
    def test_empty_state(self):
        state = ValidatorState()
        assert state.king is None
        assert state.evaluations == {}
        assert state.schema_version == SCHEMA_VERSION

    def test_record_first_eval_becomes_king(self):
        state = ValidatorState()
        ev = _make_eval(score=0.3)
        dethroned = state.record_evaluation(ev)
        assert dethroned is True
        assert state.king is not None
        assert state.king.uid == ev.uid
        assert state.king.score == ev.score
        assert state.has_evaluation(ev.hotkey, ev.commit_block)

    def test_higher_score_dethrones_king(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(uid=1, hotkey="hk1", score=0.2))
        dethroned = state.record_evaluation(
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.5)
        )
        assert dethroned is True
        assert state.king.uid == 2
        assert state.king.score == 0.5

    def test_lower_score_does_not_dethrone(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(uid=1, hotkey="hk1", score=0.5))
        dethroned = state.record_evaluation(
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.4)
        )
        assert dethroned is False
        assert state.king.uid == 1

    def test_equal_score_keeps_defender(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(uid=1, hotkey="hk1", score=0.3))
        dethroned = state.record_evaluation(
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.3)
        )
        assert dethroned is False
        assert state.king.uid == 1

    def test_disqualified_cannot_become_king(self):
        state = ValidatorState()
        dethroned = state.record_evaluation(
            _make_eval(score=0.0, disqualified=True, reason="KL too high")
        )
        assert dethroned is False
        assert state.king is None

    def test_disqualified_cannot_dethrone_king(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(uid=1, hotkey="hk1", score=0.2))
        # Even a high-score DQ'd entry should not take the throne
        dethroned = state.record_evaluation(
            _make_eval(
                uid=2, hotkey="hk2", commit_block=200,
                score=0.9, disqualified=True, reason="nan",
            )
        )
        assert dethroned is False
        assert state.king.uid == 1

    def test_zero_score_non_dq_cannot_become_king(self):
        state = ValidatorState()
        dethroned = state.record_evaluation(_make_eval(score=0.0))
        assert dethroned is False
        assert state.king is None

    def test_negative_score_cannot_become_king(self):
        state = ValidatorState()
        dethroned = state.record_evaluation(_make_eval(score=-0.1))
        assert dethroned is False
        assert state.king is None

    def test_nan_score_cannot_become_king(self):
        """NaN slips past `<= 0.0` (IEEE 754 comparisons with NaN are False).
        If it became king, nothing could dethrone it since `x > NaN` is also
        False — must be rejected explicitly."""
        state = ValidatorState()
        dethroned = state.record_evaluation(_make_eval(score=float("nan")))
        assert dethroned is False
        assert state.king is None

    def test_nan_score_cannot_dethrone_existing_king(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(uid=1, hotkey="hk1", score=0.3))
        dethroned = state.record_evaluation(
            _make_eval(uid=2, hotkey="hk2", score=float("nan"))
        )
        assert dethroned is False
        assert state.king is not None
        assert state.king.uid == 1

    def test_inf_score_cannot_become_king(self):
        state = ValidatorState()
        dethroned = state.record_evaluation(_make_eval(score=float("inf")))
        assert dethroned is False
        assert state.king is None

    def test_record_precheck_failure(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 100, "blocked import: os")
        assert state.has_precheck_failure("hk1", 100)
        assert state.is_known("hk1", 100)
        assert not state.has_evaluation("hk1", 100)

    def test_is_known_flags_both_paths(self):
        state = ValidatorState()
        state.record_precheck_failure("hk_pre", 1, "bad")
        state.record_evaluation(_make_eval(hotkey="hk_eval", commit_block=2))
        assert state.is_known("hk_pre", 1)
        assert state.is_known("hk_eval", 2)
        assert not state.is_known("hk_other", 3)

    def test_recording_eval_clears_stale_precheck_entry(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 100, "stale")
        state.record_evaluation(_make_eval(hotkey="hk1", commit_block=100))
        assert not state.has_precheck_failure("hk1", 100)
        assert state.has_evaluation("hk1", 100)


class TestScoreHistory:
    def test_history_ordered_by_commit_block(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(hotkey="hk_a", commit_block=30))
        state.record_evaluation(_make_eval(hotkey="hk_a", commit_block=10))
        state.record_evaluation(_make_eval(hotkey="hk_b", commit_block=20))
        history = state.score_history_for_hotkey("hk_a")
        assert [e.commit_block for e in history] == [10, 30]

    def test_history_empty_for_unknown_hotkey(self):
        state = ValidatorState()
        assert state.score_history_for_hotkey("hk_nobody") == []


class TestPersistence:
    def test_save_then_load_round_trip(self, tmp_path):
        state = ValidatorState()
        state.record_evaluation(_make_eval(uid=1, hotkey="hk1", score=0.4))
        state.record_evaluation(_make_eval(uid=2, hotkey="hk2",
                                           commit_block=200, score=0.6))
        state.record_precheck_failure("hk3", 300, "blocked call: eval")
        state.last_scan_block = 1234
        state.last_weights_set_block = 1234

        state.save(tmp_path)
        reloaded = ValidatorState.load(tmp_path)

        assert reloaded.king is not None
        assert reloaded.king.uid == 2
        assert reloaded.king.score == 0.6
        assert reloaded.has_evaluation("hk1", 100)
        assert reloaded.has_evaluation("hk2", 200)
        assert reloaded.has_precheck_failure("hk3", 300)
        assert reloaded.last_scan_block == 1234
        assert reloaded.last_weights_set_block == 1234
        assert reloaded.schema_version == SCHEMA_VERSION

    def test_load_missing_file_returns_empty_state(self, tmp_path):
        loaded = ValidatorState.load(tmp_path)
        assert loaded.king is None
        assert loaded.evaluations == {}

    def test_load_corrupt_file_returns_empty_state(self, tmp_path):
        (tmp_path / STATE_FILE_NAME).write_text("{not valid json")
        loaded = ValidatorState.load(tmp_path)
        assert loaded.king is None

    def test_newer_schema_rejected(self, tmp_path):
        payload = {
            "schema_version": SCHEMA_VERSION + 99,
            "king": None,
            "evaluations": {},
            "precheck_failures": {},
            "last_scan_block": 0,
            "last_weights_set_block": 0,
        }
        (tmp_path / STATE_FILE_NAME).write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="schema_version"):
            ValidatorState.load(tmp_path)

    def test_atomic_write_leaves_no_tmp_on_success(self, tmp_path):
        target = tmp_path / "x.json"
        _atomic_write_json(target, {"a": 1})
        assert target.exists()
        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []


class TestUnknownCommits:
    def test_filters_known(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval(hotkey="hk1", commit_block=100))
        state.record_precheck_failure("hk2", 200, "bad")
        incoming = [
            ("hk1", 100),      # already evaluated
            ("hk2", 200),      # already pre-rejected
            ("hk3", 300),      # unknown
            ("hk1", 150),      # same hotkey, different block → unknown
        ]
        result = unknown_commits(state, incoming)
        assert result == [("hk3", 300), ("hk1", 150)]

    def test_empty_input(self):
        state = ValidatorState()
        assert unknown_commits(state, []) == []


class TestCloneAndTimestamp:
    def test_clone_is_deep(self):
        state = ValidatorState()
        state.record_evaluation(_make_eval())
        clone = state.clone()
        clone.record_evaluation(
            _make_eval(uid=99, hotkey="hk_new", commit_block=999, score=0.9)
        )
        assert state.king.uid == 1
        assert clone.king.uid == 99

    def test_current_timestamp_is_monotonic_enough(self):
        assert current_timestamp() > 0
