"""Unit tests for validator.state — no chain, no GPU, no bittensor."""

from __future__ import annotations

import json

import pytest

from validator.config import KING_EPSILON_DECAY_BLOCKS, KING_EPSILON_INITIAL
from validator.state import (
    DUPLICATE_OF_KING_REASON,
    EvaluationRecord,
    KingRecord,
    RecordResult,
    SCHEMA_VERSION,
    STATE_FILE_NAME,
    ValidatorState,
    _atomic_write_json,
    _effective_dethrone_threshold,
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
    source_hash: str = "",
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=uid,
        hotkey=hotkey,
        commit_block=commit_block,
        repo="hf-user/policy",
        revision="a" * 40,
        score=score,
        kl_divergence=kl,
        memory_reduction=mem,
        latency_improvement=lat,
        disqualified=disqualified,
        disqualify_reason=reason,
        evaluated_at=1700000000.0,
        evaluation_block=commit_block + 10,
        source_hash=source_hash,
    )


def _record(state: ValidatorState, ev: EvaluationRecord,
            *, current_block: int | None = None) -> RecordResult:
    """Shorthand — tests that don't care about the block pass
    ``commit_block + 10`` to match the existing `_make_eval` default."""
    if current_block is None:
        current_block = ev.commit_block + 10
    return state.record_evaluation(ev, current_block=current_block)


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
        king = KingRecord.from_evaluation(ev, crowned_at_block=500)
        assert king.uid == ev.uid
        assert king.score == ev.score
        assert king.hotkey == ev.hotkey
        assert king.crowned_at_block == 500

    def test_round_trip(self):
        ev = _make_eval(source_hash="a" * 64)
        king = KingRecord.from_evaluation(ev, crowned_at_block=123)
        restored = KingRecord.from_dict(king.to_dict())
        assert restored == king
        assert restored.source_hash == ev.source_hash


class TestValidatorStateRecording:
    def test_empty_state(self):
        state = ValidatorState()
        assert state.king is None
        assert state.evaluations == {}
        assert state.schema_version == SCHEMA_VERSION

    def test_record_first_eval_becomes_king(self):
        state = ValidatorState()
        ev = _make_eval(score=0.3)
        out = _record(state, ev)
        assert out.dethroned is True
        assert out.dethrone_threshold == 0.0  # no king to beat
        assert state.king is not None
        assert state.king.uid == ev.uid
        assert state.king.score == ev.score
        assert state.has_evaluation(ev.hotkey, ev.commit_block)

    def test_higher_score_dethrones_king(self):
        state = ValidatorState()
        # Big score so decayed-epsilon moat is easy to clear
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.2))
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.5),
        )
        assert out.dethroned is True
        assert state.king.uid == 2
        assert state.king.score == 0.5

    def test_lower_score_does_not_dethrone(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.5))
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.4),
        )
        assert out.dethroned is False
        assert state.king.uid == 1

    def test_equal_score_keeps_defender(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.3))
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.3),
        )
        assert out.dethroned is False
        assert state.king.uid == 1

    def test_epsilon_moat_blocks_tiny_improvement(self):
        """A challenger whose score is > king but within the epsilon moat
        stays sub-throne. Uses a deliberately tiny improvement (0.5%) well
        below the default 1% initial epsilon."""
        state = ValidatorState()
        _record(
            state,
            _make_eval(uid=1, hotkey="hk1", score=0.5, commit_block=100),
            current_block=100,
        )
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", score=0.5025, commit_block=110),
            current_block=110,  # 10 blocks after crown: moat still ~1%
        )
        assert out.dethroned is False
        assert out.dethrone_threshold > 0.5
        assert state.king.uid == 1

    def test_epsilon_fully_decayed_allows_strict_improvement(self):
        """Once more than `KING_EPSILON_DECAY_BLOCKS` blocks have passed,
        any strict improvement over `king.score` dethrones."""
        state = ValidatorState()
        _record(
            state,
            _make_eval(uid=1, hotkey="hk1", score=0.5, commit_block=100),
            current_block=100,
        )
        far_future = 100 + KING_EPSILON_DECAY_BLOCKS + 1
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", score=0.5001, commit_block=far_future),
            current_block=far_future,
        )
        assert out.dethroned is True
        assert out.dethrone_threshold == pytest.approx(0.5)
        assert state.king.uid == 2

    def test_disqualified_cannot_become_king(self):
        state = ValidatorState()
        out = _record(
            state,
            _make_eval(score=0.0, disqualified=True, reason="KL too high"),
        )
        assert out.dethroned is False
        assert state.king is None

    def test_disqualified_cannot_dethrone_king(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.2))
        # Even a high-score DQ'd entry should not take the throne
        out = _record(
            state,
            _make_eval(
                uid=2, hotkey="hk2", commit_block=200,
                score=0.9, disqualified=True, reason="nan",
            ),
        )
        assert out.dethroned is False
        assert state.king.uid == 1

    def test_zero_score_non_dq_cannot_become_king(self):
        state = ValidatorState()
        out = _record(state, _make_eval(score=0.0))
        assert out.dethroned is False
        assert state.king is None

    def test_negative_score_cannot_become_king(self):
        state = ValidatorState()
        out = _record(state, _make_eval(score=-0.1))
        assert out.dethroned is False
        assert state.king is None

    def test_nan_score_cannot_become_king(self):
        """NaN slips past `<= 0.0` (IEEE 754 comparisons with NaN are False).
        If it became king, nothing could dethrone it since `x > NaN` is also
        False — must be rejected explicitly."""
        state = ValidatorState()
        out = _record(state, _make_eval(score=float("nan")))
        assert out.dethroned is False
        assert state.king is None

    def test_nan_score_cannot_dethrone_existing_king(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.3))
        out = _record(
            state, _make_eval(uid=2, hotkey="hk2", score=float("nan")),
        )
        assert out.dethroned is False
        assert state.king is not None
        assert state.king.uid == 1

    def test_inf_score_cannot_become_king(self):
        state = ValidatorState()
        out = _record(state, _make_eval(score=float("inf")))
        assert out.dethroned is False
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
        _record(state, _make_eval(hotkey="hk_eval", commit_block=2))
        assert state.is_known("hk_pre", 1)
        assert state.is_known("hk_eval", 2)
        assert not state.is_known("hk_other", 3)

    def test_recording_eval_clears_stale_precheck_entry(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 100, "stale")
        _record(state, _make_eval(hotkey="hk1", commit_block=100))
        assert not state.has_precheck_failure("hk1", 100)
        assert state.has_evaluation("hk1", 100)


class TestDuplicateOfKingDQ:
    """Byte-identical submission cannot tie or dethrone the earlier-committed
    king — whoever committed first holds the throne."""

    _HASH_A = "a" * 64
    _HASH_B = "b" * 64

    def test_later_duplicate_is_dqd_and_doesnt_tie(self):
        state = ValidatorState()
        _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=100, score=0.5,
            source_hash=self._HASH_A,
        ))
        ev_copy = _make_eval(
            uid=2, hotkey="hk2", commit_block=200, score=0.5,
            source_hash=self._HASH_A,
        )
        out = _record(state, ev_copy)
        assert out.dethroned is False
        assert out.stored.disqualified is True
        assert out.stored.disqualify_reason == DUPLICATE_OF_KING_REASON
        assert out.stored.score == 0.0
        # Persisted copy reflects the DQ, not the claimed score
        persisted = state.evaluations[ev_copy.eval_key]
        assert persisted.disqualified is True
        assert state.king.uid == 1

    def test_later_duplicate_with_higher_score_still_dqd(self):
        """Can't buy your way past the DQ with noise — if the source is
        byte-identical, the later committer is out regardless of score."""
        state = ValidatorState()
        _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=100, score=0.5,
            source_hash=self._HASH_A,
        ))
        out = _record(state, _make_eval(
            uid=2, hotkey="hk2", commit_block=200, score=0.99,
            source_hash=self._HASH_A,
        ))
        assert out.dethroned is False
        assert out.stored.disqualified is True
        assert state.king.uid == 1

    def test_same_hash_same_hotkey_not_dqd(self):
        """Re-committing your own winning policy at a later block is fine —
        the DQ rule targets cross-hotkey copies only."""
        state = ValidatorState()
        _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=100, score=0.5,
            source_hash=self._HASH_A,
        ))
        out = _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=200, score=0.4,
            source_hash=self._HASH_A,
        ))
        assert out.stored.disqualified is False

    def test_earlier_commit_not_dqd(self):
        """A submission at a commit_block before the king's own commit
        never gets duplicate-of-king DQ (could happen if evals land out of
        chronological order). commit_block is authoritative."""
        state = ValidatorState()
        _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=200, score=0.5,
            source_hash=self._HASH_A,
        ))
        out = _record(state, _make_eval(
            uid=2, hotkey="hk2", commit_block=100, score=0.49,
            source_hash=self._HASH_A,
        ))
        assert out.stored.disqualified is False

    def test_different_hash_not_dqd(self):
        state = ValidatorState()
        _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=100, score=0.5,
            source_hash=self._HASH_A,
        ))
        out = _record(state, _make_eval(
            uid=2, hotkey="hk2", commit_block=200, score=0.4,
            source_hash=self._HASH_B,
        ))
        assert out.stored.disqualified is False

    def test_empty_hash_does_not_trigger_dq(self):
        """Empty source_hash = unknown; never trips the DQ rule (avoids
        false-positive on legacy rows written before the field existed)."""
        state = ValidatorState()
        _record(state, _make_eval(
            uid=1, hotkey="hk1", commit_block=100, score=0.5, source_hash="",
        ))
        out = _record(state, _make_eval(
            uid=2, hotkey="hk2", commit_block=200, score=0.4, source_hash="",
        ))
        assert out.stored.disqualified is False


class TestEffectiveDethroneThreshold:
    def test_no_king_returns_zero_call_site_convention(self):
        # The helper itself requires a king_score; the zero-king case is
        # handled by `record_evaluation`, see `TestValidatorStateRecording`.
        assert _effective_dethrone_threshold(0.0, 0, 100) == 0.0

    def test_at_crowning_block_full_epsilon(self):
        th = _effective_dethrone_threshold(0.5, 1000, 1000)
        assert th == pytest.approx(0.5 * (1 + KING_EPSILON_INITIAL))

    def test_half_window_half_epsilon(self):
        half = KING_EPSILON_DECAY_BLOCKS // 2
        th = _effective_dethrone_threshold(0.5, 0, half)
        expected = 0.5 * (1 + KING_EPSILON_INITIAL * (1 - half / KING_EPSILON_DECAY_BLOCKS))
        assert th == pytest.approx(expected)

    def test_at_window_end_no_moat(self):
        th = _effective_dethrone_threshold(
            0.5, 0, KING_EPSILON_DECAY_BLOCKS,
        )
        assert th == pytest.approx(0.5)

    def test_past_window_clamped_to_score(self):
        th = _effective_dethrone_threshold(
            0.5, 0, KING_EPSILON_DECAY_BLOCKS * 10,
        )
        assert th == pytest.approx(0.5)

    def test_negative_age_clamped_to_zero(self):
        """Defensive: a current_block earlier than the crowning block
        should not inflate the moat above its initial value."""
        th = _effective_dethrone_threshold(0.5, 1000, 500)
        assert th == pytest.approx(0.5 * (1 + KING_EPSILON_INITIAL))

    def test_zero_decay_blocks_disables_moat(self):
        th = _effective_dethrone_threshold(
            0.5, 100, 200, decay_blocks=0,
        )
        assert th == 0.5

    def test_zero_initial_disables_moat(self):
        th = _effective_dethrone_threshold(
            0.5, 100, 100, epsilon_initial=0.0,
        )
        assert th == 0.5


class TestScoreHistory:
    def test_history_ordered_by_commit_block(self):
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk_a", commit_block=30))
        _record(state, _make_eval(hotkey="hk_a", commit_block=10))
        _record(state, _make_eval(hotkey="hk_b", commit_block=20))
        history = state.score_history_for_hotkey("hk_a")
        assert [e.commit_block for e in history] == [10, 30]

    def test_history_empty_for_unknown_hotkey(self):
        state = ValidatorState()
        assert state.score_history_for_hotkey("hk_nobody") == []


class TestPersistence:
    def test_save_then_load_round_trip(self, tmp_path):
        state = ValidatorState()
        _record(
            state,
            _make_eval(uid=1, hotkey="hk1", score=0.4, source_hash="a" * 64),
        )
        _record(
            state,
            _make_eval(
                uid=2, hotkey="hk2", commit_block=200, score=0.6,
                source_hash="b" * 64,
            ),
            current_block=210,
        )
        state.record_precheck_failure("hk3", 300, "blocked call: eval")
        state.last_scan_block = 1234
        state.last_weights_set_block = 1234

        state.save(tmp_path)
        reloaded = ValidatorState.load(tmp_path)

        assert reloaded.king is not None
        assert reloaded.king.uid == 2
        assert reloaded.king.score == 0.6
        assert reloaded.king.source_hash == "b" * 64
        assert reloaded.king.crowned_at_block == 210
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
        _record(state, _make_eval(hotkey="hk1", commit_block=100))
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
        _record(state, _make_eval())
        clone = state.clone()
        _record(
            clone,
            _make_eval(uid=99, hotkey="hk_new", commit_block=999, score=0.9),
        )
        assert state.king.uid == 1
        assert clone.king.uid == 99

    def test_current_timestamp_is_monotonic_enough(self):
        assert current_timestamp() > 0
