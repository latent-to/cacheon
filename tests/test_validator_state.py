"""Unit tests for validator.state -- no chain, no GPU, no bittensor."""

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
    append_king_history,
    current_timestamp,
    unknown_commits,
)

pytestmark = pytest.mark.unit


def _make_eval(
    uid: int = 1,
    hotkey: str = "hk_alice",
    commit_block: int = 100,
    score: float = 0.25,
    ttft_improvement: float = 0.15,
    throughput_improvement: float = 0.35,
    token_match_rate: float = 0.995,
    disqualified: bool = False,
    reason: str | None = None,
    image: str = "user/server:latest",
    digest: str | None = None,
) -> EvaluationRecord:
    if digest is None:
        digest = "sha256:" + format(uid, "x").zfill(64)
    return EvaluationRecord(
        uid=uid,
        hotkey=hotkey,
        commit_block=commit_block,
        image=image,
        digest=digest,
        score=score,
        ttft_improvement=ttft_improvement,
        throughput_improvement=throughput_improvement,
        token_match_rate=token_match_rate,
        disqualified=disqualified,
        disqualify_reason=reason,
        evaluated_at=1700000000.0,
        evaluation_block=commit_block + 10,
    )


def _record(
    state: ValidatorState, ev: EvaluationRecord, *, current_block: int | None = None
) -> RecordResult:
    """Shorthand -- tests that don't care about the block pass
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

    def test_new_metric_fields(self):
        ev = _make_eval(
            ttft_improvement=0.22,
            throughput_improvement=0.44,
            token_match_rate=0.998,
        )
        assert ev.ttft_improvement == 0.22
        assert ev.throughput_improvement == 0.44
        assert ev.token_match_rate == 0.998

    def test_per_prompt_defaults_to_none(self):
        ev = _make_eval()
        assert ev.per_prompt is None

    def test_per_prompt_round_trip(self):
        pp = [
            {
                "ttft_s": 0.5,
                "throughput_tps": 100.0,
                "output_tokens": 256,
                "token_match_rate": 1.0,
            },
            {
                "ttft_s": 0.6,
                "throughput_tps": 90.0,
                "output_tokens": 200,
                "token_match_rate": 0.99,
            },
        ]
        ev = EvaluationRecord(
            uid=1,
            hotkey="hk",
            commit_block=100,
            image="i:v1",
            digest="sha256:" + "a" * 64,
            score=0.5,
            ttft_improvement=0.1,
            throughput_improvement=0.2,
            token_match_rate=0.99,
            disqualified=False,
            disqualify_reason=None,
            evaluated_at=1.0,
            evaluation_block=110,
            per_prompt=pp,
        )
        d = ev.to_dict()
        assert d["per_prompt"] == pp
        restored = EvaluationRecord.from_dict(d)
        assert restored.per_prompt == pp

    def test_per_prompt_none_omitted_from_dict(self):
        ev = _make_eval()
        d = ev.to_dict()
        assert "per_prompt" not in d

    def test_from_dict_without_per_prompt_is_backward_compatible(self):
        ev = _make_eval()
        d = ev.to_dict()
        assert "per_prompt" not in d
        restored = EvaluationRecord.from_dict(d)
        assert restored.per_prompt is None


class TestKingRecord:
    def test_from_evaluation(self):
        ev = _make_eval(score=0.5)
        king = KingRecord.from_evaluation(ev, crowned_at_block=500)
        assert king.uid == ev.uid
        assert king.score == ev.score
        assert king.hotkey == ev.hotkey
        assert king.crowned_at_block == 500
        assert king.ttft_improvement == ev.ttft_improvement
        assert king.throughput_improvement == ev.throughput_improvement
        assert king.token_match_rate == ev.token_match_rate

    def test_round_trip(self):
        ev = _make_eval()
        king = KingRecord.from_evaluation(ev, crowned_at_block=123)
        restored = KingRecord.from_dict(king.to_dict())
        assert restored == king

    def test_image_digest_preserved(self):
        digest = "sha256:" + "b" * 64
        ev = _make_eval(digest=digest)
        king = KingRecord.from_evaluation(ev, crowned_at_block=123)
        assert king.digest == digest


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
        assert out.dethrone_threshold == 0.0
        assert state.king is not None
        assert state.king.uid == ev.uid
        assert state.king.score == ev.score
        assert state.has_evaluation(ev.hotkey, ev.commit_block)

    def test_higher_score_dethrones_king(self):
        state = ValidatorState()
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
        state = ValidatorState()
        _record(
            state,
            _make_eval(uid=1, hotkey="hk1", score=0.5, commit_block=100),
            current_block=100,
        )
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", score=0.5025, commit_block=110),
            current_block=110,
        )
        assert out.dethroned is False
        assert out.dethrone_threshold > 0.5
        assert state.king.uid == 1

    def test_epsilon_fully_decayed_allows_strict_improvement(self):
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
            _make_eval(
                score=0.0, disqualified=True, reason="token_match_below_threshold"
            ),
        )
        assert out.dethroned is False
        assert state.king is None

    def test_disqualified_cannot_dethrone_king(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.2))
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.9,
                disqualified=True,
                reason="nan",
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
        state = ValidatorState()
        out = _record(state, _make_eval(score=float("nan")))
        assert out.dethroned is False
        assert state.king is None

    def test_nan_score_cannot_dethrone_existing_king(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.3))
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", score=float("nan")),
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
        state.record_precheck_failure("hk1", 100, "container startup timeout")
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
    """Byte-identical Docker image (same digest) cannot tie or dethrone the
    earlier-committed king -- whoever committed first holds the throne."""

    _DIGEST_A = "sha256:" + "a" * 64
    _DIGEST_B = "sha256:" + "b" * 64

    def test_later_duplicate_is_dqd_and_doesnt_tie(self):
        state = ValidatorState()
        _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        ev_copy = _make_eval(
            uid=2,
            hotkey="hk2",
            commit_block=200,
            score=0.5,
            digest=self._DIGEST_A,
        )
        out = _record(state, ev_copy)
        assert out.dethroned is False
        assert out.stored.disqualified is True
        assert out.stored.disqualify_reason == DUPLICATE_OF_KING_REASON
        assert out.stored.score == 0.0
        persisted = state.evaluations[ev_copy.eval_key]
        assert persisted.disqualified is True
        assert state.king.uid == 1

    def test_later_duplicate_with_higher_score_still_dqd(self):
        state = ValidatorState()
        _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.99,
                digest=self._DIGEST_A,
            ),
        )
        assert out.dethroned is False
        assert out.stored.disqualified is True
        assert state.king.uid == 1

    def test_same_digest_same_hotkey_not_dqd(self):
        """Re-committing your own winning image at a later block is fine --
        the DQ rule targets cross-hotkey copies only."""
        state = ValidatorState()
        _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=200,
                score=0.4,
                digest=self._DIGEST_A,
            ),
        )
        assert out.stored.disqualified is False

    def test_earlier_commit_not_dqd(self):
        """A submission at a commit_block before the king's own commit
        never gets duplicate-of-king DQ."""
        state = ValidatorState()
        _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=200,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=100,
                score=0.49,
                digest=self._DIGEST_A,
            ),
        )
        assert out.stored.disqualified is False

    def test_different_digest_not_dqd(self):
        state = ValidatorState()
        _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.4,
                digest=self._DIGEST_B,
            ),
        )
        assert out.stored.disqualified is False

    def test_empty_digest_does_not_trigger_dq(self):
        """Empty digest = unknown; never trips the DQ rule."""
        state = ValidatorState()
        _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest="",
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.4,
                digest="",
            ),
        )
        assert out.stored.disqualified is False


class TestEffectiveDethroneThreshold:
    def test_no_king_returns_zero_call_site_convention(self):
        assert _effective_dethrone_threshold(0.0, 0, 100) == 0.0

    def test_at_crowning_block_full_epsilon(self):
        th = _effective_dethrone_threshold(0.5, 1000, 1000)
        assert th == pytest.approx(0.5 * (1 + KING_EPSILON_INITIAL))

    def test_half_window_half_epsilon(self):
        half = KING_EPSILON_DECAY_BLOCKS // 2
        th = _effective_dethrone_threshold(0.5, 0, half)
        expected = 0.5 * (
            1 + KING_EPSILON_INITIAL * (1 - half / KING_EPSILON_DECAY_BLOCKS)
        )
        assert th == pytest.approx(expected)

    def test_at_window_end_no_moat(self):
        th = _effective_dethrone_threshold(0.5, 0, KING_EPSILON_DECAY_BLOCKS)
        assert th == pytest.approx(0.5)

    def test_past_window_clamped_to_score(self):
        th = _effective_dethrone_threshold(0.5, 0, KING_EPSILON_DECAY_BLOCKS * 10)
        assert th == pytest.approx(0.5)

    def test_negative_age_clamped_to_zero(self):
        th = _effective_dethrone_threshold(0.5, 1000, 500)
        assert th == pytest.approx(0.5 * (1 + KING_EPSILON_INITIAL))

    def test_zero_decay_blocks_disables_moat(self):
        th = _effective_dethrone_threshold(0.5, 100, 200, decay_blocks=0)
        assert th == 0.5

    def test_zero_initial_disables_moat(self):
        th = _effective_dethrone_threshold(0.5, 100, 100, epsilon_initial=0.0)
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
            _make_eval(uid=1, hotkey="hk1", score=0.4, digest="sha256:" + "a" * 64),
        )
        _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.6,
                digest="sha256:" + "b" * 64,
            ),
            current_block=210,
        )
        state.record_precheck_failure("hk3", 300, "container startup timeout")
        state.last_scan_block = 1234
        state.last_weights_set_block = 1234

        state.save(tmp_path)
        reloaded = ValidatorState.load(tmp_path)

        assert reloaded.king is not None
        assert reloaded.king.uid == 2
        assert reloaded.king.score == 0.6
        assert reloaded.king.digest == "sha256:" + "b" * 64
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

    def test_load_corrupt_file_is_quarantined(self, tmp_path):
        state_path = tmp_path / STATE_FILE_NAME
        state_path.write_text("{not valid json")
        loaded = ValidatorState.load(tmp_path)
        assert loaded.king is None
        assert not state_path.exists()
        quarantined = list(tmp_path.glob(f"{STATE_FILE_NAME}.corrupt.*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text() == "{not valid json"

    def test_load_stale_field_names_does_not_crash(self, tmp_path):
        """A state file with old KV-cache field names must fall back to
        fresh state, not crash on startup."""
        old_payload = {
            "schema_version": 1,
            "king": {
                "uid": 1,
                "hotkey": "hk_old",
                "commit_block": 100,
                "image": "old/server:latest",
                "digest": "sha256:" + "a" * 64,
                "score": 0.5,
                "kl_divergence": 0.01,
                "memory_reduction": 0.4,
                "latency_improvement": 0.2,
                "evaluated_at": 123.0,
                "evaluation_block": 150,
            },
            "evaluations": {
                "hk_old:100": {
                    "uid": 1,
                    "hotkey": "hk_old",
                    "commit_block": 100,
                    "image": "old/server:latest",
                    "digest": "sha256:" + "a" * 64,
                    "score": 0.5,
                    "kl_divergence": 0.01,
                    "memory_reduction": 0.4,
                    "latency_improvement": 0.2,
                    "disqualified": False,
                    "disqualify_reason": None,
                    "evaluated_at": 123.0,
                    "evaluation_block": 150,
                },
            },
            "precheck_failures": {},
            "last_scan_block": 0,
            "last_weights_set_block": 0,
        }
        (tmp_path / STATE_FILE_NAME).write_text(json.dumps(old_payload))
        loaded = ValidatorState.load(tmp_path)
        assert loaded.king is None
        assert loaded.evaluations == {}

    def test_load_malformed_record_does_not_crash(self, tmp_path):
        bad_payload = {
            "schema_version": SCHEMA_VERSION,
            "king": None,
            "evaluations": {
                "hk:100": {"not": "a real record"},
            },
            "precheck_failures": {},
            "last_scan_block": 0,
            "last_weights_set_block": 0,
        }
        (tmp_path / STATE_FILE_NAME).write_text(json.dumps(bad_payload))
        loaded = ValidatorState.load(tmp_path)
        assert loaded.king is None
        assert loaded.evaluations == {}

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
            ("hk1", 100),
            ("hk2", 200),
            ("hk3", 300),
            ("hk1", 150),
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


class TestAppendKingHistory:
    def _king(self, uid=1, score=0.5) -> KingRecord:
        return KingRecord(
            uid=uid,
            hotkey=f"hk{uid}",
            commit_block=100,
            image="user/server:latest",
            digest="sha256:" + "a" * 64,
            score=score,
            ttft_improvement=0.15,
            throughput_improvement=0.35,
            token_match_rate=0.995,
            evaluated_at=1700000000.0,
            evaluation_block=1000,
            crowned_at_block=1000,
        )

    def test_first_king_no_prev(self, tmp_path):
        ev = _make_eval(uid=1, score=0.5)
        append_king_history(
            tmp_path, ev, None, current_block=1000, dethrone_threshold=0.0
        )
        lines = (tmp_path / "king-history.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["new_king_uid"] == 1
        assert entry["block"] == 1000
        assert "prev_king_uid" not in entry

    def test_dethronement_includes_prev_king(self, tmp_path):
        ev = _make_eval(uid=2, hotkey="hk2", score=0.6)
        prev = self._king(uid=1, score=0.4)
        append_king_history(
            tmp_path, ev, prev, current_block=2000, dethrone_threshold=0.404
        )
        entry = json.loads((tmp_path / "king-history.jsonl").read_text().strip())
        assert entry["prev_king_uid"] == 1
        assert entry["prev_king_hotkey"] == "hk1"
        assert entry["prev_king_score"] == 0.4
        assert entry["new_king_score"] == 0.6

    def test_multiple_appends(self, tmp_path):
        for i in range(3):
            ev = _make_eval(uid=i, hotkey=f"hk{i}", score=0.1 * (i + 1))
            append_king_history(
                tmp_path, ev, None, current_block=1000 + i, dethrone_threshold=0.0
            )
        lines = (tmp_path / "king-history.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[2])["new_king_uid"] == 2

    def test_write_failure_does_not_raise(self, tmp_path):
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        (ro_dir / "king-history.jsonl").write_text("")
        (ro_dir / "king-history.jsonl").chmod(0o000)
        ev = _make_eval()
        append_king_history(ro_dir, ev, None, current_block=1, dethrone_threshold=0.0)
