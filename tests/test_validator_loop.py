"""Unit tests for validator.loop.run_once — stubs out chain + eval_fn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from validator.chain import CommitmentRecord
from validator.challengers import PrecheckOutcome, PrecheckResult
from validator.loop import not_implemented_eval, run_once
from validator.state import EvaluationRecord, ValidatorState

pytestmark = pytest.mark.unit


@dataclass
class FakeMetagraph:
    hotkeys: list[str]


class FakeSubtensor:
    """Minimal in-memory substitute for bittensor.subtensor."""

    def __init__(
        self,
        *,
        hotkeys: list[str],
        revealed: dict[str, list[tuple[int, str]]],
        current_block: int = 1000,
        block_hash: str | None = "0xdeadbeef",
    ) -> None:
        self._hotkeys = hotkeys
        self._revealed = revealed
        self.block = current_block
        self._block_hash = block_hash
        self.substrate = self  # self-hosted get_block_hash
        self.set_weights_calls: list[dict[str, Any]] = []
        self._set_weights_result: Any = (True, "")

    def metagraph(self, _netuid: int = 14):
        return FakeMetagraph(hotkeys=list(self._hotkeys))

    def get_block_hash(self, _block: int):
        if self._block_hash is None:
            raise RuntimeError("block hash unavailable")
        return self._block_hash

    def get_all_revealed_commitments(self, _netuid: int):
        return self._revealed

    def set_weights(self, **kwargs):
        self.set_weights_calls.append(kwargs)
        return self._set_weights_result


FAKE_WALLET = object()


_DIGEST_DEFAULT = "sha256:" + "a" * 64


def _commit_json(image: str = "user/repo:v1", digest: str = _DIGEST_DEFAULT) -> str:
    return json.dumps({"image": image, "digest": digest})


def _make_eval_record(
    com: CommitmentRecord,
    score: float,
    disqualified: bool = False,
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=com.uid,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        image=com.image,
        digest=com.digest,
        score=score,
        ttft_improvement=score * 0.5,
        throughput_improvement=score * 0.5,
        token_match_rate=0.995,
        disqualified=disqualified,
        disqualify_reason="dq" if disqualified else None,
        evaluated_at=1700000000.0,
        evaluation_block=1000,
    )


def _seed_king(state: ValidatorState, com: CommitmentRecord, score: float) -> None:
    """Seed a pre-existing king for tests that exercise `run_once` against
    an already-decided state. Uses a current_block well past the king's
    commit_block so epsilon decay doesn't interfere with the test."""
    state.record_evaluation(
        _make_eval_record(com, score=score),
        current_block=com.commit_block + 10,
    )


class TestRunOnceNoChallengers:
    def test_empty_chain_no_king_no_weights(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1"], revealed={})
        state = ValidatorState()

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=True,
        )

        assert result.n_commitments == 0
        assert len(result.challenger_set) == 0
        assert result.evaluations_recorded == []
        assert result.king_changed is False
        assert result.weights_set is False
        assert state.king is None
        assert state.last_scan_block == 1000

    def test_dry_run_with_existing_king_sets_no_real_weights(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=0,
                hotkey="hk0",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.5,
        )

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=True,
        )
        assert result.weights_set is True
        assert st.set_weights_calls == []

    def test_live_mode_calls_set_weights_for_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1", "hk2"], revealed={})
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.4,
        )

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=False,
        )
        assert result.weights_set is True
        assert len(st.set_weights_calls) == 1
        call = st.set_weights_calls[0]
        assert call["netuid"] == 14
        assert call["uids"] == [0, 1, 2]
        assert call["weights"] == [0.0, 1.0, 0.0]
        # version_key is threaded through from VERSION_KEY; default is 1
        assert call["version_key"] == 1
        assert state.last_weights_set_block == 1000

    def test_version_key_can_be_overridden(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1"], revealed={})
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.4,
        )

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=False,
            version_key=42,
        )
        assert st.set_weights_calls[0]["version_key"] == 42


class TestRunOnceWithChallengers:
    def test_new_challenger_becomes_king(self, tmp_path):
        hotkeys = ["hk0", "hk1"]
        st = FakeSubtensor(
            hotkeys=hotkeys,
            revealed={"hk1": [(100, _commit_json("user/m1:v1", "sha256:" + "1" * 64))]},
        )
        state = ValidatorState()

        def eval_fn(challengers, *, current_block, block_hash):
            assert len(challengers) == 1
            return [_make_eval_record(challengers[0], score=0.5)]

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )

        assert result.n_commitments == 1
        assert len(result.challenger_set) == 1
        assert len(result.evaluations_recorded) == 1
        assert result.king_changed is True
        assert state.king is not None
        assert state.king.uid == 1
        assert state.king.score == 0.5

    def test_existing_evaluations_not_re_evaluated(self, tmp_path):
        hotkeys = ["hk0"]
        st = FakeSubtensor(
            hotkeys=hotkeys,
            revealed={"hk0": [(100, _commit_json())]},
        )
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=0,
                hotkey="hk0",
                commit_block=100,
                image="user/repo:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.3,
        )

        eval_calls = []

        def eval_fn(challengers, *, current_block, block_hash):
            eval_calls.append(challengers)
            return []

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )
        assert eval_calls == []  # eval_fn never called

    def test_eval_fn_exception_does_not_crash_tick(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json())]},
        )
        state = ValidatorState()

        def eval_fn(*_a, **_k):
            raise RuntimeError("pod went down")

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )
        assert result.evaluations_recorded == []
        assert result.king_changed is False

    def test_precheck_rejection_is_recorded_in_state(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={
                "hk0": [(100, _commit_json("user/a:v1", "sha256:" + "a" * 64))],
                "hk1": [(200, _commit_json("user/b:v1", "sha256:" + "b" * 64))],
            },
        )
        state = ValidatorState()

        def reject_uid_0(com):
            if com.uid == 0:
                return PrecheckResult(
                    outcome=PrecheckOutcome.REJECTED,
                    reason="blocked import: os",
                )
            return PrecheckResult(outcome=PrecheckOutcome.OK)

        ev_seen = []

        def eval_fn(challengers, **_k):
            ev_seen.extend([c.uid for c in challengers])
            return [_make_eval_record(c, score=0.2 + c.uid * 0.1) for c in challengers]

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            precheck=reject_uid_0,
            state_dir=tmp_path,
            dry_run=False,
        )

        assert ev_seen == [1]
        assert state.has_precheck_failure("hk0", 100)
        assert state.has_evaluation("hk1", 200)

    def test_default_eval_hook_raises_on_challengers(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json())]},
        )
        state = ValidatorState()

        with pytest.raises(NotImplementedError):
            run_once(
                subtensor=st,
                wallet=FAKE_WALLET,
                state=state,
                netuid=14,
                eval_fn=not_implemented_eval,
                state_dir=tmp_path,
                dry_run=False,
            )

    def test_dry_run_short_circuits_before_eval_fn(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json())]},
        )
        state = ValidatorState()

        eval_called = False

        def eval_fn(challengers, *, current_block, block_hash):
            nonlocal eval_called
            eval_called = True
            return []

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=True,
        )
        assert not eval_called
        assert len(result.challenger_set.challengers) == 1
        assert result.evaluations_recorded == []

    def test_deferred_not_recorded_in_state(self, tmp_path):
        from validator.challengers import PrecheckOutcome, PrecheckResult

        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={
                "hk0": [(100, _commit_json("user/a:v1", "sha256:" + "a" * 64))],
                "hk1": [(200, _commit_json("user/b:v1", "sha256:" + "b" * 64))],
            },
        )
        state = ValidatorState()

        def defer_uid_0(com):
            if com.uid == 0:
                return PrecheckResult(
                    outcome=PrecheckOutcome.DEFERRED,
                    reason="network flake",
                )
            return PrecheckResult(outcome=PrecheckOutcome.OK)

        ev_seen = []

        def eval_fn(challengers, **_k):
            ev_seen.extend([c.uid for c in challengers])
            return [_make_eval_record(c, score=0.2 + c.uid * 0.1) for c in challengers]

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            precheck=defer_uid_0,
            state_dir=tmp_path,
            dry_run=False,
        )

        assert ev_seen == [1]
        assert len(result.challenger_set.deferred) == 1
        assert not state.has_precheck_failure("hk0", 100)
        assert state.has_evaluation("hk1", 200)

    def test_deferred_retries_across_ticks(self, tmp_path):
        from validator.challengers import PrecheckOutcome, PrecheckResult

        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json("user/a:v1", "sha256:" + "a" * 64))]},
        )
        state = ValidatorState()

        call_count = 0

        def defer_once_then_pass(com):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PrecheckResult(
                    outcome=PrecheckOutcome.DEFERRED,
                    reason="network flake",
                )
            return PrecheckResult(outcome=PrecheckOutcome.OK)

        def eval_fn(challengers, **_k):
            return [_make_eval_record(challengers[0], score=0.5)]

        # First tick — deferred
        result1 = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            precheck=defer_once_then_pass,
            state_dir=tmp_path,
            dry_run=False,
        )
        assert len(result1.challenger_set.deferred) == 1
        assert len(result1.evaluations_recorded) == 0

        # Second tick — same commitment, now passes
        result2 = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            precheck=defer_once_then_pass,
            state_dir=tmp_path,
            dry_run=False,
        )
        assert len(result2.challenger_set.deferred) == 0
        assert len(result2.evaluations_recorded) == 1
        assert state.king is not None

    def test_dry_run_records_precheck_rejections(self, tmp_path):
        from validator.challengers import PrecheckOutcome, PrecheckResult

        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={
                "hk0": [(100, _commit_json("user/a:v1", "sha256:" + "a" * 64))],
                "hk1": [(200, _commit_json("user/b:v1", "sha256:" + "b" * 64))],
            },
        )
        state = ValidatorState()

        def reject_uid_0(com):
            if com.uid == 0:
                return PrecheckResult(
                    outcome=PrecheckOutcome.REJECTED,
                    reason="ast_blocked: import os",
                )
            return PrecheckResult(outcome=PrecheckOutcome.OK)

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            precheck=reject_uid_0,
            state_dir=tmp_path,
            dry_run=True,
        )

        assert len(result.challenger_set.newly_rejected) == 1
        assert state.has_precheck_failure("hk0", 100)
        assert not state.is_known("hk1", 200)

    def test_disqualified_challenger_does_not_dethrone(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={"hk1": [(200, _commit_json("user/b:v1", "sha256:" + "b" * 64))]},
        )
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=0,
                hotkey="hk0",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.3,
        )

        def eval_fn(challengers, **_k):
            return [_make_eval_record(challengers[0], score=0.9, disqualified=True)]

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )
        assert result.king_changed is False
        assert state.king.uid == 0


class TestRunOnceStatePersistence:
    def test_state_written_to_disk(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json())]},
        )
        state = ValidatorState()

        def eval_fn(challengers, **_k):
            return [_make_eval_record(challengers[0], score=0.5)]

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )

        reloaded = ValidatorState.load(tmp_path)
        assert reloaded.king is not None
        assert reloaded.king.uid == 0
        assert reloaded.king.score == 0.5
        assert reloaded.last_scan_block == 1000


class TestRunOnceKingHistoryJsonl:
    """Verify that king-history.jsonl is written on dethronement."""

    def test_first_king_creates_jsonl(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={"hk1": [(100, _commit_json("user/m1:v1", "sha256:" + "1" * 64))]},
        )
        state = ValidatorState()

        def eval_fn(challengers, *, current_block, block_hash):
            return [_make_eval_record(challengers[0], score=0.5)]

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )

        jsonl = tmp_path / "king-history.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["new_king_uid"] == 1
        assert "prev_king_uid" not in entry

    def test_no_jsonl_when_no_dethronement(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={"hk1": [(200, _commit_json("user/b:v1", "sha256:" + "b" * 64))]},
        )
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=0,
                hotkey="hk0",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.3,
        )

        def eval_fn(challengers, **_k):
            return [_make_eval_record(challengers[0], score=0.9, disqualified=True)]

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )

        assert not (tmp_path / "king-history.jsonl").exists()

    def test_dethronement_records_prev_king(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={"hk1": [(200, _commit_json("user/b:v1", "sha256:" + "b" * 64))]},
        )
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=0,
                hotkey="hk0",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.3,
        )

        def eval_fn(challengers, **_k):
            return [_make_eval_record(challengers[0], score=0.9)]

        run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            eval_fn=eval_fn,
            state_dir=tmp_path,
            dry_run=False,
        )

        jsonl = tmp_path / "king-history.jsonl"
        assert jsonl.exists()
        entry = json.loads(jsonl.read_text().strip())
        assert entry["new_king_uid"] == 1
        assert entry["prev_king_uid"] == 0
        assert entry["prev_king_hotkey"] == "hk0"


class TestRunOnceKingUidRecycled:
    """UID recycling guard: if `metagraph.hotkeys[king.uid]` no longer matches
    `king.hotkey`, the king's original hotkey has deregistered and the UID has
    been reassigned. Setting weights would then emit to an unevaluated
    stranger — we must drop the king instead."""

    def test_recycled_uid_clears_king_and_skips_weights(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk_new_miner"], revealed={})
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=1,
                hotkey="hk_old_king",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.5,
        )
        assert state.king is not None and state.king.uid == 1

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=False,
        )

        assert state.king is None
        assert result.weights_set is False
        assert st.set_weights_calls == []

    def test_king_uid_out_of_range_clears_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=5,
                hotkey="hk_old_king",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.5,
        )
        assert state.king is not None

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=True,
        )

        assert state.king is None
        assert result.weights_set is False

    def test_matching_hotkey_keeps_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk_king"], revealed={})
        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=1,
                hotkey="hk_king",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.5,
        )

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=False,
        )

        assert state.king is not None and state.king.uid == 1
        assert result.weights_set is True
        assert len(st.set_weights_calls) == 1


class TestRunOnceWeightsFailure:
    def test_weights_rpc_failure_is_caught(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        st._set_weights_result = (False, "retry rate limit")  # always fail

        state = ValidatorState()
        _seed_king(
            state,
            CommitmentRecord(
                uid=0,
                hotkey="hk0",
                commit_block=100,
                image="m:v1",
                digest=_DIGEST_DEFAULT,
                raw="",
            ),
            score=0.5,
        )

        result = run_once(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=tmp_path,
            dry_run=False,
            # tiny retry config so this test is fast
            chain_attempts=1,
            chain_delay_s=0,
        )
        assert result.weights_set is False
        assert result.weights_set_error is not None
