"""Unit tests for validator.loop.run_once — stubs out chain + eval_fn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from validator.chain import CommitmentRecord
from validator.challengers import PrecheckResult
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


def _commit_json(model: str = "hf/repo", rev: str = "abc") -> str:
    return json.dumps({"model": model, "revision": rev})


def _make_eval_record(
    com: CommitmentRecord, score: float, disqualified: bool = False,
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=com.uid,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        model=com.model,
        revision=com.revision,
        score=score,
        kl_divergence=0.01,
        memory_reduction=score * 0.6,
        latency_improvement=score * 0.4,
        disqualified=disqualified,
        disqualify_reason="dq" if disqualified else None,
        evaluated_at=1700000000.0,
        evaluation_block=1000,
    )


class TestRunOnceNoChallengers:
    def test_empty_chain_no_king_no_weights(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1"], revealed={})
        state = ValidatorState()

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=True,
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
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=0, hotkey="hk0", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.5,
        ))

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=True,
        )
        assert result.weights_set is True
        assert st.set_weights_calls == []

    def test_live_mode_calls_set_weights_for_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1", "hk2"], revealed={})
        state = ValidatorState()
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=1, hotkey="hk1", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.4,
        ))

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=False,
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
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=1, hotkey="hk1", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.4,
        ))

        run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=False,
            version_key=42,
        )
        assert st.set_weights_calls[0]["version_key"] == 42


class TestRunOnceWithChallengers:
    def test_new_challenger_becomes_king(self, tmp_path):
        hotkeys = ["hk0", "hk1"]
        st = FakeSubtensor(
            hotkeys=hotkeys,
            revealed={"hk1": [(100, _commit_json("hf/m1", "rev1"))]},
        )
        state = ValidatorState()

        def eval_fn(challengers, *, current_block, block_hash):
            assert len(challengers) == 1
            return [_make_eval_record(challengers[0], score=0.5)]

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, eval_fn=eval_fn,
            state_dir=tmp_path, dry_run=True,
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
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=0, hotkey="hk0", commit_block=100,
                             model="hf/repo", revision="abc", raw=""),
            score=0.3,
        ))

        eval_calls = []

        def eval_fn(challengers, *, current_block, block_hash):
            eval_calls.append(challengers)
            return []

        run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, eval_fn=eval_fn,
            state_dir=tmp_path, dry_run=True,
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
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, eval_fn=eval_fn,
            state_dir=tmp_path, dry_run=True,
        )
        assert result.evaluations_recorded == []
        assert result.king_changed is False

    def test_precheck_rejection_is_recorded_in_state(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={
                "hk0": [(100, _commit_json("hf/a", "r0"))],
                "hk1": [(200, _commit_json("hf/b", "r1"))],
            },
        )
        state = ValidatorState()

        def reject_uid_0(com):
            if com.uid == 0:
                return PrecheckResult(ok=False, reason="blocked import: os")
            return PrecheckResult(ok=True)

        ev_seen = []

        def eval_fn(challengers, **_k):
            ev_seen.extend([c.uid for c in challengers])
            return [_make_eval_record(c, score=0.2 + c.uid * 0.1) for c in challengers]

        run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, eval_fn=eval_fn, precheck=reject_uid_0,
            state_dir=tmp_path, dry_run=True,
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
                subtensor=st, wallet=FAKE_WALLET, state=state,
                netuid=14, eval_fn=not_implemented_eval,
                state_dir=tmp_path, dry_run=True,
            )

    def test_disqualified_challenger_does_not_dethrone(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={"hk1": [(200, _commit_json("hf/b", "r1"))]},
        )
        state = ValidatorState()
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=0, hotkey="hk0", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.3,
        ))

        def eval_fn(challengers, **_k):
            return [_make_eval_record(challengers[0], score=0.9, disqualified=True)]

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, eval_fn=eval_fn,
            state_dir=tmp_path, dry_run=True,
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
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, eval_fn=eval_fn,
            state_dir=tmp_path, dry_run=True,
        )

        reloaded = ValidatorState.load(tmp_path)
        assert reloaded.king is not None
        assert reloaded.king.uid == 0
        assert reloaded.king.score == 0.5
        assert reloaded.last_scan_block == 1000


class TestRunOnceKingUidRecycled:
    """UID recycling guard: if `metagraph.hotkeys[king.uid]` no longer matches
    `king.hotkey`, the king's original hotkey has deregistered and the UID has
    been reassigned. Setting weights would then emit to an unevaluated
    stranger — we must drop the king instead."""

    def test_recycled_uid_clears_king_and_skips_weights(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk_new_miner"], revealed={})
        state = ValidatorState()
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=1, hotkey="hk_old_king", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.5,
        ))
        assert state.king is not None and state.king.uid == 1

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=False,
        )

        assert state.king is None
        assert result.weights_set is False
        assert st.set_weights_calls == []

    def test_king_uid_out_of_range_clears_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        state = ValidatorState()
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=5, hotkey="hk_old_king", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.5,
        ))
        assert state.king is not None

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=True,
        )

        assert state.king is None
        assert result.weights_set is False

    def test_matching_hotkey_keeps_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk_king"], revealed={})
        state = ValidatorState()
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=1, hotkey="hk_king", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.5,
        ))

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=False,
        )

        assert state.king is not None and state.king.uid == 1
        assert result.weights_set is True
        assert len(st.set_weights_calls) == 1


class TestRunOnceWeightsFailure:
    def test_weights_rpc_failure_is_caught(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        st._set_weights_result = (False, "retry rate limit")  # always fail

        state = ValidatorState()
        state.record_evaluation(_make_eval_record(
            CommitmentRecord(uid=0, hotkey="hk0", commit_block=100,
                             model="m", revision="r", raw=""),
            score=0.5,
        ))

        result = run_once(
            subtensor=st, wallet=FAKE_WALLET, state=state,
            netuid=14, state_dir=tmp_path, dry_run=False,
            # tiny retry config so this test is fast
            chain_attempts=1, chain_delay_s=0,
        )
        assert result.weights_set is False
        assert result.weights_set_error is not None
