"""Unit tests for validator.cpu_validator -- the CPU always-on loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from validator.chain import CommitmentRecord
from validator.cpu_validator import (
    WEIGHTS_REFRESH_BLOCKS,
    _needs_weight_set,
    run_tick,
)
from validator.eval_schema import EVAL_JOB_FILE, EvalJob
from validator.state import EvaluationRecord, ValidatorState

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes (same pattern as test_validator_loop.py)
# --------------------------------------------------------------------------- #


@dataclass
class FakeMetagraph:
    hotkeys: list[str]


class FakeSubtensor:
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
        self.substrate = self
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
_DIGEST = "sha256:" + "a" * 64


def _commit_json(image: str = "user/repo:v1", digest: str = _DIGEST) -> str:
    return json.dumps({"image": image, "digest": digest})


def _make_eval_record(
    uid: int,
    hotkey: str,
    commit_block: int,
    score: float,
    eval_block: int = 500,
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=uid,
        hotkey=hotkey,
        commit_block=commit_block,
        image="user/repo:v1",
        digest=_DIGEST,
        score=score,
        ttft_improvement=score * 0.5,
        throughput_improvement=score * 0.5,
        token_match_rate=0.99,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=1700000000.0,
        evaluation_block=eval_block,
    )


def _noop(*_a, **_k):
    return 0


# --------------------------------------------------------------------------- #
# _needs_weight_set
# --------------------------------------------------------------------------- #


class TestNeedsWeightSet:
    def test_no_king(self):
        state = ValidatorState()
        assert _needs_weight_set(state, 1000) is None

    def test_first_weight_set(self):
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=500)
        state.record_evaluation(rec, current_block=500)
        assert state.king is not None
        assert state.last_weights_set_block == 0
        reason = _needs_weight_set(state, 1000)
        assert reason is not None
        assert "first" in reason

    def test_new_evals_detected(self):
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=800)
        state.record_evaluation(rec, current_block=800)
        state.last_weights_set_block = 500
        reason = _needs_weight_set(state, 1000)
        assert reason is not None
        assert "new evals" in reason

    def test_no_new_evals_not_stale(self):
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=400)
        state.record_evaluation(rec, current_block=400)
        state.last_weights_set_block = 500
        reason = _needs_weight_set(state, 600)
        assert reason is None

    def test_stale_weights_triggers_refresh(self):
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=400)
        state.record_evaluation(rec, current_block=400)
        state.last_weights_set_block = 500
        stale_block = 500 + WEIGHTS_REFRESH_BLOCKS + 1
        reason = _needs_weight_set(state, stale_block)
        assert reason is not None
        assert "stale" in reason


# --------------------------------------------------------------------------- #
# run_tick -- no challengers
# --------------------------------------------------------------------------- #


class TestRunTickNoChallengers:
    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_empty_chain_no_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1"], revealed={})
        state = ValidatorState()
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["commitments"] == 0
        assert summary["challengers"] == 0
        assert summary["weights_set"] is False
        assert summary["king_uid"] is None
        assert state.last_scan_block == 1000

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_king_with_new_evals_sets_weights_dry_run(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=800)
        state.record_evaluation(rec, current_block=800)
        state.last_weights_set_block = 0
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["weights_set"] is True
        assert summary["king_uid"] == 0
        assert state.last_weights_set_block == 1000
        assert st.set_weights_calls == []

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_king_with_new_evals_sets_weights_live(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk1"], revealed={})
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=800)
        state.record_evaluation(rec, current_block=800)
        state.last_weights_set_block = 0
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=False,
        )

        assert summary["weights_set"] is True
        assert len(st.set_weights_calls) == 1
        call = st.set_weights_calls[0]
        assert call["weights"] == [1.0, 0.0]
        assert call["uids"] == [0, 1]

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_no_weight_set_when_already_current(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={}, current_block=700)
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=400)
        state.record_evaluation(rec, current_block=400)
        state.last_weights_set_block = 500
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=False,
        )

        assert summary["weights_set"] is False
        assert st.set_weights_calls == []


# --------------------------------------------------------------------------- #
# run_tick -- with challengers
# --------------------------------------------------------------------------- #


class TestRunTickWithChallengers:
    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_writes_eval_job(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1"],
            revealed={"hk1": [(200, _commit_json("user/m:v1", "sha256:" + "b" * 64))]},
        )
        state = ValidatorState()
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["challengers"] == 1
        job = EvalJob.load(str(tmp_path))
        assert job is not None
        assert len(job.challengers) == 1
        assert job.challengers[0].uid == 1
        assert job.challengers[0].hotkey == "hk1"
        assert job.block == 1000
        assert job.block_hash == "0xdeadbeef"

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_already_evaluated_not_in_eval_job(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json())]},
        )
        state = ValidatorState()
        rec = _make_eval_record(0, "hk0", 100, 0.5, eval_block=500)
        state.record_evaluation(rec, current_block=500)
        state.last_weights_set_block = 600
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["challengers"] == 0
        assert not (tmp_path / EVAL_JOB_FILE).exists()

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_no_eval_job_when_block_hash_none(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0"],
            revealed={"hk0": [(100, _commit_json())]},
            block_hash=None,
        )
        state = ValidatorState()
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["challengers"] == 1
        assert not (tmp_path / EVAL_JOB_FILE).exists()

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_multiple_challengers_all_in_eval_job(self, tmp_path):
        st = FakeSubtensor(
            hotkeys=["hk0", "hk1", "hk2"],
            revealed={
                "hk0": [(100, _commit_json("u/a:v1", "sha256:" + "a" * 64))],
                "hk2": [(300, _commit_json("u/c:v1", "sha256:" + "c" * 64))],
            },
        )
        state = ValidatorState()
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["challengers"] == 2
        job = EvalJob.load(str(tmp_path))
        assert job is not None
        assert len(job.challengers) == 2
        uids = {c.uid for c in job.challengers}
        assert uids == {0, 2}


# --------------------------------------------------------------------------- #
# run_tick -- king UID recycling
# --------------------------------------------------------------------------- #


class TestRunTickKingRecycled:
    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_recycled_hotkey_clears_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk_new"], revealed={})
        state = ValidatorState()
        rec = _make_eval_record(1, "hk_old_king", 100, 0.5, eval_block=500)
        state.record_evaluation(rec, current_block=500)
        state.save(tmp_path)
        assert state.king is not None

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["king_uid"] is None
        assert summary["weights_set"] is False
        assert state.king is None

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_king_uid_out_of_range_clears_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        state = ValidatorState()
        rec = _make_eval_record(5, "hk_far", 100, 0.5, eval_block=500)
        state.record_evaluation(rec, current_block=500)
        state.save(tmp_path)
        assert state.king is not None

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["king_uid"] is None
        assert state.king is None

    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", _noop)
    def test_matching_hotkey_keeps_king(self, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0", "hk_king"], revealed={})
        state = ValidatorState()
        rec = _make_eval_record(1, "hk_king", 100, 0.5, eval_block=500)
        state.record_evaluation(rec, current_block=500)
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["king_uid"] == 1
        assert state.king is not None
        assert state.king.hotkey == "hk_king"


# --------------------------------------------------------------------------- #
# run_tick -- S3 download failure is non-fatal
# --------------------------------------------------------------------------- #


class TestRunTickS3Failure:
    @patch("validator.cpu_validator._try_upload", _noop)
    @patch("validator.sync.download", side_effect=RuntimeError("S3 down"))
    def test_continues_on_s3_download_failure(self, _mock_dl, tmp_path):
        st = FakeSubtensor(hotkeys=["hk0"], revealed={})
        state = ValidatorState()
        state.save(tmp_path)

        summary = run_tick(
            subtensor=st,
            wallet=FAKE_WALLET,
            state=state,
            netuid=14,
            state_dir=str(tmp_path),
            dry_run=True,
        )

        assert summary["block"] == 1000
