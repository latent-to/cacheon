"""Unit tests for validator.eval_progress and api/routes/eval_progress."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from validator.eval_progress import (
    PROGRESS_FILE,
    clear_progress,
    update_challenger_status,
    update_progress,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read(state_dir: Path) -> dict:
    path = state_dir / PROGRESS_FILE
    with open(path) as f:
        return json.load(f)


SAMPLE_CHALLENGERS = [
    {"uid": 10, "hotkey": "5Abc", "image": "docker.io/a:v1"},
    {"uid": 20, "hotkey": "5Def", "image": "docker.io/b:v2"},
]


# --------------------------------------------------------------------------- #
# update_progress
# --------------------------------------------------------------------------- #


def test_update_progress_creates_file(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    data = _read(tmp_path)
    assert data["phase"] == "challengers_found"
    assert data["round_block"] == 100
    assert data["status"] == "running"
    assert len(data["challengers"]) == 2
    assert all(c["status"] == "pending" for c in data["challengers"])
    assert data["challengers"][0]["uid"] == 10
    assert data["challengers"][1]["idx"] == 1
    assert len(data["steps"]) == 1
    assert data["steps"][0]["phase"] == "challengers_found"
    assert data["current_idx"] is None
    assert data["started_at"] > 0
    assert data["updated_at"] > 0


def test_update_progress_appends_steps(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_progress(tmp_path, phase="gpu_searching")
    update_progress(tmp_path, phase="gpu_ready", pod_id="wrk-123")
    data = _read(tmp_path)
    assert data["phase"] == "gpu_ready"
    assert len(data["steps"]) == 3
    assert data["steps"][2]["pod_id"] == "wrk-123"
    assert data["challengers"][0]["status"] == "pending"


def test_challengers_found_resets_steps(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_progress(tmp_path, phase="gpu_searching")
    assert len(_read(tmp_path)["steps"]) == 2

    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=200,
        challengers=[{"uid": 99, "hotkey": "5Xyz", "image": "img:v1"}],
    )
    data = _read(tmp_path)
    assert data["round_block"] == 200
    assert len(data["steps"]) == 1
    assert len(data["challengers"]) == 1


def test_update_progress_preserves_gpu(tmp_path):
    gpu = {"provider": "targon", "pod_id": "wrk-1", "cost_per_hr": 10.0}
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_progress(tmp_path, phase="gpu_ready", gpu=gpu)
    update_progress(tmp_path, phase="gpu_setup")
    data = _read(tmp_path)
    assert data["gpu"]["pod_id"] == "wrk-1"


# --------------------------------------------------------------------------- #
# update_challenger_status
# --------------------------------------------------------------------------- #


def test_update_challenger_status(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_challenger_status(tmp_path, 0, status="pulling", detail="pulling_image")
    data = _read(tmp_path)
    assert data["challengers"][0]["status"] == "pulling"
    assert data["challengers"][1]["status"] == "pending"
    assert data["current_idx"] == 0
    assert data["phase"] == "challenger_eval"

    update_challenger_status(
        tmp_path, 0, status="dq", score=0.0, dq_reason="pull_timeout"
    )
    data = _read(tmp_path)
    assert data["challengers"][0]["status"] == "dq"
    assert data["challengers"][0]["score"] == 0.0
    assert data["challengers"][0]["dq_reason"] == "pull_timeout"


def test_update_challenger_status_scored(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_challenger_status(tmp_path, 1, status="scored", score=0.85)
    data = _read(tmp_path)
    assert data["challengers"][1]["status"] == "scored"
    assert data["challengers"][1]["score"] == 0.85


def test_update_challenger_status_no_file(tmp_path):
    """Does not crash when progress file does not exist."""
    update_challenger_status(tmp_path, 0, status="pulling")
    assert not (tmp_path / PROGRESS_FILE).exists()


def test_update_challenger_status_bad_idx(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_challenger_status(tmp_path, 99, status="pulling")
    data = _read(tmp_path)
    assert all(c["status"] == "pending" for c in data["challengers"])


# --------------------------------------------------------------------------- #
# clear_progress
# --------------------------------------------------------------------------- #


def test_clear_progress_removes_file(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    assert (tmp_path / PROGRESS_FILE).exists()
    clear_progress(tmp_path)
    assert not (tmp_path / PROGRESS_FILE).exists()


def test_clear_progress_no_file(tmp_path):
    """Does not crash when progress file does not exist."""
    clear_progress(tmp_path)


# --------------------------------------------------------------------------- #
# Exception safety
# --------------------------------------------------------------------------- #


def test_update_progress_exception_does_not_propagate(tmp_path):
    with patch(
        "validator.state._atomic_write_json",
        side_effect=OSError("disk full"),
    ):
        update_progress(tmp_path, phase="gpu_searching")


def test_update_challenger_status_exception_does_not_propagate(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    with patch(
        "validator.state._atomic_write_json",
        side_effect=OSError("disk full"),
    ):
        update_challenger_status(tmp_path, 0, status="pulling")


# --------------------------------------------------------------------------- #
# API endpoint
# --------------------------------------------------------------------------- #


try:
    from api.server import app as _app  # noqa: F401

    _has_api_deps = True
except ImportError:
    _has_api_deps = False


@pytest.mark.skipif(not _has_api_deps, reason="API dependencies not installed")
class TestEvalProgressAPI:
    """Tests for GET /api/eval-progress using FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def _patch_state_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("api.config.STATE_DIR", tmp_path)
        from api.server import app

        from starlette.testclient import TestClient

        self.client = TestClient(app)
        self.state_dir = tmp_path

    def test_idle_when_no_file(self):
        resp = self.client.get("/api/eval-progress")
        assert resp.status_code == 200
        assert resp.json() == {"status": "idle"}

    def test_returns_data_when_running(self):
        update_progress(
            self.state_dir,
            phase="challengers_found",
            round_block=100,
            challengers=SAMPLE_CHALLENGERS,
        )
        resp = self.client.get("/api/eval-progress")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["phase"] == "challengers_found"
        assert len(body["challengers"]) == 2

    def test_stale_flag(self):
        update_progress(
            self.state_dir,
            phase="challengers_found",
            round_block=100,
            challengers=SAMPLE_CHALLENGERS,
        )
        path = self.state_dir / PROGRESS_FILE
        data = json.loads(path.read_text())
        data["updated_at"] = time.time() - 3600
        path.write_text(json.dumps(data))

        resp = self.client.get("/api/eval-progress")
        body = resp.json()
        assert body.get("possibly_stale") is True
