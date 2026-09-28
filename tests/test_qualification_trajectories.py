"""The quality-control selector joins its sources through the retained first window."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.agent_replay import AgentReplayPlan
from cacheon.eval.oci_outer_session import BatchExecutionEvidence
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.qualification_trajectories import QualificationError, _source_rows, prompt_pool, source_digest
from cacheon.eval.service_capacity import ServiceContract
from tests.test_agent_slice import _session, _write_slice


def _export(directory: Path, session_id: str, request_id: str) -> None:
    export = directory / "window1" / "aiperf"
    export.mkdir(parents=True)
    (export / "profile_export.jsonl").write_text(json.dumps({"metadata": {
        "x_request_id": request_id, "source_trace_id": session_id, "source_outer_idx": 0,
        "source_inner_idx": None, "benchmark_phase": "profiling",
    }}) + "\n")


def test_source_rows_join_the_first_retained_window_to_the_host_requests(tmp_path):
    session = _session(1, k=1, inner=0)
    plan = AgentReplayPlan(_write_slice(tmp_path, [session]), (1,), Path("/bin/aiperf"), Path("/model"),
                           tmp_path / "out", ServiceContract(20, 2, .8), "incumbent", 1, "first", windows=3)
    # The slice fixture seals an empty trace; the pool reads the recorded turn and its output budget from it.
    (plan.slice.directory / f"000_{session['id']}.json").write_text(json.dumps(
        {"id": session["id"], "models": ["m"], "block_size": 64, "hash_id_scope": "local", "requests": [{"out": 12}]}))
    request_id = f"{7:032x}"
    row = BatchExecutionEvidence(0, request_id, "a" * 32, 1.1, 1.5, 4,
                                 BatchEvidence((PromptEvidence((1, 2, 3, 4), ((),) * 4, 64),)), (), ((.1, .39),))
    _export(plan.output_directory, session["id"], request_id)
    rows = _source_rows(plan, plan.output_directory, [row])
    assert rows == {source_digest(plan, session["id"], 0, None): row}
    assert set(rows) == set(prompt_pool(plan))
    # A window layout that retains no first window is not a source the selector can use.
    with pytest.raises(FileNotFoundError):
        _source_rows(plan, tmp_path / "elsewhere", [row])
    # A request the host never made cannot be joined to a source turn.
    with pytest.raises(QualificationError, match="not one-to-one"):
        _source_rows(plan, plan.output_directory, [SimpleNamespace(request_id="b" * 32)])
