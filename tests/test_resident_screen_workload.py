"""The commissioned screen uses the scored cell lengths and measured mix."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import cacheon.eval.b300_resident_screen as screen
import cacheon.eval.b300_screen_deployment as deployment
from cacheon.arena_service import Workload, WorkloadCell
from cacheon.eval.oci_outer_session import OuterSessionTimeoutError
from cacheon.eval.resident_queue import ResidentScreenLoop
from tests.test_b300_screen_deployment import _case, _h
from tests.test_resident_queue import DIGEST_A, FakeSession, _candidate


def _mixed_inputs():
    cells = (
        WorkloadCell("short", 8192, 1024, 128, 2),
        WorkloadCell("long", 65536, 4096, 24, 3),
    )
    names = ("short", "short", "short", "long", "long", "long")
    batches = tuple(
        (f"batch-{i}",) * (128 if name == "short" else 24)
        for i, name in enumerate(names)
    )
    return SimpleNamespace(
        workload=Workload(_h("corpus"), "sealed", cells),
        prompt_batches=batches,
        prompt_batch_cells=names,
    )


class _ShapedSession(FakeSession):
    def __init__(self):
        super().__init__(100.0, {DIGEST_A: 120.0})
        self.session_id = "a" * 32
        self.shapes = []

    def execute_batch_with_shape(self, prompts, *, shape, **kwargs):
        self.shapes.append((prompts, shape))
        row = super().execute_batch(prompts, **kwargs)
        tokens = len(prompts) * shape.max_new_tokens
        rate = 100.0 if self.active is None else 120.0
        elapsed = tokens / rate
        self.clock = row.request_started_at + elapsed
        return replace(
            row, token_numerator=tokens, response_completed_at=self.clock,
        )


def test_mixed_screen_uses_all_timed_cells_and_preserves_swap_bracket():
    inputs = _mixed_inputs()
    reads = screen._screen_batches(inputs)
    assert [batch[0] for batch, _ in reads] == [f"batch-{i}" for i in range(1, 6)]
    session = _ShapedSession()
    wrapper = screen._WorkloadSession(session, reads)
    loop = ResidentScreenLoop(wrapper, prompts=reads[0][0])
    result = loop.screen(_candidate(DIGEST_A))
    assert result.passed
    assert result.verdict.speedup == pytest.approx(1.2)
    assert session.swaps == [DIGEST_A, None]
    expected = [(128, 8192, 1024)] * 2 + [(24, 65536, 4096)] * 3
    assert [(len(p), s.expected_prompt_tokens, s.max_new_tokens) for p, s in session.shapes] == expected * 4


def test_single_cell_screen_uses_that_profiles_lengths_and_concurrency():
    inputs = SimpleNamespace(
        workload=Workload(_h("other"), "sealed", (WorkloadCell("decode", 512, 128, 3, 2),)),
        prompt_batches=(("warm",) * 3, ("a",) * 3, ("b",) * 3),
        prompt_batch_cells=("decode",) * 3,
    )
    reads = screen._screen_batches(inputs)
    session = _ShapedSession()
    result = screen._WorkloadSession(session, reads).execute_batch((), canary=True)
    assert result.token_numerator == 768
    assert result.elapsed_seconds == pytest.approx(7.68)
    assert [p for p, _ in reads] == [("a",) * 3, ("b",) * 3]
    assert all(s.max_new_tokens == 128 and s.expected_prompt_tokens == 512 for _, s in reads)


def test_missing_timed_cell_is_rejected_before_opening_a_session():
    inputs = _mixed_inputs()
    inputs.prompt_batches = inputs.prompt_batches[:-1]
    inputs.prompt_batch_cells = inputs.prompt_batch_cells[:-1]
    with pytest.raises(deployment.B300ScreenDeploymentError, match="sealed timed batches"):
        screen._screen_batches(inputs)


def test_candidate_deadline_covers_the_whole_workload(monkeypatch):
    ticks = iter((0.0, 1.0, 2.0, 6.0))
    monkeypatch.setattr(screen.time, "monotonic", lambda: next(ticks))
    session = _ShapedSession()
    wrapper = screen._WorkloadSession(session, screen._screen_batches(_mixed_inputs()))
    with pytest.raises(OuterSessionTimeoutError, match="workload read"):
        wrapper.execute_batch((), canary=True, timeout_s=5.0)
    assert session.timeouts == [4.0, 3.0]


def test_commissioned_factory_wires_mixed_reads_and_long_context(tmp_path, monkeypatch):
    paths, gpus, _ = _case(tmp_path)
    inputs = deployment._authority_inputs(**paths, provisioner=None, provisioned_gpus=gpus)
    mixed = _mixed_inputs()
    inputs = replace(
        inputs, workload=mixed.workload, prompt_batches=mixed.prompt_batches,
        prompt_batch_cells=mixed.prompt_batch_cells,
    )
    captured = {}
    real_lane = screen.ResidentScreenLane

    def backend(_executor, _launch, _binding, _mount, plan, **kwargs):
        captured["plan"] = plan
        return lambda driver: driver(_ShapedSession())

    def lane(factory, **kwargs):
        captured["session"] = factory(lambda session: session)
        return real_lane(factory, **kwargs)

    monkeypatch.setattr(screen, "make_backend_lifetime_factory", backend)
    monkeypatch.setattr(screen, "ResidentScreenLane", lane)
    composition = deployment._compose(inputs)
    try:
        composition.authorities.resident_screen_factory.create()
        plan = captured["plan"]
        assert plan.engine_config.engine_kwargs["context_length"] == 65536 + 4096 + 128
        assert plan.engine_config.max_running_requests == 128
        assert plan.max_new_tokens == 1024
        row = captured["session"].execute_batch((), canary=True)
        assert row.token_numerator == 557056
        assert len(row.batches) == 5
    finally:
        composition.close()
