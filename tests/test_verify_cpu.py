"""Op-correctness test that needs torch but not a GPU.

Runs the full slot -> sandbox-load -> verify_entry path against the pure-torch
example bundle on CPU. Skipped automatically where torch is unavailable (e.g. the
dev laptop); runs on the VM.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.sandbox import load_entry  # noqa: E402
from optima.slots import get_slot  # noqa: E402
from optima.verify import verify_entry  # noqa: E402

from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent  # cwd-independent
TORCH_BUNDLE = str(_REPO / "examples/miner_silu_torch/kernels/silu_and_mul.py")
BROKEN_TORCH_BUNDLE = str(_REPO / "examples/miner_silu_broken_torch/kernels/silu_and_mul.py")


class _FakeGraphBackend:
    """CPU model of capture/replay orchestration (not CUDA semantics themselves)."""

    def __init__(self):
        self.phase = "eager"
        self.replay_index = -1

    def warmup(self, fn):
        self.phase = "warmup"
        fn()
        self.phase = "eager"

    def capture(self, fn):
        self.phase = "capture"
        fn()
        self.phase = "eager"
        return fn

    def replay(self, graph):
        self.replay_index += 1
        self.phase = "replay"
        graph()
        self.phase = "eager"

    def synchronize(self):
        pass


def _faithful_silu(x, out):
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def test_torch_silu_passes_correctness_cpu():
    entry = load_entry(TORCH_BUNDLE, "silu_and_mul")
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: max_abs={r.max_abs_err} {r.detail}" for r in result.shape_results
    )


def test_broken_torch_example_bundle_fails_cpu():
    # The committed adversarial bundle the miner guide's no-GPU walkthrough runs
    # (drops the SiLU). If this ever passes verify, the walkthrough demo is broken.
    entry = load_entry(BROKEN_TORCH_BUNDLE, "silu_and_mul")
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_wrong_kernel_fails_correctness_cpu():
    # A deliberately broken "kernel": forgets the multiply, just copies silu(gate).
    def broken(x, out):
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d]))

    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, broken, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_cpu_verify_reports_graph_proof_not_obtained():
    # Op slots execute under capture in serving, so graph proof is required by
    # default.  CPU verify remains a useful numerical PASS but must not claim that
    # capture/replay was verified.
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(
        slot, _faithful_silu, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}],
    )
    assert result.passed
    assert result.graph_required
    assert not result.graph_verified
    assert result.shape_results[0].graph_replays == 0


def test_graph_replay_orchestration_passes_all_replays_with_cpu_backend():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()
    result = verify_entry(
        slot, _faithful_silu, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert result.passed
    assert result.graph_verified
    assert result.shape_results[0].graph_replays == 3


def test_capture_only_wrong_branch_fails_graph_verification():
    # Models the real attack: eager/audit returns the reference, while a branch on
    # is_current_stream_capturing freezes a wrong kernel into the timed graph.
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def capture_branch(x, out):
        if backend.phase in {"capture", "replay"}:
            out.zero_()
        else:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, capture_branch, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed
    assert not result.graph_verified
    assert result.shape_results[0].graph_replays == 1
    assert "cuda graph replay[0]" in result.shape_results[0].detail


def test_output_poison_catches_graph_that_does_not_write():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def capture_noop(x, out):
        if backend.phase not in {"capture", "replay"}:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, capture_noop, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed
    assert "actual has non-finite values" in result.shape_results[0].detail


def test_later_graph_replay_corruption_is_not_hidden_by_first_replay():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def stateful_capture(x, out):
        if backend.phase == "replay" and backend.replay_index == 1:
            out.zero_()
        else:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, stateful_capture, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed
    assert result.shape_results[0].graph_replays == 2
    assert "cuda graph replay[1]" in result.shape_results[0].detail


def test_explicit_non_graph_safe_skips_graph_gate():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()
    result = verify_entry(
        slot, _faithful_silu, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=False,
        _graph_backend=backend,
    )
    assert result.passed
    assert not result.graph_required
    assert not result.graph_verified
    assert backend.replay_index == -1
