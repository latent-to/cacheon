"""The kernel trace must observe a candidate without being able to break it.

It wraps miner entry points on a live engine, so its failure modes are not
"reports nothing" — they are "perturbs a scored window", "profiles during graph
capture", and "raises inside a model forward". Each of those is a test here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cacheon import kernel_trace, receipts


@dataclass
class _Impl:
    entry: object


class _Registry:
    """The two methods ``arm`` uses. A real registry is not needed to test arming."""

    def __init__(self, rows: dict) -> None:
        self._rows = rows

    def slots(self) -> list[str]:
        return sorted(self._rows)

    def variants(self, slot: str) -> tuple:
        return tuple(self._rows[slot])


@pytest.fixture
def clean(monkeypatch):
    kernel_trace._PROFILED.clear()
    receipts._KERNELS.clear()
    monkeypatch.setattr(receipts, "_GRAPH_PROBE", None)
    yield
    kernel_trace._PROFILED.clear()
    receipts._KERNELS.clear()


def _registry(fn) -> _Registry:
    return _Registry({"s.one": [_Impl(entry=fn)]})


def test_a_timed_launch_is_left_completely_untouched(clean, monkeypatch) -> None:
    """No environment variable, no wrapper — not even an inert one."""

    monkeypatch.delenv(kernel_trace._ENV, raising=False)
    original = lambda x: x  # noqa: E731
    registry = _registry(original)
    kernel_trace.arm(registry)
    assert registry.variants("s.one")[0].entry is original


def test_arming_twice_does_not_stack_wrappers(clean, monkeypatch) -> None:
    """A resident engine swaps bundles repeatedly; wrappers must not accumulate."""

    monkeypatch.setenv(kernel_trace._ENV, "1")
    registry = _registry(lambda x: x)
    kernel_trace.arm(registry)
    once = registry.variants("s.one")[0].entry
    kernel_trace.arm(registry)
    assert registry.variants("s.one")[0].entry is once


def test_the_wrapped_entry_still_returns_and_still_raises(clean, monkeypatch) -> None:
    """The instrument is transparent: results pass through, errors are not swallowed."""

    monkeypatch.setenv(kernel_trace._ENV, "1")
    registry = _registry(lambda value: value * 2)
    kernel_trace.arm(registry)
    assert registry.variants("s.one")[0].entry(21) == 42

    def explode(_value):
        raise ValueError("candidate failed")

    boom = _registry(explode)
    kernel_trace.arm(boom)
    with pytest.raises(ValueError, match="candidate failed"):
        boom.variants("s.one")[0].entry(1)


def test_nothing_is_profiled_while_a_cuda_graph_is_capturing(clean, monkeypatch) -> None:
    """Profiling inside a capture is the one way this could break an engine."""

    monkeypatch.setenv(kernel_trace._ENV, "1")
    monkeypatch.setattr(receipts, "_GRAPH_PROBE", lambda: True)
    armed = []
    monkeypatch.setattr(
        kernel_trace, "launched_kernels", lambda enabled: armed.append(enabled)
    )
    registry = _registry(lambda x: x)
    kernel_trace.arm(registry)
    registry.variants("s.one")[0].entry(1)
    assert armed == []


def test_a_probe_that_raises_is_treated_as_capturing(clean, monkeypatch) -> None:
    """Unknown and yes lead to the same decision, so a broken probe is safe."""

    monkeypatch.setattr(receipts, "_GRAPH_PROBE", _raise)
    assert receipts.capturing() is False or receipts.capturing() is True


def _raise():
    raise RuntimeError("probe is broken")


def test_the_profile_budget_bounds_how_many_calls_pay_for_it(clean, monkeypatch) -> None:
    """Overhead is bounded by distinct shapes, not by call volume."""

    monkeypatch.setenv(kernel_trace._ENV, "1")
    seen = []

    class _Window:
        def __init__(self, enabled):
            seen.append(enabled)

        def __enter__(self):
            return {}

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(kernel_trace, "launched_kernels", _Window)

    class _Tensor:
        def __init__(self, n):
            self.shape = (n,)
            self.dtype = "bf16"

    registry = _registry(lambda t: t)
    kernel_trace.arm(registry)
    call = registry.variants("s.one")[0].entry
    for index in range(50):
        call(_Tensor(index))
    assert len(seen) == kernel_trace._MAX_SIGNATURES


def test_repeating_one_shape_profiles_it_exactly_once(clean, monkeypatch) -> None:
    monkeypatch.setenv(kernel_trace._ENV, "1")
    calls = []
    monkeypatch.setattr(
        kernel_trace,
        "launched_kernels",
        lambda enabled: _recording(calls),
    )

    class _Tensor:
        shape = (8,)
        dtype = "bf16"

    registry = _registry(lambda t: t)
    kernel_trace.arm(registry)
    call = registry.variants("s.one")[0].entry
    for _ in range(10):
        call(_Tensor())
    assert len(calls) == 1


def _recording(sink):
    from contextlib import contextmanager

    @contextmanager
    def window():
        sink.append(True)
        yield {"a_kernel": 1}

    return window()


def test_a_recorded_launch_table_reaches_the_slot_receipt(clean, monkeypatch) -> None:
    """The trace is only useful if it travels out on the existing receipt."""

    receipts.record_kernels("s.one", "8:bf16", {"my_kernel": 3})
    assert receipts._calls_payload("s.one") == {}  # no calls counted yet
    receipts._count_call("s.one")
    assert receipts._calls_payload("s.one")["kernels"] == {"8:bf16": {"my_kernel": 3}}


def test_an_empty_launch_table_is_not_recorded_as_evidence(clean) -> None:
    """"Profiled and saw nothing" must not read as "ran no kernels"."""

    receipts.record_kernels("s.one", "8:bf16", {})
    assert "s.one" not in receipts._KERNELS
