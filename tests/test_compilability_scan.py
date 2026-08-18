"""Static detection of Triton kernels that cannot compile.

Regression scope: on 2026-08-17, 83 live mainnet bundles from 80 distinct
hotkeys called host-side ``triton.next_power_of_2`` inside a ``@triton.jit``
body. Triton compiles on first invocation, which on the evaluation path happens
only after a resident TP4 pair is loaded, so each one cost ~15 minutes of GPU
per attempt to reject. These cases are taken from those bundles.
"""

from cacheon.sandbox import scan_compilability


def test_inline_host_helper_in_jit_body_is_reported():
    # Verbatim shape from ar_residual_rmsnorm.py:303 in the live bundles.
    source = """
import triton
import triton.language as tl

@triton.jit
def _one_shot_ar_residual_rmsnorm_kernel(out_ptr, D: tl.constexpr):
    col_offsets = tl.arange(0, triton.next_power_of_2(D))
    mask = col_offsets < D
"""
    result = scan_compilability(source, filename="ar_residual_rmsnorm.py")
    assert not result.ok
    assert len(result.violations) == 1
    finding = result.violations[0]
    assert "ar_residual_rmsnorm.py:7" in finding
    assert "triton.next_power_of_2()" in finding
    assert "_one_shot_ar_residual_rmsnorm_kernel" in finding


def test_constexpr_assignment_form_is_reported():
    # The second observed form, from the _lamport_vec_ variant.
    source = """
import triton
import triton.language as tl

@triton.jit
def _lamport_vec_kernel(out_ptr, D: tl.constexpr):
    POW2: tl.constexpr = triton.next_power_of_2(D)
"""
    result = scan_compilability(source)
    assert not result.ok
    assert "triton.next_power_of_2()" in result.violations[0]


def test_host_side_call_outside_a_kernel_is_allowed():
    """The correct idiom must not be flagged.

    Computing the bound at the launch site and passing it in as its own
    constexpr is exactly the fix, and is what Triton's own fused-softmax
    tutorial does. Flagging it would make the gate useless.
    """
    source = """
import triton
import triton.language as tl

@triton.jit
def _kernel(out_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D

def launch(out, D):
    BLOCK_D = triton.next_power_of_2(D)
    _kernel[(1,)](out, D=D, BLOCK_D=BLOCK_D)
"""
    assert scan_compilability(source).ok


def test_in_language_builtins_are_allowed_inside_a_kernel():
    source = """
import triton
import triton.language as tl

@triton.jit
def _kernel(out_ptr, n: tl.constexpr):
    x = tl.cdiv(n, 8)
    y = tl.arange(0, 64)
"""
    assert scan_compilability(source).ok


def test_bare_jit_decorator_is_covered():
    """``from triton import jit`` must not be an escape hatch."""
    source = """
import triton
from triton import jit
import triton.language as tl

@jit
def _kernel(out_ptr, D: tl.constexpr):
    x = triton.cdiv(D, 8)
"""
    result = scan_compilability(source)
    assert not result.ok
    assert "triton.cdiv()" in result.violations[0]


def test_decorator_call_form_is_covered():
    source = """
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["n"])
def _kernel(out_ptr, n, D: tl.constexpr):
    x = triton.next_power_of_2(D)
"""
    assert not scan_compilability(source).ok


def test_every_offending_call_is_reported_not_just_the_first():
    source = """
import triton
import triton.language as tl

@triton.jit
def _a(out_ptr, D: tl.constexpr):
    x = triton.next_power_of_2(D)

@triton.jit
def _b(out_ptr, D: tl.constexpr):
    y = triton.next_power_of_2(D)
"""
    result = scan_compilability(source)
    assert len(result.violations) == 2


def test_syntax_error_is_reported_not_raised():
    result = scan_compilability("def broken(:\n", filename="bad.py")
    assert not result.ok
    assert "syntax error" in result.violations[0]


def test_clean_non_triton_source_passes():
    assert scan_compilability("def f(x):\n    return x + 1\n").ok


def test_compilability_is_independent_of_the_security_scan():
    """A broken kernel is not a hostile one; the two scans must not bleed.

    ``scan_source`` is a trust boundary and its findings mean the bundle is
    adversarial. ``scan_compilability`` only means it is broken. Conflating them
    would either excuse an attack or accuse an incompetent miner of one.
    """
    from cacheon.sandbox import scan_source

    broken_but_safe = """
import triton
import triton.language as tl

@triton.jit
def _kernel(out_ptr, D: tl.constexpr):
    x = triton.next_power_of_2(D)
"""
    assert scan_source(broken_but_safe).ok
    assert not scan_compilability(broken_but_safe).ok
