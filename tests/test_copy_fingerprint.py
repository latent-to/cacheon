"""Near-copy fingerprint + cumulative cross-round copy detection.

Pins the two confirmed gaps: (1) a reformatted/recommented copy that flips the
exact content hash is still caught, and (2) copy detection now spans rounds, so a
copy revealed in a LATER round than the original is no longer mislabeled original.
"""

from pathlib import Path

from optima.commit_reveal import Ledger, make_commitment
from optima.copy_fingerprint import bundle_fingerprint, normalized_source, source_fingerprint

ORIG = '''\
"""A kernel docstring."""
import torch

def silu_and_mul(x, out):
    # compute the gate
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])
'''

# Same logic, reflowed: different whitespace, different comments, no docstring,
# extra blank lines and parens. A byte hash differs; the structure does not.
REFORMATTED = '''\
import torch


def silu_and_mul(x, out):
    d = (x.shape[-1] // 2)
    # totally different comment wording here
    out.copy_((torch.nn.functional.silu(x[..., :d]) * x[..., d:]))
'''

# A genuine logic change (drops the silu) -> must NOT collide with ORIG.
DIFFERENT = '''\
import torch

def silu_and_mul(x, out):
    d = x.shape[-1] // 2
    out.copy_(x[..., :d] * x[..., d:])
'''


def test_reformat_recomment_redocstring_fingerprints_identical():
    assert source_fingerprint(ORIG) == source_fingerprint(REFORMATTED)


def test_genuine_logic_change_fingerprints_differently():
    assert source_fingerprint(ORIG) != source_fingerprint(DIFFERENT)


def test_normalized_source_strips_docstring_and_comments():
    n = normalized_source(ORIG)
    assert "A kernel docstring" not in n
    assert "compute the gate" not in n  # comments gone


def test_bundle_fingerprint_on_a_real_example_is_stable_nonempty():
    bundle = Path(__file__).resolve().parent.parent / "examples" / "miner_silu_triton"
    fp = bundle_fingerprint(bundle)
    assert fp and len(fp) == 64
    assert bundle_fingerprint(bundle) == fp  # deterministic


def _commit_reveal(led: Ledger, hotkey: str, ch: str, salt: str, rnd: int, fp: str):
    led.commit(hotkey, make_commitment(ch, hotkey, salt), rnd)
    return led.reveal(hotkey, ch, salt, rnd, fingerprint=fp)


def test_near_copy_in_a_later_round_is_flagged():
    led = Ledger()
    F = "fingerprint-A"
    a = _commit_reveal(led, "alice", "HASH_ORIG", "s", 0, F)
    # bob reflows alice's kernel: NEW exact hash, SAME fingerprint, LATER round.
    b = _commit_reveal(led, "bob", "HASH_REFLOW", "s", 1, F)
    assert a.original is True
    assert b.original is False  # caught as a near-copy across rounds


def test_exact_copy_in_a_later_round_is_flagged():
    led = Ledger()
    a = _commit_reveal(led, "alice", "HASH_X", "s", 0, "fp1")
    c = _commit_reveal(led, "carol", "HASH_X", "s", 2, "fp1")
    assert a.original is True
    assert c.original is False  # cross-round exact copy now caught (was a gap)


def test_same_hotkey_resubmitting_own_work_is_not_a_copy():
    led = Ledger()
    a0 = _commit_reveal(led, "alice", "HASH_X", "s0", 0, "fpA")
    a1 = _commit_reveal(led, "alice", "HASH_X", "s1", 1, "fpA")
    assert a0.original is True
    assert a1.original is True  # you can't plagiarize yourself


def test_independent_distinct_kernels_both_original():
    led = Ledger()
    a = _commit_reveal(led, "alice", "HASH_A", "s", 0, "fpA")
    b = _commit_reveal(led, "bob", "HASH_B", "s", 0, "fpB")
    assert a.original and b.original
