"""Deterministic bundle identity (``cacheon/bundle_hash.py``).

``content_hash`` is what chain commitments bind to (``chain-package`` prints it,
intake re-hashes the fetched tree against it), so its stability and its
symlink-exclusion property are load-bearing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.bundle_hash import (
    CARRIER_RECEIPT_NAME,
    committed_content_hash,
    content_hash,
)

TRITON = "examples/miner_silu_triton"
BROKEN = "examples/miner_silu_broken"


def test_content_hash_stable_and_distinct():
    h1 = content_hash(TRITON)
    h2 = content_hash(TRITON)
    assert h1 == h2 and len(h1) == 64
    assert content_hash(BROKEN) != h1  # different bundle -> different hash


def test_content_hash_changes_with_content(tmp_path: Path):
    b = tmp_path / "b"
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text("bundle_id='x'\n")
    (b / "kernels" / "k.py").write_text("x = 1\n")
    h1 = content_hash(b)
    (b / "kernels" / "k.py").write_text("x = 2\n")
    assert content_hash(b) != h1


def test_content_hash_ignores_symlinks_so_identity_stays_in_bundle(tmp_path: Path):
    # A symlink must not fold an out-of-bundle file's bytes into the identity hash
    # (nor let the hash depend on the symlink target mutating). Two bundles with the
    # same real files hash identically regardless of a symlink's presence/target.
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n")
    b = tmp_path / "b"
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text("bundle_id='x'\n")
    (b / "kernels" / "k.py").write_text("x = 1\n")
    h_plain = content_hash(b)
    (b / "kernels" / "link.py").symlink_to(outside)
    assert content_hash(b) == h_plain  # symlink ignored
    outside.write_text("SECRET = 2\n")   # mutating the target does not move the hash
    assert content_hash(b) == h_plain


def test_committed_hash_of_receipt_bearing_carrier_matches_clean_bundle(
    tmp_path: Path,
):
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text("bundle_id='x'\n")
    (bundle / "kernels" / "k.py").write_text("x = 1\n")
    committed = content_hash(bundle)

    carrier = tmp_path / "carrier"
    (carrier / "kernels").mkdir(parents=True)
    (carrier / "manifest.toml").write_text("bundle_id='x'\n")
    (carrier / "kernels" / "k.py").write_text("x = 1\n")
    (carrier / CARRIER_RECEIPT_NAME).write_text('{"receipt":true}\n')

    assert committed_content_hash(carrier) == committed
    assert content_hash(carrier) != committed  # raw hash folds the receipt in


def test_committed_hash_excludes_only_the_top_level_receipt(tmp_path: Path):
    carrier = tmp_path / "carrier"
    (carrier / "kernels").mkdir(parents=True)
    (carrier / "manifest.toml").write_text("bundle_id='x'\n")
    (carrier / "kernels" / "k.py").write_text("x = 1\n")
    (carrier / CARRIER_RECEIPT_NAME).write_text('{"receipt":true}\n')
    before = committed_content_hash(carrier)

    nested = carrier / "kernels" / CARRIER_RECEIPT_NAME
    nested.write_text('{"miner":"content"}\n')
    assert committed_content_hash(carrier) != before  # nested name is miner bytes


def test_committed_hash_rejects_a_receipt_only_carrier(tmp_path: Path):
    carrier = tmp_path / "carrier"
    carrier.mkdir()
    (carrier / CARRIER_RECEIPT_NAME).write_text('{"receipt":true}\n')
    with pytest.raises(ValueError, match="no files to hash"):
        committed_content_hash(carrier)
