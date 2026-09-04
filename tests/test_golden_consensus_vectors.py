"""Golden consensus vectors for the chain-commitment content hash.

``content_hash`` is what chain commitments bind to: ``chain-package`` prints
it and intake re-hashes the fetched tree against it. Its unit tests prove
stability within one process, but a test that recomputes both sides cannot
see the hash drifting across platforms or interpreters, nor a silent
algorithm change that moves every digest at once. These vectors pin the
digest of every committed example bundle and stack fixture to bytes captured
in advance and cross-checked on two platforms before commit (macOS arm64 /
CPython 3.11 and Linux x86_64 / CPython 3.12 on a two-B200 validator host,
2026-08-09; the fixture's ``_meta`` records both).

A mismatch here is a consensus break or a deliberate identity epoch — never
a routine fixture refresh. A tree missing from the golden means a new
committed bundle whose digest must be pinned in the same change. Regenerate
only for a reviewed epoch, cross-checking on two platforms again:

    python tests/test_golden_consensus_vectors.py

prints the recomputed ``content_hashes`` document; carry ``_meta`` forward
with the new capture platforms.
"""

from __future__ import annotations

import json
from pathlib import Path

from cacheon.bundle_hash import content_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "golden_consensus_vectors.json"

_STACK_FIXTURES = (
    "tests/fixtures/stack_norm_singleton",
)


def _tracked_trees() -> tuple[str, ...]:
    examples = sorted(
        f"examples/{entry.name}"
        for entry in (REPO_ROOT / "examples").iterdir()
        if entry.is_dir() and (entry / "manifest.toml").is_file()
    )
    return (*examples, *_STACK_FIXTURES)


def _capture() -> dict[str, str]:
    return {tree: content_hash(REPO_ROOT / tree) for tree in _tracked_trees()}


def test_content_hashes_match_the_cross_platform_golden() -> None:
    golden = json.loads(FIXTURE.read_text())["content_hashes"]
    captured = _capture()
    unpinned = sorted(set(captured) - set(golden))
    assert not unpinned, (
        f"trees without a pinned consensus vector (pin them in this change): {unpinned}"
    )
    removed = sorted(set(golden) - set(captured))
    assert not removed, (
        f"pinned trees missing from the checkout (retire their vectors deliberately): {removed}"
    )
    drifted = {
        tree: (golden[tree], captured[tree])
        for tree in golden
        if golden[tree] != captured[tree]
    }
    assert not drifted, (
        "content_hash diverged from the cross-platform golden — this is a "
        f"consensus break or an unreviewed identity epoch: {drifted}"
    )


if __name__ == "__main__":
    print(json.dumps({"content_hashes": _capture()}, indent=2, sort_keys=True))
