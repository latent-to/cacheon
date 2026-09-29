"""Byte/digest goldens for the durable families Refactor D must preserve.

Refactor D (see the private plan ledger) may compact in-memory
representations only if every durable byte and digest the live subnet
reopens stays identical. Unit tests that rebuild both sides cannot see a
refactor move the bytes; these goldens pin canonical bytes and digests to
synthetic inputs captured before any refactor commit, so a slice that
changes one byte fails here first.

The inputs are synthetic and structurally equivalent to production records;
the exact retained mainnet corpus stays private (it may carry hotkeys and
publication metadata) and is checked by the host-side parity modes, not by
this file. A mismatch here is a durable-format change, never a routine
refresh. Regenerate only for a reviewed schema epoch:

    python tests/test_refactor_d_goldens.py

prints the recomputed document; carry ``_meta`` forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from tests import test_refactor_d_golden_families as golden_families

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "refactor_d_goldens.json"


def _canonical(document: Any) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


# --------------------------------------------------------------------------- #
# Golden document: {family: {case: {"inputs", "canonical", "digest", "extras"}}}
# Builders return (canonical_bytes, digest, extras); extras hold the family's
# other pinned values (ids, sibling records, reopen outcomes).
# --------------------------------------------------------------------------- #

Builder = Callable[[dict[str, Any]], tuple[bytes, str, dict[str, Any]]]

_FAMILIES: dict[str, tuple[dict[str, dict[str, Any]], Builder]] = {
    **golden_families.FAMILIES,
}


def _capture() -> dict[str, dict[str, dict[str, Any]]]:
    document: dict[str, dict[str, dict[str, Any]]] = {}
    for family, (inputs, build) in _FAMILIES.items():
        document[family] = {}
        for case, case_inputs in inputs.items():
            raw, digest, extras = build(case_inputs)
            document[family][case] = {
                "inputs": case_inputs,
                "canonical": raw.decode("utf-8"),
                "digest": digest,
                "extras": extras,
            }
    return document


def _golden() -> dict[str, dict[str, dict[str, Any]]]:
    return json.loads(FIXTURE.read_text())["families"]


@pytest.mark.parametrize(
    ("family", "case"),
    [(family, case) for family, (inputs, _) in _FAMILIES.items() for case in inputs],
)
def test_golden_bytes_and_digest_are_unchanged(family: str, case: str) -> None:
    golden = _golden()
    assert family in golden, f"family without a pinned golden (pin it in this change): {family}"
    assert case in golden[family], f"case without a pinned golden (pin it in this change): {case}"
    pinned = golden[family][case]
    raw, digest, extras = _FAMILIES[family][1](pinned["inputs"])
    assert raw.decode("utf-8") == pinned["canonical"], f"{family}/{case}: canonical bytes moved"
    assert digest == pinned["digest"], f"{family}/{case}: digest moved"
    assert extras == pinned["extras"], f"{family}/{case}: pinned sibling values moved"


def test_golden_has_no_orphaned_cases() -> None:
    golden = _golden()
    live = {family: set(inputs) for family, (inputs, _) in _FAMILIES.items()}
    orphaned = sorted(
        f"{family}/{case}"
        for family, cases in golden.items()
        for case in cases
        if case not in live.get(family, set())
    )
    assert not orphaned, f"pinned cases no builder produces any more: {orphaned}"


if __name__ == "__main__":  # pragma: no cover - regeneration helper
    import subprocess
    import sys

    sha = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(
        json.dumps(
            {
                "_meta": {
                    "purpose": "Refactor D durable-format goldens (synthetic inputs)",
                    "generated_at_commit": sha,
                    "python": sys.version.split()[0],
                },
                "families": _capture(),
            },
            indent=2,
            sort_keys=True,
        )
    )
