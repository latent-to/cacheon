#!/usr/bin/env python3
"""Replay the static runtime-quant screen over retained publication bytes.

Why this exists: before it, the only way to learn what a screen-policy change
does to the live queue was to spend a ~26 minute GPU qualification lease and
watch. That is an hours-long iteration loop for a question that is answerable
offline in seconds -- the publications are immutable bytes already on disk.

It uses the production loaders (``cacheon.manifest.load_manifest`` and
``cacheon.registry.eligibility_from_metadata``), the same two calls
``B300StaticScreenAdapter._quant_mismatch`` makes, so a verdict here is the
verdict the sealed screen would reach. It deliberately does NOT reimplement the
eligibility rule.

No target, slot, quant, pod path, or model identity is baked in: the
publication root and every requirement are arguments. Run it against any
retained corpus.

    scripts/replay_static_quant_screen.py \
        --publications /path/to/publications \
        --require moe.fused_experts=nvfp4 \
        --require moe.fused_experts_reduce=nvfp4

Exit status is 0 when the replay completes; it is a measurement, not a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from cacheon.manifest import ManifestError, load_manifest
from cacheon.registry import eligibility_from_metadata


def _requirement(text: str) -> tuple[str, str]:
    slot, _, quant = text.partition("=")
    if not slot or not quant:
        raise argparse.ArgumentTypeError(
            f"requirement must be SLOT=QUANT, got {text!r}"
        )
    return slot, quant


def _publication_roots(root: Path) -> list[Path]:
    """Every directory holding a manifest, at any depth of the shard layout."""

    return sorted(p.parent for p in root.glob("**/manifest.toml"))


def _declared_quants(bundle: Path, manifest, slot: str) -> set[str] | None:
    """Quants this bundle declares for ``slot``; None when it does not bind it."""

    variants = manifest.ops_for(slot)
    if not variants:
        return None
    declared: set[str] = set()
    for operation in variants:
        meta = None
        if operation.metadata is not None:
            raw = (bundle / operation.metadata).read_bytes()
            meta = json.loads(raw.decode("utf-8"))
        eligibility = eligibility_from_metadata(
            meta, operation.dtypes, operation.architectures
        )
        declared |= set(eligibility.quant)
    return declared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publications", required=True, type=Path)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        type=_requirement,
        metavar="SLOT=QUANT",
        help="repeatable; a bundle binding SLOT must declare QUANT",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print one line per bundle"
    )
    args = parser.parse_args(argv)

    if not args.publications.is_dir():
        print(f"no such publication root: {args.publications}", file=sys.stderr)
        return 2
    roots = _publication_roots(args.publications)
    if not roots:
        print(f"no manifests under {args.publications}", file=sys.stderr)
        return 2
    requirements = dict(args.require)

    slots = Counter()
    rejected: list[tuple[str, str, str, str]] = []
    unreadable: list[tuple[str, str]] = []
    bound = Counter()
    checked = 0

    for bundle in roots:
        try:
            manifest = load_manifest(bundle)
        except (ManifestError, OSError, ValueError) as exc:
            unreadable.append((bundle.name[:12], f"{type(exc).__name__}: {exc}"))
            continue
        checked += 1
        bundle_slots = {op.slot for op in manifest.ops}
        slots.update(bundle_slots)
        for slot, required in requirements.items():
            if slot not in bundle_slots:
                continue
            bound[slot] += 1
            try:
                declared = _declared_quants(bundle, manifest, slot)
            except (OSError, ValueError, KeyError) as exc:
                unreadable.append(
                    (bundle.name[:12], f"{slot}: {type(exc).__name__}: {exc}")
                )
                continue
            if declared is not None and required not in declared:
                rejected.append(
                    (bundle.name[:12], slot, required, ",".join(sorted(declared)) or "-")
                )
                if args.verbose:
                    print(f"REJECT {bundle.name[:12]} {slot} wants {required}")

    print(f"\npublications scanned : {len(roots)} ({checked} readable)")
    print("\nslot coverage of the retained corpus:")
    for slot, count in slots.most_common():
        share = 100.0 * count / max(checked, 1)
        mark = " <- screened" if slot in requirements else ""
        print(f"  {count:5d}  {share:5.1f}%  {slot}{mark}")

    print("\nrequirements:")
    if not requirements:
        print("  (none given -- coverage report only)")
    for slot, required in sorted(requirements.items()):
        hits = sum(1 for row in rejected if row[1] == slot)
        print(
            f"  {slot}={required}: binds {bound[slot]} bundle(s), "
            f"would REJECT {hits}"
        )

    if rejected:
        print(f"\nrejections ({len(rejected)}):")
        for digest, slot, required, declared in rejected:
            print(f"  {digest}  {slot}  needs {required}  declares [{declared}]")

    if unreadable:
        # Never silent: a bundle this tool could not read is a bundle whose
        # screen outcome this replay did NOT predict.
        print(f"\nUNREAD ({len(unreadable)}) -- outcome NOT predicted here:")
        for digest, why in unreadable[:20]:
            print(f"  {digest}  {why}")
        if len(unreadable) > 20:
            print(f"  ... and {len(unreadable) - 20} more")

    reach = 100.0 * len(rejected) / max(checked, 1)
    print(
        f"\nreplay would reject {len(rejected)} of {checked} readable bundles "
        f"({reach:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
