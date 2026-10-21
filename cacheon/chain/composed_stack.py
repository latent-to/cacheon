"""The tree identity of an evaluation stack the validator composes at settlement.

A crown joins the current stack whatever the winner's measurement-time manifest held, so when
another target crowned between the winner's measurement and its settlement, the adopted stack was
never assembled by a pod. The validator assembles it here from each measured contribution's
own publication, using the live catalog only when it matches the stack's sealed catalog.
This records composition identity, not a GPU qualification of the combined stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cacheon.engine_tree import EngineTreeError, materialize_engine_tree, reopen_materialized_engine_tree
from cacheon.settlement import SettlementCandidate
from cacheon.stack_manifest import EvaluationStackContext, EvaluationStackManifest
from cacheon.target_catalog import TargetCatalog, default_target_catalog


def composed_tree_digest(
    root: Path,
    manifest: EvaluationStackManifest,
    publications: Iterable[tuple[SettlementCandidate, Path]],
    catalog: TargetCatalog | None = None,
) -> str:
    """Assemble ``manifest`` under ``root/engine-trees`` and return its tree digest; a present tree is reopened."""

    catalog = default_target_catalog() if catalog is None else catalog
    if manifest.catalog_digest != catalog.digest:
        raise EngineTreeError("evaluation stack and live catalog differ")
    publication_roots = {
        candidate.candidate_manifest.entries[candidate.target_id].artifact_digest: path
        for candidate, path in publications
        if candidate.candidate_manifest is not None
        and candidate.candidate_manifest.entries[candidate.target_id] == manifest.entries.get(candidate.target_id)
    }
    missing = sorted(ref.artifact_digest for ref in manifest.entries.values()
                     if ref.artifact_digest not in publication_roots)
    if missing:
        raise EngineTreeError(f"composed stack entries lack a retained publication: {missing}")
    context = EvaluationStackContext(
        runtime_digest=manifest.runtime_digest, base_engine_digest=manifest.base_engine_digest,
        arena_digest=manifest.arena_digest, catalog_snapshot=manifest.catalog_snapshot,
        catalog_digest=manifest.catalog_digest,
        target_spec_digests={row["target_id"]: catalog.target_spec_digest(row["target_id"])
                             for row in catalog.snapshot()["targets"]},
    )
    trees = root / "engine-trees"
    trees.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = trees / f"composed-{manifest.digest}"
    if destination.exists():
        tree = reopen_materialized_engine_tree(destination)
    else:
        tree = materialize_engine_tree(
            manifest, context=context, catalog=catalog, destination=destination,
            resolver={("proposal", digest): path for digest, path in publication_roots.items()},
        )
    if tree.stack_digest != manifest.digest:
        raise EngineTreeError("composed tree differs from its stack")
    return tree.tree_digest


__all__ = ["composed_tree_digest"]
