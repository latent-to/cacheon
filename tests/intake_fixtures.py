"""Commission and quote a baseline for older store-level CPU scenarios."""

from dataclasses import replace

from cacheon.chain.declared_baseline import commission_baseline, current_baseline
from cacheon.stack_identity import sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog


def fixture_baseline():
    """Use a real catalog for intake-only tests that do not construct an arena."""

    catalog = default_target_catalog()
    return EvaluationStackManifest(
        runtime_digest=sha256_hex(b"runtime"), base_engine_digest=sha256_hex(b"base"),
        arena_digest=sha256_hex(b"arena"), catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest, entries={},
    )


def reserve_fixture(store, arrivals, **kwargs):
    """Supply the newly required miner declaration, then call real admission."""

    head = current_baseline(store)
    if head is None:
        stacks = store.evaluation_stacks()
        manifest = stacks[0].manifest if stacks else fixture_baseline()
        tree = stacks[0].tree_digest if stacks else sha256_hex(b"incumbent-tree")
        head = commission_baseline(store, manifest, tree)
    rows = tuple(replace(row, baseline_ref=head.manifest.digest) if not row.baseline_ref else row
                 for row in arrivals)
    return store.reserve_finalized(rows, **kwargs)


import io
import json
from functools import partial
import pytest
from cacheon.chain.submit import submit_bundle as _submit_bundle

submit_with_baseline = partial(_submit_bundle, baseline_ref=fixture_baseline().digest,
                               validator_url="https://validator.example")


@pytest.fixture(autouse=True)
def published_baseline(monkeypatch):
    """Serve a commissioned baseline without network access in miner tests."""
    from cacheon.chain import baseline_client
    monkeypatch.setattr(baseline_client, "urlopen", lambda *a, **k: io.BytesIO(
        json.dumps({"baseline_ref": fixture_baseline().digest}).encode()))


def commissioned_store(store, manifest, tree):
    """Initialize the known arena before a test submits work."""
    if current_baseline(store) is None:
        commission_baseline(store, manifest, tree)
    return store
