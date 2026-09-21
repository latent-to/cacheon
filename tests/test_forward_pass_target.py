"""The forward-pass target: a bundle names the modules of the served model it replaces."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon._strict import members_overlap
from cacheon.manifest import load_manifest
from cacheon.settlement import SettlementError, SettlementQualification
from cacheon.target_catalog import (
    SINGLETON_TARGET_IDS,
    TargetCatalog,
    TargetCatalogError,
    TargetResolutionError,
    default_target_catalog,
)
from tests.test_settlement import ROUTED, _audit_policy, _candidate, _ref, _stack
from tests.test_target_catalog import SILU, _bundle, _competition, _slot_spec


def _node_rows(*addresses: str) -> tuple[dict[str, object], ...]:
    return tuple({"slot": address} for address in addresses)


def _resolve(manifest):
    return default_target_catalog().resolve_manifest(manifest)


def test_a_node_bundle_resolves_to_the_forward_pass_with_its_own_addresses(tmp_path):
    addresses = ("model.layers.*.mlp", "logits_processor", "model.layers.*.attn")
    implicit = _resolve(load_manifest(_bundle(tmp_path / "a", rows=_node_rows(*addresses))))
    explicit = _resolve(load_manifest(_bundle(
        tmp_path / "b", rows=_node_rows(*addresses),
        competition=_competition("forward_pass", "slot"),
    )))
    assert implicit.implicit and not explicit.implicit
    for resolved in (implicit, explicit):
        assert (resolved.target_id, resolved.registered) == ("forward_pass", True)
        # The engine binds what the bundle named, so the reservation must carry that
        # and not the target's own single member.
        assert resolved.members == tuple(sorted(addresses))
    catalog = default_target_catalog()
    spec = catalog.require("forward_pass")
    assert catalog.admits(spec, implicit.members)
    assert not catalog.admits(spec, ("forward_pass",))
    assert catalog.admits(catalog.require(SILU), (SILU,))


@pytest.mark.parametrize(
    "addresses, message",
    [
        (("model.layers.*", "model.layers.3.mlp"), "overlap"),
        (("model", "model.norm"), "overlap"),
        (("lm_head",), "must sit under"),
        (("model.layers.**.mlp",), "must sit under"),
        (("model.layers.*.mlp", SILU), "must sit under"),
    ],
)
def test_a_node_bundle_outside_the_roots_or_claiming_a_node_twice_is_refused(
    tmp_path, addresses, message
):
    manifest = load_manifest(_bundle(
        tmp_path, rows=_node_rows(*addresses),
        competition=_competition("forward_pass", "slot"),
    ))
    with pytest.raises(TargetResolutionError, match=message):
        _resolve(manifest)


def test_without_node_roots_the_same_bundle_names_no_registered_target(tmp_path):
    manifest = load_manifest(_bundle(tmp_path, rows=_node_rows("model.layers.*.mlp")))
    closed = TargetCatalog([_slot_spec(SILU)])
    assert not closed.resolve_manifest(manifest).registered
    with pytest.raises(TargetCatalogError, match="node_roots"):
        TargetCatalog([replace(_slot_spec(SILU), node_roots=("model", "logits_processor"))])


def test_node_address_members_settle_and_a_malformed_member_does_not():
    catalog = default_target_catalog()
    primary = _candidate(_stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a").primary
    nodes = ("logits_processor", "model.layers.*.mlp")
    audit = _audit_policy("nodes", nodes)
    wide = replace(
        primary, members=nodes, audit_policy=audit, audit_control_digest=audit.control.digest
    )
    assert SettlementQualification.from_dict(wide.to_dict()) == wide
    with pytest.raises(SettlementError, match="member is not a canonical identifier"):
        replace(wide, members=("model.layers.**.mlp",))


def test_overlap_is_containment_for_nodes_and_equality_for_slot_ids():
    assert members_overlap(("model.layers.*",), ("model.layers.7.mlp.experts",))
    assert members_overlap(("model.layers.3.mlp",), ("model.layers.*.mlp",))
    assert not members_overlap(("model.layers.*.mlp",), ("model.layers.*.attn", "logits_processor"))
    # No slot id is a dotted prefix of another, so legacy reservations keep blocking
    # exactly the rows that set intersection blocked.
    for left in SINGLETON_TARGET_IDS:
        for right in SINGLETON_TARGET_IDS:
            assert members_overlap((left,), (right,)) == (left == right)
