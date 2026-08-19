import pytest

from cacheon.seams import SEAM_ADAPTERS
from cacheon.target_catalog import (
    SINGLETON_TARGET_IDS,
    TargetResolutionError,
    default_target_catalog,
)
from cacheon.target_servability import (
    ADAPTER_SERVED_SLOTS,
    KNOWN_UNSERVABLE_TARGETS,
    unservable_target_reason,
    unservable_targets,
)


def test_msa_block_score_has_no_adapter_and_is_unservable():
    assert "attention.msa_block_score" not in ADAPTER_SERVED_SLOTS
    reason = unservable_target_reason("attention.msa_block_score")
    assert reason is not None and "no seam adapter" in reason


def test_norm_rmsnorm_is_known_unservable_despite_adapter():
    assert "norm.rmsnorm" in ADAPTER_SERVED_SLOTS
    reason = unservable_target_reason("norm.rmsnorm")
    assert reason == KNOWN_UNSERVABLE_TARGETS["norm.rmsnorm"]


def test_exactly_the_two_known_dead_targets_are_unservable():
    assert set(unservable_targets()) == {
        "attention.msa_block_score",
        "norm.rmsnorm",
    }


def test_every_other_registered_target_is_servable():
    dead = set(unservable_targets())
    catalog = default_target_catalog()
    for row in catalog.snapshot()["targets"]:
        target_id = row["target_id"]
        if target_id in dead:
            continue
        assert unservable_target_reason(target_id, catalog=catalog) is None


def test_known_unservable_keys_are_registered_catalog_targets():
    catalog = default_target_catalog()
    for target_id in KNOWN_UNSERVABLE_TARGETS:
        catalog.require(target_id)


def test_unknown_target_fails_like_every_catalog_lookup():
    with pytest.raises(TargetResolutionError):
        unservable_target_reason("no.such.target")


def test_every_adapter_slot_is_a_registered_singleton_target():
    # The reverse cross-check of the catalog/slots test: the seam table may
    # not serve slots the catalog does not register.
    assert ADAPTER_SERVED_SLOTS <= set(SINGLETON_TARGET_IDS)


def test_adapter_served_slots_derive_from_the_seam_table():
    expected = {slot for adapter in SEAM_ADAPTERS for slot in adapter.slots}
    assert ADAPTER_SERVED_SLOTS == frozenset(expected)


def test_atomic_epilogue_target_is_servable_via_member_slots():
    assert unservable_target_reason("collective.moe_epilogue.v1") is None
