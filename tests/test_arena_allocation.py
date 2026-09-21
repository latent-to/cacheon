"""Static settings are capped before acceptance, then frozen per submission."""

import pytest

from cacheon.arena_allocation import ArenaAllocation, normalize_weights
from cacheon.economics import ArenaRewardAuthority, project_global_rewards
from tests.test_economics import _catalog, _claim, _global_context, _policy, _stack


def _schedule(*versions):
    return ArenaAllocation.from_dict({
        "activation_block": 100, "burn_hotkey": "validator",
        "sources": {name: f"/configs/{name}.json" for name in ("a", "b", "c")},
        "history": [{"from_block": block, "weights_ppm": dict(zip(("a", "b", "c"), weights))}
                    for block, weights in ((0, (1_000_000, 0, 0)), *versions)],
    })


@pytest.mark.parametrize("raw,expected", [
    ({"a": 200_000, "b": 300_000}, {"a": 200_000, "b": 300_000}),
    ({"a": 600_000, "b": 400_000}, {"a": 600_000, "b": 400_000}),
    ({"c": 400_000, "b": 400_000, "a": 400_000},
     {"a": 333_334, "b": 333_333, "c": 333_333}),
    ({"a": 0, "b": 0}, {"a": 0, "b": 0}),
    ({f"arena{i:03}": 10_000 for i in range(200)},
     {f"arena{i:03}": 5_000 for i in range(200)}),
])
def test_normalize_only_above_one_hundred_percent(raw, expected):
    assert normalize_weights(raw) == expected
    assert normalize_weights(dict(reversed(tuple(raw.items())))) == expected


@pytest.mark.parametrize("weights", [{}, {"a": -1}, {"a": True}, {"a": 1.5}])
def test_invalid_weights_are_not_coerced(weights):
    with pytest.raises(ValueError):
        normalize_weights(weights)


def test_versions_preserve_old_terms_and_submission_boundary():
    old = _schedule((100, (600_000, 400_000, 0)))
    new = _schedule((100, (600_000, 400_000, 0)), (200, (900_000, 600_000, 0)),
                    (300, (200_000, 300_000, 0)))
    assert ArenaAllocation.from_dict(new.to_dict()) == new
    for block in (0, 99, 100, 199):
        assert new.terms_at(block) == old.terms_at(block)
    assert new.terms_at(200) == {"a": 600_000, "b": 400_000, "c": 0}
    assert new.terms_at(300) == {"a": 200_000, "b": 300_000, "c": 0}


def test_waiting_bonus_versions_keep_legacy_bytes_and_freeze_at_arrival():
    old = _schedule((100, (600_000, 400_000, 0)))
    raw = old.to_dict()
    assert "stall_bonus_ppm" not in raw["history"][1]
    assert ArenaAllocation.from_dict(raw).digest == old.digest
    raw["history"].append({"from_block": 200, "weights_ppm": raw["history"][1]["weights_ppm"],
                           "stall_bonus_ppm": 250_000})
    new = ArenaAllocation.from_dict(raw)
    assert new.to_dict() == raw
    assert new.terms_at(200) == old.terms_at(200)
    assert new.stall_bonus_at(199) == old.stall_bonus_at(200) == 1_000_000
    assert new.stall_bonus_at(200) == new.stall_bonus_at(900) == 250_000


@pytest.mark.parametrize("bonus", [-1, 1_000_001, True, 0.25])
def test_waiting_bonus_rejects_invalid_strength(bonus):
    raw = _schedule((100, (600_000, 400_000, 0))).to_dict()
    raw["history"][1]["stall_bonus_ppm"] = bonus
    with pytest.raises(ValueError, match="stall bonus"):
        ArenaAllocation.from_dict(raw)


def test_waiting_bonus_cannot_rewrite_preactivation_baseline():
    raw = _schedule((100, (600_000, 400_000, 0))).to_dict()
    raw["history"][0]["stall_bonus_ppm"] = 250_000
    with pytest.raises(ValueError, match="before activation"):
        ArenaAllocation.from_dict(raw)


def test_quarter_waiting_bonus_preserves_speed_decay_and_arena_pool():
    stack = _stack(_catalog())
    first = _claim(stack, "slot.a", "alice", 1_100_000, crowned_block=100)
    later = _claim(stack, "slot.b", "bob", 1_100_000, crowned_block=7300, evidence="5")
    full = later.credit_at(7300, _policy(), predecessor_block=100)
    quarter = later.credit_at(7300, _policy(), predecessor_block=100, stall_bonus_ppm=250_000)
    assert abs(full - 2 * quarter) <= 1  # One day: multiplier 3 becomes 1.5.
    assert abs(quarter - 2 * later.credit_at(7400, _policy(), predecessor_block=100,
                                          stall_bonus_ppm=250_000)) <= 1
    assert later.credit_at(7300, _policy(), stall_bonus_ppm=250_000) == later.credit_at(7300, _policy())
    terms = {c.digest: ("a", 600_000) for c in (first, later)}
    kwargs = dict(decay_start_blocks={c.digest: None for c in (first, later)},
                  allocation_terms=terms, allocation_burn_hotkey="validator")
    full = project_global_rewards(_policy(), _global_context(7300),
                                  (ArenaRewardAuthority(stack, 1, (first, later)),), (first, later), **kwargs)
    bonuses = {first.digest: 1_000_000, later.digest: 250_000}
    quarter = project_global_rewards(_policy(), _global_context(7300),
        (ArenaRewardAuthority(stack, 1, (first, later)),), (first, later), stall_bonus_terms=bonuses, **kwargs)
    assert quarter.weights_by_hotkey == {"alice": 240_000, "bob": 360_000, "validator": 400_000}
    assert full.weights_by_hotkey == {"alice": 150_000, "bob": 450_000, "validator": 400_000}
    assert next(c.credit for c in full.standing if c.claim_digest == first.digest) == next(
        c.credit for c in quarter.standing if c.claim_digest == first.digest)
    with pytest.raises(ValueError, match="complete static allocation"):
        project_global_rewards(_policy(), _global_context(7300),
            (ArenaRewardAuthority(stack, 1, (first, later)),), (first, later),
            stall_bonus_terms={later.digest: 250_000}, **kwargs)


def _project(schedule, submissions):
    authorities, claims, terms = [], [], {}
    for index, (source, hotkey, speedup, submitted) in enumerate(submissions):
        stack = _stack(_catalog(), ("slot.a",), arena="abcdef"[index])
        claim = _claim(stack, "slot.a", hotkey, speedup, crowned_block=submitted,
                       evidence="123456"[index])
        claims.append(claim)
        authorities.append(ArenaRewardAuthority(stack, 1, (claim,)))
        terms[claim.digest] = (source, schedule.terms_at(submitted)[source])
    return project_global_rewards(
        _policy(), _global_context(400), authorities, claims,
        decay_start_blocks={claim.digest: None for claim in claims},
        allocation_terms=terms, allocation_burn_hotkey="validator",
    )


def test_arena_percentages_ignore_other_arenas_credit_and_merge_hotkeys():
    schedule = _schedule((100, (600_000, 400_000, 0)))
    rows = [("a", "alice", 1_100_000, 100), ("b", "bob", 1_900_000, 100)]
    assert _project(schedule, rows).weights_by_hotkey == {"alice": 600_000, "bob": 400_000}
    rows[1] = ("b", "alice", 1_900_000, 100)
    assert _project(schedule, rows).weights_by_hotkey == {"alice": 1_000_000}


def test_setting_changes_alone_cannot_change_existing_rewards():
    old = _schedule((100, (600_000, 400_000, 0)))
    new = _schedule((100, (600_000, 400_000, 0)), (200, (100_000, 100_000, 800_000)))
    rows = [("a", "alice", 1_100_000, 100), ("b", "bob", 1_200_000, 100)]
    assert _project(old, rows) == _project(new, rows)
    rows.append(("c", "carol", 1_100_000, 200))
    assert _project(new, rows).weights_by_hotkey == {
        "alice": 333_333, "bob": 222_222, "carol": 444_445}


def test_new_arena_winner_dilutes_payouts_without_rewriting_old_terms():
    schedule = _schedule((100, (600_000, 400_000, 0)))
    rows = [("a", "alice", 1_100_000, 99), ("b", "bob", 1_100_000, 100)]
    assert schedule.terms_at(100) == {"a": 600_000, "b": 400_000, "c": 0}
    assert _project(schedule, rows[:1]).weights_by_hotkey == {"alice": 1_000_000}
    assert _project(schedule, rows).weights_by_hotkey == {"alice": 714_286, "bob": 285_714}
    assert schedule.terms_at(99) == {"a": 1_000_000, "b": 0, "c": 0}


@pytest.mark.parametrize("new_term", [600_000, 200_000])
def test_stall_bonus_redistributes_only_inside_its_arena_even_with_mixed_terms(new_term):
    schedule = _schedule((100, (600_000, 400_000, 0)), (200, (new_term, 400_000, 0)))
    a = _stack(_catalog(), ("slot.a", "slot.b"), arena="a")
    b = _stack(_catalog(), ("slot.a",), arena="b")

    def project(later_block):
        claims = (_claim(a, "slot.a", "alice", 1_100_000, crowned_block=100, evidence="1"),
                  _claim(a, "slot.b", "bob", 1_100_000, crowned_block=later_block, evidence="2"),
                  _claim(b, "slot.a", "carol", 1_100_000, crowned_block=100, evidence="3"))
        terms = {c.digest: (source, schedule.terms_at(c.crowned_block)[source])
                 for c, source in zip(claims, ("a", "a", "b"), strict=True)}
        return project_global_rewards(
            _policy(), _global_context(400),
            (ArenaRewardAuthority(a, 1, claims[:2]), ArenaRewardAuthority(b, 1, claims[2:])),
            claims, decay_start_blocks={c.digest: None for c in claims},
            allocation_terms=terms, allocation_burn_hotkey="validator")

    short, long = project(205), project(395)
    before, after = short.weights_by_hotkey, long.weights_by_hotkey
    assert after["bob"] > before["bob"] and after["alice"] < before["alice"]
    assert before["alice"] + before["bob"] == after["alice"] + after["bob"] == (600_000+new_term)//2
    assert before["carol"] == after["carol"] == 400_000
    assert before.get("validator", 0) == after.get("validator", 0) == 300_000-new_term//2


def test_unused_share_burns_and_old_zero_offer_does_not_dilute():
    schedule = _schedule((100, (200_000, 300_000, 0)))
    rows = [("a", "alice", 1_100_000, 100), ("b", "bob", 1_100_000, 100),
            ("b", "carol", 2_000_000, 99)]
    assert _project(schedule, rows).weights_by_hotkey == {
        "alice": 200_000, "bob": 300_000, "validator": 500_000}
    assert _project(schedule, rows[-1:]).weights_by_hotkey == {"validator": 1_000_000}


def test_ineligible_recipient_uses_existing_validator_attribution():
    schedule = _schedule((100, (600_000, 400_000, 0)))
    assert _project(schedule, [("a", "gone", 1_100_000, 100)]).weights_by_hotkey == {
        "validator": 1_000_000}


def test_uncrowned_retained_pass_can_earn_without_fabricating_crown():
    stack = _stack(_catalog(), ("slot.a",))
    claim = _claim(stack, "slot.a", "alice", 1_100_000)
    empty = _stack(_catalog(), ())
    projection = project_global_rewards(
        _policy(), _global_context(), (ArenaRewardAuthority(empty, 0, ()),), (claim,),
        allocation_terms={claim.digest: ("a", 600_000)}, allocation_burn_hotkey="validator",
    )
    assert projection.weights_by_hotkey == {"alice": 600_000, "validator": 400_000}
