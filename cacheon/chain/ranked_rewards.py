"""Project unfinalized winners into reward authority without changing evaluation HEAD."""

from cacheon.stack_manifest import EvaluationStackManifest

from cacheon.chain.submission_ranking import current_winners
from cacheon.economics import ArenaRewardAuthority


def reward_authorities(store, states, earning):
    """Use each slot's strongest winner as the standing reward claim immediately."""

    candidates = {}
    for row in current_winners(store):
        retained = store._db.execute(
            "SELECT * FROM settlement_candidates WHERE reservation_id=?", (row["reservation_id"],)
        ).fetchone()
        candidate = store._settlement_candidate(retained)
        candidates[row["arena_id"], row["target_id"]] = candidate
    claims = {(row.arena_digest, row.target_id, row.contribution_digest): row for row in earning}
    authorities = []
    for state in states:
        # Generation zero can import an earlier arena's commissioned baseline.
        # Those entries are evaluation inputs, not reward claims in this arena.
        entries = dict(state.manifest.entries) if state.generation > 0 else {}
        for (arena, target), candidate in candidates.items():
            if arena == state.arena_digest:
                entries[target] = candidate.candidate_manifest.entries[target]
        if not entries:
            continue
        stack = EvaluationStackManifest(
            runtime_digest=state.manifest.runtime_digest, base_engine_digest=state.manifest.base_engine_digest,
            arena_digest=state.arena_digest, catalog_digest=state.manifest.catalog_digest,
            catalog_snapshot=state.manifest.catalog_snapshot, entries=entries,
        )
        standing = tuple(claims[state.arena_digest, target, contribution.digest]
                         for target, contribution in entries.items())
        authorities.append(ArenaRewardAuthority(stack, state.generation, standing))
    return authorities
