"""The single offer producer's locked, multi-store static allocation path."""

from contextlib import ExitStack
import json
from pathlib import Path

from cacheon.arena_allocation import ArenaAllocation, allocate_submission_weights, arena_base_credits
from cacheon.chain.intake import IntakeError, is_lock_collision
from cacheon.chain.mainnet_screen_dispatcher import load_config
from cacheon.chain.qualification_settlement import (
    _reward_projection_inputs, reconcile_follower_reward_decay,
)
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.remote_worker_spool import load_json
from cacheon.chain.sealed_config import authority_file
from cacheon.chain.weights import WeightProjection
from cacheon.economics import project_global_rewards
from cacheon.eval.evidence_store import (
    prepare_evidence_root, publish_canonical_json_evidence, reopen_evidence,
)
from cacheon.stack_identity import canonical_digest

_HISTORY = "static_arena_allocation"


def load_allocation(path: Path) -> ArenaAllocation:
    """Reload one complete owner-controlled settings file for each projection."""
    authority_file(path, "arena allocation", error=IntakeError)
    return ArenaAllocation.from_dict(load_json(path))


def require_legacy_projection(store, current_block: int) -> None:
    """Do not let removal of an activated configuration silently drop other stores."""
    row = store._db.execute("SELECT value FROM metadata WHERE key=?", (_HISTORY,)).fetchone()
    if row and current_block >= json.loads(row[0])["schedule"]["activation_block"]:
        raise IntakeError("activated arena allocation requires its configured producer")


def _bind_schedule(store, allocation, source_digests, block):
    row = store._db.execute("SELECT value FROM metadata WHERE key=?", (_HISTORY,)).fetchone()
    if row:
        previous = json.loads(row[0])
        old = ArenaAllocation.from_dict(previous["schedule"])
        if (allocation.activation_block != old.activation_block
                or allocation.burn_hotkey != old.burn_hotkey or allocation.sources != old.sources
                or allocation.history[:len(old.history)] != old.history
                or source_digests != previous["source_digests"]):
            raise IntakeError("accepted allocation history or source authority changed")
        boundary = max(block, previous["seen_through"])
        if any(row[0] <= boundary for row in allocation.history[len(old.history):]):
            raise IntakeError("allocation updates must precede their future submission boundary")
    else:
        if block >= allocation.activation_block:
            raise IntakeError("register allocation before its activation block")
        boundary = block
    store._db.execute(
        "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_HISTORY, json.dumps({"schedule": allocation.to_dict(), "seen_through": boundary,
                              "source_digests": source_digests}, sort_keys=True)),
    )


def build_static_projection(primary, *, allocation, policy, context, netuid,
                            confirmation_journal, max_lag_blocks):
    """Reopen all configured authorities before one offer; never sign or push here."""
    configs = {key: load_config(path) for key, path in allocation.sources}
    paths = [config.intake_db.resolve() for config in configs.values()]
    matches = [key for key, config in configs.items() if config.intake_db.resolve() == primary.path]
    if len(set(paths)) != len(paths) or len(matches) != 1:
        raise IntakeError("allocation sources must name distinct stores including the producer")
    primary_key = matches[0]
    if any(config.scope != primary.scope for config in configs.values()):
        raise IntakeError("allocation stores differ in chain scope")
    if context.chain_scope_digest != primary.scope.digest or netuid != primary.scope.netuid:
        raise IntakeError("allocation context differs from its source scope")
    if allocation.terms_at(0) != {key: 1_000_000 if key == primary_key else 0 for key in configs}:
        raise IntakeError("baseline allocation must preserve the existing primary rewards")
    if confirmation_journal is None:
        raise IntakeError("combined rewards require the single confirmation journal")
    with ExitStack() as lifetime:
        stores = {primary_key: primary}
        active = context.current_block >= allocation.activation_block
        if active:
            for key, config in sorted(configs.items()):
                if not config.intake_db.is_file():
                    raise IntakeError("configured reward store is absent")
                if key != primary_key:
                    try:
                        stores[key] = lifetime.enter_context(RecoverableFinalizedIntakeStore(
                            config.intake_db, config.policy, scope=config.scope))
                    except IntakeError as exc:
                        if is_lock_collision(exc):
                            from cacheon.chain.weight_offer_service import WeightOfferBusyError

                            raise WeightOfferBusyError(str(exc)) from exc
                        raise
        for store in stores.values():
            lifetime.enter_context(store._transaction())
        boundary = max([context.current_block] + [
            cursor[0] for store in stores.values() if (cursor := store.finalized_cursor())])
        _bind_schedule(primary, allocation, {key: config.digest for key, config in configs.items()}, boundary)
        if not active:
            reconcile_follower_reward_decay(primary, confirmation_journal,
                                           validator_hotkey=context.validator_hotkey)
            return primary.build_weight_projection(policy=policy, context=context, netuid=netuid)

        combined = {key: [] for key in ("arenas", "earning_claims", "discovery_claims", "earned_contributions")}
        combined["decay_start_blocks"] = {}
        terms, snapshots, adjustments, seen_reservations = {}, {}, {}, set()
        bonuses = {}
        evidence, standing_count, generation = set(), 0, 0
        cursor_hashes = {context.current_block: context.current_block_hash}
        for key, store in sorted(stores.items()):
            cursor = store.finalized_cursor()
            if (cursor is None or not 0 <= context.current_block - cursor[0] <= max_lag_blocks
                    or cursor_hashes.setdefault(cursor[0], cursor[1]) != cursor[1]):
                raise IntakeError("reward store snapshot is absent, stale, ahead, or inconsistent")
            # A second chain listener can reject the same arrival before admission;
            # that payment failure carries no publication or evaluation ownership.
            # Every arena's listener also admits every paid reveal and leaves the
            # other arena's rows unevaluated (2026-09-21: four of the six Qwen rows
            # were also in the GLM store). Only a PASS can earn, so only a PASS owns.
            ids = {row[0] for row in store._db.execute(
                "SELECT reservation_id FROM reservations WHERE decision='PASS'")}
            if seen_reservations.intersection(ids):
                raise IntakeError("reservation passed in multiple reward stores")
            seen_reservations.update(ids)
            reconcile_follower_reward_decay(store, confirmation_journal,
                                           validator_hotkey=context.validator_hotkey)
            inputs = _reward_projection_inputs(store, include_uncrowned=True)
            adjustments[key] = inputs.pop("adjustments")
            standing = inputs.pop("standing_claims")
            states = inputs.pop("states")
            standing_count += len(standing)
            generation = max(generation, max((state.generation for state in states), default=0))
            for claim in inputs["earning_claims"]:
                if claim.digest in terms or claim.crowned_block > cursor[0]:
                    raise IntakeError("reward claim is duplicated or ahead of its source")
                terms[claim.digest] = (key, allocation.terms_at(claim.crowned_block)[key])
                bonuses[claim.digest] = allocation.stall_bonus_at(claim.crowned_block)
            for claim in (*standing, *inputs["earning_claims"], *inputs["discovery_claims"]):
                evidence.add(claim.retained_evidence_digest)
            for name, values in inputs.items():
                if name == "decay_start_blocks":
                    combined[name].update(values)
                else:
                    combined[name].extend(values)
            store._bind_emissions_policy(policy)
            snapshots[key] = {"config_digest": configs[key].digest, "cursor": list(cursor),
                              "settlement_state_digest": store.settlement_state_digest()}
        projection = project_global_rewards(
            policy, context, **combined, allocation_terms=terms,
            allocation_burn_hotkey=allocation.burn_hotkey, stall_bonus_terms=bonuses)
        _, shares, paid, burned = allocate_submission_weights(
            projection.standing, terms, context, allocation.burn_hotkey,
            base_credits=arena_base_credits(combined["earning_claims"], policy, context,
                                          combined["decay_start_blocks"]))
        report = {"schema": "cacheon.static-arena-allocation.v1",
                  "allocation_digest": allocation.digest, "effective_block": context.current_block,
                  "metagraph_digest": context.metagraph_digest, "sources": snapshots,
                  "submission_terms": {key: list(value) for key, value in sorted(terms.items())},
                  "arena_weights_ppm": {key: shares.get(key, 0) for key in configs}, "burned_ppm": burned,
                  "weights_ppm": projection.weights_by_hotkey,
                  "decay_digest": canonical_digest("cacheon.static-arena-decay.v1", adjustments)}
        if any(value != 1_000_000 for value in bonuses.values()):
            report["submission_stall_bonus_ppm"] = dict(sorted(bonuses.items()))
        root = prepare_evidence_root(primary.path.parent / "weight-allocation-evidence")
        ref = publish_canonical_json_evidence(root, report, domain="weights.arena-allocation",
                                              schema=report["schema"])
        reopen_evidence(root, ref)
        evidence.add(ref.sha256)
        rewarded = tuple(sorted({claim.retained_evidence_digest for claim in combined["earning_claims"]
                                 if claim.digest in paid}))
        return WeightProjection(
            context.chain_scope_digest, netuid, context.validator_hotkey,
            canonical_digest("cacheon.static-arena-policy.v1", {
                "base_policy": policy.digest, "allocation": allocation.digest,
                "decay": report["decay_digest"]}),
            canonical_digest("cacheon.static-arena-sources.v1", snapshots), projection.digest,
            context.metagraph_digest, projection.arena_authority_digests,
            generation, context.current_block,
            standing_count, tuple(sorted(evidence)),
            tuple(sorted(projection.weights_by_hotkey.items())), ref, rewarded)
