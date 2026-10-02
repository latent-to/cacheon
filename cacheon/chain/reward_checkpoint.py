"""Durable inputs to the existing reward calculator, independent of intake locks.

Only a successful production projection replaces this checkpoint. It retains
the reopened claims, stacks, frozen allocation terms and publication clocks,
not a vector to re-stamp. Fallback runs the same calculator at the live head.
"""

from dataclasses import replace
import json
import logging
from pathlib import Path

from cacheon.chain.intake import IntakeError
from cacheon.chain.remote_worker_spool import atomic_json
from cacheon.chain.sealed_config import authority_file
from cacheon.chain.weights import WeightProjection, WeightPublicationError
from cacheon.economics import (
    ArenaRewardAuthority, DiscoveryBountyClaim, StandingRewardClaim, project_global_rewards,
)
from cacheon.stack_manifest import EvaluationStackManifest, ProposalContributionRef
from cacheon.stack_identity import canonical_digest

logger = logging.getLogger("cacheon.chain.weight_share")


class RewardSourceUnavailable(IntakeError):
    """A configured reward database is absent; a saved checkpoint may be used."""


def _encode(inputs):
    data = dict(inputs)
    data["arenas"] = [{"stack": row.stack.to_dict(), "generation": row.stack_generation,
                       "claims": [claim.to_dict() for claim in row.standing_claims]}
                      for row in inputs["arenas"]]
    for key in ("earning_claims", "discovery_claims", "earned_contributions"):
        data[key] = [row.to_dict() for row in inputs[key]]
    if "validated_claims" in inputs:
        data["validated_claims"] = [row.to_dict() for row in inputs["validated_claims"]]
    return data


def _decode(data):
    inputs = dict(data)
    inputs["arenas"] = tuple(ArenaRewardAuthority(
        EvaluationStackManifest.from_dict(row["stack"]), row["generation"],
        tuple(StandingRewardClaim.from_dict(claim) for claim in row["claims"]))
        for row in data["arenas"])
    for key, cls in (("earning_claims", StandingRewardClaim),
                     ("discovery_claims", DiscoveryBountyClaim),
                     ("earned_contributions", ProposalContributionRef)):
        inputs[key] = tuple(cls.from_dict(row) for row in data[key])
    if "validated_claims" in data:
        inputs["validated_claims"] = tuple(StandingRewardClaim.from_dict(row) for row in data["validated_claims"])
    if "allocation_terms" in inputs:
        inputs["allocation_terms"] = {key: tuple(value) for key, value in inputs["allocation_terms"].items()}
    return inputs


class RewardCheckpoint:
    """One atomic, owner-controlled checkpoint per configured weights producer."""

    def __init__(self, path, *, policy, scope, stage):
        self.path = Path(path)
        self.policy = policy
        self.scope = scope
        self.stage = stage
        self.pending = None

    def _identity(self):
        from cacheon.chain.arena_weight_projection import load_allocation

        allocation = (load_allocation(self.stage.arena_allocation_path)
                      if self.stage.arena_allocation_path is not None else None)
        return {"scope": self.scope.digest, "netuid": self.scope.netuid,
                "policy": self.policy.digest, "signer": self.stage.attribution_hotkey,
                "allocation": None if allocation is None else allocation.digest,
                "burn_hotkey": self.stage.burn_hotkey,
                "confirmation_journal": str(self.stage.confirmation_journal)}

    def capture(self, projection, inputs=None, static=None):
        """Keep only inputs from a completed production projection for later commit."""
        self.pending = {"schema": "cacheon.reward-checkpoint.v1", "identity": self._identity(),
                        "projection": projection.to_dict(),
                        "inputs": None if inputs is None else _encode(inputs), "static": static,
                        "confirmation_sequence": 0}

    def save(self):
        """Replace the checkpoint after the live store transaction has closed."""
        if self.pending is not None:
            if self.path.exists():
                try:
                    previous = json.loads(self.path.read_text())
                    if previous["identity"] == self.pending["identity"] and "last_offer" in previous:
                        self.pending["last_offer"] = WeightProjection.from_dict(previous["last_offer"]).to_dict()
                except (KeyError, TypeError, ValueError, WeightPublicationError):
                    logger.warning("weight-offer: replacing unreadable checkpoint with validated live rewards")
            atomic_json(self.path, self.pending)
            self.pending = None

    def select_offer(self, projection):
        """Retain an exact offer before HTTP so a same-block retry is idempotent."""
        saved = json.loads(self.path.read_text())
        if "last_offer" in saved:
            prior = WeightProjection.from_dict(saved["last_offer"])
            if prior.effective_block > projection.effective_block:
                raise IntakeError("reward offer finalized head regressed")
            if prior.effective_block == projection.effective_block:
                if prior.metagraph_digest != projection.metagraph_digest:
                    raise IntakeError("reward offer metagraph changed at the same block")
                return prior
        saved["last_offer"] = projection.to_dict()
        atomic_json(self.path, saved)
        return projection

    def project(self, context):
        """Recalculate saved reward inputs at the live block, preserving fixed clocks."""
        authority_file(self.path, "reward checkpoint", error=IntakeError)
        saved = json.loads(self.path.read_text())
        if saved["schema"] != "cacheon.reward-checkpoint.v1" or saved["identity"] != self._identity():
            raise IntakeError("reward checkpoint differs from configured authority")
        template = WeightProjection.from_dict(saved["projection"])
        if context.current_block < template.effective_block:
            raise IntakeError("reward checkpoint is ahead of the finalized head")
        if self.stage.arena_allocation_path is not None and saved["static"] is None:
            from cacheon.chain.arena_weight_projection import load_allocation

            if context.current_block >= load_allocation(self.stage.arena_allocation_path).activation_block:
                raise IntakeError("reward checkpoint has not observed arena allocation activation")
        if saved["inputs"] is None:
            if not set(dict(template.weights_ppm)) <= context.eligible_hotkeys:
                raise IntakeError("checkpoint burn recipient is no longer registered")
            return replace(template, effective_block=context.current_block,
                           metagraph_digest=context.metagraph_digest)
        inputs = _decode(saved["inputs"])
        journal = self.stage.confirmation_journal
        if journal is not None:
            from cacheon.chain.qualification_settlement import read_reward_confirmations

            maximum, confirmations = read_reward_confirmations(
                journal, cursor={"path": str(journal), "sequence": saved["confirmation_sequence"]},
                scope=self.scope, validator_hotkey=context.validator_hotkey)
            starts = inputs["decay_start_blocks"]
            for projection, record in confirmations:
                published = (record.confirmed_block if record.reason == "block_inclusion"
                             else record.confirmed_last_update)
                if published < projection.effective_block:
                    continue
                evidence = (projection.evidence_digests if projection.rewarded_evidence_digests is None
                            else projection.rewarded_evidence_digests)
                for claim in inputs["earning_claims"]:
                    if (claim.digest in starts and starts[claim.digest] is None
                            and claim.retained_evidence_digest in evidence
                            and dict(projection.weights_ppm).get(claim.hotkey, 0) > 0):
                        starts[claim.digest] = record.confirmed_block
            saved["confirmation_sequence"] = maximum
        projection = project_global_rewards(self.policy, context, **inputs)
        evidence = set(template.evidence_digests)
        allocation_ref = template.allocation_evidence
        rewarded = template.rewarded_evidence_digests
        if saved["static"] is not None:
            from cacheon.arena_allocation import allocate_submission_weights, arena_base_credits
            from cacheon.eval.evidence_store import prepare_evidence_root, publish_canonical_json_evidence

            static = saved["static"]
            submission_weights = {}
            _, shares, paid, burned = allocate_submission_weights(
                projection.standing, inputs["allocation_terms"], context, inputs["allocation_burn_hotkey"],
                submission_weights=submission_weights,
                base_credits=arena_base_credits(inputs["earning_claims"], self.policy, context,
                                              inputs["decay_start_blocks"], inputs["score_speedups"]))
            report = {**static["report"], "effective_block": context.current_block,
                      "metagraph_digest": context.metagraph_digest,
                      "weights_ppm": projection.weights_by_hotkey, "burned_ppm": burned,
                      "arena_weights_ppm": {key: shares.get(key, 0) for key in static["report"]["sources"]},
                      "submission_weights_ppm": {static["reservation_ids"][key]: ppm
                                                 for key, ppm in sorted(submission_weights.items())},
                      "decay_digest": canonical_digest("cacheon.checkpoint-decay.v1", inputs["decay_start_blocks"])}
            allocation_ref = publish_canonical_json_evidence(
                prepare_evidence_root(Path(static["root"])), report,
                domain="weights.arena-allocation", schema=report["schema"])
            evidence.remove(template.allocation_evidence.sha256)
            evidence.add(allocation_ref.sha256)
            rewarded = tuple(sorted({claim.retained_evidence_digest for claim in inputs["earning_claims"]
                                     if claim.digest in paid}))
        result = replace(
            template, effective_block=context.current_block, metagraph_digest=context.metagraph_digest,
            evaluation_state_digest=projection.digest, arena_state_digests=projection.arena_authority_digests,
            weights_ppm=tuple(sorted(projection.weights_by_hotkey.items())),
            evidence_digests=tuple(sorted(evidence)), allocation_evidence=allocation_ref,
            rewarded_evidence_digests=rewarded,
        )
        saved["inputs"] = _encode(inputs)
        # Clock starts are persisted before the offer is pushed; a failed push
        # or restart cannot start a claim's decay again at a later block.
        atomic_json(self.path, saved)
        logger.warning("weight-offer: projecting retained reward checkpoint from block %s at block %s",
                       template.effective_block, context.current_block)
        return result
