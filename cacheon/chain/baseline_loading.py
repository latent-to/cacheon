"""Run the operator's existing commissioner when FIFO reaches a disclosed baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from cacheon.chain.baseline_admission import promotion_target, same_baseline
from cacheon.chain.remote_worker_spool import atomic_json, load_json
from cacheon.stack_manifest import EvaluationStackManifest


def loading_config(value):
    """Validate an optional operator-owned commissioning command and rollout block."""
    if type(value) is not dict or set(value) != {"command", "timeout_seconds", "activation_block"}:
        raise ValueError("baseline_loading requires command, timeout_seconds and activation_block")
    command = value["command"]
    if (type(command) is not list or not command
            or any(type(arg) is not str or not arg or "\0" in arg for arg in command)
            or not Path(command[0]).is_absolute()):
        raise ValueError("baseline_loading command must be an argv with an absolute executable")
    if type(value["timeout_seconds"]) is not int or not 1 <= value["timeout_seconds"] <= 3600:
        raise ValueError("baseline_loading timeout_seconds must be between 1 and 3600")
    if type(value["activation_block"]) is not int or value["activation_block"] < 0:
        raise ValueError("baseline_loading activation_block must be nonnegative")
    return value


def exact_block_clock(subtensor):
    """Cache finalized chain timestamps for both disclosure and submission admission."""
    from functools import lru_cache

    @lru_cache(maxsize=4096)
    def block_time(block):
        substrate = subtensor.substrate
        block_hash = substrate.get_block_hash(block)
        value = substrate.query("Timestamp", "Now", block_hash=block_hash)
        stamp = getattr(value, "value", value)
        if stamp is None:
            raise RuntimeError(f"finalized timestamp unavailable at block {block}")
        return {"unix": int(stamp) // 1000, "estimated": False}

    return block_time


class BaselineLoader:
    """Keep failed or interrupted cutovers out of qualification until resolved."""

    def __init__(self, config, dispatcher):
        self.config = config
        self.dispatcher = dispatcher
        self.path = config.source_path
        self.pending = self.path.with_name(self.path.name + ".baseline-loading.json")

    def recover(self):
        """A restart after pointer replacement can finish bookkeeping without reloading."""
        if not self.pending.exists():
            return
        target = EvaluationStackManifest.from_dict(load_json(self.pending)["incumbent_stack"])
        if not same_baseline(target, self.dispatcher.qualification_incumbent_stack):
            raise RuntimeError("baseline loading is incomplete; inspect the retained promotion before resuming")
        self.pending.unlink()

    def __call__(self):
        from cacheon.chain.recoverable_qualification_dispatcher import QualificationCommissionRequired
        from cacheon.chain.standing_cpu_supervisor import SupervisorStageResult, load_standing_config
        from cacheon.chain.mainnet_screen_dispatcher import load_config

        self.recover()
        result = self.dispatcher.dispatch_once()
        if type(result) is not QualificationCommissionRequired:
            return result
        store, _ = self.dispatcher._open_store()
        try:
            target = promotion_target(store, self.dispatcher.qualification_incumbent_stack)
        finally:
            store.close()
        if target is None:
            return result
        settings = self.config.raw["baseline_loading"]
        request = {
            "standing_config": str(self.path),
            "incumbent_stack": target.manifest.to_dict(),
            "incumbent_tree_digest": target.tree_digest,
            "transition_event_id": target.transition_event_id,
        }
        # The durable marker precedes any external side effect. A failed command
        # must not be repeated automatically against an unknown partial cutover.
        with os.fdopen(os.open(self.pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            json.dump(request, stream)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.pending.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        completed = subprocess.run(
            settings["command"], input=json.dumps(request), text=True,
            stdout=subprocess.PIPE, check=True, timeout=settings["timeout_seconds"],
        )
        response = json.loads(completed.stdout)
        if type(response) is not dict or set(response) != {"standing_config"}:
            raise ValueError("baseline loader must return the prepared standing_config path")
        next_config = load_standing_config(response["standing_config"])
        loaded = EvaluationStackManifest.from_dict(load_json(next_config.qualification_incumbent_stack_path))
        if not same_baseline(target.manifest, loaded):
            raise ValueError("baseline loader prepared a different incumbent")
        before, after = (load_config(c.screen_dispatcher_config) for c in (self.config, next_config))
        if loaded.arena_digest != after.manifest.digest:
            raise ValueError("baseline loader incumbent differs from the prepared service")
        if (before.intake_db != after.intake_db or before.scope != after.scope
                or before.manifest.runtime != after.manifest.runtime
                or before.manifest.workload != after.manifest.workload
                or before.manifest.qualification_policy_digest != after.manifest.qualification_policy_digest
                or before.manifest.closed_targets != after.manifest.closed_targets
                or before.policy != after.policy
                or self.config.raw["baseline_loading"] != next_config.raw.get("baseline_loading")):
            raise ValueError("baseline loader changed the competition, runtime, workload or policy")
        if (not next_config.enable_qualification or not next_config.enable_settlement
                or self.config.enable_weights != next_config.enable_weights
                or self.config.raw["weights_stage_config"] != next_config.raw["weights_stage_config"]):
            raise ValueError("baseline loader changed enabled stages or weight authority")
        if load_standing_config(self.path).raw != self.config.raw:
            raise ValueError("standing configuration changed during baseline loading")
        atomic_json(self.path, next_config.raw, mode=0o400)
        self.pending.unlink()
        return SupervisorStageResult("qualification", True, disposition="baseline_loaded")
