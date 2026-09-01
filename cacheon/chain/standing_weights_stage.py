"""Eval-side weight-offer push stage for the standing CPU supervisor.

The supervisor never signs weights. When ``enable_weights`` is set, this stage
projects the current V1 offer from the intake store and HTTP-pushes it to the
serve-weights lane, which owns readback and the follow-weights signer decides
whether anything reaches the chain. Composition lives in
``standing_cpu_supervisor.build_standing_supervisor``; the sealed
``weights_stage_config`` file this module reopens is documented in
``docs/validator-guide/chain-loop.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

WEIGHTS_CONFIG_SCHEMA = "cacheon-standing-weights-config-v1"
WEIGHTS_CONFIG_DOMAIN = "cacheon.chain.standing-weights-config.v1"
_WEIGHTS_CONFIG_FIELDS = frozenset(
    {
        "attribution_hotkey",
        "burn_hotkey",
        "discovery_lifetime_blocks",
        "discovery_pool_ppm",
        "excluded_claim_digests",
        "excluded_hotkeys",
        "fallback_endpoint",
        "half_life_blocks",
        "network",
        "push_credentials",
        "push_url",
        "refresh_blocks",
        "schema",
        "time_multiplier_scale_blocks",
    }
)


@dataclass(frozen=True)
class WeightsStageConfig:
    """Sealed eval-side weight-offer push; never a chain signer."""

    network: str
    fallback_endpoint: str
    push_url: str
    push_credentials: Path
    attribution_hotkey: str
    half_life_blocks: int
    discovery_lifetime_blocks: int
    discovery_pool_ppm: int
    time_multiplier_scale_blocks: int
    excluded_hotkeys: tuple[str, ...]
    excluded_claim_digests: tuple[str, ...]
    refresh_blocks: int
    # All-uncrowned fallback target. When the intake store holds no economic
    # authority (no active claim, no crowned arena, no activated composition)
    # the stage pushes the full-pool burn offer to this hotkey instead of
    # refusing. Empty disables the fallback: a crownless store then surfaces
    # the builder's refusal as a stage error, exactly as before.
    burn_hotkey: str


@dataclass(frozen=True)
class WeightPushResult:
    """What one accepted offer push reports back to ``weights_stage``."""

    projection_digest: str
    status: str


def load_weights_config(path: str | os.PathLike[str]) -> WeightsStageConfig:
    """Strictly reopen the eval-side push-weight-offer authority file."""

    from cacheon.chain.remote_worker_spool import load_json
    from cacheon.chain.standing_cpu_supervisor import (
        StandingCpuSupervisorError,
        _absolute_path,
        _authority_file,
        _closed_config,
        _positive_int,
    )
    from cacheon.economics import EconomicsError, EmissionsPolicyManifest

    config_path = _absolute_path(os.fspath(path), "weights_stage_config")
    _authority_file(config_path, "weights stage config")
    try:
        raw = load_json(config_path)
    except Exception as exc:
        raise StandingCpuSupervisorError(
            f"weights stage config cannot reopen: {exc}"
        ) from None
    row = _closed_config(raw, _WEIGHTS_CONFIG_FIELDS, "weights stage config")
    if row["schema"] != WEIGHTS_CONFIG_SCHEMA:
        raise StandingCpuSupervisorError("weights stage config schema is unsupported")

    network = row["network"]
    fallback_endpoint = row["fallback_endpoint"]
    push_url = row["push_url"]
    attribution = row["attribution_hotkey"]
    if type(network) is not str or not network.startswith("wss://"):
        raise StandingCpuSupervisorError("weights network must be an explicit wss:// URL")
    if type(fallback_endpoint) is not str:
        raise StandingCpuSupervisorError("weights fallback_endpoint is malformed")
    if fallback_endpoint and not fallback_endpoint.startswith("wss://"):
        raise StandingCpuSupervisorError(
            "weights fallback_endpoint must be empty or an explicit wss:// URL"
        )
    if type(push_url) is not str or not push_url.startswith(("http://", "https://")):
        raise StandingCpuSupervisorError("weights push_url must be http(s)")
    if (
        type(attribution) is not str
        or not attribution
        or attribution.strip() != attribution
        or len(attribution) > 256
    ):
        raise StandingCpuSupervisorError("weights attribution_hotkey is malformed")
    burn_hotkey = row["burn_hotkey"]
    if type(burn_hotkey) is not str or burn_hotkey.strip() != burn_hotkey or len(burn_hotkey) > 256:
        raise StandingCpuSupervisorError("weights burn_hotkey is malformed")
    credentials_path = _absolute_path(row["push_credentials"], "weights push_credentials")
    _authority_file(credentials_path, "weights push credentials", secret=True)
    half_life_blocks = _positive_int(
        row["half_life_blocks"], "weights half_life_blocks", maximum=10_000_000
    )
    discovery_lifetime_blocks = _positive_int(
        row["discovery_lifetime_blocks"],
        "weights discovery_lifetime_blocks",
        maximum=10_000_000,
    )
    discovery_pool_ppm = (
        _positive_int(
            row["discovery_pool_ppm"], "weights discovery_pool_ppm", maximum=999_999
        )
        if row["discovery_pool_ppm"] != 0
        else 0
    )
    time_multiplier_scale_blocks = _positive_int(
        row["time_multiplier_scale_blocks"],
        "weights time_multiplier_scale_blocks",
        maximum=10_000_000,
    )
    try:
        policy = EmissionsPolicyManifest(
            half_life_blocks,
            discovery_lifetime_blocks,
            discovery_pool_ppm,
            time_multiplier_scale_blocks,
            excluded_hotkeys=row["excluded_hotkeys"],
            excluded_claim_digests=row["excluded_claim_digests"],
        )
    except EconomicsError as exc:
        raise StandingCpuSupervisorError(
            f"weights emissions policy is malformed: {exc}"
        ) from None
    return WeightsStageConfig(
        network=network,
        fallback_endpoint=fallback_endpoint,
        push_url=push_url,
        push_credentials=credentials_path,
        attribution_hotkey=attribution,
        half_life_blocks=half_life_blocks,
        discovery_lifetime_blocks=discovery_lifetime_blocks,
        discovery_pool_ppm=discovery_pool_ppm,
        time_multiplier_scale_blocks=time_multiplier_scale_blocks,
        excluded_hotkeys=policy.excluded_hotkeys,
        excluded_claim_digests=policy.excluded_claim_digests,
        refresh_blocks=_positive_int(
            row["refresh_blocks"], "weights refresh_blocks", maximum=86_400
        ),
        burn_hotkey=burn_hotkey,
    )


def compose_weight_offer_push(
    stage: WeightsStageConfig,
    *,
    store_factory: Callable[[], Any],
    scope: Any,
) -> Callable[[], Any]:
    """HTTP-push the current V1 offer to serve-weights; never set_weights."""

    from cacheon import chain
    from cacheon.chain.standing_cpu_supervisor import (
        StandingCpuSupervisorError,
        weights_stage,
    )
    from cacheon.chain.weight_push_auth import resolve_push_credentials
    from cacheon.chain.weight_share import (
        CurrentWeightOffer,
        WeightShareError,
        WeightShareRetryableError,
        push_current_weights,
    )
    from cacheon.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )
    from cacheon.target_catalog import default_target_catalog

    fallbacks = [stage.fallback_endpoint] if stage.fallback_endpoint else None
    subtensor = chain.connect(
        stage.network,
        fallback_endpoints=fallbacks,
        retry_forever=True,
    )
    credentials = resolve_push_credentials(str(stage.push_credentials), required=True)
    if credentials is None:
        raise StandingCpuSupervisorError("weights push credentials did not resolve")
    credential = credentials.active()[0]
    policy = EmissionsPolicyManifest(
        stage.half_life_blocks,
        stage.discovery_lifetime_blocks,
        stage.discovery_pool_ppm,
        stage.time_multiplier_scale_blocks,
        excluded_hotkeys=stage.excluded_hotkeys,
        excluded_claim_digests=stage.excluded_claim_digests,
    )
    last_push_block = 0
    netuid = int(scope.netuid)

    def publish() -> Any:
        nonlocal last_push_block
        try:
            current_block, _ = chain.read_finalized_head(subtensor)
            if last_push_block and current_block - last_push_block < stage.refresh_blocks:
                return None
            metagraph = chain.fetch_metagraph(subtensor, netuid)
            context = GlobalRewardProjectionContext(
                scope.digest,
                stage.attribution_hotkey,
                metagraph.block,
                metagraph.block_hash.lower(),
                tuple(
                    MetagraphMember(uid, hotkey)
                    for uid, hotkey in zip(
                        metagraph.uids, metagraph.hotkeys, strict=True
                    )
                ),
            )
            catalog = default_target_catalog()
            with store_factory() as store:
                states = store.evaluation_stacks()
                standing, discovery = store.active_reward_claims()
                crowned = any(state.generation > 0 for state in states)
                if standing or discovery or crowned:
                    # Real economic authority exists: project it. A mixed or
                    # torn state (claims without a crowned arena, or the
                    # reverse) is the builder's refusal to surface, not a
                    # reason to burn.
                    catalogs = {state.arena_digest: catalog for state in states}
                    projection = store.build_weight_projection(
                        policy=policy,
                        context=context,
                        catalogs=catalogs,
                        netuid=netuid,
                    )
                elif stage.burn_hotkey:
                    projection = store.build_burn_weight_projection(
                        policy=policy,
                        context=context,
                        netuid=netuid,
                        burn_hotkey=stage.burn_hotkey,
                    )
                else:
                    catalogs = {state.arena_digest: catalog for state in states}
                    projection = store.build_weight_projection(
                        policy=policy,
                        context=context,
                        catalogs=catalogs,
                        netuid=netuid,
                    )
            offer = CurrentWeightOffer.from_legacy_projection(projection)
            response = push_current_weights(
                stage.push_url,
                offer,
                credential=credential,
            )
        except StandingCpuSupervisorError:
            raise
        except (WeightShareRetryableError, WeightShareError) as exc:
            raise StandingCpuSupervisorError(f"weight offer push failed: {exc}") from None
        except Exception as exc:
            raise StandingCpuSupervisorError(
                f"weight offer push failed: {type(exc).__name__}: {exc}"
            ) from None
        last_push_block = int(current_block)
        status = response.get("status") if isinstance(response, dict) else None
        return WeightPushResult(
            projection_digest=projection.digest,
            status=status if isinstance(status, str) and status else "pushed",
        )

    return weights_stage(publish=publish)


__all__ = [
    "WEIGHTS_CONFIG_DOMAIN",
    "WEIGHTS_CONFIG_SCHEMA",
    "WeightPushResult",
    "WeightsStageConfig",
    "compose_weight_offer_push",
    "load_weights_config",
]
