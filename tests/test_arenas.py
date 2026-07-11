from dataclasses import replace

import pytest

from optima.arenas import (
    ArenaPolicyError,
    MINIMAX_M3_B300_TP4_DECODE_V1,
    MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
    huggingface_model_manifest,
    get_arena,
    derive_prompt_seed,
    list_arenas,
    referee_source_digest,
)


def test_registered_arenas_are_immutable_competable_and_distinct():
    long = MINIMAX_M3_B300_TP4_LONGPREFILL_V1
    decode = MINIMAX_M3_B300_TP4_DECODE_V1
    assert long.competable and decode.competable
    assert long.fingerprint != decode.fingerprint
    assert long.bracket != decode.bracket
    assert long.validator_image.startswith("lmsysorg/sglang@sha256:")
    with pytest.raises(TypeError):
        long.engine_kwargs["context_length"] = 1


def test_any_score_affecting_policy_change_rotates_fingerprint():
    arena = MINIMAX_M3_B300_TP4_LONGPREFILL_V1
    changed_workload = replace(
        arena,
        workload=replace(arena.workload, num_prompts=arena.workload.num_prompts + 1),
    )
    changed_engine = replace(
        arena,
        engine_kwargs={**dict(arena.engine_kwargs), "chunked_prefill_size": 16384},
    )
    changed_runtime = replace(arena, sglang_version=arena.sglang_version + ".next")
    changed_topology = replace(arena, gpu_topology_sha256="3" * 64)
    changed_gpu = replace(arena, gpu_memory_mib=arena.gpu_memory_mib - 1)
    changed_referee = replace(
        arena, referee_source_digest="sha256:" + "1" * 64
    )
    changed_release = replace(
        arena, referee_tree_digest="sha256:" + "4" * 64
    )
    changed_model = replace(
        arena, model_manifest_digest="sha256:" + "2" * 64
    )
    changed_overlays = replace(
        arena,
        runtime_overlays=(
            replace(arena.runtime_overlays[0], sha256="5" * 64),
            *arena.runtime_overlays[1:],
        ),
    )
    changed_resources = replace(
        arena,
        oci_resources=replace(
            arena.oci_resources, cpu_limit=arena.oci_resources.cpu_limit - 1
        ),
    )
    changed_deadline = replace(
        arena,
        oci_resources=replace(
            arena.oci_resources,
            batch_timeout_s=arena.oci_resources.batch_timeout_s + 1,
        ),
    )
    changed_device_envelope = replace(
        arena,
        device_state=replace(
            arena.device_state,
            maximum_temperature_c=arena.device_state.maximum_temperature_c + 1,
        ),
    )
    changed_retry_budget = replace(
        arena,
        settlement=replace(
            arena.settlement,
            retry_max_automatic_infrastructure_attempts=(
                arena.settlement.retry_max_automatic_infrastructure_attempts + 1
            ),
        ),
    )
    assert len({
        arena.fingerprint,
        changed_workload.fingerprint,
        changed_engine.fingerprint,
        changed_runtime.fingerprint,
        changed_topology.fingerprint,
        changed_gpu.fingerprint,
        changed_referee.fingerprint,
        changed_release.fingerprint,
        changed_model.fingerprint,
        changed_overlays.fingerprint,
        changed_resources.fingerprint,
        changed_deadline.fingerprint,
        changed_device_envelope.fingerprint,
        changed_retry_budget.fingerprint,
    }) == 14


def test_non_digest_image_is_not_a_valid_arena():
    with pytest.raises(ArenaPolicyError, match="immutable sha256"):
        replace(
            MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
            validator_image="lmsysorg/sglang:latest",
        )


def test_teacher_policy_rejects_boolean_or_unbounded_fields():
    with pytest.raises(ArenaPolicyError, match="clusters_per_batch"):
        replace(
            MINIMAX_M3_B300_TP4_LONGPREFILL_V1.fidelity.teacher_forced_policy,
            clusters_per_batch=False,
        )


@pytest.mark.parametrize("value", [0, False, 1.5])
def test_infrastructure_retry_budget_requires_positive_integer(value):
    arena = MINIMAX_M3_B300_TP4_LONGPREFILL_V1
    with pytest.raises(ArenaPolicyError, match="infrastructure retry attempts"):
        replace(
            arena.settlement,
            retry_max_automatic_infrastructure_attempts=value,
        )


def test_retry_attempt_budget_defaults_and_cross_field_ordering():
    settlement = MINIMAX_M3_B300_TP4_LONGPREFILL_V1.settlement
    assert settlement.retry_max_automatic_infrastructure_attempts == 3
    assert settlement.retry_max_automatic_no_decision_attempts == 4
    assert settlement.retry_max_total_attempts == 6

    with pytest.raises(ArenaPolicyError, match="no-decision/total"):
        replace(settlement, retry_max_automatic_no_decision_attempts=0)
    with pytest.raises(ArenaPolicyError, match="no-decision/total"):
        replace(settlement, retry_max_total_attempts=3)


def test_arena_lookup_is_explicit_no_adhoc_default():
    assert get_arena(MINIMAX_M3_B300_TP4_DECODE_V1.name) is MINIMAX_M3_B300_TP4_DECODE_V1
    assert MINIMAX_M3_B300_TP4_LONGPREFILL_V1.name in list_arenas()
    with pytest.raises(ArenaPolicyError, match="unknown scoring arena"):
        get_arena("debug-whatever")


def test_arena_eval_config_is_profile_authoritative_and_safe():
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    cfg = arena.eval_config_kwargs()
    assert cfg["model_path"] == arena.model_path
    assert cfg["tp_size"] == arena.tp_size
    assert cfg["timed_iters"] == arena.scoring.timed_iters
    assert cfg["warmup_iters"] == arena.scoring.warmup_iters
    assert cfg["conditioning_iters"] == arena.scoring.conditioning_iters
    assert cfg["ignore_eos"] is True
    assert cfg["disable_cuda_graph"] is False
    assert cfg["isolate"] is True
    assert cfg["allow_unsafe_no_isolation"] is False
    assert cfg["framework_mode"] is False
    assert cfg["candidate_extra_engine_kwargs"] == {}


def test_post_commit_prompt_seed_is_domain_separated_and_never_dev_zero():
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    seed = derive_prompt_seed(
        arena, bundle_hash="a" * 64, round_id=7, block_hash="0xpost-commit"
    )
    assert seed > 0
    assert seed == derive_prompt_seed(
        arena, bundle_hash="a" * 64, round_id=7, block_hash="0xpost-commit"
    )
    assert seed != derive_prompt_seed(
        arena, bundle_hash="b" * 64, round_id=7, block_hash="0xpost-commit"
    )
    assert seed != derive_prompt_seed(
        MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
        bundle_hash="a" * 64,
        round_id=7,
        block_hash="0xpost-commit",
    )


def test_referee_source_digest_tracks_path_and_bytes(tmp_path):
    package = tmp_path / "optima"
    package.mkdir()
    (package / "a.py").write_text("VALUE = 1\n")
    first = referee_source_digest(package)
    (package / "a.py").write_text("VALUE = 2\n")
    assert referee_source_digest(package) != first
    (package / "a.py").write_text("VALUE = 1\n")
    (package / "nested").mkdir()
    (package / "nested" / "a.py").write_text("VALUE = 1\n")
    assert referee_source_digest(package) != first


def test_huggingface_model_manifest_pins_revision_and_object_receipt(tmp_path):
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("a" * 40 + "\nobject-a\n0\n")
    (metadata / "model.safetensors.metadata").write_text(
        "a" * 40 + "\nobject-b\n0\n"
    )
    revision, digest = huggingface_model_manifest(tmp_path)
    assert revision == "a" * 40
    assert digest.startswith("sha256:") and len(digest) == 71
    (metadata / "model.safetensors.metadata").write_text(
        "b" * 40 + "\nobject-b\n0\n"
    )
    with pytest.raises(ArenaPolicyError, match="mixes revisions"):
        huggingface_model_manifest(tmp_path)
