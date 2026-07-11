"""Inspectable, arena-scoped SGLang system-patch product.

This is the bounded escape hatch between a frozen component slot and an
arbitrary miner-provided inference service.  A bundle declares one set of exact
unified diffs against the validator's pinned SGLang package.  Validator policy
owns the competition target, semantic region, file allowlist, arena admission,
and external fidelity requirement.

The trusted build phase copies the *entire* pinned ``sglang`` Python package into
a content-addressed overlay and applies only the declared, allowlisted text
patches.  The shared installation is never modified.  The candidate phase only
validates and imports that prebuilt overlay; it cannot build or repair it.

This module deliberately does not wire the CLI/evaluator/ledger.  The exported
``qualification_requirement``, ``prebuild_system_overlay`` and
``system_launch_environment`` hooks let those trusted-controller layers opt into
the lane without pretending a system product is a component slot.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping

from optima.compat import PINNED_SGLANG


SYSTEM_OVERLAY_SCHEMA = 1
SGLANG_INFERENCE_SYSTEM_V1 = "sglang.inference.v1"
SYSTEM_OVERLAY_IGNORED_NAMES = (
    "__pycache__",
    "*.pyc",
    # Present in the pinned editable SGLang image but never imported at runtime;
    # hidden development metadata is forbidden in immutable candidate artifacts.
    ".clang-format",
    ".claude",
)


class SystemPatchError(RuntimeError):
    """A system bundle, patch policy, or materialized overlay is invalid."""


@dataclass(frozen=True)
class SystemTargetPolicy:
    """Validator-owned meaning of one system competition target."""

    target: str
    source_target: str
    region: str
    sglang_version: str
    arenas: tuple[str, ...]
    external_quality_gate: str
    allowed_prefixes: tuple[str, ...]
    allowed_files: tuple[str, ...]
    # ``path::qualified.symbol`` rows admit only changed lines inside explicit
    # functions/methods of an otherwise mixed-trust file.
    allowed_symbol_regions: tuple[str, ...]
    allowed_suffixes: tuple[str, ...]
    forbidden_path_markers: tuple[str, ...]
    forbidden_added_source: tuple[str, ...]


_INFERENCE_POLICY = SystemTargetPolicy(
    target=SGLANG_INFERENCE_SYSTEM_V1,
    source_target="sglang",
    region="inference",
    sglang_version=PINNED_SGLANG,
    arenas=(
        "minimax-m3-b300-tp4-decode-v1",
        "minimax-m3-b300-tp4-longprefill-v1",
    ),
    external_quality_gate="controller-posthoc-teacher-forced-v2",
    # Broad model-execution surface.  These directories contain model math,
    # kernels, cache layout/management, and overlap mechanics.  They do not
    # contain the HTTP/API/tokenization boundary or the trusted timing driver.
    allowed_prefixes=(
        "sglang/srt/layers/",
        "sglang/srt/model_executor/",
        "sglang/srt/models/",
        "sglang/srt/mem_cache/",
        "sglang/srt/batch_overlap/",
    ),
    # Scheduler admission is intentionally file-by-file because the managers
    # directory also owns tokenization, output/result construction, metrics and
    # logprob plumbing.  scheduler.py itself mixes those trust surfaces, so only
    # the explicit batching/forward symbols below are admitted from that file.
    allowed_files=(
        "sglang/srt/managers/overlap_utils.py",
        "sglang/srt/managers/prefill_delayer.py",
        "sglang/srt/managers/schedule_batch.py",
        "sglang/srt/managers/schedule_policy.py",
        "sglang/srt/managers/scheduler_input_blocker.py",
        "sglang/srt/managers/scheduler_pp_mixin.py",
        "sglang/srt/managers/scheduler_recv_skipper.py",
        "sglang/srt/managers/scheduler_components/dp_attn.py",
        "sglang/srt/managers/scheduler_components/flush_wrapper.py",
        "sglang/srt/managers/scheduler_components/idle_sleeper.py",
        "sglang/srt/managers/scheduler_components/invariant_checker.py",
        "sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py",
        # Mixed-trust file: actual admission is narrowed to the symbols below.
        "sglang/srt/managers/scheduler.py",
    ),
    allowed_symbol_regions=(
        "sglang/srt/managers/scheduler.py::Scheduler.init_moe_gemm_config",
        "sglang/srt/managers/scheduler.py::Scheduler.init_tp_model_worker",
        "sglang/srt/managers/scheduler.py::Scheduler.init_target_memory_pool",
        "sglang/srt/managers/scheduler.py::Scheduler.init_memory_pools",
        "sglang/srt/managers/scheduler.py::Scheduler.init_all_backends",
        "sglang/srt/managers/scheduler.py::Scheduler.init_model_worker",
        "sglang/srt/managers/scheduler.py::Scheduler.init_chunked_prefill",
        "sglang/srt/managers/scheduler.py::Scheduler.init_schedule_policy",
        "sglang/srt/managers/scheduler.py::Scheduler.init_overlap",
        "sglang/srt/managers/scheduler.py::Scheduler.event_loop_normal",
        "sglang/srt/managers/scheduler.py::Scheduler.event_loop_overlap",
        "sglang/srt/managers/scheduler.py::Scheduler.is_disable_overlap_for_batch",
        "sglang/srt/managers/scheduler.py::Scheduler.stash_chunked_request",
        "sglang/srt/managers/scheduler.py::Scheduler._build_hisparse_decode_batch",
        "sglang/srt/managers/scheduler.py::Scheduler.get_next_batch_to_run",
        "sglang/srt/managers/scheduler.py::Scheduler.get_num_allocatable_reqs",
        "sglang/srt/managers/scheduler.py::Scheduler._should_delay_dflash_prefill_for_batching",
        "sglang/srt/managers/scheduler.py::Scheduler.get_new_batch_prefill",
        "sglang/srt/managers/scheduler.py::Scheduler._get_new_batch_prefill_raw",
        "sglang/srt/managers/scheduler.py::Scheduler._can_schedule_lora_req",
        "sglang/srt/managers/scheduler.py::Scheduler.update_running_batch",
        "sglang/srt/managers/scheduler.py::Scheduler.record_batch_in_overlap",
        "sglang/srt/managers/scheduler.py::Scheduler._forward_isolation",
        "sglang/srt/managers/scheduler.py::Scheduler.run_batch",
        "sglang/srt/managers/scheduler.py::dispatch_event_loop",
    ),
    # Inspectable source only.  A precompiled object/shared library or a build
    # script would reopen the arbitrary-image lane this product is meant to avoid.
    allowed_suffixes=(".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp"),
    forbidden_path_markers=(
        "api", "entrypoint", "http", "grpc",
        "tokenizer", "detokenizer",
        "sampler", "sampling", "logit", "logits", "logprob", "logprobs",
        "result", "results",
        "timing", "timer", "timers", "metric", "metrics",
        "profiler", "profiling",
    ),
    # A path allowlist is the main boundary.  This smaller source check prevents
    # an allowed inference file from explicitly reaching back into excluded host
    # result/sampling/API/timing modules.  It is defense in depth, not a claim that
    # static text inspection is a Python sandbox; no-egress isolation and the
    # controller's hidden one-shot comparison remain load-bearing.
    forbidden_added_source=(
        "sglang.srt.entrypoints",
        "sglang.srt.openai_api",
        "sglang.srt.managers.tokenizer",
        "sglang.srt.managers.detokenizer",
        "sglang.srt.managers.io_struct",
        "sglang.srt.sampling",
        "sglang.srt.layers.sampler",
        "sglang.srt.layers.logits_processor",
        "time.perf_counter",
        "time.monotonic",
    ),
)

SYSTEM_TARGETS: Mapping[str, SystemTargetPolicy] = MappingProxyType(
    {_INFERENCE_POLICY.target: _INFERENCE_POLICY}
)


@dataclass(frozen=True)
class SystemQualificationRequirement:
    """Settlement properties the evaluator must enforce for a system product."""

    target: str
    arena_name: str
    external_quality_gate: str
    one_shot_external: bool = True
    component_receipts: bool = False
    component_champions: bool = False


@dataclass(frozen=True)
class SystemOverlayIdentity:
    cache_key: str
    payload: dict
    stock_site_root: Path


def is_system_submission(manifest) -> bool:
    return getattr(manifest, "system", None) is not None


def uses_component_registry(manifest) -> bool:
    """System products never load component ops or emit per-slot receipts."""
    return not is_system_submission(manifest)


def system_patch_fingerprints(bundle_path: str | Path) -> tuple[str, ...]:
    """Exact + re-presentation-invariant copy signals for a system product.

    Returned separately from the per-slot fingerprint maps on purpose: assigning
    these bytes to a made-up slot would later let a system qualification leak into
    component settlement.  The chain layer can compare this product-level tuple
    within the system target/arena bracket.
    """
    from optima.copy_fingerprint import dep_patch_fingerprint
    from optima.manifest import load_manifest, resolve_system_patches

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    if manifest.system is None:
        return ()
    out: set[str] = set()
    for patch in resolve_system_patches(bundle, manifest):
        raw = patch.read_bytes()
        out.add(hashlib.sha256(raw).hexdigest())
        out.add(dep_patch_fingerprint(raw.decode("utf-8")))
    return tuple(sorted(out))


def qualification_requirement(
    manifest,
    *,
    competition_target: str,
    arena_name: str,
    external_quality_gate: str,
) -> SystemQualificationRequirement:
    """Validate and return the mandatory external system qualification contract.

    This hook is intentionally strict and arena-scoped.  In-engine receipts or
    candidate-produced logits cannot satisfy it, and a qualification from one
    arena cannot be replayed into another.
    """
    system = getattr(manifest, "system", None)
    if system is None:
        raise SystemPatchError("qualification hook requires a [system] manifest")
    policy = _require_target_policy(
        manifest, competition_target=competition_target, arena_name=arena_name
    )
    if external_quality_gate != policy.external_quality_gate:
        raise SystemPatchError(
            f"system target {competition_target!r} requires external one-shot gate "
            f"{policy.external_quality_gate!r}, not {external_quality_gate!r}"
        )
    return SystemQualificationRequirement(
        target=competition_target,
        arena_name=arena_name,
        external_quality_gate=external_quality_gate,
    )


def _require_target_policy(
    manifest,
    *,
    competition_target: str,
    arena_name: str,
) -> SystemTargetPolicy:
    system = getattr(manifest, "system", None)
    if system is None:
        raise SystemPatchError("bundle does not declare a top-level [system] product")
    policy = SYSTEM_TARGETS.get(competition_target)
    if policy is None:
        raise SystemPatchError(
            f"unknown system competition target {competition_target!r}"
        )
    if (system.target, system.region) != (policy.source_target, policy.region):
        raise SystemPatchError(
            f"system target {competition_target!r} requires target/region "
            f"{(policy.source_target, policy.region)!r}, got "
            f"{(system.target, system.region)!r}"
        )
    if arena_name not in policy.arenas:
        raise SystemPatchError(
            f"system target {competition_target!r} is not registered for arena "
            f"{arena_name!r}; admitted arenas are {policy.arenas!r}"
        )
    return policy


def _path_tokens(path: str) -> set[str]:
    import re

    return {
        token
        for component in PurePosixPath(path).parts
        for token in re.split(r"[^a-z0-9]+", component.lower())
        if token
    }


def validate_patch_path(policy: SystemTargetPolicy, path: str) -> None:
    """Enforce the explicit inference-region allowlist for one touched file."""
    p = PurePosixPath(path)
    if not path or p.is_absolute() or ".." in p.parts or p.parts[:1] != ("sglang",):
        raise SystemPatchError(f"illegal SGLang system patch path: {path!r}")
    if p.suffix not in policy.allowed_suffixes:
        raise SystemPatchError(
            f"system patch path {path!r} is not inspectable source; allowed suffixes "
            f"are {policy.allowed_suffixes!r}"
        )
    allowed = path in policy.allowed_files or any(
        path.startswith(prefix) for prefix in policy.allowed_prefixes
    )
    if not allowed:
        raise SystemPatchError(
            f"system patch path {path!r} is outside the validator-owned inference "
            "region (API/tokenization/result/timing surfaces are not patchable)"
        )
    markers = _path_tokens(path) & set(policy.forbidden_path_markers)
    if markers:
        raise SystemPatchError(
            f"system patch path {path!r} enters excluded semantic surface(s): "
            f"{sorted(markers)!r}"
        )


def _symbol_names_for_path(
    policy: SystemTargetPolicy, path: str
) -> tuple[str, ...]:
    prefix = path + "::"
    return tuple(
        row[len(prefix):]
        for row in policy.allowed_symbol_regions
        if row.startswith(prefix)
    )


def _symbol_line_ranges(source: str, wanted: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SystemPatchError(f"pinned semantic-region source is invalid Python: {exc}") from exc
    found: dict[str, tuple[int, int]] = {}

    def walk(nodes, prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                start = min(
                    [node.lineno]
                    + [decorator.lineno for decorator in node.decorator_list]
                )
                found[name] = (start, int(node.end_lineno or node.lineno))
            elif isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + ".")

    walk(tree.body)
    selected = {name: found[name] for name in wanted if name in found}
    if not selected:
        raise SystemPatchError(
            "pinned scheduler source contains none of the validator-owned semantic "
            f"regions for this patch: {wanted!r}"
        )
    return selected


def _validate_changed_lines_in_symbols(
    file_patch,
    *,
    original_source: str,
    allowed_symbols: tuple[str, ...],
) -> None:
    from optima.deppatch import apply_file_patch

    old_ranges = tuple(_symbol_line_ranges(original_source, allowed_symbols).values())

    try:
        new_source = apply_file_patch(original_source, file_patch)
    except Exception as exc:
        raise SystemPatchError(str(exc)) from exc
    new_ranges = tuple(_symbol_line_ranges(new_source, allowed_symbols).values())

    def permitted(line: int, ranges: tuple[tuple[int, int], ...]) -> bool:
        return any(start <= line <= end for start, end in ranges)

    for hunk in file_patch.hunks:
        old_line = hunk.old_start
        new_line = hunk.new_start
        for change in hunk.lines:
            tag = change[:1]
            if tag == " ":
                old_line += 1
                new_line += 1
            elif tag == "-":
                if not permitted(old_line, old_ranges):
                    raise SystemPatchError(
                        f"system patch {file_patch.path!r} changes line {old_line} "
                        "outside its validator-owned scheduler semantic regions"
                    )
                old_line += 1
            elif tag == "+":
                # Validate against the *resulting AST* as well as the old-source
                # removal coordinates.  This prevents a hunk at a method's final
                # line from inserting unindented top-level code just outside the
                # semantic region while borrowing the method's old anchor.
                if not permitted(new_line, new_ranges):
                    raise SystemPatchError(
                        f"system patch {file_patch.path!r} inserts at resulting line "
                        f"{new_line} outside its validator-owned scheduler "
                        "semantic regions"
                    )
                new_line += 1


def validate_file_patch(
    policy: SystemTargetPolicy,
    file_patch,
    *,
    original_source: str | None = None,
) -> None:
    """Apply path, semantic-symbol, and excluded-source policy."""
    validate_patch_path(policy, file_patch.path)
    symbols = _symbol_names_for_path(policy, file_patch.path)
    if symbols:
        if file_patch.is_new_file or original_source is None:
            raise SystemPatchError(
                f"system patch {file_patch.path!r} requires pinned source for "
                "semantic-region validation"
            )
        _validate_changed_lines_in_symbols(
            file_patch,
            original_source=original_source,
            allowed_symbols=symbols,
        )
    added = "\n".join(
        line[1:] for hunk in file_patch.hunks for line in hunk.lines
        if line.startswith("+")
    ).lower()
    for forbidden in policy.forbidden_added_source:
        if forbidden.lower() in added:
            raise SystemPatchError(
                f"system patch {file_patch.path!r} references excluded source surface "
                f"{forbidden!r}"
            )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _policy_payload(policy: SystemTargetPolicy) -> dict:
    return json.loads(json.dumps(asdict(policy), sort_keys=True))


def _system_overlay_root() -> Path:
    configured = os.environ.get("OPTIMA_SYSTEM_OVERLAY_ROOT", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.home() / ".cache" / "optima" / "system_overlays" / "v1"
    )


def system_overlay_base(cache_key: str, *, cache_root: Path | None = None) -> Path:
    root = Path(cache_root).resolve() if cache_root is not None else _system_overlay_root()
    return root / cache_key[:2] / cache_key


def discover_sglang_site_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemPatchError("pinned sglang package is not installed")
    package = Path(next(iter(spec.submodule_search_locations))).resolve()
    if package.name != "sglang" or not package.is_dir():
        raise SystemPatchError(f"unexpected sglang package location: {package}")
    return package.parent


def installed_sglang_version() -> str:
    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemPatchError("cannot determine installed sglang version") from exc


def system_overlay_identity(
    bundle_path: str | Path,
    *,
    competition_target: str,
    arena_name: str,
    stock_site_root: Path | None = None,
    sglang_version: str | None = None,
) -> SystemOverlayIdentity:
    """Bind an overlay to bundle, policy, arena, builder, and pinned stock tree."""
    from optima.bundle_hash import content_hash
    from optima.dep_policy import tree_hash
    from optima.manifest import load_manifest, resolve_system_patches

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    policy = _require_target_policy(
        manifest, competition_target=competition_target, arena_name=arena_name
    )
    version = sglang_version if sglang_version is not None else installed_sglang_version()
    if version != policy.sglang_version:
        raise SystemPatchError(
            f"installed sglang {version!r} differs from system target pin "
            f"{policy.sglang_version!r}"
        )
    site_root = (
        Path(stock_site_root).resolve()
        if stock_site_root is not None
        else discover_sglang_site_root()
    )
    package = site_root / "sglang"
    if not package.is_dir() or package.is_symlink():
        raise SystemPatchError(f"pinned sglang package tree missing/unsafe: {package}")
    patches = resolve_system_patches(bundle, manifest)
    builder = Path(__file__).resolve()
    payload = {
        "schema": SYSTEM_OVERLAY_SCHEMA,
        "bundle_hash": content_hash(bundle),
        "competition_target": competition_target,
        "arena_name": arena_name,
        "system": {
            "target": manifest.system.target,
            "region": manifest.system.region,
            "patches": list(manifest.system.patches),
        },
        "patch_shas": {
            rel: _sha256_file(path)
            for rel, path in zip(manifest.system.patches, patches)
        },
        "policy": _policy_payload(policy),
        "sglang_version": version,
        "stock_sglang_tree_sha256": tree_hash(package),
        "builder_sha256": _sha256_file(builder),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return SystemOverlayIdentity(
        cache_key=hashlib.sha256(encoded).hexdigest(),
        payload=payload,
        stock_site_root=site_root,
    )


def _parsed_patch_set(
    bundle: Path,
    manifest,
    policy: SystemTargetPolicy,
    *,
    stock_site_root: Path,
):
    from optima.deppatch import parse_patch_text

    parsed: list[tuple[str, tuple]] = []
    touched: set[str] = set()
    for rel in manifest.system.patches:
        patch_file = bundle / rel
        file_patches = parse_patch_text(patch_file.read_text(encoding="utf-8"))
        for fp in file_patches:
            original_source = None
            pinned_file = stock_site_root / fp.path
            if pinned_file.is_file() and not pinned_file.is_symlink():
                try:
                    original_source = pinned_file.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise SystemPatchError(
                        f"pinned patch target is not UTF-8 text: {fp.path!r}"
                    ) from exc
            validate_file_patch(policy, fp, original_source=original_source)
            if fp.path in touched:
                raise SystemPatchError(
                    f"system file {fp.path!r} is touched by more than one declared "
                    "patch; use one exact file diff"
                )
            touched.add(fp.path)
        parsed.append((rel, file_patches))
    return tuple(parsed)


_STAMP_KEYS = frozenset({
    "schema", "cache_key", "identity", "patch_files", "touched_files",
    "overlay_tree_sha256",
})


def _validate_overlay_dir(
    dest: Path,
    identity: SystemOverlayIdentity,
) -> tuple[bool, str, dict | None]:
    from optima.dep_policy import tree_hash

    if dest.is_symlink() or not dest.is_dir():
        return False, "overlay directory missing or is a symlink", None
    stamp_path = dest / "overlay.json"
    if stamp_path.is_symlink() or not stamp_path.is_file():
        return False, "overlay stamp missing or is a symlink", None
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"overlay stamp unreadable: {exc}", None
    if not isinstance(stamp, dict) or set(stamp) != _STAMP_KEYS:
        return False, "overlay stamp schema mismatch", stamp if isinstance(stamp, dict) else None
    if stamp.get("schema") != SYSTEM_OVERLAY_SCHEMA:
        return False, "overlay stamp version mismatch", stamp
    if stamp.get("cache_key") != identity.cache_key:
        return False, "overlay cache key mismatch", stamp
    if stamp.get("identity") != identity.payload:
        return False, "overlay identity payload mismatch", stamp
    site = dest / "site"
    package = site / "sglang"
    if site.is_symlink() or package.is_symlink() or not package.is_dir():
        return False, "overlay site/package tree missing or unsafe", stamp
    try:
        actual_tree = tree_hash(package)
    except RuntimeError as exc:
        return False, str(exc), stamp
    if stamp.get("overlay_tree_sha256") != actual_tree:
        return False, "overlay package tree differs from stamp", stamp
    touched = stamp.get("touched_files")
    if not isinstance(touched, dict):
        return False, "overlay touched-file map malformed", stamp
    for rel, expected_sha in touched.items():
        if not isinstance(rel, str) or not isinstance(expected_sha, str):
            return False, "overlay touched-file map malformed", stamp
        path = site / rel
        if path.is_symlink() or not path.is_file():
            return False, f"overlay touched file missing/unsafe: {rel}", stamp
        if _sha256_file(path) != expected_sha:
            return False, f"overlay touched file hash mismatch: {rel}", stamp
    return True, "", stamp


def materialize_system_overlay(
    bundle_path: str | Path,
    *,
    competition_target: str,
    arena_name: str,
    cache_root: Path | None = None,
    stock_site_root: Path | None = None,
    sglang_version: str | None = None,
) -> Path:
    """Trusted-build operation: copy pinned SGLang and apply exact allowed diffs."""
    import fcntl

    from optima.deppatch import apply_file_patch
    from optima.dep_policy import tree_hash
    from optima.manifest import load_manifest

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    policy = _require_target_policy(
        manifest, competition_target=competition_target, arena_name=arena_name
    )
    identity = system_overlay_identity(
        bundle,
        competition_target=competition_target,
        arena_name=arena_name,
        stock_site_root=stock_site_root,
        sglang_version=sglang_version,
    )
    parsed = _parsed_patch_set(
        bundle,
        manifest,
        policy,
        stock_site_root=identity.stock_site_root,
    )
    dest = system_overlay_base(identity.cache_key, cache_root=cache_root)
    lock_path = dest.parent / ".overlay.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        valid, _why, _stamp = _validate_overlay_dir(dest, identity)
        if valid:
            return dest
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink():
                dest.unlink()
            else:
                shutil.rmtree(dest)

        tmp = Path(tempfile.mkdtemp(prefix=".system.", dir=dest.parent))
        try:
            site = tmp / "site"
            site.mkdir()
            # Interpreter caches are not package source and can cause a same-size,
            # same-second patched .py to execute stale bytecode.  Everything else,
            # including trusted package data and native helpers, is copied.
            shutil.copytree(
                identity.stock_site_root / "sglang",
                site / "sglang",
                symlinks=False,
                ignore=shutil.ignore_patterns(*SYSTEM_OVERLAY_IGNORED_NAMES),
            )
            touched: dict[str, str] = {}
            for _patch_rel, file_patches in parsed:
                for fp in file_patches:
                    target = site / fp.path
                    original = None
                    if target.exists():
                        if target.is_symlink() or not target.is_file():
                            raise SystemPatchError(
                                f"system patch target is not a regular file: {fp.path!r}"
                            )
                        original = target.read_text(encoding="utf-8")
                    try:
                        new_text = apply_file_patch(original, fp)
                    except Exception as exc:
                        raise SystemPatchError(str(exc)) from exc
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(new_text, encoding="utf-8")
                    if target.suffix == ".py":
                        try:
                            compile(new_text, fp.path, "exec")
                        except SyntaxError as exc:
                            raise SystemPatchError(
                                f"patched Python is not syntactically valid: {fp.path}: {exc}"
                            ) from exc
                    touched[fp.path] = _sha256_file(target)
            stamp = {
                "schema": SYSTEM_OVERLAY_SCHEMA,
                "cache_key": identity.cache_key,
                "identity": identity.payload,
                "patch_files": list(manifest.system.patches),
                "touched_files": touched,
                "overlay_tree_sha256": tree_hash(site / "sglang"),
            }
            (tmp / "overlay.json").write_text(
                json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.rename(tmp, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    valid, why, _stamp = _validate_overlay_dir(dest, identity)
    if not valid:
        raise SystemPatchError(f"new system overlay failed integrity check: {why}")
    return dest


def _mount_is_read_only(path: Path) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def read_validated_system_overlay(
    bundle_path: str | Path,
    *,
    competition_target: str,
    arena_name: str,
    cache_root: Path | None = None,
    stock_site_root: Path | None = None,
    sglang_version: str | None = None,
    require_read_only: bool = False,
    read_only_check: Callable[[Path], bool] = _mount_is_read_only,
) -> tuple[SystemOverlayIdentity, dict, Path]:
    """Side-effect-free candidate read: verify identity, stamp, bytes, and mount."""
    identity = system_overlay_identity(
        bundle_path,
        competition_target=competition_target,
        arena_name=arena_name,
        stock_site_root=stock_site_root,
        sglang_version=sglang_version,
    )
    dest = system_overlay_base(identity.cache_key, cache_root=cache_root)
    valid, why, stamp = _validate_overlay_dir(dest, identity)
    if not valid or stamp is None:
        raise SystemPatchError(
            f"trusted prebuilt system overlay missing/stale for {identity.cache_key}: {why}"
        )
    if require_read_only and not read_only_check(dest):
        raise SystemPatchError(
            f"candidate system overlay is not on a read-only mount: {dest}"
        )
    return identity, stamp, dest


_SYSTEM_ENV_KEYS = (
    "OPTIMA_SYSTEM_OVERLAY_ARMED",
    "OPTIMA_SYSTEM_BUNDLE_PATH",
    "OPTIMA_SYSTEM_COMPETITION_TARGET",
    "OPTIMA_SYSTEM_ARENA",
    "OPTIMA_SYSTEM_OVERLAY_ROOT",
    "OPTIMA_SYSTEM_EXPECTED_CACHE_KEY",
    "OPTIMA_SYSTEM_DRIVER_PID",
    "OPTIMA_SYSTEM_PROCESS_ROLE",
    "OPTIMA_SYSTEM_ROLE_PARENT_PID",
)


def prebuild_system_overlay(
    bundle_path: str | Path,
    *,
    competition_target: str,
    arena_name: str,
    cache_root: Path | None = None,
    timeout_s: float = 1800.0,
) -> Path:
    """Build in a disposable trusted subprocess, then independently revalidate."""
    bundle = Path(bundle_path).resolve()
    root = Path(cache_root).resolve() if cache_root is not None else _system_overlay_root()
    cmd = [
        sys.executable, "-m", "optima.system_patch", "build",
        "--bundle", str(bundle),
        "--competition-target", competition_target,
        "--arena", arena_name,
        "--cache-root", str(root),
    ]
    child_env = os.environ.copy()
    child_env.update(OPTIMA_ACTIVE="0", OPTIMA_BUNDLE_PATH="")
    for key in _SYSTEM_ENV_KEYS:
        child_env.pop(key, None)
    try:
        subprocess.run(cmd, env=child_env, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise SystemPatchError(
            f"system overlay prebuild exceeded {timeout_s:g}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemPatchError(
            f"system overlay prebuild failed with exit code {exc.returncode}"
        ) from exc
    _identity, _stamp, dest = read_validated_system_overlay(
        bundle,
        competition_target=competition_target,
        arena_name=arena_name,
        cache_root=root,
    )
    return dest


def system_launch_environment(
    bundle_path: str | Path,
    *,
    competition_target: str,
    arena_name: str,
    cache_root: Path | None = None,
) -> dict[str, str]:
    """Controller hook: validate prebuild and arm scheduler-role bootstrap.

    The returned environment must be applied only around construction of the
    candidate engine.  It contains no direct overlay path: scheduler bootstrap
    independently derives and validates the content-addressed location.
    """
    root = Path(cache_root).resolve() if cache_root is not None else _system_overlay_root()
    identity, _stamp, _dest = read_validated_system_overlay(
        bundle_path,
        competition_target=competition_target,
        arena_name=arena_name,
        cache_root=root,
    )
    return {
        "OPTIMA_SYSTEM_OVERLAY_ARMED": "1",
        "OPTIMA_SYSTEM_BUNDLE_PATH": str(Path(bundle_path).resolve()),
        "OPTIMA_SYSTEM_COMPETITION_TARGET": competition_target,
        "OPTIMA_SYSTEM_ARENA": arena_name,
        "OPTIMA_SYSTEM_OVERLAY_ROOT": str(root),
        "OPTIMA_SYSTEM_EXPECTED_CACHE_KEY": identity.cache_key,
        "OPTIMA_SYSTEM_DRIVER_PID": str(os.getpid()),
        # Role is deliberately absent.  The validator-owned multiprocessing hook
        # sets it only while spawning SGLang's exact scheduler target.
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m optima.system_patch")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--bundle", required=True)
    build.add_argument("--competition-target", required=True)
    build.add_argument("--arena", required=True)
    build.add_argument("--cache-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        dest = materialize_system_overlay(
            args.bundle,
            competition_target=args.competition_target,
            arena_name=args.arena,
            cache_root=Path(args.cache_root),
        )
        print(dest, flush=True)
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess hook
    raise SystemExit(_main())
