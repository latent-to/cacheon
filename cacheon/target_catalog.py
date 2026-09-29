"""Validator-owned contribution target identity.

The manifest answers which implementation rows should be loaded.  This module
answers the separate question: which smallest validator-registered semantic
delta does that set of rows propose to replace?

The catalog is deliberately policy-only.  It contains no score, champion,
settlement, chain, qualification, execution-trust, or whole-serving authority.
Untrusted implementation code still runs as a complete isolated engine; its
execution form does not create or deny a contribution identity.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from typing import Iterable, Mapping

from cacheon._strict import NODE_ADDRESS, require_node_members
from cacheon.manifest import CompetitionEntry, DEFAULT_VARIANT, Manifest
from cacheon.stack_identity import canonical_digest

# Catalog identities are consensus-bearing and survive the product/package
# rename. Existing crowns and evaluation stacks bind these exact domains.
_TARGET_CONTRACT_DOMAIN = "cacheon.target-contract"
_TARGET_CATALOG_DOMAIN = "cacheon.target-catalog"
_TARGET_SPEC_DOMAIN = "cacheon.target-spec"

class TargetKind(str, Enum):
    SLOT = "slot"

# Feature names are validator vocabulary, never miner-selected permissions.
# Dynamic/unknown manifest fields and rebuild steps are still observed, but no
# registered target admits them until validator code names the capability here.
FEATURE_ENTRY = "entry"
FEATURE_VARIANTS = "variants"
FEATURE_PREPARE = "prepare"
FEATURE_SETUP = "setup"
# Retired with the op slots; no manifest can declare it (resolution refuses override
# points), but it stays in the admitted sets because forward_pass's target-spec digest
# binds them.
FEATURE_OVERRIDE = "override"
FEATURE_CUDA_SOURCES = "cuda_sources"
FEATURE_REBUILD_BUILD_CUDA_EXT = "rebuild:build_cuda_ext"

KNOWN_CONTRIBUTION_FEATURES = frozenset(
    {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_SETUP,
        FEATURE_OVERRIDE,
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    }
)

_STANDARD_COMPONENT_FEATURES = frozenset(
    {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_OVERRIDE,
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    }
)
_ID_RE = re.compile(r"^[0-9A-Za-z._\-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_SCHEMA_VERSION = 2
_CATALOG_POLICY_VERSION = "target-catalog.v2"

class TargetCatalogError(ValueError):
    """Validator target policy is internally invalid."""


class TargetResolutionError(ValueError):
    """A bundle cannot resolve to the registered target it requested."""


def _decimal_string(value: object, *, field: str) -> str:
    """Return the frozen, float-free decimal representation used in identity JSON."""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TargetCatalogError(f"{field} must be a finite decimal") from exc
    if not number.is_finite():
        raise TargetCatalogError(f"{field} must be a finite decimal")
    if number == 0:
        return "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


@dataclass(frozen=True)
class CorrectnessContractRef:
    mode: str = "allclose"
    top_k: int = 0
    min_ratio: str = "1"
    min_cosine: str = "0"
    max_rel_norm_err: str = "0"
    min_overlap: str = "0"

    def __post_init__(self) -> None:
        if self.mode not in {
            "allclose",
            "matched_ratio",
            "cosine",
            "topk_overlap",
        }:
            raise TargetCatalogError("correctness mode is not registered")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise TargetCatalogError("correctness top_k must be a non-negative integer")
        for name in (
            "min_ratio",
            "min_cosine",
            "max_rel_norm_err",
            "min_overlap",
        ):
            object.__setattr__(
                self,
                name,
                _decimal_string(getattr(self, name), field=f"correctness {name}"),
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "min_ratio": self.min_ratio,
            "min_cosine": self.min_cosine,
            "max_rel_norm_err": self.max_rel_norm_err,
            "min_overlap": self.min_overlap,
        }


@dataclass(frozen=True)
class ToleranceContractRef:
    dtype: str
    atol: str
    rtol: str

    def __post_init__(self) -> None:
        _simple_id(self.dtype, field="tolerance dtype")
        object.__setattr__(
            self, "atol", _decimal_string(self.atol, field=f"{self.dtype} atol")
        )
        object.__setattr__(
            self, "rtol", _decimal_string(self.rtol, field=f"{self.dtype} rtol")
        )

    def snapshot(self) -> dict[str, str]:
        return {"dtype": self.dtype, "atol": self.atol, "rtol": self.rtol}


@dataclass(frozen=True)
class TargetContractRef:
    """Stdlib-only, versioned identity of one target's serving contract."""

    schema_version: int
    slot_id: str
    kind: str
    entry: str
    prepare: str | None
    graph_dynamic_inputs: tuple[str, ...]
    input_abi_id: str
    output_abi_id: str
    reference_id: str
    verification_profile_id: str
    binding_family_id: str
    correctness: CorrectnessContractRef
    tolerances: tuple[ToleranceContractRef, ...]
    kl_threshold: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise TargetCatalogError("target contract schema_version must be 1")
        _simple_id(self.slot_id, field="contract slot_id")
        if self.kind not in {"op", "block", "collective"}:
            raise TargetCatalogError(f"contract {self.slot_id!r} has invalid kind")
        if not isinstance(self.entry, str) or not self.entry.isidentifier():
            raise TargetCatalogError(f"contract {self.slot_id!r} entry is invalid")
        if self.prepare is not None and (
            not isinstance(self.prepare, str) or not self.prepare.isidentifier()
        ):
            raise TargetCatalogError(f"contract {self.slot_id!r} prepare is invalid")
        if isinstance(self.graph_dynamic_inputs, str):
            raise TargetCatalogError("graph_dynamic_inputs must be an ordered sequence")
        dynamic = tuple(self.graph_dynamic_inputs)
        if len(set(dynamic)) != len(dynamic) or not all(
            isinstance(name, str) and name.isidentifier() for name in dynamic
        ):
            raise TargetCatalogError(
                f"contract {self.slot_id!r} graph_dynamic_inputs are invalid"
            )
        object.__setattr__(self, "graph_dynamic_inputs", dynamic)
        for name in (
            "input_abi_id",
            "output_abi_id",
            "reference_id",
            "verification_profile_id",
            "binding_family_id",
        ):
            _simple_id(getattr(self, name), field=f"contract {name}")
        if not isinstance(self.correctness, CorrectnessContractRef):
            raise TargetCatalogError("contract correctness must be CorrectnessContractRef")
        tolerances = tuple(self.tolerances)
        if not all(isinstance(row, ToleranceContractRef) for row in tolerances):
            raise TargetCatalogError("contract tolerances must be ToleranceContractRef rows")
        dtype_names = tuple(row.dtype for row in tolerances)
        if dtype_names != tuple(sorted(dtype_names)) or len(set(dtype_names)) != len(
            dtype_names
        ):
            raise TargetCatalogError("contract tolerances must be dtype-sorted and unique")
        object.__setattr__(self, "tolerances", tolerances)
        if self.kl_threshold is not None:
            object.__setattr__(
                self,
                "kl_threshold",
                _decimal_string(self.kl_threshold, field="contract kl_threshold"),
            )

    def snapshot(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "slot_id": self.slot_id,
            "kind": self.kind,
            "entry": self.entry,
            "prepare": self.prepare,
            "graph_dynamic_inputs": list(self.graph_dynamic_inputs),
            "input_abi_id": self.input_abi_id,
            "output_abi_id": self.output_abi_id,
            "reference_id": self.reference_id,
            "verification_profile_id": self.verification_profile_id,
            "binding_family_id": self.binding_family_id,
            "correctness": self.correctness.snapshot(),
            "tolerances": [row.snapshot() for row in self.tolerances],
            "kl_threshold": self.kl_threshold,
        }
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(_TARGET_CONTRACT_DOMAIN, self.snapshot())


@dataclass(frozen=True)
class TargetSpec:
    """One validator-owned reward-unit identity.

    ``node_roots`` opens node addresses. A manifest whose slots all name modules at
    or under one of these roots of the served model resolves to this target, and its
    members are the addresses it declared, not ``members``. Two targets never share
    a node: overlap inside one bundle is refused by address containment, and the
    model and the prefix cache have disjoint roots.
    """

    target_id: str
    kind: TargetKind
    members: tuple[str, ...]
    allowed_features: frozenset[str] = frozenset({FEATURE_ENTRY})
    contract_ref: TargetContractRef | None = None
    node_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node_roots, str):
            object.__setattr__(self, "node_roots", tuple(self.node_roots))
        if not isinstance(self.members, str):
            object.__setattr__(self, "members", tuple(self.members))
        if not isinstance(self.allowed_features, str):
            object.__setattr__(self, "allowed_features", frozenset(self.allowed_features))


@dataclass(frozen=True)
class ResolvedTarget:
    """Canonical proposal identity, independent of manifest row order.

    ``registered=False`` is a discovery result, not a crownability judgment.
    Explicit requests that lie about a registered target fail instead of
    producing an unregistered result.
    """

    target_id: str | None
    kind: TargetKind | None
    members: tuple[str, ...]
    registered: bool
    implicit: bool
    observed_features: frozenset[str]
    features_complete: bool
    reason: str | None = None
    contract_digest: str | None = None

    def require_registered(self) -> "ResolvedTarget":
        if not self.registered:
            raise TargetResolutionError(
                self.reason or "proposal has no registered contribution target"
            )
        return self

    def require_complete_features(self) -> "ResolvedTarget":
        if not self.features_complete:
            raise TargetResolutionError(
                "target identity lacks complete trusted bundle-feature evidence"
            )
        return self


def _simple_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TargetCatalogError(f"{field} must be a non-empty canonical string")
    if not _ID_RE.fullmatch(value):
        raise TargetCatalogError(f"{field} has illegal characters: {value!r}")
    return value


def _semantic_members(manifest: Manifest) -> tuple[str, ...]:
    """Distinct slots in declaration order; variant rows count once."""
    return tuple(dict.fromkeys(op.slot for op in manifest.ops))


def manifest_declared_features(manifest: Manifest) -> frozenset[str]:
    """Derive contribution features from parsed manifest data.

    This does not inspect ``rebuild.json``.  Trusted intake must pass exact
    observed patcher capabilities through ``observed_features``; the catalog
    deliberately does not duplicate the rebuild parser or execute a plan.
    """

    features: set[str] = {FEATURE_ENTRY}
    counts: dict[str, int] = {}
    for op in manifest.ops:
        counts[op.slot] = counts.get(op.slot, 0) + 1
        if op.variant != DEFAULT_VARIANT:
            features.add(FEATURE_VARIANTS)
        if op.prepare is not None:
            features.add(FEATURE_PREPARE)
        if op.setup is not None:
            features.add(FEATURE_SETUP)
        if op.base_kernel is not None or op.override_point is not None:
            raise TargetResolutionError(
                f"{op.slot}: override points are retired; submit the node"
            )
        if op.cuda_sources:
            features.add(FEATURE_CUDA_SOURCES)
        # Unknown op keys are retained by Manifest for forward compatibility.
        # They are observable capabilities, not an implicit permission bypass.
        features.update(f"op_extra:{key}" for key in op.extra)
    if any(count > 1 for count in counts.values()):
        features.add(FEATURE_VARIANTS)
    return frozenset(features)


def _validate_complete_feature_evidence(
    manifest: Manifest, features: frozenset[str]
) -> None:
    """Validate trusted external build evidence for static targets."""

    has_cuda = FEATURE_CUDA_SOURCES in features
    builds_cuda = FEATURE_REBUILD_BUILD_CUDA_EXT in features
    if has_cuda and not builds_cuda:
        raise TargetResolutionError(
            "complete feature evidence has CUDA sources without "
            "rebuild:build_cuda_ext"
        )
    if builds_cuda and not has_cuda:
        raise TargetResolutionError(
            "complete feature evidence selects rebuild:build_cuda_ext "
            "without declared CUDA sources"
        )
    if builds_cuda and not any(
        path.endswith(".cu") for op in manifest.ops for path in op.cuda_sources
    ):
        raise TargetResolutionError(
            "rebuild:build_cuda_ext requires a declared .cu compilation unit"
        )


class TargetCatalog:
    """Immutable, deterministic validator policy for registered targets."""

    def __init__(self, specs: Iterable[TargetSpec]):
        if isinstance(specs, (str, bytes, Mapping)):
            raise TargetCatalogError("target specs must be an iterable of TargetSpec")
        rows = tuple(specs)
        if not rows:
            raise TargetCatalogError("target catalog must not be empty")

        by_id: dict[str, TargetSpec] = {}
        for index, spec in enumerate(rows):
            if not isinstance(spec, TargetSpec):
                raise TargetCatalogError(
                    f"target spec {index} is not a TargetSpec: {type(spec).__name__}"
                )
            target_id = _simple_id(spec.target_id, field="target_id")
            if target_id in by_id:
                raise TargetCatalogError(f"duplicate target ID {target_id!r}")
            if spec.kind is not TargetKind.SLOT:
                raise TargetCatalogError(
                    f"target {target_id!r} kind must be TargetKind"
                )
            if tuple(spec.members) != (target_id,):
                raise TargetCatalogError(
                    f"slot target {target_id!r} must have itself as its sole member"
                )
            if not isinstance(spec.contract_ref, TargetContractRef):
                raise TargetCatalogError(
                    f"slot target {target_id!r} requires a TargetContractRef"
                )
            if spec.contract_ref.slot_id != target_id:
                raise TargetCatalogError(
                    f"slot target {target_id!r} contract_ref names "
                    f"{spec.contract_ref.slot_id!r}"
                )
            roots = spec.node_roots
            if roots != tuple(sorted(set(roots))) or any(
                "*" in root or NODE_ADDRESS.fullmatch(root) is None for root in roots
            ):
                raise TargetCatalogError(
                    f"slot target {target_id!r} node_roots must be sorted unique names"
                )
            if isinstance(spec.allowed_features, str):
                raise TargetCatalogError(
                    f"target {target_id!r} allowed_features must be a set"
                )
            unknown_features = spec.allowed_features - KNOWN_CONTRIBUTION_FEATURES
            if unknown_features:
                raise TargetCatalogError(
                    f"target {target_id!r} allows unknown features "
                    f"{tuple(sorted(unknown_features))!r}"
                )
            if FEATURE_ENTRY not in spec.allowed_features:
                raise TargetCatalogError(
                    f"target {target_id!r} must allow the entry feature"
                )
            if FEATURE_SETUP in spec.allowed_features:
                raise TargetCatalogError(
                    f"target {target_id!r} may not allow engine-wide setup"
                )
            by_id[target_id] = spec

        ordered = dict(sorted(by_id.items()))
        self._by_id = ordered
        self._target_snapshots = {
            target_id: self._build_target_snapshot(spec)
            for target_id, spec in ordered.items()
        }
        self._snapshot = {
            "schema_version": _CATALOG_SCHEMA_VERSION,
            "policy_version": _CATALOG_POLICY_VERSION,
            "targets": [self._target_snapshots[target_id] for target_id in ordered],
        }
        self._digest = canonical_digest(_TARGET_CATALOG_DOMAIN, self._snapshot)

    @staticmethod
    def _build_target_snapshot(spec: TargetSpec) -> dict[str, object]:
        # The relation keys are empty since node targets replaced composed slots;
        # they stay in the identity so retained target-spec digests are unchanged.
        common: dict[str, object] = {
            "target_id": spec.target_id,
            "kind": spec.kind.value,
            "members": list(spec.members),
            "displaces": [],
            "conflicts_with": [],
            "requires": [],
            "allowed_features": sorted(spec.allowed_features),
        }
        if spec.node_roots:  # absent otherwise: existing targets keep their spec digests
            common["node_roots"] = list(spec.node_roots)
        assert spec.contract_ref is not None
        common["contract_ref"] = spec.contract_ref.snapshot()
        common["contract_digest"] = spec.contract_ref.digest
        return common

    def require(self, target_id: str) -> TargetSpec:
        if not isinstance(target_id, str):
            raise TargetResolutionError("target ID must be a string")
        try:
            return self._by_id[target_id]
        except KeyError:
            raise TargetResolutionError(
                f"unknown contribution target {target_id!r}; target IDs are validator-owned"
            ) from None

    def snapshot(self) -> dict[str, object]:
        """Return a fresh canonical JSON projection of complete catalog policy."""
        return deepcopy(self._snapshot)

    @property
    def digest(self) -> str:
        return self._digest

    def target_spec_digest(self, target_id: str) -> str:
        self.require(target_id)
        return canonical_digest(
            _TARGET_SPEC_DOMAIN, self._target_snapshots[target_id]
        )

    def contract_digest(self, target_id: str) -> str:
        self.require(target_id)
        value = self._target_snapshots[target_id]["contract_digest"]
        assert isinstance(value, str)
        return value

    def admitted_members(self, spec: TargetSpec, declared: Iterable[str]) -> tuple[str, ...]:
        """The manifest's own node addresses under ``spec``'s roots, or a resolution error."""

        return require_node_members(
            tuple(declared), roots=spec.node_roots, error=TargetResolutionError
        )

    def admits(self, spec: TargetSpec, members: Iterable[str]) -> bool:
        """Whether a finalized reservation's members are this target's canonical form."""

        members = tuple(members)
        try:
            return self.admitted_members(spec, members) == members
        except TargetResolutionError:
            return False

    def validate_active_targets(self, target_ids: Iterable[str]) -> tuple[str, ...]:
        if isinstance(target_ids, (str, bytes)):
            raise TargetResolutionError("active target IDs must be an iterable")
        active = tuple(target_ids)
        if not all(isinstance(target_id, str) for target_id in active):
            raise TargetResolutionError("active target IDs must be strings")
        if len(set(active)) != len(active):
            raise TargetResolutionError("active target IDs contain duplicates")
        for target_id in active:
            self.require(target_id)
        return tuple(sorted(active))

    def resolve_manifest(
        self,
        manifest: Manifest,
        *,
        observed_features: Iterable[str] | None = None,
    ) -> ResolvedTarget:
        if not isinstance(manifest, Manifest):
            raise TypeError("manifest must be an cacheon.manifest.Manifest")
        features_complete = observed_features is not None
        if isinstance(observed_features, (str, bytes)):
            raise TargetResolutionError("observed_features must be an iterable of strings")
        extra_features = tuple(observed_features or ())
        if not all(isinstance(feature, str) and feature for feature in extra_features):
            raise TargetResolutionError(
                "observed_features must contain non-empty strings"
            )
        features = frozenset(
            set(manifest_declared_features(manifest)) | set(extra_features)
        )
        members_in_manifest = _semantic_members(manifest)
        member_set = frozenset(members_in_manifest)
        request = manifest.competition

        if request is not None:
            if not isinstance(request, CompetitionEntry):
                raise TargetResolutionError(
                    "manifest competition request must be a CompetitionEntry"
                )
            if (
                not isinstance(request.target, str)
                or not _ID_RE.fullmatch(request.target)
                or not isinstance(request.mode, str)
            ):
                raise TargetResolutionError(
                    "manifest competition target/mode are malformed"
                )

        if request is not None and request.mode == "system":
            members = tuple(sorted(member_set))
            return ResolvedTarget(
                target_id=None,
                kind=None,
                members=members,
                registered=False,
                implicit=False,
                observed_features=features,
                features_complete=features_complete,
                reason=(
                    "legacy competition mode 'system' is unregistered; migrate to "
                    "a registered target or the future discovery lane"
                ),
            )

        if request is None:
            # A node bundle's target is the one whose roots hold every address it declared.
            rooted = [
                row for row in self._by_id.values()
                if member_set and row.node_roots and all(
                    any(m == r or m.startswith(r + ".") for r in row.node_roots)
                    for m in member_set
                )
            ]
            spec = rooted[0] if len(rooted) == 1 else None
            if spec is None:
                members = tuple(sorted(member_set))
                return ResolvedTarget(
                    target_id=None,
                    kind=None,
                    members=members,
                    registered=False,
                    implicit=True,
                    observed_features=features,
                    features_complete=features_complete,
                    reason=(
                        f"proposal {manifest.bundle_id!r} has no registered exact target "
                        f"for members {members!r}; classify it for future discovery"
                    ),
                )
            implicit = True
        else:
            if request.mode not in {kind.value for kind in TargetKind}:
                raise TargetResolutionError(
                    f"unknown competition mode {request.mode!r}"
                )
            spec = self.require(request.target)
            implicit = False

        members = self.admitted_members(spec, members_in_manifest)
        unexpected = features - spec.allowed_features
        if unexpected:
            if FEATURE_SETUP in unexpected:
                detail = "engine-wide setup belongs in the fenced discovery lane"
            else:
                detail = "features are not registered for this target"
            raise TargetResolutionError(
                f"target {spec.target_id!r} rejects observed features "
                f"{tuple(sorted(unexpected))!r}: {detail}"
            )
        if features_complete:
            _validate_complete_feature_evidence(manifest, features)
        return ResolvedTarget(
            target_id=spec.target_id,
            kind=spec.kind,
            members=members,
            registered=True,
            implicit=implicit,
            observed_features=features,
            features_complete=features_complete,
        )

    def resolve_intake(
        self,
        manifest: Manifest,
        *,
        observed_features: Iterable[str],
    ) -> ResolvedTarget:
        """Resolve with a required trusted projection of external features."""
        return self.resolve_manifest(
            manifest, observed_features=observed_features
        ).require_registered().require_complete_features()


# The model's forward pass and the scheduler's prefix cache are the two targets a
# bundle may replace; a bundle names the modules or the cache object it swaps in.
# Model and cache replacements are distinct stack entries. Sharing one target
# erased the incumbent kernels when a miner submitted only a cache (2026-09-28).
FORWARD_PASS_TARGET = "forward_pass"
FORWARD_PASS_ROOTS = ("logits_processor", "model")
PREFIX_CACHE_TARGET = "prefix_cache"


def _node_contract(slot_id: str, **fields: object) -> TargetContractRef:
    return TargetContractRef(
        schema_version=1, slot_id=slot_id, kind="block", entry="entry",
        graph_dynamic_inputs=(), kl_threshold=None, **fields,  # type: ignore[arg-type]
    )


@lru_cache(maxsize=1)
def default_target_catalog() -> TargetCatalog:
    contracts = {
        # The numbers are sglang_nodes' row bar and relative floor; the honest
        # twin scales the rest.
        FORWARD_PASS_TARGET: _node_contract(
            FORWARD_PASS_TARGET, prepare="prepare",
            input_abi_id="node.stock-forward-arguments.input.v1",
            output_abi_id="node.stock-forward-result.output.v1",
            reference_id="node.stock-module-in-engine.reference.v1",
            verification_profile_id="node.stock-twin-rows.verify.v1",
            binding_family_id="sglang.named-module.v1",
            correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.75"),
            tolerances=(ToleranceContractRef("bfloat16", "0", "0.02"),),
        ),
        PREFIX_CACHE_TARGET: _node_contract(
            PREFIX_CACHE_TARGET, prepare=None,
            input_abi_id="prefix-cache.runtime-object.input.v1",
            output_abi_id="prefix-cache.runtime-subclass.output.v1",
            reference_id="prefix-cache.engine-state.reference.v1",
            verification_profile_id="prefix-cache.content.verify.v1",
            binding_family_id="sglang.prefix-cache.v1",
            correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="1"),
            tolerances=(),
        ),
    }
    return TargetCatalog(
        TargetSpec(
            target_id=target, kind=TargetKind.SLOT, members=(target,),
            allowed_features=_STANDARD_COMPONENT_FEATURES,
            contract_ref=contracts[target], node_roots=roots,
        )
        for target, roots in ((FORWARD_PASS_TARGET, FORWARD_PASS_ROOTS),
                              (PREFIX_CACHE_TARGET, ("tree_cache",)))
    )
