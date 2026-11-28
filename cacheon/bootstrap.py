"""Version-independent seam loader via a post-import hook.

SGLang's released versions don't all ship the ``sglang.srt.plugins`` framework,
and ``sglang.Engine`` runs the model in a spawned scheduler process. To install
the Cacheon seam reliably in *every* interpreter in the venv — including that
spawned child — we drop a ``.pth`` file in site-packages containing the single
line ``import cacheon.bootstrap``. Python executes ``.pth`` imports at interpreter
startup (in spawned children too), so this module loads everywhere.

At startup sglang is not yet imported, and importing it here would be heavy and
fragile. So instead we register a meta-path finder that defers patching until
a module the seam table names is actually imported, then runs ``seam.activate``
against the freshly-loaded module. ``seam.activate`` is env-driven, so a baseline
process (CACHEON_ACTIVE unset) just installs a pass-through dispatcher.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

# Modules whose import should trigger seam installation — derived from the single seam
# table (cacheon/seams.py), so adding a seam there is the only edit. seams.py is stdlib-only
# (no torch/sglang), safe to import at interpreter startup. seam.activate() installs
# whatever is loaded.
from cacheon.seams import TARGET_MODULES as _TARGETS

_ENTROPY_TARGET = "sglang.srt.layers.logprob_processor"


def _run_activate() -> None:
    from cacheon import seam

    seam.activate()


def _install_teacher_entropy(mod) -> None:
    """Pristine-T-only hook: emit [argmax, -H(p_t)] in k=2 top-logprobs without extra forward passes."""
    if getattr(mod, "_cacheon_entropy_wrapped", False):
        return
    orig = mod.get_top_logprobs_chunk

    def wrapped(
        logprobs, top_k_nums, pruned_lens, top_logprobs_val, top_logprobs_idx,
        split_pruned_len, log_normalizer=None, precomputed_topk=None, copy_to_pinned_cpu=False,
    ):
        if top_k_nums and all(k == 2 for k in top_k_nums) and logprobs.shape[0] > 0:
            import torch

            n = logprobs.shape[0]
            ent = torch.empty(n, dtype=torch.float32, device=logprobs.device)
            for lo in range(0, n, 256):
                hi = min(lo + 256, n)
                if log_normalizer is not None:
                    rm, rls = log_normalizer
                    lp = (logprobs[lo:hi].float() - rm[lo:hi, None]) - rls[lo:hi, None]
                else:
                    lp = logprobs[lo:hi].float()
                ent[lo:hi] = -torch.where(lp > -80.0, lp.exp() * lp, 0.0).sum(dim=-1)
            vals, idxs = precomputed_topk if precomputed_topk is not None else logprobs.topk(2, dim=1)
            if log_normalizer is not None:
                rm, rls = log_normalizer
                vals = (vals.float() - rm[:, None]) - rls[:, None]
            else:
                vals = vals.float()
            vals, idxs = vals.clone(), idxs.clone()
            vals[:, 1] = -ent.clamp_min(0.0)
            idxs[:, 1] = 0
            zeros = torch.zeros(n, dtype=torch.float32, device=logprobs.device)
            return orig(
                logprobs, top_k_nums, pruned_lens, top_logprobs_val, top_logprobs_idx,
                split_pruned_len, log_normalizer=(zeros, zeros), precomputed_topk=(vals, idxs),
                copy_to_pinned_cpu=copy_to_pinned_cpu,
            )
        return orig(
            logprobs, top_k_nums, pruned_lens, top_logprobs_val, top_logprobs_idx,
            split_pruned_len, log_normalizer=log_normalizer, precomputed_topk=precomputed_topk,
            copy_to_pinned_cpu=copy_to_pinned_cpu,
        )

    mod.get_top_logprobs_chunk = wrapped
    mod._cacheon_entropy_wrapped = True


def _wrap_loader(loader, *, entropy: bool = False):
    orig_exec = loader.exec_module

    def exec_module(module):
        orig_exec(module)
        if entropy:
            _install_teacher_entropy(module)
        else:
            _run_activate()

    loader.exec_module = exec_module
    return loader


class _SeamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        is_entropy = fullname == _ENTROPY_TARGET and os.environ.get("CACHEON_TEACHER_ENTROPY") == "1"
        if fullname not in _TARGETS and not is_entropy:
            return None
        # Resolve SGLang through the standard path machinery instead of walking
        # ``sys.meta_path`` ourselves: PathFinder honors the normal filesystem
        # and zip path hooks used by installed SGLang distributions without
        # re-entering this seam finder.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _wrap_loader(spec.loader, entropy=is_entropy)
        return spec


def install() -> None:
    if any(t in sys.modules for t in _TARGETS):
        # Something already imported (e.g. re-entry); patch what's available now.
        _run_activate()
    if os.environ.get("CACHEON_TEACHER_ENTROPY") == "1" and _ENTROPY_TARGET in sys.modules:
        _install_teacher_entropy(sys.modules[_ENTROPY_TARGET])
    if not any(isinstance(f, _SeamFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _SeamFinder())


install()
