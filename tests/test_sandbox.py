"""Unit tests for inference_engine.sandbox and inference_engine.runner.

No GPU, no model download required.
Run with: pytest tests/test_sandbox.py -v
"""

import textwrap

import pytest

from inference_engine.sandbox import CheckResult, check
from inference_engine.runner import run_check, _validate_output
from inference_engine.policy import CacheConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_policy(extra_imports: str = "", extra_body: str = "") -> str:
    """Return source for a minimal valid KVCachePolicy subclass."""
    return textwrap.dedent(f"""\
        import torch
        import torch.nn.functional as F
        {extra_imports}
        from inference_engine.policy import KVCachePolicy, CacheConfig, AttentionOutput

        class MyPolicy(KVCachePolicy):
            def setup(self, config):
                self.config = config
                self.k_cache = [None] * config.num_layers
                self.v_cache = [None] * config.num_layers

            def write(self, keys, values, layer_idx, positions):
                if self.k_cache[layer_idx] is None:
                    self.k_cache[layer_idx] = keys
                    self.v_cache[layer_idx] = values
                else:
                    self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], keys], dim=2)
                    self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], values], dim=2)

            def attend(self, query, layer_idx, **kwargs):
                k = self.k_cache[layer_idx]
                v = self.v_cache[layer_idx]
                n_rep = self.config.num_heads // self.config.num_kv_heads
                if n_rep > 1:
                    bsz, nkv, slen, hd = k.shape
                    k = k[:, :, None, :, :].expand(bsz, nkv, n_rep, slen, hd).reshape(bsz, nkv * n_rep, slen, hd)
                    v = v[:, :, None, :, :].expand(bsz, nkv, n_rep, slen, hd).reshape(bsz, nkv * n_rep, slen, hd)
                scale = self.config.head_dim ** -0.5
                attn = torch.matmul(query, k.transpose(-2, -1)) * scale
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
                output = torch.matmul(attn, v)
                return AttentionOutput(output=output, attention_weights=attn)

            def memory_bytes(self):
                total = 0
                for k, v in zip(self.k_cache, self.v_cache):
                    if k is not None:
                        total += k.nelement() * k.element_size()
                    if v is not None:
                        total += v.nelement() * v.element_size()
                return total

            def get_config(self):
                return {{"name": "test"}}
            {extra_body}
    """)


SMALL_CONFIG = CacheConfig(
    num_layers=1,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    max_seq_len=64,
    dtype=None,  # not used by runner (always float32 in worker)
)


# ---------------------------------------------------------------------------
# Layer 1 — AST static analysis
# ---------------------------------------------------------------------------

class TestBlockedImports:
    def test_blocked_import_os(self):
        src = "import os\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked import" in r.reason

    def test_blocked_import_sys(self):
        src = "import sys\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked import" in r.reason

    def test_blocked_from_import(self):
        src = "from os import path\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked from-import" in r.reason

    def test_blocked_import_subprocess(self):
        src = "import subprocess\n" + _minimal_policy()
        r = check(src)
        assert not r.ok

    def test_relative_import_rejected(self):
        src = "from . import foo\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "relative" in r.reason

    def test_relative_dotdot_import_rejected(self):
        src = "from ..utils import bar\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "relative" in r.reason


class TestBlockedCalls:
    def test_blocked_eval(self):
        src = _minimal_policy(extra_body='    def extra(self): eval("1+1")')
        r = check(src)
        assert not r.ok
        assert "eval" in r.reason

    def test_blocked_exec(self):
        src = _minimal_policy(extra_body='    def extra(self): exec("x=1")')
        r = check(src)
        assert not r.ok
        assert "exec" in r.reason

    def test_blocked_open(self):
        src = _minimal_policy(extra_body='    def extra(self): open("/etc/passwd")')
        r = check(src)
        assert not r.ok
        assert "open" in r.reason

    def test_blocked_dunder_import(self):
        src = _minimal_policy(extra_body='    def extra(self): __import__("os")')
        r = check(src)
        assert not r.ok
        assert "__import__" in r.reason


class TestBlockedIntrospectionCalls:
    """getattr/globals/locals/vars etc. are classic AST sandbox escapes."""

    def test_blocked_getattr(self):
        src = _minimal_policy(extra_body='    def extra(self): getattr(self, "k")')
        r = check(src)
        assert not r.ok
        assert "getattr" in r.reason

    def test_blocked_setattr(self):
        src = _minimal_policy(extra_body='    def extra(self): setattr(self, "k", 1)')
        r = check(src)
        assert not r.ok
        assert "setattr" in r.reason

    def test_blocked_delattr(self):
        src = _minimal_policy(extra_body='    def extra(self): delattr(self, "k")')
        r = check(src)
        assert not r.ok
        assert "delattr" in r.reason

    def test_blocked_globals(self):
        src = _minimal_policy(extra_body='    def extra(self): globals()')
        r = check(src)
        assert not r.ok
        assert "globals" in r.reason

    def test_blocked_locals(self):
        src = _minimal_policy(extra_body='    def extra(self): locals()')
        r = check(src)
        assert not r.ok
        assert "locals" in r.reason

    def test_blocked_vars(self):
        src = _minimal_policy(extra_body='    def extra(self): vars()')
        r = check(src)
        assert not r.ok
        assert "vars" in r.reason

    def test_blocked_dir(self):
        src = _minimal_policy(extra_body='    def extra(self): dir(self)')
        r = check(src)
        assert not r.ok
        assert "dir" in r.reason

    def test_getattr_chain_attack_blocked(self):
        src = _minimal_policy(
            extra_body='    def extra(self): getattr(__builtins__, "__import__")("os").system("id")'
        )
        r = check(src)
        assert not r.ok
        assert "getattr" in r.reason

    def test_globals_chain_attack_blocked(self):
        src = _minimal_policy(
            extra_body='    def extra(self): globals()["__builtins__"]["__import__"]("os")'
        )
        r = check(src)
        assert not r.ok
        assert "globals" in r.reason

    def test_method_getattr_not_blocked(self):
        """obj.getattr(...) as a method call must not trigger the bare-call check."""
        src = _minimal_policy(
            extra_body='    def extra(self): self.some_obj.getattr("x")'
        )
        r = check(src)
        assert r.ok, f"method .getattr() falsely blocked: {r.reason}"


class TestMethodCallsNotFalsePositive:
    """Method calls on allowed objects must not be confused with bare
    blocked built-in calls (e.g. torch.compile != compile)."""

    def test_torch_compile_allowed(self):
        src = _minimal_policy(
            extra_body='    def extra(self): return torch.compile(self.attend)'
        )
        r = check(src)
        assert r.ok, f"torch.compile() falsely blocked: {r.reason}"

    def test_model_eval_allowed(self):
        src = _minimal_policy(
            extra_body='    def extra(self): self.model.eval()'
        )
        r = check(src)
        assert r.ok, f"model.eval() falsely blocked: {r.reason}"

    def test_tensor_open_method_allowed(self):
        """Hypothetical .open() method on an allowed object must not trigger."""
        src = _minimal_policy(
            extra_body='    def extra(self): self.file_handle.open()'
        )
        r = check(src)
        assert r.ok, f".open() method falsely blocked: {r.reason}"

    def test_bare_eval_still_blocked(self):
        src = _minimal_policy(extra_body='    def extra(self): eval("1+1")')
        r = check(src)
        assert not r.ok
        assert "eval" in r.reason

    def test_bare_compile_still_blocked(self):
        src = _minimal_policy(extra_body='    def extra(self): compile("x=1", "<>", "exec")')
        r = check(src)
        assert not r.ok
        assert "compile" in r.reason


class TestBlockedAttrs:
    def test_blocked_os_attr(self):
        src = _minimal_policy(extra_body='    def extra(self): os.system("echo hi")')
        r = check(src)
        assert not r.ok
        assert "os" in r.reason

    def test_blocked_dunder_builtins_import(self):
        src = _minimal_policy(
            extra_body='    def extra(self): __builtins__.__import__("os").system("id")'
        )
        r = check(src)
        assert not r.ok
        assert "__builtins__" in r.reason

    def test_blocked_dunder_builtins_getattr_style(self):
        src = _minimal_policy(
            extra_body='    def extra(self): __builtins__.open("/etc/passwd")'
        )
        r = check(src)
        assert not r.ok
        assert "__builtins__" in r.reason


class TestSubmoduleEscapeBlocked:
    """Verify that importing dangerous stdlib objects through inference_engine
    submodules is caught — regression tests for the re-export escape vector."""

    def test_from_runner_import_os(self):
        src = "from inference_engine.runner import os as myos\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked from-import" in r.reason

    def test_import_runner_directly(self):
        src = "import inference_engine.runner\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked import" in r.reason

    def test_from_sandbox_import(self):
        src = "from inference_engine.sandbox import check\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked from-import" in r.reason

    def test_from_harness_import(self):
        src = "from inference_engine.harness import Harness\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked from-import" in r.reason

    def test_from_scoring_import(self):
        src = "from inference_engine.scoring import ScoreResult\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked from-import" in r.reason

    def test_from_passthrough_import(self):
        src = "from inference_engine.passthrough import PassthroughPolicy\n" + _minimal_policy()
        r = check(src)
        assert not r.ok
        assert "blocked from-import" in r.reason

    def test_policy_submodule_still_allowed(self):
        src = _minimal_policy()
        r = check(src)
        assert r.ok, f"legitimate policy import rejected: {r.reason}"

    def test_top_level_ie_import_allowed(self):
        src = "import inference_engine\n" + _minimal_policy()
        r = check(src)
        assert r.ok, f"top-level inference_engine import rejected: {r.reason}"


class TestAllowedImports:
    def test_allowed_torch(self):
        src = _minimal_policy()
        r = check(src)
        assert r.ok

    def test_allowed_numpy(self):
        src = _minimal_policy(extra_imports="import numpy")
        r = check(src)
        assert r.ok

    def test_allowed_math(self):
        src = _minimal_policy(extra_imports="import math")
        r = check(src)
        assert r.ok

    def test_allowed_einops(self):
        src = _minimal_policy(extra_imports="import einops")
        r = check(src)
        assert r.ok

    def test_allowed_torch_submodule(self):
        src = _minimal_policy(extra_imports="import torch.nn")
        r = check(src)
        assert r.ok


class TestStructuralChecks:
    def test_missing_write(self):
        src = textwrap.dedent("""\
            import torch
            from inference_engine.policy import KVCachePolicy, AttentionOutput
            class Bad(KVCachePolicy):
                def setup(self, config): pass
                def attend(self, query, layer_idx, **kw):
                    return AttentionOutput(output=query)
                def memory_bytes(self): return 0
                def get_config(self): return {}
        """)
        r = check(src)
        assert not r.ok
        assert "write" in r.reason

    def test_missing_attend(self):
        src = textwrap.dedent("""\
            import torch
            from inference_engine.policy import KVCachePolicy
            class Bad(KVCachePolicy):
                def setup(self, config): pass
                def write(self, keys, values, layer_idx, positions): pass
                def memory_bytes(self): return 0
                def get_config(self): return {}
        """)
        r = check(src)
        assert not r.ok
        assert "attend" in r.reason

    def test_no_policy_class(self):
        src = textwrap.dedent("""\
            import torch
            class NotAPolicy:
                def setup(self): pass
        """)
        r = check(src)
        assert not r.ok
        assert "no class subclassing KVCachePolicy" in r.reason

    def test_valid_minimal_policy(self):
        r = check(_minimal_policy())
        assert r.ok

    def test_syntax_error_rejected(self):
        r = check("def broken(:\n  pass")
        assert not r.ok
        assert "syntax error" in r.reason


# ---------------------------------------------------------------------------
# Layer 2 — subprocess execution + output validation
# ---------------------------------------------------------------------------

class TestRunnerValidation:
    """Test _validate_output directly with crafted result dicts."""

    def test_valid_result_passes(self):
        result = {
            "output_shape": [1, 4, 1, 8],
            "output_dtype": "torch.float32",
            "output_has_nan": False,
            "output_has_inf": False,
            "output_min": -2.0,
            "output_max": 3.0,
            "memory_bytes": 1024,
            "attn_weights_shape": [1, 4, 1, 4],
            "attn_weights_sum_last_dim": 1.0,
        }
        assert _validate_output(result, SMALL_CONFIG) is None

    def test_nan_rejected(self):
        result = {
            "output_shape": [1, 4, 1, 8],
            "output_dtype": "torch.float32",
            "output_has_nan": True,
            "output_has_inf": False,
            "output_min": -2.0,
            "output_max": 3.0,
            "memory_bytes": 0,
            "attn_weights_shape": None,
        }
        err = _validate_output(result, SMALL_CONFIG)
        assert err is not None
        assert "NaN" in err

    def test_inf_rejected(self):
        result = {
            "output_shape": [1, 4, 1, 8],
            "output_dtype": "torch.float32",
            "output_has_nan": False,
            "output_has_inf": True,
            "output_min": -2.0,
            "output_max": 3.0,
            "memory_bytes": 0,
            "attn_weights_shape": None,
        }
        err = _validate_output(result, SMALL_CONFIG)
        assert err is not None
        assert "Inf" in err

    def test_wrong_shape_rejected(self):
        result = {
            "output_shape": [1, 8, 1, 8],  # wrong num_heads
            "output_dtype": "torch.float32",
            "output_has_nan": False,
            "output_has_inf": False,
            "output_min": -2.0,
            "output_max": 3.0,
            "memory_bytes": 0,
            "attn_weights_shape": None,
        }
        err = _validate_output(result, SMALL_CONFIG)
        assert err is not None
        assert "shape" in err

    def test_out_of_range_rejected(self):
        result = {
            "output_shape": [1, 4, 1, 8],
            "output_dtype": "torch.float32",
            "output_has_nan": False,
            "output_has_inf": False,
            "output_min": -2.0,
            "output_max": 150.0,
            "memory_bytes": 0,
            "attn_weights_shape": None,
        }
        err = _validate_output(result, SMALL_CONFIG)
        assert err is not None
        assert "range" in err

    def test_worker_error_reported(self):
        result = {"error": "something broke"}
        err = _validate_output(result, SMALL_CONFIG)
        assert err is not None
        assert "worker error" in err


class TestRunnerEndToEnd:
    """Full subprocess runs — these take a few seconds each."""

    def test_valid_policy_passes(self):
        r = run_check(_minimal_policy(), SMALL_CONFIG, timeout=60)
        assert r.ok, f"Expected ok=True, got reason: {r.reason}"

    def test_blocked_import_caught_before_execution(self):
        src = "import os\n" + _minimal_policy()
        r = run_check(src, SMALL_CONFIG, timeout=60)
        assert not r.ok
        assert "blocked import" in r.reason

    def test_timeout_kills_subprocess(self):
        """Verify that the subprocess.TimeoutExpired path in run_check works.

        The hang must live inside a method the worker actually calls
        (attend) and must not use imports outside ALLOWED_IMPORTS.
        """
        src = textwrap.dedent("""\
            import torch
            from inference_engine.policy import KVCachePolicy, CacheConfig, AttentionOutput

            class HangPolicy(KVCachePolicy):
                def setup(self, config):
                    self.k = [None] * config.num_layers
                    self.v = [None] * config.num_layers

                def write(self, keys, values, layer_idx, positions):
                    self.k[layer_idx] = keys
                    self.v[layer_idx] = values

                def attend(self, query, layer_idx, **kwargs):
                    while True:
                        pass

                def memory_bytes(self):
                    return 0

                def get_config(self):
                    return {"name": "hang"}
        """)
        r = run_check(src, SMALL_CONFIG, timeout=3)
        assert not r.ok
        assert "timed out" in r.reason
