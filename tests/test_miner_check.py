"""Miner diagnostics reuse serving receipts without making a qualification claim."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cacheon import miner_check
from cacheon.cli import build_parser


def _bundle(root, body="def forward(module, *args, **kwargs):\n    return None\n"):
    root.mkdir()
    (root / "entry.py").write_text(body)
    (root / "manifest.toml").write_text(
        'bundle_id="check-test"\nabi_version="cacheon-op-abi-v0"\n'
        '[[ops]]\nslot="model.layers.*.mlp"\nsource="entry.py"\nentry="forward"\n'
    )
    return root


@pytest.mark.parametrize("body, expected", [
    ("def forward(module, x, batch):\n    return x\n", 0),
    ("def forward():\n    return None\n", 2),
    ("forward = 7\n", 2),
])
def test_verify_resolves_real_candidate_imports_in_a_child(tmp_path, body, expected):
    bundle = _bundle(tmp_path / "bundle", body)
    assert miner_check.verify_nodes(str(bundle)) == expected
    assert "cacheon_kernel_entry" not in sys.modules


def test_smoke_checks_prepare_arity_without_executing_a_fake_model(tmp_path, capsys):
    bundle = _bundle(tmp_path / "bundle",
                     "def forward(state, x):\n    return x\n"
                     "def prepare(module, missing):\n    return module\n")
    with (bundle / "manifest.toml").open("a") as handle:
        handle.write('prepare="prepare"\n')
    assert miner_check.verify_nodes(str(bundle)) == 2
    assert "missing" in capsys.readouterr().out


def test_verify_loads_bundle_local_helpers_without_importing_them_in_controller(tmp_path):
    bundle = _bundle(tmp_path / "bundle", "from glm_check_helper import forward\n")
    (bundle / "glm_check_helper.py").write_text(
        "def forward(module, *args, **kwargs):\n    return module.forward(*args, **kwargs)\n"
    )
    assert miner_check.verify_nodes(str(bundle)) == 0
    assert "glm_check_helper" not in sys.modules


def test_child_timeout_reaps_its_owned_process(tmp_path, monkeypatch):
    popen = subprocess.Popen
    children = []

    def sleeping(*_args, **kwargs):
        child = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", sleeping)
    with pytest.raises(subprocess.TimeoutExpired):
        miner_check._child([], tmp_path / "log", timeout=0.02)
    assert children[0].poll() is not None


def _args(tmp_path, model, ranks):
    bundle = _bundle(tmp_path / "bundle")
    (tmp_path / model).mkdir()
    (tmp_path / "engine.json").write_text(json.dumps({"tp_size": ranks, "dtype": "bfloat16"}))
    (tmp_path / "requests.json").write_text(json.dumps([
        {"prompt": ["Explain a loop."], "sampling_params": {"max_new_tokens": 16}},
    ]))
    return build_parser().parse_args([
        "check", str(bundle), "--model", str(tmp_path / model),
        "--engine-config", str(tmp_path / "engine.json"),
        "--requests", str(tmp_path / "requests.json"), "--output", str(tmp_path / "out"),
    ])


@pytest.mark.parametrize("model,ranks", [("qwen", 1), ("glm", 4)])
@pytest.mark.parametrize("windows,violations,expected", [(4, 0, 0), (0, 0, 3), (4, 1, 2)])
def test_check_grades_every_rank_and_stops_before_graph_on_failed_audit(
    tmp_path, monkeypatch, model, ranks, windows, violations, expected, capsys,
):
    args = _args(tmp_path, model, ranks)
    phases = []

    def child(arguments, log, **_kwargs):
        phase, inputs, folder = arguments
        phases.append(phase)
        config = json.loads(Path(inputs).read_text())["config"]
        assert config["model_path"] == str(tmp_path / model)
        assert config["tp_size"] == ranks
        rows = [{"slot": "model.layers.*.mlp", "rank": rank, "world_size": ranks,
                 "pid": 100 + rank, "n": windows, "violations": violations,
                 "worst_frac": 0.0 if violations else 1.0, "min_ratio": 0.75,
                 "mode": "matched_ratio", "compare_errors": 0, "baseline_refused": 0}
                for rank in range(ranks)]
        report = {"decision": "COMPLETED", "receipts": {"audit": rows, "completed": []}}
        Path(folder, "result.json").write_text(json.dumps(report))
        log.write_text("retained engine output")

    monkeypatch.setattr(miner_check, "_child", child)
    assert miner_check.check(args) == expected
    assert phases == (["audit", "graph"] if expected == 0 else ["audit"])
    assert (Path(args.output) / "audit" / "engine.log").read_text() == "retained engine output"
    assert f"model.layers.*.mlp | {ranks - 1} |" in capsys.readouterr().out


def test_check_retains_original_failure_and_does_not_overwrite_a_previous_run(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path, "qwen", 1)

    def failed(arguments, log, **_kwargs):
        log.write_text("original candidate traceback")
        raise RuntimeError("candidate load failed")

    monkeypatch.setattr(miner_check, "_child", failed)
    assert miner_check.check(args) == 2
    assert "candidate load failed" in capsys.readouterr().err
    assert miner_check.check(args) == 2
    assert "FileExistsError" in capsys.readouterr().err
    assert (Path(args.output) / "audit" / "engine.log").read_text() == "original candidate traceback"


@pytest.mark.parametrize("phase", ["audit", "graph"])
def test_engine_uses_real_bootstrap_environment_and_preserves_serving_options(tmp_path, monkeypatch, phase):
    from cacheon import receipts, seam
    from cacheon.eval import engine_worker

    seen = {}

    class Engine:
        def __init__(self, **options):
            seen.update(options=options, audit=os.environ["CACHEON_SLOT_AUDIT"])

        def generate(self, **request):
            seen["request"] = request

        def shutdown(self):
            seen["shutdown"] = True

    fake = SimpleNamespace(Engine=Engine, __version__="test-runtime")
    monkeypatch.setitem(sys.modules, "sglang", fake)
    monkeypatch.setattr(seam, "_IS_DRIVER", False)
    monkeypatch.setattr(engine_worker, "_report_running_engine_failures", lambda _: None)
    monkeypatch.setattr(receipts, "require", lambda *_a, **_k: [{"pid": 100, "slots": ["model.layers.*.mlp"]}])
    monkeypatch.setattr(engine_worker, "_require_execution_completion", lambda *_a, **kw: seen.update(coverage=kw))
    for key in ("CACHEON_ACTIVE", "CACHEON_BUNDLE_PATH", "CACHEON_FRAMEWORK_MODE", "CACHEON_SEAM_RECEIPT_DIR",
                "SGLANG_PLUGINS", "CACHEON_SLOT_AUDIT", "CACHEON_SLOT_AUDIT_SEED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "")
    request = {"input_ids": [[1, 2, 3]], "sampling_params": {"max_new_tokens": 5}}
    miner_check._engine({"model_path": "/model", "tp_size": 1, "mem_fraction_static": 0.93},
                        "/bundle", [request], tmp_path, phase=phase, seed=7)
    assert seen["options"]["mem_fraction_static"] == 0.93
    assert seen["options"].get("disable_cuda_graph", False) == (phase == "audit")
    assert seen["audit"] == ("1" if phase == "audit" else "0")
    assert seen["coverage"]["require_captured"] == (phase == "graph")
    assert seen["request"] == request and seen["shutdown"]
    assert json.loads((tmp_path / "result.json").read_text())["sglang_version"] == "test-runtime"
