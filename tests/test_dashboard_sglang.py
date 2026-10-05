"""SGLang labels come from the exact commissioned image, without fixed revisions."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from dashboard import sglang


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    registration = tmp_path / "registration.json"
    record = {"worker_readiness": {"runtime_digest": "runtime"}, "ready_receipt_digest": "ready",
              "pod_host": "worker", "pod_port": 22, "pod_user": "operator", "known_hosts_path": "/known-hosts"}
    registration.write_text(json.dumps(record))
    image = "registry/arena@sha256:" + "a" * 64
    (stage / "ready-receipt.json").write_text(json.dumps({"receipt_digest": "ready", "worker_image": image}))
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps({"ai.sglang.build.commit": "b" * 40,
                                                  "ai.sglang.image.tag": "lmsysorg/sglang:v9.8.7"}))

    monkeypatch.setattr(sglang.subprocess, "run", run)
    sglang._image_build.cache_clear()
    source = SimpleNamespace(values={"REGISTRATION_PATH": registration, "STAGE_ROOT": stage})
    return source, calls, record


def test_fetches_immutable_image_labels_and_caches_them(runtime):
    source, calls, _ = runtime
    expected = {"commit": "b" * 40, "version": "v9.8.7"}
    assert sglang.sglang_build("runtime", source) == expected
    assert sglang.sglang_build("runtime", source) == expected
    assert len(calls) == 1
    args, options = calls[0]
    assert args[-2] == "operator@worker"
    assert "docker image inspect registry/arena@sha256:" + "a" * 64 in args[-1]
    assert options["timeout"] == 5 and "StrictHostKeyChecking=yes" in args
    assert sglang.sglang_build("historical", source) == {}
    assert len(calls) == 1


def test_mismatched_receipt_does_not_inspect_another_image(runtime):
    source, calls, record = runtime
    record["ready_receipt_digest"] = "other"
    source.values["REGISTRATION_PATH"].write_text(json.dumps(record))
    assert sglang.sglang_build("runtime", source) == {}
    assert calls == []


def test_fetch_failure_is_logged_and_retried_after_cache_interval(runtime, monkeypatch, caplog):
    source, _, _ = runtime
    count = []

    def failed(*args, **kwargs):
        count.append(1)
        raise subprocess.TimeoutExpired("ssh", 5)

    monkeypatch.setattr(sglang.subprocess, "run", failed)
    monkeypatch.setattr(sglang.time, "time", lambda: 1000)
    assert sglang.sglang_build("runtime", source) == {}
    assert sglang.sglang_build("runtime", source) == {}
    assert len(count) == 1 and "timed out" in caplog.text
    monkeypatch.setattr(sglang.time, "time", lambda: 1300)
    assert sglang.sglang_build("runtime", source) == {}
    assert len(count) == 2


def test_source_rotation_fetches_new_image_instead_of_stale_commit(runtime, monkeypatch):
    source, calls, record = runtime
    assert sglang.sglang_build("runtime", source)["commit"] == "b" * 40
    record["worker_readiness"]["runtime_digest"] = "new-runtime"
    record["ready_receipt_digest"] = "new-ready"
    source.values["REGISTRATION_PATH"].write_text(json.dumps(record))
    receipt = source.values["STAGE_ROOT"] / "ready-receipt.json"
    receipt.write_text(json.dumps({"receipt_digest": "new-ready", "worker_image": "registry/new@sha256:" + "c" * 64}))
    monkeypatch.setattr(sglang.subprocess, "run", lambda *args, **kw: SimpleNamespace(stdout=json.dumps(
        {"ai.sglang.build.commit": "d" * 40, "ai.sglang.image.tag": "lmsysorg/sglang:v10.0.0"})))
    assert sglang.sglang_build("new-runtime", source) == {"commit": "d" * 40, "version": "v10.0.0"}
    assert sglang.sglang_build("runtime", source) == {}
