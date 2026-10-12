"""A failed pooled comparison remains explainable after its worker exits."""

import json

import pytest
import torch

from cacheon import audit
from cacheon.integrations import sglang_nodes as nodes


@pytest.fixture(autouse=True)
def clean_audit(monkeypatch):
    for module, name, value in ((nodes, "_noise", {}), (nodes, "_pooled", {}),
                                 (audit, "_stats", {}), (audit, "_diagnostic_bytes", 0),
                                 (audit, "_diagnostic_limit_reported", False)):
        monkeypatch.setattr(module, name, value)
    monkeypatch.delenv("CACHEON_SEAM_RECEIPT_DIR", raising=False)


def test_failed_window_keeps_prior_call_errors_and_their_own_tolerances(capsys):
    # The October 11 failure was 183/256, pooled across calls. The diagnostic
    # must explain all 256 rows, not just the final call that completed it.
    for passed in (93, 90):
        errors = torch.cat((torch.zeros(passed), torch.ones(128 - passed)))
        nodes._grade("model", 1, [errors], [torch.zeros(128)], node_name="model")
    line = next(x for x in capsys.readouterr().err.splitlines()
                if x.startswith("CACHEON-AUDIT-WINDOW: "))
    event = json.loads(line.partition(": ")[2])
    calls = event["calls"]
    replayed = sum(sum(error <= call["tolerance"] for error in call["row_errors"])
                   for call in calls) / sum(call["rows"] for call in calls)
    assert replayed == event["passing_fraction"] == 183 / 256
    assert event["rows_omitted"] == 0 and event["node"] == "model"
    assert len(calls) == 2 and all(len(call["reference_row_errors"]) == 128 for call in calls)
    assert all(call["reference_p90"] == call["history_p90"] == 0 for call in calls)
    assert audit._stats["model"]["violations"] == 1


def test_diagnostics_bound_rows_and_bytes_without_changing_the_verdict(capsys, monkeypatch):
    nodes._grade("model", 1, [torch.ones(8192)], [torch.zeros(8192)])
    event = json.loads(capsys.readouterr().err.partition(": ")[2])
    assert event["rows_omitted"] == 4096 and event["calls"][0]["rows"] == 8192
    assert event["passing_fraction"] == 0
    monkeypatch.setattr(audit, "_DIAGNOSTIC_LIMIT", 1)
    for _ in range(2):
        nodes._grade("model", 1, [torch.ones(256)], [torch.zeros(256)])
    assert capsys.readouterr().err.count("CACHEON-AUDIT-DIAGNOSTICS-LIMIT:") == 1
    assert (audit._stats["model"]["n"], audit._stats["model"]["violations"]) == (3, 3)


def test_nonfinite_rows_remain_explicit_in_the_diagnostic(capsys):
    nodes._grade("model", 1, [torch.full((256,), float("nan"))], [torch.zeros(256)])
    event = json.loads(capsys.readouterr().err.partition(": ")[2])
    assert event["calls"][0]["row_errors"] == ["nan"] * 256
    assert event["passing_fraction"] == 0 and audit._stats["model"]["compare_errors"] == 0
