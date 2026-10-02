"""Fee displays follow the credit consumed by the selected submission."""

import sqlite3
from dataclasses import replace

import pytest

from tests.test_chain_intake import _arrival, _fingerprint, _publish, _reserve, _store
from tests.test_dashboard_metrics import client  # noqa: F401


@pytest.mark.parametrize("target", ["forward_pass", "prefix_cache"])
@pytest.mark.parametrize("funding", [
    "spent_credit", "available_credit", "other_submission", "other_hotkey",
    "returned_credit", "payment", "unpaid",
])
def test_submission_fee_reads_consumed_credit_without_mutating_it(tmp_path, client, monkeypatch, target, funding):  # noqa: F811
    from dashboard import app

    with _store(tmp_path) as store:
        arrival = _arrival(0, block=9009700)
        if funding == "payment":
            arrival = replace(arrival, payment_block=9009690, payment_extrinsic_index=2)
        reservation_id = _reserve(store, (arrival,), block=9009700)[0].reservation_id
        _publish(store, reservation_id, _fingerprint(target, "member"), digest="a" * 64, root=str(tmp_path))
        path = store.path
    monkeypatch.setattr(app, "DB_PATH", path)
    with sqlite3.connect(path) as con:
        if funding not in ("payment", "unpaid"):
            reservation = {"available_credit": "", "returned_credit": "", "other_submission": "earlier"}.get(funding, reservation_id)
            con.execute("INSERT INTO eval_cost_credits "
                        "(credit_id,hotkey,amount_tao_rao,note,granted_at,reservation_id,spent_block) "
                        "VALUES ('credit',?,500000000,'make-good','2026-10-01',?,?)",
                        ("another-miner" if funding == "other_hotkey" else "miner", reservation,
                         0 if funding in ("available_credit", "returned_credit") else 9009700))
        before = con.execute("SELECT * FROM eval_cost_credits").fetchall()
    expected = {"credit_id": "credit", "amount_tao": .5, "spent_block": 9009700} if funding == "spent_credit" else None
    for endpoint, field in ((f"/api/submissions/{reservation_id}", None), ("/api/submissions", "items"), ("/api/queue", "pending")):
        response = client.get(endpoint)
        assert response.status_code == 200, response.text
        payload = response.json()
        row = payload[field][0] if field else payload
        assert row["fee_credit"] == expected
        if funding == "payment":
            assert row["payment"]["ref"] == "9009690-2"
            assert row["payment"]["links"]["tao_app"].endswith("/blocks/9009690/extrinsics/2")
        else:
            assert row["payment"] is None
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT * FROM eval_cost_credits").fetchall() == before
