"""Social cards preserve source, stock identity and retained score attribution."""

from dataclasses import replace
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image

from dashboard import social
from dashboard.social_image import render_card
from dashboard.sources import DashboardSource, install_sources, selected


@pytest.fixture
def cards(tmp_path, monkeypatch):
    path = tmp_path / "intake.sqlite3"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE settlement_qualifications "
                    "(reservation_id, reproduction_index, qualification_json, attempt_ref_json)")
        for index, gain in enumerate((1.15, 1.068866)):
            qualification = {"speedup": gain, "incumbent_manifest": {"entries": {}, "runtime_digest": "a" * 64},
                             "candidate_manifest": {"runtime_digest": "a" * 64}}
            con.execute("INSERT INTO settlement_qualifications VALUES (?,?,?,?)",
                        ("submission", index, json.dumps(qualification), str(index)))

    def connection():
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        return con

    def detail(reservation, response):
        if reservation not in ("submission", "pending"):
            raise HTTPException(404, "reservation not found")
        source = selected.get()
        return {"reservation_id": reservation, "target_id": "forward_pass", "decision": "PASS" if reservation == "submission" else "",
                "status": "qualified" if reservation == "submission" else "qualifying",
                "competition": source.model if source else "Qwen3.6-35B"}

    def speed(reference, roots, target):
        assert reference == "1"  # Do not splice the faster primary into reproduction's score.
        return {"lanes": [{"role": role, "tokens_per_second": rate} for role, rate in
                          (("B", 1279.4), ("B_prime", 1279.6), ("C", 1367.8))]}

    monkeypatch.setattr(social, "retained_speed", speed)
    monkeypatch.setattr(social, "sglang_build", lambda runtime, source:
                        {"commit": "1234567" + "b" * 33, "version": "v0.5.20"} if source else {})
    api = {"submission_detail": detail, "intake_conn": connection, "evidence_roots": lambda con: (),
           "STATIC_DIR": Path(__file__).parents[1] / "dashboard/static"}
    return api, path


def test_card_uses_one_stock_qualification(cards):
    api, _ = cards
    card = social.submission_card("submission", api)
    assert card.gain == pytest.approx((1367.8 / 1279.6 - 1) * 100)
    assert (card.submission, card.stock) == (1367.8, 1279.6)
    assert "+6.89% throughput improvement over stock SGLang" in card.description
    assert "commit unavailable" in card.description
    assert Image.open(BytesIO(render_card(card))).size == (1200, 630)


@pytest.mark.parametrize("manifest", [{}, {"entries": {}}, {"entries": {"prefix_cache": {}}, "runtime_digest": "a" * 64},
                                      {"entries": {}, "runtime_digest": "b" * 64}])
def test_incumbent_missing_or_other_runtime_is_never_stock(cards, manifest):
    api, path = cards
    with sqlite3.connect(path) as con:
        for index, raw in con.execute("SELECT reproduction_index,qualification_json FROM settlement_qualifications").fetchall():
            q = json.loads(raw)
            q["incumbent_manifest"] = manifest
            con.execute("UPDATE settlement_qualifications SET qualification_json=? WHERE reproduction_index=?",
                        (json.dumps(q), index))
    card = social.submission_card("submission", api)
    assert card.gain is None and card.stock is None
    assert card.submission == 1367.8
    assert "stock SGLang" not in card.description
    assert "submission 1,367.8 tok/s" in card.description


def test_pending_card_has_no_pass_or_fabricated_measurements(cards):
    card = social.submission_card("pending", cards[0])
    assert card.status == "QUALIFYING"
    assert (card.gain, card.stock, card.submission) == (None, None, None)
    assert render_card(card) != render_card(replace(card, status="FAILED · FAIL"))


def test_replay_headline_is_improvement_in_displayed_throughput(cards, monkeypatch):
    speed = {"speedup": 1.068866, "windows": 2, "window_limit": 4,
             "grading": {"required_speedup": 1.01, "detail": "credited replay"},
             "lanes": [{"role": role, "decode_tps": rate, "mean_ttft_s": 1}
                       for role, rate in (("B", 90), ("B", 110), ("C", 120), ("C", 130))]}
    monkeypatch.setattr(social, "retained_speed", lambda *args: speed)
    card = social.submission_card("submission", cards[0])
    assert (card.metric, card.stock, card.submission) == ("DECODE THROUGHPUT", 100, 125)
    assert card.gain == 25
    assert (card.ttft, card.stock_ttft) == (1, 1)
    assert "TTFT 1,000.0 ms" in card.description


def test_single_measurement_layout_and_even_pill_padding():
    card = social.SubmissionCard("submission", "GLM-5.3", "forward_pass", "QUALIFIED · PASS",
                                 submission=104.1, ttft=.123, metric="DECODE THROUGHPUT")
    image = Image.open(BytesIO(render_card(card)))
    pill = image.crop((958, 38, 1147, 79))
    mask = Image.new("L", pill.size)
    mask.putdata([255 if pill.getpixel((x, y))[1] > 100 else 0
                  for y in range(pill.height) for x in range(pill.width)])
    left, top, right, bottom = mask.getbbox()
    assert abs(top - (41 - bottom)) <= 2
    assert left >= 19 and 189 - right >= 19
    assert "stock SGLang" not in card.description and "TTFT 123.0 ms" in card.description
    assert render_card(card) != render_card(replace(card, stock=100, stock_ttft=.15))


@pytest.mark.parametrize("key,slug,model", [("qwen", "qwen3.6", "Qwen3.6-35B"), ("glm", "glm-5.3", 'GLM-5.3 <"test">')])
def test_html_and_png_work_without_javascript_for_paths_and_query_aliases(cards, monkeypatch, key, slug, model):
    monkeypatch.delenv("CACHEON_DASH_SOURCES", raising=False)
    api, _ = cards
    app = FastAPI()

    @app.get("/")
    def index(request: Request):
        return social.submission_page(request, api)

    social.install_social(app, api)
    install_sources(app, {**api, "index": index})
    app.state.dashboard_sources = {key: DashboardSource(key, key, model, {}, {}, False, None)}
    app.state.dashboard_default = key
    client = TestClient(app)
    for url in (f"/{slug}?submission=submission&arena=wrong", f"/?arena={key}&submission=submission",
                f"/?arena={slug}&submission=submission"):
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        tags = {}

        class Metadata(HTMLParser):
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "meta":
                    tags[attrs.get("property", attrs.get("name"))] = attrs.get("content")

        Metadata().feed(response.text)
        assert tags["og:title"] == f"Cacheon · {model} · forward_pass"
        assert tags["og:url"] == f"https://dash.cacheon.ai/{slug}?submission=submission"
        assert tags["twitter:card"] == "summary_large_image"
        assert "SGLang v0.5.20 · 1234567" in tags["og:description"]
        png = client.get(tags["og:image"])
        assert png.headers["content-type"] == "image/png"
        assert png.headers["cache-control"] == "public, max-age=60"
        assert Image.open(BytesIO(png.content)).size == (1200, 630)
    assert client.get(f"/{slug}?submission=missing").status_code == 404
    assert client.get("/api/submissions/missing/preview.png").status_code == 404
    assert selected.get() is None
