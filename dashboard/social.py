"""Submission share metadata from the dashboard's retained evaluation readers."""

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from dashboard.forensics import retained_speed
from dashboard.sources import selected
from dashboard.sglang import sglang_build
from dashboard.winners import result_summary


@dataclass(frozen=True)
class SubmissionCard:
    """Public measurements with explicit attribution for a separate stock run."""

    reservation: str
    model: str
    target: str
    status: str
    submission: float | None = None
    stock: float | None = None
    ttft: float | None = None
    stock_ttft: float | None = None
    metric: str = "DECODE THROUGHPUT"
    commit: str = ""
    version: str = ""
    stock_reference_date: str = ""

    @property
    def comparison(self):
        """A throughput comparison requires a measured, positive stock rate."""
        return self.submission is not None and self.stock is not None and self.stock > 0

    @property
    def gain(self):
        """The percentage corresponds to the throughput numbers printed on this card."""
        return (self.submission / self.stock - 1) * 100 if self.comparison else None

    @property
    def runtime_label(self):
        """Use the recorded upstream build, rather than the current worker version."""
        if not self.commit:
            return "SGLang commit unavailable"
        return "SGLang " + (self.version + " · " if self.version else "@ ") + self.commit[:7]

    @property
    def description(self):
        """Text alternative for crawlers and screen readers."""
        rate = lambda n: f"{n:,.1f} tok/s" if n is not None else "unavailable"
        measurement = f"{self.metric.title()}: submission {rate(self.submission)}"
        if self.ttft is not None:
            measurement += f", TTFT {self.ttft * 1000:,.1f} ms"
        if self.comparison:
            measurement = (f"{self.gain:+.2f}% throughput improvement over stock SGLang. " + measurement
                           + f"; stock SGLang {rate(self.stock)}")
            if self.stock_ttft is not None:
                measurement += f", TTFT {self.stock_ttft * 1000:,.1f} ms"
            if self.stock_reference_date:
                measurement += f". Stock reference {self.stock_reference_date}; separate runs, not a paired qualification"
        return f"{self.model} · {self.target} · {self.status}. {measurement}. {self.runtime_label}."


def stock_reference(qualification, speed):
    """Match an explicitly configured retained stock run to this replay identity."""
    path = os.environ.get("CACHEON_DASH_STOCK_REFERENCES")
    if not path:
        return {}
    candidate = qualification.get("candidate_manifest") or {}
    matches = []
    for entry in json.loads(Path(path).read_text()):
        summary = json.loads(Path(entry["summary"]).read_text())
        inputs = json.loads(Path(entry["inputs"]).read_text())
        stock = inputs["stock_manifest"]
        if stock["_entries"] != [] or summary["kind"] != "stock_sglang_reference":
            raise ValueError("OG stock reference must have an empty contribution stack")
        if (any(not candidate.get(key) or candidate[key] != stock[key]
                for key in ("arena_digest", "runtime_digest", "base_engine_digest"))
                or not speed.get("workload_digest")
                or speed["workload_digest"] != inputs["workload_digest"]
                or speed["workload_digest"] != summary["workload_digest"]
                or speed.get("load") != summary["load"]):
            continue
        rate, ttft = summary["mean_decode_tps"], summary["mean_ttft_s"]
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(ttft) or ttft < 0:
            raise ValueError("OG stock reference measurements must be finite and valid")
        matches.append(dict(stock=rate, stock_ttft=ttft, stock_reference_date=
                            datetime.fromtimestamp(summary["completed_unix"], timezone.utc).date().isoformat()))
    if len(matches) > 1:
        raise ValueError("Multiple OG stock references match this replay; configure one")
    return matches[0] if matches else {}


def submission_card(reservation, api):
    """Read candidate rates from one qualification, with paired or reference stock."""
    detail = api["submission_detail"](reservation, Response())
    rid, target = detail["reservation_id"], detail["target_id"]
    fields = dict(reservation=rid, model=detail["competition"], target=target,
                  status=" · ".join(filter(None, (detail["status"], detail["decision"]))).upper())
    with closing(api["intake_conn"]()) as con:
        rows = con.execute("SELECT qualification_json, attempt_ref_json FROM settlement_qualifications "
                           "WHERE reservation_id=? ORDER BY reproduction_index", (rid,)).fetchall()
        if not rows:
            result = detail.get("result")
            return SubmissionCard(**fields, submission=result["decode_tps"][1] if result else None,
                                  ttft=result["ttft_s"][1] if result else None)
        # Historical paired qualifications retain the lower accepted score.
        row = min(rows, key=lambda r: float(json.loads(r["qualification_json"])["speedup"]))
        qualification = json.loads(row["qualification_json"])
        speed = retained_speed(row["attempt_ref_json"], api["evidence_roots"](con), target)
    incumbent = qualification.get("incumbent_manifest") or {}
    runtime = (qualification.get("candidate_manifest") or {}).get("runtime_digest")
    stock = bool(runtime) and incumbent.get("entries") == {} and incumbent.get("runtime_digest") == runtime
    source = selected.get()
    fields.update(sglang_build(runtime, source))
    result = result_summary(speed)
    if result:
        baseline, candidate = result["decode_tps"]
        fields.update(submission=candidate, stock=baseline if stock else None,
                      ttft=result["ttft_s"][1], stock_ttft=result["ttft_s"][0] if stock else None)
        if not stock and candidate is not None:
            fields.update(stock_reference(qualification, speed))
    return SubmissionCard(**fields)


def submission_page(request: Request, api):
    """Serve crawler-visible metadata in the same HTML used by the interactive app."""
    path = api["STATIC_DIR"] / "index.html"
    reservation = request.query_params.get("submission")
    if not reservation:
        return FileResponse(path)
    card = submission_card(reservation, api)
    source = selected.get()
    slug = source.public()["slug"] if source else ""
    canonical = "https://dash.cacheon.ai/" + slug + "?" + urlencode({"submission": card.reservation})
    query = "?" + urlencode({"arena": source.key}) if source else ""
    image = f"https://dash.cacheon.ai/api/submissions/{card.reservation}/preview.png{query}"
    title = f"Cacheon · {card.model} · {card.target}"
    tags = {"og:type": "website", "og:site_name": "Cacheon", "og:title": title,
            "og:description": card.description, "og:url": canonical, "og:image": image,
            "og:image:type": "image/png", "og:image:width": "1200", "og:image:height": "630",
            "og:image:alt": card.description, "twitter:card": "summary_large_image",
            "twitter:title": title, "twitter:description": card.description,
            "twitter:image": image, "twitter:image:alt": card.description}
    markup = "\n".join(f'<meta {"name" if k.startswith("twitter:") else "property"}="{k}" '
                       f'content="{escape(v, quote=True)}"/>' for k, v in tags.items())
    return HTMLResponse(path.read_text().replace("</head>", markup + "\n</head>"),
                        headers={"Cache-Control": "no-store"})


def install_social(app, api):
    """Register PNGs alongside the existing source-scoped submission API."""
    from dashboard.social_image import render_card

    @app.get("/api/submissions/{reservation_id}/preview.png", include_in_schema=False)
    def preview(reservation_id: str):
        card = submission_card(reservation_id, api)
        return Response(render_card(card), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=60"})
