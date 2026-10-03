"""Submission share metadata from the dashboard's retained evaluation readers."""

from contextlib import closing
from dataclasses import dataclass
from html import escape
import json
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from dashboard.forensics import retained_speed
from dashboard.sources import selected
from dashboard.sglang import sglang_build
from dashboard.winners import candidate_measurement, result_summary


@dataclass(frozen=True)
class SubmissionCard:
    """Public display fields; stock numbers belong to the same retained attempt."""

    reservation: str
    model: str
    target: str
    status: str
    submission: float | None = None
    stock: float | None = None
    ttft: float | None = None
    stock_ttft: float | None = None
    metric: str = "OUTPUT THROUGHPUT"
    commit: str = ""
    version: str = ""

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
        return f"{self.model} · {self.target} · {self.status}. {measurement}. {self.runtime_label}."


def submission_card(reservation, api):
    """Read rates and TTFT from one retained qualification and its stock identity."""
    detail = api["submission_detail"](reservation, Response())
    rid, target = detail["reservation_id"], detail["target_id"]
    fields = dict(reservation=rid, model=detail["competition"], target=target,
                  status=" · ".join(filter(None, (detail["status"], detail["decision"]))).upper())
    with closing(api["intake_conn"]()) as con:
        rows = con.execute("SELECT qualification_json, attempt_ref_json FROM settlement_qualifications "
                           "WHERE reservation_id=? ORDER BY reproduction_index", (rid,)).fetchall()
        if not rows:
            result = detail.get("result")
            return SubmissionCard(**fields, submission=result["decode_tps"][1] if result else detail.get("tokens_per_second"),
                                  ttft=result["ttft_s"][1] if result else None,
                                  metric="DECODE THROUGHPUT" if result else "OUTPUT THROUGHPUT")
        # Historical paired qualifications retain the lower accepted score.
        row = min(rows, key=lambda r: float(json.loads(r["qualification_json"])["speedup"]))
        qualification = json.loads(row["qualification_json"])
        speed = retained_speed(row["attempt_ref_json"], api["evidence_roots"](con), target)
    incumbent = qualification.get("incumbent_manifest") or {}
    runtime = (qualification.get("candidate_manifest") or {}).get("runtime_digest")
    stock = bool(runtime) and incumbent.get("entries") == {} and incumbent.get("runtime_digest") == runtime
    source = selected.get()
    fields.update(sglang_build(runtime, source))
    if speed:
        result = result_summary(speed)
        if result:
            baseline, candidate = result["decode_tps"]
            fields.update(metric="DECODE THROUGHPUT", submission=candidate, stock=baseline if stock else None,
                          ttft=result["ttft_s"][1], stock_ttft=result["ttft_s"][0] if stock else None)
        else:
            baseline = [lane["tokens_per_second"] for lane in speed["lanes"]
                        if lane["role"] in ("B", "B_prime", "B_double_prime")
                        and lane.get("tokens_per_second") is not None]
            fields.update(submission=candidate_measurement([speed])["tokens_per_second"],
                          stock=max(baseline) if stock and baseline else None)
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
