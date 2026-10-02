"""Render Cacheon's branded 1200×630 submission cards without a browser."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dashboard.social import SubmissionCard

ASSETS = Path(__file__).parent / "static"
FG, MUTED, ACCENT, LINE = "#f4f8f7", "#9aaaa7", "#30d3be", "#193b36"


@lru_cache(maxsize=128)
def render_card(card: SubmissionCard) -> bytes:
    """Cache only fully specified cards so a new evaluation changes the image."""
    image = Image.new("RGB", (1200, 630), "#030a09")
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=24, color=FG, width=1092, anchor=None):
        font = ImageFont.truetype(str(ASSETS / "Ubuntu.ttf"), size)
        while draw.textlength(value, font=font) > width and size > 12:
            size -= 1
            font = ImageFont.truetype(str(ASSETS / "Ubuntu.ttf"), size)
        draw.text((x, y), value, font=font, fill=color, anchor=anchor)

    draw.rectangle((0, 0, 1199, 7), fill=ACCENT)
    with Image.open(ASSETS / "icon-192.png") as original:
        logo = original.convert("RGBA").resize((44, 44), Image.Resampling.LANCZOS)
    image.paste(logo, (54, 36), logo)
    text(110, 35, "Cacheon", 34)
    text(280, 46, "COMPETITIVE INFERENCE OPTIMIZATION", 19, MUTED, width=650)
    draw.rounded_rectangle((958, 38, 1146, 78), 20, fill="#0c302a")
    text(1052, 45, card.status, 18, ACCENT, width=165, anchor="mt")
    text(54, 118, card.model, 53)
    text(57, 186, "TARGET  /  " + card.target.replace("_", " ").upper(), 21, MUTED)
    draw.line((54, 242, 1146, 242), fill=LINE)
    if card.gain is not None:
        text(54, 278, f"{card.gain:+.2f}%", 106, ACCENT, width=510)
        text(59, 400, "improvement over stock SGLang", 23)
    else:
        text(54, 300, "Stock comparison", 38, MUTED, width=510)
        text(54, 350, "unavailable", 38, MUTED)
    text(59, 440, "Retained evaluation result", 20, MUTED)
    draw.line((594, 282, 594, 487), fill=LINE)
    text(640, 282, card.metric, 19, MUTED)
    maximum = max(card.submission or 0, card.stock or 0)
    for label, value, y, color in (("Submission", card.submission, 316, ACCENT),
                                   ("Stock SGLang", card.stock, 403, "#60716e")):
        text(640, y + 12, label, 23)
        text(1146, y, f"{value:,.1f}" if value is not None else "—", 40, width=240, anchor="rt")
        if value is not None:
            text(1060, y + 49, "tok/s", 19, MUTED)
            width = round(407 * value / maximum) if maximum else 0
            if width:
                draw.rounded_rectangle((640, y + 57, 640 + width, y + 67), 5, fill=color)
    draw.line((54, 531, 1146, 531), fill=LINE)
    runtime = "SGLang @ " + card.commit[:7] if card.commit else "SGLang commit unavailable"
    text(54, 560, runtime, 22, MUTED, width=620)
    text(720, 563, "SUBMISSION  " + card.reservation[:12], 19, MUTED)
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
