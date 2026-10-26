from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ST = getSampleStyleSheet()
GRID = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)])


def make_pdf(path: Path, flow_items, title="Fleet price list") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    flow = []
    for item in flow_items:
        if item == "PAGE":
            flow.append(PageBreak())
        elif isinstance(item, tuple):  # (style, text)
            flow.append(Paragraph(item[1], ST[item[0]]))
        else:  # a table: list of rows
            flow.append(Table(item, style=GRID))
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(flow)
    return path


FLEET = [("Title", "Fleet price list 2026"), ("Heading2", "Delivery bikes"),
         ("BodyText", "Fleet customers get volume pricing on the Ebb E-Cargo."),
         [["Bikes", "Price per bike", "Lead time"], ["1-4", "4,950 credits", "3 weeks"],
          ["5-9", "4,700 credits", "4 weeks"], ["10+", "4,450 credits", "6 weeks"]],
         "PAGE", ("Heading2", "Service plans"), ("BodyText", "The fleet service plan costs 300 credits per bike per year.")]


def text_image(lines, size=(1240, 400)):
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default(size=34)
    for i, line in enumerate(lines):
        d.text((40, 40 + i * 70), line, fill=0, font=font)
    return img


def make_scanned_pdf(path: Path, lines) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_image(lines).convert("RGB").save(path, "PDF", resolution=150)
    return path
