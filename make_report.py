"""Build Kabdulla_CP4.pdf from the report text and the saved F1 figure.

From the extracted project: python -m pip install reportlab pillow matplotlib
Then run: python make_report.py
"""

from io import BytesIO
import os
from pathlib import Path
import re
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parent


def register_fonts():
    folder = Path(os.environ.get("SystemRoot", "C:/Windows")) / "Fonts"
    names = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]
    if not all((folder / name).is_file() for name in names):
        from matplotlib import get_data_path
        folder = Path(get_data_path()) / "fonts" / "ttf"
        names = ["DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
                 "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"]
    for face, name in zip(("Paper", "PaperBold", "PaperItalic", "PaperBoldItalic"), names):
        pdfmetrics.registerFont(TTFont(face, str(folder / name)))
    pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="PaperBold",
                                  italic="PaperItalic", boldItalic="PaperBoldItalic")


register_fonts()
TITLE = ParagraphStyle("Title", fontName="PaperBold", fontSize=17, leading=20, spaceAfter=8)
HEADING = ParagraphStyle("Heading", fontName="PaperBold", fontSize=13, leading=16,
                         spaceBefore=5, spaceAfter=4, keepWithNext=True)
BODY = ParagraphStyle("Body", fontName="Paper", fontSize=11.5, leading=15, spaceAfter=7)
NOTE = ParagraphStyle("Note", fontName="PaperItalic", fontSize=9.5, leading=12, spaceAfter=6)


def inline(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def data_table(lines, width):
    rows = [[cell.strip() for cell in line.strip("|").split("|")]
            for line in lines if not re.fullmatch(r"[|:\s-]+", line)]
    widths = {2: (0.60, 0.40), 4: (0.15, 0.25, 0.26, 0.34),
              5: (0.16, 0.18, 0.24, 0.18, 0.24)}[len(rows[0])]
    formatted = []
    for row_num, row in enumerate(rows):
        cells = []
        for col_num, cell in enumerate(row):
            style = ParagraphStyle("Cell", fontName="PaperBold" if row_num == 0 else "Paper",
                                   fontSize=10, leading=12, alignment=TA_LEFT if col_num == 0 else TA_RIGHT)
            cells.append(Paragraph(inline(cell), style))
        formatted.append(cells)
    table = Table(formatted, colWidths=[width * w for w in widths], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
    ]))
    return table


def page_number(canvas, doc):
    canvas.setFont("Paper", 10)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))


def main():
    doc = SimpleDocTemplate(str(ROOT / "Kabdulla_CP4.pdf"), pagesize=A4,
                            leftMargin=22*mm, rightMargin=22*mm, topMargin=18*mm,
                            bottomMargin=18*mm, author="Kabdulla Danial",
                            title="Checkpoint 4 – Capacity penalties in VM placement")
    width = doc.width - 12
    content = [Paragraph("Introduction to Optimization · Checkpoint 4", BODY)]
    lines = (ROOT / "Kabdulla_CP4.md").read_text(encoding="utf-8").splitlines()
    pos = 0
    while pos < len(lines):
        line = lines[pos].strip()
        pos += 1
        if not line or line.startswith("### "):
            continue
        if line == "---":
            content.append(PageBreak())
        elif line.startswith("|"):
            block = [line]
            while pos < len(lines) and lines[pos].startswith("|"):
                block.append(lines[pos])
                pos += 1
            content.extend([data_table(block, width), Spacer(1, 6)])
        elif line.startswith("!["):
            filename = re.search(r"\(([^)]+)\)$", line).group(1)
            stream = BytesIO()
            with PILImage.open(ROOT / filename) as source:
                w, h = source.size
                source.convert("L").save(stream, format="PNG")
            stream.seek(0)
            content.append(Image(stream, width=width, height=width * h / w))
        elif line.startswith("## "):
            content.append(Paragraph(inline(line[3:]), HEADING))
        elif line.startswith("# "):
            content.append(Paragraph(inline(line[2:]), TITLE))
        elif line.startswith("_") and line.endswith("_"):
            content.append(Paragraph(inline(line[1:-1]), NOTE))
        else:
            content.append(Paragraph(inline(line), BODY))
    doc.build(content, onFirstPage=page_number, onLaterPages=page_number)


if __name__ == "__main__":
    main()
