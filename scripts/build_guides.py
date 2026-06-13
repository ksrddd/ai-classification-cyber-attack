"""build_guides.py -- render docs/*_guide.md into matching PDFs.

Self-contained minimal-Markdown -> PDF converter using reportlab. Handles
the subset of Markdown used in our guides:

    - # / ## / ### headings
    - ``` fenced code blocks
    - | table | rows | (with header separator |---|---|)
    - - bullet lists
    - 1. numbered lists
    - regular paragraphs
    - inline `code` (rendered as monospace span)

Uses Tahoma (ships with Windows) so Thai characters render correctly.
On non-Windows systems, falls back to the default Helvetica (Thai chars
will display as boxes -- run on Windows for the canonical PDFs).

Usage
-----
    python scripts/build_guides.py
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.colors import HexColor, lightgrey
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Preformatted,
    PageBreak,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


# ---------------------------------------------------------------------------
# Font registration -- Tahoma supports Thai on Windows
# ---------------------------------------------------------------------------
def register_thai_fonts() -> tuple[str, str, str]:
    """Register Thai-capable fonts and return (regular, bold, mono).

    - Tahoma / Tahoma-Bold: proportional, supports Thai. Ships with Windows.
    - Consolas: monospace, supports Thai on Win10+. Used for code blocks
      and inline `code` so Thai comments inside code render correctly.

    Falls back to Helvetica/Courier on non-Windows (Thai will not render).
    """
    win_fonts = Path("C:/Windows/Fonts")
    tahoma = win_fonts / "tahoma.ttf"
    tahoma_bd = win_fonts / "tahomabd.ttf"
    consolas = win_fonts / "consola.ttf"
    if tahoma.exists() and tahoma_bd.exists():
        pdfmetrics.registerFont(TTFont("Tahoma", str(tahoma)))
        pdfmetrics.registerFont(TTFont("Tahoma-Bold", str(tahoma_bd)))
        reg, bold = "Tahoma", "Tahoma-Bold"
    else:
        reg, bold = "Helvetica", "Helvetica-Bold"
    if consolas.exists():
        pdfmetrics.registerFont(TTFont("Consolas", str(consolas)))
        mono = "Consolas"
    else:
        mono = "Courier"
    return reg, bold, mono


# ---------------------------------------------------------------------------
# Markdown parser -- line-based state machine
# ---------------------------------------------------------------------------
class MdBlock:
    """One renderable block: heading / paragraph / code / table / list."""
    def __init__(self, kind: str, lines: list[str], level: int = 0):
        self.kind = kind     # "h1" "h2" "h3" "p" "code" "table" "ul" "ol"
        self.lines = lines
        self.level = level


def parse_markdown(text: str) -> list[MdBlock]:
    """Return a list of MdBlock structures from the given markdown text."""
    blocks: list[MdBlock] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # blank line: skip
        if not stripped:
            i += 1
            continue

        # fenced code block
        if stripped.startswith("```"):
            j = i + 1
            buf: list[str] = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            blocks.append(MdBlock("code", buf))
            i = j + 1
            continue

        # heading
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            blocks.append(MdBlock(f"h{level}", [m.group(2).strip()], level=level))
            i += 1
            continue

        # table (line starts with |)
        if stripped.startswith("|"):
            buf = [stripped]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                buf.append(lines[j].strip())
                j += 1
            blocks.append(MdBlock("table", buf))
            i = j
            continue

        # bullet list
        if re.match(r"^[-*]\s+", stripped):
            buf = [re.sub(r"^[-*]\s+", "", stripped)]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if re.match(r"^[-*]\s+", nxt):
                    buf.append(re.sub(r"^[-*]\s+", "", nxt))
                    j += 1
                else:
                    break
            blocks.append(MdBlock("ul", buf))
            i = j
            continue

        # numbered list
        if re.match(r"^\d+\.\s+", stripped):
            buf = [re.sub(r"^\d+\.\s+", "", stripped)]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if re.match(r"^\d+\.\s+", nxt):
                    buf.append(re.sub(r"^\d+\.\s+", "", nxt))
                    j += 1
                else:
                    break
            blocks.append(MdBlock("ol", buf))
            i = j
            continue

        # horizontal rule (--- on its own line) -> skip; visual only in MD
        if stripped == "---":
            i += 1
            continue

        # paragraph -- consume contiguous non-blank lines
        buf = [stripped]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if (not nxt) or nxt.startswith(("#", "```", "|", "-", "*"))\
                    or re.match(r"^\d+\.\s+", nxt):
                break
            buf.append(nxt)
            j += 1
        blocks.append(MdBlock("p", [" ".join(buf)]))
        i = j
    return blocks


# ---------------------------------------------------------------------------
# Inline formatting: `code` -> monospace span; **bold** -> bold
# ---------------------------------------------------------------------------
def render_inline(text: str, mono_font: str = "Consolas") -> str:
    """Escape XML special chars then re-apply inline markup as Paragraph tags."""
    out = (text
           .replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;"))
    # bold **text**
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    # inline code `text` -- monospace font that supports Thai
    out = re.sub(r"`([^`]+?)`",
                 rf'<font name="{mono_font}" size="9">\1</font>',
                 out)
    return out


# ---------------------------------------------------------------------------
# Render blocks -> reportlab flowables
# ---------------------------------------------------------------------------
def build_styles(regular: str, bold: str, mono: str) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "Base", fontName=regular, fontSize=10.5, leading=15,
        textColor=HexColor("#1f2937"),
        spaceAfter=4,
    )
    return {
        "h1": ParagraphStyle("h1", parent=base, fontName=bold, fontSize=20,
                             leading=26, spaceBefore=6, spaceAfter=12,
                             textColor=HexColor("#0f172a")),
        "h2": ParagraphStyle("h2", parent=base, fontName=bold, fontSize=15,
                             leading=20, spaceBefore=14, spaceAfter=8,
                             textColor=HexColor("#0f172a")),
        "h3": ParagraphStyle("h3", parent=base, fontName=bold, fontSize=12,
                             leading=16, spaceBefore=10, spaceAfter=6,
                             textColor=HexColor("#334155")),
        "p":  ParagraphStyle("p",  parent=base, spaceAfter=6),
        "li": ParagraphStyle("li", parent=base, leftIndent=14, bulletIndent=2,
                             spaceAfter=2),
        "code": ParagraphStyle("code", parent=base, fontName=mono,
                               fontSize=9, leading=12, leftIndent=8,
                               backColor=HexColor("#f1f5f9"),
                               borderColor=HexColor("#cbd5e1"),
                               borderWidth=0.5, borderPadding=6,
                               textColor=HexColor("#0f172a")),
        "table_cell": ParagraphStyle("tc", parent=base, fontSize=9,
                                     leading=12, spaceAfter=0),
        "table_head": ParagraphStyle("th", parent=base, fontName=bold,
                                     fontSize=9, leading=12, spaceAfter=0,
                                     textColor=HexColor("#0f172a")),
    }


def split_table_row(row: str) -> list[str]:
    """| a | b | c | -> ['a', 'b', 'c']"""
    cells = row.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def render_table(block: MdBlock, styles: dict[str, ParagraphStyle]) -> Table:
    """Convert a Markdown table block to a reportlab Table."""
    # block.lines[0] = header, [1] = separator |---|---|, [2:] = body
    header = split_table_row(block.lines[0])
    body_rows = [split_table_row(r) for r in block.lines[2:]]

    data = []
    data.append([Paragraph(render_inline(c), styles["table_head"]) for c in header])
    for row in body_rows:
        # Pad short rows
        while len(row) < len(header):
            row.append("")
        data.append([Paragraph(render_inline(c), styles["table_cell"])
                     for c in row[:len(header)]])

    # Equal column widths fitting within page text width
    n_cols = max(1, len(header))
    page_w, _ = A4
    avail = page_w - 4 * cm
    col_w = avail / n_cols

    tbl = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e2e8f0")),
        ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#94a3b8")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return tbl


def render_blocks(blocks: list[MdBlock], styles: dict[str, ParagraphStyle]):
    """Yield reportlab flowables from a list of MdBlocks."""
    for b in blocks:
        if b.kind in ("h1", "h2", "h3"):
            yield Paragraph(render_inline(b.lines[0]), styles[b.kind])
        elif b.kind == "p":
            yield Paragraph(render_inline(b.lines[0]), styles["p"])
        elif b.kind == "code":
            # Preformatted preserves whitespace + monospace.
            text = "\n".join(b.lines) if b.lines else ""
            yield Preformatted(text, styles["code"])
            yield Spacer(1, 4)
        elif b.kind == "ul":
            for item in b.lines:
                yield Paragraph("• " + render_inline(item), styles["li"])
            yield Spacer(1, 4)
        elif b.kind == "ol":
            for n, item in enumerate(b.lines, start=1):
                yield Paragraph(f"{n}. " + render_inline(item), styles["li"])
            yield Spacer(1, 4)
        elif b.kind == "table":
            yield render_table(b, styles)
            yield Spacer(1, 6)


def render_pdf(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    blocks = parse_markdown(text)

    regular, bold, mono = register_thai_fonts()
    styles = build_styles(regular, bold, mono)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=md_path.stem.replace("_", " ").title(),
        author="Cyber Attack Classification project",
    )
    flow = list(render_blocks(blocks, styles))
    doc.build(flow)


def main() -> int:
    guides = sorted(DOCS_DIR.glob("*_guide.md"))
    if not guides:
        print(f"No *_guide.md found under {DOCS_DIR}")
        return 1
    for md in guides:
        pdf = md.with_suffix(".pdf")
        print(f"Rendering {md.name} -> {pdf.name}")
        render_pdf(md, pdf)
        print(f"  written {pdf.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
