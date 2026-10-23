"""Render docs/PROJECT_REPORT.md to a printable PDF. Helvetica only."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "PROJECT_REPORT.md"
OUTPUT = ROOT / "docs" / "PROJECT_REPORT.pdf"


def ascii(text: str) -> str:
    replacements = {
        "—": "-",
        "–": "-",
        "→": "->",
        "×": "x",
        "²": "2",
        "≥": ">=",
        "≤": "<=",
        "≠": "!=",
        "η": "eta",
        "χ": "chi",
        "λ": "lambda",
        "Σ": "sum",
        "√": "sqrt",
        "🔒": "[lock]",
        "·": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "`": "'",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "PRAVAH project report  -  Koushal Acharya", align="C")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, str(self.page_no()), align="C")


def strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = text.replace("`", "'")
    return ascii(text)


def emit_wrapped(pdf: ReportPDF, text: str, size: int = 10, style: str = "", indent: float = 0) -> None:
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(20, 20, 20)
    usable = pdf.w - pdf.l_margin - pdf.r_margin - indent
    if indent:
        pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(usable, 5, text)


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_left_margin(16)
    pdf.set_right_margin(16)
    in_code = False
    code_buf: list[str] = []
    table_buf: list[str] = []

    def flush_code() -> None:
        nonlocal code_buf
        if not code_buf:
            return
        pdf.set_fill_color(245, 242, 235)
        pdf.set_font("Courier", "", 7.5)
        block = ascii("\n".join(code_buf))
        pdf.multi_cell(0, 3.6, block, fill=True)
        pdf.ln(2)
        code_buf = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        rows = []
        for raw in table_buf:
            cells = [strip_inline(part.strip()) for part in raw.strip("|").split("|")]
            if all(set(cell) <= set("-: ") for cell in cells):
                continue
            rows.append(cells)
        pdf.set_font("Helvetica", "", 8)
        usable = pdf.w - pdf.l_margin - pdf.r_margin
        for index, row in enumerate(rows):
            line = " | ".join(row)
            if index == 0:
                pdf.set_font("Helvetica", "B", 8)
            else:
                pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(usable, 4.2, line)
        pdf.ln(2)
        table_buf = []

    for raw in lines:
        if raw.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_buf.append(raw)
            continue
        if raw.startswith("|"):
            table_buf.append(raw)
            continue
        flush_table()
        if not raw.strip():
            pdf.ln(2)
            continue
        if raw.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, strip_inline(raw[2:]))
            pdf.ln(2)
        elif raw.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 7, strip_inline(raw[3:]))
            pdf.ln(1)
        elif raw.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, strip_inline(raw[4:]))
        elif raw.startswith("#### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 5.5, strip_inline(raw[5:]))
        elif raw.startswith("- "):
            emit_wrapped(pdf, "- " + strip_inline(raw[2:]), 10, indent=4)
        elif re.match(r"\d+\. ", raw):
            emit_wrapped(pdf, strip_inline(raw), 10, indent=4)
        elif raw.startswith("> "):
            emit_wrapped(pdf, strip_inline(raw[2:]), 10, "I")
        else:
            emit_wrapped(pdf, strip_inline(raw), 10)
    flush_code()
    flush_table()
    pdf.output(str(OUTPUT))
    print(OUTPUT, OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
