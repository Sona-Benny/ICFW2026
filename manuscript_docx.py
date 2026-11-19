"""Helpers that turn lightweight markup into a journal-style Word manuscript.

Inline markup understood by add_text():
    **bold**   *italic*   _<subscript>   ^<superscript>   [[1,2]] -> superscript citation
"""
import re
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_FONT = "Times New Roman"
TOKEN = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|_<.+?>|\^<.+?>|\[\[.+?\]\])")
INK = RGBColor(0x0B, 0x0B, 0x0B)


class Manuscript:
    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        self.eq_no = 0
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.3))
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        normal = self.doc.styles["Normal"]
        normal.font.name = BODY_FONT
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        normal.font.size = Pt(11)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15
        for level, size in ((1, 13), (2, 11.5), (3, 11)):
            st = self.doc.styles[f"Heading {level}"]
            st.font.name = BODY_FONT
            st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.italic = level == 3
            st.font.color.rgb = INK
            rfonts = st.element.rPr.rFonts
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rfonts.attrib.pop(qn(attr), None)
            rfonts.set(qn("w:ascii"), BODY_FONT)
            rfonts.set(qn("w:hAnsi"), BODY_FONT)
            st.paragraph_format.space_before = Pt(12 if level == 1 else 8)
            st.paragraph_format.space_after = Pt(4)
            st.paragraph_format.keep_with_next = True
        self._add_page_numbers()

    def _add_page_numbers(self):
        p = self.doc.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)

    # ------------------------------------------------------------------ text
    @staticmethod
    def add_text(par, text, size=None, bold=False, italic=False, colour=None):
        for part in TOKEN.split(text):
            if not part:
                continue
            b, i, sub, sup = bold, italic, False, False
            if part.startswith("**") and part.endswith("**"):
                part, b = part[2:-2], True
            elif part.startswith("[[") and part.endswith("]]"):
                part, sup = part[2:-2], True
            elif part.startswith("_<") and part.endswith(">"):
                part, sub = part[2:-1], True
            elif part.startswith("^<") and part.endswith(">"):
                part, sup = part[2:-1], True
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                part, i = part[1:-1], True
            # nested italics inside sub/superscripts, e.g. _<*u*>
            for k, piece in enumerate(re.split(r"(\*.+?\*)", part)):
                if not piece:
                    continue
                it = i
                if piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
                    piece, it = piece[1:-1], True
                run = par.add_run(piece)
                run.bold = b
                run.italic = it
                run.font.subscript = sub
                run.font.superscript = sup
                if size:
                    run.font.size = Pt(size)
                if colour:
                    run.font.color.rgb = colour
        return par

    def para(self, text, align="justify", size=None, space_after=None, indent=False, style=None):
        p = self.doc.add_paragraph(style=style)
        p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                       "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        if indent:
            p.paragraph_format.first_line_indent = Cm(0.6)
        self.add_text(p, text, size=size)
        return p

    def heading(self, text, level=1):
        h = self.doc.add_heading(level=level)
        self.add_text(h, text)
        return h

    def bullets(self, items):
        for item in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self.add_text(p, item)

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ------------------------------------------------------------------ equations
    def equation(self, latex, width_cm=None, fontsize=13):
        """Rendered display equation with a right-aligned number. Returns the number."""
        self.eq_no += 1
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=fontsize, math_fontfamily="stix")
        buf = BytesIO()
        fig.savefig(buf, dpi=400, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        table = self.doc.add_table(rows=1, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        widths = (Cm(14.4), Cm(2.0))
        for col, cell, w in zip(table.columns, table.rows[0].cells, widths):
            col.width = w  # Word lays out by the grid columns, not only the cell widths
            cell.width = w
        left, right = table.rows[0].cells
        p = left.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img = plt.imread(buf)
        buf.seek(0)
        h_px, w_px = img.shape[:2]
        width = Cm(width_cm) if width_cm else Cm(min(13.5, w_px / 400 * 2.54))
        p.add_run().add_picture(buf, width=width)
        q = right.paragraphs[0]
        q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        q.add_run(f"({self.eq_no})")
        for c in (left, right):
            c.vertical_alignment = 1  # center
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return self.eq_no

    # ------------------------------------------------------------------ figures
    def figure(self, path, caption, width_cm=16.0, prefix="Figure"):
        if prefix == "Figure":
            self.fig_no += 1
            label = f"Figure {self.fig_no}."
        else:
            label = f"{prefix}."
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.space_after = Pt(10)
        self.add_text(c, f"**{label}** ", size=9.5)
        self.add_text(c, caption, size=9.5)
        return self.fig_no

    # ------------------------------------------------------------------ tables
    @staticmethod
    def _border(cell, edges):
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right"):
            el = OxmlElement(f"w:{edge}")
            if edge in edges:
                el.set(qn("w:val"), "single")
                el.set(qn("w:sz"), str(edges[edge]))
                el.set(qn("w:color"), "000000")
            else:
                el.set(qn("w:val"), "nil")
            borders.append(el)
        tc_pr.append(borders)

    def table(self, caption, header, rows, footnote=None, col_widths=None, font_size=8.5, prefix="Table"):
        if prefix == "Table":
            self.tab_no += 1
            label = f"Table {self.tab_no}."
        else:
            label = f"{prefix}."
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.keep_with_next = True
        c.paragraph_format.space_after = Pt(3)
        self.add_text(c, f"**{label}** ", size=9.5)
        self.add_text(c, caption, size=9.5)
        t = self.doc.add_table(rows=len(rows) + 1, cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = col_widths is None
        if col_widths:
            for col, w in zip(t.columns, col_widths):
                col.width = Cm(w)
        n = len(rows)
        for r, values in enumerate([header] + rows):
            for j, value in enumerate(values):
                cell = t.cell(r, j)
                if col_widths:
                    cell.width = Cm(col_widths[j])
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = r < n  # keep short tables on one page
                self.add_text(p, str(value), size=font_size, bold=(r == 0))
                edges = {}
                if r == 0:
                    edges = {"top": 12, "bottom": 6}
                if r == n:
                    edges["bottom"] = 12
                self._border(cell, edges)
        if footnote:
            f = self.doc.add_paragraph()
            f.paragraph_format.space_before = Pt(2)
            self.add_text(f, footnote, size=8.5)
        else:
            self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return self.tab_no

    def landscape_section(self):
        sec = self.doc.add_section()
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
        return sec

    def portrait_section(self):
        sec = self.doc.add_section()
        sec.orientation = WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        return sec

    def save(self, path):
        self.doc.save(str(path))
