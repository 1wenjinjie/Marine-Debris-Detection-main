#!/usr/bin/env python3
"""交底书 Word 精排：在 md_to_docx 通用转换之上叠加正式文档排版。

- 标题：黑体、黑色，分级字号；文档题名居中
- 插图：居中，图题（alt 文本）居中加粗置于图下；块级公式居中
- 正文：两端对齐、首行缩进 2 字符、1.3 倍行距；去除灰色字符分隔线
- 表格：居中、表头加粗浅灰底纹、单元格垂直居中
- 页面：A4，页脚居中页码

用法：python make_pretty_docx.py -i xxx.md -o xxx.docx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SKILL_TOOLS = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "patent-disclosure-skill" / "tools"
)
sys.path.insert(0, str(_SKILL_TOOLS))

import md_to_docx as m  # noqa: E402

from docx.enum.section import WD_SECTION  # noqa: E402,F401
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

BLACK = RGBColor(0, 0, 0)


def _patched_embed_picture(doc, path, *, alt, src, max_w_in, max_h_in, center):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    try:
        dims = m._image_pixel_size(path)
        run = p.add_run()
        run.font.bold = False
        if dims:
            w_in, h_in = m._fit_image_display_inches(
                *dims, max_w_in=max_w_in, max_h_in=max_h_in
            )
            run.add_picture(str(path.resolve()), width=w_in, height=h_in)
        else:
            run.add_picture(str(path.resolve()), width=m.Inches(max_w_in))
    except Exception:
        p.add_run(f"[图片无法嵌入: {alt or src} — {path}]")
        return
    caption = (alt or "").strip()
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(2)
        cp.paragraph_format.space_after = Pt(10)
        run = cp.add_run(caption)
        m._set_run_font(run, "黑体", 9, bold=False)
        run.font.color.rgb = BLACK


def _noop_horizontal_rule(doc):
    pass


_MATH_DPI = 200.0
_MATH_MAX_W_IN = 5.5


def _patched_embed_picture_inline(paragraph, path, *, max_h_in, max_w_in=None):
    """公式 PNG 按渲染 DPI 的天然尺寸嵌入（与 10.5pt 正文同高），不再压成固定 0.17in。"""
    try:
        dims = m._image_pixel_size(path)
        run = paragraph.add_run()
        run.font.bold = False
        if dims:
            px_w, px_h = dims
            w_in = px_w / _MATH_DPI
            h_in = px_h / _MATH_DPI
            if w_in > _MATH_MAX_W_IN:
                h_in *= _MATH_MAX_W_IN / w_in
                w_in = _MATH_MAX_W_IN
            run.add_picture(str(path.resolve()), width=m.Inches(w_in), height=m.Inches(h_in))
        else:
            run.add_picture(str(path.resolve()), height=m.Inches(0.15))
    except Exception:
        paragraph.add_run(f"[行内公式图缺失: {path}]")


def _style_headings(doc):
    specs = {
        "Heading 1": ("黑体", 18, WD_ALIGN_PARAGRAPH.CENTER, 18, 12),
        "Heading 2": ("黑体", 15, None, 14, 8),
        "Heading 3": ("黑体", 13, None, 10, 6),
        "Heading 4": ("黑体", 12, None, 8, 4),
        "Heading 5": ("黑体", 12, None, 8, 4),
    }
    for name, (font, size, align, before, after) in specs.items():
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = BLACK
        rpr = st.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:eastAsia"), font)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing = 1.2
        if align is not None:
            st.paragraph_format.alignment = align
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            level = p.style.name.split()[-1]
            size = {"1": 18, "2": 15, "3": 13}.get(level, 12)
            for run in p.runs:
                m._set_run_font(run, "黑体", size, bold=True)
                run.font.color.rgb = BLACK


def _has_drawing(p) -> bool:
    return bool(p._element.xpath(".//w:drawing"))


def _style_body(doc):
    for p in doc.paragraphs:
        name = p.style.name
        if name.startswith("Heading") or name.startswith("List"):
            continue
        if p.alignment == WD_ALIGN_PARAGRAPH.CENTER:
            continue
        if _has_drawing(p):
            texty = any(r.text.strip() for r in p.runs)
            if not texty:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.first_line_indent = None
                continue
        if not p.text.strip() and not _has_drawing(p):
            continue
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Pt(21)
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(4)


def _set_cell_shading(cell, hex_fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _style_tables(doc):
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                tc_pr = cell._tc.get_or_add_tcPr()
                valign = OxmlElement("w:vAlign")
                valign.set(qn("w:val"), "center")
                tc_pr.append(valign)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    if i == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for run in p.runs:
                            run.font.bold = True
                if i == 0:
                    _set_cell_shading(cell, "F2F2F2")


def _setup_page(doc):
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.8)
        section.right_margin = Cm(2.8)


def _add_page_number_footer(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._element.append(fld_begin)
    run._element.append(instr)
    run._element.append(fld_end)
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"


_BLOCK_LATEX_LINE = re.compile(r"^\s*\\\[.*\\\]\s*$")
_FTHETA_PNG = "math_figures/inline_ftheta.png"


def _preprocess_md(md_text: str, base: Path) -> str:
    """单行 \\[...\\] 后已跟 PNG 注释时删掉裸 LaTeX 行；补渲染无图的 f_theta 单元格。"""
    lines = md_text.splitlines()
    out = []
    for idx, ln in enumerate(lines):
        if _BLOCK_LATEX_LINE.match(ln):
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and m._HIDDEN_MD_IMAGE_COMMENT_RE.match(lines[j].strip()):
                continue
        out.append(ln)
    text = "\n".join(out)

    raw = "\\(f_\\theta(\\cdot)\\)"
    if raw + "<!--" not in text.replace(" ", "") and raw in text:
        png = base / _FTHETA_PNG
        if not png.exists():
            try:
                from math_render import render_latex_to_png
                render_latex_to_png(r"f_\theta(\cdot)", png, dpi=200, fontsize=10.5)
            except Exception as e:
                print(f"[make_pretty_docx] f_theta 渲染失败，保留原文: {e}", file=sys.stderr)
        if png.is_file():
            text = text.replace(
                raw + " |",
                f"<!-- ![公式·行内]({_FTHETA_PNG}) --> |",
            )
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="交底书 Word 精排")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)

    in_path = Path(args.input).resolve()
    base = in_path.parent
    md_text = _preprocess_md(in_path.read_text(encoding="utf-8"), base)

    m._embed_picture = _patched_embed_picture
    m._embed_picture_inline = _patched_embed_picture_inline
    m._add_horizontal_rule = _noop_horizontal_rule

    doc = m.convert_md_to_docx(md_text, base_dir=base)

    _setup_page(doc)
    _style_headings(doc)
    _style_body(doc)
    _style_tables(doc)
    _add_page_number_footer(doc)

    out = Path(args.output).resolve()
    doc.save(str(out))
    print(f"已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
