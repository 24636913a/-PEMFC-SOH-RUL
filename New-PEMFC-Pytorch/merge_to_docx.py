# -*- coding: utf-8 -*-
"""
将论文 Markdown 章节合并为一个 Word 文档。
处理: 标题、粗体、表格、数学公式、<mark>高亮</mark>、列表
"""
import re
import docx
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

BASE = r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch\文章章节"

FILES = [
    "论文题目.md",
    "论文-摘要.md",
    "论文第一章-引言.md",
    "论文第二章-相关工作.md",
    "论文第三章-数据与预处理.md",
    "论文第四章-方法.md",
    "论文第五章-实验与结果.md",
    "论文第六章-结论.md",
    "论文-参考文献.md",
]

doc = Document()

# --- 全局样式 ---
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# --- 辅助函数 ---

def set_cell_font(cell, text, bold=False, size=Pt(10)):
    """设置表格单元格文字"""
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = size
    run.bold = bold

# 标签配置：(tag_name, open, close, highlight_color)
TAGS = [
    ('mark', '<mark>', '</mark>', 'FFFF00'),   # 审稿内容修订 → 黄色
    ('AI', '<AI>', '</AI>', '92D050'),          # 去AI味语言润色 → 绿色
]

def add_mark_aware_paragraph(doc, text, style_name='Normal', bold=False, font_size=None):
    """
    添加段落，解析 <mark> (黄色) 和 <AI> (绿色) 标签
    同时处理内联 **粗体** 和普通文本
    """
    p = doc.add_paragraph(style=style_name)
    _add_tagged_runs(p, text)

    if bold:
        for run in p.runs:
            run.bold = True
    if font_size:
        for run in p.runs:
            run.font.size = font_size
    return p

def _add_tagged_runs(p, text):
    """解析 <mark> <AI> 标签 + **粗体**，添加到段落"""
    # 按所有标签分隔
    pattern = '|'.join(f'({re.escape(t[1])}.*?{re.escape(t[2])})' for t in TAGS)
    parts = re.split(f'({pattern})', text)
    for part in parts:
        if not part:
            continue
        matched = False
        for tag_name, tag_open, tag_close, color in TAGS:
            if part.startswith(tag_open) and part.endswith(tag_close):
                inner = part[len(tag_open):-len(tag_close)]
                _add_bold_runs(p, inner, highlight_color=color)
                matched = True
                break
        if not matched:
            _add_bold_runs(p, part, highlight_color=None)

def _add_bold_runs(p, text, highlight_color=None):
    """处理 **粗体** 并可选高亮"""
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            inner = part[2:-2]
            run = p.add_run(inner)
            run.bold = True
        else:
            run = p.add_run(part)
        if highlight_color:
            _add_highlight(run, highlight_color)

def _add_highlight(run, color):
    """给 run 添加文字高亮（字符底纹）"""
    rPr = run._element.get_or_add_rPr()
    highlight = OxmlElement('w:highlight')
    highlight.set(qn('w:val'), color)
    rPr.append(highlight)

def add_heading_custom(doc, text, level=1):
    """添加标题，支持 <mark> 和 <AI>"""
    h = doc.add_heading(level=level)
    h.clear()
    pattern = '|'.join(f'({re.escape(t[1])}.*?{re.escape(t[2])})' for t in TAGS)
    parts = re.split(f'({pattern})', text)
    for part in parts:
        if not part:
            continue
        matched = False
        for tag_name, tag_open, tag_close, color in TAGS:
            if part.startswith(tag_open) and part.endswith(tag_close):
                inner = part[len(tag_open):-len(tag_close)]
                run = h.add_run(inner)
                _add_highlight(run, color)
                matched = True
                break
        if not matched:
            h.add_run(part)
    return h

def add_table_from_lines(doc, lines):
    """从 Markdown 表格行列表创建 Word 表格"""
    if not lines:
        return
    # 解析表头
    header_cells = [c.strip() for c in lines[0].split('|') if c.strip()]
    # 跳过分隔行 (|---|---|)
    data_lines = [l for l in lines[1:] if not re.match(r'^[\s\|\-:]+$', l)]

    rows_data = []
    for line in data_lines:
        cells = [c.strip() for c in line.split('|') if c.strip()]
        rows_data.append(cells)

    if not rows_data:
        return

    max_cols = len(header_cells)
    table = doc.add_table(rows=1 + len(rows_data), cols=max_cols, style='Table Grid')
    table.autofit = True

    # 表头
    for i, cell_text in enumerate(header_cells):
        if i < max_cols:
            set_cell_font(table.rows[0].cells[i], cell_text, bold=True, size=Pt(9))
            # 表头灰色底纹
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), 'D9E2F3')
            table.rows[0].cells[i]._element.get_or_add_tcPr().append(shading)

    # 数据行
    for r, row_data in enumerate(rows_data):
        for c, cell_text in enumerate(row_data):
            if c < max_cols:
                # 处理 <mark> 和 <AI> 标签
                clean = cell_text
                for t in TAGS:
                    clean = clean.replace(t[1], '').replace(t[2], '')
                clean = clean.replace('**', '')
                set_cell_font(table.rows[r + 1].cells[c], clean, bold=False, size=Pt(9))
                # <mark> → 黄色底纹, <AI> → 绿色底纹
                for tag_name, tag_open, tag_close, color in TAGS:
                    if tag_open in cell_text:
                        shading = OxmlElement('w:shd')
                        shading.set(qn('w:fill'), color)
                        table.rows[r + 1].cells[c]._element.get_or_add_tcPr().append(shading)
                        break  # 一种颜色即可

    doc.add_paragraph()  # 表后空行
    return table


def process_file(filepath):
    """处理单个 Markdown 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0
    table_buffer = []
    in_table = False
    in_math = False
    math_buffer = []

    while i < len(lines):
        line = lines[i]

        # 数学公式块 $$...$$
        if line.strip().startswith('$$') and not in_math:
            in_math = True
            math_buffer = []
            i += 1
            continue
        if in_math:
            if line.strip().startswith('$$'):
                # 公式结束，输出为居中斜体
                math_text = '\n'.join(math_buffer)
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(math_text)
                run.italic = True
                run.font.size = Pt(11)
                in_math = False
                i += 1
                continue
            else:
                math_buffer.append(line)
                i += 1
                continue

        # 表格检测（连续的 |...| 行）
        if line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                table_buffer = []
                in_table = True
            table_buffer.append(line)
            i += 1
            # 如果下一行也是表格或者是分隔行，继续收集
            if i < len(lines) and (lines[i].strip().startswith('|') or re.match(r'^[\s\|\-:]+$', lines[i])):
                continue
            else:
                # 表格结束，输出
                add_table_from_lines(doc, table_buffer)
                table_buffer = []
                in_table = False
                i += 1
                continue

        # 如果正在收集表格但当前行不是表格行
        if in_table:
            add_table_from_lines(doc, table_buffer)
            table_buffer = []
            in_table = False

        # 空行
        if line.strip() == '':
            i += 1
            continue

        # **[此处插入图X]** 占位符
        if line.strip().startswith('**[此处插入图'):
            p = doc.add_paragraph()
            run = p.add_run(line.strip().replace('**', ''))
            run.italic = True
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(128, 128, 128)
            i += 1
            continue

        # 标题 # ## ### ####
        stripped = line.strip()
        if stripped.startswith('# ') or stripped == '#':
            level = len(stripped.split()[0]) if stripped[0] == '#' else 1
            title_text = stripped.lstrip('#').strip()
            add_heading_custom(doc, title_text, level=min(level, 4))
            i += 1
            continue
        if re.match(r'^#{2,4}\s', stripped):
            level = len(stripped.split()[0])
            title_text = stripped.lstrip('#').strip()
            add_heading_custom(doc, title_text, level=min(level, 4))
            i += 1
            continue

        # 无序列表
        if stripped.startswith('- ') or stripped.startswith('* '):
            text = stripped[2:]
            p = doc.add_paragraph(style='List Bullet')
            p.clear()
            add_mark_to_runs(p, text)
            i += 1
            continue

        # 有序列表 1. 2.
        if re.match(r'^\d+\.\s', stripped):
            text = re.sub(r'^\d+\.\s', '', stripped)
            p = doc.add_paragraph(style='List Number')
            p.clear()
            add_mark_to_runs(p, text)
            i += 1
            continue

        # 普通段落
        add_mark_aware_paragraph(doc, stripped)
        i += 1


def add_mark_to_runs(p, text):
    """给段落添加 run，支持 <mark> <AI> 和 **粗体**"""
    _add_tagged_runs(p, text)


# --- 主流程 ---
print("开始生成 Word 文档...")

for fname in FILES:
    fpath = os.path.join(BASE, fname)
    if os.path.exists(fpath):
        print(f"  处理: {fname}")
        process_file(fpath)
        # 每章之间加分页符（题目和摘要之间除外）
        if fname != "论文题目.md":
            doc.add_page_break()
    else:
        print(f"  ⚠ 跳过 (不存在): {fname}")

# --- 保存 ---
output_path = os.path.join(BASE, "..", "论文全文-PEMFC退化预测.docx")
doc.save(output_path)
print(f"\nDone: {output_path}")
