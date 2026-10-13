from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "paper" / "figures"

BLACK = (20, 20, 20)
DARK = (56, 56, 56)
MID = (110, 110, 110)
LINE = (68, 68, 68)
WHITE = (255, 255, 255)
FILL = (250, 250, 250)
FILL2 = (242, 242, 242)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


F_TITLE = font(48, True)
F_HEAD = font(38, True)
F_BODY = font(32)
F_SMALL = font(27)
F_TINY = font(23)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=8)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> str:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = ""
        for character in raw_line:
            candidate = line + character
            if not line or draw.textlength(candidate, font=fnt) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = character
        if line:
            lines.append(line)
    return "\n".join(lines)


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill=BLACK,
    spacing: int = 8,
) -> None:
    x1, y1, x2, y2 = xy
    wrapped = wrap_text(draw, text, fnt, x2 - x1 - 24)
    width, height = text_size(draw, wrapped, fnt)
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2),
        wrapped,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt=F_SMALL, fill=DARK) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color=LINE, width: int = 3) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 16
    spread = math.radians(23)
    left = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    right = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, left, right], fill=color)


def scientific_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    subtitle: str = "",
    fill=WHITE,
    title_font=F_HEAD,
    subtitle_font=F_SMALL,
) -> None:
    draw.rectangle(xy, fill=fill, outline=LINE, width=3)
    x1, y1, x2, y2 = xy
    if subtitle:
        centered_text(draw, (x1 + 16, y1 + 14, x2 - 16, y1 + 102), title, title_font)
        centered_text(draw, (x1 + 18, y1 + 110, x2 - 18, y2 - 16), subtitle, subtitle_font, fill=DARK, spacing=7)
    else:
        centered_text(draw, (x1 + 14, y1 + 14, x2 - 14, y2 - 14), title, title_font)


def fusion_node(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    radius = 35
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=WHITE, outline=LINE, width=3)
    centered_text(draw, (x - 28, y - 27, x + 28, y + 27), "W", F_SMALL, fill=BLACK)


def make_pipeline() -> Path:
    w, h = 2600, 930
    image = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(image)

    label(draw, (100, 52), "(a) 端到端实例检测流程", F_HEAD, BLACK)
    draw.line((100, 118, w - 100, 118), fill=(185, 185, 185), width=2)

    for x1, x2, title in [
        (120, 1210, "数据准备与预处理"),
        (1325, 1705, "检测网络"),
        (1820, 2480, "后处理与输出"),
    ]:
        draw.line((x1, 185, x2, 185), fill=(125, 125, 125), width=2)
        centered_text(draw, (x1, 132, x2, 176), title, F_SMALL, fill=MID)

    y, box_width, box_height, gap = 250, 330, 200, 75
    xs = [120 + index * (box_width + gap) for index in range(6)]
    steps = [
        ("MARIDA 样本", "11 波段影像块\n256 x 256 像素"),
        ("七通道构造", "B4, B3, B2, B8\nB11, B12, NDWI-inv"),
        ("预处理", "p01-p99 缩放\nletterbox: 512 / 1024"),
        ("YOLOv11n\n+ BiFPN", "CBAM 注意力\nP3-P5 检测头"),
        ("后处理", "NMS\n阈值 0.25；Platt 校准"),
        ("实例级输出", "边界框与置信度\n数量、P、R、F1"),
    ]
    for index, (title, subtitle) in enumerate(steps):
        scientific_box(
            draw,
            (xs[index], y, xs[index] + box_width, y + box_height),
            title,
            subtitle,
            fill=FILL2 if index == 3 else FILL,
            title_font=F_BODY,
        )
        if index < len(steps) - 1:
            arrow(draw, (xs[index] + box_width + 14, y + box_height // 2), (xs[index + 1] - 14, y + box_height // 2))

    draw.line((120, 570, w - 120, 570), fill=(185, 185, 185), width=2)
    label(draw, (120, 622), "评价设置", F_SMALL, BLACK)
    label(
        draw,
        (330, 622),
        "scene-clean 数据划分；NDWI-inv 仅由输入影像计算；消融实验报告三个随机种子的均值和标准差。",
        F_SMALL,
        DARK,
    )
    label(
        draw,
        (120, 730),
        "注：imgsz=1024 指 letterbox 上采样后特征图单元数量增加，并非传感器空间分辨率提升。",
        F_TINY,
        MID,
    )
    output = OUT_DIR / "report_fig3_1_pipeline_paper.png"
    image.save(output, dpi=(300, 300))
    return output


def make_modules() -> Path:
    w, h = 2600, 1180
    image = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(image)

    draw.line((100, 118, w - 100, 118), fill=(185, 185, 185), width=2)
    draw.line((1300, 175, 1300, 950), fill=(190, 190, 190), width=2)

    # CBAM: show its two masks separately, then show their sequential application.
    label(draw, (120, 52), "(a) CBAM：通道注意力与空间注意力", F_HEAD, BLACK)
    main_y = 285
    scientific_box(draw, (140, main_y, 310, main_y + 125), "输入特征\nF", fill=FILL, title_font=F_BODY)
    scientific_box(draw, (370, main_y, 600, main_y + 125), "通道注意力\nM_c(F)", fill=FILL2, title_font=F_BODY)
    scientific_box(draw, (660, main_y, 800, main_y + 125), "中间特征\nF_c", fill=FILL, title_font=F_BODY)
    scientific_box(draw, (860, main_y, 1090, main_y + 125), "空间注意力\nM_s(F_c)", fill=FILL2, title_font=F_BODY)
    scientific_box(draw, (1150, main_y, 1270, main_y + 125), "Fout", fill=FILL, title_font=F_SMALL)
    for start, end in [
        ((325, main_y + 62), (355, main_y + 62)),
        ((615, main_y + 62), (645, main_y + 62)),
        ((815, main_y + 62), (845, main_y + 62)),
        ((1105, main_y + 62), (1135, main_y + 62)),
    ]:
        arrow(draw, start, end)

    draw.line((370, 500, 600, 500), fill=(155, 155, 155), width=2)
    centered_text(draw, (370, 530, 600, 582), "AvgPool / MaxPool", F_SMALL, fill=DARK)
    arrow(draw, (485, 594), (485, 650), width=2)
    centered_text(draw, (370, 665, 600, 725), "共享 MLP\n-> Sigmoid", F_SMALL, fill=DARK)
    label(draw, (400, 790), "F_c = M_c(F) x F", F_SMALL, BLACK)

    draw.line((860, 500, 1090, 500), fill=(155, 155, 155), width=2)
    centered_text(draw, (860, 530, 1090, 582), "AvgPool_c / MaxPool_c", F_SMALL, fill=DARK)
    arrow(draw, (975, 594), (975, 650), width=2)
    centered_text(draw, (860, 655, 1090, 735), "Concat\n-> Conv 7 x 7 -> Sigmoid", F_SMALL, fill=DARK)
    label(draw, (865, 790), "F_out = M_s(F_c) x F_c", F_SMALL, BLACK)
    label(draw, (140, 910), "注意力图先后作用于特征张量；模块收益由消融实验而非结构假设判定。", F_TINY, MID)

    # BiFPN: an explicit top-down and bottom-up graph with weighted fusion nodes.
    label(draw, (1370, 52), "(b) BiFPN：双向跨尺度加权融合", F_HEAD, BLACK)
    levels = [(285, "P5 in", "stride 32"), (490, "P4 in", "stride 16"), (695, "P3 in", "stride 8")]
    for y, name, stride in levels:
        scientific_box(draw, (1390, y - 43, 1540, y + 43), name, fill=FILL, title_font=F_SMALL)
        centered_text(draw, (1390, y + 50, 1540, y + 82), stride, F_TINY, fill=MID)

    scientific_box(draw, (1690, 242, 1840, 328), "P5 td", fill=FILL2, title_font=F_SMALL)
    fusion_node(draw, 1785, 490)
    fusion_node(draw, 1785, 695)
    arrow(draw, (1540, 285), (1675, 285))
    arrow(draw, (1765, 328), (1785, 450))
    arrow(draw, (1540, 490), (1745, 490))
    arrow(draw, (1785, 530), (1785, 655))
    arrow(draw, (1540, 695), (1745, 695))

    fusion_node(draw, 2085, 490)
    fusion_node(draw, 2085, 285)
    arrow(draw, (1825, 695), (2035, 695))
    arrow(draw, (2038, 675), (2085, 530))
    arrow(draw, (1825, 490), (2045, 490))
    arrow(draw, (1540, 520), (1600, 520), width=2)
    draw.line((1600, 520, 1600, 570), fill=LINE, width=2)
    arrow(draw, (1600, 570), (2040, 570), width=2)
    arrow(draw, (2048, 460), (2085, 325))
    arrow(draw, (1840, 285), (2045, 285))
    arrow(draw, (1540, 255), (1600, 255), width=2)
    draw.line((1600, 255, 1600, 205), fill=LINE, width=2)
    draw.line((1600, 205, 2085, 205), fill=LINE, width=2)
    arrow(draw, (2085, 205), (2085, 245), width=2)

    for y, text in [(285, "Detect P5"), (490, "Detect P4"), (695, "Detect P3")]:
        scientific_box(draw, (2280, y - 43, 2465, y + 43), text, fill=FILL, title_font=F_SMALL)
    arrow(draw, (2125, 285), (2265, 285))
    arrow(draw, (2125, 490), (2265, 490))
    arrow(draw, (1825, 695), (2265, 695))

    draw.line((1370, 825, 2470, 825), fill=(185, 185, 185), width=2)
    label(draw, (1370, 870), "W：归一化非负权重融合", F_SMALL, BLACK)
    label(draw, (1370, 925), "O = sum(w_i I_i) / (sum(w_i) + eps),  w_i >= 0", F_SMALL, DARK)
    label(draw, (100, h - 80), "注：精确层数、连接关系及检测头配置以模型 YAML 和实际消融配置为准。", F_TINY, MID)
    output = OUT_DIR / "report_fig4_1_cbam_bifpn_paper.png"
    image.save(output, dpi=(300, 300))
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(make_pipeline())
    print(make_modules())


if __name__ == "__main__":
    main()
