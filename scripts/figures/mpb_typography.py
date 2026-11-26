"""Shared Arial typography and RGB PNG helper for the MPB figures."""

from pathlib import Path
from matplotlib import font_manager, rcParams
from PIL import Image


def configure_typography() -> None:
    for weight, style in (("normal", "normal"), ("bold", "normal"), ("normal", "italic")):
        prop = font_manager.FontProperties(family="Arial", weight=weight, style=style)
        font_manager.findfont(prop, fallback_to_default=False)

    rcParams.update({
        "font.family": ["Arial"],
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "font.weight": "normal",
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "axes.labelweight": "normal",
        "axes.unicode_minus": True,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "text.color": "#202020",
        "axes.labelcolor": "#202020",
        "xtick.labelcolor": "#202020",
        "ytick.labelcolor": "#202020",
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
        "mathtext.cal": "Arial:italic",
        "mathtext.fallback": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def flatten_png_to_rgb(png_path: str | Path, fallback_dpi: float = 600.0) -> None:
    """Flattens an RGBA PNG onto a pure white RGB canvas, preserving DPI metadata."""
    p = Path(png_path)
    if not p.is_file():
        return
    with Image.open(p) as im:
        dpi = im.info.get("dpi", (fallback_dpi, fallback_dpi))
        if im.mode == "RGB":
            return
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
    bg.save(p, format="PNG", dpi=dpi)

