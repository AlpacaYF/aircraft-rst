"""Shared style + layout toolkit for the RST framework figures.

Follows scipilot-figure-skill's publishing rules (final-size rendering,
vector-first, Okabe-Ito colorblind palette, >=6 pt text, 600 DPI line art)
while drawing schematics, which the skill itself does not cover.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

SKILL_SCRIPTS = Path.home() / ".qoder-cn" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

from export_figure import export_figure
from setup_style import setup_style
from visual_qa import audit_layout

CM = 1.0 / 2.54
FIG_W_CM = 14.65          # thesis text width: A4 21.0 - 3.17 x 2
PNG_DPI = 600

# Okabe-Ito subset (plot_recipes.md); #F0E442 dropped -- invisible on white print.
MODULE_C = {
    "M1": "#0072B2",   # data preprocessing / entity construction
    "M2": "#009E73",   # region-scene visual features
    "M3": "#D55E00",   # MLLM semantic knowledge (offline)
    "M4": "#CC79A7",   # RST fusion and recognition
    "M5": "#E69F00",   # training and inference
    "IO": "#000000",   # input / output
    "GRAY": "#555555",
}
# Grayscale luminance of M2 and M3 is close, so colour is never the only cue.
# Texture goes on the band headers only: hatching every box buries the labels.
MODULE_HATCH = {"M1": None, "M2": "/", "M3": "....", "M4": "\\\\", "M5": "xx",
                "IO": None, "GRAY": None}

FS_HEAD, FS_BLOCK, FS_SUB, FS_MIN = 7.5, 7.0, 6.5, 6.0
LW_BOX, LW_FLOW, LW_AUX, LW_LANE = 0.8, 1.0, 0.7, 0.6

SOLID, OFFLINE, SUPERVISE = "-", (0, (2.0, 1.4)), (0, (0.8, 1.2))


def apply_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_BLOCK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "hatch.linewidth": 0.4,
        "patch.force_edgecolor": True,
        "savefig.dpi": PNG_DPI,
        "savefig.pad_inches": 0.0,
        "savefig.facecolor": "white",
        "path.simplify": False,
    })


def canvas(h_cm: float):
    """Full-bleed axes where 1 data unit == 1 cm at the final print size."""
    fig = plt.figure(figsize=(FIG_W_CM * CM, h_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W_CM)
    ax.set_ylim(0, h_cm)
    ax.axis("off")
    ax.set_facecolor("white")
    return fig, ax


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def c(self): return (self.x + self.w / 2, self.y + self.h / 2)
    @property
    def l(self): return (self.x, self.y + self.h / 2)
    @property
    def r(self): return (self.x + self.w, self.y + self.h / 2)
    @property
    def t(self): return (self.x + self.w / 2, self.y + self.h)
    @property
    def b(self): return (self.x + self.w / 2, self.y)
    @property
    def lt(self): return (self.x, self.y + self.h)
    @property
    def rt(self): return (self.x + self.w, self.y + self.h)
    @property
    def lb(self): return (self.x, self.y)
    @property
    def rb(self): return (self.x + self.w, self.y)

    def shifted(self, dx=0.0, dy=0.0):
        return Rect(self.x + dx, self.y + dy, self.w, self.h)


def lane(ax, r: Rect, color, ls=SOLID, lw=LW_LANE, alpha=0.045, hatch=None, z=0):
    ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, transform=ax.transData,
                           facecolor=color, alpha=alpha, edgecolor=color,
                           linewidth=lw, linestyle=ls, hatch=hatch, zorder=z))


def box(ax, r: Rect, text="", *, color, hatch=None, ls=SOLID, fs=FS_BLOCK,
        weight="normal", z=2, fill_alpha=0.13, rounding=0.14, text_color="#000000",
        ha="center", va="center", dy=0.0):
    ax.add_patch(FancyBboxPatch(
        (r.x, r.y), r.w, r.h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        mutation_aspect=1, facecolor=color, alpha=fill_alpha, edgecolor=color,
        linewidth=LW_BOX, linestyle=ls, hatch=hatch, zorder=z))
    if text:
        ax.text(r.x + r.w / 2, r.y + r.h / 2 + dy, text, fontsize=fs,
                fontweight=weight, color=text_color, ha=ha, va=va,
                linespacing=1.28, zorder=z + 1, clip_on=False)
    return r


def arrow(ax, p0, p1, *, color="#000000", lw=LW_FLOW, ls=SOLID, rad=0.0,
          head=7.0, shrink=1.6, z=1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=head,
        connectionstyle=f"arc3,rad={rad}", color=color, linewidth=lw,
        linestyle=ls, shrinkA=shrink, shrinkB=shrink, zorder=z,
        joinstyle="miter"))


def elbow(ax, p0, p1, *, color="#000000", lw=LW_FLOW, ls=SOLID, z=1, head=7.0,
          shrink=1.6):
    """Orthogonal 3-segment route: horizontal, vertical, then arrowed horizontal."""
    x0, y0 = p0
    x1, y1 = p1
    xm = (x0 + x1) / 2
    ax.plot([x0, xm], [y0, y0], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")
    ax.plot([xm, xm], [y0, y1], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")
    arrow(ax, (xm, y1), (x1, y1), color=color, lw=lw, ls=ls, zorder=z,
          head=head, shrink=shrink)


def poly_arrow(ax, points, *, color="#000000", lw=LW_FLOW, ls=SOLID, z=1,
               head=7.0):
    """Orthogonal route through waypoints; only the final segment gets a head."""
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle="butt")
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls, z=z,
          head=head)


def label(ax, x, y, text, *, fs=FS_SUB, color="#000000", weight="normal",
          style="normal", ha="center", va="center", z=4, ls=1.28):
    ax.text(x, y, text, fontsize=fs, color=color, fontweight=weight,
            style=style, ha=ha, va=va, zorder=z, linespacing=ls, clip_on=False)


def eqtag(ax, x, y, tag, *, color=MODULE_C["GRAY"]):
    label(ax, x, y, f"({tag})", fs=FS_MIN, color=color, style="italic")


def chip(ax, r: Rect, text, *, color, fs=FS_MIN):
    box(ax, r, text, color=color, fs=fs, fill_alpha=0.22, rounding=0.10, lw=0.6)


def export(fig, basename: str, h_cm: float, out_dir: Path) -> list[str]:
    """Vector + exact-size raster. Grayscale preview is made here, not by
    export_figure: its _grayscale_from() re-saves the PNG with a hardcoded
    bbox_inches='tight', which would clobber the exact-size raster."""
    paths = export_figure(
        fig, str(out_dir / basename), formats=["pdf", "svg", "png"],
        size_inches=(FIG_W_CM * CM, h_cm * CM), dpi=PNG_DPI,
        tight=False, pad_inches=0.0, grayscale_preview=False)
    from PIL import Image
    png = f"{out_dir / basename}.png"
    gray = f"{out_dir / basename}_grayscale.png"
    Image.open(png).convert("L").save(gray)
    paths.append(gray)
    return paths


def selfcheck(fig, basename: str, out_dir: Path, h_cm: float) -> list[str]:
    """Machine gate: missing glyphs / clipping, text-text and text-bound overlap,
    axes still full-bleed, PNG pixel size and DPI equal the declared print size."""
    problems: list[str] = []

    for item in audit_layout(fig):
        problems.append(f"audit_layout: {item[0]} {item[1]}")

    pos = fig.axes[0].get_position()
    if max(abs(pos.x0), abs(pos.y0), abs(1 - pos.x1), abs(1 - pos.y1)) > 1e-6:
        problems.append(f"axes not full-bleed: {pos.bounds}")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.axes[0].transData.inverted()
    texts = [t for t in fig.axes[0].texts if t.get_text().strip()]
    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        boxes.append((t, bb))
        for xy in ((bb.x0, bb.y0), (bb.x1, bb.y1)):
            dx, dy = inv.transform(xy)
            if not (-0.02 <= dx <= FIG_W_CM + 0.02 and -0.02 <= dy <= h_cm + 0.02):
                problems.append(f"out of canvas: {t.get_text()[:28]!r} at ({dx:.2f},{dy:.2f})")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 1.5 and oy > 1.5:
                problems.append(
                    f"text overlap {ox:.1f}x{oy:.1f}px: "
                    f"{boxes[i][0].get_text()[:22]!r} vs {boxes[j][0].get_text()[:22]!r}")

    png = out_dir / f"{basename}.png"
    from PIL import Image
    with Image.open(png) as im:
        want = round(FIG_W_CM / 2.54 * PNG_DPI)
        if abs(im.width - want) > 2:
            problems.append(f"PNG width {im.width}px, expected {want}px "
                            "(figure was rescaled)")
        dpi = im.info.get("dpi")
        if not dpi or round(dpi[0]) != PNG_DPI:
            problems.append(f"PNG dpi metadata {dpi}, expected ({PNG_DPI},{PNG_DPI})")
    return problems


def report(name: str, problems: list[str]) -> bool:
    if problems:
        print(f"[FAIL] {name}")
        for p in problems:
            print("   -", p)
        return False
    print(f"[PASS] {name}")
    return True
