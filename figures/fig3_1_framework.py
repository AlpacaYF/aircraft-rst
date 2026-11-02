"""图3-1  RST multimodal fusion framework (overall).

Five modules of Chapter 3 as full-width bands, so every box keeps enough room
for English labels at >=6 pt inside the 14.65 cm thesis text width.
Reading order follows the paper: 1 preprocessing -> 2 region/scene features
-> 3 offline MLLM knowledge -> 4 RST fusion -> 5 training & inference.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from figstyle import (FS_HEAD, FS_MIN, MODULE_C, MODULE_HATCH, OFFLINE, Rect,
                      apply_style, arrow, box, canvas, export, label, lane,
                      report, selfcheck)

OUT = Path(__file__).resolve().parent
H = 12.45
X0 = 0.25
IN_L = 0.50
BANDS = {"M1": (10.10, 2.10), "M2": (6.65, 3.00), "M3": (6.65, 3.00),
         "M4": (3.30, 2.90), "M5": (0.25, 2.60)}


def badge(ax, cx, cy, n, color):
    ax.add_patch(plt.Circle((cx, cy), 0.20, facecolor=color, edgecolor="none",
                            zorder=5))
    ax.text(cx, cy - 0.008, str(n), fontsize=FS_MIN + 0.5, color="white",
            ha="center", va="center", fontweight="bold", zorder=6)


def band(ax, key, n, title, x=X0, w=14.15, dashed=False, tag=None):
    """Band outline + numbered header strip. The header carries the module
    texture so content boxes stay clean enough to read in grayscale."""
    y, h = BANDS[key]
    c = MODULE_C[key]
    lane(ax, Rect(x, y, w, h), c, ls=OFFLINE if dashed else "-", alpha=0.04)
    box(ax, Rect(x, y + h - 0.62, w, 0.62), "", color=c, fill_alpha=0.20,
        rounding=0.12, z=1, hatch=MODULE_HATCH[key])
    badge(ax, x + 0.42, y + h - 0.31, n, c)
    label(ax, x + 0.72, y + h - 0.31, title, fs=FS_HEAD, weight="bold", ha="left")
    if tag:
        label(ax, x + w - 0.20, y + h - 0.31, tag, fs=FS_MIN,
              color=MODULE_C["GRAY"], ha="right", style="italic")


def hchain(ax, y, h, x_first, widths, items, color, gap=0.30):
    rects, x = [], x_first
    for w in widths:
        rects.append(Rect(x, y, w, h))
        x += w + gap
    for r, txt in zip(rects, items):
        box(ax, r, txt, color=color, fs=FS_MIN)
    for a, b in zip(rects[:-1], rects[1:]):
        arrow(ax, a.r, b.l)
    return rects


def main():
    apply_style()
    fig, ax = canvas(H)

    # ---------------- 1  data preprocessing & entity construction ----------
    band(ax, "M1", 1, "Data preprocessing & entity construction")
    r1 = box(ax, Rect(IN_L, 10.42, 2.90, 1.00), "Raw image $S_{ij}$\n+ instance boxes",
             color=MODULE_C["M1"], fs=FS_MIN)
    r2 = box(ax, Rect(3.70, 10.42, 4.40, 1.00),
             "Crop by bounding box (3-1)\nRegion $R_{ij}$ · Scene $S_{ij}$",
             color=MODULE_C["M1"], fs=FS_MIN)
    r3 = box(ax, Rect(8.40, 10.42, 5.60, 1.00),
             "Per-entity sample\nLabel $Y_{ij}$ · Attribute $D_{ij}$",
             color=MODULE_C["M1"], fs=FS_MIN)
    for a, b in ((r1, r2), (r2, r3)):
        arrow(ax, a.r, b.l)
    label(ax, IN_L, 10.20, "Each aircraft in the image becomes an independent entity.",
          fs=FS_MIN, color=MODULE_C["GRAY"], ha="left", style="italic")

    # ---------------- 2  region-scene visual features ----------------------
    band(ax, "M2", 2, "Region–scene visual features", x=X0, w=6.95)
    hchain(ax, 8.10, 0.85, IN_L, [1.25, 1.50, 2.60],
           ["Region\n$R_{ij}$", "Visual\nencoder", "Local tokens\n$F^{R}$\n(3-2, 3-3)"],
           MODULE_C["M2"])
    hchain(ax, 7.10, 0.85, IN_L, [1.25, 1.50, 2.60],
           ["Scene\n$S_{ij}$", "Scene\nencoder", "Global context\n$F^{S}$\n(3-4)"],
           MODULE_C["M2"])
    label(ax, IN_L, 6.80,
          "Overlapping tiles + 2-D position encoding, hierarchical\n"
          "token compression; scene encoded once, shared by entities.",
          fs=FS_MIN, color=MODULE_C["GRAY"], ha="left", style="italic")

    # ---------------- 3  offline MLLM semantic knowledge -------------------
    band(ax, "M3", 3, "MLLM semantic knowledge", x=7.45, w=6.95, dashed=True,
         tag="offline, frozen")
    xL, xR, wB = 7.70, 10.90, 3.10
    a1 = box(ax, Rect(xL, 8.10, wB, 0.85), "MLLM knowledge\ngeneration (3-5)\n$K_c$",
             color=MODULE_C["M3"], fs=FS_MIN)
    a2 = box(ax, Rect(xR, 8.10, wB, 0.85), "Consistency\nfilter (3-6)\n$S_{v,t}$",
             color=MODULE_C["M3"], fs=FS_MIN)
    a3 = box(ax, Rect(xL, 7.10, wB, 0.85), "Hard-negative\ndifference (3-7)\n$H_{ab}$",
             color=MODULE_C["M3"], fs=FS_MIN)
    a4 = box(ax, Rect(xR, 7.10, wB, 0.85), "Text encoder\n$F_c^{T}$ (3-8)",
             color=MODULE_C["M3"], fs=FS_MIN)
    arrow(ax, a1.r, a2.l)
    arrow(ax, a1.b, a3.t)
    arrow(ax, a3.r, a4.l)
    arrow(ax, a2.b, a4.t)
    label(ax, xL, 6.80, "Input: class name + real images, not entity crops\n"
          "$K_c$: appearance · structure · cues · differences",
          fs=FS_MIN, color=MODULE_C["GRAY"], ha="left", style="italic")

    # ---------------- 4  RST fusion & recognition --------------------------
    # Two rows: online path (F^R, F^S -> fusion -> H -> heads) above, cached
    # text prototypes (F^T -> S_c -> type head) below.
    band(ax, "M4", 4, "RST multimodal fusion & entity recognition")
    cR = box(ax, Rect(IN_L, 5.05, 1.05, 0.42), "$F^{R}$", color=MODULE_C["M4"])
    cS = box(ax, Rect(IN_L, 4.55, 1.05, 0.42), "$F^{S}$", color=MODULE_C["M4"])
    cT = box(ax, Rect(IN_L, 3.66, 1.05, 0.42), "$F^{T}$", color=MODULE_C["M4"])
    f1 = box(ax, Rect(1.90, 4.55, 3.90, 0.95),
             "Multimodal fusion\nregion–scene attention\n+ dynamic gating\n(3-9 ~ 3-15)",
             color=MODULE_C["M4"], fs=FS_MIN)
    f2 = box(ax, Rect(6.15, 4.55, 2.60, 0.95),
             "Entity\nrepresentation\n$H$ (3-15)", color=MODULE_C["M4"], fs=FS_MIN)
    f3 = box(ax, Rect(9.05, 4.55, 4.90, 0.95), "Attribute head  $D$  (3-16)",
             color=MODULE_C["M4"], fs=FS_MIN)
    f4 = box(ax, Rect(9.05, 3.42, 4.90, 0.90), "Type head  $Y$  (3-17)",
             color=MODULE_C["M4"], fs=FS_MIN)
    sc = box(ax, Rect(6.15, 3.42, 2.60, 0.90), "Semantic\nmatching $S_c$\n(3-12)",
             color=MODULE_C["M4"], ls=OFFLINE, fs=FS_MIN)
    arrow(ax, cR.r, (1.90, 5.26))
    arrow(ax, cS.r, (1.90, 4.76))
    arrow(ax, f1.r, f2.l)
    arrow(ax, f2.r, f3.l)
    arrow(ax, f2.b, sc.t)
    arrow(ax, sc.r, (9.05, 3.87))
    arrow(ax, (8.75, 4.62), (9.05, 4.20))
    arrow(ax, cT.r, sc.l, ls=OFFLINE, lw=0.8)
    label(ax, 3.85, 4.06, "cached class prototypes", fs=FS_MIN,
          color=MODULE_C["GRAY"], style="italic")

    # ---------------- 5  training & inference ------------------------------
    band(ax, "M5", 5, "Model training & inference strategy")
    label(ax, IN_L, 1.75, "Training\n(staged)", fs=FS_MIN, ha="left",
          color="#000000", weight="bold")
    hchain(ax, 1.30, 0.90, 1.85, [2.85, 2.85, 2.85, 2.85],
           ["Stage 1\nvisual base\n$L_{\\mathrm{base}}$ (3-18)",
            "Stage 2\n+ text modality\n$L_{\\mathrm{con}}$ (3-19)",
            "Stage 3\nhard negative\n$L_{\\mathrm{hard}}$ (3-20)",
            "Optimization\nTotal loss $L$\nwith $L_{\\mathrm{diff}}$ (3-21)"],
           MODULE_C["M5"], gap=0.26)
    label(ax, IN_L, 0.85, "Inference", fs=FS_MIN, ha="left",
          color="#000000", weight="bold")
    hchain(ax, 0.45, 0.80, 1.85, [2.10, 2.10, 2.10, 2.20, 2.60],
           ["Full image", "Detector\nYOLO11", "Candidate\nboxes",
            "RST\nrecognition", "Entity output\nloc·attr·type"],
           MODULE_C["M5"], gap=0.26)
    label(ax, 14.10, 0.32, "MLLM is not called at inference.", fs=FS_MIN,
          color=MODULE_C["GRAY"], ha="right", style="italic")

    # ---------------- inter-band flow --------------------------------------
    for x, y_from, y_to in ((3.70, 10.10, 9.65), (3.70, 6.65, 6.20),
                            (10.90, 6.65, 6.20), (7.30, 3.30, 2.85)):
        arrow(ax, (x, y_from), (x, y_to))

    paths = export(fig, "fig3_1_framework", H, OUT)
    ok = report("fig3_1_framework",
                selfcheck(fig, "fig3_1_framework", OUT, H))
    print("files:", [Path(p).name for p in paths])
    plt.close(fig)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
