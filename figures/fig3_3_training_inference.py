"""图3-3  Staged training and inference flow, detail (Section 3.5).

Expands module 5 of 图3-1: the three training stages with the loss term each
one adds (eq. 3-18 ~ 3-21), and the inference chain, where the detector supplies
candidate boxes while scene features and class prototypes are reused from cache,
so no MLLM call happens at inference time.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from figstyle import (FS_HEAD, FS_MIN, MODULE_C, OFFLINE, SUPERVISE, Rect,
                      apply_style, arrow, box, canvas, export, label, lane,
                      poly_arrow, report, selfcheck)

OUT = Path(__file__).resolve().parent
H = 8.6
C = MODULE_C["M5"]
GRAY = MODULE_C["GRAY"]


def note(ax, x, y, text, **kw):
    label(ax, x, y, text, fs=FS_MIN, color=GRAY, style="italic", **kw)


def main():
    apply_style()
    fig, ax = canvas(H)

    ax.add_patch(plt.Rectangle((0.25, 7.95), 14.15, 0.60, facecolor=C,
                               alpha=0.20, edgecolor=C, lw=0.8, zorder=1))
    ax.add_patch(plt.Circle((0.67, 8.25), 0.20, facecolor=C, edgecolor="none",
                            zorder=5))
    ax.text(0.67, 8.242, "5", fontsize=6.5, color="white", ha="center",
            va="center", fontweight="bold", zorder=6)
    label(ax, 0.97, 8.25, "Model training & inference strategy — detail",
          fs=FS_HEAD, weight="bold", ha="left")

    # ---------------- training ---------------------------------------------
    lane(ax, Rect(0.25, 4.55, 14.15, 3.30), C, alpha=0.04)
    label(ax, 0.45, 7.72, "Training (staged)", fs=FS_MIN, ha="left", color="#000000", weight="bold")
    s1 = box(ax, Rect(0.45, 6.30, 4.20, 1.30),
             "Stage 1 · visual base\n$F^{R}$ from Region crops\n"
             "$L_{\\mathrm{base}}=L_{\\mathrm{domain}}+L_{\\mathrm{class}}$\n(3-18)",
             color=C, fs=FS_MIN)
    s2 = box(ax, Rect(4.95, 6.30, 4.20, 1.30),
             "Stage 2 · multimodal fusion\nintroduce $F^{T}$, joint training\n"
             "$+\\lambda_1 L_{\\mathrm{con}}$, temperature $\\tau$\n(3-19)",
             color=C, fs=FS_MIN)
    s3 = box(ax, Rect(9.45, 6.30, 4.20, 1.30),
             "Stage 3 · hard negative\nsimilar-class contrast, margin $m$\n"
             "$+\\lambda_2 L_{\\mathrm{hard}}$ (3-20)\ndifference cues $L_{\\mathrm{diff}}$",
             color=C, fs=FS_MIN)
    arrow(ax, s1.r, s2.l)
    arrow(ax, s2.r, s3.l)
    loss = box(ax, Rect(1.10, 4.85, 12.50, 0.85),
               "Total objective\n"
               "$L=L_{\\mathrm{domain}}+L_{\\mathrm{class}}+\\lambda_1 L_{\\mathrm{con}}"
               "+\\lambda_2 L_{\\mathrm{hard}}+\\lambda_3 L_{\\mathrm{diff}}$",
               color=C, fs=FS_MIN)
    note(ax, 13.85, 5.28, "(3-21)", ha="left")
    for x in (2.55, 7.05, 11.55):
        arrow(ax, (x, 6.30), (x, 5.70), ls=SUPERVISE, lw=0.8)
    note(ax, 0.45, 4.68,
         "MLLM only builds knowledge offline; $F_c^{T}$ is encoded before training "
         "and cached, so the recognition model never calls it.", ha="left")

    # ---------------- inference --------------------------------------------
    lane(ax, Rect(0.25, 0.35, 14.15, 3.90), C, alpha=0.04)
    label(ax, 0.45, 4.05, "Inference", fs=FS_MIN, ha="left",
          color="#000000", weight="bold")
    widths = [1.80, 1.95, 1.95, 2.35, 2.35, 2.45]
    items = ["Full image", "Detector\nYOLO11", "Candidate\nboxes",
             "Region\nfeature $F^{R}$", "RST fusion\n$H$ (3-15)",
             "Entity output\nloc·attr·type"]
    x = 0.35
    chain = []
    for w in widths:
        chain.append(Rect(x, 3.05, w, 0.75))
        x += w + 0.18
    for r, txt in zip(chain, items):
        box(ax, r, txt, color=C, fs=FS_MIN)
    for a, b in zip(chain[:-1], chain[1:]):
        arrow(ax, a.r, b.l)
    sc1 = box(ax, Rect(0.35, 1.55, 6.60, 0.80),
              "Scene feature $F^{S}$: encoded once per image\n"
              "and shared by all entities in it (3-4)",
              color=C, ls=OFFLINE, fs=FS_MIN)
    sc2 = box(ax, Rect(7.50, 1.55, 6.60, 0.80), "Cached class prototypes\n"
              "$F_c^{T}$ (3-8)", color=C, ls=OFFLINE, fs=FS_MIN)
    poly_arrow(ax, [(3.65, 2.35), (3.65, 2.68), (10.30, 2.68), (10.30, 3.05)],
               ls=OFFLINE)
    arrow(ax, (10.80, 2.35), (10.80, 3.05), ls=OFFLINE)
    note(ax, 0.45, 0.80,
         "No MLLM call at inference: the detector supplies boxes and the cached "
         "prototypes supply semantics, so latency is set by detection plus fusion.",
         ha="left")
    paths = export(fig, "fig3_3_training_inference", H, OUT)
    ok = report("fig3_3_training_inference",
                selfcheck(fig, "fig3_3_training_inference", OUT, H))
    print("files:", [Path(p).name for p in paths])
    plt.close(fig)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
