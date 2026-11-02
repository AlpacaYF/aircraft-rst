"""图3-2  RST multimodal fusion and entity recognition, detail (Section 3.4).

Expands module 4 of 图3-1: region-scene cross-attention with the residual that
yields the scene-enhanced entity feature, the semantic matching branch, the
gated dynamic fusion, and the two parallel heads -- including the detail that
S_c is added into the type head through beta (eq. 3-17).

Four horizontal rows keep every connection short: attention, fusion, heads,
text branch. A left->right line in one row would otherwise have to cross the
top->bottom lines coming out of H.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from figstyle import (FS_HEAD, FS_MIN, MODULE_C, OFFLINE, Rect, apply_style,
                      arrow, box, canvas, export, label, poly_arrow, report,
                      selfcheck)

OUT = Path(__file__).resolve().parent
H = 8.6
C = MODULE_C["M4"]
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
    ax.text(0.67, 8.242, "4", fontsize=6.5, color="white", ha="center",
            va="center", fontweight="bold", zorder=6)
    label(ax, 0.97, 8.25, "RST multimodal fusion & entity recognition — detail",
          fs=FS_HEAD, weight="bold", ha="left")

    # ---- row 1: region-scene cross-attention ------------------------------
    cR = box(ax, Rect(0.45, 6.75, 2.05, 0.60), "Region tokens\n$F^{R}$: $M\\times d$",
             color=C, fs=FS_MIN)
    cS = box(ax, Rect(0.45, 6.05, 2.05, 0.60), "Scene tokens\n$F^{S}$",
             color=C, fs=FS_MIN)
    q = box(ax, Rect(2.90, 6.75, 1.85, 0.60), "Query\n$Q=F^{R}W_Q$", color=C, fs=FS_MIN)
    kv = box(ax, Rect(2.90, 6.05, 1.85, 0.60), "Key / Value\n$F^{S}W_{K,V}$",
             color=C, fs=FS_MIN)
    attn = box(ax, Rect(5.15, 6.15, 3.60, 1.20),
               "Multi-head cross-attention\n$A_{RS}\\propto\\mathrm{Softmax}(QK^{\\top})$\n(3-10)",
               color=C, fs=FS_MIN)
    res = box(ax, Rect(9.15, 6.15, 3.60, 1.20),
              "Scene-enhanced\nentity feature\n(3-11)", color=C, fs=FS_MIN)
    note(ax, 10.95, 5.92, "$F_{RS}=F^{R}+A_{RS}V$", ha="center")
    arrow(ax, cR.r, q.l)
    arrow(ax, cS.r, kv.l)
    arrow(ax, q.r, (5.15, 7.05), rad=-0.06)
    arrow(ax, kv.r, (5.15, 6.45), rad=0.06)
    arrow(ax, attn.r, res.l)

    # ---- row 2: projections, gating, unified representation ---------------
    proj = box(ax, Rect(2.90, 4.35, 2.60, 0.90), "Projections\n$z_R,z_S,z_T$ (3-13)",
               color=C, fs=FS_MIN)
    gate = box(ax, Rect(5.90, 4.35, 3.60, 0.90),
               "Gated fusion (3-14)\n$\\alpha_R,\\alpha_S,\\alpha_T$", color=C, fs=FS_MIN)
    head_h = box(ax, Rect(9.90, 4.35, 3.45, 0.90), "Entity representation\n$H$ (3-15)",
                 color=C, fs=FS_MIN)
    note(ax, 7.70, 4.10, "$H=\\alpha_R z_R+\\alpha_S z_S+\\alpha_T z_T$", ha="center")
    poly_arrow(ax, [(12.75, 6.15), (12.75, 5.70), (3.80, 5.70), (3.80, 5.25)])
    poly_arrow(ax, [(1.47, 6.05), (1.47, 5.10), (2.90, 5.10)])
    arrow(ax, proj.r, gate.l)
    arrow(ax, gate.r, head_h.l)

    # ---- row 3: parallel heads --------------------------------------------
    hD = box(ax, Rect(6.60, 2.60, 2.80, 0.85), "Attribute head\n$D$ (3-16)",
             color=C, fs=FS_MIN)
    hY = box(ax, Rect(9.90, 2.60, 2.80, 0.85), "Type head\n$Y$ (3-17)",
             color=C, fs=FS_MIN)
    note(ax, 8.00, 2.38, "$D=\\mathrm{Head}_D(H)$", ha="center")
    note(ax, 11.30, 2.38, "$Y=\\mathrm{Head}_Y(H+\\beta S_c)$", ha="center")
    poly_arrow(ax, [(10.60, 4.35), (10.60, 3.80), (8.00, 3.80), (8.00, 3.45)])
    arrow(ax, (11.90, 4.35), (11.90, 3.45))

    # ---- row 4: text branch ------------------------------------------------
    cT = box(ax, Rect(0.45, 0.90, 2.05, 0.75), "Class prototypes\n$F_c^{T}$ (3-8)",
             color=C, fs=FS_MIN)
    sc = box(ax, Rect(2.90, 0.90, 3.60, 0.75),
             "Semantic matching, per token\n$S_c=\\mathrm{Sim}(F_{RS},F_c^{T})$ (3-12)",
             color=C, ls=OFFLINE, fs=FS_MIN)
    arrow(ax, cT.r, sc.l)
    poly_arrow(ax, [(1.47, 1.65), (1.47, 4.55), (2.90, 4.55)], ls=OFFLINE)
    poly_arrow(ax, [(6.50, 1.275), (13.35, 1.275), (13.35, 3.02), (12.70, 3.02)])
    note(ax, 9.60, 1.90, "class-specific match enters the type head", ha="center")
    note(ax, 13.45, 2.20, "$\\beta$", ha="left")

    paths = export(fig, "fig3_2_rst_fusion", H, OUT)
    ok = report("fig3_2_rst_fusion",
                selfcheck(fig, "fig3_2_rst_fusion", OUT, H))
    print("files:", [Path(p).name for p in paths])
    plt.close(fig)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
