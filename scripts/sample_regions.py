"""Verify the context-expanded Region crops actually work.

Two checks:
  geometric - does expansion swallow a neighbouring aircraft that the GT box
              excluded? A contaminated Region teaches the model to predict two
              labels for one box.
  visual    - a contact sheet of raw-bbox vs expanded crops to eyeball whether
              25% is enough to keep wingtips and tail surfaces.

Writes index/region_preview.png and index/region_check.json.
"""

import json
import os
import random
from collections import defaultdict

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "index")
EXPAND = json.load(open(os.path.join(IDX, "class_meta.json"), encoding="utf-8"))["params"]["region_expand"]


def load_instances():
    out = []
    with open(os.path.join(IDX, "instances.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def overlap(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def protected_expand(raw, others, W, H, ratio, cover_limit=0.15):
    """Expand for context, then pull an edge back if the result would swallow
    another entity. The largest edge move that restores the limit and still
    fully contains the GT box wins, so the target keeps as much context as it can."""
    box = expand_with_clamp(raw, W, H, ratio)
    for _ in range(6):
        worst, worst_cov = None, cover_limit
        for o in others:
            cov = overlap(box, o)
            if cov > worst_cov:
                worst, worst_cov = o, cov
        if worst is None:
            break
        ox1, oy1, ox2, oy2 = worst
        x1, y1, x2, y2 = box
        # drop the neighbour by moving the crop edge just past its far side; only
        # possible when the neighbour sits entirely outside the GT box on that axis
        cands = []
        if ox2 + 1 <= x2 and ox2 + 1 <= raw[0]:
            cands.append([ox2 + 1, y1, x2, y2])
        if ox1 - 1 >= x1 and ox1 - 1 >= raw[2]:
            cands.append([x1, y1, ox1 - 1, y2])
        if oy2 + 1 <= y2 and oy2 + 1 <= raw[1]:
            cands.append([x1, oy2 + 1, x2, y2])
        if oy1 - 1 >= y1 and oy1 - 1 >= raw[3]:
            cands.append([x1, y1, x2, oy1 - 1])
        if not cands:
            break
        box = max(cands, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    return [round(v, 1) for v in box]


def expand_with_clamp(raw, W, H, ratio):
    bw, bh = raw[2] - raw[0], raw[3] - raw[1]
    ex, ey = bw * ratio, bh * ratio
    x1, y1, x2, y2 = raw[0] - ex, raw[1] - ey, raw[2] + ex, raw[3] + ey
    if x1 < 0:
        x2 = min(W, x2 - x1); x1 = 0
    if y1 < 0:
        y2 = min(H, y2 - y1); y1 = 0
    if x2 > W:
        x1 = max(0, x1 - (x2 - W)); x2 = W
    if y2 > H:
        y1 = max(0, y1 - (y2 - H)); y2 = H
    return [x1, y1, x2, y2]


def sweep(by_img, ratios):
    """How much neighbour contamination does each expansion ratio buy?"""
    rows = []
    for ratio in ratios:
        contam = minor = clamped = pcontam = pminor = 0
        area_gain, parea, n_multi = [], [], 0
        for key, rws in by_img.items():
            W, H = rws[0]["image_size"]
            boxes = [r["bbox"] for r in rws]
            for i, r in enumerate(rws):
                raw = r["bbox"]
                exp = expand_with_clamp(raw, W, H, ratio)
                others = [b for j, b in enumerate(boxes) if j != i]
                a0 = max(1.0, (raw[2] - raw[0]) * (raw[3] - raw[1]))
                a1 = max(1.0, (exp[2] - exp[0]) * (exp[3] - exp[1]))
                if others:
                    n_multi += 1
                    hits = [(overlap(exp, o), overlap(raw, o)) for o in others]
                    after, before = max(hits, key=lambda t: t[0])
                    if after >= 0.5 > before:
                        contam += 1
                    elif after >= 0.15 > before:
                        minor += 1
                    pb = protected_expand(raw, others, W, H, ratio)
                    phits = [(overlap(pb, o), overlap(raw, o)) for o in others]
                    pafter, pbefore = max(phits, key=lambda t: t[0])
                    if pafter >= 0.5 > pbefore:
                        pcontam += 1
                    elif pafter >= 0.15 > pbefore:
                        pminor += 1
                    parea.append(((pb[2] - pb[0]) * (pb[3] - pb[1])) / a1)
                area_gain.append(a1 / a0)
                if (exp[2] - exp[0]) < (raw[2] - raw[0]) * (1 + 2 * ratio) - 1:
                    clamped += 1
        parea.sort()
        rows.append({"ratio": ratio,
                     "contaminated": contam,
                     "contaminated_pct": round(100 * contam / max(1, n_multi), 2),
                     "minor_touch_pct": round(100 * minor / max(1, n_multi), 2),
                     "protected_contaminated_pct": round(100 * pcontam / max(1, n_multi), 2),
                     "protected_minor_touch_pct": round(100 * pminor / max(1, n_multi), 2),
                     "protected_median_area_vs_naive":
                         round(parea[len(parea) // 2], 3) if parea else None,
                     "border_clamped_pct": round(100 * clamped / len(area_gain), 1),
                     "median_area_gain": round(sorted(area_gain)[len(area_gain) // 2], 2)})
    return rows


def main():
    inst = load_instances()
    by_img = defaultdict(list)
    for r in inst:
        by_img[r["image"]].append(r)

    swallowed = defaultdict(int)
    examples = []
    for key, rows in by_img.items():
        if len(rows) < 2:
            continue
        for r in rows:
            raw, exp = r["bbox"], r["bbox_expanded"]
            # judge each instance once, against whichever neighbour it swallows most
            hits = [(overlap(exp, o["bbox"]), o) for o in rows if o is not r]
            if not hits:
                continue
            after, o = max(hits, key=lambda t: t[0])
            before = overlap(raw, o["bbox"])
            if after >= 0.5 > before:
                swallowed["new_majority_overlap"] += 1
                swallowed["same_class" if o["class"] == r["class"] else "diff_class"] += 1
                if len(examples) < 400:
                    examples.append({"image": key, "class": r["class"],
                                     "other_class": o["class"],
                                     "overlap_before": round(before, 3),
                                     "overlap_after": round(after, 3)})
            elif after >= 0.15 > before:
                swallowed["new_minor_overlap"] += 1

    n_multi = sum(1 for rows in by_img.values() if len(rows) > 1)
    total_single = sum(len(rows) for rows in by_img.values() if len(rows) > 1)
    report = {
        "expand_ratio": EXPAND,
        "images_with_multiple_entities": n_multi,
        "instances_in_multi_entity_images": total_single,
        "expansion_newly_swallows_neighbour_majority": swallowed["new_majority_overlap"],
        "of_which_same_class": swallowed["same_class"],
        "of_which_different_class": swallowed["diff_class"],
        "expansion_newly_touches_neighbour_15pct": swallowed["new_minor_overlap"],
        "pct_of_multi_entity_instances_contaminated": round(
            100 * swallowed["new_majority_overlap"] / max(1, total_single), 2),
        "worst_examples": sorted(examples, key=lambda e: -e["overlap_after"])[:15],
    }

    # ---- contact sheet ----
    random.seed(3)
    pool = [r for r in inst if r["image"] in by_img]
    tiny = sorted(pool, key=lambda r: r["long_side"])[:200]
    wide = sorted(pool, key=lambda r: -max(r["bbox"][2] - r["bbox"][0], 1) /
                  max(1, r["bbox"][3] - r["bbox"][1]))[:200]
    border = [r for r in pool if any(
        (a <= 1 and b > 0) for a, b in zip(r["bbox"], r["image_size"]))][:200]
    picks = (random.sample(pool, 10) + random.sample(tiny, 6)
             + random.sample(wide, 6) + random.sample(border, 6))

    Cell, Lab = 200, 18
    cols = 4
    rows_n = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (Cell * 2 + 8), rows_n * (Cell + Lab + 8)), "white")
    dr = ImageDraw.Draw(sheet)
    cache = {}
    for i, r in enumerate(picks):
        key = r["image"]
        if key not in cache:
            cache.clear()
            cache[key] = Image.open(os.path.join(ROOT, r["scene_file"])).convert("RGB")
        im = cache[key]
        for j, (box, tag) in enumerate(((r["bbox"], "gt"), (r["bbox_expanded"], "exp"))):
            crop = im.crop([int(v) for v in box])
            cw, ch = crop.size
            s = min(Cell / max(1, cw), Cell / max(1, ch))
            crop = crop.resize((max(1, int(cw * s)), max(1, int(ch * s))))
            canvas = Image.new("RGB", (Cell, Cell), (245, 245, 245))
            canvas.paste(crop, ((Cell - crop.size[0]) // 2, (Cell - crop.size[1]) // 2))
            ox, oy = (i % cols) * (Cell * 2 + 8) + j * (Cell + 8), (i // cols) * (Cell + Lab + 8)
            sheet.paste(canvas, (ox, oy))
            dr.text((ox + 2, oy + Cell + 2),
                    "%s %s %dx%d %s" % (r["class"], tag, cw, ch,
                                        "%.1fx" % (1 / s if s else 0)),
                    fill="black")
    sheet.save(os.path.join(IDX, "region_preview.png"))
    report["contact_sheet"] = "index/region_preview.png"
    report["contact_sheet_samples"] = len(picks)

    # ---- expansion-ratio sweep, naive vs neighbour-protected ----
    report["ratio_sweep"] = sweep(by_img, [0.0, 0.05, 0.10, 0.15, 0.25, 0.40])
    at_expand = [s for s in report["ratio_sweep"] if s["ratio"] == EXPAND][0]
    report["at_configured_ratio"] = at_expand

    with open(os.path.join(IDX, "region_check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("worst_examples", "ratio_sweep")},
                     ensure_ascii=False, indent=2))
    print("\nworst contamination examples:")
    for e in report["worst_examples"][:8]:
        print("  %-10s vs %-10s before=%.2f after=%.2f"
              % (e["class"], e["other_class"], e["overlap_before"], e["overlap_after"]))
    print("\nexpansion-ratio sweep (multi-entity images only):")
    print("  ratio  naive_contam%  prot_contam%  naive_minor%  prot_minor%  clamped%  area  prot_area")
    for s in report["ratio_sweep"]:
        print("  %5.2f  %13.2f  %12.2f  %13.2f  %11.2f  %8.1f  %5.2f  %9.3f"
              % (s["ratio"], s["contaminated_pct"], s["protected_contaminated_pct"],
                 s["minor_touch_pct"], s["protected_minor_touch_pct"],
                 s["border_clamped_pct"], s["median_area_gain"],
                 s["protected_median_area_vs_naive"] or 0))


if __name__ == "__main__":
    main()
