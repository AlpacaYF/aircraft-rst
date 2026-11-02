#!/usr/bin/env python
"""Direction-4 KB upgrade: pairwise difference sentences from a real VLM.

For each of the hardest confusable class pairs (mined in 3.3.3 from real visual
similarity), crop a few TRAINING images of class A, ask Qwen3-VL (served with
vLLM on localhost) what distinguishes an A from a B as visible in that photo,
and keep the answer as a pairwise-difference sentence attached to class A.
Both directions of every pair are generated.

Output: knowledge/pairdiff_strings.json in the same {classes, items} format
tools/translate_kb_strings.py writes, so rst.build_text_features --strings can
encode it unchanged. The file contains the ORIGINAL english diff sentences plus
the new pairwise ones (both kind "diff"), so the prototype selection
(--proto-kinds diff) sees the old set plus the upgrades.

  python -m tools.gen_pairdiff_kb --base knowledge/kb_strings_en.json \
      --pairs knowledge/hard_pairs.json --topk 50 --out knowledge/pairdiff_strings.json
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:8321/v1/chat/completions"
MODEL = "Qwen3-VL-8B-Instruct"

PROMPT = ("This photo shows a {a} aircraft. It is frequently confused with the "
          "{b}. Based ONLY on structural features visible in this photo (wing plan, "
          "engine count and placement, tail, fuselage, intakes, nose), write one "
          "short factual English sentence describing how a {a} differs from a {b}. "
          "Start exactly with: \"Compared to the {b}, the {a}\". "
          "If the photo is unclear, answer with the single word UNKNOWN.")


def norm_name(name):
    """F18 -> F-18, Boeing787 -> Boeing 787, EF2000 stays EF-2000-ish; the VLM
    understands the dashed forms reliably and the raw strings are sometimes
    ambiguous."""
    m = re.match(r"^(Boeing|Airbus)(\d+)$", name)
    if m:
        return "%s %s" % (m.group(1), m.group(2))
    m = re.match(r"^([A-Za-z]+)(\d+[A-Za-z]*)$", name)
    if m:
        return "%s-%s" % (m.group(1), m.group(2))
    return name


def crop_b64(scene_path, bbox, margin=0.15, max_side=448, quality=88):
    img = Image.open(scene_path).convert("RGB")
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    w, h = x1 - x0, y1 - y0
    x0, y0 = x0 - w * margin, y0 - h * margin
    x1, y1 = x1 + w * margin, y1 + h * margin
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    sub = img.crop((int(x0), int(y0), int(x1), int(y1)))
    s = max_side / max(1, max(sub.size))
    if s < 1:
        sub = sub.resize((max(1, int(sub.width * s)), max(1, int(sub.height * s))),
                         Image.BILINEAR)
    buf = io.BytesIO()
    sub.save(buf, format="JPEG", quality=quality)
    img.close()
    return base64.b64encode(buf.getvalue()).decode()


def ask(a_name, b_name, b64, retries=2):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text",
                 "text": PROMPT.format(a=norm_name(a_name), b=norm_name(b_name))},
            ],
        }],
        "max_tokens": 64,
        "temperature": 0.2,
    }
    for t in range(retries + 1):
        try:
            r = requests.post(BASE_URL, json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            txt = re.sub(r"\s+", " ", txt)
            if "UNKNOWN" in txt.upper() and len(txt) < 40:
                return None
            # keep only the sentence itself
            m = re.search(r"(Compared to.*)", txt)
            if m:
                txt = m.group(1)
            if 20 <= len(txt) <= 300 and "Compared to" in txt:
                return txt
            return None
        except Exception as e:
            if t == retries:
                print("  FAIL %s vs %s: %s" % (a_name, b_name, str(e)[:90]))
                return None
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--base", default="knowledge/kb_strings_en.json",
                    help="existing english strings (their diff sentences are kept)")
    ap.add_argument("--pairs", default="knowledge/hard_pairs.json")
    ap.add_argument("--index", default="index/instances.jsonl")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--images-per-class", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="knowledge/pairdiff_strings.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)

    with open(args.base, encoding="utf-8") as fh:
        base = json.load(fh)
    classes = [str(c) for c in base["classes"]]
    cidx = {c: i for i, c in enumerate(classes)}
    old_items = [[int(r), str(k), str(t)] for r, k, t in base["items"]]

    with open(args.pairs, encoding="utf-8") as fh:
        pairs = json.load(fh)["hardest_pairs"][:args.topk]
    print("classes %d | base items %d | pair candidates %d"
          % (len(classes), len(old_items), len(pairs)))

    # training images per class, up to --images-per-class
    shots = {}
    with open(args.index, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("split") != "train":
                continue
            shots.setdefault(d["class"], []).append(d)
    for c in shots:
        shots[c] = shots[c][:args.images_per_class]

    tasks = []          # (classA, classB, record)
    for p in pairs:
        a, b = p["a"], p["b"]
        if a not in cidx or b not in cidx:
            continue
        for (x, y) in ((a, b), (b, a)):
            for rec in shots.get(x, []):
                tasks.append((x, y, rec))
    print("generation tasks: %d (pairs x 2 directions x %d images)"
          % (len(tasks), args.images_per_class))

    done = [0]
    new_sentences = {}

    def work(t):
        x, y, rec = t
        b64 = crop_b64(rec["scene_file"], rec["bbox"])
        txt = ask(x, y, b64)
        done[0] += 1
        if done[0] % 25 == 0:
            print("  %d/%d" % (done[0], len(tasks)), flush=True)
        return (x, txt)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for x, txt in ex.map(work, tasks):
            if txt:
                new_sentences.setdefault(x, []).append(txt)

    # dedupe, keep at most 4 pairwise sentences per class
    items = list(old_items)
    n_new = 0
    for c, sents in sorted(new_sentences.items()):
        seen = set()
        kept = 0
        for s in sents:
            if s in seen:
                continue
            seen.add(s)
            items.append([cidx[c], "diff", s])
            kept += 1
            n_new += 1
            if kept >= 4:
                break
    print("kept %d new pairwise sentences across %d classes"
          % (n_new, len(new_sentences)))

    out = {"classes": classes, "items": items}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote %s  (items %d -> %d)" % (args.out, len(old_items), len(items)))
    print("sample:", items[len(old_items)][2][:110] if len(items) > len(old_items) else "(none)")


if __name__ == "__main__":
    main()
