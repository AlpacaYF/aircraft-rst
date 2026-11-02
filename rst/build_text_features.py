"""Turn the semantic knowledge base into the Text Features of 3.3.4.

Each class keeps its knowledge as *separate* tokens rather than one long
sentence, so the cross-attention in 3.4.2 can match a visual region to
"tail configuration" or "intake layout" specifically.

  python -m rst.build_text_features [--kb ...] [--out knowledge/text_features.npz]
"""

import argparse
import json
import os

import numpy as np
import torch

from .encoders import SigLIP2Encoder
from .config import Config

KIND_ORDER = ["desc", "attr", "feat", "diff"]


def collect(kb_path):
    with open(kb_path, encoding="utf-8") as fh:
        kb = json.load(fh)
    entries = {e["class"]: e for e in kb["entries"] if e.get("ok")}
    classes = sorted(entries)
    items = []          # (class_row, kind, text)
    for row, c in enumerate(classes):
        k = entries[c]["knowledge"]
        desc = (k.get("overall_description") or "").strip()
        if desc:
            items.append((row, "desc", desc))
        for a, v in (k.get("structural_attributes") or {}).items():
            v = str(v).strip()
            if v and v != "不可见":
                items.append((row, "attr", "%s：%s" % (a, v)))
        for f in k.get("discriminative_features") or []:
            if str(f).strip():
                items.append((row, "feat", str(f).strip()))
        for d in k.get("similar_class_differences") or []:
            if str(d).strip():
                items.append((row, "diff", str(d).strip()))
    return classes, items


def apply_verification(items, verdict_path):
    """3.3.2 output drops the sentences that failed visual consistency."""
    if not verdict_path or not os.path.exists(verdict_path):
        return items, None
    with open(verdict_path, encoding="utf-8") as fh:
        verdict = json.load(fh)
    dropped = {str(x) for x in verdict.get("drop", [])}
    return [(r, kind, t) for (r, kind, t) in items if t not in dropped], verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--kb", default="")
    ap.add_argument("--verdict", default="knowledge/kb_verification.json")
    ap.add_argument("--names-only", action="store_true",
                    help="one token per class holding just the class name. This is the "
                         "control that isolates the thesis's core claim: if the MLLM "
                         "attribute knowledge base does not beat bare class names under "
                         "an otherwise identical training run, the knowledge is not what "
                         "is helping. Write to a separate --out so the real Text Features "
                         "stay in place.")
    ap.add_argument("--out", default="knowledge/text_features.npz")
    ap.add_argument("--anchor-names", action="store_true",
                    help="prefix every sentence with its class name. Measured with the "
                         "CLIP-L tower on 600 crops: bare class names retrieve at "
                         "35.68%% but the knowledge sentences only at 11.83%%, because a "
                         "description like 'swept wing, twin engine, T-tail' fits dozens "
                         "of types while a name fits one. Anchoring keeps the attribute "
                         "content for L_diff and gives the retrieval side the one token "
                         "that is actually unique.")
    ap.add_argument("--proto-kinds", default="",
                    help="comma-separated sentence kinds to use as class prototypes, "
                         "e.g. 'diff' or 'desc,diff'. Measured on 600 test crops with the "
                         "CLIP-L tower: diff sentences retrieve at 11.83%% Top-1 and are "
                         "the best separated (prototype cosine 0.78), while attr and feat "
                         "sit at 2.00%%/1.83%% with cosine 0.93/0.89 -- attributes are "
                         "shared across types, so they cannot identify one. Mixing all "
                         "four kinds lets the 78%% generic sentences dilute the diff "
                         "signal (10.00%% overall). The attribute matrix still uses every "
                         "feat sentence so L_diff keeps working.")
    ap.add_argument("--strings", default="",
                    help="JSON written by tools/translate_kb_strings.py: encode THESE "
                         "strings instead of re-reading the knowledge base. Needed "
                         "because the KB is Chinese and an English-BPE tower (CLIP) "
                         "shreds it -- measured 8.00%% retrieval on Chinese versus "
                         "35.68%% for English class names.")
    ap.add_argument("--model", default="")
    ap.add_argument("--text-encoder", default="siglip2", choices=["siglip2", "clip"],
                    help="which frozen text tower encodes the knowledge base. The "
                         "SigLIP 2 tower pools at a position its fast tokenizer fills "
                         "with <pad>, and even after the left-padding fix its 114 class "
                         "embeddings stay nearly collinear (pairwise cosine 0.90-0.95) "
                         "so frozen retrieval sits at chance; CLIP-ViT-L/14 separates the "
                         "same strings to 0.63 and reaches 35.7%% zero-shot on these "
                         "crops. Prototypes are only comparable within one tower, so "
                         "always write a different --out.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--cache-dir", default="",
                    help="defaults to $HF_HOME set by env.sh")
    args = ap.parse_args()

    cfg = Config()
    cfg.root = os.path.abspath(args.root)
    kb_path = args.kb or os.path.join(cfg.root, "knowledge",
                                      "aircraft_semantic_kb_filtered.json")
    if not os.path.exists(kb_path):
        kb_path = os.path.join(cfg.root, "knowledge", "aircraft_semantic_kb.json")
    classes, items = collect(kb_path)
    if args.names_only:
        items = [(row, "name", c) for row, c in enumerate(classes)]
        args.verdict = ""                       # nothing to verify against
        print("names-only control: %d classes, 1 token each" % len(classes))
    if args.strings:
        with open(os.path.join(cfg.root, args.strings), encoding="utf-8") as fh:
            blob = json.load(fh)
        if [str(x) for x in blob["classes"]] != classes:
            raise SystemExit("class order in %s differs from %s -- refusing to mix "
                             "prototypes with the wrong labels" % (args.strings, kb_path))
        before = len(items)
        items = [(int(r), str(k), str(t)) for r, k, t in blob["items"] if str(t).strip()]
        print("strings from %s: %d of %d kept (%d empty dropped)"
              % (args.strings, len(items), before, before - len(items)))
        # the stored verdict was computed on the Chinese sentences, so its drop list
        # cannot apply to their translations
        args.verdict = ""
    items, verdict = apply_verification(items, os.path.join(cfg.root, args.verdict)
                                        if args.verdict else "")
    print("knowledge base : %s" % os.path.relpath(kb_path, cfg.root))
    print("classes %d | semantic strings %d" % (len(classes), len(items)))

    model_name = args.model or (cfg.vision_model if args.text_encoder == "siglip2"
                                else "openai/clip-vit-large-patch14")
    if args.text_encoder == "clip":
        from .baselines import CLIPEncoder
        enc = CLIPEncoder(model_name,
                          cache_dir=(os.path.join(cfg.root, args.cache_dir)
                                     if args.cache_dir else None))
    else:
        enc = SigLIP2Encoder(model_name, freeze=True,
                             cache_dir=(os.path.join(cfg.root, args.cache_dir)
                                        if args.cache_dir else None))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    if device == "cuda":
        enc = enc.to(torch.bfloat16)

    # attr_matrix is built from every feat sentence regardless of --proto-kinds, so
    # restricting the prototypes does not silently disable the difference loss
    all_items = items
    if args.proto_kinds:
        keep = {k.strip() for k in args.proto_kinds.split(",") if k.strip()}
        items = [x for x in items if x[1] in keep]
        print("prototypes restricted to kinds %s: %d of %d sentences"
              % (sorted(keep), len(items), len(all_items)))
        if not items:
            raise SystemExit("no sentence matches --proto-kinds %s" % args.proto_kinds)

    if args.anchor_names:
        items = [(r, k, "%s: %s" % (classes[r], t)) for (r, k, t) in items]
        print("anchored every sentence with its class name")

    texts = [t for _, _, t in items]
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            chunk = texts[i:i + args.batch]
            f = enc.encode_text(chunk)
            feats.append(f.float().cpu().numpy())
            if (i // args.batch) % 10 == 0:
                print("  encoded %d/%d" % (min(i + args.batch, len(texts)), len(texts)),
                      flush=True)
    feats = np.concatenate(feats, axis=0)

    lengths = enc.text_token_lengths(texts)
    n_trunc = sum(1 for n in lengths if n > enc.max_text_len)
    print("text tokens: max %d, limit %d, would-truncate %d (%.1f%%)"
          % (max(lengths), enc.max_text_len, n_trunc, 100 * n_trunc / max(1, len(lengths))))

    C = len(classes)
    per_class = np.bincount(np.array([r for r, _, _ in items], dtype=np.int64),
                            minlength=C)
    N = int(per_class.max()) if len(per_class) else 1
    tokens = np.zeros((C, N, feats.shape[1]), dtype=np.float32)
    mask = np.zeros((C, N), dtype=bool)
    seen = [0] * C
    for i, (row, kind, text) in enumerate(items):
        j = seen[row]
        if j < N:
            tokens[row, j] = feats[i]
            mask[row, j] = True
            seen[row] += 1

    feat_strings = sorted({t for _, k, t in all_items if k == "feat"})
    feat_index = {s: i for i, s in enumerate(feat_strings)}
    attr_matrix = np.zeros((C, len(feat_strings)), dtype=np.float32)
    for row, kind, text in all_items:
        if kind == "feat":
            attr_matrix[row, feat_index[text]] = 1.0

    out_path = os.path.join(cfg.root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path, tokens=tokens, mask=mask, features=feats,
        rows=np.array([r for r, _, _ in items], dtype=np.int32),
        attr_matrix=attr_matrix, attr_strings=np.array(feat_strings, dtype=object),
        classes=np.array(classes, dtype=object),
        strings=np.array([t for _, _, t in items], dtype=object),
        kinds=np.array([k for _, k, _ in items], dtype=object),
        text_dim=np.array([feats.shape[1]], dtype=np.int32))
    print("wrote %s  tokens%s  attr_matrix%s" % (out_path, tokens.shape, attr_matrix.shape))


if __name__ == "__main__":
    main()
