#!/usr/bin/env python
"""Is the SigLIP 2 zero-shot row a real measurement, or is the checkpoint being loaded
by the wrong model class?

The number in 表4-2 is 1.00% Top-1 -- chance for 114 classes -- while CLIP-ViT-L/14 on
the identical crops and the identical prompts gets 35.68%. Two web-scale contrastive
models do not differ by 35x on the same task, so something is wrong on our side.

The suspicion this settles: transformers 4.49.0 (the training venv) predates proper
SigLIP 2 support and `AutoModel` maps the checkpoint to the SigLIP **v1** class. If
components are dropped or reinterpreted, the vision tower can still produce usable
features -- and it does: frozen mean-pooled features reach 38.6% prototype Top-1 and
44.8% 1-NN -- while the image/text *alignment* is destroyed, which is exactly the
observed symptom (text embeddings for the 114 class names have mean pairwise cosine
0.97, i.e. nearly collinear).

Run this under BOTH venvs and compare. Any difference in class name, projection layers,
or loading warnings is the answer.

  .venv/bin/python     -m tools.probe_siglip_loader --n 400 --gpu 3
  .venv-det/bin/python -m tools.probe_siglip_loader --n 400 --gpu 3
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rst.baselines import RegionCollator
from rst.config import Config
from rst.data import EntityDataset
from rst.eval import macro_f1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="google/siglip2-base-patch16-384")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--template", default="a photo of a {c} aircraft")
    ap.add_argument("--pad-side", default="right", choices=["left", "right"],
                    help="SiglipTextTransformer pools at last_hidden_state[:, -1, :] "
                         "assuming 'sticky' EOS tokenisation, i.e. that the final "
                         "position IS the terminator. A right-padded fast tokenizer "
                         "puts a <pad> there instead, so the embedding is pooled from "
                         "a padding token. Left padding restores the assumption.")
    args = ap.parse_args()

    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = Config()
    cfg.root = os.path.abspath(args.root)
    os.chdir(cfg.root)
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("transformers %s | torch %s | model %s" % (transformers.__version__,
                                                     torch.__version__, args.model))
    c = AutoConfig.from_pretrained(args.model)
    print("config class %-18s model_type %-10s architectures %s"
          % (type(c).__name__, getattr(c, "model_type", None), getattr(c, "architectures", None)))
    print("projection_dim %s | text hidden %s | vision hidden %s | max_position_embeddings %s"
          % (getattr(c, "projection_dim", None),
             getattr(getattr(c, "text_config", None), "hidden_size", None),
             getattr(getattr(c, "vision_config", None), "hidden_size", None),
             getattr(getattr(c, "text_config", None), "max_position_embeddings", None)))

    m, info = AutoModel.from_pretrained(args.model, output_loading_info=True)
    print("loaded as %s" % type(m).__name__)
    for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "_tied_weights_keys"):
        v = info.get(k) or []
        print("  %-18s %d %s" % (k, len(v), (v[:4] if v else "")))
    for attr in ("visual_projection", "text_projection", "logit_scale", "logit_bias"):
        obj = getattr(m, attr, None)
        shape = tuple(obj.weight.shape) if hasattr(obj, "weight") else None
        print("  %-18s %s %s" % (attr, type(obj).__name__ if obj is not None else "ABSENT",
                                 shape or ""))
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = args.pad_side
    ids = tok("a photo of an aircraft")["input_ids"]
    padded = tok(text=["a photo of an aircraft", "x"], padding="max_length",
                 max_length=64, return_tensors="pt")
    last_id = int(padded["input_ids"][0, -1])
    print("tokenizer %s | pad_side %s | eos %r pad %r" % (type(tok).__name__,
                                                          tok.padding_side,
                                                          tok.eos_token, tok.pad_token))
    print("  unpadded ids -> %s" % tok.convert_ids_to_tokens(ids))
    print("  POOLED POSITION (-1) holds %r  <- SiglipTextTransformer reads exactly this"
          % tok.convert_ids_to_tokens([last_id])[0])

    # ---- data ------------------------------------------------------------
    meta = json.load(open(os.path.join(cfg.index_dir, "class_meta.json"), encoding="utf-8"))
    classes = [str(x) for x in meta["classes"]]
    ds = EntityDataset(cfg, "test", train=False)
    if args.n and args.n < len(ds):
        ds.records = ds.records[:args.n]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=RegionCollator(ds), pin_memory=(device.type == "cuda"))
    m = m.to(device).eval()

    texts = [args.template.replace("{c}", x) for x in classes]
    enc = tok(text=texts, padding="max_length", truncation=True, max_length=64,
              return_tensors="pt", return_attention_mask=True)
    t = F.normalize(m.get_text_features(input_ids=enc["input_ids"].to(device),
                                        attention_mask=enc["attention_mask"].to(device))
                    .float(), dim=-1)
    G = (t @ t.T).fill_diagonal_(0)
    print("\ntext side: mean pairwise cos %.4f (a healthy tower is well below 0.9)"
          % float(G.mean()))

    imgs, ys = [], []
    for b in loader:
        px = b["region"][:, :3].to(device)
        if px.shape[-1] != getattr(m.config.vision_config, "image_size", 384):
            px = F.interpolate(px, size=m.config.vision_config.image_size,
                               mode="bilinear", align_corners=False)
        imgs.append(F.normalize(m.get_image_features(pixel_values=px).float(), dim=-1).cpu())
        ys.append(b["class"])
    v = torch.cat(imgs)
    y = torch.cat(ys).numpy()
    pred = (v @ t.cpu().T).argmax(1).numpy()
    print("zero-shot on %d crops: Top-1 %.2f%% (chance %.2f%%)  MacroF1 %.2f%%  "
          "distinct classes %d"
          % (len(y), 100 * float((pred == y).mean()), 100 / len(classes),
             100 * macro_f1(y, pred, len(classes))[0], len(np.unique(pred))))
    print("mean image-text cosine %+.4f (CLIP on the same crops: about +0.18)"
          % float((v @ t.cpu().T).mean()))

    print("\n-- coarse sanity: every crop is an aircraft --")
    for good, bad in (("a photo of an aircraft", "a photo of a banana"),
                      ("a photo of an airplane", "a photo of a dog"),
                      ("a photo of a military jet fighter", "a photo of a bowl of soup")):
        e = tok(text=[good, bad], padding="max_length", truncation=True, max_length=64,
                return_tensors="pt", return_attention_mask=True)
        tt = F.normalize(m.get_text_features(input_ids=e["input_ids"].to(device),
                                             attention_mask=e["attention_mask"].to(device))
                         .float(), dim=-1)
        win = float(((v[:64].to(device) @ tt.T).argmax(1) == 0).float().mean())
        print("   '%s' beats '%s' on %.0f%% of crops | cos(prompts) %.4f"
              % (good.split(" of a ")[-1], bad.split(" of a ")[-1], 100 * win,
                 float(tt[0] @ tt[1])))

    # ---- controls that separate "our code is wrong" from "the model cannot do
    # fine-grained aircraft type names" --------------------------------------
    # (a) military vs civilian: a 2-way decision on the same crops with plain English
    #     prompts. If the image/text alignment were broken this would sit at ~50%.
    dom = np.asarray([0 if ds.store.domain_of[classes[yi]] == "military" else 1
                      for yi in y])
    e = tok(text=["a photo of a military aircraft", "a photo of a civilian airliner"],
            padding="max_length", truncation=True, max_length=64, return_tensors="pt",
            return_attention_mask=True)
    tt = F.normalize(m.get_text_features(input_ids=e["input_ids"].to(device),
                                         attention_mask=e["attention_mask"].to(device))
                     .float(), dim=-1)
    pd = (v.to(device) @ tt.T).argmax(1).cpu().numpy()
    print("\ncontrol A  military-vs-civilian 2-way: %.2f%% correct (chance 50%%, n=%d)"
          % (100 * float((pd == dom).mean()), len(dom)))

    # (b) the same measurement through the model's OWN processor, starting from PIL
    #     pixels, so none of our normalisation or tokenisation is in the loop. Needs
    #     sentencepiece; reported as skipped when it is not installed.
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model)
    except Exception as exc:
        print("control B  skipped: AutoProcessor unavailable (%s)" % str(exc)[:90])
    else:
        from PIL import Image
        pil, py = [], []
        for b in loader:
            for i in range(b["region"].shape[0]):
                arr = ((b["region"][i, :3].permute(1, 2, 0).numpy() + 1.0)
                       * 127.5).clip(0, 255).astype(np.uint8)
                pil.append(Image.fromarray(arr))
                py.append(int(b["class"][i]))
            if len(pil) >= args.n:
                break
        inp = proc(text=texts, images=pil, padding="max_length", truncation=True,
                   max_length=64, return_tensors="pt",
                   return_attention_mask=True).to(device)
        tf = F.normalize(m.get_text_features(input_ids=inp["input_ids"],
                                             attention_mask=inp["attention_mask"]).float(),
                         dim=-1).cpu()
        vf = F.normalize(m.get_image_features(pixel_values=inp["pixel_values"]).float(),
                         dim=-1).cpu()
        py = np.asarray(py)
        pp = (vf @ tf.T).argmax(1).numpy()
        Gp = (tf @ tf.T).fill_diagonal_(0)
        print("control B  official AutoProcessor from PIL: Top-1 %.2f%%  MacroF1 %.2f%%  "
              "text pairwise cos %.4f  mean img-text cos %+.4f  (n=%d)"
              % (100 * float((pp == py).mean()), 100 * macro_f1(py, pp, len(classes))[0],
                 float(Gp.mean()), float((vf @ tf.T).mean()), len(py)))
        inp2 = proc(text=["a photo of a military aircraft", "a photo of a civilian airliner"],
                    padding="max_length", truncation=True, max_length=64,
                    return_tensors="pt", return_attention_mask=True).to(device)
        t2 = F.normalize(m.get_text_features(input_ids=inp2["input_ids"],
                                             attention_mask=inp2["attention_mask"]).float(),
                         dim=-1)
        d2 = np.asarray([0 if ds.store.domain_of[classes[c]] == "military" else 1
                         for c in py])
        print("control B2 military-vs-civilian via processor: %.2f%% correct"
              % (100 * float(((vf.to(device) @ t2.T).argmax(1).cpu().numpy() == d2).mean())))
    print("=" * 78)


if __name__ == "__main__":
    main()
