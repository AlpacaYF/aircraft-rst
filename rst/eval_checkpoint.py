#!/usr/bin/env python
"""Evaluate a trained checkpoint on a held-out split, and diagnose the Text branch.

Produces the numbers tables 4-3 and 4-6 need (test-split Top-1 / Top-5 / Macro-F1,
military-only Top-1, and the attribute P/R/F1), which `rst.train` never computes:
it only ever validates on the validation split.

Two traps this avoids, both of which would silently invalidate a comparison:

1. `runs/<name>/config.json` records the STAGE-1 capability mask -- every modality
   off -- because train.py saves it before the stage loop mutates cfg. Reading it
   here would evaluate the full model as if it were region-only. The per-stage
   truth lives in the checkpoint's own "cfg" entry, so that is what we load.

2. beta (`sem_scale`) in eq. 3-17 multiplies a vector added to the class logits.
   Softmax gradients sum to zero across classes, so beta only receives gradient to
   the extent S_c actually differs between classes. If the 114 text prototypes are
   near-collinear, beta stays pinned at its initial value however long training
   runs, and S_c contributes a near-constant offset plus noise. `--diagnose`
   measures exactly that: the spread of S_c across classes, how often argmax S_c
   is the true class, and the prototypes' off-diagonal cosine.

  python -m rst.eval_checkpoint --run runs/align_full_kb --ckpt best.pt --split test
  python -m rst.eval_checkpoint --run runs/align_full_kb --ckpt best.pt --diagnose
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import EntityDataset, RSTCollator
from .eval import evaluate, format_result, macro_f1, stratify_by_size
from .model import RSTModel
from .train import load_semantic


def cfg_from_ckpt(ck, root):
    """The checkpoint's own cfg, not runs/<name>/config.json -- see trap 1."""
    cfg = Config()
    for k, v in ck["cfg"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
    cfg.root = root
    # A checkpoint trained before the gate floor existed must be scored without
    # one: alpha = eps + (1-3*eps)*softmax(g) changes H, so applying today's
    # default to an old run would silently alter its predictions and make the
    # before/after comparison meaningless.
    if "gate_floor" not in ck["cfg"]:
        cfg.gate_floor = 0.0
    return cfg


@torch.no_grad()
def diagnose(model, loader, device, cfg, store, max_batches=0):
    """Is the Text branch carrying class information, or just an offset?"""
    model.eval()
    alpha_sum, sem_all, sem_only_hit, y_all = [], [], [], []
    sem_raw_spread = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and cfg.amp)):
            out = model(batch)
        if out["gate"] is not None:
            alpha_sum.append(out["gate"].float().cpu())
        y_all.append(batch["class"].cpu())
        if "sem_raw" in out:
            sr = out["sem_raw"].float().cpu()
            sem_all.append(sr)
            sem_only_hit.append((sr.argmax(1) == batch["class"].cpu()))
            sem_raw_spread.append(sr.std(dim=1))
    y = torch.cat(y_all)
    # A plain_head run has no gate to report: forward() returns out["gate"] = None and
    # the classification path never touches 3.4.3. Say so in the JSON instead of either
    # crashing or printing a fabricated (1, 0, 0).
    gate = {} if alpha_sum else {"gate_alpha_mean": None, "plain_head": True}
    if alpha_sum:
        a = torch.cat(alpha_sum).mean(0)
        # The gate is the whole claim of 3.4.3, so report it even when the Text branch is
        # off -- a softmax gate that saturates to (1, 0, 0) means the fusion is not
        # dynamic at all and Scene/Text are being discarded. A weight on a disabled
        # branch is inert, not a leak: forward() zeroes that branch's feature, so alpha
        # multiplies a zero vector, and out_norm is a LayerNorm which is invariant to the
        # scale the wasted mass removes. Listed explicitly because reading "text: 0.173"
        # on a run trained with Text off otherwise looks exactly like a broken ablation.
        # tools/selftest_synthetic.py proves the inertness by scrambling the disabled
        # branch and requiring bit-identical logits.
        gate = {
            "gate_alpha_mean": {"region": round(float(a[0]), 4),
                                "scene": round(float(a[1]), 4),
                                "text": round(float(a[2]), 4)},
            "gate_alpha_min_over_classes": round(float(torch.cat(alpha_sum).min(0).values[1]), 4),
        }
    base = {
        "text_active": bool(sem_all),
        "n_entities": int(len(y)),
        "gate_branches_disabled": [n for n, on in (("scene", cfg.use_scene),
                                                   ("text", cfg.use_text)) if not on],
        **gate,
    }
    if not sem_all:
        return base

    sr = torch.cat(sem_all)                      # [N, C] = S_c / beta
    hit = torch.cat(sem_only_hit)

    # how collinear are the 114 class prototypes after the trained projection?
    tok, mask = model.class_text_tokens()        # [C, N, h], [C, N]
    m = mask.unsqueeze(-1).to(tok.dtype)
    proto = (tok * m).sum(1) / m.sum(1).clamp(min=1.0)
    p = F.normalize(proto.float(), dim=-1)
    G = p @ p.T
    C = G.size(0)
    off = G[~torch.eye(C, dtype=torch.bool, device=G.device)]

    beta = float(model.sem_scale.detach().cpu())
    base.update({
        "beta_sem_scale": beta,
        "beta_moved_from_init": round(beta - 10.0, 4),
        # S_c alone, ignoring the visual head entirely
        "sem_only_top1_pct": round(100 * float(hit.float().mean()), 2),
        "chance_pct": round(100.0 / C, 2),
        # spread of S_c across classes for one entity: this is what beta scales,
        # and what the softmax can actually see
        "sem_raw_std_across_classes_mean": round(float(torch.cat(sem_raw_spread).mean()), 5),
        "sem_raw_std_x_beta": round(beta * float(torch.cat(sem_raw_spread).mean()), 4),
        "sem_raw_range_mean": round(float((sr.max(1).values - sr.min(1).values).mean()), 5),
        "hard_margin_over_sem_std": round(
            cfg.hard_margin / max(1e-6, float(torch.cat(sem_raw_spread).mean())), 2),
        # prototype collinearity
        "proto_offdiag_cos_mean": round(float(off.mean()), 4),
        "proto_offdiag_cos_p99": round(float(off.quantile(0.99)), 4),
        "proto_offdiag_cos_max": round(float(off.max()), 4),
    })
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run", required=True, help="runs/<name>")
    ap.add_argument("--ckpt", default="best.pt", help="best.pt | last.pt | stage2.pt ...")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--region-input", type=int, default=0,
                    help="override the region crop resolution. SigLIP2 interpolates its "
                         "position embeddings, so a checkpoint trained at 384 can be "
                         "scored at 512 without retraining -- this tests the resolution "
                         "mismatch behind the >=384px accuracy deficit directly")
    ap.add_argument("--region-expand", dest="region_expand", type=float, default=None,
                    help="override the region context expansion ratio (default: the "
                         "checkpoint's value). The multi-scale TTA control evaluates "
                         "0.15/0.25/0.35 separately and averages logits offline")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--diag-batches", type=int, default=200)
    ap.add_argument("--dump-samples", action="store_true",
                    help="also write runs/<name>/samples_<split>.npz: per-entity logits "
                         "(with and without S_c), semantic score matrix, gate weights and "
                         "prediction. Subset analysis and the modality-complementarity "
                         "test are then done offline, without re-running the model")
    ap.add_argument("--out", default="", help="where to write the JSON (default: in the run dir)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    run_dir = args.run if os.path.isabs(args.run) else os.path.join(root, args.run)
    ck_path = os.path.join(run_dir, args.ckpt)
    if not os.path.exists(ck_path):
        raise SystemExit("no such checkpoint: %s" % ck_path)

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = cfg_from_ckpt(ck, root)
    if args.region_input:
        cfg.region_input = args.region_input
    if args.region_expand is not None:
        cfg.region_expand = args.region_expand
    cfg.num_workers = args.workers
    print("checkpoint : %s" % ck_path)
    print("region input %d px (%d tokens per region)"
          % (cfg.region_input, (cfg.region_input // 16) ** 2))
    print("stage cfg  : scene=%s text=%s hard=%s diff=%s mask=%s  (from the checkpoint, "
          "NOT config.json)" % (cfg.use_scene, cfg.use_text, cfg.use_hard_negative,
                                cfg.use_diff_loss, cfg.use_mask_channel))
    print("step %s | val at save time: %s" % (ck.get("step"), format_result(ck["val"]))
          if "val" in ck else "step %s" % ck.get("step"))

    protos, attr_matrix, _ = load_semantic(cfg)
    sd = ck["model"]
    attr_dim = int(sd["head_attr.weight"].shape[0]) if "head_attr.weight" in sd else 0
    model = RSTModel(cfg, protos, attr_dim=attr_dim)
    model.load_state_dict(sd)
    model.to(device).eval()

    ds = EntityDataset(cfg, args.split, train=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=RSTCollator(ds),
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=cfg.num_workers > 0)
    print("split %-10s entities %d | batches %d" % (args.split, len(ds), len(loader)))

    # table 4-7's Hard Pair Accuracy needs the confusable sets mined in 3.3.3
    hard_path = os.path.join(root, cfg.index_dir, "hard_table.npy")
    hard_table = np.load(hard_path) if os.path.exists(hard_path) else None
    if hard_table is None:
        print("NOTE %s missing -- Hard Pair Accuracy omitted" % hard_path)

    res = evaluate(model, loader, device, cfg, ds.store, desc="test", use_amp=cfg.amp,
                   return_samples=True, hard_table=hard_table)
    print("\nWITH the semantic term S_c (eq. 3-17 as written):")
    print("  " + format_result(res))
    strat = stratify_by_size(res, ds.store, len(ds.store.classes))
    print("  by entity long side (tests 3.2.3's small-target claim):")
    for k, v in strat.items():
        print("    %-9s n=%-5d top1 %6.2f  macroF1 %6.2f"
              % (k, v["n"], 100 * v["top1"], 100 * v["macro_f1"]))

    if args.dump_samples:
        # Everything 4.5's subset analysis and the modality-complementarity question need
        # is derivable from these five arrays, so the run is scored once and every later
        # question (which entities flip between configurations, where the gate puts its
        # weight, how large the semantic margin is) is answered offline instead of by
        # re-running the model or by training another variant.
        g = res["_got"]
        path = os.path.join(run_dir, "samples_%s.npz" % args.split)
        np.savez_compressed(
            path,
            ids=np.array(g["ids"]), y=g["y"],
            pred=res["_pred"],
            logits=g["logits"].numpy(),                    # as the table scores it
            logits_visual=g["logits_visual"].numpy(),      # eq. 3-17 without S_c
            sem_raw=g["sem_raw"].numpy(),                  # S_c / beta
            gate=g["gate"].numpy(),
            sem_scale=float(model.sem_scale.detach().cpu()),
            gate_present=bool(g["gate_present"]), sem_present=bool(g["sem_present"]),
            run=os.path.basename(run_dir.rstrip("/")), split=args.split,
            use_scene=bool(cfg.use_scene), use_text=bool(cfg.use_text),
            ckpt=args.ckpt)
        print("  per-sample dump -> %s (%d entities, gate=%s, S_c=%s)"
              % (path, len(g["ids"]), g["gate_present"], g["sem_present"]))

    res_no_sem = None
    if cfg.use_text:
        # Re-score with S_c dropped from eq. 3-17 only. The forward pass is
        # untouched, so z_t and the gated H stay exactly as trained: this
        # isolates "S_c is unhelpful at inference" from "the contrastive term
        # reshaped the visual features during training", which persists either way.
        res_no_sem = evaluate(model, loader, device, cfg, ds.store,
                              desc="test-noSem", use_amp=cfg.amp, use_semantic=False,
                              hard_table=hard_table)
        print("WITHOUT S_c at inference (same H, visual head only):")
        print("  " + format_result(res_no_sem))
        d = 100 * (res["macro_f1"] - res_no_sem["macro_f1"])
        print("  -> S_c changes Macro-F1 by %+.2f points at inference" % d)

    diag = None
    if args.diagnose:
        diag = diagnose(model, loader, device, cfg, ds.store, args.diag_batches)
        print("\ndiagnostics:")
        for k, v in diag.items():
            print("  %-34s %s" % (k, v))

    out = {"run": os.path.basename(run_dir), "ckpt": args.ckpt, "split": args.split,
           "region_input": cfg.region_input,
           "stage_cfg": {"use_scene": cfg.use_scene, "use_text": cfg.use_text,
                         "use_hard_negative": cfg.use_hard_negative,
                         "use_diff_loss": cfg.use_diff_loss,
                         "use_mask_channel": cfg.use_mask_channel},
           "with_semantic": {k: v for k, v in res.items() if not k.startswith("_")},
           "without_semantic": res_no_sem,
           "by_entity_size": strat, "diagnostics": diag}
    tag = "" if not args.region_input else "_r%d" % args.region_input
    path = args.out or os.path.join(run_dir, "eval_%s%s.json" % (args.split, tag))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
