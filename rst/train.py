"""Three-stage training for the RST model (3.5.1 - 3.5.3).

Stage 1  Region only                     -> visual basis
Stage 2  + Scene + Text                  -> multimodal fusion
Stage 3  + hard negatives + diff loss    -> fine-grained discrimination

Stages share one model and one optimiser; a stage only decides which branches
and losses are live, so progress carries over instead of restarting.

  python -m rst.train --epochs 30 --batch 16
  python -m rst.train --stages 1            # baseline only
  python -m rst.train --ablation region_only   # tables 4-6 / 4-7
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import EntityDataset, RSTCollator, BalancedEntitySampler
from .eval import evaluate, format_result
from .losses import RSTLoss, entity_metrics
from .model import RSTModel

try:
    from tqdm import tqdm
except ImportError:                                    # pragma: no cover
    def tqdm(x, **kw):
        return x

# which switches each stage turns on
STAGE_FLAGS = {
    1: dict(use_scene=False, use_text=False, use_hard_negative=False, use_diff_loss=False),
    2: dict(use_scene=True, use_text=True, use_hard_negative=False, use_diff_loss=False),
    3: dict(use_scene=True, use_text=True, use_hard_negative=True, use_diff_loss=True),
}

# named runs that reproduce the paper's ablation rows
#
# Every variant keeps the full 1-2-3 stage curriculum and the same budget as the
# full model; only `caps` differs. Giving the ablations stages=(3,) instead would
# have compared a 15~20 epoch single-stage run against the full model's 50 epoch
# curriculum, so table 4-6/4-7 would have measured the curriculum rather than the
# modality. For region_only the later stages simply have nothing left to add.
ABLATIONS = {
    "full": {},
    "region_only": dict(use_scene=False, use_text=False, use_hard_negative=False,
                        use_diff_loss=False),
    "region_scene": dict(use_scene=True, use_text=False, use_hard_negative=False,
                         use_diff_loss=False),
    "region_text": dict(use_scene=False, use_text=True, use_hard_negative=False,
                        use_diff_loss=False),
    "no_hard": dict(use_hard_negative=False, use_diff_loss=False),
    "no_diff": dict(use_diff_loss=False),
    "no_mask": dict(use_mask_channel=False),
}

CAP_KEYS = ("use_scene", "use_text", "use_hard_negative", "use_diff_loss")

# 3.5 as revised: a stage runs at least its lower budget and may end early once
# validation military Macro-F1 has been flat for PLATEAU epochs.
STAGE_BUDGETS = {1: (10, 10), 2: (15, 20), 3: (15, 20)}
PLATEAU = 3


def default_caps():
    return {k: True for k in CAP_KEYS}


def caps_from(ablation="", overrides=None):
    """What the run is allowed to turn on. `overrides` comes from explicit
    --use-*/--no-use-* flags and wins over the named ablation."""
    caps = default_caps()
    if ablation:
        for k, v in ABLATIONS[ablation].items():
            if k in CAP_KEYS:
                caps[k] = bool(v)
    for k, v in (overrides or {}).items():
        if k in CAP_KEYS and v is not None:
            caps[k] = bool(v)
    return caps


def plateau_advance(watch, best_stage, stall, epochs_done, lo_b, plateau, auto):
    """3.5 stage hand-off, as a pure function so it can be tested without training.

    A stage may end early once its lower budget `lo_b` is met and the watched
    metric (validation military Macro-F1) has been flat for `plateau` epochs.
    Before the lower budget, a plateau is ignored on purpose: leaving a stage
    after two lucky epochs would be worse than finishing its budget.
    """
    if watch > best_stage + 1e-4:
        return watch, 0, False
    stall += 1
    advance = bool(auto and epochs_done >= lo_b and stall >= plateau)
    return best_stage, stall, advance


def resolve_stage(cfg, stage, caps):
    """A stage says what MAY be on; caps say what is ALLOWED. Intersect them."""
    for k in CAP_KEYS:
        setattr(cfg, k, bool(STAGE_FLAGS[stage][k]) and caps[k])
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_param_groups(model, cfg):
    """Backbone and fusion get different learning rates; biases and norm weights
    skip weight decay."""
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith(("region_enc.", "scene_enc.")) else head).append(p)

    def split(params, lr):
        decay = [p for p in params if p.ndim > 1]
        nodecay = [p for p in params if p.ndim <= 1]
        return [{"params": decay, "lr": lr, "base_lr": lr, "weight_decay": cfg.weight_decay},
                {"params": nodecay, "lr": lr, "base_lr": lr, "weight_decay": 0.0}]
    return split(backbone, cfg.backbone_lr) + split(head, cfg.head_lr)


def lr_at(base_lr, step, total, warmup):
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def load_semantic(cfg):
    path = os.path.join(cfg.root, getattr(cfg, "text_features",
                                          "knowledge/text_features.npz"))
    if not os.path.exists(path):
        print("WARNING: %s missing -- Text branch disabled" % path)
        return None, None, 0
    z = np.load(path, allow_pickle=True)
    protos = {"tokens": torch.from_numpy(z["tokens"]),
              "mask": torch.from_numpy(z["mask"]),
              "class_of_token": None}
    return protos, torch.from_numpy(z["attr_matrix"]), z["tokens"].shape[-1]


@torch.no_grad()
def remine_hard_table(model, val_loader, device, cfg, store, old_table, topk=8):
    """3.5.3 as revised: the confusable sets for stage 3 come from THIS model's
    validation confusion, not from the frozen SigLIP2 similarity the 3.3.3 table
    was mined with. A class that confuses the model but never appeared in the
    frozen-embedding neighbours (and vice versa) is exactly the disagreement this
    catches. Rows keep the 3.3.3 neighbours as filler so the table stays dense.

    Returns the new [C, K] table; the caller decides whether to install it.
    """
    from .eval import evaluate as _evaluate
    res = _evaluate(model, val_loader, device, cfg, store, desc="remine",
                    use_amp=cfg.amp, return_samples=True)
    y, pred = np.asarray(res["_pred"]), np.asarray(res["_y"])
    C = len(store.classes)
    conf = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y, pred):
        if t != p:
            conf[t, p] += 1
    old = np.asarray(old_table, dtype=np.int64)
    new = old.copy()
    n_rows = 0
    for c in range(C):
        errs = [int(e) for e in np.argsort(-conf[c]) if conf[c, e] > 0 and e != c][:topk]
        if not errs:
            continue
        fill = [int(v) for v in old[c] if int(v) != c and int(v) not in errs]
        row = errs + fill
        row = (row + [c] * topk)[:topk]
        new[c] = row
        n_rows += 1
    pairs = int((conf.sum() + 1) // 2) if conf.sum() else 0
    print("remine: %d/%d val confusions, %d/%d classes get a refreshed hard row"
          % (int(conf.sum()), len(y), n_rows, C))
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=None,
                    help="run-to-run noise is 2.05 Macro-F1 points, so a single "
                         "ablation comparison says nothing on its own")
    ap.add_argument("--text-features", dest="text_features", default=None,
                    help="point the Text branch at a different prototype source, e.g. "
                         "knowledge/text_features_names.npz for the class-name-only control")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=None)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    ap.add_argument("--backbone-lr", dest="backbone_lr", type=float, default=None)
    ap.add_argument("--head-lr", dest="head_lr", type=float, default=None)
    ap.add_argument("--stage-epochs", type=int, default=0)
    ap.add_argument("--stage-switch", default="auto", choices=["auto", "fixed"],
                    help="auto: leave a stage once its lower budget is met and "
                         "validation military Macro-F1 has been flat for --plateau "
                         "epochs (3.5); fixed: run each stage for --stage-epochs")
    ap.add_argument("--plateau", type=int, default=PLATEAU)
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ablation", default="", choices=[""] + list(ABLATIONS))
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="")
    ap.add_argument("--cache-dir", default=".cache/hf")
    ap.add_argument("--device", default="")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=0,
                    help="limit validation to N batches for fast dev loops")
    ap.add_argument("--region-pool", dest="region_query_pool", type=int, default=None,
                    help="pool the region patch grid to NxN; 8 halves attention cost")
    ap.add_argument("--scene-max-tiles", dest="scene_max_tiles", type=int, default=None,
                    help="cap tiles encoded per scene; the main scene-memory knob")
    ap.add_argument("--scene-token-cap", dest="scene_token_cap", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None,
                    help="fusion width override; the DEDF capacity-matched control "
                         "runs --hidden 1536 (default 768, 12 heads divide it)")
    ap.add_argument("--scene-cache", dest="use_scene_cache", action="store_true",
                    help="read pooled scene features from cache/ instead of encoding "
                         "tiles each step (run rst.cache_scene_features first)")
    ap.add_argument("--resume", default="")
    for f in ("use_scene", "use_text", "use_hard_negative", "use_diff_loss",
              "use_mask_channel", "balance_class_loss", "freeze_vision",
              "grad_checkpoint", "use_scene_cache", "sem_at_inference", "plain_head",
              "gate_active_only"):
        ap.add_argument("--" + f.replace("_", "-"), dest=f, action="store_true")
        ap.add_argument("--no-" + f.replace("_", "-"), dest=f, action="store_false")
        ap.set_defaults(**{f: None})
    ap.add_argument("--lambda-con", dest="lambda_con", type=float, default=None)
    ap.add_argument("--lambda-hard", dest="lambda_hard", type=float, default=None)
    ap.add_argument("--lambda-diff", dest="lambda_diff", type=float, default=None)
    # --- v2 mechanisms; every one defaults to the v1 behaviour ---
    ap.add_argument("--z-s-env", dest="z_scene_env", action="store_true", default=None,
                    help="z_S pools environment tiles only (far tiles), so the Scene "
                         "branch stops duplicating what F_RS already fused")
    ap.add_argument("--proto-learnable", dest="proto_learnable", action="store_true",
                    default=None, help="KB-initialised learnable prototypes P=P_kb+delta")
    ap.add_argument("--margin-hard", dest="margin_hard", action="store_true", default=None,
                    help="replace the dead hinge with an additive-margin CE over "
                         "{y} u hard(y)")
    ap.add_argument("--margin-m", dest="margin_m", type=float, default=None)
    ap.add_argument("--margin-tau", dest="margin_tau", type=float, default=None)
    ap.add_argument("--con-hard-weight", dest="con_hard_weight", type=float, default=None,
                    help="upweight hard negatives in L_con's denominator (1 = off)")
    ap.add_argument("--remine-stage3", dest="remine_stage3", action="store_true",
                    default=None, help="rebuild the hard table from validation "
                         "confusion at the stage-2 -> stage-3 boundary")
    ap.add_argument("--erase-prob", dest="erase_prob", type=float, default=None,
                    help="probability of random-erasing one rectangle on a train region")
    ap.add_argument("--expand-jitter", dest="expand_jitter", type=float, default=None,
                    help="+/- jitter of the context expansion, sampled per entity")
    ap.add_argument("--few-shot", dest="few_shot", type=int, default=None,
                    help="cap TRAIN entities per class at K (fixed shared subset across "
                         "configurations); val/test stay full. Route-B knowledge-value "
                         "experiment: hard_table and text_features are label-independent "
                         "priors whose relative value should grow as K shrinks")
    ap.add_argument("--few-shot-seed", dest="few_shot_seed", type=int, default=None,
                    help="subset sampling seed; same seed + same K => identical subset "
                         "for every configuration")
    # --- Route-C text-path probes ---
    ap.add_argument("--proto-head", dest="proto_head", action="store_true", default=None,
                    help="classifier weight IS the pooled class prototypes (text path "
                         "is structurally inside the decision)")
    ap.add_argument("--adaptive-sem", dest="adaptive_sem", action="store_true",
                    default=None, help="logits = (1-w)*visual + w*S_c, w learned "
                         "from the visual confidence/margin (mutually exclusive with "
                         "--proto-head)")
    ap.add_argument("--con-detach", dest="con_detach", action="store_true", default=None,
                    help="contrastive terms see a detached visual side, so alignment "
                         "only trains text_proj/prototypes and not the visual tower")
    args = ap.parse_args()

    cfg = Config()
    cfg.root = os.path.abspath(args.root)

    overrides = {k: getattr(args, k) for k in CAP_KEYS}
    overrides["use_mask_channel"] = args.use_mask_channel
    caps = caps_from(args.ablation, overrides)

    for k, v in vars(args).items():
        if v is None or k in CAP_KEYS or k in ("no_amp", "ablation", "stages",
                                               "root", "name", "model"):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if args.use_mask_channel is not None:
        cfg.use_mask_channel = bool(args.use_mask_channel)
    elif args.ablation and "use_mask_channel" in ABLATIONS[args.ablation]:
        cfg.use_mask_channel = bool(ABLATIONS[args.ablation]["use_mask_channel"])
    if args.model:
        cfg.vision_model = args.model
    if args.batch:
        cfg.batch_size = args.batch
    if args.workers:
        cfg.num_workers = args.workers
    if args.max_batches:
        cfg.max_train_batches = args.max_batches
    cfg.amp = cfg.amp and not args.no_amp
    if cfg.proto_head and cfg.adaptive_sem:
        raise SystemExit("--proto-head and --adaptive-sem are mutually exclusive: "
                         "with a prototype-parameterised classifier there is no "
                         "separate visual side for the adaptive weight to gate")
    if cfg.proto_head and not (caps["use_text"] and cfg.sem_at_inference):
        raise SystemExit("--proto-head needs the Text branch live at inference "
                         "(caps use_text=%s, sem_at_inference=%s); it would be inert "
                         "here" % (caps["use_text"], cfg.sem_at_inference))
    stages = list(args.stages)
    if args.ablation and "stages" in ABLATIONS[args.ablation]:
        stages = list(ABLATIONS[args.ablation]["stages"])
    resolve_stage(cfg, stages[0], caps)

    set_seed(cfg.seed)
    run_name = args.name or time.strftime("%Y%m%d_%H%M%S") + (
        "_" + args.ablation if args.ablation else "")
    out_dir = os.path.join(cfg.root, cfg.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    cfg.save(os.path.join(out_dir, "config.json"))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device: %s" % device)
    if device.type == "cuda":
        print("gpu: %s  (%.1f GB)" % (torch.cuda.get_device_name(0),
                                      torch.cuda.get_device_properties(0).total_memory / 1e9))

    train_ds = EntityDataset(cfg, "train", train=True)
    val_ds = EntityDataset(cfg, "validation", train=False)
    collate = RSTCollator(train_ds)
    sampler = (BalancedEntitySampler(train_ds, cfg.seed)
               if cfg.sampler == "balanced" else None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=cfg.num_workers,
                              collate_fn=collate, pin_memory=(device.type == "cuda"),
                              drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=max(2, cfg.num_workers // 2),
                            collate_fn=RSTCollator(val_ds),
                            pin_memory=(device.type == "cuda"),
                            persistent_workers=cfg.num_workers > 0)
    print("train entities %d (%s sampler) | val %d | %.0f batches/epoch"
          % (len(train_ds), cfg.sampler, len(val_ds), len(train_loader)))

    protos, attr_matrix, text_dim = load_semantic(cfg)
    hard_path = os.path.join(cfg.root, cfg.index_dir, "hard_table.npy")
    hard_table = (torch.from_numpy(np.load(hard_path)).long()
                  if os.path.exists(hard_path) else None)
    if hard_table is None and cfg.use_hard_negative:
        print("WARNING index/hard_table.npy missing -- the hard-negative term of "
              "eq. 3-20/3-21 is DISABLED (L_hard == 0), so stage 3 will not do what "
              "3.5.3 describes. Run `python -m rst.mine_hard_pairs` first.")

    model = RSTModel(cfg, protos, attr_dim=int(attr_matrix.shape[1])
                     if attr_matrix is not None else 0).to(device)
    # RSTModel.__init__ already installed the prototypes; calling set_prototypes
    # again here used to re-register them from CPU tensors after .to(device).
    if attr_matrix is not None:
        attr_t = attr_matrix.float()
    else:
        attr_t = None
    crit = RSTLoss(cfg, train_ds.store.class_weight_tensor("inverse_freq"),
                   attr_t, hard_table).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model: %.1fM trainable params" % (n_trainable / 1e6))
    opt = torch.optim.AdamW(optimizer_param_groups(model, cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    start_step = 0
    if args.resume:
        ck = torch.load(os.path.join(cfg.root, args.resume), map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print("resumed %s at step %d" % (args.resume, start_step))

    hist_path = os.path.join(out_dir, "history.jsonl")
    best = -1.0
    step = start_step
    # One schedule across all stages. Per-stage schedules made warmup (counted in
    # full-epoch steps) longer than a short stage, and restarted cosine while the
    # global step kept growing, so the learning rate collapsed to ~1e-9.
    #
    # Stage lengths come from the 3.5 budgets (table 4-2: 10 / 15~20 / 15~20), not
    # from an even split of cfg.epochs. Splitting evenly gave every stage 10
    # epochs, so stages 2 and 3 could never satisfy `epochs_done >= lo_b` and the
    # plateau hand-off was unreachable. The horizon uses each stage's UPPER bound;
    # ending a stage early just leaves the cosine slightly short of zero.
    if args.stage_epochs:
        budgets = {s: (args.stage_epochs, args.stage_epochs) for s in stages}
    else:
        fallback = max(1, cfg.epochs // len(stages))
        budgets = {s: STAGE_BUDGETS.get(s, (fallback, fallback)) for s in stages}
    total_steps = sum(budgets[s][1] for s in stages) * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    # A short run (small --stage-epochs, or a truncated smoke test) can make the
    # warmup as long as the whole schedule, which silently pins the LR inside the
    # ramp for the entire training. Cap it at a quarter of the horizon.
    if warmup_steps * 4 > total_steps:
        capped = max(1, total_steps // 4)
        print("WARNING warmup %d steps would span the whole %d-step schedule; capping to %d"
              % (warmup_steps, total_steps, capped))
        warmup_steps = capped
    print("schedule: %d total steps (%s epochs max), %d warmup"
          % (total_steps, "+".join(str(budgets[s][1]) for s in stages), warmup_steps))
    for si, stage in enumerate(stages):
        if cfg.use_text and protos is None:
            caps["use_text"] = False
        resolve_stage(cfg, stage, caps)
        if stage == 3 and si > 0 and cfg.remine_stage3 and crit.has_hard:
            # 3.5.3's stage boundary: swap the 3.3.3 frozen-similarity neighbours
            # for what this model actually confuses, then stage 3's margin term
            # works on live confusions. The critic's buffer is replaced in place
            # so the loss picks it up on the next batch.
            new_table = remine_hard_table(model, val_loader, device, cfg,
                                          train_ds.store, crit.hard.cpu().numpy())
            crit.hard.copy_(torch.from_numpy(new_table).long().to(crit.hard.device))
        lo_b, hi_b = budgets[stage]
        n_epochs = hi_b
        best_stage, stall = -1.0, 0
        print("\n=== stage %d: scene=%s text=%s hard=%s diff=%s | budget %d~%d epochs ==="
              % (stage, cfg.use_scene, cfg.use_text, cfg.use_hard_negative,
                 cfg.use_diff_loss, lo_b, hi_b))

        for ep in range(n_epochs):
            train_ds.set_epoch(ep + si * 100)
            if sampler is not None:
                sampler.set_epoch(ep + si * 100)
            model.train()
            bar = tqdm(train_loader, total=len(train_loader),
                       desc="s%d e%d/%d" % (stage, ep + 1, n_epochs), dynamic_ncols=True)
            acc, n_b = {}, 0
            t0 = time.time()
            for batch in bar:
                if cfg.max_train_batches and n_b >= cfg.max_train_batches:
                    break
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(g["base_lr"], step - start_step,
                                    total_steps, warmup_steps)
                with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                    out = model(batch)
                    loss, parts = crit(out, batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                step += 1
                n_b += 1
                m = entity_metrics(out, batch)
                for k, v in {**{"loss": loss.item()}, **{("L_" + a): b.item()
                                                          for a, b in parts.items()}, **m}.items():
                    acc[k] = acc.get(k, 0.0) + v
                bar.set_postfix(
                    loss="%.3f" % (acc["loss"] / n_b),
                    top1="%.3f" % (acc["top1"] / n_b),
                    lr="%.1e" % opt.param_groups[-1]["lr"],
                    mem="%.1fG" % (torch.cuda.max_memory_allocated() / 1e9)
                    if device.type == "cuda" else "-")
            for k in list(acc):
                acc[k] /= max(1, n_b)
            acc.update({"stage": stage, "epoch_in_stage": ep + 1, "lr": opt.param_groups[-1]["lr"],
                        "sec": round(time.time() - t0, 1), "split": "train"})
            with open(hist_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(acc) + "\n")

            if (ep + 1) % cfg.eval_every_epoch == 0 or ep + 1 == n_epochs:
                res = evaluate(model, val_loader, device, cfg, val_ds.store,
                               desc="validate", use_amp=cfg.amp,
                               max_batches=args.eval_batches)
                res.update({"stage": stage, "epoch_in_stage": ep + 1, "split": "val"})
                with open(hist_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res) + "\n")
                print("  val " + format_result(res))
                if res["macro_f1"] > best:
                    best = res["macro_f1"]
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "cfg": cfg.to_dict(), "step": step, "val": res},
                               os.path.join(out_dir, "best.pt"))
                    print("  saved best (macro-F1 %.2f) -> %s" % (100 * best, out_dir))

                # 3.5 stage hand-off criterion, tracked on military Macro-F1
                watch = (res["military_macro_f1"] if args.stage_switch == "auto"
                         else res["macro_f1"])
                best_stage, stall, advance = plateau_advance(
                    watch, best_stage, stall, ep + 1, lo_b, args.plateau,
                    args.stage_switch == "auto")
                if advance:
                    print("  stage %d stable: watched metric flat for %d epochs at "
                          "%.2f (budget %d~%d, ran %d) -> advancing"
                          % (stage, stall, 100 * best_stage, lo_b, hi_b, ep + 1))
                    break

            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "cfg": cfg.to_dict(), "step": step},
                       os.path.join(out_dir, "last.pt"))

        # 3.3.2 as revised verifies the knowledge base in the space this stage
        # just aligned, so the stage boundary needs its own artifact -- best.pt is
        # picked on validation Macro-F1 and can come from any stage.
        stage_ck = os.path.join(out_dir, "stage%d.pt" % stage)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "cfg": cfg.to_dict(), "step": step, "stage": stage}, stage_ck)
        print("  stage %d finished at step %d -> %s" % (stage, step, stage_ck))

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"best_val_macro_f1": best, "run": run_name,
                   "ablation": args.ablation, "stages": stages}, fh, indent=2)
    print("\nbest val macro-F1 %.2f  ->  %s" % (100 * best, out_dir))


if __name__ == "__main__":
    main()
