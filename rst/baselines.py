#!/usr/bin/env python
"""Table 4-3 baselines: pure-vision backbones and zero-shot VLMs.

Fairness (4.5.1 requires identical splits, inputs and metrics):
  * regions come from the same `build_region` call as our model, so the crop
    geometry, the 25% context expansion, the colour augmentation, the random
    erasing and the expansion jitter are identical;
  * the mask channel is opt-in via --mask-channel; the fair-384 rerun turns it
    on so every row sees the same 4-channel input our Region uses;
  * metrics come from the same `rst.eval.evaluate`, so Top-1/Top-5/Macro-F1 and
    the military-only numbers are computed by one code path;
  * the balanced sampler and inverse-frequency class weights match our training.
The v1 rows ran each backbone at its native 224 with the plain colour aug and a
single-group 3e-4 optimizer -- the rerun (input 384, mask channel, erase+jitter,
grouped 1e-5/1e-4, 50 epochs) removes the resolution/training-recipe confound
that 4-3's footnote had to carry. What remains different is the pretraining
corpus (ImageNet vs SigLIP2 web data), which belongs to the backbone choice and
stays in the footnote.

  python -m rst.baselines train --backbone resnet50 --gpu 0 --input-size 384 \
      --mask-channel --tag fair384
  python -m rst.baselines eval --backbone resnet50 --tag fair384 --gpu 0
  python -m rst.baselines zeroshot --model siglip2 --gpu 0
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import EntityDataset, BalancedEntitySampler, build_region
from .eval import evaluate, format_result, macro_f1, stratify_by_size, topk_correct
from .train import lr_at, set_seed

try:
    from tqdm import tqdm
except ImportError:                                    # pragma: no cover
    def tqdm(x, **kw):
        return x

BACKBONES = {
    # name: (torchvision factory, weights enum, feature dim, native input)
    "resnet50": ("resnet50", "ResNet50_Weights.IMAGENET1K_V2", 2048, 224),
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_V1", 768, 224),
    "swin_b": ("swin_b", "Swin_B_Weights.IMAGENET1K_V1", 1024, 224),
    "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights.IMAGENET1K_V1", 1024, 224),
    # transformers CLIP, handled by a dedicated branch in VisionBaseline: the
    # patch size is 14 so the closest multiple-of-14 input to our 384 is 336,
    # and the position embeddings are bicubically interpolated 16x16 -> 24x24
    "clip_vit_l14": ("(transformers CLIPVisionModel)", "openai/clip-vit-large-patch14", 1024, 336),
}


def mask_channel_conv(old):
    """Widen a backbone's first conv from 3 to 4 inputs for the GT-box mask channel.

    The new channel starts at exactly zero, so at initialisation the network computes
    precisely what the ImageNet-pretrained 3-channel weights did: the mask can only add
    information, never destroy the pretrained features. This is the same trick
    `SigLIP2Encoder.extra_conv` uses for our own model, and it is what makes the
    baseline comparison fair -- 3.1 gives our Region four channels, so a baseline that
    is denied the fourth is not being compared on equal inputs.
    """
    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, bias=old.bias is not None,
                    padding_mode=old.padding_mode)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class RegionCollator:
    """Region crops only -- no scene tiles, so none of the tiling work is done."""

    def __init__(self, dataset, mask_channel=False):
        self.ds = dataset
        self.cfg = dataset.cfg
        self.root = dataset.cfg.root
        self.mask_channel = mask_channel

    def __call__(self, items):
        regions, ys, dom, ids = [], [], [], []
        augment = self.ds.train and self.cfg.augment_color
        # same augmentation set as the RST training path: erase and expansion
        # jitter ride on build_region exactly as RSTCollator passes them
        erase_p = getattr(self.cfg, "erase_prob", 0.0)
        expand_j = getattr(self.cfg, "expand_jitter", 0.0)
        for it in items:
            rec = it["rec"]
            img = Image.open(os.path.join(self.root, rec["scene_file"])).convert("RGB")
            W, H = rec["image_size"]
            region, _ = build_region(
                img, W, H, rec, it["flip"], self.cfg, augment,
                erase=(self.ds.train and erase_p > 0
                       and random.random() < erase_p),
                expand_jit=(expand_j if self.ds.train else 0.0))
            img.close()
            # RGB only unless the run asked for the mask channel too
            if self.mask_channel and region.shape[0] < 4:
                raise RuntimeError("mask_channel requested but build_region returned %d "
                                   "channels -- cfg.use_mask_channel is not set"
                                   % region.shape[0])
            regions.append(region if self.mask_channel else region[:3])
            ys.append(it["class"])
            dom.append(it["domain"])
            ids.append(it["instance_id"])
        return {"region": torch.stack(regions),
                "class": torch.tensor(ys, dtype=torch.long),
                "domain": torch.tensor(dom, dtype=torch.long),
                "instance_ids": ids}


def interpolate_pos_embedding(pos, new_grid, old_grid):
    """(1, 1+N, D) ViT position embedding -> bicubic resize of the patch tokens.

    cls carries through untouched. Without this, feeding 384 to a 224-trained
    torchvision ViT dies on the pos-embedding add -- the same constraint our
    SigLIP2Encoder solves with its own interpolation."""
    cls, patch = pos[:, :1], pos[:, 1:]
    d = patch.shape[-1]
    patch = patch.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(new_grid, new_grid), mode="bicubic",
                          align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, d)
    return torch.cat([cls, patch], dim=1)


class VisionBaseline(nn.Module):
    """A torchvision backbone plus the two heads, speaking the same output dict as
    RSTModel so rst.eval can score it without a second metrics implementation."""

    def __init__(self, name, n_classes, n_domains=2, pretrained=True, dropout=0.1,
                 in_channels=3, input_size=0):
        super().__init__()
        from torchvision import models
        factory, weights_name, feat, native = BACKBONES[name]

        if name.startswith("clip_vit"):
            # transformers CLIP fine-tuning branch. Input must be resized to a
            # multiple of the 14px patch grid (336) and the RGB channels moved
            # from our SigLIP [-1,1] to CLIP's mean/std. Position embeddings are
            # handled by transformers' own interpolate_pos_encoding=True at call
            # time; image_size is widened so its consistency check passes.
            from transformers import CLIPVisionModel
            hf_name = weights_name
            self._clip = True
            self._clip_native = 336
            vmodel = CLIPVisionModel.from_pretrained(hf_name)
            emb = vmodel.vision_model.embeddings
            if in_channels == 4:
                emb.patch_embedding = mask_channel_conv(emb.patch_embedding)
            emb.image_size = self._clip_native
            vmodel.config.image_size = self._clip_native
            self.register_buffer("clip_mean", torch.tensor(
                [0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer("clip_std", torch.tensor(
                [0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
            self.features = vmodel
            self.native = self._clip_native
            self.in_channels = in_channels
            self.head_class = nn.Linear(feat, n_classes)
            self.head_domain = nn.Linear(feat, n_domains)
            return

        weights = models.get_weight(weights_name) if pretrained else None
        if weights is not None and weights.meta.get("crop_size"):
            native = int(weights.meta["crop_size"])
        pretrained_size = native
        if input_size:
            native = input_size
        net_factory = getattr(models, factory)
        # torchvision is not uniform here: ViT and Swin take `dropout`, ResNet and
        # ConvNeXt do not, so pass it only where the signature accepts it
        kw = {"dropout": dropout} if name.startswith(("vit", "swin")) else {}
        net = net_factory(weights=weights, **kw)
        self.native = native
        self.in_channels = in_channels
        if in_channels == 4:
            if name == "resnet50":
                net.conv1 = mask_channel_conv(net.conv1)
            elif name.startswith("vit"):
                net.conv_proj = mask_channel_conv(net.conv_proj)
            elif name.startswith(("swin", "convnext")):
                net.features[0][0] = mask_channel_conv(net.features[0][0])
            else:
                raise ValueError("no first-conv recipe for %s" % name)
        if name == "resnet50":
            self.features = nn.Sequential(*list(net.children())[:-1], nn.Flatten(1))
        elif name.startswith("vit"):
            net.heads = nn.Identity()
            self._pos_interpolated = False
            # torchvision >=0.19 keeps pos_embedding on the ENCODER submodule, not
            # on the VisionTransformer; the resolution assert, however, reads
            # VisionTransformer.image_size (an int, compared against h directly)
            pos = (getattr(net, "pos_embedding", None)
                   or getattr(net.encoder, "pos_embedding", None))
            if pretrained_size != native and pos is not None:
                holder = net if hasattr(net, "pos_embedding") else net.encoder
                g_new = native // net.patch_size
                g_old = pretrained_size // net.patch_size
                holder.pos_embedding.data = interpolate_pos_embedding(
                    holder.pos_embedding.data, g_new, g_old)
                self._pos_interpolated = True
                if hasattr(net, "image_size"):
                    net.image_size = native            # int, matches the assert
            self.features = net
        elif name.startswith("swin"):
            net.head = nn.Identity()
            self.features = net
        elif name.startswith("convnext"):
            # torchvision's ConvNeXt classifier is (LayerNorm2d, Flatten, Linear);
            # keep the norm, drop the Linear
            net.classifier = nn.Sequential(net.classifier[0], nn.Flatten(1))
            self.features = net
        else:
            raise ValueError(name)
        self.head_class = nn.Linear(feat, n_classes)
        self.head_domain = nn.Linear(feat, n_domains)

    def forward(self, batch):
        x = batch["region"][:, :self.in_channels]
        if getattr(self, "_clip", False):
            # resize to the 14px-multiple input, then move the RGB channels from
            # our SigLIP [-1,1] to CLIP's mean/std expectation; the mask channel
            # (if present) stays as-is, mirroring the training-path convention
            x = F.interpolate(x, size=self._clip_native, mode="bilinear",
                              align_corners=False)
            rgb = (x[:, :3] + 1.0) / 2.0
            rgb = (rgb - self.clip_mean) / self.clip_std
            x = torch.cat([rgb, x[:, 3:]], dim=1) if x.size(1) > 3 else rgb
            f = self.features(x, interpolate_pos_encoding=True).pooler_output
        else:
            # resnet/convnext/swin accept any resolution natively, and a ViT whose pos
            # embedding was interpolated runs at its new size directly. The resize only
            # fires for legacy checkpoints whose backbone is locked to its pretrained size.
            if x.shape[-1] != self.native and not getattr(self, "_pos_interpolated", False):
                x = F.interpolate(x, size=self.native, mode="bilinear", align_corners=False)
            f = self.features(x)
        f = f.flatten(1)
        return {"logits_class": self.head_class(f), "logits_domain": self.head_domain(f),
                "entity": f, "gate": f.new_zeros(f.size(0), 3)}


def report_test(cfg, args, model, out_dir, meta, n_classes):
    """Test-split numbers + the same size stratification used for our model, so a
    table 4-3/4-6 row can be compared cell by cell. Split out of `train_baseline`
    so a finished run can be re-scored at a different bucket resolution without
    retraining."""
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    test_ds = EntityDataset(cfg, "test", train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=cfg.num_workers,
                             collate_fn=RegionCollator(test_ds, args.mask_channel),
                             pin_memory=(device.type == "cuda"))
    model.eval()
    test = evaluate(model, test_loader, device, cfg, test_ds.store, desc="test",
                    return_samples=True)
    print("  " + format_result(test))
    strat = stratify_by_size(test, test_ds.store, n_classes)
    for nm, v in strat.items():
        print("    %-9s n=%-5d top1 %6.2f macroF1 %6.2f"
              % (nm, v["n"], 100 * v["top1"], 100 * v["macro_f1"]))
    slim = {k: v for k, v in test.items() if not k.startswith("_")}
    if getattr(args, "dump_samples", False):
        # per-entity arrays in the same schema as rst.eval_checkpoint's dump, so
        # tools/subset_report can run paired McNemar tests baseline-vs-RST on the
        # same 3002 entities without any format glue
        g = test["_got"]
        np.savez_compressed(
            os.path.join(out_dir, "samples_test.npz"),
            ids=np.array(g["ids"]), y=g["y"], pred=test["_pred"],
            logits=g["logits"].numpy(),
            gate=np.zeros((len(g["ids"]), 3)), sem_raw=np.zeros((len(g["ids"]), n_classes)),
            gate_present=False, sem_present=False,
            run=os.path.basename(out_dir), split="test")
        print("  per-sample dump -> %s" % os.path.join(out_dir, "samples_test.npz"))
    path = os.path.join(out_dir, "eval_test.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(meta, test=slim, by_entity_size=strat), fh,
                  ensure_ascii=False, indent=2, default=float)
    print("  wrote %s" % path)


def evaluate_only(cfg, args):
    # --tag keeps a re-run from overwriting the archived baseline: history.jsonl is
    # appended to and best.pt would simply be replaced, silently losing the row that
    # is already in the thesis
    out_dir = os.path.join(cfg.root, cfg.out_dir,
                           "baseline_" + args.backbone + ("_" + args.tag if args.tag else ""))
    ck_path = os.path.join(out_dir, "best.pt")
    if not os.path.exists(ck_path):
        raise SystemExit("no checkpoint at %s" % ck_path)
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    n_classes = len(json.load(open(os.path.join(cfg.root, cfg.index_dir, "class_meta.json"),
                                   encoding="utf-8"))["classes"])
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    # input_size comes from the checkpoint's own metadata: a fair384 ViT has a
    # 384-grid pos embedding, so rebuilding it at 224 would fail on load
    model = VisionBaseline(args.backbone, n_classes, dropout=cfg.dropout,
                           in_channels=int(ck.get("in_channels", 4 if args.mask_channel else 3)),
                           input_size=int(ck.get("native_input", 0))).to(device)
    model.load_state_dict(ck["model"])
    print("%s best.pt (epoch %d, val macro-F1 %.2f) -> rescoring test"
          % (args.backbone, ck["epoch"], 100 * ck["val"]["macro_f1"]))
    report_test(cfg, args, model, out_dir,
                {"backbone": args.backbone, "native_input": model.native,
                 "epochs": "restored", "best_val_macro_f1": ck["val"]["macro_f1"]},
                n_classes)


class CLIPEncoder(nn.Module):
    """CLIP zero-shot needs its own preprocessing: our crops leave `to_tensor` in the
    SigLIP [-1, 1] range, CLIP was trained on mean/std-normalised pixels at 224. Same
    two methods as SigLIP2Encoder so the zero-shot loop is shared."""

    def __init__(self, name, cache_dir=None):
        super().__init__()
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
        self.model = CLIPModel.from_pretrained(name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        pp = AutoImageProcessor.from_pretrained(name, cache_dir=cache_dir)
        size = pp.size["shortest_edge"] if isinstance(pp.size, dict) else pp.size
        self.native_image_size = int(size)
        self.register_buffer("mean", torch.tensor(pp.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(pp.image_std).view(1, 3, 1, 1))
        self.max_text_len = 77

    def text_token_lengths(self, texts):
        """Same contract as SigLIP2Encoder, so build_text_features can report how many
        sentences would be truncated by either tower."""
        return [len(self.tokenizer.tokenize(t)) for t in texts]

    def image_feature(self, px):
        x = (px[:, :3] + 1.0) / 2.0                     # [-1,1] -> [0,1]
        if x.shape[-1] != self.native_image_size:
            x = F.interpolate(x, size=self.native_image_size, mode="bilinear",
                              align_corners=False)
        return self.model.get_image_features(pixel_values=(x - self.mean) / self.std)

    def contrastive_feature(self, px):
        """Same contract as SigLIP2Encoder.contrastive_feature: the embedding that
        lives in the shared image-text space. For CLIP that is get_image_features."""
        return self.image_feature(px)

    def encode_text(self, texts):
        tok = self.tokenizer(list(texts), padding=True, truncation=True, max_length=77,
                             return_tensors="pt")
        return self.model.get_text_features(
            input_ids=tok["input_ids"].to(self.mean.device),
            attention_mask=tok["attention_mask"].to(self.mean.device))


def baseline_param_groups(model, cfg, head_lr):
    """The same two-group layout rst.train uses: backbone at 1e-5, heads at
    1e-4, decay on matrices only. A single 3e-4 group was the v1 recipe and is
    one of the confounds the fair rerun removes."""
    backbone, head = [], []
    for name, p in model.named_parameters():
        (head if name.startswith("head_") else backbone).append(p)
    groups = []
    for params, lr in ((backbone, cfg.backbone_lr), (head, head_lr)):
        groups.append({"params": [p for p in params if p.ndim > 1],
                       "lr": lr, "base_lr": lr, "weight_decay": cfg.weight_decay})
        groups.append({"params": [p for p in params if p.ndim <= 1],
                       "lr": lr, "base_lr": lr, "weight_decay": 0.0})
    return groups


def train_baseline(cfg, args):
    set_seed(cfg.seed)
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    n_classes = len(json.load(open(os.path.join(cfg.root, cfg.index_dir, "class_meta.json"),
                                   encoding="utf-8"))["classes"])

    train_ds = EntityDataset(cfg, "train", train=True)
    val_ds = EntityDataset(cfg, "validation", train=False)

    def mk(ds, bs, sampler=None):
        return DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=(sampler is None),
                          num_workers=cfg.num_workers,
                          collate_fn=RegionCollator(ds, args.mask_channel),
                          pin_memory=(device.type == "cuda"),
                          persistent_workers=cfg.num_workers > 0)

    train_loader = mk(train_ds, args.batch, BalancedEntitySampler(train_ds, cfg.seed))
    val_loader = mk(val_ds, 64)

    model = VisionBaseline(args.backbone, n_classes, dropout=cfg.dropout,
                           in_channels=4 if args.mask_channel else 3,
                           input_size=args.input_size).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("%s: %.1fM trainable params, input %d, %d channels (%s), "
          "%d train / %d val entities"
          % (args.backbone, n_par / 1e6, model.native, model.in_channels,
             "RGB + GT-box mask" if args.mask_channel else "RGB only",
             len(train_ds), len(val_ds)))

    cw = train_ds.store.class_weight_tensor("inverse_freq").to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)
    crit_d = nn.CrossEntropyLoss()
    if args.single_lr:
        opt = torch.optim.AdamW(model.parameters(), lr=args.single_lr,
                                weight_decay=cfg.weight_decay)
        for g in opt.param_groups:
            g["base_lr"] = args.single_lr
    else:
        opt = torch.optim.AdamW(baseline_param_groups(model, cfg, args.head_lr))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    total_steps = args.epochs * len(train_loader)
    warmup = min(cfg.warmup_epochs * len(train_loader), max(1, total_steps // 4))
    out_dir = os.path.join(cfg.root, cfg.out_dir,
                           "baseline_" + args.backbone + ("_" + args.tag if args.tag else ""))
    # history.jsonl is appended to and best.pt simply replaced, so re-running without a
    # --tag silently destroys an archived baseline -- that is exactly what happened to
    # the ViT row of 表4-2. Refuse rather than guess.
    if os.path.exists(os.path.join(out_dir, "best.pt")):
        raise SystemExit("refusing to overwrite %s: it already holds a finished baseline. "
                         "Pass --tag <name> for a separate run directory." % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg.save(os.path.join(out_dir, "config.json"))

    best, step = -1.0, 0
    hist = os.path.join(out_dir, "history.jsonl")
    for ep in range(args.epochs):
        train_ds.set_epoch(ep)
        model.train()
        acc, n_b, t0 = {}, 0, time.time()
        bar = tqdm(train_loader, total=len(train_loader), desc="%s e%d/%d"
                   % (args.backbone, ep + 1, args.epochs), dynamic_ncols=True)
        for batch in bar:
            if args.max_batches and n_b >= args.max_batches:
                break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(g["base_lr"], step, total_steps, warmup)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                out = model(batch)
                loss = crit(out["logits_class"], batch["class"]) \
                    + crit_d(out["logits_domain"], batch["domain"])
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            n_b += 1
            pred = out["logits_class"].argmax(1)
            for k, v in {"loss": float(loss),
                         "top1": float((pred == batch["class"]).float().mean())}.items():
                acc[k] = acc.get(k, 0.0) + v
            bar.set_postfix(loss="%.3f" % (acc["loss"] / n_b), top1="%.3f" % (acc["top1"] / n_b))
        res = evaluate(model, val_loader, device, cfg, val_ds.store, desc="val")
        rec = {"epoch": ep + 1, "train_loss": acc["loss"] / max(1, n_b),
               "train_top1": acc["top1"] / max(1, n_b), "sec": round(time.time() - t0, 1),
               "val_top1": res["top1"], "val_macro_f1": res["macro_f1"],
               "val_military_macro_f1": res["military_macro_f1"]}
        with open(hist, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("  ep%-2d train top1 %.3f loss %.3f | val %s"
              % (ep + 1, rec["train_top1"], rec["train_loss"], format_result(res)))
        if res["macro_f1"] > best:
            best = res["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": ep + 1, "val": res,
                        "backbone": args.backbone, "native_input": model.native,
                        "in_channels": model.in_channels},
                       os.path.join(out_dir, "best.pt"))
            print("    saved best (val macro-F1 %.2f)" % (100 * best))

    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print("\n=== %s TEST (best val macro-F1 %.2f at epoch %d) ==="
          % (args.backbone, 100 * best, ck["epoch"]))
    report_test(cfg, args, model, out_dir,
                {"backbone": args.backbone, "native_input": model.native,
                 "epochs": args.epochs, "best_val_macro_f1": best}, n_classes)


def zeroshot(cfg, args):
    """4.4.2 zero-shot: class names only, no training on this dataset."""
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(cfg.root, cfg.index_dir, "class_meta.json"),
                          encoding="utf-8"))
    classes = [str(c) for c in meta["classes"]]
    test_ds = EntityDataset(cfg, "test", train=False)
    if args.limit and args.limit < len(test_ds):
        test_ds.records = test_ds.records[:args.limit]      # smoke test without a full pass
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=RegionCollator(test_ds),
                        pin_memory=(device.type == "cuda"))

    from .encoders import SigLIP2Encoder
    name = cfg.vision_model if args.model in ("", "siglip2") else args.model
    # the tag doubles as the output filename, so the default stays "siglip2" and the
    # already-collected row does not get orphaned under a new name
    tag = "siglip2" if args.model in ("", "siglip2") else name.rsplit("/", 1)[-1].lower()
    enc = (SigLIP2Encoder(name, freeze=True, load_text=True) if "siglip" in tag
           else CLIPEncoder(name))
    enc = enc.to(device).eval()
    print("zero-shot encoder: %s (native input %d)" % (name, enc.native_image_size))
    templates = {"raw": ["{c}"],
                 "photo": ["a photo of a {c} aircraft", "a photo of a {c}"]}
    results = {}
    with torch.no_grad():
        # image features do not depend on the prompt, so they are computed once and
        # reused for every template set and every control
        imgs, ys = [], []
        for batch in tqdm(loader, desc="zeroshot-images"):
            px = batch["region"][:, :3].to(device)
            if px.shape[-1] != enc.native_image_size:
                px = F.interpolate(px, size=enc.native_image_size, mode="bilinear",
                                   align_corners=False)
            imgs.append(F.normalize(enc.contrastive_feature(px).float(), dim=-1).cpu())
            ys.append(batch["class"])
        v = torch.cat(imgs)
        y = torch.cat(ys).numpy()
        mil = np.isin(y, list(test_ds.store.military_index))
        dom = np.asarray([0 if test_ds.store.domain_of[classes[c]] == "military" else 1
                          for c in y])

        # A chance-level 114-way score is ambiguous: it could mean "this frozen encoder
        # carries no usable signal" or "our pipeline is broken". These two controls
        # settle it, and they are recorded in the JSON so the table row can be
        # defended without re-running anything.
        #   domain_control -- a coarse 2-way decision with plain English prompts. If the
        #     image/text alignment were broken this would sit at 50%.
        #   text_pairwise_cos -- how far apart the 114 class-name embeddings are. Near
        #     1.0 means the prompts are effectively the same vector, so no retrieval
        #     can work however good the image side is.
        ctl = enc.encode_text(["a photo of a military aircraft",
                               "a photo of a civilian airliner"])
        ctl = F.normalize(ctl.float(), dim=-1).cpu()
        controls = {
            "chance_pct": round(100.0 / len(classes), 3),
            "military_vs_civilian_acc": round(
                100 * float(((v @ ctl.T).argmax(1).numpy() == dom).mean()), 2),
            "n": int(len(y)),
        }
        print("control: military-vs-civilian 2-way %.2f%% correct (chance 50%%)"
              % controls["military_vs_civilian_acc"])

        for tname, tpls in templates.items():
            texts = [t for c in classes for t in [x.replace("{c}", c) for x in tpls]]
            tf = enc.encode_text(texts)
            tf = F.normalize(tf.float(), dim=-1).view(len(classes), len(tpls), -1).mean(1)
            # v is on CPU (3002 x d is a few MB), so keep the text side there too
            tf = F.normalize(tf, dim=-1).cpu()                   # [C, d]
            G = (tf @ tf.T).fill_diagonal_(0)
            sims = v @ tf.T
            pred = sims.argmax(1).numpy()
            res = {"n": int(len(y)), "top1": float((pred == y).mean()),
                   "top5": float(np.asarray(topk_correct(sims, torch.as_tensor(y), 5)).mean()),
                   "macro_f1": macro_f1(y, pred, len(classes))[0],
                   "military_top1": float((pred[mil] == y[mil]).mean()),
                   "military_macro_f1": macro_f1(y[mil], pred[mil], len(classes))[0],
                   "templates": tpls, "model": name,
                   "text_pairwise_cos_mean": round(float(G.mean()), 4),
                   "mean_image_text_cos": round(float(sims.mean()), 4),
                   "distinct_predicted_classes": int(len(np.unique(pred)))}
            results[tname] = res
            print("%s zero-shot [%-5s] Top-1 %.2f  Top-5 %.2f  MacroF1 %.2f | "
                  "mil Top-1 %.2f mil F1 %.2f | text cos %.4f  img-text cos %+.4f  "
                  "%d distinct classes predicted"
                  % (tag, tname, 100 * res["top1"], 100 * res["top5"], 100 * res["macro_f1"],
                     100 * res["military_top1"], 100 * res["military_macro_f1"],
                     res["text_pairwise_cos_mean"], res["mean_image_text_cos"],
                     res["distinct_predicted_classes"]))
    results["controls"] = controls
    out = args.out or os.path.join(cfg.root, cfg.out_dir,
                                   "baseline_%s_zeroshot.json" % tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "zeroshot"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input-size", dest="input_size", type=int, default=384,
                    help="run the backbone at this resolution instead of its "
                         "pretrained crop size; a ViT gets its position embeddings "
                         "bicubically interpolated (the same treatment our "
                         "SigLIP2 backbone's 384 input represents). 0 = pretrained "
                         "size, which is what the archived v1 rows used")
    ap.add_argument("--single-lr", dest="single_lr", type=float, default=0.0,
                    help="one param group at this lr (the v1 recipe, 3e-4); default "
                         "0 uses the grouped backbone 1e-5 / head 1e-4 our model "
                         "trains with")
    ap.add_argument("--head-lr", dest="head_lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="cfg.seed; the default (0) matches the RST runs")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--mask-channel", dest="mask_channel", action="store_true",
                    help="feed the baselines the same 4th channel our Region uses "
                         "(the GT-box binary mask of 3.1). Without this the comparison "
                         "in 表4-2 is not input-matched: our model sees where the "
                         "aircraft is inside the context-expanded crop and they do not.")
    ap.add_argument("--tag", default="", help="suffix for the run directory, so a "
                                              "re-run with different hyper-parameters "
                                              "does not clobber the archived baseline")
    ap.add_argument("--dump-samples", dest="dump_samples", action="store_true",
                    help="write per-entity test predictions in the rst format so "
                         "tools/subset_report can run paired tests against our model")
    ap.add_argument("--erase-prob", dest="erase_prob", type=float, default=None,
                    help="same random-erasing augmentation as the RST training path "
                         "(v2 fair recipe: 0.25)")
    ap.add_argument("--expand-jitter", dest="expand_jitter", type=float, default=None,
                    help="same context-expansion jitter as the RST path (v2: 0.10)")
    ap.add_argument("--limit", type=int, default=0, help="zeroshot: score only N entities")
    ap.add_argument("--out", default="", help="zeroshot: where to write the JSON")
    args = ap.parse_args()

    cfg = Config()
    cfg.root = os.path.abspath(args.root)
    cfg.num_workers = args.workers
    cfg.use_scene = cfg.use_text = cfg.use_hard_negative = cfg.use_diff_loss = False
    # build_region only emits the 4th channel when cfg.use_mask_channel is set, so the
    # flag has to live on cfg and not just on args -- otherwise the collator hands the
    # model 3 channels while its first conv expects 4 and the run dies on step one.
    cfg.use_mask_channel = bool(getattr(args, "mask_channel", False))
    if getattr(args, "erase_prob", None) is not None:
        cfg.erase_prob = args.erase_prob
    if getattr(args, "expand_jitter", None) is not None:
        cfg.expand_jitter = args.expand_jitter
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if args.mode == "train":
        train_baseline(cfg, args)
    elif args.mode == "eval":
        evaluate_only(cfg, args)
    else:
        zeroshot(cfg, args)


if __name__ == "__main__":
    main()
