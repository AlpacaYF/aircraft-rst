#!/usr/bin/env python
"""Framework self-test that needs no images.

Pushes synthetic region/scene tensors through one real forward+backward step on
the GPU, using the actual offline Text Features, so the fusion (eq. 3-13), the
gate, both heads and every loss term are verified before 8 GB of images arrive.
Also re-checks the two pieces of training logic that are pure functions: the
plateau stage hand-off (3.5) and the ablation capability mask.

  python tools/selftest_synthetic.py            # cpu if no gpu
  python tools/selftest_synthetic.py --gpu 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rst.config import Config
from rst.train import (STAGE_BUDGETS, PLATEAU, caps_from, default_caps,
                       plateau_advance, resolve_stage)

OK, BAD = "[ ok ]", "[FAIL]"
fails = []


def check(cond, msg):
    print("%s %s" % (OK if cond else BAD, msg))
    if not cond:
        fails.append(msg)


def test_plateau():
    print("\n-- 3.5 stage hand-off: switch only after the lower budget AND a plateau --")
    lo, hi = STAGE_BUDGETS[2]
    watch, stall, adv = 0.50, 0, False
    for ep in range(1, hi + 1):
        watch, stall, adv = plateau_advance(0.50, watch, stall, ep, lo, PLATEAU, True)
        if adv:
            break
    check(adv and ep == max(lo, PLATEAU),
          "metric flat from the start advances at epoch %d = max(lower budget %d, plateau %d)"
          % (ep, lo, PLATEAU))

    watch, stall, adv = 0.50, 0, False
    ever = False
    for ep in range(1, hi + 2):
        watch, stall, adv = plateau_advance(0.50 + 0.01 * ep, watch, stall, ep,
                                            lo, PLATEAU, True)
        ever = ever or adv
    check(not ever, "a metric that improves every epoch never triggers a hand-off")

    watch, stall, adv = 0.50, 0, False
    for ep in range(1, lo):
        watch, stall, adv = plateau_advance(0.50, watch, stall, ep, lo, PLATEAU, True)
    check(not adv, "plateau reached before the lower budget (%d) does not advance" % lo)

    watch, stall, adv = 0.50, 0, False
    for ep in range(1, hi + 5):
        watch, stall, adv = plateau_advance(0.50, watch, stall, ep, lo, PLATEAU, False)
    check(not adv, "--stage-switch fixed never advances automatically")

    # realistic case: improves past the lower budget, then goes flat
    watch, stall, adv = 0.50, 0, False
    stop_at = lo + 1
    for ep in range(1, hi + 1):
        w = 0.50 + 0.01 * min(ep, stop_at)
        watch, stall, adv = plateau_advance(w, watch, stall, ep, lo, PLATEAU, True)
        if adv:
            break
    check(adv and ep == stop_at + PLATEAU,
          "improving to epoch %d then flat advances at %d (stop + plateau)"
          % (stop_at, ep))


def test_caps():
    print("\n-- ablation mask: a stage may only enable what the ablation allows --")
    caps = default_caps()
    check(all(caps.values()), "full model: every capability on")

    cfg = Config()
    resolve_stage(cfg, 3, caps_from("region_only"))
    check(not (cfg.use_scene or cfg.use_text or cfg.use_hard_negative or cfg.use_diff_loss),
          "region_only stays region-only inside stage 3 (stage flags must not override it)")

    cfg = Config()
    resolve_stage(cfg, 3, caps_from("region_scene"))
    check(cfg.use_scene and not cfg.use_text and not cfg.use_hard_negative,
          "region_scene: Scene on, Text and hard negatives off in stage 3")

    cfg = Config()
    resolve_stage(cfg, 3, caps_from("no_hard"))
    check(cfg.use_text and not cfg.use_hard_negative,
          "no_hard: Text on, hard negatives off in stage 3")

    cfg = Config()
    resolve_stage(cfg, 1, default_caps())
    check(not (cfg.use_scene or cfg.use_text),
          "stage 1 is region-only by design (3.5.1)")
    resolve_stage(cfg, 2, default_caps())
    check(cfg.use_scene and cfg.use_text and not cfg.use_hard_negative,
          "stage 2 adds Scene and Text (3.5.2)")
    resolve_stage(cfg, 3, default_caps())
    check(cfg.use_hard_negative and cfg.use_diff_loss,
          "stage 3 adds hard negatives and the diff loss (3.5.3)")

    caps = caps_from("region_scene", {"use_text": True})
    check(caps["use_text"], "an explicit --use-* flag beats the named ablation")


def build_batch(cfg, device, n_ent=6, n_img=3, n_tiles=4, per_tile=49, dim=768,
                scene_mode="cached"):
    """Same layout RSTCollator produces, for either scene path."""
    g = torch.Generator().manual_seed(0)
    ch = 3 + (1 if cfg.use_mask_channel else 0)
    region = torch.rand(n_ent, ch, cfg.region_input, cfg.region_input, generator=g)
    e2s = torch.arange(n_ent, dtype=torch.long) % n_img
    meta = torch.zeros(n_img, n_tiles, 4)
    meta[..., 0] = torch.arange(n_tiles).float()          # gi
    meta[..., 1] = 0.0                                     # gj
    meta[..., 2] = torch.tensor([1.0, 0.0, 1.0, 0.0][:n_tiles])   # near/far mix
    meta[..., 3] = 0.0                                     # 0 = real tile
    meta[:, -1, 2] = 0.0                                   # last tile is far
    meta[:, -1, 3] = 1.0                                   # ... and padded
    out = {
        "region": region.to(device),
        "class": torch.randint(0, cfg.n_classes, (n_ent,), generator=g).to(device),
        "domain": torch.randint(0, cfg.n_domains, (n_ent,), generator=g).to(device),
        "gt_box": torch.rand(n_ent, 4, generator=g).to(device),
        "entity_to_scene": e2s.to(device),
        "scene_tile_meta": meta.to(device),
        "images": ["synthetic%d" % i for i in range(n_img)],
        "instance_ids": list(range(n_ent)),
    }
    if scene_mode == "cached":
        out["scene_tokens"] = torch.randn(
            n_img, n_tiles * per_tile, dim, generator=g).to(device)
    else:
        out["scene_tiles"] = torch.rand(
            n_img, n_tiles, 3, cfg.scene_input, cfg.scene_input, generator=g).to(device)
    return out


def test_forward(cfg, device, scene_mode="cached"):
    print("\n-- forward + backward, %s scene path --" % scene_mode)
    from rst.model import RSTModel
    from rst.losses import RSTLoss, entity_metrics

    p = os.path.join(cfg.root, "knowledge", "text_features.npz")
    check(os.path.exists(p), "offline Text Features present at %s" % p)
    z = np.load(p, allow_pickle=True)
    check(tuple(z["tokens"].shape) == (cfg.n_classes, 19, 768),
          "tokens %s cover all %d classes" % (tuple(z["tokens"].shape), cfg.n_classes))
    protos = {"tokens": torch.from_numpy(z["tokens"]),
              "mask": torch.from_numpy(z["mask"]), "class_of_token": None}
    attr_dim = int(z["attr_matrix"].shape[1])

    model = RSTModel(cfg, protos, attr_dim=attr_dim).to(device)
    # regression: re-installing prototypes from CPU tensors used to strand `proto`
    # on the CPU after .to(device), which blew up the first time stage 2 touched
    # the text branch
    model.set_prototypes(protos)
    check(model.proto.device.type == device.type,
          "set_prototypes keeps proto on %s when handed CPU tensors" % device.type)
    batch = build_batch(cfg, device, scene_mode=scene_mode)
    crit = RSTLoss(cfg, attr_matrix=torch.from_numpy(z["attr_matrix"])).to(device)

    # 3.2.3 two-level budget, which the cached and live paths must agree on: an
    # entity-bearing tile keeps kn*kn tokens, a background tile keeps kf*kf with
    # the rest of its row masked, and a padded tile keeps none.
    kn, kf = cfg.scene_near_pool, min(cfg.scene_far_pool, cfg.scene_near_pool)
    if scene_mode == "live":
        T, n_img = batch["scene_tiles"].size(1), batch["scene_tiles"].size(0)
        tok, pad = model.encode_scene(batch["scene_tiles"], batch["scene_tile_meta"])
    else:
        T = batch["scene_tile_meta"].size(1)
        n_img = batch["scene_tokens"].size(0)
        tok, pad = model.cached_scene(batch["scene_tokens"], batch["scene_tile_meta"])
    check(tuple(tok.shape) == (n_img, T * kn * kn, cfg.hidden),
          "%s scene tokens %s = %d tiles x %d slots"
          % (scene_mode, tuple(tok.shape), T, kn * kn))
    pm = pad.view(-1, T, kn * kn)
    keep = ~pm                                                # True = a real token
    real = ~(batch["scene_tile_meta"][..., 3] > 0.5)          # [B,T] not padded
    near = (batch["scene_tile_meta"][..., 2] > 0.5) & real
    far = (~(batch["scene_tile_meta"][..., 2] > 0.5)) & real
    if near.any():
        check(int(keep[near].sum(-1).unique().item()) == kn * kn,
              "near tiles keep all %d tokens" % (kn * kn))
    if far.any():
        check(int(keep[far].sum(-1).unique().item()) == kf * kf,
              "far tiles keep %d tokens, %d masked" % (kf * kf, kn * kn - kf * kf))
    if (~real).any():
        check(bool(pm[~real].all()), "padded tiles are fully masked")
    check(T * kn * kn <= cfg.scene_token_cap,
          "scene token budget %d <= cap %d" % (T * kn * kn, cfg.scene_token_cap))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = model(batch)
    n = batch["region"].size(0)
    check(tuple(out["logits_class"].shape) == (n, cfg.n_classes),
          "logits_class %s" % (tuple(out["logits_class"].shape),))
    check(tuple(out["logits_domain"].shape) == (n, cfg.n_domains),
          "logits_domain %s" % (tuple(out["logits_domain"].shape),))
    check(tuple(out["logits_semantic"].shape) == (n, cfg.n_classes),
          "logits_semantic %s (eq. 3-12 over all candidate classes)"
          % (tuple(out["logits_semantic"].shape),))
    check(tuple(out["gate"].shape) == (n, 3), "gate %s" % (tuple(out["gate"].shape),))
    check(torch.allclose(out["gate"].sum(-1), torch.ones(n, device=device), atol=1e-5),
          "gate weights are a convex combination (3.4.3)")
    eps = cfg.gate_floor
    check(bool((out["gate"] >= eps - 1e-6).all()),
          "gate floor %.2f: no branch can be starved of gradient (the measured "
          "collapse was alpha=(1.000, 0.000, 0.000))" % eps)
    check(bool((out["gate"] <= eps + (1.0 - 3.0 * eps) + 1e-6).all()),
          "gate ceiling %.2f: a branch that earns weight can still dominate"
          % (eps + 1.0 - 3.0 * eps))
    check(out["logits_attr"].shape[-1] == attr_dim,
          "logits_attr %s against %d attribute sentences"
          % (tuple(out["logits_attr"].shape), attr_dim))
    check(torch.isfinite(out["logits_class"]).all() and torch.isfinite(out["logits_semantic"]).all(),
          "no NaN/Inf in the logits")

    loss, parts = crit(out, batch)
    missing = {"domain", "class", "con", "hard", "diff"} - set(parts)
    check(not missing, "every loss term present: %s" % ", ".join(sorted(parts)))
    for k, v in sorted(parts.items()):
        check(torch.isfinite(v).all(), "  L_%-7s = %.4f" % (k, float(v)))
    check(float(parts["hard"]) == 0.0,
          "no hard_table.npy -> L_hard is exactly 0, not a silent pull toward class 0")

    # regression: an absent hard table used to leave self.hard zero-filled, which
    # made class 0 everybody's hard negative and trained the model to suppress it
    hard = ((torch.arange(cfg.n_classes) + 1) % cfg.n_classes).unsqueeze(1)
    crit_h = RSTLoss(cfg, attr_matrix=torch.from_numpy(z["attr_matrix"]),
                     hard_table=hard).to(device)
    with torch.no_grad():
        out2 = model(batch)
    _, parts_h = crit_h(out2, batch)
    check(float(parts_h["hard"]) > 0.0,
          "with a real hard table L_hard = %.4f is active (margin %.2f)"
          % (float(parts_h["hard"]), cfg.hard_margin))

    loss.backward()
    got = sum(1 for q in model.parameters() if q.grad is not None and q.grad.abs().sum() > 0)
    tot = sum(1 for q in model.parameters() if q.requires_grad)
    check(got > tot * 0.5, "backward: loss=%.4f, %d/%d trainable tensors got gradient"
          % (float(loss), got, tot))
    check(any(q.grad is not None and q.grad.abs().sum() > 0
              for q in model.rs_attn.parameters()),
          "region-scene attention (3.4.1) receives gradient")
    check(any(q.grad is not None and q.grad.abs().sum() > 0
              for q in model.text_proj.parameters()),
          "text projection receives gradient while the text encoder stays frozen")
    frozen = [n for n, q in model.named_parameters()
              if "text" in n and q.requires_grad and "text_proj" not in n]
    check(not frozen, "no text-encoder parameter is trainable (%d offenders)" % len(frozen))

    m = entity_metrics(out, batch)
    check(set(m) == {"top1", "top5", "domain_acc"}, "metrics: %s" % m)
    if device.type == "cuda":
        print("       peak GPU memory for one step: %.2f GB on %s"
              % (torch.cuda.max_memory_allocated(device) / 1e9,
                 torch.cuda.get_device_name(device)))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def test_hard_pair_metric():
    """Table 4-7's Hard Pair Accuracy. The first version of this metric gathered with
    the ROW index instead of the true CLASS index, which only crashes once the split
    has more entities than classes -- i.e. never in a smoke test, always on the real
    test set. These checks pin the index semantics down with n > C on purpose."""
    print("\n-- Hard Pair Accuracy (table 4-7) --")
    from rst.eval import hard_pair_accuracy
    C = 3
    table = torch.tensor([[1, 2], [0, 2], [0, 1]])          # C x K confusable sets
    y = np.array([0, 1, 2, 0, 1, 2, 0])                     # 7 entities, 3 classes
    win = torch.full((7, C), -1.0)
    win[torch.arange(7), torch.as_tensor(y)] = 1.0          # true class always highest
    check(abs(hard_pair_accuracy(win, y, table) - 1.0) < 1e-6,
          "true class beats every hard negative -> 1.000")
    lose = -win
    check(abs(hard_pair_accuracy(lose, y, table) - 0.0) < 1e-6,
          "true class last -> 0.000")
    tie = torch.zeros(7, C)
    check(abs(hard_pair_accuracy(tie, y, table) - 1.0) < 1e-6,
          "exact ties count as the true class winning (argmax takes column 0)")
    check(hard_pair_accuracy(win, y, None) is None, "no hard table -> None, not a crash")


def test_baseline_mask_channel(device):
    """表4-2 fairness: the baselines must be able to take the same 4th channel our
    Region uses. The conv surgery is per-architecture (resnet.conv1, vit.conv_proj,
    swin/convnext features[0][0]), so a silent failure on one backbone would leave a
    table row that quietly compares 3-channel inputs against our 4.

    Also pins the zero-init property: with the mask set to zero the network must
    compute exactly what the pretrained 3-channel weights computed, so adding the
    channel can only add information.
    """
    print("\n-- baseline 4-channel (mask) surgery --")
    from rst.baselines import VisionBaseline
    # Both ends of the mask path have to agree: build_region only emits the 4th channel
    # when cfg.use_mask_channel is set, and baselines.main() used to hardcode that to
    # False while the model was built with 4 input channels -- which dies on step one
    # with a conv shape error after the whole dataset has been scanned.
    from PIL import Image
    from rst.data import build_region
    cfgm = Config()
    img = Image.new("RGB", (200, 160), (90, 120, 150))
    rec = {"bbox": [40, 30, 120, 110], "bbox_expanded": [20, 15, 140, 125]}
    cfgm.use_mask_channel = True
    check(build_region(img, 200, 160, rec, False, cfgm, jitter=False)[0].shape[0] == 4,
          "build_region emits 4 channels when cfg.use_mask_channel is set")
    cfgm.use_mask_channel = False
    check(build_region(img, 200, 160, rec, False, cfgm, jitter=False)[0].shape[0] == 3,
          "build_region emits 3 channels when it is not")
    for name in ("resnet50", "vit_b_16", "swin_b", "convnext_base"):
        try:
            m = VisionBaseline(name, 114, pretrained=False, dropout=0.1,
                               in_channels=4).to(device).eval()
        except Exception as exc:
            check(False, "%s: 4-channel construction failed (%s)" % (name, exc))
            continue
        convs = [mod for mod in m.features.modules() if isinstance(mod, torch.nn.Conv2d)]
        check(bool(convs) and convs[0].in_channels == 4,
              "%s: first conv takes 4 channels" % name)
        x = {"region": torch.randn(2, 4, 224, 224, device=device)}
        with torch.no_grad():
            out = m(x)
            x0 = dict(x)
            x0["region"] = torch.cat([x["region"][:, :3],
                                      torch.zeros(2, 1, 224, 224, device=device)], 1)
            # non-zero mask must change the logits once the channel is trained; at
            # init the extra weights are zero, so zeroing the mask is a no-op
            out0 = m(x0)
        check(out["logits_class"].shape == (2, 114), "%s: logits shape %s"
              % (name, tuple(out["logits_class"].shape)))
        w = convs[0].weight[:, 3]
        check(float(w.abs().max()) == 0.0,
              "%s: mask channel weights start at exactly zero" % name)
        check(torch.allclose(out["logits_class"], out0["logits_class"], atol=1e-4),
              "%s: zero mask reproduces the 3-channel output" % name)


def test_ablation_isolation(cfg, device):
    """A switched-off branch must contribute EXACTLY nothing -- not even via the gate.

    This exists because a real diagnostic raised a real doubt: the region_scene run
    (Text off) reports gate_alpha_mean = {region 0.777, scene 0.050, text 0.173}, which
    reads as if 17% of the fused representation were still text. It is not. In
    `RSTModel.forward` a disabled branch sets its feature to zeros (z_t = 0 when
    cfg.use_text is False, z_s = 0 when cfg.use_scene is False), so the gate weight
    multiplies a zero vector, and `out_norm` is a LayerNorm -- invariant to the overall
    scale that the wasted mass removes, so only the ratio between the live branches
    survives. The gate is still asked for three weights because eq. 3-14 defines it that
    way; the weights of dead branches are simply inert.

    Reading the code is an argument, not a proof, so this randomises every parameter and
    buffer belonging to the disabled branch and requires bit-identical logits. If a dead
    branch ever leaks into the prediction, the numbers move and this fails.
    """
    print("\n-- ablation isolation: a disabled branch must not reach the logits --")
    from rst.model import RSTModel
    z = np.load(os.path.join(cfg.root, "knowledge", "text_features.npz"), allow_pickle=True)
    protos = {"tokens": torch.from_numpy(z["tokens"]), "mask": torch.from_numpy(z["mask"]),
              "class_of_token": None}
    attr_dim = int(z["attr_matrix"].shape[1])

    text_mods = ("text_proj", "tm_attn", "text_memory", "proto", "sem_scale")
    scene_mods = ("scene_proj", "rs_attn")
    cases = [("region_scene", text_mods, "Text"),
             ("region_only", scene_mods + text_mods, "Scene and Text")]

    for ablation, off, label in cases:
        c = Config()
        c.root = cfg.root
        resolve_stage(c, 3, caps_from(ablation))
        model = RSTModel(c, protos, attr_dim=attr_dim).to(device).eval()
        batch = build_batch(c, device, scene_mode="cached")
        with torch.no_grad():
            out0 = model(batch)
        logits0 = out0["logits_class"].clone()
        check("logits_semantic" not in out0,
              "%s: emits no logits_semantic (Text off, so eq. 3-17 has no S_c term)"
              % ablation)
        alpha = out0["gate"]
        check(abs(float(alpha.sum(1).mean()) - 1.0) < 1e-4,
              "%s: gate weights still sum to 1 (%s)"
              % (ablation, [round(float(x), 3) for x in alpha.mean(0)]))

        g = torch.Generator().manual_seed(1234)
        n = 0
        for name, p in model.named_parameters():
            if name.startswith(off):
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g).to(p.device, p.dtype) * 5.0)
                n += 1
        for name, buf in model.named_buffers():
            if name.startswith(off) and buf.is_floating_point():
                with torch.no_grad():
                    buf.copy_(torch.randn(buf.shape, generator=g).to(buf.device, buf.dtype) * 5.0)
                n += 1
        check(n > 0, "%s: scrambled %d tensors of the disabled %s branch"
              % (ablation, n, label))
        with torch.no_grad():
            out1 = model(batch)
        check(torch.equal(logits0, out1["logits_class"]),
              "%s: class logits BIT-IDENTICAL after scrambling %s -> the ablation is real"
              % (ablation, label))


def test_plain_head(cfg, device):
    """The plain "region image + dropout + FC" control must really exclude 3.4.

    Two things are asserted, and the second one is the reason this row had to be
    trained at all. (a) Scrambling the gate, the projections, AttentionPool and out_norm
    leaves the plain head's logits bit-identical, so nothing of 3.4 is in its path.
    (b) For the gated Region Only model, the fused H differs from out_norm(z_region) by
    well under one percent, because a disabled branch contributes a zero vector and
    LayerNorm is invariant to the scale the dead gate weight removes apart from its own
    eps. So bypassing the gate ALONE would have been a no-op -- the plain-head control
    has to drop the projection and pooling machinery too, or it would retrain 87.57
    under a new name.
    """
    print("\n-- plain head: region image + dropout + FC, with no 3.4 in the path --")
    from rst.model import RSTModel
    z = np.load(os.path.join(cfg.root, "knowledge", "text_features.npz"), allow_pickle=True)
    protos = {"tokens": torch.from_numpy(z["tokens"]), "mask": torch.from_numpy(z["mask"]),
              "class_of_token": None}
    attr_dim = int(z["attr_matrix"].shape[1])
    dead = ("gate", "region_proj", "scene_proj", "text_proj", "rs_attn", "tm_attn",
            "pool", "out_norm", "text_memory", "proto", "sem_scale")

    c = Config()
    c.root = cfg.root
    resolve_stage(c, 3, caps_from("region_only"))
    c.plain_head = False
    # one model, both paths: the comparison below is only meaningful if the weights are
    # the same and the architecture is the only difference
    model = RSTModel(c, protos, attr_dim=attr_dim).to(device).eval()
    batch = build_batch(c, device, scene_mode="cached")

    with torch.no_grad():
        og = model(batch, need_features=True)
    # Removing ONLY the gate is close to a no-op: with the other branches zeroed, H is
    # LN(alpha_r * z_r) and LayerNorm is scale-invariant up to its own eps, which leaves
    # a sub-percent residue (measured, not assumed). So the gate was never the thing that
    # made Region Only more than a plain classifier -- region_proj, the 2-D positional
    # grid and AttentionPool are, and that is what plain_head takes out.
    rel = float((og["entity"] - model.out_norm(og["z_region"])).abs().mean()
                / og["entity"].abs().mean())
    check(rel < 0.05,
          "Region Only: fused H differs from out_norm(z_region) by only %.2f%% on "
          "average, so bypassing the gate alone would be a relabelled rerun" % (100 * rel))

    c.plain_head = True
    with torch.no_grad():
        op = model(batch)
    check(op["gate"] is None, "plain_head emits no gate weights (3.4.3 not executed)")
    check(op["logits_class"].shape == (op["entity"].shape[0], c.n_classes)
          and op["entity"].shape[1] == c.hidden,
          "plain_head: %s feature -> logits %s"
          % (tuple(op["entity"].shape), tuple(op["logits_class"].shape)))
    check(not torch.equal(op["logits_class"], og["logits_class"]),
          "plain_head and Region Only are DIFFERENT models on identical weights "
          "(not a relabelled rerun)")
    logits0 = op["logits_class"].clone()

    g = torch.Generator().manual_seed(4321)
    n = 0
    for name, p in model.named_parameters():
        if name.startswith(dead):
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g).to(p.device, p.dtype) * 5.0)
            n += 1
    for name, buf in model.named_buffers():
        if name.startswith(dead) and buf.is_floating_point():
            with torch.no_grad():
                buf.copy_(torch.randn(buf.shape, generator=g).to(buf.device, buf.dtype) * 5.0)
            n += 1
    check(n > 0, "plain_head: scrambled %d tensors of the projection/gate/pool modules" % n)
    with torch.no_grad():
        out1 = model(batch)
    check(torch.equal(logits0, out1["logits_class"]),
          "plain_head: class logits BIT-IDENTICAL with 3.4 randomised -> real control")

    c.use_scene = True
    try:
        model(batch)
        check(False, "plain_head with use_scene on must raise, not discard Scene")
    except ValueError:
        check(True, "plain_head + use_scene raises instead of silently dropping a branch")
    c.use_scene = False
    c.plain_head = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--gpu", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.root = os.path.abspath(args.root)
    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    elif args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print("=" * 74)
    print("RST synthetic self-test  |  device=%s  |  root=%s" % (device, cfg.root))
    print("torch %s  transformers-compatible build cuda=%s"
          % (torch.__version__, torch.version.cuda))
    print("=" * 74)

    test_plateau()
    test_caps()
    test_hard_pair_metric()
    test_baseline_mask_channel(device)
    test_ablation_isolation(cfg, device)
    test_plain_head(cfg, device)
    test_forward(cfg, device, "cached")
    test_forward(cfg, device, "live")

    print("\n" + "=" * 74)
    if fails:
        print("FAILED %d check(s):" % len(fails))
        for f in fails:
            print("  - " + f)
    else:
        print("all checks passed")
    print("=" * 74)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
