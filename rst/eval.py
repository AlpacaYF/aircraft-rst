"""Metrics for the paper's tables: Top-1/Top-5, Macro-F1 overall and military-only,
plus the military/civilian attribute scores of table 4-5.

Macro-F1 is computed directly rather than via sklearn so the server environment
stays small.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:                                    # pragma: no cover
    def tqdm(x, **kw):
        return x


def macro_f1(y_true, y_pred, n_classes):
    """Mean of per-class F1 over classes that actually appear."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1, present = [], []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if (y_true == c).sum() == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        present.append(c)
    return float(np.mean(f1)) if f1 else 0.0, present


def binary_prf(y_true, y_pred, positive):
    tp = int(((y_true == positive) & (y_pred == positive)).sum())
    fp = int(((y_true != positive) & (y_pred == positive)).sum())
    fn = int(((y_true == positive) & (y_pred != positive)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def topk_correct(logits, y, k):
    pred = logits.topk(min(k, logits.size(1)), dim=1).indices
    return (pred == y[:, None]).any(1).cpu().numpy()


# Shared by rst.eval_checkpoint and rst.baselines. The entity long side is the one
# variable that separates "resolution mismatch" from "the model is weak": every crop
# is forced to 384^2, so accuracy should peak near that size and fall off both ways.
# Anything coarser than this hides the effect -- over half the test entities sit in
# the single >=384 bucket.
SIZE_BUCKETS = [(0, 64, "<64px"), (64, 128, "64-128"), (128, 256, "128-256"),
                (256, 384, "256-384"), (384, 768, "384-768"), (768, 1536, "768-1536"),
                (1536, 10 ** 9, ">=1536px")]


def hard_pair_accuracy(logits, y, hard_table):
    """Table 4-7's "Hard Pair Accuracy": restrict the decision to the true class plus
    the confusable types mined in 3.3.3, and ask whether the true class still wins.

    Overall Top-1 is dominated by easy classes, so a mechanism whose entire purpose is
    separating visually similar types (the hard-negative term, the difference
    knowledge) can move this metric while leaving Top-1 inside the noise floor. This
    is the measurement that can actually show it working.
    """
    if hard_table is None:
        return None
    table = torch.as_tensor(hard_table, dtype=torch.long, device=logits.device)
    yt = torch.as_tensor(y, dtype=torch.long, device=logits.device)
    # column 0 must be the TRUE CLASS index, not the row index: gather selects
    # columns, and argmax == 0 then reads as "the true class beat every hard rival"
    cand = torch.cat([yt.unsqueeze(1), table[yt]], dim=1)
    return float((logits.gather(1, cand).argmax(1) == 0).float().mean())


def entity_long_sides(store, ids):
    size = {"%s#%d" % (r["image"], r["instance_index"]): r.get("long_side", 0)
            for r in store.instances}
    return np.array([size.get(i, 0) for i in ids], dtype=np.float64)


def stratify_by_size(res, store, n_classes):
    """Metrics per entity-size bucket. Needs res["_pred"], res["_y"], res["_ids"]."""
    ls = entity_long_sides(store, res["_ids"])
    pred, y = res["_pred"], res["_y"]
    out = {}
    for lo, hi, name in SIZE_BUCKETS:
        m = (ls >= lo) & (ls < hi)
        if not m.any():
            continue
        out[name] = {"n": int(m.sum()),
                     "top1": round(float((pred[m] == y[m]).mean()), 4),
                     "macro_f1": round(macro_f1(y[m], pred[m], n_classes)[0], 4)}
    return out


@torch.no_grad()
def collect_logits(model, loader, device, cfg, use_amp=True, max_batches=0,
                   use_semantic=True, with_extra=False):
    """One forward pass over `loader` returning raw arrays, no metrics.

    Split out so a caller can combine logits before argmax rather than averaging
    per-pass accuracies, and so the hard-pair metric can reuse the same logits.

    `logits_visual` is the same forward WITHOUT the additive S_c term, so the
    with/without comparison is a property of one set of predictions rather than of two
    passes that could disagree for other reasons.
    """
    model.eval()
    logits_c, logits_v, logits_d, ys, ys_d, ids = [], [], [], [], [], []
    gates, sems = [], []
    for bi, batch in enumerate(tqdm(loader, total=max_batches or len(loader),
                                    leave=False)):
        if max_batches and bi >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and use_amp)):
            out = model(batch)
        lv = out["logits_class"]
        lc = lv + out["logits_semantic"] if (use_semantic and "logits_semantic" in out) else lv
        logits_c.append(lc.float().cpu())
        logits_v.append(lv.float().cpu())
        logits_d.append(out["logits_domain"].float().cpu())
        ys.append(batch["class"].cpu())
        ys_d.append(batch["domain"].cpu())
        ids.extend(batch.get("instance_ids") or [])
        if with_extra:
            # a plain_head run has no gate; record zeros so the array still aligns with
            # the entity list and the reader can check `gate_present` instead of guessing
            gates.append(torch.zeros(lv.size(0), 3) if out["gate"] is None
                         else out["gate"].float().cpu())
            sr = out.get("sem_raw")
            # torch.zeros(...) lands on the CPU by default; zeros_like(lv) would inherit
            # the GPU and then .numpy() would fail on exactly the runs that have no S_c
            sems.append(torch.zeros(lv.size(0), lv.size(1)) if sr is None
                        else sr.float().cpu())
    got = {"logits": torch.cat(logits_c), "logits_visual": torch.cat(logits_v),
           "logits_domain": torch.cat(logits_d),
           "y": torch.cat(ys).numpy(), "yd": torch.cat(ys_d).numpy(), "ids": ids}
    if with_extra:
        got["gate"] = torch.cat(gates)
        got["sem_raw"] = torch.cat(sems)
        got["gate_present"] = bool(torch.cat(gates).abs().sum() > 0)
        got["sem_present"] = bool(torch.cat(sems).abs().sum() > 0)
    return got


def score_predictions(logits, logits_domain, y, yd, store, top5_hit=None):
    n = len(store.classes)
    pred = logits.argmax(1).numpy()
    pd = logits_domain.argmax(1).numpy()
    if top5_hit is None:
        top5_hit = topk_correct(logits, torch.as_tensor(y), 5)

    mil = set(store.military_index)
    mil_mask = np.isin(y, list(mil))
    res = {
        "n": int(len(y)),
        "top1": float((pred == y).mean()),
        "top5": float(np.asarray(top5_hit).mean()),
        "macro_f1": macro_f1(y, pred, n)[0],
        "military_top1": float((pred[mil_mask] == y[mil_mask]).mean()) if mil_mask.any() else 0.0,
        "military_macro_f1": macro_f1(y[mil_mask], pred[mil_mask], n)[0] if mil_mask.any() else 0.0,
        "military_classes_evaluated": int(len(np.unique(y[mil_mask]))) if mil_mask.any() else 0,
        "domain_acc": float((pd == yd).mean()),
    }
    res["military"] = binary_prf(yd, pd, 0)
    res["civilian"] = binary_prf(yd, pd, 1)

    conf = {}
    wrong = pred != y
    if wrong.any():
        pairs = {}
        for a, b in zip(y[wrong], pred[wrong]):
            key = (int(a), int(b))
            pairs[key] = pairs.get(key, 0) + 1
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:20]
        conf = {"%s->%s" % (store.classes[a], store.classes[b]): c
                for (a, b), c in top}
    res["top_confusions"] = conf
    return res, pred


def evaluate(model, loader, device, cfg, store, desc="eval", use_amp=True, max_batches=0,
             use_semantic=None, return_samples=False, hard_table=None, with_extra=False):
    """max_batches > 0 gives a fast, noisy development-time read instead of a
    full pass over the split.

    `use_semantic` overrides only whether S_c is added to the class logits
    (eq. 3-17). Do not ablate it by setting cfg.use_text=False: that also zeroes
    z_t inside the forward, so the gated entity representation H changes and the
    comparison measures a crippled model rather than the missing term.
    """
    if use_semantic is None:
        use_semantic = cfg.use_text and getattr(cfg, "sem_at_inference", True)
    got = collect_logits(model, loader, device, cfg, use_amp=use_amp,
                         max_batches=max_batches, use_semantic=use_semantic,
                         with_extra=with_extra or return_samples)
    res, pred = score_predictions(got["logits"], got["logits_domain"], got["y"],
                                  got["yd"], store,
                                  top5_hit=topk_correct(got["logits"],
                                                        torch.as_tensor(got["y"]), 5))
    if hard_table is not None:
        res["hard_pair_acc"] = hard_pair_accuracy(got["logits"], got["y"], hard_table)
    if return_samples:
        # per-sample arrays so a caller can stratify (by entity size, by tail
        # frequency, by hard-pair membership) without re-running the model
        res["_pred"], res["_y"], res["_ids"] = pred, got["y"], got["ids"]
        res["_got"] = got
    return res


def format_result(res):
    out = ("Top-1 %.2f  Top-5 %.2f  MacroF1 %.2f | mil Top-1 %.2f mil F1 %.2f "
           "(%d cls) | domain %.2f milF1 %.2f civF1 %.2f"
           % (100 * res["top1"], 100 * res["top5"], 100 * res["macro_f1"],
              100 * res["military_top1"], 100 * res["military_macro_f1"],
              res["military_classes_evaluated"], 100 * res["domain_acc"],
              100 * res["military"]["f1"], 100 * res["civilian"]["f1"]))
    if res.get("hard_pair_acc") is not None:
        out += " | hardPair %.2f" % (100 * res["hard_pair_acc"])
    return out
