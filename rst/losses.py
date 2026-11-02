"""Training objectives, eq. 3-18 through 3-21.

L = L_domain + L_class + lambda1*L_con + lambda2*L_hard + lambda3*L_diff

L_con and L_hard both reuse the model's raw entity-vs-prototype similarity
matrix rather than recomputing it, which keeps the extra cost at one gather.
"""

import torch
from torch import nn
import torch.nn.functional as F


class RSTLoss(nn.Module):
    def __init__(self, cfg, class_weights=None, attr_matrix=None, hard_table=None):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("cw", class_weights if class_weights is not None
                             else torch.ones(cfg.n_classes))
        # [C, n_attr] 0/1 which attribute sentences belong to each class
        self.register_buffer("attr", attr_matrix if attr_matrix is not None
                             else torch.zeros(1, 1))
        # [C, K] indices of each class's hardest negative classes. A zero-filled
        # stand-in must NOT be read as "class 0 is everyone's hard negative":
        # that silently trains the model to suppress one arbitrary class, so the
        # loss is gated on has_hard the same way the diff term is gated on attr.
        self.has_hard = hard_table is not None
        self.register_buffer("hard", hard_table if hard_table is not None
                             else torch.zeros(cfg.n_classes, 1, dtype=torch.long))

    def forward(self, out, batch):
        cfg = self.cfg
        y = batch["class"]
        d = batch["domain"]
        parts = {}

        parts["domain"] = F.cross_entropy(out["logits_domain"], d)

        logits = out["logits_class"]
        # respect sem_at_inference here too, so the head is never trained against a
        # term that will be absent when the model is scored
        if "logits_semantic" in out and cfg.use_text and cfg.sem_at_inference:
            logits = logits + out["logits_semantic"]
        w = self.cw if cfg.balance_class_loss else None
        parts["class"] = F.cross_entropy(logits, y, weight=w,
                                         label_smoothing=cfg.label_smoothing)

        if "sem_raw" in out and cfg.use_text:
            # con_detach routes the contrastive terms through a stop-gradient
            # visual side (sem_raw_con) when present; the additive S_c used by
            # the class loss keeps its gradients.
            con_src = out.get("sem_raw_con", out["sem_raw"])
            parts["con"] = self._contrastive(con_src, y)
            if cfg.use_hard_negative and self.has_hard:
                parts["hard"] = (self._margin(con_src, y) if cfg.margin_hard
                                 else self._hard(con_src, y))
            else:
                parts["hard"] = logits.new_zeros(())
        else:
            parts["con"] = logits.new_zeros(())
            parts["hard"] = logits.new_zeros(())

        if cfg.use_diff_loss and out.get("logits_attr") is not None \
                and self.attr.size(1) > 1:
            parts["diff"] = F.binary_cross_entropy_with_logits(
                out["logits_attr"], self.attr[y], pos_weight=self._pos_weight())
        else:
            parts["diff"] = logits.new_zeros(())

        total = (parts["domain"] + parts["class"]
                 + cfg.lambda_con * parts["con"]
                 + cfg.lambda_hard * parts["hard"]
                 + cfg.lambda_diff * parts["diff"])
        return total, parts

    def _pos_weight(self):
        pos = self.attr.sum(0).clamp(min=1.0)
        return ((self.attr.size(0) - pos) / pos).clamp(max=50.0).to(self.attr.device)

    def _contrastive(self, sem_raw, y):
        """CE against all prototypes, with hard negatives upweighted in the
        denominator by cfg.con_hard_weight. logsumexp over [logits, log(w-1) +
        hard logits] adds (w-1)*exp(s_hard) to the partition function, so w=2 is
        exactly "count each hard negative twice" without rebuilding the class
        axis. w=1 (or hard off) is the plain CE the v1 runs used."""
        logits = sem_raw / self.cfg.temperature
        w = float(self.cfg.con_hard_weight)
        if w <= 1.0 or not self.has_hard or not self.cfg.use_hard_negative:
            return F.cross_entropy(logits, y)
        cand = self.hard[y].clamp(0, logits.size(1) - 1)          # [B, K]
        extra = torch.log(sem_raw.new_tensor(w - 1.0)) \
            + logits.gather(1, cand)
        extra = extra.masked_fill(cand == y[:, None], float("-inf"))
        z = torch.logsumexp(torch.cat([logits, extra], dim=1), dim=1)
        return (z - logits.gather(1, y[:, None]).squeeze(1)).mean()

    def _margin(self, sem_raw, y):
        """Additive-margin CE restricted to {y} u hard(y) (ArcFace-style).

        The hinge this replaces was identically zero through stage 3: balanced
        sampling drove train Top-1 to 1.000, every true-vs-hard pair cleared the
        margin and the term contributed no gradient at all. A CE over the
        restricted candidate set keeps a softmax gradient as long as any
        confusable rival stays competitive, which is the regime the term exists
        for; the additive margin additionally demands the true class win by
        margin_m, not merely win."""
        cand = self.hard[y]                                       # [B, K]
        valid = (cand != y[:, None]).any(dim=1)
        if not valid.any():
            return sem_raw.new_zeros(())
        cand = cand.clamp(0, sem_raw.size(1) - 1)
        full = torch.cat([y[:, None], cand], dim=1)               # [B, 1+K]
        sim = sem_raw.gather(1, full)
        # margin on the true class only, applied functionally so autograd sees
        # a clean graph; degenerate hard columns (a class's row pointing at
        # itself, the 3.3.3 sentinel for uncovered classes) are masked to -inf
        # so they contribute nothing to the restricted partition function --
        # column 0 is the target and must never be masked
        bias = torch.zeros_like(sim)
        bias[:, 0] = self.cfg.margin_m
        degen = torch.cat([torch.zeros(y.size(0), 1, dtype=torch.bool, device=y.device),
                           cand == y[:, None]], dim=1)
        sim = (sim - bias).masked_fill(degen, float("-inf"))
        loss = F.cross_entropy(sim / self.cfg.margin_tau,
                               torch.zeros_like(y), reduction="none")
        return loss[valid].mean()

    def _hard(self, sem_raw, y):
        """Online hard negative: among the true class's pre-mined confusable
        classes, take the one the model currently scores highest."""
        cand = self.hard[y]                                  # [B, K]
        valid = (cand != y[:, None]).any(dim=1)
        if not valid.any():
            return sem_raw.new_zeros(())
        scores = sem_raw.gather(1, cand.clamp(max=sem_raw.size(1) - 1))  # [B,K]
        scores = scores.masked_fill(cand == y[:, None], float("-inf"))
        neg_idx = scores.argmax(dim=1)
        neg = cand.gather(1, neg_idx[:, None]).squeeze(1).clamp(max=sem_raw.size(1) - 1)
        pos_s = sem_raw.gather(1, y[:, None]).squeeze(1)
        neg_s = sem_raw.gather(1, neg[:, None]).squeeze(1)
        loss = F.relu(self.cfg.hard_margin + neg_s - pos_s)
        return loss[valid].mean()


def entity_metrics(out, batch):
    """Top-1 / Top-5 / domain accuracy for the progress bar."""
    logits = out["logits_class"]
    if "logits_semantic" in out:
        logits = logits + out["logits_semantic"]
    y = batch["class"]
    top1 = (logits.argmax(1) == y).float().mean().item()
    k = min(5, logits.size(1))
    top5 = (logits.topk(k, dim=1).indices == y[:, None]).any(1).float().mean().item()
    dom = (out["logits_domain"].argmax(1) == batch["domain"]).float().mean().item()
    return {"top1": top1, "top5": top5, "domain_acc": dom}
