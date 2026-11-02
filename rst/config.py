"""Hyper-parameters for the Region-Scene-Text model.

Defaults follow table 4-2 of the proposal; every field is overridable from the
command line so the ablations in tables 4-6 and 4-7 are flag changes rather
than code changes.
"""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class Config:
    # --- paths ---
    root: str = "."
    index_dir: str = "index"
    knowledge_dir: str = "knowledge"
    text_features: str = "knowledge/text_features.npz"
    out_dir: str = "runs"

    # --- encoders ---
    vision_model: str = "google/siglip2-base-patch16-384"
    text_model: str = "google/siglip2-base-patch16-384"
    share_vision: bool = True          # one vision tower feeds Region and Scene
    freeze_vision: bool = False
    grad_checkpoint: bool = True   # without it an 8 GB card cannot train the backbone

    # --- input geometry (table 4-2) ---
    region_input: int = 384
    scene_input: int = 384             # each tile is resized to this before encoding
    region_expand: float = 0.25
    region_query_pool: int = 12        # pool 24x24 patch grid -> 12x12 to bound attention cost
    scene_token_cap: int = 1024
    # 3.2.3 two-level scene budget: a tile that carries an aircraft entity keeps a
    # 7x7 pooled grid, a background tile is squeezed to 2x2.
    scene_near_pool: int = 7         # 49 tokens per entity-bearing tile
    scene_far_pool: int = 2          # 4 tokens per background tile
    # Worst case every kept tile is entity-bearing, so the budget has to hold at
    # near_pool**2 per tile; RSTModel asserts this when it is built.
    scene_max_tiles: int = 20        # 20 * 49 = 980 <= scene_token_cap
    use_scene_cache: bool = False      # read pooled scene tokens from disk (3.5.4)
    scene_cache_dir: str = "cache"
    augment_flip: bool = True
    augment_color: bool = True

    # --- which modalities are active (drives tables 4-6 / 4-7) ---
    use_scene: bool = True
    use_text: bool = True
    use_hard_negative: bool = True
    use_diff_loss: bool = True
    use_mask_channel: bool = True

    # --- v2 mechanisms (all default OFF; existing scripts and runs reproduce) ---
    # 1) Scene de-redundancy. z_R pools F_RS, which already attended to the
    #    entity-bearing tiles, so a z_S over the SAME tokens is a second view of
    #    the same pixels and the gate floors alpha_Scene (measured 0.050 in every
    #    v1 run). With this on, z_S pools only the environment tiles (far tiles),
    #    which z_R cannot see: other aircraft, airfield, background.
    z_scene_env: bool = False
    # 2) Learnable text prototypes, KB-initialised: P = P_kb + delta. The frozen
    #    buffer lets training only reshape a projection around fixed directions;
    #    a residual keeps the KB as the init while letting the prototype space
    #    adapt. A random-init control under the same architecture isolates the
    #    knowledge's contribution.
    proto_learnable: bool = False
    # 3) L_hard replacement. The hinge on sem_raw is identically zero once train
    #    Top-1 reaches 1.0 (every pair clears margin 0.1, stage 3 gets no
    #    gradient at all). A CE over the restricted candidate set {y} u hard(y)
    #    with an additive margin on the true class keeps a gradient as long as a
    #    confusable rival stays competitive -- exactly the regime it targets.
    margin_hard: bool = False
    margin_m: float = 0.1           # additive margin on the true-class cosine;
                                    # 0.1 ~ 0.9 sigma of the measured S_c spread
    margin_tau: float = 0.05        # cos/tau, i.e. logit scale ~20
    # 4) Hard-negative reweighting inside L_con's softmax denominator. Adding
    #    (w-1) copies of each hard negative's logit upweights it by w; 1.0 = the
    #    plain CE the v1 runs used.
    con_hard_weight: float = 1.0
    # 5) Re-mine the confusable sets from THIS model's validation confusion at
    #    the stage-2 -> stage-3 boundary (3.5.3 made model-aware). Classes with
    #    no validation errors keep their 3.3.3 row.
    remine_stage3: bool = False
    # 7) Augmentation that keeps train Top-1 below 1.0, which is what let every
    #    discriminative loss saturate to zero in the v1 runs. Erase zeroes one
    #    rectangle across all region channels (RGB mid-gray + mask background);
    #    expand jitter samples the context expansion per entity.
    erase_prob: float = 0.0
    expand_jitter: float = 0.0      # +/- around region_expand, e.g. 0.10 for 0.15..0.35
    # Route B (few-shot knowledge-value experiment): cap TRAIN entities per class
    # at K, sampled with a FIXED seed so every configuration and seed sees the
    # same subset -- the ablation then compares architectures on identical data.
    # Val/test stores are never capped (EntityDataset passes shots only when
    # train=True), and eval_checkpoint's stores are untouched by construction.
    # hard_table and text_features are sample-count independent, which is the
    # point: they are the prior whose value should grow as labels get scarce.
    few_shot: int = 0               # 0 = off
    few_shot_seed: int = 12345
    # --- Route-C text-path probes (all default off) ---
    # proto_head: the classifier weight IS the class prototypes -- W = text_proj
    # (pooled P_kb + dP) -- so the text path is physically part of the decision
    # instead of an additive term a visual head can bypass. Requires use_text.
    proto_head: bool = False
    # adaptive_sem: logits = (1-w)*visual + w*S_c with w = sigmoid(mlp(confidence,
    # margin)): the knowledge prior gets more say exactly when the visual head is
    # unsure (the conf:low subset is where S_c already helped most). Mutually
    # exclusive with proto_head (there the prototype IS the visual side).
    adaptive_sem: bool = False
    # con_detach: L_con/L_hard/L_margin see sem_raw computed from a DETACHED
    # fused representation, so contrastive alignment updates only text_proj and
    # the prototypes instead of also leaking into the visual tower -- the leak is
    # why S_c is redundant at inference in the first place.
    con_detach: bool = False

    # --- heads ---
    n_classes: int = 114
    n_domains: int = 2
    hidden: int = 768
    attn_heads: int = 12
    dropout: float = 0.1

    # --- losses (eq. 3-21) ---
    lambda_con: float = 1.0          # table 4-2: lambda1 / lambda2 / lambda3
    lambda_hard: float = 1.0
    lambda_diff: float = 0.5
    temperature: float = 0.07
    hard_margin: float = 0.1
    # Measured on the first full runs: the spread of S_c across the 114 classes is
    # only ~0.112, so the original margin 0.3 asked for a 2.7-sigma separation on
    # every hard pair and over-constrained the embedding (full scored 0.8 points
    # below the same model without the term). 0.1 is ~0.9 sigma.
    gate_floor: float = 0.05         # min alpha per branch, see RSTModel.forward
    # Gate over ACTIVE branches only. The three-way softmax lets a disabled
    # branch's channel swallow softmax mass that multiplies a zero vector: with
    # region_text (use_scene off) the dead Scene channel took 0.82 of the mass
    # and z_R's effective share fell to 46%, which is how the first v2_rt run
    # produced a distorted representation that scored well on val and poorly on
    # test. With this on, disabled channels are masked out of the softmax and
    # get alpha exactly 0, while the all-branches-on case is bit-identical to
    # the old formula (k=3 reduces to eps + (1-3*eps)*softmax). Default off so
    # existing checkpoints keep scoring exactly as they were trained.
    gate_active_only: bool = False
    # Whether eq. 3-17's additive beta*S_c term is used at INFERENCE. Set False to make
    # the Text branch a training-time regulariser only: L_con / L_hard / L_diff still
    # shape the shared visual features, but the class logits come from head_class alone.
    # Motivated by measurement, not taste -- L_con trains S_c on the same `fused`
    # features the visual head uses, so the two are redundant at inference (S_c alone
    # scored 86.98% against the whole model's 87.14%, a net +0.03), while the near
    # collinear prototypes make the additive term mostly noise.
    sem_at_inference: bool = True
    # Table 4-6's "region image only" control: the mean-pooled SigLIP 2 region feature
    # goes through dropout and one fully-connected layer, and the whole 3.4 path
    # (region_proj, the 2-D positional grid, AttentionPool, the gate, out_norm) is out of
    # the classification path. Without it, the Region Only row still runs 3.4's pooling
    # and head machinery, so it measures "the region modality only", not "a plain
    # classifier on a region image" -- which is the number the chapter claims.
    # Requires use_scene and use_text off; RSTModel raises otherwise rather than letting
    # a trained branch vanish silently.
    plain_head: bool = False
    balance_class_loss: bool = True    # long-tail: weight CE by inverse frequency

    # --- optimisation (table 4-2) ---
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 16
    epochs: int = 30
    warmup_epochs: int = 3
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    amp: bool = True
    grad_clip: float = 1.0
    num_workers: int = 8
    sampler: str = "balanced"          # balanced | natural

    # --- runtime ---
    seed: int = 0
    stages: tuple = (1, 2, 3)
    stage_epochs: int = 0              # 0 -> split cfg.epochs across stages
    log_every: int = 20
    eval_every_epoch: int = 1
    max_train_batches: int = 0         # >0 truncates, for smoke tests
    resume: str = ""

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = Config()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
        return cfg
