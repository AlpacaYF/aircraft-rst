"""The Region-Scene-Text fusion model (chapter 3).

Data flow per entity:

    region pixels --E_R--> F^R (patch tokens)          3.2.1
    scene tiles   --E_S--> F^S (pooled tile tokens)    3.2.2 / 3.2.3
    F^R attends to F^S                              -> F_RS   3.4.1
    F_RS attends to the class-agnostic semantic memory -> z_T  3.3.4
    gate(z_R, z_S, z_T)                             -> H      3.4.3
    heads: domain, class, semantic matching, attribute 3.4.4

Text is class-specific, so it enters two ways: as a memory the entity queries
(class-agnostic, safe to fuse before the label is known) and as per-class
prototypes used for the matching logits and the contrastive losses.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import SigLIP2Encoder, TilePositionalEncoding, Sincos2D


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv, kv_mask=None):
        out, w = self.attn(q, kv, kv, key_padding_mask=kv_mask, need_weights=False)
        return self.norm(q + out), w


class AttentionPool(nn.Module):
    """Learned query that collapses a token set to one vector."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)

    def forward(self, x, mask=None):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=mask, need_weights=False)
        return out.squeeze(1)


class RSTModel(nn.Module):
    def __init__(self, cfg, prototypes=None, attr_dim=0):
        """prototypes: [n_classes, n_tokens, text_dim] offline Text Features (3.3.4)."""
        super().__init__()
        self.cfg = cfg
        extra = 1 if cfg.use_mask_channel else 0
        self.region_enc = SigLIP2Encoder(cfg.vision_model, cfg.freeze_vision, extra,
                                         load_text=False,
                                         grad_checkpoint=cfg.grad_checkpoint)
        if cfg.share_vision:
            self.scene_enc = self.region_enc
        else:
            self.scene_enc = SigLIP2Encoder(cfg.vision_model, cfg.freeze_vision, 0,
                                            load_text=False,
                                            grad_checkpoint=cfg.grad_checkpoint)
        d = self.region_enc.embed_dim

        # SigLIP ships a learned positional embedding sized for exactly one input
        # resolution. Scene stays at it because the offline tile cache (3.5.4) was
        # built at that size; the region path may deviate, in which case the encoder
        # bicubically interpolates the position embedding (SigLIP2Encoder._run_vision).
        native = self.region_enc.native_image_size
        if cfg.scene_input != native:
            raise ValueError("cfg.scene_input=%d but %s only accepts %d, and the scene "
                             "token cache was built at that size"
                             % (cfg.scene_input, cfg.vision_model, native))
        if cfg.region_input != native:
            if cfg.region_input % 16:
                raise ValueError("region_input=%d must be a multiple of the 16px patch "
                                 "grid to interpolate positions" % cfg.region_input)
            print("NOTE region_input=%d != the encoder's native %d; position embeddings "
                  "are interpolated (576 -> %d region tokens)"
                  % (cfg.region_input, native, (cfg.region_input // 16) ** 2))
        # 3.2.3's token budget has to hold in the worst case, where every tile that
        # survives the cap carries an entity and so keeps its full kn x kn grid.
        worst = cfg.scene_max_tiles * cfg.scene_near_pool ** 2
        if worst > cfg.scene_token_cap:
            raise ValueError(
                "scene_max_tiles=%d x scene_near_pool^2=%d = %d scene tokens, over "
                "scene_token_cap=%d; lower scene_max_tiles or scene_near_pool"
                % (cfg.scene_max_tiles, cfg.scene_near_pool ** 2, worst,
                   cfg.scene_token_cap))
        h = cfg.hidden

        self.region_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.scene_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.tile_pos = TilePositionalEncoding(h)
        self.region_pos = Sincos2D(h)

        self.rs_attn = CrossAttention(h, cfg.attn_heads, cfg.dropout)      # 3.4.1
        self.text_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.tm_attn = CrossAttention(h, cfg.attn_heads, cfg.dropout)      # entity -> semantic memory
        self.pool = AttentionPool(h)

        # 3.4.3 dynamic gating
        self.gate = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(h, 3))
        self.out_norm = nn.LayerNorm(h)

        self.head_domain = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(h, cfg.n_domains))
        self.head_class = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(h, cfg.n_classes))
        self.head_attr = nn.Linear(h, attr_dim) if attr_dim else None
        self.sem_scale = nn.Parameter(torch.tensor(10.0))
        # adaptive inference weighting (Route C, direction 3): w = sigmoid(mlp(
        # [max visual logit, margin])), initialised near sigmoid(0) = 0.5
        self.sem_gate = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.sem_gate[-1].weight)
        nn.init.zeros_(self.sem_gate[-1].bias)

        # learnable residual on the KB prototypes; None unless cfg.proto_learnable
        self.proto_delta = None
        if prototypes is not None:
            self.set_prototypes(prototypes)
        else:
            self.register_buffer("proto", torch.zeros(1, 1, d))
            self.register_buffer("proto_class", torch.zeros(1, dtype=torch.long))
            self.register_buffer("proto_mask", torch.zeros(1, 1, dtype=torch.bool))

    # ---- offline semantic knowledge (3.3.4) ----
    def set_prototypes(self, prototypes):
        """prototypes: [C, N, d] float tensor, plus a bool mask of real vs padded tokens.

        Safe to call after the module has been moved to a device: the buffers are
        registered on whatever device the parameters already live on, so a caller
        that hands in CPU tensors (the usual case, they come from an npz) does not
        silently strand `proto` on the CPU.
        """
        dev = next(self.parameters()).device
        if isinstance(prototypes, dict):
            proto = prototypes["tokens"]
            mask = prototypes["mask"]
            cls = prototypes["class_of_token"]
        else:
            proto, mask, cls = prototypes, None, None
        self.register_buffer("proto", proto.float().to(dev))
        self.register_buffer("proto_mask",
                             (torch.ones_like(proto, dtype=torch.bool)[..., 0]
                              if mask is None else mask.bool()).to(dev))
        C, N, _ = proto.shape
        self.register_buffer("proto_class",
                             (torch.arange(C).unsqueeze(1).expand(C, N).reshape(-1)
                              if cls is None else cls.long()).to(dev))
        # KB-initialised learnable prototypes: registered as a real Parameter so
        # .to(device) moves it with the model and the state dict round-trips it.
        # Creating it here (not in __init__) is what makes the shape follow the
        # prototypes that were actually loaded.
        if getattr(self.cfg, "proto_learnable", False):
            self.proto_delta = nn.Parameter(torch.zeros(C, N, proto.size(-1)))
        else:
            self.proto_delta = None

    def _proto_tokens(self):
        """The prototype bank the model actually matches against: the frozen KB
        tokens plus their learned residual, or the tokens as loaded."""
        if self.proto_delta is None:
            return self.proto
        return self.proto + self.proto_delta

    @property
    def text_memory(self):
        """All classes' semantic tokens as one queryable bank (class-agnostic)."""
        keep = self.proto_mask.reshape(-1)
        return self.text_proj(self._proto_tokens().reshape(-1, self.proto.size(-1)))[keep]

    def class_text_tokens(self):
        """[C, N, h] projected prototypes with a padding mask."""
        h = self.text_proj(self._proto_tokens())
        return h, self.proto_mask

    # ---- forward ----
    def cached_scene(self, tokens, tile_meta):
        """Scene tokens precomputed by rst.cache_scene_features.

        The cache stores the same rectangular [T, kn*kn] layout encode_scene
        builds, so a background tile occupies only its first kf*kf slots and the
        rest of its row has to be masked here exactly as it is there -- otherwise
        the cached and live paths would hand the attention different key sets.
        The vision tower is frozen for the scene branch only: region features
        still track it. That is the deliberate trade that makes a trainable
        backbone fit in memory.
        """
        B, T = tile_meta.shape[:2]
        slots = tokens.size(1) // max(1, T)
        kn = int(round(slots ** 0.5))
        if kn * kn != slots:
            raise ValueError(
                "scene cache stores %d tokens per tile, which is not a square grid; "
                "rebuild it with the current scene_near_pool" % slots)
        kf = min(self.cfg.scene_far_pool, kn)
        tok = self.scene_proj(tokens)
        tok = tok + self.tile_pos(tile_meta).repeat_interleave(slots, dim=1)
        valid = torch.ones(B, T, slots, dtype=torch.bool, device=tokens.device)
        valid[..., kf * kf:] = False                          # background tiles
        valid |= (tile_meta[..., 2] > 0.5).unsqueeze(-1)      # entity tiles keep all
        valid &= ~(tile_meta[..., 3] > 0.5).unsqueeze(-1)     # padded tiles keep none
        return tok, ~valid.reshape(B, T * slots)

    def encode_scene(self, tiles, tile_meta):
        """tiles [B_img,T,3,S,S] -> [B_img, T*kn*kn, h] plus a padding mask.

        3.2.3's two-level budget: a tile that carries an aircraft entity keeps a
        kn x kn pooled grid, a background tile is squeezed to kf x kf, so the
        token budget is spent where the target actually is. Far tiles are written
        into the first kf*kf slots of a kn*kn-wide row and the remaining slots are
        masked, which keeps one rectangular tensor instead of ragged per-image
        sequences. The near/far distinction additionally survives as an embedding
        in TilePositionalEncoding.
        """
        B, T = tiles.shape[:2]
        cfg = self.cfg
        kn = cfg.scene_near_pool
        kf = min(cfg.scene_far_pool, kn)
        slots = kn * kn

        flat = tiles.reshape(B * T, *tiles.shape[2:])
        raw = self.scene_enc.patch_tokens(flat)              # [B*T, g*g, d]
        g = self.scene_enc.grid_side(raw)
        d = raw.size(-1)

        near = tile_meta[..., 2] > 0.5                       # [B,T]
        p_near = self.scene_enc.pool_grid(raw, g, kn)        # [B*T, kn*kn, d]
        p_far = self.scene_enc.pool_grid(raw, g, kf)         # [B*T, kf*kf, d]
        if p_far.size(1) < slots:
            p_far = torch.cat([p_far, p_far.new_zeros(p_far.size(0), slots - p_far.size(1), d)], 1)
        pooled = torch.where(near.reshape(-1, 1, 1), p_near, p_far)

        valid = torch.ones(B * T, slots, dtype=torch.bool, device=raw.device)
        valid[~near.reshape(-1), kf * kf:] = False           # far tiles: only kf^2 real
        valid &= ~(tile_meta[..., 3] > 0.5).reshape(-1, 1)   # padded tiles: none real

        tok = self.scene_proj(pooled).view(B, T * slots, -1)
        tok = tok + self.tile_pos(tile_meta).repeat_interleave(slots, dim=1)
        return tok, ~valid.view(B, T * slots)

    def _env_pad(self, batch, s_pad, e2s, total_tokens):
        """Padding mask for the environment view of the scene: everything except
        the entity-bearing tiles. z_R pools F_RS, which already attended to those
        tiles, so a z_S over the same tokens hands the gate two views of the same
        pixels and it correctly floors alpha_Scene (0.050 in every v1 run). Far
        tiles carry what z_R cannot see -- other aircraft, airfield, background.

        An entity whose whole scene is entity-bearing tiles has no environment;
        the row falls back to the full scene view, because an all-masked key set
        makes MultiheadAttention return NaN.
        """
        meta = batch["scene_tile_meta"][e2s]                    # [B, T, 4]
        T = max(1, meta.size(1))
        slots = max(1, total_tokens // T)
        near = meta[..., 2] > 0.5                               # [B, T]
        env_invalid = s_pad | near.repeat_interleave(slots, dim=1)
        all_bad = env_invalid.all(dim=1, keepdim=True)
        return torch.where(all_bad, s_pad, env_invalid)

    def forward(self, batch, need_features=False):
        cfg = self.cfg
        rtokens, rside = self.region_enc.pooled_tokens(
            batch["region"], cfg.region_query_pool)

        if getattr(cfg, "plain_head", False):
            # Table 4-6's "region image only" control: the encoder's mean-pooled region
            # feature goes through dropout and ONE fully-connected layer. Everything 3.4
            # adds -- region_proj, the 2-D positional grid, AttentionPool, the gate and
            # out_norm -- is out of the path, so this is the same backbone, input and
            # training budget as Region Only but with a plain classifier, and it lands on
            # the same footing as the pure-vision baselines of table 4-2.
            # mean(12x12 block-pooled tokens) is exactly mean(24x24 patch tokens), so the
            # pooling the fusion uses does not change what this head sees.
            if cfg.use_scene or cfg.use_text:
                raise ValueError("plain_head replaces the fusion head outright; run it "
                                 "with use_scene/use_text off or the branch it was "
                                 "trained on is silently thrown away")
            v = rtokens.mean(dim=1)
            out = {"logits_domain": self.head_domain(v),
                   "logits_class": self.head_class(v),
                   "logits_attr": self.head_attr(v) if self.head_attr is not None else None,
                   "gate": None, "entity": v}
            if need_features:
                out["tokens_region"] = rtokens
                out["z_region"] = v
            return out

        rq = self.region_proj(rtokens)
        g = rtokens.size(1) // max(1, rside * rside) if rside else 1
        if rside:
            rq = rq + self.region_pos(rside, rq.device).repeat_interleave(g, 0).unsqueeze(0)

        e2s = batch["entity_to_scene"]
        scene = None
        if cfg.use_scene and "scene_tokens" in batch:
            scene = self.cached_scene(batch["scene_tokens"], batch["scene_tile_meta"])
        elif cfg.use_scene and "scene_tiles" in batch:
            scene = self.encode_scene(batch["scene_tiles"], batch["scene_tile_meta"])

        if scene is not None:
            s_tok, s_pad = scene
            s_tok = s_tok[e2s]
            s_pad = s_pad[e2s]
            fused, _ = self.rs_attn(rq, s_tok, s_pad)
            # eq. 3-13 as revised: the region branch carries the scene-enhanced
            # representation F_RS, while the scene branch keeps the raw F_S so its
            # gate weight measures the whole-image contribution on its own.
            # v2's z_scene_env narrows that F_S to the environment tiles only, so
            # the two branches stop being two views of the same pixels.
            z_r = self.pool(fused)
            z_s = self.pool(s_tok, mask=(self._env_pad(batch, s_pad, e2s, s_tok.size(1))
                                         if cfg.z_scene_env else None))
        else:
            fused = rq
            z_r = self.pool(rq)
            z_s = torch.zeros_like(z_r)

        has_text = cfg.use_text and self.proto.size(0) == cfg.n_classes and self.proto.size(1) > 1
        if has_text:
            mem = self.text_memory.unsqueeze(0).expand(z_r.size(0), -1, -1)
            z_t, _ = self.tm_attn(fused, mem, None)
            z_t = self.pool(z_t)
        else:
            z_t = torch.zeros_like(z_r)

        # A plain softmax gate collapses. Region is by far the most informative
        # branch at initialisation, so it wins the gate, which starves z_s and z_t
        # of gradient and locks them out for the rest of training. Measured on the
        # first full runs: alpha = (1.000, 0.000, 0.000) for region_scene, i.e.
        # the Scene branch contributed exactly nothing and the whole tiling /
        # two-level token budget of 3.2.3 had no effect on any prediction.
        # The floor keeps alpha a convex combination, so eq. 3-14 and 3-15 are
        # unchanged, while guaranteeing every branch both contributes and keeps
        # receiving gradient -- a branch that earns it can still rise to
        # eps + (1 - 3*eps) = 0.90.
        eps = cfg.gate_floor
        g_logits = self.gate(torch.cat([z_r, z_s, z_t], dim=-1))
        if getattr(cfg, "gate_active_only", False):
            # Softmax over ACTIVE branches only. A disabled branch carries a zero
            # vector, but the unmasked three-way softmax still hands it mass that
            # then multiplies zero -- with region_text the dead Scene channel took
            # 0.82 of the mass and z_R's effective share collapsed to 46%. Masked,
            # the dead channels get alpha exactly 0 and the live channels renormalise:
            # k=3 reduces bit-identically to eps + (1-3*eps)*softmax(g_logits), so
            # the full model is untouched.
            alive = torch.tensor([True,
                                  cfg.use_scene and scene is not None,
                                  bool(has_text)], device=g_logits.device)
            g = torch.softmax(g_logits.masked_fill(~alive, float("-inf")), dim=-1)
            k = int(alive.sum().item())
            alphas = torch.where(alive.unsqueeze(0),
                                 eps + (1.0 - k * eps) * g,
                                 torch.zeros_like(g))
        else:
            g = torch.softmax(g_logits, dim=-1)
            alphas = eps + (1.0 - 3.0 * eps) * g
        H = self.out_norm(alphas[:, 0:1] * z_r + alphas[:, 1:2] * z_s
                          + alphas[:, 2:3] * z_t)

        logits_domain = self.head_domain(H)
        logits_class = self.head_class(H)
        if getattr(cfg, "proto_head", False) and has_text:
            # Route C direction 1: the classifier weight IS the (masked-mean
            # pooled) prototypes, so the text path is structurally inside the
            # decision. Scaled by 1/sqrt(h) to keep the logit scale sane.
            wtok, wmask = self.class_text_tokens()                # [C,N,h], [C,N]
            m = wmask.unsqueeze(-1).to(wtok.dtype)
            W = (wtok * m).sum(1) / m.sum(1).clamp(min=1.0)       # [C,h]
            logits_class = F.linear(H, W * (W.size(-1) ** -0.5),
                                    self.head_class[-1].bias)
        logits_attr = self.head_attr(H) if self.head_attr is not None else None

        # eq. 3-12 and 3.5.2: the visual side of the entity<->prototype match is
        # the token set F_RS, deliberately not H and not its pooled form. H
        # already contains z_T, so matching H against text prototypes would score
        # text against itself; the pooled vector would collapse the region-level
        # correspondence 3.4.2 relies on.
        sem = self.semantic_logits(fused) if has_text else None
        sem_weight = None
        if sem is not None and getattr(cfg, "adaptive_sem", False):
            # Route C direction 3: confidence-adaptive convex combination --
            # knowledge gets more say exactly where the visual head is unsure.
            # The gate input is detached: only the tiny gate mlp trains on it.
            with torch.no_grad():
                top2 = logits_class.detach().topk(2, dim=1).values
                feat = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1]], dim=1)
            sem_weight = torch.sigmoid(self.sem_gate(feat))       # [B,1]
            logits_class = (1.0 - sem_weight) * logits_class + sem_weight * sem

        out = {"logits_domain": logits_domain, "logits_class": logits_class,
               "logits_attr": logits_attr, "gate": alphas, "entity": H}
        if sem is not None:
            # In adaptive mode the semantic term is already inside logits_class;
            # re-adding it in eval's lc = lv + logits_semantic would double-count,
            # so the field is omitted there (sem_raw stays for the contrastive
            # losses and for diagnostics).
            if sem_weight is None:
                out["logits_semantic"] = sem
            else:
                out["sem_weight"] = sem_weight
            out["sem_raw"] = sem / self.sem_scale
            if getattr(cfg, "con_detach", False):
                # Route C direction 2: contrastive terms see a DETACHED visual
                # side, so alignment trains only text_proj + prototypes and the
                # visual tower keeps no private copy of the alignment.
                out["sem_raw_con"] = self.semantic_logits(fused.detach()) / self.sem_scale
        if need_features:
            out["tokens_region"] = rq
            # z_r is the pooled F_RS of eq. 3-13, i.e. the visual side `v` of the
            # contrastive objective 3-19. 3.3.2 verifies the knowledge base in
            # exactly this space once stage 2 has aligned it, so it has to be
            # readable from outside the model.
            out["z_region"] = z_r
        return out

    def semantic_logits(self, H, class_ids=None):
        """3.4.2: S_c for every candidate class.

        `H` may be the region-token set F_RS [B,M,h] or an already pooled entity
        vector [B,h]. With the token set, each of a class's semantic tokens is
        matched to the single visual region that explains it best and the
        per-token scores are averaged, which is what makes "tail structure" bind
        to the tail and "engine layout" to the engines as 3.4.2 describes; a
        pooled vector cannot express that correspondence. Averaging over a
        class's semantic tokens (rather than taking their max) keeps one
        spuriously high attribute from deciding the class on its own.
        """
        tok, mask = self.class_text_tokens()          # [C,N,h], [C,N]
        k = F.normalize(tok, dim=-1)                  # [C,N,h]
        if H.dim() == 2:
            q = F.normalize(H, dim=-1).unsqueeze(1).unsqueeze(1)   # [B,1,1,h]
            sim = (q * k.unsqueeze(0)).sum(-1)                     # [B,C,N]
        else:
            q = F.normalize(H, dim=-1)                             # [B,M,h]
            sim = torch.einsum("bmh,cnh->bcnm", q, k).max(dim=-1).values  # [B,C,N]
        keep = mask.unsqueeze(0).to(sim.dtype)                     # [1,C,N]
        per_class = (sim * keep).sum(-1) / keep.sum(-1).clamp(min=1.0)
        if class_ids is not None:
            per_class = per_class[:, class_ids]
        return self.sem_scale * per_class
