"""SigLIP 2 wrappers.

Two non-obvious pieces:

* Patch tokens. ``SiglipVisionModel.last_hidden_state`` is the *pre* layernorm
  stream, so ``post_layernorm`` has to be applied by hand before the tokens are
  used as attention keys/values -- otherwise the Region branch and the offline
  text prototypes live at different scales.

* The Region input can carry a mask channel (GT-box interior) so the encoder can
  tell the target from the context ring added by expansion. Conv1 stays at its
  pretrained 3 channels and a zero-initialised 1-channel branch is summed in, so
  loading a checkpoint is unaffected and the mask starts as a no-op.
"""

import math

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def _vision_body(model):
    for attr in ("vision_model", "visual_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            return sub
    return model


def _find_module(module, kind):
    """The patch embedding lives at .embeddings.patch_embedding in some
    transformers versions and at .patch_embedding in others."""
    for sub in module.modules():
        if isinstance(sub, kind):
            return sub
    return None


def _load_pretrained(cls, name, cache_dir):
    """Prefer an already-downloaded snapshot.

    transformers otherwise issues a HEAD request on every load, so on a host that
    cannot reach huggingface.co a perfectly good local cache still costs five
    retry sleeps and then an OSError. Trying offline first makes a warm cache load
    instantly and only reaches the network when something is genuinely missing.
    """
    base = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        return cls.from_pretrained(name, local_files_only=True, **base)
    except Exception:
        return cls.from_pretrained(name, **base)


class SigLIP2Encoder(nn.Module):
    def __init__(self, name, freeze=False, extra_image_channels=0, cache_dir=None,
                 load_text=True, grad_checkpoint=False):
        super().__init__()
        self.model = _load_pretrained(AutoModel, name, cache_dir)
        # AutoProcessor routes to the slow SiglipTokenizer, which needs a
        # spiece.model file these repos do not ship. The fast Gemma tokenizer
        # loads from tokenizer.json and needs no sentencepiece at all.
        self.tokenizer = _load_pretrained(AutoTokenizer, name, cache_dir)
        vis = _vision_body(self.model)
        self.vision = vis
        self.text = getattr(self.model, "text_model", None)
        self.post_layernorm = getattr(vis, "post_layernorm", None)
        cfg = getattr(vis, "config", None)
        self.embed_dim = int(getattr(cfg, "hidden_size", 768))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.num_heads = int(getattr(cfg, "num_heads", 12))
        self.native_image_size = int(getattr(cfg, "image_size", 384))
        tcfg = getattr(self.text, "config", None) if self.text is not None else None
        self.max_text_len = min(int(getattr(tcfg, "max_position_embeddings", 64)), 64)
        self.freeze = freeze
        self.extra_image_channels = extra_image_channels

        if extra_image_channels:
            conv = _find_module(vis, nn.Conv2d)
            if conv is None:
                raise RuntimeError("could not locate the vision patch-embedding conv")
            self.extra_conv = nn.Conv2d(extra_image_channels, conv.out_channels,
                                        kernel_size=conv.kernel_size, stride=conv.stride,
                                        padding=conv.padding, bias=False)
            nn.init.zeros_(self.extra_conv.weight)
        else:
            self.extra_conv = None

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
        # Without this, a trainable ViT-B at 384 keeps ~200 MB of activations per
        # image and an 8 GB card cannot fit even batch 4 once scene tiles are added.
        # Checkpointing recomputes them in backward: ~40% slower, several-fold less
        # memory.
        if grad_checkpoint:
            self._enable_checkpointing()
        # Prototypes are built offline by build_text_features, so during training
        # the ~1.2 GB text tower is never called. Dropping it here saves that VRAM
        # and keeps checkpoints to the size of the part that actually trains.
        if not load_text and self.text is not None:
            del self.model.text_model
            self.text = None

    # ---- image ----
    def _enable_checkpointing(self):
        """Only SiglipModel is a PreTrainedModel; the vision submodules are plain
        nn.Modules without gradient_checkpointing_enable, so try the levels in
        order and fall back to running without it rather than crashing."""
        kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        for target in (self.model, getattr(self.model, "vision_model", None),
                       getattr(self.vision, "encoder", None)):
            fn = getattr(target, "gradient_checkpointing_enable", None)
            if fn is not None:
                try:
                    fn(**kwargs)
                    return True
                except Exception:
                    continue
        if getattr(self.vision, "encoder", None) is not None:
            self.vision.encoder.gradient_checkpointing = True
            return True
        return False

    def _run_vision(self, rgb):
        """SigLIP's learned position embedding exists for exactly one resolution,
        but HF keeps a bicubic interpolation of it behind an explicit flag. Passing
        it is what lets the 384-pretrained tower accept a 512 region crop, which is
        how the resolution-mismatch hypothesis behind the >=384px accuracy deficit
        gets tested without retraining (`--region-input`)."""
        return self.vision(pixel_values=rgb,
                           interpolate_pos_encoding=rgb.shape[-1] != self.native_image_size)

    def patch_tokens(self, px):
        """[B,C,H,W] -> [B, g*g, d] with the post layernorm applied."""
        rgb = px[:, :3]
        out = self._run_vision(rgb)
        tokens = out.last_hidden_state
        if self.extra_conv is not None and px.size(1) > 3:
            add = self.extra_conv(px[:, 3:3 + self.extra_image_channels])
            add = add.flatten(2).transpose(1, 2)
            tokens = tokens + add
        if self.post_layernorm is not None:
            tokens = self.post_layernorm(tokens)
        return tokens

    @staticmethod
    def grid_side(tokens):
        n = tokens.size(1)
        g = int(round(math.sqrt(n)))
        return g if g * g == n else None

    @staticmethod
    def pool_grid(tokens, g, k):
        """[N, g*g, d] -> [N, k*k, d] by average-pooling the patch grid.

        Split out from pooled_tokens so the scene branch can pool entity-bearing
        tiles and background tiles to different widths in one encoder pass
        (3.2.3's two-level token budget).
        """
        if g is None or k <= 0 or k >= g:
            return tokens
        d = tokens.size(-1)
        grid = tokens.reshape(tokens.size(0), g, g, d).permute(0, 3, 1, 2)
        grid = nn.functional.adaptive_avg_pool2d(grid, k)
        return grid.permute(0, 2, 3, 1).reshape(tokens.size(0), k * k, d)

    def pooled_tokens(self, px, k):
        """Patch grid adaptively pooled to k x k, keeping spatial order."""
        tokens = self.patch_tokens(px)
        g = self.grid_side(tokens)
        if g is None or k <= 0 or k >= g:
            return tokens, g or 1
        return self.pool_grid(tokens, g, k), k

    def contrastive_feature(self, px):
        """The image embedding SigLIP was actually trained to align with text: the
        attention pooling head's output (`get_image_features`).

        Keep this separate from `image_feature`. That one averages the patch tokens,
        which is a fine visual descriptor for image-to-image similarity (3.3.3 mines
        hard pairs with it) but it is NOT in the shared space, so comparing it against
        a text embedding measures nothing -- which is exactly how the SigLIP 2
        zero-shot row of 表4-2 came to be computed the first time.
        """
        return self.model.get_image_features(pixel_values=px[:, :3])

    def image_feature(self, px):
        """A single vector per crop, mean-pooled over the patch tokens.

        This is NOT SigLIP's contrastive image embedding (that one is the attention
        pooling head's output, which HF returns from `get_image_features`). It is a
        visual descriptor, and it is the right thing for the only caller that needs
        it: `rst.extract_embeddings`, whose features are compared image-to-image by
        3.3.3's hard-pair mining. Measured on the test split these mean-pooled
        features reach 38.6% prototype Top-1 and 44.8% 1-NN Top-1 against a 0.88%
        chance rate, and `index/hard_table.npy` agrees with their top-8 neighbours
        100% of the time -- so the visual side of the frozen pipeline is sound.

        Do not use this for image-to-text similarity -- use `contrastive_feature`.
        tools/probe_siglip_loader.py measures why: on this class set the frozen *text*
        embeddings are the degenerate side (mean pairwise cosine 0.95 for the class
        names), which is why 3.3.2's retrieval verdict sits at chance.
        """
        out = self._run_vision(px[:, :3])
        if getattr(out, "image_embeds", None) is not None:
            return out.image_embeds
        return self.post_layernorm(out.last_hidden_state).mean(dim=1)

    # ---- text ----
    def encode_text(self, texts):
        """SigLIP pools the text tower at `last_hidden_state[:, -1, :]` -- HF's
        SiglipTextTransformer hard-codes that position with the comment "Assuming
        sticky EOS tokenization, last token is always EOS". It does not consult the
        attention mask.

        So the terminator has to BE the last column. The fast Gemma tokenizer this
        repo ships pads on the right, which puts a <pad> at position -1 and silently
        pools every prompt from a padding token. Measured on this dataset that cost a
        lot: class-prompt pairwise cosine 0.9785 instead of 0.9038, and a
        military-vs-civilian 2-way control 86.50% instead of 97.00%. Left padding
        restores the assumption the model was built on.

        The batch must also carry a real attention mask -- GemmaTokenizerFast does not
        return one unless asked -- and be padded to a fixed length so position -1 is
        the terminator for every row regardless of the longest string in the batch.
        """
        if self.text is None:
            raise RuntimeError("checkpoint has no text tower; cannot build prototypes")
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            tok = self.tokenizer(text=list(texts), padding="max_length", truncation=True,
                                 max_length=self.max_text_len, return_tensors="pt",
                                 return_attention_mask=True)
        finally:
            self.tokenizer.padding_side = side
        dev = next(self.text.parameters()).device
        return self.model.get_text_features(input_ids=tok["input_ids"].to(dev),
                                            attention_mask=tok["attention_mask"].to(dev))

    def text_token_lengths(self, texts):
        return [len(self.tokenizer.tokenize(t)) for t in texts]


class TilePositionalEncoding(nn.Module):
    """Marks where each scene tile came from, so pooled tokens from different
    parts of the image are not interchangeable."""

    def __init__(self, dim, max_grid=32):
        super().__init__()
        self.row = nn.Embedding(max_grid + 1, dim // 2)
        self.col = nn.Embedding(max_grid + 1, dim // 2)
        self.near = nn.Embedding(2, dim)
        nn.init.zeros_(self.row.weight)
        nn.init.zeros_(self.col.weight)
        nn.init.zeros_(self.near.weight)

    def forward(self, meta):
        """meta [B,T,4] = (gi, gj, near, is_pad) -> [B,T,dim]"""
        gi = meta[..., 0].clamp(0, 31).long()
        gj = meta[..., 1].clamp(0, 31).long()
        near = meta[..., 2].clamp(0, 1).long()
        emb = torch.cat([self.row(gj), self.col(gi)], dim=-1) + self.near(near)
        return emb * (1.0 - meta[..., 3:4])


class Sincos2D(nn.Module):
    """Fixed 2-D positions for a token grid; used so region queries and scene keys
    both know their own layout."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, side, device):
        half = self.dim // 4
        pos = torch.arange(side, device=device, dtype=torch.float32)
        ang = torch.arange(half, device=device, dtype=torch.float32)
        ang = torch.exp(ang * (-math.log(10000.0) / max(1, half - 1)))
        ph = pos[:, None] * ang[None, :]
        pe = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1)
        rows = pe[:, None, :].expand(side, side, pe.size(-1))
        cols = pe[None, :, :].expand(side, side, pe.size(-1))
        out = torch.cat([rows, cols], dim=-1).reshape(side * side, self.dim)
        pad = self.dim - out.size(-1)
        if pad:
            out = torch.nn.functional.pad(out, (0, pad))
        return out
