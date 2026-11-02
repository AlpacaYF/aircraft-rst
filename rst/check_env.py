"""Server preflight: environment, model availability, data integrity, and one
real forward+backward step so nothing surprises you mid-training.

  python -m rst.check_env
  python -m rst.check_env --download-model      # also fetch the encoder weights
"""

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request

OK, WARN, BAD = "[ ok ]", "[warn]", "[FAIL]"


def line(tag, msg):
    print("%s %s" % (tag, msg))


def check_torch():
    line(OK, "python %s" % sys.version.split()[0])
    try:
        import torch
    except ImportError:
        line(BAD, "torch not installed -- run setup.sh first")
        return None
    line(OK if torch.cuda.is_available() else WARN,
         "torch %s | cuda available: %s | build cuda: %s"
         % (torch.__version__, torch.cuda.is_available(), torch.version.cuda))
    if not torch.cuda.is_available():
        line(WARN, "CPU only -- training will be impractically slow")
        return torch
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        p = torch.cuda.get_device_properties(i)
        line(OK, "cuda:%d %s  %.1f/%.1f GB free  sm_%d%d  %d MiB reported"
             % (i, p.name, free / 1e9, total / 1e9, p.major, p.minor,
                int(p.total_memory / 1e6)))
        if p.major < 7:
            line(WARN, "  compute capability < 7.0: bf16 unavailable, use fp16/amp")
    return torch


def check_network(model_name):
    hosts = {"huggingface.co": "https://huggingface.co",
             "hf-mirror.com": "https://hf-mirror.com"}
    reachable = {}
    for name, url in hosts.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rst"})
            with urllib.request.urlopen(req, timeout=8):
                reachable[name] = True
            line(OK, "%s reachable" % name)
        except Exception as exc:
            reachable[name] = False
            line(WARN, "%s unreachable (%s)" % (name, type(exc).__name__))
    if not reachable.get("huggingface.co") and reachable.get("hf-mirror.com"):
        line(WARN, "set  export HF_ENDPOINT=https://hf-mirror.com")
    elif not any(reachable.values()):
        line(BAD, "no model host reachable -- pre-download the encoder and pass --model <path>")
    return reachable


def check_model_present(model_name, cache_dir):
    try:
        from transformers import AutoConfig
        kw = {} if cache_dir is None else {"cache_dir": cache_dir}
        AutoConfig.from_pretrained(model_name, local_files_only=True, **kw)
        line(OK, "encoder %s already in cache (%s)"
             % (model_name, cache_dir or "$HF_HOME = " + os.environ.get("HF_HOME", "?")))
        return True
    except Exception:
        line(WARN, "encoder %s not cached locally" % model_name)
        return False


def check_artifacts(root):
    need = {
        "index/instances.jsonl": "entity index",
        "index/images.jsonl": "scene tiling plan",
        "index/class_meta.json": "classes, domain, sampling weights",
        "knowledge/aircraft_semantic_kb_filtered.json": "livery-filtered knowledge base",
        "knowledge/domain_map.json": "military/civilian map",
    }
    optional = {
        "knowledge/text_features.npz": "Text Features (python -m rst.build_text_features)",
        "index/region_embeddings.npz": "region embeddings (rst.extract_embeddings)",
        "index/hard_table.npy": "hard-negative table (rst.mine_hard_pairs)",
        "knowledge/kb_verification.json": "3.3.2 verdict (rst.verify_knowledge)",
    }
    ok = True
    for rel, what in need.items():
        p = os.path.join(root, rel)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            line(OK, "%s  (%.1f MB) %s" % (rel, os.path.getsize(p) / 1e6, what))
        else:
            line(BAD, "%s MISSING -- %s" % (rel, what))
            ok = False
    for rel, how in optional.items():
        p = os.path.join(root, rel)
        if os.path.exists(p):
            line(OK, "%s present" % rel)
        else:
            line(WARN, "%s absent -- run %s" % (rel, how))
    ds = os.path.join(root, "dataset")
    files = os.listdir(ds) if os.path.isdir(ds) else []
    n_jpg = sum(1 for f in files if f.endswith(".jpg"))
    n_csv = sum(1 for f in files if f.endswith(".csv"))
    # only the .jpg are needed: bbox geometry lives in index/instances.jsonl, so the
    # per-image .csv files the raw dataset ships are redundant
    if n_jpg == 25015:
        line(OK, "dataset/ has %d jpg (complete); %d redundant csv present" % (n_jpg, n_csv))
    else:
        line(BAD, "dataset/ has %d jpg, expected 25015 -- upload incomplete" % n_jpg)
        ok = False
    if not os.path.isdir(os.path.join(root, "crop")):
        line(OK, "crop/ absent -- not needed, regions are cut from dataset/ at train time")
    return ok


def check_dataset(cfg):
    from .config import Config
    from .data import EntityDataset, RSTCollator
    cfg.root = os.path.abspath(cfg.root)
    ds = EntityDataset(cfg, "train", train=True)
    ds.set_epoch(0)
    coll = RSTCollator(ds)
    items = [ds[i] for i in range(4)]
    batch = coll(items)
    scene_key = "scene_tokens" if cfg.use_scene_cache else "scene_tiles"
    line(OK, "collator: region%s scene(%s)%s  entities->scenes %s/%d images"
         % (tuple(batch["region"].shape), scene_key,
            tuple(batch[scene_key].shape),
            tuple(batch["entity_to_scene"].shape), len(batch["images"])))
    assert batch["region"].shape[1] in (3, 4), "unexpected region channels"
    assert batch[scene_key].shape[0] == len(batch["images"])
    return batch


def check_forward(cfg, batch, model_name, cache_dir, do_backward=True):
    import torch
    from .model import RSTModel
    from .losses import RSTLoss
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protos = None
    p = os.path.join(cfg.root, "knowledge", "text_features.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        protos = {"tokens": torch.from_numpy(z["tokens"]),
                  "mask": torch.from_numpy(z["mask"]), "class_of_token": None}
    else:
        cfg.use_text = False
        line(WARN, "no text_features.npz -- forward test runs with the Text branch off")
    model = RSTModel(cfg, protos, attr_dim=(0 if protos is None
                                            else int(np.load(p, allow_pickle=True)["attr_matrix"].shape[1])))
    model.to(device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = model(batch)
    line(OK, "forward: class%s domain%s gate%s"
         % (tuple(out["logits_class"].shape), tuple(out["logits_domain"].shape),
            tuple(out["gate"].shape)))
    if do_backward:
        crit = RSTLoss(cfg).to(device)
        loss, parts = crit(out, batch)
        loss.backward()
        got = sum(1 for q in model.parameters() if q.grad is not None and q.grad.abs().sum() > 0)
        tot = sum(1 for q in model.parameters() if q.requires_grad)
        line(OK if got > tot * 0.5 else WARN,
             "backward: loss=%.3f  %d/%d param tensors received gradient" % (loss.item(), got, tot))
        if torch.cuda.is_available():
            line(OK, "peak GPU memory for one step: %.2f GB"
                 % (torch.cuda.max_memory_allocated() / 1e9))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="google/siglip2-base-patch16-384")
    ap.add_argument("--cache-dir", default="",
                    help="defaults to $HF_HOME set by env.sh; only set this to override")
    ap.add_argument("--download-model", action="store_true")
    ap.add_argument("--skip-forward", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)
    # Passing cache_dir=<HF_HOME> is wrong: transformers looks for models--*
    # directly under cache_dir, while HF_HOME's real layout is <HF_HOME>/hub/.
    # Supplying it therefore misses the cache and silently re-downloads 1.5 GB,
    # so leave it unset and let HF_HOME (exported by env.sh) govern.
    cache = os.path.join(root, args.cache_dir) if args.cache_dir else None
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
    elif not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.path.join(root, ".cache", "hf")
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    print("=" * 78)
    print("RST preflight -- %s" % root)
    print("=" * 78)
    torch = check_torch()
    check_network(args.model)
    if torch is None:
        sys.exit(1)

    present = check_model_present(args.model, cache)
    if args.download_model or not present:
        try:
            from transformers import AutoModel, AutoTokenizer
            print("loading %s (this downloads weights if needed)..." % args.model)
            kw = {} if cache is None else {"cache_dir": cache}
            AutoModel.from_pretrained(args.model, **kw)
            AutoTokenizer.from_pretrained(args.model, **kw)
            line(OK, "encoder + tokenizer load")
            present = True
        except Exception as exc:
            line(BAD, "encoder failed to load: %r" % (exc,))
            present = False

    artifacts_ok = check_artifacts(root)
    if artifacts_ok and present and not args.skip_forward:
        from .config import Config
        cfg = Config()
        cfg.root = root
        cfg.vision_model = args.model
        # The live path encodes every tile inside the autograd graph, which is the
        # thing an 8 GB card cannot afford -- so test whatever this machine will
        # actually train on, and shrink the tile count if there is no cache yet.
        from .scene_cache import SceneCache
        cache_bin = os.path.join(root, cfg.scene_cache_dir, "scene_tokens.bin")
        cache_idx = os.path.join(root, cfg.scene_cache_dir, "scene_index.npz")
        if SceneCache.exists(root, cfg.scene_cache_dir) and os.path.exists(cache_idx):
            cfg.use_scene_cache = True
            line(OK, "pipeline test uses the scene cache (the configured training path)")
        else:
            cfg.scene_max_tiles = min(cfg.scene_max_tiles, 8)
            line(WARN, "no scene cache -- pipeline test runs the live scene path with "
                       "%d tiles, which is not what an 8 GB card would train on"
                 % cfg.scene_max_tiles)
        try:
            batch = check_dataset(cfg)
            check_forward(cfg, batch, args.model, cache)
        except Exception as exc:
            import traceback
            line(BAD, "pipeline test raised: %r" % (exc,))
            traceback.print_exc()
            artifacts_ok = False

    print("-" * 78)
    print("next steps (this is what run_pipeline.sh does, in this order):")
    print("  1. python -m rst.build_text_features --verdict knowledge/.no_verdict")
    print("  2. python -m rst.extract_embeddings            # frozen, feeds 3.3.3")
    print("  3. python -m rst.mine_hard_pairs               # 3.3.3")
    print("  4. bash tools/build_scene_cache.sh 4           # 3.2.3 / 3.5.4")
    print("  5. bash run_train.sh --name align_full_kb      # 3.5.1-3.5.3, one call")
    print("  6. python -m rst.extract_embeddings --ckpt runs/align_full_kb/stage2.pt")
    print("  7. python -m rst.verify_knowledge              # 3.3.2, post-3.5.2")
    print("3.3.2 deliberately comes AFTER the 3.5.2 alignment: a frozen encoder gives no")
    print("usable text->image signal on these 114 types, so verifying earlier measures noise.")
    print("=" * 78)
    sys.exit(0 if artifacts_ok and present else 1)


if __name__ == "__main__":
    main()
