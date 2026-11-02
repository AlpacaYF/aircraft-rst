"""Memory-mapped reader for the scene feature cache written by
``rst.cache_scene_features``.

Returns pooled patch tokens *before* the trainable projection, so `scene_proj`
and `tile_pos` keep learning; only the vision tower itself is frozen and skipped.
"""

import os

import numpy as np


class SceneCache:
    def __init__(self, bin_path, index_path):
        if not (os.path.exists(bin_path) and os.path.exists(index_path)):
            raise FileNotFoundError(
                "scene cache missing -- run `python -m rst.cache_scene_features` first")
        idx = np.load(index_path, allow_pickle=True)
        self.dim = int(idx["dim"][0])
        self.tile_tokens = int(idx["tile_tokens"][0])
        # absent from caches written before the 3.2.3 two-level budget
        self.near_pool = int(idx["near_pool"][0]) if "near_pool" in idx.files else 0
        self.far_pool = int(idx["far_pool"][0]) if "far_pool" in idx.files else 0
        self.fingerprint = (str(idx["fingerprint"][0])
                            if "fingerprint" in idx.files else "")
        self.row_ptr = idx["row_ptr"].astype(np.int64)
        self.n_tiles = idx["n_tiles"].astype(np.int32)
        self.meta = idx["meta"].astype(np.float32)
        images = [str(k) for k in idx["images"]]
        self.pos = {k: i for i, k in enumerate(images)}
        # cumulative meta row per image, precomputed so lookups stay O(1)
        self.meta_ptr = np.concatenate([[0], np.cumsum(self.n_tiles)]).astype(np.int64)
        self.total_tokens = int(self.row_ptr[-1])
        self.bin_path = bin_path
        self._mm = None
        self.model_name = str(idx["model"][0])

    @staticmethod
    def exists(root, subdir="cache"):
        return os.path.exists(os.path.join(root, subdir, "scene_tokens.bin"))

    @property
    def mm(self):
        # opened lazily and once per process: forked DataLoader workers would
        # otherwise each inherit a stale mapping
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=np.float16, mode="r",
                                 shape=(self.total_tokens, self.dim))
        return self._mm

    def __contains__(self, key):
        return key in self.pos

    def __len__(self):
        return len(self.pos)

    def get(self, key):
        """-> (tokens [n_tiles*tile_tokens, dim] float32, meta [n_tiles, 4]) or None."""
        i = self.pos.get(key)
        if i is None:
            return None
        start, end = int(self.row_ptr[i]), int(self.row_ptr[i + 1])
        n = int(self.n_tiles[i])
        if n <= 0 or end <= start:
            return None
        tokens = np.asarray(self.mm[start:end], dtype=np.float32)
        meta = self.meta[self.meta_ptr[i]:self.meta_ptr[i] + n]
        return tokens, meta
