"""
PaPie char-embedding cache, with a fix for its eviction bug.

PaPie 0.6.0 caches the char BiRNN encoding of each word form. On CPU this speeds up tagging
~2.5x (word forms repeat a lot); on GPU it is slower and uses more VRAM, so it is CPU-only.
Its `_encode_words_cached` evicts old entries before re-reading the cache hits of the current
batch, which raises KeyError once the cache is full. The fixed copy below evicts afterwards.
"""

import types
from typing import Iterable

import torch


def _encode_words_cached_fixed(self, char, nchars):
    """Copy of pie 0.6.0 RNNEmbedding._encode_words_cached, with eviction moved after reassembly."""
    cache = self._cache
    nchars_list = nchars.tolist()
    keys = [tuple(char[:n, j].tolist()) for j, n in enumerate(nchars_list)]

    miss_cols, seen = [], set()
    for j, key in enumerate(keys):
        if key not in cache and key not in seen:
            seen.add(key)
            miss_cols.append(j)

    if miss_cols:
        idx = torch.tensor(miss_cols, dtype=torch.int64, device=char.device)
        emb_miss, outs_miss = self._encode_words(char[:, idx], nchars[idx])
        for i, j in enumerate(miss_cols):
            n = nchars_list[j]
            cache[keys[j]] = (emb_miss[i].clone(), outs_miss[:n, i].clone())

    sample = next(iter(cache.values()))[0]
    out_len = max(nchars_list)
    emb = torch.zeros((len(keys), self.embedding_dim), dtype=sample.dtype, device=sample.device)
    outs = torch.zeros((out_len, len(keys), self.embedding_dim), dtype=sample.dtype, device=sample.device)
    for j, key in enumerate(keys):
        word_emb, word_outs = cache[key]
        cache.move_to_end(key)
        emb[j] = word_emb
        outs[: word_outs.size(0), j] = word_outs

    # Evict only once every word of the batch has been read back
    while len(cache) > self._cache_maxsize:
        cache.popitem(last=False)

    return emb, outs


def iter_char_embeddings(tagger) -> Iterable:
    """Char-embedding modules (RNNEmbedding) of every sub-model of a pie tagger."""
    for model, _tasks in tagger.models:
        cemb = getattr(model, "cemb", None)
        if cemb is not None and hasattr(cemb, "enable_cache"):
            yield cemb


def enable_char_embedding_cache(char_embeddings: Iterable, max_entries: int) -> None:
    """Enable the fixed LRU cache on each char-embedding module (max_entries=0 disables it)."""
    for cemb in char_embeddings:
        if max_entries <= 0:
            cemb.disable_cache()
            continue
        cemb.enable_cache(max_entries)
        cemb._encode_words_cached = types.MethodType(_encode_words_cached_fixed, cemb)
