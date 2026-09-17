"""Tests for the PaPie char-embedding cache workaround (real PaPie code, tiny model)."""

import pytest
import torch
from pie.models.embedding import RNNEmbedding

from app.core.papie_cache import enable_char_embedding_cache


def _batches():
    """Word batches (char ids, one column per word) that make the cache evict a word re-read in the same batch."""
    words = [(1, 2), (3, 4, 5), (6, 7), (1, 2), (8, 9), (3, 4, 5), (2, 6, 7), (6, 7), (9, 1), (1, 2)]
    for start in range(0, len(words) - 2, 2):
        batch = words[start : start + 4]
        length = max(len(w) for w in batch)
        char = torch.tensor([[w[i] if i < len(w) else 0 for w in batch] for i in range(length)])
        yield char, torch.tensor([len(w) for w in batch])


@pytest.fixture
def embedding():
    torch.manual_seed(0)
    module = RNNEmbedding(num_embeddings=12, embedding_dim=4, padding_idx=0)
    module.eval()
    return module


def test_papie_cache_bug_is_reproduced(embedding):
    """Guard: PaPie 0.6.0's own cache raises KeyError once full (if this fails, the upstream bug is fixed)."""
    embedding.enable_cache(3)
    with pytest.raises(KeyError), torch.no_grad():
        for char, nchars in _batches():
            embedding._encode_words_cached(char, nchars)


def test_fixed_cache_never_raises_and_matches_uncached_output(embedding):
    enable_char_embedding_cache([embedding], max_entries=3)
    with torch.no_grad():
        for char, nchars in _batches():
            cached_emb, cached_outs = embedding._encode_words_cached(char, nchars)
            expected_emb, expected_outs = embedding._encode_words(char, nchars)
            assert torch.allclose(cached_emb, expected_emb, atol=1e-6)
            assert torch.allclose(cached_outs, expected_outs, atol=1e-6)
    assert len(embedding._cache) <= 3


def test_zero_entries_disables_cache(embedding):
    enable_char_embedding_cache([embedding], max_entries=0)
    assert embedding._cache is None
