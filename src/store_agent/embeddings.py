"""Text embeddings for semantic product search.

The default runs a small model locally (no API key, data never leaves the
machine), which suits on-prem and edge deployments. `FakeEmbedder` is a
deterministic bag-of-words hash used by the tests.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

DIM = 384


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents (product texts)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Retrieval models like bge embed queries differently from documents."""


# Multilingual, so Greek queries ("ψάρι για τη σχάρα") work as well as English.
# Chosen by the search evals: 20/20 cases against 17/20 for BAAI/bge-small-en-v1.5
# (run `uv run store-agent-evals`). Still 384 dimensions, so the schema is unchanged.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # heavy import, only when used

        self._model = TextEmbedding(model)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_embedder(kind: str) -> Embedder:
    if kind == "fastembed":
        return FastEmbedder()
    if kind == "fake":
        return FakeEmbedder()
    raise ValueError(f"unknown embedder {kind!r} (expected 'fastembed' or 'fake')")


def to_pgvector(v: Sequence[float]) -> str:
    """Format a vector as a pgvector literal, used with a `%s::vector` cast."""
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
