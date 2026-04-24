import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class CVIndex:
    def __init__(self):
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.index: faiss.Index | None = None
        self.chunks: List[str] = []
        self.metadata: List[dict] = []

    def build(self, documents: List[Tuple[str, str]]):
        """documents: list of (candidate_name, full_text)"""
        self.chunks = []
        self.metadata = []

        for name, text in documents:
            for chunk in _split(text):
                self.chunks.append(chunk)
                self.metadata.append({"name": name})

        embeddings = self.encoder.encode(self.chunks, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self.index is None:
            return []
        vec = self.encoder.encode([query])
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        scores, indices = self.index.search(vec.astype("float32"), top_k)
        return [
            {"chunk": self.chunks[i], "meta": self.metadata[i], "score": float(scores[0][j])}
            for j, i in enumerate(indices[0])
            if i >= 0
        ]


def _split(text: str, chunk_size: int = 300, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks
