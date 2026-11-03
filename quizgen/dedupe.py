"""Step 6b - Remove near-duplicate questions by similarity.

Uses a multilingual embedding model (sentence-transformers) when installed; otherwise
falls back to character n-gram TF-IDF (works immediately, no GPU, but only catches
surface-level duplicates, not paraphrases).
A question whose cosine similarity to an already KEPT question exceeds the threshold
is tagged 'duplicate'.
"""
from __future__ import annotations

import numpy as np

from .generate import GeneratedQuestion

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def embed(texts: list[str], use_st: bool = True) -> np.ndarray:
    if use_st:
        try:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(EMBED_MODEL)
            return np.asarray(m.encode(texts, normalize_embeddings=True))
        except ImportError:
            pass
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    X = X.toarray()
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def mark_duplicates(questions: list[GeneratedQuestion], threshold: float = 0.9,
                    use_st: bool = True) -> list[GeneratedQuestion]:
    cands = [q for q in questions if q.ok]
    if len(cands) < 2:
        return questions
    E = embed([q.mcq.question + " " + q.mcq.answer for q in cands], use_st=use_st)
    kept: list[int] = []
    for i, q in enumerate(cands):
        if kept and float(np.max(E[kept] @ E[i])) > threshold:
            q.rejected.append("duplicate")
        else:
            kept.append(i)
    return questions
