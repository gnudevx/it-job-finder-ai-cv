"""
Similarity Service
──────────────────
Cosine similarity dùng numpy — nhanh, không cần vector DB ở scale vài nghìn CV.

Khi nào cần upgrade lên vector DB (Qdrant/Weaviate)?
→ > 50k CV hoặc cần filter phức tạp (location, experience range, v.v.)
"""

import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Trả về float trong khoảng [0, 1]."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query_vec: list[float], doc_vecs: list[list[float]]) -> list[float]:
    """
    Tính similarity của 1 query với N documents cùng lúc.
    Dùng matrix ops → nhanh hơn loop đơn lẻ.
    """
    q = np.array(query_vec, dtype=np.float32)
    D = np.array(doc_vecs, dtype=np.float32)          # shape: (N, dim)

    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return [0.0] * len(doc_vecs)

    q_unit = q / q_norm
    d_norms = np.linalg.norm(D, axis=1, keepdims=True)  # shape: (N, 1)

    # Tránh chia cho 0
    d_norms = np.where(d_norms == 0, 1e-10, d_norms)
    D_unit = D / d_norms                                # shape: (N, dim)

    scores = D_unit @ q_unit                            # shape: (N,)
    return scores.tolist()


def to_percentage(score: float) -> float:
    """Convert [-1,1] cosine score → [0,100] dễ đọc."""
    return round(max(0.0, score) * 100, 2)