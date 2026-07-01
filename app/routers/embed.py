"""
POST /embed { resumeId }
─────────────────────────
RAG-style chunking + embedding:
  - Chunk rawText thành nhiều đoạn có overlap
  - Embed từng chunk
  - Lưu mean pooling vector vào Resume.embedding (dùng cho ranking)
  - Lưu từng chunk vector vào ParsedResume.chunks (dùng cho retrieval chi tiết sau này)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from bson import ObjectId
import logging
import numpy as np

from app.db.mongo import get_parsed_resumes_col, get_resumes_col
from app.services.embedding_service import embed_document, embed_batch_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/embed", tags=["embed"])


class EmbedRequest(BaseModel):
    resumeId: str


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Chia text thành chunks theo từ, có overlap.

    Tại sao overlap?
    → Thông tin quan trọng thường nằm ở biên giữa 2 chunk.
    → Overlap đảm bảo không mất context.

    chunk_size=300 words ≈ 400 token — phù hợp text-embedding-004 (limit 2048).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap  # slide với overlap

    return chunks


def mean_pool(vectors: list[list[float]]) -> list[float]:
    """
    Tính mean vector từ nhiều chunk vectors.
    Đây là cách đơn giản nhất để represent toàn bộ document bằng 1 vector.
    Alternatives: max pooling, weighted by position — nhưng mean đủ tốt cho CV ranking.
    """
    arr = np.array(vectors, dtype=np.float32)
    return arr.mean(axis=0).tolist()


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("")
async def embed_cv(body: EmbedRequest):
    """
    Flow:
    1. Load rawText + shortSummary từ ParsedResume
    2. Chunk rawText (có overlap)
    3. Prepend shortSummary vào chunk đầu để boost signal
    4. Embed batch tất cả chunks
    5. Mean pool → lưu vào Resume.embedding (dùng cho cosine ranking)
    6. Lưu chunk texts vào ParsedResume.chunkTexts (dùng cho RAG retrieval sau)
    """
    try:
        oid = ObjectId(body.resumeId)
    except Exception:
        raise HTTPException(status_code=400, detail="resumeId không hợp lệ")

    # Load ParsedResume
    parsed_col = get_parsed_resumes_col()
    parsed = await parsed_col.find_one(
        {"resumeId": oid},
        {"rawText": 1, "shortSummary": 1, "summary": 1},
    )

    if not parsed:
        raise HTTPException(
            status_code=404,
            detail=f"ParsedResume không tồn tại cho resumeId: {body.resumeId}",
        )

    raw_text = parsed.get("rawText", "")
    short_summary = parsed.get("shortSummary") or parsed.get("summary", "")

    if not raw_text and not short_summary:
        raise HTTPException(status_code=422, detail="CV không có text để embed")

    # Chunk rawText
    chunks = chunk_text(raw_text, chunk_size=300, overlap=50)

    if not chunks:
        # Fallback: dùng shortSummary nếu rawText rỗng
        chunks = [short_summary]

    # Prepend shortSummary vào chunk đầu tiên để boost signal tổng quát
    # shortSummary đã chứa: tên, role, skills, experience — rất quan trọng
    if short_summary and chunks[0] != short_summary:
        chunks[0] = f"{short_summary}\n\n{chunks[0]}"

    logger.info(f"resumeId={body.resumeId}: {len(chunks)} chunks")

    # Embed batch tất cả chunks
    chunk_vectors = await embed_batch_documents(chunks)

    # Lọc chunk bị lỗi (vector rỗng)
    valid_pairs = [
        (chunk, vec)
        for chunk, vec in zip(chunks, chunk_vectors)
        if vec and len(vec) > 0
    ]

    if not valid_pairs:
        raise HTTPException(status_code=500, detail="Embedding thất bại cho tất cả chunks")

    valid_chunks, valid_vecs = zip(*valid_pairs)

    # Mean pooling → 1 vector đại diện cho toàn bộ CV
    doc_vector = mean_pool(list(valid_vecs))

    # Lưu mean vector vào Resume.embedding (dùng cho ranking)
    resumes_col = get_resumes_col()
    await resumes_col.update_one(
        {"_id": oid},
        {"$set": {"jdEmbedding": doc_vector}},
    )

    # Lưu chunk texts vào ParsedResume để dùng cho RAG retrieval sau
    # (khi cần tìm đoạn cụ thể trong CV thay vì rank tổng thể)
    await parsed_col.update_one(
        {"resumeId": oid},
        {"$set": {"chunkTexts": list(valid_chunks)}},
    )

    logger.info(
        f"✅ Embedded resumeId={body.resumeId} | "
        f"chunks={len(valid_chunks)} | dim={len(doc_vector)}"
    )

    return {
        "success": True,
        "resumeId": body.resumeId,
        "chunks": len(valid_chunks),
        "embeddingDim": len(doc_vector),
    }