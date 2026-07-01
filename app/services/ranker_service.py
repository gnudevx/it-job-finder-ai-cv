"""
CV Ranker Service
─────────────────
Orchestrate toàn bộ pipeline:
1. Load ParsedResumes từ MongoDB
2. Embed CV nếu chưa có (cache vào RESUMES collection)
3. Batch cosine similarity
4. Sort → top K
5. Sinh matchReason song song
"""

import logging
from bson import ObjectId

from app.db.mongo import get_parsed_resumes_col, get_resumes_col
from app.services.embedding_service import embed_query, embed_batch_documents
from app.services.similarity_service import batch_cosine_similarity, to_percentage
from app.services.reason_service import generate_reasons_batch
from app.models.schemas import RankedCV

logger = logging.getLogger(__name__)


async def _load_parsed_resumes() -> list[dict]:
    """Load tất cả ParsedResume có summary."""
    col = get_parsed_resumes_col()
    cursor = col.find(
        {"shortSummary": {"$exists": True, "$ne": ""}},
        {
            "resumeId": 1,
            "candidateId": 1,
            "skills": 1,
            "totalYearsExperience": 1,
            "detectedRole": 1,
            "summary": 1,
            "shortSummary": 1,
        },
    )
    return await cursor.to_list(length=None)


async def _get_resume_meta(resume_id: ObjectId) -> dict:
    """Load metadata từ RESUMES collection (fileName, fileType, embedding cache)."""
    col = get_resumes_col()
    doc = await col.find_one(
        {"_id": resume_id},
        {"fileUrl": 1, "fileName": 1, "fileType": 1, "embedding": 1, "skills": 1},
    )
    return doc or {}


async def _cache_embedding(resume_id: ObjectId, embedding: list[float]):
    """Lưu embedding vào Resume document để tái sử dụng."""
    col = get_resumes_col()
    await col.update_one(
        {"_id": resume_id},
        {"$set": {"jdEmbedding": embedding}},
    )


async def rank_cvs(job_description: str, top_k: int = 5) -> list[RankedCV]:
    # ── 1. Embed JD ──────────────────────────────────────────────────────────
    jd_vec = await embed_query(job_description)

    # ── 2. Load parsed resumes ───────────────────────────────────────────────
    parsed_list = await _load_parsed_resumes()
    if not parsed_list:
        logger.warning("No parsed resumes found in DB")
        return []

    # ── 3. Lấy embedding — cache nếu chưa có ─────────────────────────────────
    resume_metas: dict[str, dict] = {}
    cv_vecs: list[list[float]] = []
    needs_embed_indices: list[int] = []
    needs_embed_texts: list[str] = []

    for i, pr in enumerate(parsed_list):
        resume_id = pr.get("resumeId")
        if not resume_id:
            cv_vecs.append([])
            continue

        meta = await _get_resume_meta(resume_id)
        resume_metas[str(resume_id)] = meta

        cached_emb = meta.get("jdEmbedding", [])
        if cached_emb and len(cached_emb) > 0:
            cv_vecs.append(cached_emb)
        else:
            # Đánh dấu cần embed
            cv_vecs.append([])
            needs_embed_indices.append(i)
            needs_embed_texts.append(
                pr.get("shortSummary") or pr.get("summary", "")
            )

    # Batch embed những CV chưa có cache
    if needs_embed_texts:
        logger.info(f"Embedding {len(needs_embed_texts)} CVs (not cached)...")
        new_embeddings = await embed_batch_documents(needs_embed_texts)

        for idx, emb in zip(needs_embed_indices, new_embeddings):
            cv_vecs[idx] = emb
            resume_id = parsed_list[idx].get("resumeId")
            if resume_id and emb:
                await _cache_embedding(resume_id, emb)

    # ── 4. Lọc CV có vector hợp lệ ───────────────────────────────────────────
    valid_pairs = [
        (i, vec) for i, vec in enumerate(cv_vecs)
        if vec and len(vec) > 0
    ]
    if not valid_pairs:
        return []

    valid_indices, valid_vecs = zip(*valid_pairs)

    # ── 5. Batch cosine similarity ────────────────────────────────────────────
    scores = batch_cosine_similarity(jd_vec, list(valid_vecs))

    # ── 6. Sort → top K ───────────────────────────────────────────────────────
    ranked = sorted(
        zip(valid_indices, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    top_candidates = [parsed_list[i] for i, _ in ranked]
    top_scores = [s for _, s in ranked]

    # ── 7. Sinh matchReason song song ─────────────────────────────────────────
    for pr, score in zip(top_candidates, top_scores):
        pr["matchScore"] = to_percentage(score)

    reasons = await generate_reasons_batch(job_description, top_candidates)

    # ── 8. Build response ─────────────────────────────────────────────────────
    results: list[RankedCV] = []
    for pr, score, reason in zip(top_candidates, top_scores, reasons):
        resume_id = pr.get("resumeId")
        meta = resume_metas.get(str(resume_id), {})

        results.append(
            RankedCV(
                resumeId=str(resume_id),
                candidateId=str(pr.get("candidateId", "")),
                title=pr.get("detectedRole", ""),
                skills=pr.get("skills", []),
                experienceYears=pr.get("totalYearsExperience", 0),
                summary=pr.get("summary", ""),
                shortSummary=pr.get("shortSummary", ""),
                matchScore=to_percentage(score),
                matchReason=reason,
                fileType=meta.get("fileType", "pdf"),
                fileName=meta.get("fileName", ""),
            )
        )

    return results