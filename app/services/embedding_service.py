import asyncio
import logging
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                logger.info("Loading embedding model...")
                _model = SentenceTransformer(
                    "paraphrase-multilingual-mpnet-base-v2",
                    device="cpu",  # tránh meta tensor bug
                )
                # Warm up để torch allocate memory thật ngay lúc này
                _model.encode(["warmup"], normalize_embeddings=True)
                logger.info(f"✅ Model loaded, dim={_model.get_sentence_embedding_dimension()}")
    return _model


def _embed_sync(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return np.array(vector).tolist()


def _embed_batch_sync(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.array(vectors).tolist()


async def embed_query(text: str) -> list[float]:
    return await asyncio.to_thread(_embed_sync, text)


async def embed_document(text: str) -> list[float]:
    return await asyncio.to_thread(_embed_sync, text)


async def embed_batch_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        return await asyncio.to_thread(_embed_batch_sync, texts)
    except Exception as e:
        logger.error(f"Batch embed failed: {e}")
        results = []
        for text in texts:
            try:
                results.append(await embed_document(text))
            except Exception as ex:
                logger.error(f"Single embed failed: {ex}")
                results.append([])
        return results