import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.db.mongo import get_client, close_client
from app.routers import recommend, embed
import asyncio
from app.services.embedding_service import get_model
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — ping MongoDB để fail fast nếu connection sai
    try:
        client = get_client()
        await client.admin.command("ping")
        await asyncio.to_thread(get_model)
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise

    yield

    # Shutdown
    await close_client()
    logger.info("MongoDB connection closed")


app = FastAPI(
    title="CV Recommendation AI Service",
    description="Semantic CV ranking dùng Gemini Embedding + Cosine Similarity",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — cho phép Node.js backend gọi vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://localhost:3000",
        os.getenv("NODE_BACKEND_URL", "http://backend:5000"),
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Routers
app.include_router(recommend.router)
app.include_router(embed.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-service"}