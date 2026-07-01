from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "")

# Motor client — kết nối 1 lần, dùng suốt vòng đời app
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            MONGO_URI,
            server_api=ServerApi("1"),
            # Timeout hợp lý cho Atlas
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
        )
    return _client


def get_db():
    return get_client()[MONGO_DB_NAME]


# Collections — đặt tên khớp với Node.js model
def get_resumes_col():
    return get_db()["RESUMES"]


def get_parsed_resumes_col():
    return get_db()["PARSED_RESUMES"]


async def close_client():
    global _client
    if _client:
        _client.close()
        _client = None