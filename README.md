# IT Job Finder AI — AI Matching Service

Microservice Python (FastAPI) chịu trách nhiệm **matching CV ↔ Job Description** bằng AI: sinh embedding, tính độ tương đồng, và giải thích lý do phù hợp (matchReason) bằng Gemini. Service này đóng vai trò `ai-service` trong một hệ thống lớn hơn gồm Frontend + Backend Node.js + MongoDB.

## Mục lục

- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Luồng xử lý](#luồng-xử-lý)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt & chạy local](#cài-đặt--chạy-local)
- [Biến môi trường](#biến-môi-trường)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [API](#api)

## Kiến trúc tổng quan

Service này **không** đứng độc lập — nó là một microservice AI được Backend Node.js gọi qua HTTP, và cả hai cùng đọc/ghi chung một MongoDB Atlas.

```
Frontend
   │  JD input / CV upload
   ▼
Node.js Backend
   │  POST http://ai-service:8000/recommend
   │  body: { jobDescription, jobId? }
   ▼
Python FastAPI (ai-service)  ← repo này
   ├─ đọc thẳng MongoDB (chung DB với Node.js)
   ├─ embed (Gemini) + rank (cosine similarity) + reason (Gemini)
   └─ trả về [{ resumeId, matchScore, matchReason, skills, ... }]
   ▲
   │  response
Node.js Backend
   │  forward kết quả
   ▼
Frontend
```

Luồng lưu embedding khi có CV mới:

```
CV upload → Node.js parse → Python embed → lưu Resume.embedding vào MongoDB Atlas
```

## Cấu trúc thư mục

```
it-job-finder-ai-cv/
├── app/
│   ├── main.py                     # FastAPI entrypoint
│   ├── routers/
│   │   └── recommend.py            # POST /recommend
│   ├── services/
│   │   ├── embedding_service.py    # Gọi Gemini embedding API
│   │   ├── similarity_service.py   # Tính cosine similarity (numpy)
│   │   ├── reason_service.py       # Gemini chat → sinh matchReason
│   │   └── ranker_service.py       # Điều phối pipeline embed → rank → reason
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response schemas
│   └── db/
│       └── mongo.py                # Motor (async MongoDB client)
├── test.py
├── requirements.txt
├── package.json / package-lock.json
├── .gitignore
└── README.md
```

## Luồng xử lý

1. Node.js Backend nhận job description từ Frontend, gọi `POST /recommend` sang service này.
2. `recommend.py` nhận request, chuyển cho `ranker_service.py` điều phối pipeline.
3. `embedding_service.py` gọi Gemini để embed job description.
4. `similarity_service.py` tính cosine similarity giữa embedding job và embedding các CV đã lưu trong MongoDB.
5. `reason_service.py` gọi Gemini chat để sinh giải thích (`matchReason`) cho từng kết quả phù hợp.
6. Trả về danh sách CV được xếp hạng kèm điểm số và lý do, Node.js forward về Frontend.

## Yêu cầu hệ thống

- Python 3.10+
- MongoDB Atlas (hoặc MongoDB instance dùng chung với Backend Node.js)
- API key Gemini (Google AI Studio)
- Docker & Docker Compose (nếu chạy container hóa)

## Cài đặt & chạy local

```bash
# 1. Clone repo
git clone https://github.com/gnudevx/it-job-finder-ai-cv.git
cd it-job-finder-ai-cv

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Tạo file .env (xem mục Biến môi trường bên dưới)
cp .env.example .env   # nếu có, hoặc tự tạo mới

# 5. Chạy service
uvicorn app.main:app --reload --port 8000
```

Service sẽ chạy tại `http://localhost:8000`, Swagger docs tại `http://localhost:8000/docs`.

## Biến môi trường

| Biến             | Mô tả                                                    |
| ---------------- | -------------------------------------------------------- |
| `MONGODB_URI`    | Connection string MongoDB Atlas (dùng chung với Backend) |
| `GEMINI_API_KEY` | API key cho Gemini embedding & chat                      |
| `PORT`           | Port chạy service (mặc định `8000`)                      |

## Chạy bằng Docker

```bash
docker compose up -d --build
```

Sau khi đổi `.env`, restart lại container:

```bash
docker compose up -d --force-recreate ai-service
```

## API

### `POST /recommend`

**Request body:**

```json
{
  "jobDescription": "string (bắt buộc)",
  "jobId": "string (tuỳ chọn)"
}
```

**Response:**

```json
[
  {
    "resumeId": "string",
    "matchScore": 0.87,
    "matchReason": "string",
    "skills": ["string"]
  }
]
```

---

> Ghi chú: đây là service AI trong một hệ thống nhiều thành phần (Frontend / Backend Node.js / MongoDB Atlas). Repo Backend và Frontend được quản lý riêng.
