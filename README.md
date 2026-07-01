# IT Job Finder AI — CV Recommendation AI Service

Microservice Python (FastAPI) chịu trách nhiệm **gợi ý CV phù hợp nhất với một JD (mô tả công việc)**:

- Embedding văn bản chạy **local trên chính server** bằng `sentence-transformers` (không gọi API ngoài, không tốn phí).
- Cosine similarity bằng numpy để rank CV theo JD.
- Gemini (`gemini-2.5-flash`) chỉ dùng để **sinh lời giải thích (matchReason)** cho từng CV, có fallback rule-based khi Gemini lỗi/hết quota.

Service này đóng vai trò `ai-service` trong một hệ thống lớn hơn gồm Frontend + Backend Node.js + MongoDB (dùng chung 2 collection `RESUMES` và `PARSED_RESUMES`).

## Mục lục

- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Luồng xử lý](#luồng-xử-lý)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt & chạy local](#cài-đặt--chạy-local)
- [Biến môi trường](#biến-môi-trường)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [API](#api)
- [Lưu ý khi deploy (Render, v.v.)](#lưu-ý-khi-deploy-render-vv)

## Kiến trúc tổng quan

Service này **không** đứng độc lập — nó là một microservice AI được Backend Node.js gọi qua HTTP, và cả hai cùng đọc/ghi chung một MongoDB Atlas (collection `RESUMES` và `PARSED_RESUMES`, đặt tên khớp với model Node.js).

```
Frontend
   │  JD input / CV upload
   ▼
Node.js Backend
   │  POST http://ai-service:8000/recommend
   │  body: { jobDescription, topK? }
   ▼
Python FastAPI (ai-service)  ← repo này
   ├─ load CV (PARSED_RESUMES) từ MongoDB
   ├─ embed JD + embed CV (local, sentence-transformers, cache vào RESUMES.jdEmbedding)
   ├─ rank bằng cosine similarity (numpy)
   └─ sinh matchReason cho top K bằng Gemini (fallback rule-based nếu lỗi/hết quota)
   ▲
   │  { success, total, results: [{ resumeId, matchScore, matchReason, skills, ... }] }
Node.js Backend
   │  forward kết quả
   ▼
Frontend
```

Luồng lưu embedding khi có CV mới:

```
CV upload → Node.js parse text → PARSED_RESUMES.rawText
   → POST /embed { resumeId }
   → chunk rawText (300 từ/chunk, overlap 50 từ)
   → embed từng chunk (local, sentence-transformers)
   → mean-pooling → lưu vào RESUMES.jdEmbedding
   → lưu chunkTexts vào PARSED_RESUMES (dự phòng cho RAG retrieval sau này, hiện chưa dùng)
```

## Cấu trúc thư mục

```
it-job-finder-ai-cv/
├── app/
│   ├── main.py                     # FastAPI entrypoint, CORS, lifespan (ping Mongo + load model)
│   ├── routers/
│   │   ├── recommend.py            # POST /recommend
│   │   └── embed.py                # POST /embed — chunk + embed 1 CV, lưu vào Mongo
│   ├── services/
│   │   ├── embedding_service.py    # Load & chạy sentence-transformers (local, CPU)
│   │   ├── similarity_service.py   # Cosine similarity hàng loạt (numpy)
│   │   ├── reason_service.py       # Gọi Gemini sinh matchReason + fallback rule-based
│   │   └── ranker_service.py       # Điều phối pipeline: load CV → embed/cache → rank → reason
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response schemas
│   └── db/
│       └── mongo.py                # Motor (async MongoDB client), collection RESUMES / PARSED_RESUMES
├── test.py                         # Script thử nghiệm Gemini API (không phải một phần của service)
├── requirements.txt
├── package.json / package-lock.json
├── .gitignore
└── README.md
```

> `test.py` và `package.json` hiện không liên quan tới logic chính của service (script thử nghiệm và dependency frontend React không dùng ở đây) — cân nhắc dọn khỏi repo.

## Luồng xử lý

**`POST /embed`** (gọi khi có CV mới, thường do Node.js trigger sau khi parse xong):

1. Load `rawText` (+ `shortSummary`) của CV từ `PARSED_RESUMES`.
2. Chia `rawText` thành các chunk 300 từ, overlap 50 từ (tránh mất ngữ cảnh ở ranh giới đoạn).
3. Gắn `shortSummary` vào chunk đầu tiên để tăng tín hiệu tổng quát (tên, role, skill, kinh nghiệm).
4. Embed từng chunk bằng model local (`embedding_service.py`), sau đó **mean-pooling** thành 1 vector đại diện.
5. Lưu vector vào `RESUMES.jdEmbedding`, lưu các chunk text vào `PARSED_RESUMES.chunkTexts`.

**`POST /recommend`** (gọi khi cần tìm CV phù hợp với 1 JD):

1. Embed job description bằng cùng model local.
2. Load các CV có `shortSummary` trong `PARSED_RESUMES`; CV nào chưa có `jdEmbedding` cache sẽ được embed on-the-fly và lưu lại.
3. `similarity_service.py` tính cosine similarity hàng loạt (numpy) giữa JD và toàn bộ CV.
4. Sort lấy top K, quy điểm về thang 0–100.
5. `reason_service.py` gọi Gemini (`gemini-2.5-flash`) sinh 2 câu nhận xét tiếng Việt cho từng CV trong top K (giới hạn 1 request/lần + delay 4s để tránh vượt free tier 15 RPM); nếu Gemini lỗi hoặc hết quota, dùng fallback rule-based nêu kỹ năng trùng khớp.
6. Trả kết quả về Node.js, Node.js forward về Frontend.

## Yêu cầu hệ thống

- Python 3.10+
- MongoDB Atlas (hoặc MongoDB instance dùng chung với Backend Node.js), collection `RESUMES` và `PARSED_RESUMES`
- API key Gemini (Google AI Studio) — chỉ dùng để sinh `matchReason`, không dùng để embedding
- **≥ 1–2GB RAM khả dụng** cho tiến trình — cần để load `torch` + model `sentence-transformers` vào RAM (xem thêm mục [Lưu ý khi deploy](#lưu-ý-khi-deploy-render-vv))
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

| Biến               | Mô tả                                                                            |
| ------------------ | -------------------------------------------------------------------------------- |
| `MONGO_URI`        | Connection string MongoDB Atlas (dùng chung với Backend Node.js)                 |
| `MONGO_DB_NAME`    | Tên database chứa `RESUMES` và `PARSED_RESUMES`                                  |
| `GEMINI_API_KEY`   | API key cho Gemini — chỉ dùng để sinh `matchReason`, **không** dùng để embedding |
| `NODE_BACKEND_URL` | Origin của Backend Node.js, dùng cho CORS (mặc định `http://backend:5000`)       |

> Model embedding (`sentence-transformers`) chạy hoàn toàn local, không cần API key riêng.

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

Nhận JD, trả về danh sách CV đã rank + giải thích.

**Request body:**

```json
{
  "jobDescription": "string (bắt buộc, tối thiểu 20 ký tự)",
  "topK": 5
}
```

**Response:**

```json
{
  "success": true,
  "total": 5,
  "results": [
    {
      "resumeId": "string",
      "candidateId": "string",
      "title": "string",
      "skills": ["python", "sql"],
      "experienceYears": 3,
      "summary": "string",
      "shortSummary": "string",
      "matchScore": 87.34,
      "matchReason": "string (2 câu tiếng Việt)",
      "fileType": "pdf",
      "fileName": "string"
    }
  ]
}
```

### `POST /embed`

Embed 1 CV cụ thể và lưu vector vào MongoDB — thường được Node.js gọi ngay sau khi parse xong CV mới.

**Request body:**

```json
{
  "resumeId": "string (ObjectId hợp lệ)"
}
```

**Response:**

```json
{
  "success": true,
  "resumeId": "string",
  "chunks": 4,
  "embeddingDim": 768
}
```

### `GET /health`

Health check đơn giản, trả `{ "status": "ok", "service": "ai-service" }`.

## Lưu ý khi deploy (Render, v.v.)

Model embedding chạy **ngay trên server đang host service** (không gọi API ngoài), nên khi deploy lên các nền tảng như Render cần lưu ý:

- **RAM**: `torch` + model transformer cần khoảng 1–2GB RAM để load và chạy — free tier của nhiều nền tảng (thường 512MB) có thể không đủ, dẫn đến OOM khi khởi động.
- **Filesystem ephemeral**: nếu không gắn persistent disk, model weights (tải từ HuggingFace Hub) sẽ bị xoá mỗi lần container restart/deploy lại, khiến lần khởi động sau phải tải lại từ đầu. Nên cân nhắc **bake model vào Docker image lúc build** thay vì tải lúc runtime.
- **Cold start**: model được load trong `lifespan` startup của FastAPI trước khi app sẵn sàng nhận request — nếu health check gọi quá sớm trong lúc model đang tải, có thể bị đánh dấu unhealthy.
- **CPU only**: `device="cpu"` được ép cứng trong code, không cần GPU nhưng suy luận sẽ chậm hơn so với GPU.

---

> Ghi chú: đây là service AI trong một hệ thống nhiều thành phần (Frontend / Backend Node.js / MongoDB Atlas). Repo Backend và Frontend được quản lý riêng.
