# it-job-finder-ai-cv
Project it job AI help find employee  
it-job-finder-ai-cv/                          ← repo riêng 
├── app/
│   ├── main.py                      ← FastAPI entrypoint
│   ├── routers/
│   │   └── recommend.py             ← POST /recommend
│   ├── services/
│   │   ├── embedding_service.py     ← Gemini embed API
│   │   ├── similarity_service.py    ← cosine similarity numpy
│   │   ├── reason_service.py        ← Gemini chat → matchReason
│   │   └── ranker_service.py        ← orchestrate pipeline
│   ├── models/
│   │   └── schemas.py               ← Pydantic request/response
│   └── db/
│       └── mongo.py                 ← motor (async MongoDB client)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env


# Inter-service call
Node.js Backend
    │
    │  POST http://ai-service:8000/recommend
    │  { jobDescription, jobId? }
    ▼
Python FastAPI (ai-service)
    ├─ query MongoDB trực tiếp (same DB)
    ├─ embed + rank + reason
    └─ trả [{resumeId, matchScore, matchReason, skills, ...}]
    
Node.js Backend
    └─ forward response về Frontend

flow:
CV upload → Node.js parse → Python embed → lưu Resume.embedding vào MongoDB Atlas
                                                        ↑
Frontend JD input → Node.js → Python FastAPI đọc thẳng MongoDB → rank → trả về Node → Frontend




Đổi .env → docker compose up -d --force-recreate ai-service