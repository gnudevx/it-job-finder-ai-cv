from pydantic import BaseModel, Field
from typing import Optional


# ── Request ──────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    jobDescription: str = Field(..., min_length=20, description="Nội dung JD / yêu cầu tuyển dụng")
    topK: int = Field(default=5, ge=1, le=20, description="Số CV trả về")
    employerId: Optional[str] = Field(default=None, description="ID nhà tuyển dụng để chỉ xét các ứng viên đã apply vào các job của họ")


# ── Internal ─────────────────────────────────────────────────────────────────

class RankedCV(BaseModel):
    resumeId: str
    candidateId: Optional[str] = None
    fullName: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    title: Optional[str] = None          # detectedRole
    skills: list[str] = []
    experienceYears: Optional[float] = None
    summary: Optional[str] = None
    shortSummary: Optional[str] = None
    matchScore: float                    # 0-100
    matchReason: Optional[str] = None
    avatar: Optional[str] = None
    fileType: Optional[str] = None
    fileName: Optional[str] = None


# ── Response ──────────────────────────────────────────────────────────────────

class RecommendResponse(BaseModel):
    success: bool = True
    total: int
    results: list[RankedCV]