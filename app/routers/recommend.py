from fastapi import APIRouter, HTTPException
from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.ranker_service import rank_cvs
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
async def recommend(body: RecommendRequest):
    """
    Nhận JD → trả về danh sách CV ranked theo độ phù hợp.
    
    - **jobDescription**: Nội dung JD hoặc yêu cầu tuyển dụng (tối thiểu 20 ký tự)
    - **topK**: Số CV trả về (mặc định 5, tối đa 20)
    """
    try:
        results = await rank_cvs(
            job_description=body.jobDescription,
            top_k=body.topK,
        )
        return RecommendResponse(
            success=True,
            total=len(results),
            results=results,
        )
    except Exception as e:
        logger.exception(f"Recommend failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))