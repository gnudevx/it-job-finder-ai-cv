import asyncio
import logging
import os

from google import genai
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

_PROMPT_TEMPLATE = """
Bạn là chuyên gia tuyển dụng. Viết ĐÚNG 5 câu tiếng Việt nhận xét ứng viên.

YÊU CẦU TUYỂN DỤNG: {job_description}

ỨNG VIÊN:
- Kinh nghiệm: {experience_years} năm
- Kỹ năng: {skills}
- Điểm phù hợp: {match_score}%

QUY TẮC BẮT BUỘC:
- Nếu {match_score} >= 60: câu 1 nêu kỹ năng CỤ THỂ khớp với JD, câu 2 nêu điểm còn thiếu
- Nếu {match_score} < 60: câu 1 nêu kỹ năng CỤ THỂ còn thiếu so với JD, câu 2 nêu điểm có thể tận dụng
- PHẢI so sánh trực tiếp kỹ năng ứng viên với yêu cầu JD
- PHẢI gọi tên kỹ năng cụ thể (python, spark, hadoop,...)
- TUYỆT ĐỐI KHÔNG viết chung chung kiểu "chưa phù hợp", "không phù hợp", "ứng viên chưa đáp ứng"

Ví dụ tốt (score 63%): "Ứng viên có python và sql server phù hợp yêu cầu, đồng thời có kinh nghiệm hadoop/Big Data là lợi thế. Tuy nhiên thiếu 3 năm kinh nghiệm theo yêu cầu và chưa có spark, kafka."
Ví dụ xấu: "Ứng viên chưa phù hợp với vị trí này."
""".strip()


def _make_fallback(
    experience_years: float,
    skills: list[str],
    detected_role: str,
    job_description: str,
    match_score: float,
) -> str:
    matched = [s for s in skills if s.lower() in job_description.lower()]
    score_comment = (
        "tốt" if match_score >= 70
        else "trung bình" if match_score >= 50
        else "thấp"
    )
    if matched:
        return (
            f"Độ phù hợp {score_comment} ({match_score}%): ứng viên có {experience_years} năm kinh nghiệm "
            f"với các kỹ năng liên quan: {', '.join(matched[:3])}."
        )
    return (
        f"Độ phù hợp {score_comment} ({match_score}%): ứng viên có {experience_years} năm "
        f"kinh nghiệm trong lĩnh vực {detected_role or 'liên quan'} nhưng thiếu kỹ năng chuyên biệt theo yêu cầu."
    )


def _is_quota_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = getattr(exc, "status_code", None)
        if code in (429, 503):
            return True
        if "quota" in str(exc).lower() or "exhausted" in str(exc).lower():
            return True
    return False


# ✅ Fix: chỉ retry lỗi tạm thời (5xx), không retry quota (429)
def _is_retryable(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    if _is_quota_error(exc):
        return False
    code = getattr(exc, "status_code", None)
    return code in (500, 502, 503) if code else False


@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=True,
)
def _generate_sync(prompt: str) -> str:
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "max_output_tokens": 1024,
            "temperature": 0.4,
            "thinking_config": {"thinking_budget": 0},
        },
    )
    return (response.text or "").strip()


async def generate_reason(
    job_description: str,
    detected_role: str,
    experience_years: float,
    skills: list[str],
    summary: str,
    match_score: float = 0.0,  # ✅ thêm param này
) -> str:
    # ✅ Clean summary — tránh summary rỗng kiểu "Developer with experience in ,"
    jd_words = set(job_description.lower().split())
    tech_keywords = {"python", "sql", "spark", "hadoop", "kafka", "airflow", 
                     "bigquery", "dbt", "etl", "pipeline", "tableau", "powerbi",
                     "java", "scala", "aws", "gcp", "azure", "docker", "kubernetes"}
    required_skills = [w for w in tech_keywords if w in jd_words]

    clean_summary = summary.strip() if summary else ""
    if not clean_summary or "experience in ," in clean_summary:
        clean_summary = ""


    prompt = _PROMPT_TEMPLATE.format(
        job_description=job_description[:600],
        detected_role=detected_role or "Chưa xác định",
        experience_years=experience_years or 0,
        skills=", ".join(skills[:15]) if skills else "Không có thông tin",
        match_score=round(match_score, 1),
    )
    try:
        return await asyncio.to_thread(_generate_sync, prompt)
    except Exception as e:
        if _is_quota_error(e):
            logger.warning("Gemini quota exhausted, dùng fallback reason")
        else:
            logger.warning(f"Reason generation failed: {e}")
        return _make_fallback(experience_years, skills, detected_role, job_description, match_score)


async def generate_reasons_batch(
    job_description: str,
    candidates: list[dict],
) -> list[str]:
    """15 RPM free tier → semaphore(1) + delay 4s"""
    semaphore = asyncio.Semaphore(1)

    async def _with_sem(c: dict) -> str:
        async with semaphore:
            logger.info(f"Candidate keys: {list(c.keys())}")  # ← thêm đây
            logger.info(f"Candidate data: {c}")               # ← thêm đây
            result = await generate_reason(
                job_description=job_description,
                detected_role=c.get("detectedRole", ""),
                experience_years=c.get("totalYearsExperience", 0),
                skills=c.get("skills", []),
                summary=c.get("shortSummary") or c.get("summary", ""),
                match_score=c.get("matchScore", 0.0),  # ✅ truyền score vào
            )
            await asyncio.sleep(4)
            return result

    return await asyncio.gather(*[_with_sem(c) for c in candidates])