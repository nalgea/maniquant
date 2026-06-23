"""
06_api_server.py
ManiAgent FastAPI 서버 — 실시간 RAG 쿼리 엔드포인트

엔드포인트:
    POST /v1/chat          — RAG 질의응답 (locale + domain 지원)
    GET  /health           — 서버 상태 확인
    POST /v1/chat/stream   — 스트리밍 응답 (SSE)
    GET  /v1/domains       — 지원 도메인 및 카테고리 목록
    GET  /v1/stats         — Milvus 컬렉션 통계

지원 도메인:
    anti_aging — 항노화·피부 과학 (기본)
    plastic    — 강남 성형외과 뷰티투어

실행:
    uvicorn maniagent.06_api_server:app --host 0.0.0.0 --port 8000 --reload
    또는
    python maniagent/06_api_server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

# ─── 의존성 ────────────────────────────────────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[ERROR] 필수 패키지 없음: {e}")
    print("       pip install fastapi uvicorn[standard] pydantic python-dotenv")
    sys.exit(1)

# ─── 인메모리 메트릭 ──────────────────────────────────────────────────────────
_metrics: dict = {
    "start_time":     time.time(),
    "total_requests": 0,
    "total_errors":   0,
    "endpoints":      defaultdict(lambda: {"count": 0, "errors": 0, "total_ms": 0.0}),
}

import importlib.util as _ilu

_HERE = Path(__file__).parent


def _load_rag_engine():
    spec = _ilu.spec_from_file_location("06_rag_engine", _HERE / "06_rag_engine.py")
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rag_mod = _load_rag_engine()
get_engine      = _rag_mod.get_engine
ManiAgentRAG    = _rag_mod.ManiAgentRAG
ManiAgentResponse = _rag_mod.ManiAgentResponse
_DOMAIN_CONFIGS = _rag_mod._DOMAIN_CONFIGS

load_dotenv(_HERE.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 앱 라이프사이클
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작 시 모든 도메인의 RAG 엔진을 사전 로드합니다."""
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

    for domain in _DOMAIN_CONFIGS:
        logger.info(f"RAG 엔진 초기화 중: domain={domain}")
        try:
            get_engine(
                domain=domain,
                host=host, port=port,
                top_k=top_k, score_threshold=threshold,
            )
            logger.info(f"RAG 엔진 초기화 완료: {domain}")
        except Exception as e:
            logger.warning(f"RAG 엔진 초기화 실패 ({domain}): {e} — 요청 시 재시도")
    yield
    logger.info("서버 종료")


# ════════════════════════════════════════════════════════════════════════════
# FastAPI 앱
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ManiAgent API",
    description=(
        "Maniquant AI 에이전트 — 항노화·뷰티·성형 도메인 RAG 기반 다국어 질의응답 API.\n\n"
        "지원 언어: 한국어(ko), 중국어(zh), 일본어(ja), 영어(en)\n\n"
        "LLM: DeepSeek-V4 (주력) → Qwen-max → Claude (폴백)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# 요청 로깅 + 메트릭 미들웨어
# ════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """모든 HTTP 요청의 응답시간·상태코드를 기록합니다."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    path = request.url.path
    status = response.status_code
    is_error = status >= 400

    _metrics["total_requests"] += 1
    if is_error:
        _metrics["total_errors"] += 1

    ep = _metrics["endpoints"][path]
    ep["count"]    += 1
    ep["total_ms"] += elapsed_ms
    if is_error:
        ep["errors"] += 1

    logger.info(
        f"{request.method} {path} → {status}  {elapsed_ms:.1f}ms"
    )
    return response


# ════════════════════════════════════════════════════════════════════════════
# 요청 / 응답 스키마
# ════════════════════════════════════════════════════════════════════════════

LocaleType  = Literal["ko", "zh", "ja", "en"]
DomainType  = Literal["anti_aging", "plastic"]


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="질문 (한국어, 중국어, 일본어, 영어 가능)",
        examples=["바노바기 성형외과 가격 알려줘", "强南哪家整形医院评价最好？"],
    )
    locale: LocaleType = Field(
        default="ko",
        description="응답 언어: ko(한국어) / zh(중국어) / ja(일본어) / en(영어)",
    )
    domain: DomainType = Field(
        default="anti_aging",
        description="도메인: anti_aging(항노화) / plastic(강남 성형외과 뷰티투어)",
    )
    collection: Optional[str] = Field(
        default=None,
        description="Milvus 컬렉션 이름 (미입력 시 도메인 기본값 사용)",
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="검색할 최대 청크 수",
    )
    score_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="유사도 임계값 (IP 기준, 높을수록 엄격)",
    )
    filter_expr: str = Field(
        default="",
        description="Milvus 필터 표현식 (예: 'year > 2020')",
        examples=["year > 2022", "domain == 'anti_aging'"],
    )


class SourceInfo(BaseModel):
    source_file: str
    page_number: int
    score:       float
    domain:      str
    category:    str
    year:        int
    author:      str


class ChatResponse(BaseModel):
    query:        str
    locale:       str
    answer:       str
    sources:      list[SourceInfo]
    llm_provider: str
    model:        str
    token_usage:  dict
    elapsed_ms:   float


class HealthResponse(BaseModel):
    status:     str
    version:    str
    domains:    list[str]


# ════════════════════════════════════════════════════════════════════════════
# 의존성
# ════════════════════════════════════════════════════════════════════════════

def get_rag_engine(domain: DomainType = "anti_aging") -> ManiAgentRAG:
    try:
        return get_engine(domain=domain)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG 엔진 준비 실패 ({domain}): {e}")


# ════════════════════════════════════════════════════════════════════════════
# 라우터
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """서버 상태 및 버전을 반환합니다."""
    return HealthResponse(
        status="ok",
        version="1.1.0",
        domains=list(_DOMAIN_CONFIGS.keys()),
    )


@app.post("/v1/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    RAG 기반 다국어 질의응답.

    - **query**: 자연어 질문 (한/중/일/영 모두 가능)
    - **locale**: 응답 언어 (`ko` / `zh` / `ja` / `en`)
    - **domain**: `anti_aging` (항노화) 또는 `plastic` (강남 성형외과 뷰티투어)
    - **filter_expr**: Milvus 부울 필터 (예: `year > 2021`)

    **예시 요청 (성형외과):**
    ```json
    {"query": "바노바기 코성형 가격은?", "locale": "zh", "domain": "plastic"}
    ```
    """
    t0 = time.perf_counter()
    engine = get_rag_engine(domain=request.domain)
    engine.top_k           = request.top_k
    engine.score_threshold = request.score_threshold

    try:
        response: ManiAgentResponse = engine.generate(
            query=request.query,
            locale=request.locale,
            filter_expr=request.filter_expr,
        )
    except Exception as exc:
        logger.exception(f"generate() 오류: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ChatResponse(
        query=response.query,
        locale=response.locale,
        answer=response.answer,
        sources=[SourceInfo(**s) for s in response.sources],
        llm_provider=response.llm_provider,
        model=response.model,
        token_usage=response.token_usage,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.post("/v1/chat/stream", tags=["RAG"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    RAG 질의응답 — 서버-전송 이벤트(SSE) 스트리밍 응답.

    현재 구현: 완성된 응답을 청크 단위로 분할하여 스트리밍합니다.
    (Phase 1 이후 LLM 스트리밍 API 적용 예정)
    """
    engine = get_rag_engine(domain=request.domain)
    engine.top_k           = request.top_k
    engine.score_threshold = request.score_threshold

    async def event_generator() -> AsyncIterator[str]:
        try:
            # 비동기 컨텍스트에서 동기 generate 실행
            loop     = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: engine.generate(
                    query=request.query,
                    locale=request.locale,
                    filter_expr=request.filter_expr,
                ),
            )

            # 응답을 50자 단위로 청크 스트리밍
            answer = response.answer
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i : i + chunk_size]
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            # 메타데이터 이벤트
            meta = {
                "event":        "done",
                "llm_provider": response.llm_provider,
                "model":        response.model,
                "sources":      response.sources,
                "token_usage":  response.token_usage,
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/v1/domains", tags=["Metadata"])
async def list_domains() -> dict:
    """지원하는 도메인 및 카테고리 목록을 반환합니다."""
    import anti_aging_config as _aa  # noqa: PLC0415
    import plastic_config     as _pl  # noqa: PLC0415
    return {
        "anti_aging": {
            cat: list(subcats.keys())
            for cat, subcats in _aa.ANTI_AGING_KEYWORDS.items()
        },
        "plastic": {
            cat: list(subcats.keys())
            for cat, subcats in _pl.PLASTIC_KEYWORDS.items()
        },
    }


@app.get("/v1/stats", tags=["Metadata"])
async def collection_stats(domain: DomainType = "anti_aging") -> dict:
    """Milvus 컬렉션 통계를 반환합니다."""
    engine = get_rag_engine(domain=domain)
    try:
        return {
            "domain":       domain,
            "collection":   engine.collection_name,
            "num_entities": engine._collection.num_entities,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics", tags=["System"])
async def get_metrics() -> dict:
    """서버 운영 메트릭을 반환합니다 (요청 수·오류율·엔드포인트별 평균 응답시간)."""
    uptime_s = time.time() - _metrics["start_time"]
    total    = _metrics["total_requests"]
    errors   = _metrics["total_errors"]

    endpoints_summary = {}
    for path, ep in _metrics["endpoints"].items():
        cnt = ep["count"]
        endpoints_summary[path] = {
            "requests":   cnt,
            "errors":     ep["errors"],
            "avg_ms":     round(ep["total_ms"] / cnt, 2) if cnt else 0.0,
            "error_rate": round(ep["errors"] / cnt * 100, 1) if cnt else 0.0,
        }

    return {
        "uptime_seconds":  round(uptime_s, 1),
        "total_requests":  total,
        "total_errors":    errors,
        "error_rate_pct":  round(errors / total * 100, 1) if total else 0.0,
        "endpoints":       endpoints_summary,
    }


# ════════════════════════════════════════════════════════════════════════════
# 에러 핸들러
# ════════════════════════════════════════════════════════════════════════════

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception(f"처리되지 않은 예외: {exc}")
    raise HTTPException(status_code=500, detail="내부 서버 오류")


# ════════════════════════════════════════════════════════════════════════════
# 진입점
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "07_api_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )
