FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY manidata/ ./manidata/
COPY maniagent/ ./maniagent/

# ── bge-m3 모델 빌드 시 미리 다운로드 (시작 시간 단축) ──────────────────────
# Railway 배포 시 콜드 스타트 타임아웃 방지
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('bge-m3 모델 다운로드 중...'); \
m = SentenceTransformer('BAAI/bge-m3', cache_folder='/app/.cache/models'); \
print('완료:', m.get_sentence_embedding_dimension(), 'dim')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# 캐시 경로 설정 (다운로드된 모델 재사용)
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/models
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "maniagent.07_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
