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

# ── 임베딩 모델 빌드 시 미리 다운로드 ─────────────────────────────────────────
# paraphrase-multilingual-mpnet-base-v2:
#   - 크기: ~280MB (Railway 무료 플랜 호환)
#   - 차원: 768d (Zilliz 스키마와 동일)
#   - 지원 언어: 한국어, 중국어, 일본어, 영어 포함 50+ 언어
ARG EMBED_MODEL=paraphrase-multilingual-mpnet-base-v2
ENV EMBED_MODEL_NAME=${EMBED_MODEL}
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/models
ENV HF_HOME=/app/.cache/huggingface

RUN python -c "\
import os; \
from sentence_transformers import SentenceTransformer; \
model_name = os.getenv('EMBED_MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2'); \
print(f'모델 다운로드: {model_name}'); \
m = SentenceTransformer(model_name, cache_folder='/app/.cache/models'); \
dim = m.get_sentence_embedding_dimension(); \
print(f'완료 — 차원: {dim}d')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "maniagent.07_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
