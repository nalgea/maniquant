FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치 (캐시 레이어 분리)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY manidata/ ./manidata/
COPY maniagent/ ./maniagent/
COPY data/processed/ ./data/processed/

# bge-m3 모델 사전 다운로드 (빌드 시 캐싱)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "maniagent.07_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
