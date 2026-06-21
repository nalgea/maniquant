FROM python:3.11-slim

WORKDIR /app

# v4
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY manidata/ ./manidata/
COPY maniagent/ ./maniagent/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn maniagent.07_api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
