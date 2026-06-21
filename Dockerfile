FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY manidata/ ./manidata/
COPY maniagent/ ./maniagent/
COPY data/processed/ ./data/processed/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000

CMD ["uvicorn", "maniagent.07_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
