FROM python:3.11-slim

RUN adduser --system --no-create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY ui/ ./ui/
COPY main.py .

USER appuser

ENV PORT=8080
ENV HF_HOME=/app/.cache/huggingface
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]