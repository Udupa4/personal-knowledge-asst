FROM python:3.11-slim

RUN adduser --system --no-create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY ui/ ./ui/
COPY main.py .

# Download the model during build, into a fixed path we control.
# We run this as root (before USER appuser) so we can write anywhere.
# Set HF cache to a known directory, then chown it to appuser.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')" \
    && chown -R appuser:0 /app/.cache

USER appuser

ENV PORT=8080
ENV HF_HOME=/app/.cache/huggingface
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]