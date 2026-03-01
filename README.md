# Personal Knowledge Assistant

A FastAPI-based personal knowledge assistant that combines document retrieval (RAG) with session-based short-term memory. Drop `.txt` files into `data/docs/`, ingest them into a local Chroma vector store, and query against them with per-session conversation context backed by Redis.

---

## Architecture

```
main.py                         # FastAPI app entry point
src/
├── auth/auth.py                # API key authentication (X-Api-Key header)
├── common/utils/singletone.py  # SingletonMeta metaclass
├── config/event_handler.py     # Shutdown event: flushes all Redis sessions
├── dto/
│   ├── qa_dto.py               # QAIn, QAResp Pydantic models
│   └── session_dto.py          # CreateSessionResp, TurnIn Pydantic models
├── memory/stm_manager.py           # Redis-backed short-term memory (StmMemoryManager)
├── qa/retriever.py             # Document loader + Chroma vector retriever
└── router/
    ├── session_router.py       # Session lifecycle endpoints
    └── qa_router.py            # Ingest + QA endpoints
data/docs/                      # Place .txt knowledge documents here
chroma_store/                   # Auto-created Chroma persistence directory
```

---

## Prerequisites

- Python 3.10+
- Redis server running on `localhost:6379`
- HuggingFace model `all-MiniLM-L6-v2` cached locally

---

## Setup

1. **Clone and create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** — copy `.env-example` to `.env` and fill in values:
   ```
   REDIS_URL=redis://localhost:6379/0
   API_KEY=your_api_key_here
   STM_TTL_SECONDS=86400
   CHROMA_PERSIST_DIR=./chroma_store
   ```

4. **Start Redis:**
   ```bash
   redis-server
   # or with Docker:
   docker run -d --name redis -p 6379:6379 redis:7-alpine
   ```

5. **Add knowledge documents** — place `.txt` files in `data/docs/`.

6. **Run the server:**
   ```bash
   python main.py
   ```
   Server starts at `http://0.0.0.0:8080`.

---

## API Endpoints

All endpoints require the header: `X-Api-Key: <your_api_key>`

### Session

| Method   | Path                                | Description                                               |
|----------|-------------------------------------|-----------------------------------------------------------|
| `POST`   | `/session`                          | Create a new session, returns `session_id`                |
| `GET`    | `/session`                          | List all active session IDs                               |
| `POST`   | `/session/{session_id}/turn`        | Append a turn (`role`: `user`/`assistant`/`tool`, `text`) |
| `GET`    | `/session/{session_id}/context?k=6` | Retrieve last `k` turns                                   |
| `DELETE` | `/session/{session_id}/end`         | Delete a specific session                                 |
| `DELETE` | `/session`                          | Delete all sessions                                       |

### QA & Ingestion

| Method | Path      | Description                                           |
|--------|-----------|-------------------------------------------------------|
| `POST` | `/ingest` | Re-ingest docs from `data/docs/` into Chroma          |
| `POST` | `/qa`     | Query against the knowledge base with session context |

#### `POST /qa` request body:
```json
{
  "session_id": "uuid-string",
  "question": "What is ...?",
  "top_k": 3
}
```

---

## Notes

- Documents are **auto-ingested on startup** from `data/docs/`. Only `.txt` files are supported for now.
- The `POST /ingest` endpoint can be used to re-ingest after adding new documents without restarting.
- The HuggingFace embedding model runs with `local_files_only=True` — ensure the model is cached before running in network-restricted environments.
- `StmMemoryManager` is a **singleton** — the same Redis connection is shared across all routers.
- On server shutdown, all Redis session keys are flushed via `flushdb`.
- The LLM answerer (`synthesize_answer`) is not yet implemented — `/qa` currently returns raw retrieved chunks and STM context.

---

## Interactive Docs

FastAPI auto-generates docs at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
