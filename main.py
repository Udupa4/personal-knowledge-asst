from dotenv import load_dotenv
load_dotenv()

import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from src.router.auth_router import router as auth_router
from src.router.ingest_router import router as ingest_router
from src.router.session_router import router as session_router
from src.router.agent_router import router as agent_router
from src.router.memory_router import router as memory_router
from src.router.probes_router import router as probes_router
from src.config.event_handler import custom_shutdown_event_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal Knowledge Assistant")

app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(session_router)
app.include_router(agent_router)
app.include_router(memory_router)
app.include_router(probes_router)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    custom_shutdown_event_handler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # Use uvicorn for now for development
    logger.info("Starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", log_config=None)
    logger.info("Server started at http://0.0.0.0:8080")