from dotenv import load_dotenv
load_dotenv()

import logging
import uvicorn
from fastapi import FastAPI
from src.router.session_router import router as session_router
from src.router.qa_router import router as qa_router
from src.config.event_handler import custom_shutdown_event_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal Copilot (Phase 2)")

app.include_router(session_router)
app.include_router(qa_router)

app.add_event_handler("shutdown", custom_shutdown_event_handler())

if __name__ == "__main__":
    # Use uvicorn for now for development
    logger.info("Starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", log_config=None)
    logger.info("Server started at http://0.0.0.0:8080")