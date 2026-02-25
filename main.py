# app/main.py
import uvicorn
from fastapi import FastAPI
from src.router.session_router import router as session_router
from src.router.qa_router import router as qa_router
from dotenv import load_dotenv
from src.config.event_handler import custom_shutdown_event_handler

load_dotenv()

app = FastAPI(title="Personal Copilot (Phase 2)")

app.include_router(session_router)
app.include_router(qa_router)

app.add_event_handler("shutdown", custom_shutdown_event_handler())

if __name__ == "__main__":
    # Use uvicorn for now for development
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True, log_level="info")