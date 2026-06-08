import logging
from fastapi import APIRouter, Depends, HTTPException
import uuid

from src.auth.auth import require_api_key
from src.dto.session_dto import CreateSessionResp, TurnIn
from src.memory.ltm_manager import LtmManager
from src.memory.stm_manager import StmMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["session"])
stm_mm = StmMemoryManager()
ltm_mm = LtmManager()

@router.post("/session/{user_id}", response_model=CreateSessionResp, dependencies=[Depends(require_api_key)])
async def create_session(user_id: str):
    session_id = str(uuid.uuid4())
    await stm_mm.create_session(session_id, user_id=user_id)
    return {"session_id": session_id}

@router.get("/session", dependencies=[Depends(require_api_key)])
async def get_sessions():
    sessions = await stm_mm.list_sessions()
    return {"sessions": sessions}

@router.delete("/session/{session_id}/end", dependencies=[Depends(require_api_key)])
async def end_session(session_id: str, user_id: str):
    summary = await stm_mm.flush_to_ltm(session_id, user_id, ltm_mm)
    return {"summary": summary, "user_id": user_id, "status": "flushed to LTM"}

@router.delete("/session", dependencies=[Depends(require_api_key)])
async def end_all_sessions():
    await stm_mm.clear_all_sessions()
    return {"status": "deleted all sessions"}

@router.post("/session/{session_id}/turn", dependencies=[Depends(require_api_key)])
async def add_turn(session_id: str, turn: TurnIn):
    item = await stm_mm.write_stm(session_id, turn.user, turn.assistant)
    return item

@router.get("/session/{session_id}/context", dependencies=[Depends(require_api_key)])
async def get_context(session_id: str, k: int = 6):
    items = await stm_mm.read_stm(session_id=session_id, k=k)
    return items
