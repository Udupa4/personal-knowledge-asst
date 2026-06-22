import logging
from fastapi import APIRouter, Depends, HTTPException
import uuid

from src.auth.dependencies import get_current_user, CurrentUser
from src.dto.session_dto import CreateSessionResp, TurnIn
from src.memory.ltm_manager import LtmManager
from src.memory.stm_manager import StmMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["session"])
stm_mm = StmMemoryManager()
ltm_mm = LtmManager()

@router.post("/session", response_model=CreateSessionResp)
async def create_session(current_user: CurrentUser = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    await stm_mm.create_session(session_id, user_id=current_user.user_id)
    return {"session_id": session_id}

@router.get("/session")
async def get_sessions(current_user: CurrentUser = Depends(get_current_user)):
    sessions = await stm_mm.list_sessions()
    return {"sessions": sessions}

@router.delete("/session/{session_id}/end")
async def end_session(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    summary = await stm_mm.flush_to_ltm(session_id, current_user.user_id, ltm_mm)
    return {"summary": summary, "user_id": current_user.user_id, "status": "flushed to LTM"}

@router.delete("/session")
async def end_all_sessions(current_user: CurrentUser = Depends(get_current_user)):
    await stm_mm.clear_all_sessions()
    return {"status": "deleted all sessions"}

@router.post("/session/{session_id}/turn")
async def add_turn(session_id: str, turn: TurnIn, current_user: CurrentUser = Depends(get_current_user)):
    item = await stm_mm.write_stm(session_id, turn.user, turn.assistant)
    return item

@router.get("/session/{session_id}/context")
async def get_context(session_id: str, k: int = 6, current_user: CurrentUser = Depends(get_current_user)):
    items = await stm_mm.read_stm(session_id=session_id, k=k)
    return items
