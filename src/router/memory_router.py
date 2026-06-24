# src/router/memory_router.py
import logging
from fastapi import APIRouter, Depends
from src.auth.dependencies import get_current_user, CurrentUser
from src.memory.ltm_manager import LtmManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])
ltm_mm = LtmManager()

@router.get("/ltm")
async def get_ltm_for_user(current_user: CurrentUser = Depends(get_current_user)):
    """Get all LTM saved for a specific user."""
    return ltm_mm.get_all_for_user(current_user.user_id)

@router.delete("/ltm")
async def delete_ltm_for_user(current_user: CurrentUser = Depends(get_current_user)):
    """Delete all LTM entries of a specific user."""
    ltm_mm.delete_for_user(current_user.user_id)
    return {"user_id": current_user.user_id, "status": "ltm cleared"}
