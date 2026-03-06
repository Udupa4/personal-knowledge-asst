# src/router/memory_router.py
import logging
from fastapi import APIRouter, Depends
from src.auth.auth import require_api_key
from src.memory.ltm_manager import LtmManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])
ltm_mm = LtmManager()

@router.get("/ltm", dependencies=[Depends(require_api_key)])
async def get_all_ltm():
    """Get all LTM entries across all users, grouped by user_id."""
    return ltm_mm.get_all()

@router.get("/ltm/{user_id}", dependencies=[Depends(require_api_key)])
async def get_ltm_for_user(user_id: str):
    """Get all LTM entries for a specific user."""
    return ltm_mm.get_all_for_user(user_id)

@router.delete("/ltm/{user_id}", dependencies=[Depends(require_api_key)])
async def delete_ltm_for_user(user_id: str):
    """Delete all LTM entries for a specific user."""
    ltm_mm.delete_for_user(user_id)
    return {"user_id": user_id, "status": "ltm cleared"}
