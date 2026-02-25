from pydantic import BaseModel
from typing import List, Dict, Any

class QAIn(BaseModel):
    session_id: str
    question: str
    top_k: int = 3

class QAResp(BaseModel):
    answer: str
    evidence: List[Dict[str, Any]]
    used_stm: List[Dict[str, Any]]