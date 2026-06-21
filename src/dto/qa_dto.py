from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EvidenceItem(BaseModel):
    title: str
    filename: str
    snippet: str

class QAIn(BaseModel):
    session_id: str
    question: str
    top_k: int = 3

class QAResp(BaseModel):
    answer: str
    evidence: List[EvidenceItem]
    stm_context: List[Dict[str, Any]]
    ltm_context: List[str]
    metadata: Optional[Dict[str, Any]] = None