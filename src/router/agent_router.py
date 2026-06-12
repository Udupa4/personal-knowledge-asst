import logging
from fastapi import APIRouter, Depends, HTTPException

from src.agent.state import AgentState
from src.auth.auth import require_api_key
from src.dto.qa_dto import QAIn, QAResp, EvidenceItem
from src.agent.graph import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["agent"])

@router.post("/agent/qa", response_model=QAResp, dependencies=[Depends(require_api_key)])
async def agent_qa(payload: QAIn):
    """
    Main QA endpoint which flows through the LangGraph.
    """
    # Populate the input fields in initial State.
    initial_state: AgentState = {
        "question": payload.question,
        "session_id": payload.session_id,
        "user_id": payload.user_id,
        "top_k": payload.top_k,

        "route": "",
        "route_reasoning": "",
        "stm_context": [],
        "ltm_context": [],
        "retrieved_docs": [],
        "answer": "",
        "metadata": {},
        "clarification_question": "",
        "error": ""
    }

    final_state: AgentState = await graph.ainvoke(initial_state)

    return QAResp(
        answer=final_state.get("answer", "Couldn't generate an answer"),
        evidence=[EvidenceItem(**d) for d in final_state.get("retrieved_docs", [])],
        stm_context=final_state.get("stm_context", []),
        ltm_context=final_state.get("ltm_context", []),
        metadata=final_state.get("metadata", {}),
    )