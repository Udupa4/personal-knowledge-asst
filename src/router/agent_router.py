import logging
from fastapi import APIRouter, Depends
from langchain_core.messages import AIMessage, ToolMessage

from src.auth.auth import require_api_key
from src.dto.qa_dto import QAIn, QAResp, EvidenceItem
from src.agent.graph import graph
from src.agent.tools import UserContext
from src.utils.agent_utils import get_message_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["agent"])


def _extract_final_answer(final_state: dict) -> str:
    if final_state.get("answer") and isinstance(final_state["answer"], str):
        return final_state["answer"]

    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = get_message_text(msg)
        if text:
            return text

    return final_state.get("error") or "(no response)"


def _extract_retrieved_docs(final_state: dict) -> list:
    """
    Extract retrieved docs from ToolMessage results in state["messages"].

    search_knowledge_base returns a formatted string of doc snippets.
    We parse that string back into EvidenceItem-compatible dicts so the
    evidence array in the API response is populated correctly.

    ToolMessage.name tells us which tool produced the result, so we only
    parse messages from search_knowledge_base, not web_search or memory.
    """
    docs = []
    messages = final_state.get("messages", [])

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        # Only parse knowledge base results, not web search or memory
        if getattr(msg, "name", "") != "search_knowledge_base":
            continue

        content = msg.content or ""
        if not content or content.startswith("No relevant"):
            continue

        # search_knowledge_base formats results as:
        # [filename.txt]\nsnippet text\n\n[filename2.txt]\nsnippet text
        # Split on double newline to separate individual doc blocks
        blocks = content.split("\n\n")
        for block in blocks:
            lines = block.strip().splitlines()
            if not lines:
                continue

            # First line is [filename.txt]
            first_line = lines[0].strip()
            if first_line.startswith("[") and first_line.endswith("]"):
                filename = first_line[1:-1]
                snippet = "\n".join(lines[1:]).strip()
                docs.append({
                    "title": filename.replace(".txt", "").replace("_", " ").title(),
                    "filename": filename,
                    "snippet": snippet,
                })

    return docs


@router.post("/agent/qa", response_model=QAResp, dependencies=[Depends(require_api_key)])
async def agent_qa(payload: QAIn):
    initial_state = {
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
        "clarification_question": None,
        "error": None,
        "messages": [],
    }

    final_state = await graph.ainvoke(initial_state, context=UserContext(payload.user_id))
    answer = _extract_final_answer(final_state)
    retrieved_docs = _extract_retrieved_docs(final_state)

    return QAResp(
        answer=answer,
        evidence=[EvidenceItem(**d) for d in retrieved_docs],
        stm_context=final_state.get("stm_context", []),
        ltm_context=final_state.get("ltm_context", []),
        metadata=final_state.get("metadata", {}),
    )