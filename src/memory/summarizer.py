from typing import List, Dict, Any
from src.llm.llm import get_llm

SUMMARIZE_PROMPT = """You are a memory distillation assistant.
Given a conversation between a user and an AI assistant, extract 3-7 concise bullet points
capturing: key facts the user shared, topics explored, preferences expressed, and decisions made.
Output only the bullet points, nothing else."""

async def summarize_turns(turns: List[Dict[str, Any]]) -> str:
    """Convert a list of STM turn-pairs into a bullet-point LTM summary."""
    if not turns:
        return ""
    # turns are stored most-recent-first, reverse for chronological order
    conversation = "\n".join([
        f"User: {t['user']}\nAssistant: {t['assistant']}"
        for t in reversed(turns)
    ])
    llm = get_llm()
    response = await llm.ainvoke([
        ("system", SUMMARIZE_PROMPT),
        ("human", conversation)
    ])
    return response.content.strip()