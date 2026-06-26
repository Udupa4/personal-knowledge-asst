from typing import List, Dict, Any
from src.llm.llm import get_llm

SUMMARIZE_PROMPT = """You are a long-term memory (LTM) distillation assistant for an AI agent. 
Your job is to analyze a conversation and extract core insights about the USER to help the agent personalize future interactions.

Ignore generic educational content, explanations of technical concepts, or raw facts about the world. Instead, extract 3-7 concise bullet points strictly capturing:
1. User Profile & Context: What is the user working on? (e.g., building a platform, optimizing a RAG system).
2. User Preferences & Constraints: Specific technologies they use, version constraints, or preferred architectural styles.
3. User Intent & Interests: Topics they actively inquired about, problems they are trying to solve, or goals they expressed.
4. Decisions & Action Items: What the user decided to implement or move forward with.

Frame every bullet point from the perspective of the user's state or actions. Do not output conversational filler; output only the bullet points."""

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