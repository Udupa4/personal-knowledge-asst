import os
import logging
from typing import Dict, List, Tuple, Any

from src.llm.llm import get_llm, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from src.llm.prompt_template import template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_ai_resp(ai_response: Any) -> Tuple[str, Dict]:
    """
    Extract content and metadata (model_name, finish_reason, prompt_feedback, usage) from ai_response
    """
    metadata = {}
    if hasattr(ai_response, "content"):
        text = ai_response.content
        resp_metadata = getattr(ai_response, "response_metadata", {}) or {}
        usage_metadata = getattr(ai_response, "usage_metadata", {}) or {}

        # Extract required metadata
        metadata["model_name"] = resp_metadata.get("model_name", "")
        metadata["finish_reason"] = resp_metadata.get("finish_reason", "")
        metadata["prompt_feedback"] = resp_metadata.get("prompt_feedback", "")
        metadata["usage"] = usage_metadata

        return text, metadata
    else:
        return str(ai_response), {}

def compose_prompt(stm_context: List[Dict[str, Any]], retrieved: List[Dict[str, Any]], user_question: str) -> str:
    parts = []

    if stm_context:
        parts.append("\nRecent conversations (most recent first):")
        for turn in stm_context:
            parts.append(f"- ({turn.get('role')}) {turn.get('text')}")

    if retrieved:
        parts.append("\nEvidence (top results):")
        for i, r in enumerate(retrieved, start=1):
            parts.append(f"[EVIDENCE {i}] Title: {r.get('title')}\nSnippet: {r.get('snippet')}\n")
    else:
        parts.append("\nNo Evidence Found.")

    parts.append(f"\nUser question: {user_question}")
    return "\n".join(parts)

async def synthesize_answer(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                            temperature: float = DEFAULT_TEMPERATURE) -> Tuple[str, Dict]:
    try:
        llm = get_llm()

        messages = [
            ("system", template),
            ("human", prompt)
        ]

        llm_response = await llm.ainvoke(messages)
        response_text, metadata = _parse_ai_resp(llm_response)
        return response_text, metadata
    except Exception as ex:
        logger.error(f"Error while synthesizing answer: {ex}")
        return "No response from the llm.", {}