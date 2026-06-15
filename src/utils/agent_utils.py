import logging
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def get_message_text(msg: AIMessage) -> str:
    """
    Extract plain text from an AIMessage, handling both content shapes:

    - content as a string (Ollama / OpenAI-compatible models):
        "Here is the answer..."

    - content as a list of blocks (Gemini 2.5 with thought signatures):
        [{"type": "text", "text": "Here is the answer...", "signature": "..."}]

    Only "text" type blocks are extracted — "thinking" or other block types
    are skipped, since they aren't part of the user-facing answer.
    """
    content = getattr(msg, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()

    logger.warning(f"Unexpected message content type: {type(content)}")
    return ""