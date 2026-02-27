import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from src.auth.auth import require_api_key
from src.dto.qa_dto import QAIn, QAResp
from src.memory.manager import MemoryManager
from src.qa.answerer import compose_prompt, synthesize_answer
from src.qa.retriever import ChunkedDocLoader, VectorRetriever
# from src.qa.answerer import compose_prompt, synthesize_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["qa"])
mm = MemoryManager()
# initialize retriever and loader
loader = ChunkedDocLoader(chunk_size=800, chunk_overlap=200)
vector_retriever = VectorRetriever()
docs = loader.load_and_split()
vector_retriever.build_or_load(docs)

# create an endpoint to (re)ingest docs into Chroma via API
@router.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest_docs():
    """
    Create or update the local Chroma store from a list of Documents (chunks).
    """
    documents = loader.load_and_split()
    count = len(documents)
    vector_retriever.build_or_load(documents)
    return {"status": "ingested", "chunks": count}

@router.post("/qa", dependencies=[Depends(require_api_key)])
async def ask_question(payload: QAIn):
    stm_context = await mm.read_stm(payload.session_id, k=6)
    retrieved = vector_retriever.retrieve(payload.question, top_k=payload.top_k)
    prompt = compose_prompt(stm_context=stm_context, retrieved=retrieved, user_question=payload.question)
    answer, metadata = await synthesize_answer(prompt)
    await mm.write_stm(payload.session_id, "user", payload.question)    # Write user's question into STM
    await mm.write_stm(payload.session_id, "assistant", answer)       # Write assistant's answer into STM
    results = {
        "answer": answer,
        "retrieved": retrieved,
        "stm_context": stm_context,
        "prompt": prompt,
        "metadata": metadata
    }
    return results