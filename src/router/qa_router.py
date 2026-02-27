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
    stm = await mm.read_stm(payload.session_id, k=6)
    retrieved = vector_retriever.retrieve(payload.question, top_k=payload.top_k)
    prompt = compose_prompt(stm_context=stm, retrieved=retrieved, user_question=payload.question)
    answer, metadata = await synthesize_answer(prompt)
    results = {"answer": answer, "metadata": metadata}
    return results

    # TODO: implement answerer
    # prompt = compose_prompt(stm, retrieved, payload.question)
    # answer = synthesize_answer(prompt)
    # return {"answer": answer, "evidence": retrieved, "used_stm": stm}