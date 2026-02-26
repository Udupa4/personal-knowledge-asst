# app/routers/qa_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from src.auth.auth import require_api_key
from src.dto.qa_dto import QAIn, QAResp
from src.memory.manager import MemoryManager
from src.qa.retriever import ChunkedDocLoader, VectorRetriever
# from src.qa.answerer import compose_prompt, synthesize_answer

router = APIRouter(prefix="", tags=["qa"])
mm = MemoryManager()
# initialize retriever and loader
loader = ChunkedDocLoader(chunk_size=800, chunk_overlap=200)
vector_retriever = VectorRetriever()

# ingest docs on startup
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
    stm = await mm.read_stm(payload.session_id, k=6)
    retrieved = vector_retriever.retrieve(payload.question, top_k=payload.top_k)
    results = {"matching_docs": retrieved, "used_stm": stm}
    return results

    # TODO: implement answerer
    # prompt = compose_prompt(stm, retrieved, payload.question)
    # answer = synthesize_answer(prompt)
    # return {"answer": answer, "evidence": retrieved, "used_stm": stm}