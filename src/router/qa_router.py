import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from src.auth.auth import require_api_key
from src.dto.qa_dto import QAIn, QAResp
from src.memory.ltm_manager import LtmManager
from src.memory.stm_manager import StmMemoryManager
from src.qa.answerer import compose_prompt, synthesize_answer
from src.qa.retriever import ChunkedDocLoader, VectorRetriever
# from src.qa.answerer import compose_prompt, synthesize_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["qa"])
stm_mm = StmMemoryManager()
ltm_mm = LtmManager()

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
    stm_context = await stm_mm.read_stm(payload.session_id, k=6)
    ltm_context = ltm_mm.retrieve(payload.question, payload.user_id, k=3)

    matching_docs = vector_retriever.retrieve(payload.question, top_k=payload.top_k)
    prompt = compose_prompt(
        stm_context=stm_context,
        ltm_context=ltm_context,
        retrieved=matching_docs,
        user_question=payload.question
    )
    answer, metadata = await synthesize_answer(prompt)

    await stm_mm.write_stm(payload.session_id, user_text=payload.question, assistant_text=answer, meta=metadata)

    results = {
        "answer": answer,
        "matching_docs": matching_docs,
        "stm_context": stm_context,
        "ltm_context": ltm_context,
        "prompt": prompt,
        "metadata": metadata
    }
    return results