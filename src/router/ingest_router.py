from fastapi import APIRouter, Depends
from src.auth.dependencies import get_current_user, CurrentUser
from src.qa.retriever import VectorRetriever, ChunkedDocLoader

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/")
async def ingest_documents(current_user: CurrentUser = Depends(get_current_user)):
    loader = ChunkedDocLoader(chunk_size=800, chunk_overlap=200)
    docs = loader.load_and_split(only_new=True)
    retriever = VectorRetriever(user_id=current_user.user_id)
    retriever.build_or_load(docs=docs, add_new=True)
    return {"user_id": current_user.user_id, "docs_ingested": len(docs)}