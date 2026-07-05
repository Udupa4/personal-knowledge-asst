from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from langchain_core.documents import Document

from src.auth.dependencies import get_current_user, CurrentUser
from src.ingest.extractor import extract_text, ALLOWED_MIME_TYPES, ALLOWED_EXTENSIONS
from src.ingest.file_registry import save_file_record, list_files, delete_file_record
from src.qa.retriever import VectorRetriever

router = APIRouter(prefix="/ingest", tags=["ingest"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB hard ceiling

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...), current_user: CurrentUser = Depends(get_current_user)):
    # ── validate ──────────────────────────────────────────────────────────
    content_type = (file.content_type or "").split(";")[0].strip()
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if content_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: .txt, .pdf",
        )

    # ── read + size-check ─────────────────────────────────────────────────
    # extractor.py calls file.read() internally; we need size before that.
    # Reset approach: read once here, pass bytes through — but extractor
    # takes UploadFile. Simpler: read raw bytes here, check size, then
    # delegate. We wrap in a thin shim so extractor stays reusable.
    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds 10 MB limit.",
        )

    # ── extract text ──────────────────────────────────────────────────────
    try:
        text = await _extract_from_bytes(raw, content_type, ext)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the file.",
        )

    # ── ingest into Qdrant ────────────────────────────────────────────────
    doc = Document(
        page_content=text,
        metadata={"source": filename, "user_id": current_user.user_id},
    )
    retriever = VectorRetriever(user_id=current_user.user_id)
    retriever.build_or_load(docs=[doc], add_new=True)

    # build_or_load returns the underlying retriever; chunk count comes
    # from the child splitter. Approximate via the retriever's child
    # vectorstore search — but that's expensive. Instead, replicate the
    # chunk count logic cheaply:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunk_count = len(child_splitter.split_text(text))

    # ── persist file record ───────────────────────────────────────────────
    try:
        file_id = save_file_record(
            user_id=current_user.user_id,
            filename=filename,
            size_bytes=len(raw),
            mime_type=content_type or ext,
            chunk_count=chunk_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    return {
        "file_id": file_id,
        "filename": filename,
        "chunks_ingested": chunk_count,
    }

@router.get("/files")
async def get_files(current_user: CurrentUser = Depends(get_current_user)):
    return list_files(current_user.user_id)

@router.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(file_id: str, current_user: CurrentUser = Depends(get_current_user)):
    records = list_files(current_user.user_id)
    if not any(f["file_id"] == file_id for f in records):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    delete_file_record(current_user.user_id, file_id)
    return {"status": f"successfully deleted file with file_id {file_id}"}



# ── internal helper ────────────────────────────────────────────────────────────
# Mirrors extractor.py but operates on pre-read bytes so we can size-check
# before parsing. Keeps extractor.py's UploadFile interface intact for
# standalone use, while router avoids a double-read.

import io
# pyrefly: ignore [missing-import]
from pypdf import PdfReader


async def _extract_from_bytes(raw: bytes, content_type: str, ext: str) -> str:
    if content_type == "text/plain" or ext == ".txt":
        return raw.decode("utf-8", errors="replace")

    if content_type == "application/pdf" or ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    raise ValueError(f"Unsupported file type: {content_type!r} / {ext!r}")