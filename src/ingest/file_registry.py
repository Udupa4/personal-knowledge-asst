import uuid
from datetime import datetime, timezone
from typing import Any

from src.auth.firestore_client import FirestoreClient

MAX_FILES_PER_USER = 3

def _files_col(user_id: str):
    db = FirestoreClient().get_db()
    return db.collection("users").document(user_id).collection("files")


def save_file_record(
    user_id: str,
    filename: str,
    size_bytes: int,
    mime_type: str,
    chunk_count: int,
) -> str:
    col = _files_col(user_id)

    existing = list(col.stream())
    if len(existing) >= MAX_FILES_PER_USER:
        raise ValueError(
            f"Upload limit reached. You can store at most {MAX_FILES_PER_USER} files. "
            "Delete an existing file before uploading a new one."
        )

    file_id = str(uuid.uuid4())
    record: dict[str, Any] = {
        "filename": filename,
        "size_bytes": size_bytes,
        "mime_type": mime_type,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
    }
    col.document(file_id).set(record)
    return file_id

def list_files(user_id: str) -> list[dict[str, Any]]:
    """
    Return all file records for a user, newest first.
    """
    docs = (
        _files_col(user_id)
        .order_by("uploaded_at", direction="DESCENDING")
        .stream()
    )
    return [{"file_id": doc.id, **doc.to_dict()} for doc in docs]


def delete_file_record(user_id: str, file_id: str) -> None:
    """
    Remove a file record from Firestore.
    Note: does NOT remove vectors from Qdrant — out of scope.
    """
    _files_col(user_id).document(file_id).delete()