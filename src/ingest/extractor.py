import io
from fastapi import UploadFile
from pypdf import PdfReader


ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
ALLOWED_EXTENSIONS = {".txt", ".pdf"}


async def extract_text(file: UploadFile) -> str:
    """
    Read an UploadFile and return its text content as a plain string.
    Raises ValueError for unsupported types.
    """
    content_type = (file.content_type or "").split(";")[0].strip()
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    raw = await file.read()

    if content_type == "text/plain" or ext == ".txt":
        return raw.decode("utf-8", errors="replace")

    if content_type == "application/pdf" or ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    raise ValueError(
        f"Unsupported file type: content_type={content_type!r}, ext={ext!r}. "
        f"Allowed: .txt, .pdf"
    )