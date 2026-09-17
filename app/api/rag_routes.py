"""RAG API routes — document ingestion, retrieval admin, and document management."""

import asyncio
import os
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])

UPLOAD_DIR = "./temp_uploads"


# ---------------------------------------------------------------------------
# POST /rag/ingest
# ---------------------------------------------------------------------------


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    session_id: str = Form(default="default"),
    force: bool = Query(
        default=False,
        description=(
            "If true, delete the existing vectors for a duplicate file and "
            "re-ingest from scratch. Has no effect on genuinely new files."
        ),
    ),
):
    """Upload a document (.txt, .pdf, or .md) to the RAG vector store.

    Pipeline
    --------
    validation → parsing → cleaning → structure detection →
    metadata extraction → intelligent chunking → embedding → vector storage

    Deduplication
    -------------
    The file's SHA-256 hash is checked before any embedding work begins.
    Uploading an identical file a second time returns ``status="duplicate"``
    with no re-embedding — unless ``force=true`` is passed, which will delete
    the old vectors and re-ingest the document entirely.

    Context fields
    --------------
    ``user_id`` and ``session_id`` are stored in per-chunk metadata so that
    retrieval results can be attributed to their uploader context.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = os.path.basename(file.filename or "upload")
    unique_filename = f"{uuid.uuid4().hex}_{safe_filename}"
    temp_file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save the uploaded file to a temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the blocking pipeline off the async event loop
        result = await asyncio.to_thread(
            rag_service.ingest_file,
            file_path=temp_file_path,
            metadata={
                "filename": safe_filename,
                "user_id": user_id,
                "session_id": session_id,
            },
            force=force,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc

    finally:
        # Always clean up the temporary upload file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    status = result.get("status")

    if status == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Failed to ingest '{safe_filename}'.",
                "error": result.get("error"),
                "document_id": result.get("document_id"),
            },
        )

    if status == "duplicate":
        return {
            "message": (
                f"'{safe_filename}' was already ingested as "
                f"'{result['duplicate_of']}' on {result['originally_ingested_at']}. "
                "Skipped to avoid duplicate chunks. Pass force=true to re-ingest."
            ),
            "status": "duplicate",
            "document_id": result["document_id"],
            "chunks_added": 0,
            "duplicate_of": result["duplicate_of"],
            "originally_ingested_at": result["originally_ingested_at"],
            "user_id": result["user_id"],
            "session_id": result["session_id"],
        }

    return {
        "message": f"Successfully ingested '{safe_filename}'.",
        "status": "ingested",
        "document_id": result["document_id"],
        "chunks_added": result["chunks_added"],
        "filename": result["filename"],
        "document_type": result["document_type"],
        "user_id": result["user_id"],
        "session_id": result["session_id"],
    }


# ---------------------------------------------------------------------------
# GET /rag/documents
# ---------------------------------------------------------------------------


@router.get("/documents")
async def list_ingested_documents():
    """List all documents tracked in the RAG vector store.

    Returns ingestion status, chunk counts, document IDs, and uploader context.
    """
    docs = await asyncio.to_thread(rag_service.list_documents)
    return {
        "total": len(docs),
        "documents": docs,
    }


# ---------------------------------------------------------------------------
# GET /rag/documents/{document_id}
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Fetch metadata for a single document by its document_id."""
    doc = await asyncio.to_thread(rag_service.get_document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with document_id='{document_id}'.",
        )
    return doc


# ---------------------------------------------------------------------------
# DELETE /rag/documents/{document_id}
# ---------------------------------------------------------------------------


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document's vectors from Qdrant and its DB record.

    This is permanent — the document must be re-uploaded to be available
    for retrieval again.
    """
    result = await asyncio.to_thread(rag_service.delete_document, document_id)

    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"No document found with document_id='{document_id}'.",
        )

    return {
        "message": f"Document '{document_id}' deleted successfully.",
        "status": "deleted",
        "document_id": document_id,
        "vectors_deleted": result["vectors_deleted"],
    }
