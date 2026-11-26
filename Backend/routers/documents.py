import os
import tempfile

import pymupdf4llm
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from Backend.dependencies import get_current_user
from Backend.utils.vector_store import push_batch

router = APIRouter(tags=["documents"])
ALLOWED_FILE_EXTENSIONS = {
    ".pdf",
    ".xps",
    ".epub",
    ".cbz",
    ".mobi",
    ".fb2",
    ".svg",
    ".txt",
    ".md",
}


@router.post("/ingest")
@traceable(name="INGESTION")
async def ingest_pdf(
    thread_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    extension = os.path.splitext(file.filename or "")[1]
    if not file.filename or extension not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed.")

    contents = await file.read()
    if not contents or not contents.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded file contains no readable content.",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            data = pymupdf4llm.to_markdown(temp_path)
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse file: {error}",
            ) from error

        if not data or not data.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in file.",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks = splitter.split_text(data)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document produced no chunks.",
            )

        await push_batch(
            email=user["email"],
            chunks=chunks,
            thread_id=thread_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "message": "File successfully processed and stored.",
                "filename": file.filename,
                "chunks_ingested": len(chunks),
            },
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
