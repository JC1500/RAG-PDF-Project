from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse,FileResponse
from psycopg_pool import ConnectionPool
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import push_batch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
import os
from model import graph
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from psycopg.rows import dict_row
import pymupdf4llm


chatbot = None
checkpointer = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, checkpointer, store
    pool=ConnectionPool(conninfo=str(os.getenv('DB_URI')),min_size=2,max_size=20,max_lifetime=3600.0,kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    checkpointer=PostgresSaver(pool)
    store=PostgresStore(pool)
    chatbot = graph.compile(checkpointer=checkpointer, store=store)
    yield 
    
    pool.close()
    print("Database connections closed successfully.")

app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    inp: str
    user_id: str
    thread_id: str

# @app.get('/')
# def default():
#     return FileResponse("index.html")

@app.post("/ingest")
async def ingest_pdf(email: str,thread_id:str, file: UploadFile = File(...)):
    file_extensions = [
    ".pdf",   # Portable Document Format
    ".xps",   # XML Paper Specification
    ".epub",  # Electronic Publication
    ".cbz",   # Comic Book Zip
    ".mobi",  # Mobipocket eBook
    ".fb2",   # FictionBook (XML-based eBook)
    ".svg",   # Scalable Vector Graphics
    ".txt",   # Plain Text
    ".md"     # Markdown
    ]

    if not file.filename or not any(file.filename.endswith(ext) for ext in file_extensions):
        raise HTTPException(status_code=400, detail="File type not allowed.")
 
    contents = await file.read()
    if not contents or not contents.strip():
        raise HTTPException(status_code=400, detail="The uploaded file contains no readable content.")
 
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
 
        try:
            data = pymupdf4llm.to_markdown(temp_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")
 
        if not data or not data.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in file.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks = splitter.split_text(data)
 
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced no chunks.")
 
        push_batch(email=email, chunks=chunks, thread_id=thread_id)
 
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


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    res = chatbot.invoke(
        {"messages": [HumanMessage(content=request.inp)]},
        config={"configurable": {"thread_id": request.thread_id, "user_id": request.user_id}}
    )
    
    # Extract response content cleanly to return JSON
    messages = res.get("messages", [])
    latest_message = messages[-1].content if messages else ""
    
    return {
        "response": latest_message,
        "summary": res.get("summary", "")
    }