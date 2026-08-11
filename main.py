from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse,FileResponse
from langchain_docling import loader
import io
from psycopg_pool import ConnectionPool
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import push
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
import os
from model import graph
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


chatbot = None
checkpointer = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, checkpointer, store
    pool=ConnectionPool(conninfo=str(os.getenv('DB_URI')),min_size=2,max_size=20,max_lifetime=3600.0,kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    checkpointer=PostgresSaver(pool)
    store=PostgresStore(pool)
    checkpointer.setup()
    store.setup()
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
async def ingest_pdf(email: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    tempfile1=None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile()

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no readable text.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(full_text)

        for chunk in chunks:
            push(email=email, chunks=chunk)

        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF successfully processed and stored.",
                "filename": file.filename,
                "chunks_ingested": len(chunks)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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