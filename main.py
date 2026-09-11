from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse,HTMLResponse
from psycopg_pool import ConnectionPool
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.vector_store import push_batch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
import os
from model import graph
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from psycopg.rows import dict_row
import pymupdf4llm
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from utils.classes import ChatRequest

chatbot = None
checkpointer = None
store = None

#OpenID Connect setup (GOOGLE)
oauth=OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, checkpointer, store
    pool=ConnectionPool(conninfo=str(os.getenv('DB_URI')),min_size=2,max_size=20,max_lifetime=3600.0,kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    checkpointer=PostgresSaver(pool)
    store=PostgresStore(pool)
    chatbot = graph.compile(checkpointer=checkpointer, store=store)
    yield 
    
    pool.close()

app = FastAPI(lifespan=lifespan)

@app.get('/',response_class=HTMLResponse)
def default():
    return FileResponse("index.html")

app.add_middleware(
    SessionMiddleware,
    secret_key=str(os.getenv("SESSION_SECRET")),
    same_site="lax",
    https_only=os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true",
)




def get_current_user(request: Request) -> dict:
    """Reads the authenticated user from the server-side session.
    Raises 401 if no one is logged in — the frontend is expected to
    redirect to /auth/login in that case."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user




#Auth

@app.get("/auth/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    print("LOGIN session id:", request.session)
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    print("CALLBACK session before exchange:", request.session)
    try:
        print(str(request))
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Login failed: {e}")

    userinfo = token.get("userinfo")
    if not userinfo or not userinfo.get("email"):
        raise HTTPException(status_code=400, detail="Google did not return an email.")

    request.session["user"] = {
        "email": userinfo["email"],
        "name": userinfo.get("name", userinfo["email"]),
    }
    return RedirectResponse(url="/")


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")


@app.get("/auth/me")
async def me(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


#Data ingestion

@app.post("/ingest")
async def ingest_pdf(
    thread_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    email = user["email"]
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
 
        await push_batch(email=email, chunks=chunks, thread_id=thread_id)
 
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
def chat_endpoint(request: ChatRequest, user: dict = Depends(get_current_user)):
    email = user["email"]
    res = chatbot.invoke(
        {"messages": [HumanMessage(content=request.inp)], 'email': email},
        config={"configurable": {"thread_id": request.thread_id, "user_id": email}}
    )
    
    # Extract response content cleanly to return JSON
    messages = res.get("messages", [])
    latest_message = messages[-1].content if messages else ""
    
    return {
        "response": latest_message,
        "summary": res.get("summary", "")
    }