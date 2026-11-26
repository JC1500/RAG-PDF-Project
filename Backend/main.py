import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from Backend.lifespan import lifespan
from Backend.routers import auth, chat, documents

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=str(os.getenv("SESSION_SECRET")),
    same_site=os.getenv("SESSION_SAME_SITE", "lax"),
    https_only=os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health_check")
def health_check():
    return {"message": "ALL GOOD"}
