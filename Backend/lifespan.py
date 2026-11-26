import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore


load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
    from Backend.model.model import graph

    pool = AsyncConnectionPool(
        conninfo=str(os.getenv("DB_URI")),
        min_size=2,
        max_size=20,
        max_lifetime=3600.0,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(pool)
        # await checkpointer.setup()
        store = AsyncPostgresStore(pool)
        # await store.setup()
        app.state.chatbot = graph.compile(checkpointer=checkpointer, store=store)
        yield
    finally:
        await pool.close()

