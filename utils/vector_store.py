from qdrant_client import AsyncQdrantClient, models
import os
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
from typing import List, Dict
import uuid
import asyncio

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
collection_name = 'test'

client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global client
    if client is None:
        client = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            cloud_inference=True,
            timeout=100,
        )
    return client


async def ensure_collection(client: AsyncQdrantClient):
    if not await client.collection_exists(collection_name):
        size = client.get_embedding_size(model_name)
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE)
        )
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="email",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="thread_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )


async def get_from_store(email: str, query: str, thread_id: str):
    client = get_client()
    if not await client.collection_exists(collection_name):
        return ['Sorry the data doesnt exits']
    response = (await client.query_points(
        collection_name=collection_name,
        query=Document(text=query, model=model_name),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="email",
                    match=models.MatchValue(value=email),
                ),
                models.FieldCondition(
                    key="thread_id",
                    match=models.MatchValue(value=thread_id),
                )
            ]
        ),
        with_payload=True,
        limit=5,
    )).points
    ans = []
    for i in response:
        ans.append(i.payload['data'])  # type:ignore
    return ans


async def push(email: str, chunks: str, thread_id: str):
    client = get_client()
    await ensure_collection(client)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=Document(text=chunks, model=model_name),
        payload={
            "data": chunks,
            "email": email,
            "thread_id": thread_id
        }
    )
    await client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[point]
    )


async def push_batch(email: str, chunks: list[str], thread_id: str):
    client = get_client()
    await ensure_collection(client)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=Document(text=c, model=model_name),
            payload={"data": c, "email": email, "thread_id": thread_id, "chunk_index": i},
        )
        for i, c in enumerate(chunks)
    ]
    # Fire batches concurrently instead of awaiting each one sequentially.
    tasks = [
        client.upsert(collection_name=collection_name, wait=False, points=points[i:i + 100])
        for i in range(0, len(points), 100)
    ]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    async def _main():
        em = 'abc@gmail.com'
        await push_batch(chunks=['hello', 'hi', 'bye'], email=em, thread_id='13')
        a = await get_from_store(query='hello', email=em, thread_id='13')
        print(a)

    asyncio.run(_main())