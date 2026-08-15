from qdrant_client import QdrantClient,models
import os
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, PointStruct,Document
from typing import List,Dict
import uuid

load_dotenv()

model_name="sentence-transformers/all-MiniLM-L6-v2"
collection_name='test'


def get(email: str, query: str,thread_id:str):
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,
    )
    if (not client.collection_exists(collection_name)):
         return ['Sorry the data doesnt exits']
    response = client.query_points(
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
    ).points
    ans=[]
    for i in response:
        ans.append(i.payload['data'])  #type:ignore
    return ans


def push(email:str,chunks:str,thread_id:str):
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),cloud_inference=True)
    size = client.get_embedding_size(model_name)
    if not client.collection_exists(collection_name):  ###chatbot_memorystore
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="email",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
    point=PointStruct(
        id=str(uuid.uuid4()),
        vector=Document(text=chunks, model=model_name),
        payload={
            "data":chunks,
            "email":email,
            "thread_id":thread_id
    }
    )
    client.upsert(
         collection_name=collection_name,
         wait=True,
         points=[point]
    )

def push_batch(email: str, chunks: list[str], thread_id: str):
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),cloud_inference=True,timeout=100)
    size = client.get_embedding_size(model_name)
    if not client.collection_exists(collection_name):  ###chatbot_memorystore
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE)
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="email",
                    field_schema=models.PayloadSchemaType.KEYWORD
    )
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=Document(text=c, model=model_name),
            payload={"data": c, "email": email, "thread_id": thread_id,"chunk_index": i,},
        )
        for i,c in enumerate(chunks)
    ]
    for i in range(0, len(points), 100):
        client.upsert(collection_name=collection_name, wait=False, points=points[i:i+100])


if __name__=='__main__':
    em='abc@gmail.com'
    #   push(chunks='hi there ',email=em)
    a=get(query='hello',email=em,thread_id='13')
    # print(a)
    # for i in a:
    #     print(i)
    # print(type(a))
    

