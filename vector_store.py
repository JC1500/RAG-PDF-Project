from qdrant_client import QdrantClient,models
import os
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, PointStruct,Document
from typing import List,Dict
import uuid

load_dotenv()

model_name="sentence-transformers/all-MiniLM-L6-v2"
collection_name='test'


def get(email: str, query: str):
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=Document(text=query, model=model_name),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="email",
                    match=models.MatchValue(value=email),
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


def push(email:str,chunks:str):
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
            "email":email
    }
    )
    client.upsert(
         collection_name=collection_name,
         wait=True,
         points=[point]
    )


if __name__=='__main__':
    em='abc@gmail.com'
    #   push(chunks='hi there ',email=em)
    a=get(query='hello',email=em)
    # print(a)
    # for i in a:
    #     print(i)
    # print(type(a))
    

