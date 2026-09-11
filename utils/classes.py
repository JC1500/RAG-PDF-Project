from pydantic import BaseModel,Field
from typing import Annotated
from langgraph.graph import MessagesState




class Extactorschema(BaseModel):
    summary: str = Field(description="Concise 3-5 sentence factual summary of the conversation.")
    is_present: Annotated[bool, Field(description='Return True if their is memory present that should be added to the Long-Term Memory, else return False.')]
    memories: Annotated[list[str], Field(description='Memories to add to the long term memory.')]



class ChatSchema(MessagesState):
    summary: str
    docs: list[str]
    query:str



class ChatRequest(BaseModel):
    inp: str
    thread_id: str
