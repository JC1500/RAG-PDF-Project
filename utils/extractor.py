import asyncio
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Annotated
from typing import Optional
from langchain.messages import SystemMessage,AnyMessage,HumanMessage
"""
    Function to extract Long-Term Memory is defined here
"""


class schema(BaseModel):
    summary: str = Field(description="Concise 3-5 sentence factual summary of the conversation.")
    is_present: Annotated[bool, Field(description='Return True if their is memory present that should be added to the Long-Term Memory, else return False.')]
    memories: Annotated[list[str], Field(description='Memories to add to the long term memory.')]


llm = ChatOllama(model='gemma3:4b', temperature=0.0)
extractor_model = llm.with_structured_output(schema)


async def extract_memory(message: list[AnyMessage], memory: Optional[list[str]] = None) -> None | dict[str,str]:
    """
    This function is used to extract the Long-Term Memory from the user input messages.
    Runs asynchronously so it doesn't block the event loop while waiting on the LLM.
    """
    system_prompt="""
    You have two tasks. Perform both using the conversation below.

    ### Task 1 — Summarize
    Summarize the user's recent messages and conversation history.
    - Don't start with phrases like "Here is the summary..." or "The summary is...".
    - Focus only on message content, not URLs, tab IDs, or technical metadata.
    - Provide a concise, neutral, factual summary in 3-5 sentences.
    - Highlight main topics, recurring themes, and the user's intent or goals.
    - Avoid speculation or unnecessary detail.
    {existing_summary_block}
    ### Task 2 — Extract long-term memory
    Analyze the user's messages and extract durable facts/preferences/context useful for
    future personalization (name, preferences, goals, skills, interests). Do NOT extract
    transient/short-term details ("I'm hungry", "I'll call tomorrow"). Normalize each fact
    into a short, clear string (e.g. "User's name is Janmejai"). Do not duplicate anything
    already present in existing memory below.
    Existing memory: {existing_memory}
    ### Conversation
    {messages}
    """
    existing_summary_block = f"Existing Summary: {memory}" if memory else ""
    
    prompt = system_prompt.format(
            existing_summary_block=existing_summary_block,
            existing_memory=memory,
            messages=message,
        )
    try:
        result=await extractor_model.ainvoke([SystemMessage(content=prompt)])
    except Exception as e:
        ## IF error in extraction return None
        return None
    if result.is_present:  # type:ignore
        return {'memory': result.memories,"summary": result.summary}  # type:ignore
    return None

if __name__ == '__main__':
    a=input('input')
    res = asyncio.run(extract_memory(message=[HumanMessage(content=a)]))
    print(res)
    if res is not None:
        print(res)