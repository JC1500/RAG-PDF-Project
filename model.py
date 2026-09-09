import asyncio
import os
import uuid
from typing import Optional
from langchain.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from langchain.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy
from langgraph.store.base import BaseStore
from pydantic import BaseModel
from utils.extractor import extract_memory
from utils.vector_store import get_from_store
from langgraph.config import get_stream_writer
from retries.api_fault import api_retry
load_dotenv()


class State(MessagesState):
    summary: str
    docs: list[str]
    query:str


model =ChatOpenRouter(model='poolside/laguna-s-2.1:free',temperature=0.0)
summariser_model = ChatOllama(model='gemma3:4b', temperature=0.0)


async def extract_ltm_summarise(s: State, config: RunnableConfig, store: BaseStore):
    writer = get_stream_writer() 
    writer({'custom_key':"Extracting Ltm and writing summary......"})
    namespace = ('users', str(config['configurable']['user_id']).replace(".", "_dot_"), 'details')  # type:ignore
    items = await store.asearch(namespace)
    existing_memories = [item.value.get('memory', '') for item in items] if items else []
    msg = s['messages']
    res = await extract_memory(msg, existing_memories)  # type:ignore
    if res is not None:
        for item in res['memories']:
            await store.aput(namespace=namespace, key=str(uuid.uuid4()), value={'memory': item})  # type:ignore

    return {'summary': res['summary'], 'messages': [RemoveMessage(id=m.id) for m in msg[:-4] if m.id]} #type:ignore


@tool(description="This is the retreiver which retreives the documents from the store. You need to pass the query for the retreiver search.")
async def get_from_memory(query: str, config: RunnableConfig):
    writer = get_stream_writer() 
    writer({'custom_key':"Extracting data from documents......"})
    # get_from_store hits Qdrant with a blocking client, using multi-threading to avoid blocking 
    res = await asyncio.to_thread(
        get_from_store,
        email=str(config['configurable']['user_id']), #type:ignore
        query=query,
        thread_id=str(config['configurable']['thread_id']), #type:ignore
    )
    return {'docs': res}


async def chat_node(s: State, config: RunnableConfig, store: BaseStore):
    """
    Main chat node that invokes the llm.
    Here we used the memory store to store the long-term memory and memory-saver for storing the short-term memory.
    """
    namespace = ('users', str(config['configurable']['user_id']).replace(".", "_dot_"), 'details')  # type:ignore
    items = await store.asearch(namespace)
    if items:
        ltm = [item.value.get('memory', '') for item in items]
        ltm = ' \n '.join(ltm)  # type:ignore
    else:
        ltm = ""

    hist = s.get('summary', '')
    docs = s.get('docs', [])
    docs_section = "\n".join(f"- {d}" for d in docs) if docs else "No relevant document context retrieved."

    memory_section = f"User Profile / Memories:\n{ltm}" if ltm else "User Profile / Memories: None available yet."

    SYSTEM_PROMPT = f"""You are an expert, friendly AI collaborator and technical assistant.
    {memory_section}

    Relevant Document Context (retrieved from the user's uploaded documents):
    {docs_section}

    Instructions for Personalization and Response:
    - If user profile data or a name is available above, weave it in naturally (e.g., greet the user by name, acknowledge their tools or frameworks like Python, Qdrant, or LangChain, and reference ongoing project context).
    - If relevant document context is available above, ground your answer in it and cite it naturally; do not fabricate content that isn't there.
    - Maintain a warm, engaging, and professional tone tailored specifically to this user.
    - STRICT RULE: DO NOT invent, assume, or hallucinate personal details, names, or project histories that are not explicitly provided in the profile data above. If a detail is missing, rely on general technical expertise without fabricating a personal history.
    - Keep responses concise, actionable, and structured with clear formatting when explaining code or technical steps.
    - At the end of your response, naturally suggest 3 relevant follow-up questions or next steps based on the current context.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if hist:
        messages.append(SystemMessage(content=f"Conversation Summary of the same chat till now:\n{hist}"))
    messages.extend(s['messages']) #type:ignore
    # model_with_tools=model.bind_tools([get_from_memory])
    writer = get_stream_writer() 
    writer({'custom_key':"Model is thinking......"})
    res = await model.ainvoke(messages)
    return {'messages': [res]}

def condition_check(s: State):
    if len(s['messages']) > 4:
        return 'create_summary'
    return '__end__'


graph = StateGraph(State)
tools=ToolNode([get_from_memory])
graph.add_node('tools',tools)
graph.add_node('chat_node', chat_node,retry_policy=RetryPolicy(max_attempts=3,jitter=True,retry_on=api_retry,initial_interval=1.0,backoff_factor=2))  # type:ignore
graph.add_node('extract_ltm_summarise', extract_ltm_summarise)  # type:ignore
graph.add_edge(START, 'chat_node')
graph.add_edge('extract_ltm_summarise',END)
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_conditional_edges(START, condition_check, {'extract_ltm_summarise': 'extract_ltm_summarise', '__end__': '__end__'})
graph.add_edge('chat_node', END)


if __name__ == '__main__':
    async def main():
        checkpointer=InMemorySaver()
        store=InMemoryStore()
        chatbot = graph.compile(checkpointer=checkpointer, store=store)
        while True:
            inp = input('Enter msg: ')
            if inp == 'exit':
                break
            res = await chatbot.ainvoke(
                {'messages': [HumanMessage(content=inp)], 'docs': [], 'summary': ''},
                config={'configurable': {'thread_id': '1','user_id': 'abc@gmail.com'}},
            )
            msgs = res.get('messages', [])
            for m in msgs:
                print(m.content)
            print('-' * 30)
            print(res.get('summary', ''))

    asyncio.run(main())