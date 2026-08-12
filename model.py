from langchain_openrouter import ChatOpenRouter
import os
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END,MessagesState
from langchain.messages import SystemMessage,HumanMessage,RemoveMessage,AIMessage,ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv
from langgraph.store.base import BaseStore
from pydantic import BaseModel
from utils.extractor import extract_memory
import uuid
from typing import Optional

load_dotenv()


class State(MessagesState):
    summary:str


model=ChatOllama(model='gemma3:4b',temperature=0.0)
summariser_model=ChatOllama(model='gemma3:4b',temperature=0.0)
# model=ChatOpenRouter(model='openai/gpt-oss-20b:free',temperature=0.5,api_key=os.getenv('OPENROUTER_API_KEY')) #type:ignore
# embed=HuggingFaceEmbeddings(cache_folder='./embeddings_model')
# summariser_model=ChatOpenRouter(model='openai/gpt-oss-20b:free',temperature=0,api_key=os.getenv('OPENROUTER_API_KEY')) #type:ignore


def extract_ltm(s:State,config:RunnableConfig,store:BaseStore):
    namespace=('users',str(config['configurable']['user_id']),'details') #type:ignore
    memories=store.search(namespace)
    msg=s['messages'][-2].content
    res=extract_memory(msg,memories) #type:ignore
    if res is not None:
        for item in res:
            store.put(namespace=namespace,key=str(uuid.uuid4()),value={'memory':item})  #type:ignore
    
def chat_node(s:State,config:RunnableConfig,store:BaseStore):
    """
    Main chat node that invokes the llm.
    Here we used the memory store to store the long-term memory and memory-saver for storing the short-term memory.
    """
    namespace=('users',str(config['configurable']['user_id']),'details') #type:ignore
    items=store.search(namespace)
    if items:
        ltm=[item.value.get('memory','') for item in items]
        print(ltm)
        ltm=' \n '.join(ltm) #type:ignore
    else:
        ltm=""
    
    hist=s.get('summary','')
    messages=s['messages']
    memory_section = f"User Profile / Memories:\n{ltm}" if ltm else "User Profile / Memories: None available yet."

    SYSTEM_PROMPT = f"""You are an expert, friendly AI collaborator and technical assistant.
    {memory_section}

    Instructions for Personalization and Response:
    - If user profile data or a name is available above, weave it in naturally (e.g., greet the user by name, acknowledge their tools or frameworks like Python, Qdrant, or LangChain, and reference ongoing project context).
    - Maintain a warm, engaging, and professional tone tailored specifically to this user.
    - STRICT RULE: DO NOT invent, assume, or hallucinate personal details, names, or project histories that are not explicitly provided in the profile data above. If a detail is missing, rely on general technical expertise without fabricating a personal history.
    - Keep responses concise, actionable, and structured with clear formatting when explaining code or technical steps.
    - At the end of your response, naturally suggest 3 relevant follow-up questions or next steps based on the current context.
    """
    messages = []
    messages.append(SystemMessage(content=SYSTEM_PROMPT))
    if hist:
        messages.append(SystemMessage(content=f"Conversation Summary of the same chat till now:\n{hist}"))
    messages.extend(s['messages'])

    res=model.invoke(messages)
    return {'messages':[res]}

def create_summary(s:State):
    """
    Using Summarising method.
    """
    existing_summary=s.get('summary','')
    chats=s['messages'][:4]
    msg=[]
    for m in chats:
        if(isinstance(m,AIMessage)):
            msg.append({'ai':m.content})
        elif (isinstance(m,ToolMessage)):
            msg.append({'tool':m.content})
        elif (isinstance(m,HumanMessage)):
            msg.append({'user':m.content})
    if existing_summary:
        prompt=f"""
                Existing Summary: {existing_summary}
                You are an assistant tasked with summarizing the user’s recent messages and conversation history. 
                - Don't start the summary by stating like 'Here is the summary of the conversation...' or 'The summary of the conversation is...'.
                - Focus only on the content of the user’s messages, not on URLs, tab IDs, or technical metadata. 
                - Provide a concise, neutral, and factual summary in 3–5 sentences. 
                - Highlight the main topics discussed, recurring themes, and the user’s intent or goals. 
                - Avoid speculation, unnecessary detail, or references to browsing metadata. 
                - Ensure the summary is cohesive, easy to read, and captures the essence of the conversation.
                The messages are given below.\n {msg}
                """
    else:
        prompt=f"""You are an assistant tasked with summarizing the user’s recent messages and conversation history. 
        - Don't start the summary by stating like 'Here is the summary of the conversation...' or 'The summary of the conversation is...'.
        - Focus only on the content of the user’s messages, not on URLs, tab IDs, or technical metadata. 
        - Provide a concise, neutral, and factual summary in 3–5 sentences. 
        - Highlight the main topics discussed, recurring themes, and the user’s intent or goals. 
        - Avoid speculation, unnecessary detail, or references to browsing metadata. 
        - Ensure the summary is cohesive, easy to read, and captures the essence of the conversation.
        The messages are given below.\n {msg} """
    msg=[SystemMessage(content=prompt)]
    summary=summariser_model.invoke(msg)
    return {'summary':summary.content,'messages':[RemoveMessage(id=m.id) for m in chats if m.id]}

def condition_check(s:State):
    if(len(s['messages'])>4): return 'create_summary'
    return '__end__'

    


graph=StateGraph(State)

graph.add_node('chat_node',chat_node) #type:ignore
graph.add_node('extract_ltm',extract_ltm)  #type:ignore
graph.add_node('create_summary',create_summary)  #type:ignore

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node','extract_ltm')
graph.add_conditional_edges('extract_ltm',condition_check,{'create_summary':'create_summary','__end__':'__end__'})
graph.add_edge('create_summary',END)
    


if __name__=='__main__':
    with PostgresSaver.from_conn_string(str(os.getenv('DB_URI'))) as checkpointer,PostgresStore.from_conn_string(str(os.getenv('DB_URI'))) as store:
        store.setup()
        checkpointer.setup()
        chatbot = graph.compile(checkpointer=checkpointer, store=store)
        while True:
            inp=input('Enter msg: ')
            if inp=='exit':
                break
            res=chatbot.invoke(State(messages=[HumanMessage(content=inp)]),config={'configurable':{'thread_id':'1','user_id':'u1'}})
            msg=res.get('messages','')
            print(len(msg))
            for m in msg:
                print(m.content)
            print('-'*30)
            print(res.get('summary',''))
            print("store: ", store.search(('users','u1','details')))