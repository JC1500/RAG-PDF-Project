from langchain_openrouter import ChatOpenRouter
import os
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END,MessagesState
from langchain.messages import SystemMessage,HumanMessage,RemoveMessage,AIMessage,ToolMessage
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from dotenv import load_dotenv
from pydantic import BaseModel
from extractor import extract_memory
import uuid


load_dotenv()


class State(MessagesState):
    summary:str


model=ChatOllama(model='gemma3:4b')
summariser_model=ChatOllama(model='gemma3:4b')
# model=ChatOpenRouter(model='openai/gpt-oss-20b:free',temperature=0.5,api_key=os.getenv('OPENROUTER_API_KEY')) #type:ignore
embed=HuggingFaceEmbeddings(cache_folder='./embeddings_model')
# summariser_model=ChatOpenRouter(model='openai/gpt-oss-20b:free',temperature=0,api_key=os.getenv('OPENROUTER_API_KEY')) #type:ignore
store=InMemoryStore(index={'embed':embed,'dims':1356})

def extract_ltm(s:State,config=RunnableConfig):
    namespace=('users',str(config['configurable']['user_id']),'details') #type:ignore
    memories=store.search(namespace)
    msg=s['messages'][-2].content
    res=extract_memory(msg,memories)
    if res:
        for item in res:
            store.put(namespace=namespace,key=str(uuid.uuid4()),value=item)
    
def chat_node(s:State,config:RunnableConfig):
    """
    Main chat node that invokes the llm.
    Here we used the memory store to store the long-term memory and memory-saver for storing the short-term memory.
    """
    namespace=('users',str(config['configurable']['user_id']),'details') #type:ignore
    items=store.search(namespace)
    if items:
        data=[item.value for item in items]
        print(data)
        data=' \n '.join(data)
    else:
        data=""
    
    SYSTEM_PROMPT_TEMPLATE = f"""You are a helpful assistant with memory capabilities.
    If user-specific memory is available, use it to personalize 
    your responses based on what you know about the user.

    Your goal is to provide relevant, friendly, and tailored 
    assistance that reflects the user’s preferences, context, and past interactions.

    If the user’s name or relevant personal context is available, always personalize your responses by:
        – Always Address the user by name (e.g., "Sure, JC...") when appropriate
        – Referencing known projects, tools, or preferences (e.g., "your MCP  server python based project")
        – Adjusting the tone to feel friendly, natural, and directly aimed at the user

    Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." 
    say "Since your project is built with TypeScript..."

    Use personalization especially in:
        – Greetings and transitions
        – Help or guidance tailored to tools and frameworks the user uses
        – Follow-up messages that continue from past context

    Always ensure that personalization is based only on known user details and not assumed.

    In the end suggest 3 relevant further questions based on the current response and user profile

    The user’s memory (which may be empty) is provided as: {data}
    """
    hist=s.get('summary','')
    messages=s['messages']
    if hist:
        messages=[SystemMessage(content=hist)]+messages
    messages=[SystemMessage(content=SYSTEM_PROMPT_TEMPLATE)]+messages

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

graph.add_node('chat_node',chat_node)
graph.add_node('extract_ltm',extract_ltm)
graph.add_node('create_summary',create_summary)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node','extract_ltm')
graph.add_conditional_edges('extract_ltm',condition_check,{'create_summary':'create_summary','__end__':'__end__'})
graph.add_edge('create_summary',END)

checkpointer=InMemorySaver()
chatbot=graph.compile(checkpointer=checkpointer)


if __name__=='__main__':
    while True:
        # print(chatbot.get_state(config={'configurable':{'thread_id':'1',}}))
        inp=input('Enter msg: ')
        if inp=='exit':
            break
        res=chatbot.invoke({'messages':[HumanMessage(content=inp)]},config={'configurable':{'thread_id':'1','user_id':'u1'}})
        msg=res.get('messages','')
        print(len(msg))
        for m in msg:
            print(m.content)
        print('-'*30)
        print(res.get('summary',''))
        print("store: ", store.search(('users','u1','details')))
