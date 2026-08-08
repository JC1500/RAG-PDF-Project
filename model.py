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

load_dotenv()
class State(MessagesState):
    summary:str

model=ChatOllama(model='gemma3:4b')
summariser_model=ChatOllama(model='gemma3:4b')
# model=ChatOpenRouter(model='openai/gpt-oss-20b:free',temperature=0.5,api_key=os.getenv('OPENROUTER_API_KEY')) #type:ignore
# embed=HuggingFaceEmbeddings(cache_folder='./embeddings_model')
# summariser_model=ChatOpenRouter(model='openai/gpt-oss-20b:free',temperature=0,api_key=os.getenv('OPENROUTER_API_KEY')) #type:ignore
# store=InMemoryStore(index={'embed':embed,'dims':1356})


def chat_node(s:State,config:RunnableConfig):
    """
    Main chat node that invokes the llm.
    Here we used the memory store to store the long-term memory and memory-saver for storing the short-term memory.
    """
    # namespace=('users',config['configurable']['user_id'],'details') #type:ignore
    # items=store.search(namespace)
    # if items:
    #     data=[item.value.get('data','') for item in items]
    #     data='/n'.join(data)
    # else:
    #     data=""
    # SYSTEM_PROMPT_TEMPLATE = f"""You are a helpful assistant with memory capabilities.
    # If user-specific memory is available, use it to personalize 
    # your responses based on what you know about the user.

    # Your goal is to provide relevant, friendly, and tailored 
    # assistance that reflects the user’s preferences, context, and past interactions.

    # If the user’s name or relevant personal context is available, always personalize your responses by:
    #     – Always Address the user by name (e.g., "Sure, JC...") when appropriate
    #     – Referencing known projects, tools, or preferences (e.g., "your MCP  server python based project")
    #     – Adjusting the tone to feel friendly, natural, and directly aimed at the user

    # Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." 
    # say "Since your project is built with TypeScript..."

    # Use personalization especially in:
    #     – Greetings and transitions
    #     – Help or guidance tailored to tools and frameworks the user uses
    #     – Follow-up messages that continue from past context

    # Always ensure that personalization is based only on known user details and not assumed.

    # In the end suggest 3 relevant further questions based on the current response and user profile

    # The user’s memory (which may be empty) is provided as: {data}
    # """
    hist=s['summary']
    messages=s['messages']
    if hist:
        messages=[SystemMessage(content=hist)]+messages
    # messages=[SystemMessage(content=SYSTEM_PROMPT_TEMPLATE)]+messages

    res=model.invoke(messages)
    return {'messages':[res]}


def create_summary(s:State):
    """
    Using Summarising method.
    """
    existing_summary=s['summary']
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


# graph=StateGraph(State)

# graph.add_node('chat_node',chat_node)
# graph.add_node('create_summary',create_summary)

# graph.add_edge(START,'chat_node')
# graph.add_conditional_edges('chat_node',condition_check,{'create_summary':'create_summary','__end__':'__end__'})
# graph.add_edge('create_summary',END)

# checkpointer=InMemorySaver()
# chatbot=graph.compile(checkpointer=checkpointer)


if __name__=='__main__':
    # while True:
    #     # print(chatbot.get_state(config={'configurable':{'thread_id':'1',}}))
    #     inp=input('Enter msg: ')
    #     if inp=='exit':
    #         break
    #     res=chatbot.invoke(State(messages=[HumanMessage(content=inp)],summary=''),config={'configurable':{'thread_id':'1','user_id':'u1'}})
    #     print(res['messages'])
    #     print('-'*30)
    #     print(res['summary'])
    messages=[HumanMessage(content='Hi my name is Jc', additional_kwargs={}, response_metadata={}, id='c595ffb8-d971-4c25-ac91-ccd970bca27b'), AIMessage(content="Hi Jc! It's nice to meet you. 😊 What can I do for you today? Do you want to chat, play a game, or maybe just need some information?", additional_kwargs={}, response_metadata={'model': 'gemma3:4b', 'created_at': '2026-08-08T04:30:08.3817594Z', 'done': True, 'done_reason': 'stop', 'total_duration': 14842396300, 'load_duration': 10407294400, 'prompt_eval_count': 16, 'prompt_eval_duration': 400112000, 'eval_count': 39, 'eval_duration': 4012159000, 'logprobs': None, 'model_name': 'gemma3:4b', 'model_provider': 'ollama'}, id='lc_run--019fdfa2-73fc-7063-95df-df5d9b364ff0-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 16, 'output_tokens': 39, 'total_tokens': 55}), HumanMessage(content='What is my name?', additional_kwargs={}, response_metadata={}, id='9dae7e43-6060-45d4-bec1-2ba0ffb9ce6c'), AIMessage(content="Your name is Jc! 😄 \n\nI just confirmed it when you introduced yourself. 😊 \n\nIs there anything else you'd like to know about yourself, or would you like to talk about something specific?", additional_kwargs={}, response_metadata={'model': 'gemma3:4b', 'created_at': '2026-08-08T04:30:46.2394741Z', 'done': True, 'done_reason': 'stop', 'total_duration': 6795273400, 'load_duration': 878983800, 'prompt_eval_count': 71, 'prompt_eval_duration': 1062477000, 'eval_count': 45, 'eval_duration': 4847158000, 'logprobs': None, 'model_name': 'gemma3:4b', 'model_provider': 'ollama'}, id='lc_run--019fdfa3-2752-73b0-a7ac-7edece3bc4f4-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 71, 'output_tokens': 45, 'total_tokens': 116})]
    s=State({'summary':''},messages=messages)
    res=create_summary(s)
    print(res['summary'])
    print('-'*30)
    print(res['messages'])

