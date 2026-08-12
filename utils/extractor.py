from langchain_ollama import ChatOllama  
from pydantic import BaseModel,Field
from typing import Annotated
from typing import Optional
"""
    Function to extract Long-Term Memory is defined here
"""


class schema(BaseModel):
    is_present:Annotated[bool,Field(description='Return True if their is memory present that should be added to the Long-Term Memory, else return False.')]
    memories:Annotated[list[str],Field(description='Memories to add to the long term memory.')]



llm=ChatOllama(model='gemma3:4b',temperature=0.0)
extractor_model=llm.with_structured_output(schema)


def extract_memory(message:str, memory:Optional[list[str]]=None)->None| list[str]:
    """
    This function is used to extract the Long-Term Memory from the user input messges
    """
    system_prompt="""
            You are a memory extraction assistant. Your role is to analyze the user’s messages and identify information that should be stored as long-term memory.
            Guidelines:
            1. Extract only durable facts, preferences, or context that are useful for future personalization.
            – Examples: "My name is Janmejai", "I prefer Python for ML projects", "I attended IIT Jammu Summer School".
            – PreferredName: if the user shares their name or nickname.
            – Preferences: tools, frameworks, styles, or routines the user favors.
            – Goals: long-term objectives or projects.
            – Skills: technical or professional abilities.
            – Interests: hobbies, topics, or domains the user enjoys.
            – Do NOT extract transient or short-term details like "I’m hungry", "I’ll call you tomorrow".
            2. Normalize the extracted memory into single, clear string. Keep it short and factual.
            – Example: "User’s name is Janmejai" instead of "Call me Janmejai".
            4. Ignore speculative, emotional, or temporary states.
            5. The output memory should be concise and as short as possible.
            Existing memory(which can be empyty) is given as: {memory}
            Your task: From the following user messages, extract all relevant long-term memory facts as concise strings, ensuring no duplication with the already provided memory.\n {message}
        """
    msg=system_prompt.format(memory=memory,message=message)
    response=extractor_model.invoke(msg)
    if response.is_present: #type:ignore
        return response.memories  #type:ignore
    return None

if(__name__=='__main__'):
    res=extract_memory(message='i will call you tomorrow')
    print(res)
    if res is not None:
        print(res)