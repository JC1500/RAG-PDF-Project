from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage

from Backend.dependencies import get_chatbot, get_current_user
from Backend.utils.classes import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat_endpoint(
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
    chatbot=Depends(get_chatbot),
):
    email = user["email"]
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content=chat_request.inp)], "email": email},
        config={
            "run_name": "test01",
            "configurable": {
                "thread_id": chat_request.thread_id,
                "user_id": email,
            },
        },
    )

    messages = result.get("messages", [])
    latest_message = messages[-1].content if messages else ""
    return {
        "response": latest_message,
        "summary": result.get("summary", ""),
    }
