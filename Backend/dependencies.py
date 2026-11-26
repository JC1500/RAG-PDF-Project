from typing import Any

from fastapi import HTTPException, Request


def get_current_user(request: Request) -> dict:
    """Return the authenticated user stored in the session."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


def get_chatbot(request: Request) -> Any:
    return request.app.state.chatbot
