from pathlib import Path
import sys
import asyncio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).parents[1]))
from starlette.requests import Request
from Backend.dependencies import get_current_user
from Backend.main import app
from Backend.routers.auth import me




def test_me_unauthenticated():
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/auth/me",
        "headers": [],
        "session":{}
    })
    with pytest.raises(HTTPException) as error:
        asyncio.run(me(request))
    assert error.value.status_code==401


def test_health_check():
    response = TestClient(app).get("/health_check")
    assert response.json() == {"message": "ALL GOOD"}


def test_cors_preflight_for_local_frontend():
    response = TestClient(app).options(
        "/chat",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5500"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_backend_root_does_not_serve_frontend():
    response = TestClient(app).get("/")
    assert response.status_code == 404


if __name__ =='__main__':
    test_health_check()
    test_me_unauthenticated()