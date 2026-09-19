from pathlib import Path
import sys
import asyncio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).parents[1]))
from Backend.main import me,Request,health_check,ingest_pdf,app,get_current_user





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
    assert health_check()['message']=='ALL GOOD'


if __name__ =='__main__':
    test_health_check()
    test_me_unauthenticated()