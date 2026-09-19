from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from Backend.retries.api_fault import api_retry
from openrouter import errors


def test_api_fault():
    rate_limit_error = errors.TooManyRequestsResponseError.__new__(errors.TooManyRequestsResponseError)
    
    assert api_retry(rate_limit_error) == True
    assert api_retry(ValueError()) == False



if __name__=='__main__':
    test_api_fault()