from openrouter import errors
import logging

def api_retry(e:BaseException)->bool:
    if isinstance(e,errors.TooManyRequestsResponseError):
        logging.warning("Retrying")
        return True
    return False


def default_run():
    pass
