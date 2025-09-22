from src.logger import logging
import sys
from src.exception import MyException

try:
    a= 1 +"cf"
except Exception as e:
    logging.info(e)
    raise MyException(e,sys) from e