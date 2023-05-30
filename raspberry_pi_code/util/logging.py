import time
import logging
from typing import Optional
import sys

def log_time(f):
    def timed_f(*args, **kwargs):
        start = time.monotonic()
        out = f(*args, **kwargs)
        elapsed = time.monotonic() - start

        self_inst = args[0]
        self_name = self_inst.__class__.__qualname__
        logging.info(f"{self_name}@{hex(id(self_inst))}.{f.__name__} executed in {elapsed:.4f}s")

        return out
    
    return timed_f

def setup_logger(logger_name: Optional[str] = None,
               log_file: str = "logs.txt",
               level: int = logging.DEBUG,
               include_console: bool = True):
    logFormatter = logging.Formatter('[%(asctime)s] [%(funcName)s-%(threadName)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    fileHandler = logging.FileHandler(log_file, encoding="utf-8")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if not include_console:
        return logger

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger
