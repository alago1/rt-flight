import time
import logging

def log_time(f):
    def timed_f(*args, **kwargs):
        start = time.monotonic()
        out = f(*args, **kwargs)
        elapsed = time.monotonic() - start

        self_inst = args[0]
        self_name = self_inst.__class__.__qualname__
        logging.getLogger().info(f"[{self_name}@{hex(id(self_inst))}] INFO: {f.__name__} executed in {elapsed:.4f}s")

        return out
    
    return timed_f
