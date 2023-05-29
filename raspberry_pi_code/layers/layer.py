from typing import Optional
import logging

class PipelineLayer:
    def __init_subclass__(cls):
        assert hasattr(cls, "run") and callable(getattr(cls, "run")), "PipelineLayer subclasses must implement a run method"

    def __init__(self, logger: Optional[logging.Logger] = None, *args, **kwargs):
        self._logger = logger or logging.getLogger()
