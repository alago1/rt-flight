class PipelineLayer:
    def __init_subclass__(cls):
        assert hasattr(cls, "run") and callable(getattr(cls, "run")), "PipelineLayer subclasses must implement a run method"
