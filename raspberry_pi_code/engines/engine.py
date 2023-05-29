import importlib
from typing import Type, TypeVar


T = TypeVar('T', bound='AbstractEngine')

SUPPORTED_ENGINES = ['tflite', 'coral', 'onnx', 'tensorrt']
U = TypeVar('U', *SUPPORTED_ENGINES)

class AbstractEngine:
    def __init_subclass__(cls) -> None:
        assert hasattr(cls, 'load_model') and callable(getattr(cls, 'load_model')), f"Load model function must be defined in {cls.__qualname__}"
        assert hasattr(cls, '__call__'), f"class {cls.__qualname__} must be callable"
        assert hasattr(cls, 'get_input_shape') and callable(getattr(cls, 'get_input_shape')), f"get_input_shape must be defined in {cls.__qualname__}"


class EngineLoader:

    @staticmethod
    def load(model_path: str, engine: U='tflite', *args, **kwargs) -> Type[T]:
        if engine not in SUPPORTED_ENGINES:
            raise NotImplementedError(f"Engine '{engine}' is not supported. Please choose one of {SUPPORTED_ENGINES}")
        
        engine_module = importlib.import_module(f'.{engine}', 'engines')
        engine_class = getattr(engine_module, f'{engine.capitalize()}Engine')

        return engine_class(model_path=model_path, *args, **kwargs)
