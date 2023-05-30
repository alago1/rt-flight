import importlib
from typing import Literal, Type, TypeVar, Union

T = TypeVar('T', bound='AbstractEngine')

SUPPORTED_ENGINES = ['tflite', 'coral', 'onnx', 'tensorrt']
U = Union[TypeVar('U', *SUPPORTED_ENGINES), Literal['auto']]

known_extensions = {
    'edgetpu.tflite': 'coral',
    'tflite': 'tflite',
    'onnx': 'onnx',
    'trt': 'tensorrt'
}

class AbstractEngine:
    def __init_subclass__(cls) -> None:
        assert hasattr(cls, 'load_model') and callable(getattr(cls, 'load_model')), f"Load model function must be defined in {cls.__qualname__}"
        assert hasattr(cls, '__call__'), f"class {cls.__qualname__} must be callable"
        assert hasattr(cls, 'get_input_shape') and callable(getattr(cls, 'get_input_shape')), f"get_input_shape must be defined in {cls.__qualname__}"


class EngineLoader:

    @staticmethod
    def load(model_path: str, engine: U = 'auto', *args, **kwargs) -> Type[T]:
        if engine == 'auto':
            for extension, engine_name in known_extensions.items():
                if model_path.lower().endswith(extension):
                    engine = engine_name
                    break
            else:
                raise ValueError(f"Could not infer engine from model path '{model_path}'. Please specify engine manually.")

        if engine not in SUPPORTED_ENGINES:
            raise NotImplementedError(f"Engine '{engine}' is not supported. Please choose one of {SUPPORTED_ENGINES}")
        
        engine_module = importlib.import_module(f'.{engine}', 'engines')
        engine_class = getattr(engine_module, f'{engine.capitalize()}Engine')

        return engine_class(model_path=model_path, *args, **kwargs)
