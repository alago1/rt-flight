import importlib
import pathlib
from typing import Type, Tuple, List

import numpy as np

SUPPORTED_ENGINES = ['tflite', 'coral', 'onnx', 'tensorrt']

class AbstractEngine:
    def __init_subclass__(cls) -> None:
        assert hasattr(cls, 'load_model') and callable(getattr(cls, 'load_model')), f"Load model function must be defined in {cls.__qualname__}"
        assert hasattr(cls, '__call__'), f"class {cls.__qualname__} must be callable"
        assert hasattr(cls, 'get_input_shape') and callable(getattr(cls, 'get_input_shape')), f"get_input_shape must be defined in {cls.__qualname__}"

    def load_model(self) -> None:
        pass

    def get_input_shape(self) -> Tuple[int, int]:
        pass

    def __call__(self, input: np.ndarray) -> List[np.ndarray]:
        pass


class EngineLoader:

    @staticmethod
    def load(model_path: str, engine='tflite', *args, **kwargs) -> Type[AbstractEngine]:
        if engine not in SUPPORTED_ENGINES:
            raise NotImplementedError(f"Engine '{engine}' is not supported. Please choose one of {SUPPORTED_ENGINES}")
                
        parent_path = pathlib.Path(__file__).parent.resolve()
        # engine_module = importlib.import_module(f'{parent_path}/{engine}.py')
        engine_module = importlib.import_module(f'.{engine}', 'engines')
        engine_class = getattr(engine_module, f'{engine.capitalize()}Engine')

        return engine_class(model_path=model_path, *args, **kwargs)

